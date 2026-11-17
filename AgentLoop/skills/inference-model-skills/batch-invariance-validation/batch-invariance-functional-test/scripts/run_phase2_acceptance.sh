#!/bin/bash
# Parameterized batch invariance acceptance runner.
# The caller defines one scenario matrix through VLLM_TEST_* variables.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEST_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ENVIRONMENT_ROOT=$(cd "${TEST_ROOT}/../batch-invariance-environment" && pwd)
PYTEST_CASE="${TEST_ROOT}/scripts/test_batch_invariant.py::test_logprobs_bitwise_batch_invariance_bs1_vs_bsN"

export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_TEST_LOG_STATS_INTERVAL:-1}"

SCENARIO="${VLLM_TEST_SCENARIO:-phase2}"
MODEL="${VLLM_TEST_MODEL:-}"
TP_LIST="${VLLM_TEST_TP_LIST:-}"
MAX_PROMPT_NUM_LIST="${VLLM_TEST_MAX_PROMPT_NUM_LIST:-}"
PROMPT_MAX_TOKENS_PAIRS="${VLLM_TEST_PROMPT_MAX_TOKENS_PAIRS:-}"
TEMPERATURE="${VLLM_TEST_TEMPERATURE:-0.6}"
REQUIRED_ROUNDS="${VLLM_TEST_REQUIRED_ROUNDS:-1}"
BATCHED_TOKENS_OFFSET="${VLLM_TEST_BATCHED_TOKENS_OFFSET:-101}"
FIXED_MAX_NUM_BATCHED_TOKENS="${VLLM_TEST_MAX_NUM_BATCHED_TOKENS:-}"
AUTO_BATCHED_TOKENS_CAP="${VLLM_TEST_AUTO_MAX_NUM_BATCHED_TOKENS_CAP:-131072}"
MODEL_MAX_LEN="${VLLM_TEST_MAX_MODEL_LEN:-22578}"
SEED_BASE="${VLLM_TEST_SEED:-12345}"
ENABLE_PREFIX_CACHING="${VLLM_TEST_ENABLE_PREFIX_CACHING:-False}"
LOG_ROOT="${VLLM_TEST_LOG_DIR:-./logs}"
RUN_CONFIG="${VLLM_TEST_RUN_CONFIG:-}"
POST_RUN_WAIT_TIME="${VLLM_TEST_POST_RUN_WAIT_TIME:-5}"
TRANSIENT_RESOURCE_RETRY_COUNT="${VLLM_TEST_TRANSIENT_RESOURCE_RETRY_COUNT:-2}"
TRANSIENT_RESOURCE_RETRY_DELAY="${VLLM_TEST_TRANSIENT_RESOURCE_RETRY_DELAY:-10}"
TRANSIENT_RESOURCE_PATTERN='free memory on device.*(<|less than).*desired'
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="${LOG_ROOT%/}/batch_invariant_results_${timestamp}"
mkdir -p "$output_dir"

summary_csv="$output_dir/summary.csv"

echo "Scenario,Model,ParallelMode,TP,Temperature,MaxTokens,MaxPrompts,PromptLength,MaxNumBatchedTokens,EnablePrefixCaching,SchedulerStatus,KVCapacityTokens,RequiredKVTokens,MaxKVCacheUsage,Preemptions,Status,RequiredRounds,TotalAttempts,TotalFailures,StepMismatch,TokenMismatch,LogprobMismatch,FirstFailureReason,FailureManifests,SchedulerReports,LogFiles" > "$summary_csv"

calculate_batched_tokens() {
    local max_prompts=$1
    local prompt_len=$2
    local max_tokens=$3
    if [ -n "$FIXED_MAX_NUM_BATCHED_TOKENS" ]; then
        echo "$FIXED_MAX_NUM_BATCHED_TOKENS"
    else
        local requested=$(((prompt_len + BATCHED_TOKENS_OFFSET) * max_prompts))
        local minimum=$((prompt_len + max_tokens))
        if [ "$minimum" -lt "$MODEL_MAX_LEN" ]; then
            minimum=$MODEL_MAX_LEN
        fi
        local planned=$requested
        if [ "$planned" -gt "$AUTO_BATCHED_TOKENS_CAP" ]; then
            planned=$AUTO_BATCHED_TOKENS_CAP
        fi
        if [ "$planned" -lt "$minimum" ]; then
            planned=$minimum
        fi
        echo "$planned"
    fi
}

require_non_empty() {
    local name=$1
    local value=$2
    if [ -z "$value" ]; then
        echo "ERROR: $name must be set." >&2
        return 1
    fi
}

validate_positive_int() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid $name value: '$value'. Use positive integers." >&2
        return 1
    fi
}

validate_non_negative_int() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid $name value: '$value'. Use a non-negative integer." >&2
        return 1
    fi
}

reject_unsupported_sampling_parameters() {
    local name
    local unsupported=(
        VLLM_TEST_TOP_P
        VLLM_TEST_TOP_K
        VLLM_TEST_MIN_P
        VLLM_TEST_PRESENCE_PENALTY
        VLLM_TEST_FREQUENCY_PENALTY
        VLLM_TEST_REPETITION_PENALTY
    )
    for name in "${unsupported[@]}"; do
        if [[ -v "$name" ]]; then
            echo "ERROR: UNSUPPORTED_SAMPLING_PARAMETER: $name. Only VLLM_TEST_TEMPERATURE is supported." >&2
            return 1
        fi
    done
}

parse_positive_int_list() {
    local name=$1
    local raw=$2
    local normalized="${raw//,/ }"
    local value
    local parsed=0

    for value in $normalized; do
        validate_positive_int "$name" "$value" || return 1
        echo "$value"
        parsed=$((parsed + 1))
    done

    if [ "$parsed" -eq 0 ]; then
        echo "ERROR: $name list is empty." >&2
        return 1
    fi
}

parse_prompt_max_tokens_pairs() {
    local raw=$1
    local normalized="${raw//,/ }"
    local pair
    local prompt_len
    local max_tokens
    local parsed=0

    for pair in $normalized; do
        pair="${pair//->/:}"
        pair="${pair//-/:}"
        IFS=':' read -r prompt_len max_tokens extra <<< "$pair"
        if [ -n "${extra:-}" ] || [ -z "${prompt_len:-}" ] || [ -z "${max_tokens:-}" ]; then
            echo "ERROR: invalid prompt/max_tokens pair: '$pair'. Use prompt_len:max_tokens, e.g. '2048:20480 4096:4096'." >&2
            return 1
        fi
        validate_positive_int "prompt length" "$prompt_len" || return 1
        validate_positive_int "max tokens" "$max_tokens" || return 1
        echo "${prompt_len}:${max_tokens}"
        parsed=$((parsed + 1))
    done

    if [ "$parsed" -eq 0 ]; then
        echo "ERROR: prompt/max_tokens pair list is empty." >&2
        return 1
    fi
}

parse_failure_stats() {
    local stats_file=$1
    local total_failures=0
    local step_mismatch=0
    local token_mismatch=0
    local logprob_mismatch=0
    local first_reason=""

    if [ -f "$stats_file" ]; then
        while IFS='=' read -r key value; do
            case "$key" in
                "TOTAL_FAILURES") total_failures=$value ;;
                "STEP_MISMATCH") step_mismatch=$value ;;
                "TOKEN_MISMATCH") token_mismatch=$value ;;
                "LOGPROB_MISMATCH") logprob_mismatch=$value ;;
                "FIRST_FAILURE_REASON") first_reason=$value ;;
            esac
        done < "$stats_file"
        rm -f "$stats_file"
    fi

    echo "$total_failures,$step_mismatch,$token_mismatch,$logprob_mismatch,$first_reason"
}

verify_batch_invariant_ops() {
    echo "========================================"
    echo "Preflight: batch invariant custom op"
    echo "========================================"

    local preflight_exit_code
    if python3 "${ENVIRONMENT_ROOT}/scripts/check_batch_invariant_env.py" \
        --run-config "$RUN_CONFIG" \
        --report-json "$output_dir/preflight.json" \
        --timeout "${VLLM_TEST_OP_CHECK_TIMEOUT:-120s}"; then
        echo "[PASS] batch invariant custom op is available"
        return 0
    else
        preflight_exit_code=$?
    fi

    echo "ERROR: batch invariant custom op preflight failed."
    if [ "$preflight_exit_code" -eq 4 ]; then
        echo "BLOCKED_OPERATOR_INSTALLATION"
        echo "Automatic operator installation did not complete; inspect preflight.json and enter bounded recovery."
    elif [ "$preflight_exit_code" -eq 7 ]; then
        echo "BLOCKED_CHIP_DETECTION"
        echo "Check SOC_VERSION and npu-smi output; unknown chips must not default to A3."
    else
        echo "Install or rebuild the required batch invariant environment manually, then rerun the preflight."
    fi
    return "$preflight_exit_code"
}

verify_batch_invariant_mode() {
    if [ "$VLLM_BATCH_INVARIANT" != "1" ]; then
        echo "ERROR: BATCH_INVARIANT_MODE_REQUIRED: VLLM_BATCH_INVARIANT must be exactly 1; got '$VLLM_BATCH_INVARIANT'." >&2
        return 1
    fi
    echo "[PASS] VLLM_BATCH_INVARIANT=1"
}

write_combo_summary() {
    local scenario=$1
    local model_short=$2
    local tp=$3
    local temp=$4
    local mt=$5
    local mp=$6
    local plen=$7
    local bt=$8
    local pc=$9
    local status=${10}
    local required_rounds=${11}
    local attempts=${12}
    local total_failures=${13}
    local step_mismatch=${14}
    local token_mismatch=${15}
    local logprob_mismatch=${16}
    local reason=${17}
    local log_files=${18}
    local scheduler_status=${19}
    local kv_capacity=${20}
    local required_kv_tokens=${21}
    local max_kv_usage=${22}
    local preemptions=${23}
    local scheduler_reports=${24}
    local failure_manifests=${25}

    echo "$scenario,$model_short,TP,$tp,$temp,$mt,$mp,$plen,$bt,$pc,$scheduler_status,$kv_capacity,$required_kv_tokens,$max_kv_usage,$preemptions,$status,$required_rounds,$attempts,$total_failures,$step_mismatch,$token_mismatch,$logprob_mismatch,\"$reason\",\"$failure_manifests\",\"$scheduler_reports\",\"$log_files\"" >> "$summary_csv"
}

run_combo() {
    local scenario=$1
    local tp=$2
    local temp=$3
    local mt=$4
    local mp=$5
    local plen=$6
    local required_rounds=$7
    local pc=${8:-$ENABLE_PREFIX_CACHING}

    local model_short
    model_short=$(basename "$MODEL")
    local bt
    bt=$(calculate_batched_tokens "$mp" "$plen" "$mt")
    local combo_id="${scenario}_Model${model_short}_TP${tp}_T${temp}_MT${mt}_MP${mp}_PLEN${plen}_B${bt}_C${pc}_token_logprob"

    echo ""
    echo "========================================"
    echo "Running $scenario: $combo_id"
    echo "  Model: $MODEL"
    echo "  TP: $tp"
    echo "  Max prompts: $mp"
    echo "  Prompt length: $plen"
    echo "  Max tokens: $mt"
    echo "  Temperature: $temp"
    echo "  Compare metric: token_ids + chosen-token logprob"
    echo "  Required rounds: $required_rounds"
    echo "  BATCH_INVARIANCE_MEMORY_PLAN: requested=$(((plen + BATCHED_TOKENS_OFFSET) * mp)), cap=$AUTO_BATCHED_TOKENS_CAP, selected=$bt, minimum=$MODEL_MAX_LEN"
    echo "========================================"

    local consecutive_success=0
    local total_attempts=0
    local combo_status="UNKNOWN"
    local total_failures=0
    local step_mismatch=0
    local token_mismatch=0
    local logprob_mismatch=0
    local reason=""
    local log_files=""
    local scheduler_status="UNKNOWN"
    local kv_capacity=""
    local required_kv_tokens=""
    local max_kv_usage=""
    local preemptions=0
    local scheduler_reports=""
    local failure_manifests=""
    local transient_resource_retries=0

    export VLLM_TEST_MODEL="$MODEL"
    export VLLM_TEST_TP_SIZE="$tp"
    export VLLM_TEST_TEMPERATURE="$temp"
    export VLLM_TEST_MAX_TOKENS="$mt"
    export VLLM_TEST_MAX_PROMPT_NUM="$mp"
    export VLLM_TEST_MAX_NUM_BATCHED_TOKENS="$bt"
    export VLLM_TEST_MAX_MODEL_LEN="$MODEL_MAX_LEN"
    export VLLM_TEST_ENABLE_PREFIX_CACHING="$pc"
    export VLLM_TEST_MAX_PROMPT_LEN="$plen"
    while [ "$consecutive_success" -lt "$required_rounds" ]; do
        total_attempts=$((total_attempts + 1))
        local run_log="$output_dir/${combo_id}_attempt${total_attempts}.log"
        local run_failure_stats="$output_dir/${combo_id}_attempt${total_attempts}_failure_stats.txt"
        local run_failure_manifest="$output_dir/${combo_id}_attempt${total_attempts}_failure_manifest.json"
        if [ -z "$log_files" ]; then
            log_files="$run_log"
        else
            log_files="${log_files};${run_log}"
        fi

        echo "  Attempt $total_attempts/$required_rounds (consecutive successes: $consecutive_success/$required_rounds)"
        echo "  Log file: $run_log"
        export VLLM_TEST_SEED="$((SEED_BASE + consecutive_success))"
        export VLLM_TEST_FAILURE_STATS_FILE="$run_failure_stats"
        export VLLM_TEST_FAILURE_MANIFEST="$run_failure_manifest"
        rm -f "$run_failure_stats" "$run_failure_manifest"

        {
            echo "BATCH_INVARIANCE_ATTEMPT_START: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "Scenario: $scenario"
            echo "Model: $MODEL"
            echo "TP: $tp"
            echo "BS: $mp"
            echo "ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
        } | tee "$run_log"

        set +e
        pytest -v -s "$PYTEST_CASE" 2>&1 | tee -a "$run_log"
        local exit_code=${PIPESTATUS[0]}
        echo "BATCH_INVARIANCE_ATTEMPT_EXIT: exit_code=$exit_code" | tee -a "$run_log"
        set -e

        local run_status="UNKNOWN"
        local run_failures=0
        local run_step_mismatch=0
        local run_token_mismatch=0
        local run_logprob_mismatch=0
        local run_reason=""
        local scheduler_report="$output_dir/${combo_id}_attempt${total_attempts}_scheduler.json"
        local scheduler_tsv
        local scheduler_exit_code

        set +e
        scheduler_tsv=$(python3 "${TEST_ROOT}/scripts/analyze_scheduler_log.py" \
            --log "$run_log" \
            --expected-batch-size "$mp" \
            --static-batch \
            --max-sequence-length "$((plen + mt))" \
            --block-size 128 \
            --report "$scheduler_report" \
            --tsv)
        scheduler_exit_code=$?
        set -e

        local run_scheduler_status
        local run_kv_capacity
        local run_required_kv_tokens
        local run_max_kv_usage
        local run_preemptions=0
        local scheduler_reason
        IFS=$'\t' read -r run_scheduler_status run_kv_capacity \
            run_required_kv_tokens run_max_kv_usage scheduler_reason <<< "$scheduler_tsv"
        scheduler_status="$run_scheduler_status"
        kv_capacity="$run_kv_capacity"
        required_kv_tokens="$run_required_kv_tokens"
        max_kv_usage="$run_max_kv_usage"
        preemptions="$run_preemptions"
        if [ -z "$scheduler_reports" ]; then
            scheduler_reports="$scheduler_report"
        else
            scheduler_reports="${scheduler_reports};${scheduler_report}"
        fi

        if [ -f "$run_failure_manifest" ]; then
            if [ -z "$failure_manifests" ]; then
                failure_manifests="$run_failure_manifest"
            else
                failure_manifests="${failure_manifests};${run_failure_manifest}"
            fi
        fi

        if grep -qE '^BATCH_INVARIANCE_RESOURCE_BLOCKED:' "$run_log"; then
            run_status="SKIPPED_TP_RESOURCE_UNAVAILABLE"
            run_reason=$(grep -E '^BATCH_INVARIANCE_RESOURCE_BLOCKED:' "$run_log" | tail -1)
        elif grep -qiE "$TRANSIENT_RESOURCE_PATTERN" "$run_log"; then
            run_reason=$(grep -iE "$TRANSIENT_RESOURCE_PATTERN" "$run_log" | tail -1)
            if [ "$transient_resource_retries" -lt "$TRANSIENT_RESOURCE_RETRY_COUNT" ]; then
                run_status="RETRY_TRANSIENT_RESOURCE"
            else
                run_status="ERROR"
            fi
        elif grep -qE 'LLM initialization failed:' "$run_log"; then
            run_status="ERROR"
            run_reason=$(grep -E 'LLM initialization failed:' "$run_log" | tail -1)
        elif [ "$scheduler_exit_code" -ne 0 ]; then
            run_status="BLOCKED"
            run_reason="${run_scheduler_status}: ${scheduler_reason}"
        elif grep -q "Logits are not available on RequestOutput" "$run_log"; then
            run_status="SKIP"
            run_reason="Logprobs not available"
        elif grep -q "BATCH INVARIANCE FAILURES:" "$run_log"; then
            run_status="FAIL"
            local stats
            stats=$(parse_failure_stats "$run_failure_stats")
            IFS=',' read -r run_failures run_step_mismatch run_token_mismatch run_logprob_mismatch run_reason <<< "$stats"
            if [ -z "$run_failures" ] || [ "$run_failures" = "0" ]; then
                local failures_line
                failures_line=$(grep "BATCH INVARIANCE FAILURES:" "$run_log")
                run_failures=$(echo "$failures_line" | sed -E 's/.*BATCH INVARIANCE FAILURES: ([0-9]+)\/[0-9]+.*/\1/')
                run_reason=$(grep -E "Reason:" "$run_log" | head -1 | sed 's/.*Reason: //')
            fi
            if [ ! -f "$run_failure_manifest" ]; then
                run_reason="${run_reason}; structured failure manifest missing"
            fi
        elif [ "$exit_code" -eq 0 ]; then
            run_status="PASS"
        else
            run_status="ERROR"
            run_reason="Pytest exit code $exit_code"
        fi

        case "$run_status" in
            PASS)
                consecutive_success=$((consecutive_success + 1))
                echo "  Attempt $total_attempts PASSED"
                if [ "$consecutive_success" -lt "$required_rounds" ]; then
                    sleep "$POST_RUN_WAIT_TIME"
                fi
                ;;
            RETRY_TRANSIENT_RESOURCE)
                transient_resource_retries=$((transient_resource_retries + 1))
                echo "  Transient startup memory check failed; retry $transient_resource_retries/$TRANSIENT_RESOURCE_RETRY_COUNT after ${TRANSIENT_RESOURCE_RETRY_DELAY}s."
                sleep "$TRANSIENT_RESOURCE_RETRY_DELAY"
                ;;
            FAIL|BLOCKED|SKIP|SKIPPED_TP_RESOURCE_UNAVAILABLE|ERROR)
                combo_status="$run_status"
                reason="$run_reason"
                total_failures=$((total_failures + run_failures))
                step_mismatch=$((step_mismatch + run_step_mismatch))
                token_mismatch=$((token_mismatch + run_token_mismatch))
                logprob_mismatch=$((logprob_mismatch + run_logprob_mismatch))
                echo "  Attempt $total_attempts ended with $run_status, stopping $scenario."
                break
                ;;
            *)
                combo_status="ERROR"
                reason="Unknown run status: $run_status"
                break
                ;;
        esac
    done

    if [ "$consecutive_success" -eq "$required_rounds" ]; then
        combo_status="PASS"
        reason=""
    elif [ "$combo_status" = "UNKNOWN" ]; then
        combo_status="ERROR"
        reason="Did not reach $required_rounds consecutive PASS in $total_attempts attempts"
    fi

    write_combo_summary "$scenario" "$model_short" "$tp" "$temp" "$mt" "$mp" "$plen" "$bt" "$pc" "$combo_status" "$required_rounds" "$total_attempts" "$total_failures" "$step_mismatch" "$token_mismatch" "$logprob_mismatch" "$reason" "$log_files" "$scheduler_status" "$kv_capacity" "$required_kv_tokens" "$max_kv_usage" "$preemptions" "$scheduler_reports" "$failure_manifests"

    LAST_COMBO_STATUS="$combo_status"
    [ "$combo_status" = "PASS" ]
}

generate_summary_md() {
    python3 - "$summary_csv" "$output_dir/summary.md" <<'PY'
import csv
import pathlib
import sys

summary_csv = pathlib.Path(sys.argv[1])
summary_md = pathlib.Path(sys.argv[2])

with summary_csv.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

with summary_md.open("w", encoding="utf-8") as md:
    md.write("# Batch Invariance Summary\n\n")
    if rows:
        header = rows[0].keys()
        md.write("| " + " | ".join(header) + " |\n")
        md.write("|" + "|".join(["---"] * len(rows[0])) + "|\n")
        for row in rows:
            md.write("| " + " | ".join(row.get(k, "") for k in header) + " |\n")
    else:
        md.write("No batch invariance rows.\n")
PY
}

run_required_combo() {
    if run_combo "$@"; then
        return 0
    fi
    generate_summary_md
    if [ "$LAST_COMBO_STATUS" = "SKIPPED_TP_RESOURCE_UNAVAILABLE" ]; then
        return 2
    fi
    return 1
}

main() {
    cd "$TEST_ROOT"

    require_non_empty "ASCEND_RT_VISIBLE_DEVICES" "${ASCEND_RT_VISIBLE_DEVICES:-}"
    verify_batch_invariant_mode

    echo "========================================"
    echo "Batch Invariance Test"
    echo "Output dir: $output_dir"
    echo "Visible devices: $ASCEND_RT_VISIBLE_DEVICES"
    echo "Scenario: $SCENARIO"
    echo "Model: $MODEL"
    echo "Parallel mode: TP (supported)"
    echo "TP list: $TP_LIST"
    echo "Max prompt number list: $MAX_PROMPT_NUM_LIST"
    echo "Prompt/max_tokens pairs: $PROMPT_MAX_TOKENS_PAIRS"
    echo "Temperature: $TEMPERATURE"
    echo "Required rounds: $REQUIRED_ROUNDS"
    echo "Log stats interval: $VLLM_LOG_STATS_INTERVAL seconds"
    echo "========================================"

    require_non_empty "VLLM_TEST_MODEL" "$MODEL"
    require_non_empty "VLLM_TEST_RUN_CONFIG" "$RUN_CONFIG"
    require_non_empty "VLLM_TEST_TP_LIST" "$TP_LIST"
    require_non_empty "VLLM_TEST_MAX_PROMPT_NUM_LIST" "$MAX_PROMPT_NUM_LIST"
    require_non_empty "VLLM_TEST_PROMPT_MAX_TOKENS_PAIRS" "$PROMPT_MAX_TOKENS_PAIRS"
    if [[ ! "$SCENARIO" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: VLLM_TEST_SCENARIO may contain only letters, digits, '.', '_' and '-'." >&2
        return 1
    fi
    validate_positive_int "VLLM_TEST_REQUIRED_ROUNDS" "$REQUIRED_ROUNDS"
    validate_non_negative_int "VLLM_TEST_BATCHED_TOKENS_OFFSET" "$BATCHED_TOKENS_OFFSET"
    validate_non_negative_int "VLLM_TEST_SEED" "$SEED_BASE"
    validate_non_negative_int "VLLM_TEST_POST_RUN_WAIT_TIME" "$POST_RUN_WAIT_TIME"
    validate_non_negative_int "VLLM_TEST_TRANSIENT_RESOURCE_RETRY_COUNT" "$TRANSIENT_RESOURCE_RETRY_COUNT"
    validate_non_negative_int "VLLM_TEST_TRANSIENT_RESOURCE_RETRY_DELAY" "$TRANSIENT_RESOURCE_RETRY_DELAY"
    validate_positive_int "VLLM_TEST_AUTO_MAX_NUM_BATCHED_TOKENS_CAP" "$AUTO_BATCHED_TOKENS_CAP"
    validate_positive_int "VLLM_TEST_MAX_MODEL_LEN" "$MODEL_MAX_LEN"
    validate_positive_int "VLLM_TEST_LOG_STATS_INTERVAL" "$VLLM_LOG_STATS_INTERVAL"
    reject_unsupported_sampling_parameters
    if [ -n "$FIXED_MAX_NUM_BATCHED_TOKENS" ]; then
        validate_positive_int "VLLM_TEST_MAX_NUM_BATCHED_TOKENS" "$FIXED_MAX_NUM_BATCHED_TOKENS"
        if [ "$FIXED_MAX_NUM_BATCHED_TOKENS" -lt "$MODEL_MAX_LEN" ]; then
            echo "ERROR: VLLM_TEST_MAX_NUM_BATCHED_TOKENS ($FIXED_MAX_NUM_BATCHED_TOKENS) must be at least VLLM_TEST_MAX_MODEL_LEN ($MODEL_MAX_LEN) when chunked prefill is disabled." >&2
            return 1
        fi
    fi
    if [[ ! "$TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "ERROR: invalid VLLM_TEST_TEMPERATURE value: '$TEMPERATURE'." >&2
        return 1
    fi

    local tp_text
    tp_text=$(parse_positive_int_list "TP" "$TP_LIST")
    local max_prompt_num_text
    max_prompt_num_text=$(parse_positive_int_list "max prompt number" "$MAX_PROMPT_NUM_LIST")
    local pairs_text
    pairs_text=$(parse_prompt_max_tokens_pairs "$PROMPT_MAX_TOKENS_PAIRS")
    local tps=()
    local max_prompt_nums=()
    local pairs=()
    mapfile -t tps <<< "$tp_text"
    mapfile -t max_prompt_nums <<< "$max_prompt_num_text"
    mapfile -t pairs <<< "$pairs_text"

    verify_batch_invariant_ops

    local tp
    local max_prompt_num
    local pair
    local plen
    local mt
    local passed_combo_count=0
    local combo_exit_code
    for tp in "${tps[@]}"; do
        for max_prompt_num in "${max_prompt_nums[@]}"; do
            for pair in "${pairs[@]}"; do
                IFS=':' read -r plen mt <<< "$pair"
                if run_required_combo "$SCENARIO" "$tp" "$TEMPERATURE" "$mt" \
                    "$max_prompt_num" "$plen" "$REQUIRED_ROUNDS"; then
                    passed_combo_count=$((passed_combo_count + 1))
                else
                    combo_exit_code=$?
                    if [ "$combo_exit_code" -eq 2 ]; then
                        echo "Skipping remaining combinations for TP=$tp because the model weights cannot load with this TP."
                        break 2
                    fi
                    return "$combo_exit_code"
                fi
            done
        done
    done

    if [ "$passed_combo_count" -eq 0 ]; then
        echo "ERROR: Phase 2 has no runnable TP with completed acceptance evidence." >&2
        generate_summary_md
        return 1
    fi

    generate_summary_md

    echo ""
    echo "========================================"
    echo "Batch invariance scenario '$SCENARIO' passed."
    echo "Summary CSV: $summary_csv"
    echo "Markdown summary: $output_dir/summary.md"
    echo "========================================"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
