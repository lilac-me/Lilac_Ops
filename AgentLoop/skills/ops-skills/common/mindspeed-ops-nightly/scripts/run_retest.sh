#!/usr/bin/env bash
# run_retest.sh — skill 复测（retest）编排（skill v0.6, 2026-09-01 新增）
# 场景: 日常/版本看护（daily/version）完成后, 用户下发"复测"命令。
# 只重跑上一轮失败项（UT 失败测试文件 + ATK 精度/性能失败算子）, 判定每项是否稳定复现。
# 判定: 稳定复现（再次失败）| 未复现（本次通过）| 无法判定（无结果/未执行/桶超时/文件缺失）。
#
# 用法: bash run_retest.sh [REF_DIR] [DATE]
#   REF_DIR  参考报告目录（默认 $WS/20*/ 下最新含 report.json 者, 即最近一轮看护; 一般无需指定）
#   DATE     默认今日; 复测产物 $WS/{DATE}_retest/（每天新建目录, 与当日日常看护 $WS/{DATE}/ 隔离）
# 阶段: init → ut_retest → acc_retest → perf_retest → summarize_retest（.phase 断点续跑, 换容器可续）
# 环境变量: CONTAINER（报告元信息, 默认 unknown）; RETEST_UT_TIMEOUT（UT 每桶限时秒, 默认 3600）
#       NCARDS（默认 6, 服务器方式按实际可用逻辑卡数覆盖）; ASCEND_RT_VISIBLE_DEVICES（默认 0..5）
# 说明: 复测不重装/不拉取（沿用当前已装源码 HEAD）; atk 命令与日常看护完全一致（--max_task 10 -to 1200）;
#       perf 复测不覆盖 $WS/triton/ 基准 xlsx。
WS="${WS:-/mnt/share/ops_nightly_agent}"
CONTAINER="${CONTAINER:-unknown}"
REF="${1:-}"
DATE="${2:-$(date +%Y%m%d)}"
export DATE
OUT="$WS/${DATE}_retest"
LOG="$WS/logs/run_retest_${DATE}_$(date +%H%M%S).log"
UT_TIMEOUT="${RETEST_UT_TIMEOUT:-3600}"
NCARDS="${NCARDS:-6}"
mkdir -p "$OUT/atk_logs" "$WS/logs"

mark() { echo "$1" > "$OUT/.phase"; }
phase() { [ "$(cat "$OUT/.phase" 2>/dev/null || echo none)" = "$1" ]; }

# 阶段运行 + 心跳（每 30s 写 OUT/heartbeat, 观测容器低占用时段）
run_stage() {
  local name="$1" fn="$2"; shift 2
  ( "$fn" "$@" ) </dev/null >> "$LOG" 2>&1 &
  local pid=$!
  ( while kill -0 "$pid" 2>/dev/null; do echo "$(date +%s) $name" >> "$OUT/heartbeat"; sleep 30; done ) &
  local hb=$!
  wait "$pid"; local rc=$?
  kill "$hb" 2>/dev/null || true
  return $rc
}

echo "[run_retest] REF=${REF:-<auto>} DATE=$DATE OUT=$OUT CONTAINER=$CONTAINER UT_TIMEOUT=$UT_TIMEOUT" | tee -a "$LOG"

# ---------- init: 定位参考报告 + 提取失败清单 ----------
if ! phase init; then
  echo "== phase: init ==" | tee -a "$LOG"
  if [ -z "$REF" ]; then
    for f in $WS/20*/report.json; do
      [ -f "$f" ] || continue
      d="${f%/report.json}"
      if [ -z "$REF" ] || [ "$d" -nt "$REF" ]; then REF="$d"; fi
    done
  fi
  if [ -z "$REF" ] || [ ! -f "$REF/report.json" ]; then
    echo "[run_retest] 找不到参考报告（REF=${REF:-空}）"; exit 1
  fi
  source "$WS/env.sh"
  TOTAL=$(python - "$REF" "$OUT" <<'PYEOF' 2>>"$LOG"
import json, sys
ref, out = sys.argv[1], sys.argv[2]
r = json.load(open(ref + "/report.json", encoding="utf-8"))
ut = [f["file"] for f in (r.get("ut") or {}).get("failed_files") or []]
acc = [f["op"] for f in (r.get("atk_accuracy") or {}).get("failed_ops") or []]
perf = [f["op"] for f in (r.get("atk_performance") or {}).get("failed_ops") or []]
for name, lst in (("ut_failed_files.txt", ut), ("acc_failed_ops.txt", acc), ("perf_failed_ops.txt", perf)):
    open(out + "/" + name, "w", encoding="utf-8").write("\n".join(lst) + ("\n" if lst else ""))
print(len(ut) + len(acc) + len(perf))
PYEOF
)
  echo "[run_retest] REF=$REF 失败项清单: $(wc -l < "$OUT/ut_failed_files.txt") UT 文件 / $(wc -l < "$OUT/acc_failed_ops.txt") ACC / $(wc -l < "$OUT/perf_failed_ops.txt") PERF" | tee -a "$LOG"
  if [ "${TOTAL:-0}" = "0" ]; then
    echo "[run_retest] 参考报告无失败项, 无需复测" | tee -a "$LOG"
    mark init; mark summarize_retest
    exit 0
  fi
  mark init
fi

# ---------- ut_retest: 只跑失败测试文件, 6 桶并行, 每桶 timeout 防挂起 ----------
ut_retest() {
  source "$WS/env.sh"
  cd "$WS/MindSpeed-Ops/tests" || return 1
  export SOC_VERSION=ascend910_93
  export PYTHONPATH="$WS/MindSpeed-Ops${PYTHONPATH:+:$PYTHONPATH}"
  local T0=$(date +%s)
  local RESULT="$OUT/result_ut_retest.log"
  : > "$RESULT"
  local -a FILES=()
  mapfile -t FILES < "$OUT/ut_failed_files.txt" 2>/dev/null || true
  if [ "${#FILES[@]}" -eq 0 ]; then
    echo "=== ut_retest: 无失败文件（跳过） ===" > "$RESULT"
    echo "ut 0" >> "$OUT/timing.txt"
    return 0
  fi
  local -a EXIST=() f
  for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then EXIST+=("$f"); else echo "SKIP missing file: $f" >> "$RESULT"; fi
  done
  if [ "${#EXIST[@]}" -eq 0 ]; then
    echo "=== ut_retest: 所有失败文件均缺失（跳过运行） ===" >> "$RESULT"
    echo "ut 0" >> "$OUT/timing.txt"
    return 0
  fi
  local -a BUCKETS
  local i idx=0
  for i in $(seq 0 $((NCARDS-1))); do BUCKETS[$i]=""; done
  for f in "${EXIST[@]}"; do
    BUCKETS[$((idx % NCARDS))]+=" $f"
    idx=$((idx+1))
  done
  local -a pids=()
  for i in $(seq 0 $((NCARDS-1))); do
    [ -n "${BUCKETS[$i]// }" ] || continue
    ( export ASCEND_RT_VISIBLE_DEVICES=$i
      if command -v timeout >/dev/null 2>&1; then
        timeout "$UT_TIMEOUT" pytest ${BUCKETS[$i]} -p no:cacheprovider > "$OUT/ut_rpart$i.log" 2>&1
      else
        pytest ${BUCKETS[$i]} -p no:cacheprovider > "$OUT/ut_rpart$i.log" 2>&1
      fi ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  python - "$OUT" "$NCARDS" "${EXIST[@]}" <<'PYEOF' >> "$RESULT"
import os, re, sys
date_dir, n = sys.argv[1], int(sys.argv[2])
files = sys.argv[3:]
tot_f = tot_p = tot_s = 0
failed, seen, per_part, part_files = [], set(), [], []
buckets = {i: [] for i in range(n)}
for idx, f in enumerate(files):
    buckets[idx % n].append(f)
for i in range(n):
    p = os.path.join(date_dir, "ut_rpart%d.log" % i)
    if not os.path.exists(p):
        continue
    c = open(p, encoding="utf-8", errors="ignore").read()
    for cc in re.findall(r"^FAILED\s+(.*)$", c, re.MULTILINE):
        if cc not in seen:
            seen.add(cc); failed.append(cc)
    def lastnum(pat):
        ms = re.findall(pat, c)
        return int(ms[-1]) if ms else 0
    f_ = lastnum(r"(\d+)\s+failed\b"); p_ = lastnum(r"(\d+)\s+passed\b"); s_ = lastnum(r"(\d+)\s+skipped\b")
    has_summary = f_ or p_ or s_ or ("error" in c and re.search(r"\d+\s+error", c))
    if f_ or p_ or s_:
        per_part.append("rpart%d: %d failed, %d passed, %d skipped" % (i, f_, p_, s_))
        tot_f += f_; tot_p += p_; tot_s += s_
    elif has_summary:
        per_part.append("rpart%d: 无 failed/passed 计数（had error）" % i)
    elif c.strip():
        per_part.append("rpart%d: 无汇总行（超时/中断, FAILED 行 %d）" % (i, c.count("FAILED ")))
    if buckets[i]:
        part_files.append("rpart_files%d: %s" % (i, " ".join(buckets[i])))
out = ["###### UT RETEST (%d buckets) ######" % n]
out += per_part
out += part_files
out.append("###### FAILED CASES (retest) ######")
out += ["FAILED " + cc for cc in failed]
out.append("===== UT_RETEST TOTAL: %d failed, %d passed, %d skipped =====" % (tot_f, tot_p, tot_s))
open(os.path.join(date_dir, "result_ut_retest.log"), "a", encoding="utf-8").write("\n".join(out) + "\n")
print("UT_RETEST_MERGED total_failed=%d total_passed=%d" % (tot_f, tot_p))
PYEOF
  echo "ut $(( $(date +%s) - T0 ))" >> "$OUT/timing.txt"
  echo "=== ut_retest done ==="
}

# ---------- acc/perf 复测（共享实现, 命令与日常看护完全一致, NCARDS 并发池） ----------
atk_retest_stage() {
  local KIND="$1"   # acc | perf
  source "$WS/env.sh"
  cd "$WS/MindSpeed-Ops/tests" || return 1
  export SOC_VERSION=ascend910_93
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}"
  export PYTHONPATH="$WS/MindSpeed-Ops${PYTHONPATH:+:$PYTHONPATH}"
  local TARGET_DIR="atk_tests/triton"
  local TKEY="acc"; [ "$KIND" = "perf" ] && TKEY="perf"
  local RESULT="$OUT/result_atk_${KIND}_retest.log"
  : > "$RESULT"
  local T0=$(date +%s)
  local -a NAMES=()
  mapfile -t NAMES < "$OUT/${KIND}_failed_ops.txt" 2>/dev/null || true
  if [ "${#NAMES[@]}" -eq 0 ]; then
    echo "=== ${KIND}_retest: 无失败算子（跳过） ===" > "$RESULT"
    echo "$TKEY 0" >> "$OUT/timing.txt"
    return 0
  fi
  local -a V=() nm y g t
  for nm in "${NAMES[@]}"; do
    y="${TARGET_DIR}/${nm}/${nm}.yaml"; g="${TARGET_DIR}/${nm}/generate_${nm}.py"; t="${TARGET_DIR}/${nm}/triton_${nm}.py"
    if [ -f "$y" ] && [ -f "$g" ] && [ -f "$t" ]; then V+=("$nm"); else echo "SKIP op 三件套缺失: $nm" >> "$RESULT"; fi
  done
  if [ "${#V[@]}" -eq 0 ]; then
    echo "=== ${KIND}_retest: 无有效算子（跳过运行） ===" >> "$RESULT"
    echo "$TKEY 0" >> "$OUT/timing.txt"
    return 0
  fi
  # 慢 op 先入池（读最近一轮同款 op_timing 文件, 秒数降序）
  local TIMING_REF="" f
  for f in $WS/20*/op_timing_${TKEY}.txt; do
    [ -f "$f" ] || continue
    if [ -z "$TIMING_REF" ] || [ "$f" -nt "$TIMING_REF" ]; then TIMING_REF="$f"; fi
  done
  local -A OP_SEC=()
  if [ -n "$TIMING_REF" ]; then
    while read -r _ op sec; do [ -n "$op" ] && [ "$sec" -eq "$sec" ] 2>/dev/null && OP_SEC[$op]=$sec; done < "$TIMING_REF"
  fi
  local -a SORTED=()
  mapfile -t SORTED < <(for n in "${V[@]}"; do echo "${OP_SEC[$n]:-0} $n"; done | sort -k1,1 -rn | awk '{print $2}')
  # NCARDS 并发作业池
  local -a pids=()
  local idx=-1 dir_name LOG_FILE yaml_file py_script triton_file device_id
  local BACKEND="cpu"; local TASK="accuracy"; local GREP="acc_pass_result"
  if [ "$KIND" = "perf" ]; then BACKEND="npu"; TASK="performance_device"; GREP="is_device_perf_pass"; fi
  for dir_name in "${SORTED[@]}"; do
    idx=$((idx+1))
    LOG_FILE="$OUT/atk_logs/retest_${KIND}_${dir_name}.log"
    yaml_file="${TARGET_DIR}/${dir_name}/${dir_name}.yaml"
    py_script="${TARGET_DIR}/${dir_name}/generate_${dir_name}.py"
    triton_file="${TARGET_DIR}/${dir_name}/triton_${dir_name}.py"
    while :; do
      local ALIVE=0 p NEW=()
      for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && { ALIVE=$((ALIVE+1)); NEW+=("$p"); }; done
      pids=("${NEW[@]}")
      [ "$ALIVE" -lt "$NCARDS" ] && break
      wait -n 2>/dev/null || sleep 2
    done
    device_id=$((idx % NCARDS))
    rm -rf "result/${dir_name}"
    if ! atk case -f "$yaml_file" -p "$py_script" > /dev/null 2>&1; then
      echo "[$dir_name] case gen FAILED" >> "$RESULT"
      continue
    fi
    (
      local t_op=$(date +%s)
      atk node --backend triton --devices "$device_id" node --backend "$BACKEND" --devices "$device_id" \
        task --max_task 10 -c "result/${dir_name}/json/all_${dir_name}.json" --task "$TASK" \
        -tup ./ -p "$triton_file" -to 1200 > "$LOG_FILE" 2>&1
      echo "$TKEY $dir_name $(( $(date +%s) - t_op ))" >> "$OUT/op_timing_${TKEY}.txt"
    ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
  # 结果提取（按执行顺序; case gen FAILED 的 op 无 status → NO_RESULT 由脚本标注）
  for dir_name in "${SORTED[@]}"; do
    LOG_FILE="$OUT/atk_logs/retest_${KIND}_${dir_name}.log"
    local STATUS
    STATUS=$(grep -oP "${GREP}:\K[^\s]+" "$LOG_FILE" | tail -1)
    if [ -z "$STATUS" ]; then STATUS="NO_RESULT"; fi
    echo "[$dir_name] status: $STATUS" >> "$RESULT"
  done
  echo "$TKEY $(( $(date +%s) - T0 ))" >> "$OUT/timing.txt"
  echo "=== ${KIND}_retest done ==="
}

# ---------- summarize_retest: 稳定复现判定 + retest 报告 ----------
summarize_retest() {
  source "$WS/env.sh"
  python "$WS/scripts/summarize_retest.py" "$OUT" "$REF" "$CONTAINER"
}

# ---------- 阶段编排（.phase 断点续跑） ----------
if ! phase ut_retest; then
  echo "== phase: ut_retest ==" | tee -a "$LOG"
  run_stage ut_retest ut_retest || true
  mark ut_retest
fi
if ! phase acc_retest; then
  echo "== phase: acc_retest ==" | tee -a "$LOG"
  run_stage acc_retest atk_retest_stage acc || true
  mark acc_retest
fi
if ! phase perf_retest; then
  echo "== phase: perf_retest ==" | tee -a "$LOG"
  run_stage perf_retest atk_retest_stage perf || true
  mark perf_retest
fi
if ! phase summarize_retest; then
  echo "== phase: summarize_retest ==" | tee -a "$LOG"
  run_stage summarize summarize_retest || { echo "summarize_retest 失败"; exit 1; }
  mark summarize_retest
fi

echo "[run_retest] ALL DONE — 复测报告: $OUT/retest_report.md, retest_report.json" | tee -a "$LOG"
