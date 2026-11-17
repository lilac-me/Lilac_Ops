#!/usr/bin/env bash
# Run msProbe acc_check for one dump.json or all rank*/dump.json under a directory.

set +u

usage() {
    cat <<'EOF'
Usage: run_msprobe_api_precheck.sh (--api-info FILE | --dump-root DIR) --out-dir DIR [options]

Options:
  --api-info FILE          A single .../rankN/dump.json.
  --dump-root DIR          Search recursively for rank*/dump.json.
  --out-dir DIR            Output root; one subdirectory per rank.
  --env-script PATH        Source CANN/NNAL activation script; repeatable.
  --python PATH            Python executable. Default: python3
  --msprobe PATH           msprobe CLI. Default: msprobe
  --runner NAME            acc_check or multi_acc_check. Default: acc_check
  --num-splits N           multi_acc_check -n value. Default: 8
  --device IDS             multi_acc_check -d value, for example "0 1".
  --no-save-error-data     Do not pass -save_error_data.
  -h, --help               Show this help.
EOF
}

API_INFO=""
DUMP_ROOT=""
OUT_DIR=""
PYTHON_BIN="python3"
MSPROBE_BIN="msprobe"
RUNNER="acc_check"
NUM_SPLITS="8"
DEVICES=""
SAVE_ERROR_DATA=1
ENV_SCRIPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-info) API_INFO="$2"; shift 2 ;;
        --dump-root) DUMP_ROOT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --env-script) ENV_SCRIPTS+=("$2"); shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --msprobe) MSPROBE_BIN="$2"; shift 2 ;;
        --runner) RUNNER="$2"; shift 2 ;;
        --num-splits) NUM_SPLITS="$2"; shift 2 ;;
        --device) DEVICES="$2"; shift 2 ;;
        --no-save-error-data) SAVE_ERROR_DATA=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    echo "--out-dir is required" >&2
    usage
    exit 2
fi
if [[ -z "$API_INFO" && -z "$DUMP_ROOT" ]]; then
    echo "provide --api-info or --dump-root" >&2
    usage
    exit 2
fi
if [[ -n "$API_INFO" && -n "$DUMP_ROOT" ]]; then
    echo "use either --api-info or --dump-root, not both" >&2
    exit 2
fi
if [[ "$RUNNER" != "acc_check" && "$RUNNER" != "multi_acc_check" ]]; then
    echo "--runner must be acc_check or multi_acc_check" >&2
    exit 2
fi

for script in "${ENV_SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        echo "source $script"
        source "$script"
    else
        echo "MISSING env script: $script" >&2
        exit 2
    fi
done

if ! command -v "$MSPROBE_BIN" >/dev/null 2>&1; then
    echo "msprobe CLI not found: $MSPROBE_BIN" >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python not found: $PYTHON_BIN" >&2
    exit 2
fi

FILES=()
if [[ -n "$API_INFO" ]]; then
    FILES=("$API_INFO")
else
    mapfile -t FILES < <(find "$DUMP_ROOT" -type f -name dump.json | sort -V)
fi
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "no dump.json found" >&2
    exit 3
fi

mkdir -p "$OUT_DIR"
FAILED=0
for api in "${FILES[@]}"; do
    if [[ ! -f "$api" ]]; then
        echo "MISSING $api" >&2
        FAILED=1
        continue
    fi
    rank="$(basename "$(dirname "$api")")"
    rank_out="$OUT_DIR/$rank"
    mkdir -p "$rank_out"
    log="$rank_out/run.log"

    cmd=("$MSPROBE_BIN" "$RUNNER" -api_info "$api" -o "$rank_out")
    if [[ "$RUNNER" == "multi_acc_check" ]]; then
        cmd+=(-n "$NUM_SPLITS")
        if [[ -n "$DEVICES" ]]; then
            # DEVICES is intentionally whitespace-separated, for example "0 1".
            read -r -a device_array <<< "$DEVICES"
            cmd+=(-d "${device_array[@]}")
        fi
    fi
    if [[ "$SAVE_ERROR_DATA" == "1" ]]; then
        cmd+=(-save_error_data)
    fi

    echo "== $rank =="
    printf 'command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}" > "$log" 2>&1
    rc=$?
    echo "exit_code=$rc log=$log"
    if [[ $rc -ne 0 ]]; then
        FAILED=1
        tail -n 80 "$log" || true
    fi
done

echo "== summary =="
"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import collections
import csv
import glob
import os
import sys

root = sys.argv[1]
result_files = sorted(glob.glob(os.path.join(root, "**", "accuracy_checking_result_*.csv"), recursive=True))
detail_files = sorted(glob.glob(os.path.join(root, "**", "accuracy_checking_details_*.csv"), recursive=True))
if not result_files:
    print("no accuracy_checking_result_*.csv found")
    raise SystemExit(0)

for path in result_files:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    status = collections.Counter(row.get("Forward Test Success", "") for row in rows)
    print(f"{path}: {dict(status)}")
    for row in rows:
        value = row.get("Forward Test Success", "")
        if value.lower() in ("warning", "error", "skip"):
            print(f"  {value.upper():7} {row.get('API name', '')} {row.get('Message', '') or ''}".rstrip())

if detail_files:
    print("details:")
    for path in detail_files:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        status = collections.Counter(row.get("Status", "") for row in rows)
        print(f"{path}: {dict(status)}")
PY

exit $FAILED