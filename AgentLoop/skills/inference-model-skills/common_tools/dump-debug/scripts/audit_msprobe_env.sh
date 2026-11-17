#!/usr/bin/env bash
# Collect the environment actually used by msProbe.

set +u

usage() {
    cat <<'EOF'
Usage: audit_msprobe_env.sh [options]

Options:
  --env-script PATH   Source a CANN/NNAL/Python activation script; repeatable.
  --python PATH       Python executable. Default: python3
  --msprobe PATH      msprobe CLI path. Default: msprobe
  --repo PATH         Optional msProbe repository path.
  -h, --help          Show this help.
EOF
}

PYTHON_BIN="python3"
MSPROBE_BIN="msprobe"
REPO_PATH=""
ENV_SCRIPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-script) ENV_SCRIPTS+=("$2"); shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --msprobe) MSPROBE_BIN="$2"; shift 2 ;;
        --repo) REPO_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

echo "== timestamp =="
date '+%F %T %z'

echo "== os =="
cat /etc/os-release 2>/dev/null | head -n 8
uname -a

echo "== env scripts =="
for script in "${ENV_SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        echo "source $script"
        # CANN scripts reference optional variables and are not safe under set -u.
        source "$script"
    else
        echo "MISSING $script"
    fi
done

echo "== executables =="
command -v "$PYTHON_BIN" || true
command -v "$MSPROBE_BIN" || true
"$PYTHON_BIN" -V 2>&1 || true
"$MSPROBE_BIN" --version 2>&1 | head -n 20 || true

echo "== python packages =="
"$PYTHON_BIN" - <<'PY'
import importlib.util
from importlib.metadata import version, PackageNotFoundError

for name in ("mindstudio-probe", "torch", "torch-npu", "transformers", "triton-ascend"):
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name}: NOT_INSTALLED")

spec = importlib.util.find_spec("msprobe")
print(f"msprobe_import_path: {getattr(spec, 'origin', None)}")
try:
    import torch
    print(f"torch_runtime: {torch.__version__}")
except Exception as exc:
    print(f"torch_runtime: ERROR {exc}")
try:
    import torch_npu
    print(f"torch_npu_runtime: {torch_npu.__version__}")
except Exception as exc:
    print(f"torch_npu_runtime: ERROR {exc}")
PY

echo "== msprobe availability =="
if "$PYTHON_BIN" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("msprobe") else 1)' 2>/dev/null; then
    echo "msprobe_importable: yes"
else
    echo "msprobe_importable: no"
    echo "install: cd \"\$HOME\" && git clone https://gitcode.com/Ascend/msprobe.git -b master && cd msprobe && python3 build.py && python3 -m pip install artifacts/mindstudio_probe*.whl"
    echo "or: $PYTHON_BIN -m pip install mindstudio-probe"
fi

echo "== msprobe repo =="
if [[ -n "$REPO_PATH" ]]; then
    if [[ -d "$REPO_PATH/.git" ]]; then
        git -C "$REPO_PATH" rev-parse HEAD 2>&1 || true
        git -C "$REPO_PATH" log -1 --format='%H %ci %s' 2>&1 || true
        echo "-- status --"
        git -C "$REPO_PATH" status --short 2>&1 | head -n 50 || true
    else
        echo "NOT_A_GIT_REPO $REPO_PATH"
    fi
else
    echo "not provided; pass --repo /path/to/msprobe"
fi

echo "== canonical files =="
CANON="$("$PYTHON_BIN" - <<'PY'
import importlib.util
spec = importlib.util.find_spec("msprobe")
print(spec.origin if spec and spec.origin else "")
PY
)"
if [[ -n "$CANON" ]]; then
    PKG="$(dirname "$CANON")"
    echo "package_dir=$PKG"
    echo "function_factory=$PKG/pytorch/dump/function_factory.py"
    echo "bench_functions=$PKG/pytorch/bench_functions"
    ls -1 "$PKG/pytorch/bench_functions" 2>/dev/null | head -n 100 || true
fi

echo "== registered fusion benches =="
"$PYTHON_BIN" - <<'PY'
try:
    from msprobe.pytorch.dump.function_factory import npu_custom_functions
    names = list(npu_custom_functions.keys())
    print("registered_count:", len(names))
    print("registered:", ",".join(sorted(names)))
    print("fia_registered:", "npu_fused_infer_attention_score" in names)
except Exception as exc:
    print(f"ERROR: {exc}")
PY

echo "== npu =="
npu-smi info 2>/dev/null | head -n 40 || echo "npu-smi unavailable"