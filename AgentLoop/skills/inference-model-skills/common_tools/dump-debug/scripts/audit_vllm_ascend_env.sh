#!/usr/bin/env bash
set +u

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "=== date / host ==="
date
hostname
uname -a

if [ -n "${CANN_SET_ENV:-}" ] && [ -f "${CANN_SET_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV}"
fi
if [ -n "${ATB_SET_ENV:-}" ] && [ -f "${ATB_SET_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${ATB_SET_ENV}"
fi

echo "=== python ==="
command -v "${PYTHON_BIN}" || true
"${PYTHON_BIN}" --version || true

echo "=== packages / actual imports ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import os

for module_name in ("vllm", "vllm_ascend", "torch", "torch_npu", "transformers"):
    try:
        module = __import__(module_name)
        print(f"{module_name}.file={getattr(module, '__file__', '<none>')}")
        print(f"{module_name}.version={getattr(module, '__version__', '<none>')}")
    except Exception as exc:
        print(f"{module_name}=ERROR {type(exc).__name__}: {exc}")

for dist_name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "triton-ascend"):
    try:
        print(f"dist.{dist_name}={md.version(dist_name)}")
    except Exception as exc:
        print(f"dist.{dist_name}=ERROR {exc}")

print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')}")
print(f"VLLM_TEST_MODEL={os.environ.get('VLLM_TEST_MODEL', '<unset>')}")
print(f"VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT', '<unset>')}")
print(f"VLLM_ENFORCE_EAGER={os.environ.get('VLLM_ENFORCE_EAGER', '<unset>')}")
print(f"http_proxy={os.environ.get('http_proxy', '<unset>')}")
print(f"https_proxy={os.environ.get('https_proxy', '<unset>')}")
PY

echo "=== git commits ==="
"${PYTHON_BIN}" - <<'PY'
import pathlib
import subprocess

for module_name in ("vllm", "vllm_ascend"):
    try:
        module = __import__(module_name)
        path = pathlib.Path(module.__file__).resolve()
    except Exception as exc:
        print(f"{module_name}: import failed: {exc}")
        continue
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            commit = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=False, text=True, capture_output=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(parent), "status", "--short"],
                check=False, text=True, capture_output=True,
            ).stdout.strip()
            print(f"{module_name}.git={parent}")
            print(f"{module_name}.commit={commit or '<unknown>'}")
            print(f"{module_name}.dirty={'yes' if status else 'no'}")
            if status:
                print(status[:2000])
            break
    else:
        print(f"{module_name}.git=<not found>")
PY

echo "=== CANN ==="
for candidate in "${CANN_ROOT:-}" /usr/local/Ascend /mnt/share/*/CANN /mnt/share/*/CANN/*; do
  [ -n "$candidate" ] || continue
  [ -e "$candidate" ] || continue
  echo "--- $candidate"
  find "$candidate" -maxdepth 4 -type f \( -name 'version.cfg' -o -name 'ascend_toolkit_install.info' \) -print 2>/dev/null | head -n 10
done

echo "=== NPU ==="
command -v npu-smi >/dev/null 2>&1 && npu-smi info || echo "npu-smi not found"