#!/bin/bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "请直接执行 setup_transformers_runtime.sh，不要 source。" >&2
  return 2
fi

TRANSFORMERS_REPOSITORY=${TRANSFORMERS_REPOSITORY:-https://github.com/huggingface/transformers.git}
TRANSFORMERS_COMMIT=${TRANSFORMERS_COMMIT:?请设置 TRANSFORMERS_COMMIT}
TRANSFORMERS_CHECKOUT=${TRANSFORMERS_CHECKOUT:?请设置 TRANSFORMERS_CHECKOUT}
TRANSFORMERS_VENV=${TRANSFORMERS_VENV:?请设置 TRANSFORMERS_VENV}
BASE_PYTHON=${BASE_PYTHON:-python3}
RUNTIME_REPORT=${RUNTIME_REPORT:-${TRANSFORMERS_VENV}/runtime_environment.json}

if [[ ! -d "${TRANSFORMERS_CHECKOUT}/.git" ]]; then
  git clone --no-checkout --filter=blob:none "${TRANSFORMERS_REPOSITORY}" "${TRANSFORMERS_CHECKOUT}"
fi
git -C "${TRANSFORMERS_CHECKOUT}" fetch --depth 1 origin "${TRANSFORMERS_COMMIT}"
git -C "${TRANSFORMERS_CHECKOUT}" checkout --detach FETCH_HEAD

"${BASE_PYTHON}" -m venv --system-site-packages "${TRANSFORMERS_VENV}"
PYTHONPATH= PYTHONHOME= PIP_REQUIRE_VIRTUALENV=true \
  "${TRANSFORMERS_VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir --upgrade \
  'huggingface-hub>=1.5.0,<2.0' 'safetensors>=0.8.0' 'tokenizers>=0.22.0,<=0.23.0' 'accelerate>=1.1.0'

PYTHONPATH="${TRANSFORMERS_CHECKOUT}/src" \
  "${TRANSFORMERS_VENV}/bin/python" - "${TRANSFORMERS_CHECKOUT}" "${RUNTIME_REPORT}" <<'PY'
import json
import pathlib
import sys
import torch
import transformers

checkout = pathlib.Path(sys.argv[1]).resolve()
report = pathlib.Path(sys.argv[2])
transformers_file = pathlib.Path(transformers.__file__).resolve()
if checkout not in transformers_file.parents:
    raise RuntimeError(f"transformers imported from {transformers_file}, expected {checkout}")
payload = {
    "python": sys.executable,
    "sys_prefix": sys.prefix,
    "sys_base_prefix": sys.base_prefix,
    "transformers_version": transformers.__version__,
    "transformers_file": str(transformers_file),
    "torch_version": torch.__version__,
}
try:
    import torch_npu
    payload["torch_npu_version"] = torch_npu.__version__
except Exception as exc:
    payload["torch_npu_error"] = repr(exc)
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
