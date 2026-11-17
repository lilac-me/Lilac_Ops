#!/usr/bin/env python3
"""Collect local CANN, NPU, torch, and torch_npu facts as validation evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], timeout: int = 20) -> dict:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": type(exc).__name__, "message": str(exc)}


def py_probe(code: str) -> dict:
    return run([sys.executable, "-c", code])


def find_env_scripts(cann_path: Path | None) -> list[str]:
    if not cann_path or not cann_path.exists():
        return []
    names = {"set_env.sh", "ascend_toolkit_install.info"}
    matches = []
    for path in cann_path.rglob("*"):
        if path.name in names:
            matches.append(str(path))
        if len(matches) >= 40:
            break
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-path", type=Path)
    args = parser.parse_args()

    cann_path = args.cann_path.expanduser().resolve() if args.cann_path else None
    npu_smi = shutil.which("npu-smi")
    result = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "cann": {
            "path": str(cann_path) if cann_path else None,
            "path_exists": bool(cann_path and cann_path.exists()),
            "env_scripts": find_env_scripts(cann_path),
            "env": {key: os.environ.get(key) for key in ["ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"]},
        },
        "commands": {
            "npu_smi_info": run([npu_smi, "info"]) if npu_smi else {"error": "npu-smi not found"},
        },
        "python_probes": {
            "torch": py_probe("import torch, json; print(json.dumps({'version': torch.__version__}))"),
            "torch_npu": py_probe("import torch_npu, json; print(json.dumps({'version': getattr(torch_npu, '__version__', None)}))"),
            "npu": py_probe(
                "import torch, torch_npu, json; "
                "print(json.dumps({'available': torch.npu.is_available(), 'count': torch.npu.device_count()}))"
            ),
        },
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
