#!/usr/bin/env python3
"""Run a Ray actor NPU allocation smoke and print JSON evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def is_conda_base(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return "/envs/" not in normalized and any(
        marker in normalized for marker in ("/miniconda3/bin/python", "/anaconda3/bin/python")
    )


def run_smoke(python: str, timeout: int) -> dict[str, Any]:
    code = r'''
import json
result = {"status": "PASS", "errors": [], "evidence": []}
try:
    import ray
    import torch
    import torch_npu  # noqa: F401
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1, logging_level="ERROR")
    @ray.remote(resources={"NPU": 1})
    def probe():
        torch.npu.set_device(0)
        tensor = torch.ones(1, device="npu")
        return {
            "assigned_npu_ids": ray.get_runtime_context().get_accelerator_ids().get("NPU", []),
            "tensor_device": str(tensor.device),
        }
    value = ray.get(probe.remote())
    result["actor_result"] = value
    if not value.get("assigned_npu_ids") or "npu" not in value.get("tensor_device", "").lower():
        result["errors"].append("Ray task did not receive and use an NPU resource")
    else:
        result["evidence"].append("Ray task received an NPU resource and allocated an NPU tensor")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
finally:
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
if result["errors"]:
    result["status"] = "BLOCKED"
print(json.dumps(result, sort_keys=True))
'''
    try:
        completed = subprocess.run([python, "-c", code], text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"status": "BLOCKED", "errors": [f"Ray smoke timed out after {timeout}s"], "stdout_tail": str(exc.stdout or "")[-2000:], "stderr_tail": str(exc.stderr or "")[-2000:]}
    result: dict[str, Any] = {"status": "BLOCKED", "errors": ["Ray smoke did not emit JSON"]}
    for line in reversed(completed.stdout.splitlines()):
        if line.strip().startswith("{"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                pass
            break
    result.update({"selected_python": python, "exit_code": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]})
    if completed.returncode != 0:
        result["status"] = "BLOCKED"
        result.setdefault("errors", []).append(f"Ray smoke exited {completed.returncode}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.self_test:
        assert Path(sys.executable).is_file()
        print("SELF_TEST_PASS")
        return 0
    selected = str(Path(args.python).expanduser().resolve())
    if not Path(selected).is_file() or is_conda_base(selected):
        result = {"status": "BLOCKED", "errors": ["selected Python must be an existing non-base absolute path"]}
    else:
        result = run_smoke(selected, args.timeout)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
