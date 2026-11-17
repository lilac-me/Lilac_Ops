#!/usr/bin/env python3
"""Run a focused vLLM-Ascend generation smoke and print JSON evidence."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def module_version(name: str) -> str | None:
    try:
        return metadata.version(name.replace("_", "-"))
    except Exception:
        try:
            module = importlib.import_module(name)
        except Exception:
            return None
        return str(getattr(module, "__version__", "unknown"))


def release_line(value: str | None) -> str:
    match = re.match(r"^\s*(\d+\.\d+\.\d+)", str(value or ""))
    return match.group(1) if match else str(value or "").split("+", 1)[0].strip()


def expect_version(result: dict[str, Any], key: str, expected: str, label: str) -> bool:
    if not expected or expected.lower() == "auto":
        return True
    actual = str(result.get(key) or "")
    if release_line(actual) == release_line(expected):
        result["evidence"].append(f"{label} release line matches {expected}: {actual}")
        return True
    result["error"] = f"{label} version mismatch: expected {expected}, got {actual}"
    return False


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked_at": utc_now(),
        "status": "BLOCKED",
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dry_run": bool(args.dry_run),
        "evidence": [],
    }
    if args.dry_run:
        result.update({"status": "DRY_RUN", "planned_action": "import NPU and vLLM packages, select NPU device, instantiate vllm.LLM, generate text"})
        return result
    try:
        import torch  # type: ignore
        import torch_npu  # type: ignore  # noqa: F401

        result["torch_version"] = str(torch.__version__)
        result["torch_npu_version"] = module_version("torch_npu")
        torch.npu.set_device(0)
        result["npu_device_count"] = int(torch.npu.device_count())
        result["evidence"].append("selected torch NPU logical device 0")
    except Exception as exc:
        result["error"] = f"base NPU gate failed: {type(exc).__name__}: {exc}"
        return result
    if not expect_version(result, "torch_version", args.expected_torch_version, "torch"):
        return result
    if not expect_version(result, "torch_npu_version", args.expected_torch_npu_version, "torch_npu"):
        return result

    try:
        from vllm import LLM, SamplingParams  # type: ignore

        result["vllm_version"] = module_version("vllm")
        result["vllm_ascend_version"] = module_version("vllm_ascend")
        if result["vllm_ascend_version"] is None:
            raise ImportError("vllm_ascend is not importable")
    except Exception as exc:
        result["error"] = f"vLLM import failed: {type(exc).__name__}: {exc}"
        return result
    if not expect_version(result, "vllm_version", args.expected_vllm_version, "vllm"):
        return result
    if not expect_version(result, "vllm_ascend_version", args.expected_vllm_ascend_version, "vllm_ascend"):
        return result

    model = Path(args.model).expanduser().resolve()
    if not model.exists():
        result["error"] = f"model path does not exist: {model}"
        return result
    try:
        kwargs: dict[str, Any] = {
            "model": str(model),
            "trust_remote_code": args.trust_remote_code,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
        }
        if args.enforce_eager:
            kwargs["enforce_eager"] = True
        llm = LLM(**kwargs)
        outputs = llm.generate([args.prompt], SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature))
        generated = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        result.update({"status": "PASS", "generated_text": generated})
        result["evidence"].append("vLLM generation completed on the selected NPU environment")
    except Exception as exc:
        result["error"] = f"vLLM generation failed: {type(exc).__name__}: {exc}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt", default="Hello from Ascend NPU.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-torch-version", default="")
    parser.add_argument("--expected-torch-npu-version", default="")
    parser.add_argument("--expected-vllm-version", default="")
    parser.add_argument("--expected-vllm-ascend-version", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def self_test() -> int:
    assert release_line("2.10.0") == release_line("2.10.0.post4")
    args = parse_args()
    args.self_test = False
    args.dry_run = True
    result = run_smoke(args)
    if result["status"] != "DRY_RUN":
        return 1
    print("SELF_TEST_PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    result = run_smoke(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
