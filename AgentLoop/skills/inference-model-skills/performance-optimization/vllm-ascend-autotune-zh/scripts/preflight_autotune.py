#!/usr/bin/env python3
"""Require version/device confirmation before starting vLLM-Ascend tuning."""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version as distribution_version


DEFAULT_REFERENCE_DOC_URL = "https://docs.vllm.ai/projects/ascend/en/latest/"


def installed_version(distribution: str, module: str) -> str:
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        pass
    except Exception:
        pass

    try:
        package = importlib.import_module(module)
        return str(getattr(package, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001 - report the environment state
        return f"不可用（{exc.__class__.__name__}）"


def normalize_devices(raw: str) -> str:
    devices = [part.strip() for part in raw.split(",")]
    if not devices or any(not part.isdigit() for part in devices):
        raise ValueError("设备集合必须是逗号分隔的数字 ID，例如 4,5,6,7")
    return ",".join(devices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM-Ascend 寻优启动前强制版本/设备确认")
    parser.add_argument("--devices", required=True, help="设备集合，例如 4,5,6,7")
    parser.add_argument("--reference-doc-url", default=DEFAULT_REFERENCE_DOC_URL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        devices = normalize_devices(args.devices)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    vllm_version = installed_version("vllm", "vllm")
    ascend_version = installed_version("vllm-ascend", "vllm_ascend")

    print("=== vLLM-Ascend 寻优启动前强制预检 ===")
    print(f"当前 vLLM 版本: {vllm_version}")
    print(f"当前 vLLM-Ascend 版本: {ascend_version}")
    print(f"参考文档 URL: {args.reference_doc_url}")
    print(f"设备集合（将强制写入子进程）: ASCEND_RT_VISIBLE_DEVICES={devices}")
    print("在确认前不得启动基线、候选 benchmark 或推理 server。")

    try:
        answer = input("是否需要更改版本或设备集合？[y/N] ").strip().lower()
    except EOFError:
        print("未收到明确确认，已停止寻优。")
        return 2

    if answer in {"y", "yes", "是", "需要", "更改"}:
        print("请先更改目标环境或设备参数，然后重新运行本预检。")
        return 2
    if answer in {"n", "no", "否", "不", "继续", "确认"}:
        print("已确认沿用当前版本和设备集合，可以启动寻优。")
        return 0

    print("回答不明确，已停止寻优；请明确回答是否需要更改。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
