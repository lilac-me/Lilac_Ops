#!/usr/bin/env python3
"""Run a minimal HCCL process-group smoke test under torchrun or equivalent."""

from __future__ import annotations

import json
import os
import traceback


def main() -> int:
    result = {
        "backend": "hccl",
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        result["status"] = "NOT_LAUNCHED"
        result["message"] = "Run with torchrun or a launcher that sets RANK and WORLD_SIZE."
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    try:
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401

        torch.npu.set_device(result["local_rank"])
        dist.init_process_group(backend="hccl")
        tensor = torch.tensor([result["rank"] + 1.0], device=f"npu:{result['local_rank']}")
        dist.all_reduce(tensor)
        dist.barrier()
        result["all_reduce_value"] = float(tensor.cpu().item())
        result["status"] = "PASS"
        print(json.dumps(result, indent=2, sort_keys=True))
        dist.destroy_process_group()
        return 0
    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAIL"
        result["error"] = type(exc).__name__
        result["message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
