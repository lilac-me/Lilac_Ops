#!/usr/bin/env python3
"""Run minimal torch_npu import, visibility, tensor, matmul, and backward checks."""

from __future__ import annotations

import json
import traceback


def main() -> int:
    result: dict = {"checks": []}
    try:
        import torch
        import torch_npu

        result["torch_version"] = torch.__version__
        result["torch_npu_version"] = getattr(torch_npu, "__version__", None)
        available = bool(torch.npu.is_available())
        count = int(torch.npu.device_count())
        result["npu_available"] = available
        result["npu_count"] = count
        if not available or count < 1:
            raise RuntimeError("NPU is not available")

        device = torch.device("npu:0")
        x = torch.randn(32, 32, device=device)
        y = torch.randn(32, 32, device=device)
        z = x @ y
        result["checks"].append({"name": "matmul", "shape": list(z.shape), "dtype": str(z.dtype)})

        layer = torch.nn.Linear(16, 8).to(device)
        inp = torch.randn(4, 16, device=device)
        loss = layer(inp).pow(2).mean()
        loss.backward()
        grad_finite = all(param.grad is not None and bool(torch.isfinite(param.grad).all().item()) for param in layer.parameters())
        result["checks"].append({"name": "forward_backward", "loss": float(loss.detach().cpu()), "grad_finite": grad_finite})
        if not grad_finite:
            raise RuntimeError("non-finite or missing gradient")

        result["status"] = "PASS"
        print(json.dumps(result, indent=2, sort_keys=True))
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
