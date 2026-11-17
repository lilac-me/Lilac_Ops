#!/usr/bin/env python3
"""Shared payload, hashing, layout, and serialization helpers for logits alignment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


FORMAT_VERSION = 1


def safe_torch_load(path: str | Path) -> Any:
    """Load a tensor-only payload on CPU without enabling arbitrary pickle objects."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with PyTorch releases that predate weights_only.
        return torch.load(path, map_location="cpu")


def input_sha256(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
    """Hash tensor shapes and contiguous int64 bytes using the shared artifact contract."""
    digest = hashlib.sha256()
    for tensor in (input_ids, attention_mask):
        value = tensor.detach().cpu().to(torch.int64).contiguous()
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def normalize_fixed_input(
    payload: Mapping[str, Any],
    *,
    require_right_padding: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Validate and normalize the shared input payload to contiguous CPU int64 tensors."""
    if "input_ids" not in payload:
        raise KeyError("The fixed input payload is missing input_ids.")

    input_ids = payload["input_ids"]
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)
    input_ids = input_ids.detach().cpu().to(torch.long).contiguous()
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.numel() == 0:
        raise ValueError(f"input_ids must be non-empty [B,S], got {tuple(input_ids.shape)}")
    if torch.any(input_ids < 0):
        raise ValueError("input_ids contains negative token IDs.")

    attention_mask = payload.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    elif not torch.is_tensor(attention_mask):
        attention_mask = torch.tensor(attention_mask)
    attention_mask = attention_mask.detach().cpu().to(torch.long).contiguous()
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask and input_ids must have the same shape.")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("attention_mask must contain only 0 and 1.")
    if torch.any(attention_mask.sum(dim=-1) == 0):
        raise ValueError("Every input row must contain at least one valid token.")
    if require_right_padding:
        seen_padding = attention_mask.eq(0).cummax(dim=-1).values
        if torch.any(seen_padding & attention_mask.eq(1)):
            raise ValueError("Only no padding or right padding is supported by this baseline.")

    digest = input_sha256(input_ids, attention_mask)
    recorded_digest = payload.get("input_sha256")
    if recorded_digest is not None and recorded_digest != digest:
        raise ValueError(
            f"input_sha256={recorded_digest!r} does not match recomputed digest={digest!r}."
        )
    return input_ids, attention_mask, digest


def load_fixed_input(
    path: str | Path,
    *,
    require_right_padding: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Fixed input does not exist: {input_path}")
    payload = safe_torch_load(input_path)
    if not isinstance(payload, dict):
        raise TypeError("The fixed input must be a dictionary payload.")
    input_ids, attention_mask, digest = normalize_fixed_input(
        payload,
        require_right_padding=require_right_padding,
    )
    return input_ids, attention_mask, digest, payload


def normalize_logits_layout(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    requested_layout: str,
) -> torch.Tensor:
    """Normalize 3-D logits to [batch, sequence, vocabulary]."""
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError(f"logits must be a 3-D tensor, got {type(logits)!r}")
    batch, sequence = input_ids.shape
    if requested_layout == "bsv":
        if logits.shape[:2] != (batch, sequence):
            raise ValueError(f"layout=bsv conflicts with raw shape {tuple(logits.shape)}")
        return logits
    if requested_layout == "sbv":
        if logits.shape[:2] != (sequence, batch):
            raise ValueError(f"layout=sbv conflicts with raw shape {tuple(logits.shape)}")
        return logits.transpose(0, 1).contiguous()
    if requested_layout != "auto":
        raise ValueError(f"Unsupported logits layout: {requested_layout}")

    bsv = logits.shape[:2] == (batch, sequence)
    sbv = logits.shape[:2] == (sequence, batch)
    if bsv and not sbv:
        return logits
    if sbv and not bsv:
        return logits.transpose(0, 1).contiguous()
    if bsv and sbv:
        raise ValueError("B == S makes layout ambiguous; pass --logits-layout=bsv or sbv.")
    raise ValueError(
        f"Raw logits shape {tuple(logits.shape)} does not match input {tuple(input_ids.shape)}."
    )


def convert_logits_dtype(logits: torch.Tensor, dtype_name: str) -> torch.Tensor:
    conversions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_name == "origin":
        return logits
    if dtype_name not in conversions:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return logits.to(conversions[dtype_name])


def select_output_tensor(output: Any, selector: str = "auto") -> torch.Tensor:
    """Select logits from a tensor, mapping, object attribute, or sequence output."""
    if selector == "auto":
        if torch.is_tensor(output):
            return output
        if hasattr(output, "logits") and torch.is_tensor(output.logits):
            return output.logits
        if isinstance(output, Mapping) and torch.is_tensor(output.get("logits")):
            return output["logits"]
        if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
            candidates = [item for item in output if torch.is_tensor(item) and item.ndim == 3]
            if len(candidates) == 1:
                return candidates[0]
            raise TypeError(f"Cannot uniquely select logits from {len(candidates)} 3-D tensors.")
        raise TypeError(f"Unsupported forward output type: {type(output)!r}")

    kind, separator, value = selector.partition(":")
    if not separator or not value:
        raise ValueError("--logits-output-selector must be auto, attr:NAME, key:NAME, or index:N.")
    if kind == "attr":
        selected = getattr(output, value)
    elif kind == "key":
        if not isinstance(output, Mapping):
            raise TypeError(f"key selector requires a mapping, got {type(output)!r}")
        selected = output[value]
    elif kind == "index":
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
            raise TypeError(f"index selector requires a sequence, got {type(output)!r}")
        selected = output[int(value)]
    else:
        raise ValueError(f"Unsupported output selector kind: {kind!r}")
    if not torch.is_tensor(selected):
        raise TypeError(f"Selected output is not a tensor: {type(selected)!r}")
    return selected


def parse_json_object(value: str | None) -> dict[str, Any]:
    """Parse an inline JSON object or @path-to-json; return an empty object for None."""
    if value is None:
        return {}
    if value.startswith("@"):
        source = Path(value[1:])
        parsed = json.loads(source.read_text(encoding="utf-8"))
    else:
        parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("Alignment metadata JSON must be an object.")
    return parsed


def to_json_compatible(value: Any) -> Any:
    """Recursively normalize framework metadata into deterministic JSON values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        if tensor.numel() <= 256:
            return tensor.tolist()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [to_json_compatible(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return to_json_compatible(item_method())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)
