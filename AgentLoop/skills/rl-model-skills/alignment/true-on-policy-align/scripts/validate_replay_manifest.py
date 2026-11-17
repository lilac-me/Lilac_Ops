#!/usr/bin/env python3
"""Validate a guarded prefill Tensor replay chain."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


DIFF_ID = re.compile(r"^D[0-9]{3,}$")
SHA256 = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


class ManifestError(ValueError):
    pass


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON {path}: {exc}") from exc


def req_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"{key} must be an object")
    return value


def req_string(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{key} must be a non-empty string")
    return value.strip()


def req_strings(parent: dict[str, Any], key: str) -> list[str]:
    value = parent.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ManifestError(f"{key} must be a non-empty list of strings")
    return [item.strip() for item in value]


def validate_fixed_relative_path(raw: str) -> None:
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or re.match(r"^[A-Za-z]:/", normalized):
        raise ManifestError("tensor_file must be relative to the replay manifest")
    if ".." in path.parts or any(token in normalized for token in ("*", "?", "[", "]")):
        raise ManifestError("tensor_file must be a fixed in-tree relative path")


def validate_context(context: dict[str, Any]) -> None:
    for key in ("model_digest", "config_digest", "sample_digest"):
        if not SHA256.match(req_string(context, key)):
            raise ManifestError(f"context.{key} must be a full sha256 value")
    prompt_length = context.get("prompt_token_length")
    if not isinstance(prompt_length, int) or isinstance(prompt_length, bool) or prompt_length < 1:
        raise ManifestError("context.prompt_token_length must be a positive integer")
    req_string(context, "token_slice")
    layer = context.get("layer")
    call_index = context.get("call_index")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
        raise ManifestError("context.layer must be a non-negative integer")
    if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 0:
        raise ManifestError("context.call_index must be a non-negative integer")
    req_string(context, "symbol")
    shape = context.get("shape")
    if not isinstance(shape, list) or not shape or not all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape
    ):
        raise ManifestError("context.shape must be a non-empty list of non-negative integers")
    req_string(context, "dtype")


def validate(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root must be an object")
    if manifest.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    if manifest.get("phase") != "prefill":
        raise ManifestError("phase must be prefill")
    if manifest.get("canonical_side") != "train":
        raise ManifestError("canonical_side must be train")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ManifestError("entries must be a list")

    ids: list[str] = []
    enabled = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"entries[{index}] must be an object")
        diff_id = req_string(entry, "diff_id")
        if not DIFF_ID.match(diff_id):
            raise ManifestError(f"entries[{index}].diff_id is invalid")
        ids.append(diff_id)
        if not isinstance(entry.get("enabled"), bool):
            raise ManifestError(f"entries[{index}].enabled must be boolean")
        enabled += int(entry["enabled"])
        req_string(entry, "boundary")
        tensor_file = req_string(entry, "tensor_file")
        validate_fixed_relative_path(tensor_file)
        if not SHA256.match(req_string(entry, "sha256")):
            raise ManifestError(f"entries[{index}].sha256 must be a full sha256 value")
        if entry.get("weights_only") is not True:
            raise ManifestError(f"entries[{index}].weights_only must be true")
        validate_context(req_dict(entry, "context"))
        req_string(entry, "return_selector")
        req_strings(entry, "side_effects_covered")
        gate = req_dict(entry, "activation_gate")
        hits = gate.get("expected_hits")
        if not isinstance(hits, int) or isinstance(hits, bool) or hits < 1:
            raise ManifestError("activation_gate.expected_hits must be a positive integer")
        if gate.get("post_injection_torch_equal") is not True:
            raise ManifestError("activation_gate.post_injection_torch_equal must be true")
        max_abs = gate.get("post_injection_max_abs_diff")
        if isinstance(max_abs, bool) or not isinstance(max_abs, (int, float)) or not math.isfinite(max_abs) or max_abs != 0:
            raise ManifestError("activation_gate.post_injection_max_abs_diff must be finite zero")
        if gate.get("immediate_downstream_input_exact") is not True:
            raise ManifestError("activation_gate.immediate_downstream_input_exact must be true")

    if len(ids) != len(set(ids)):
        raise ManifestError("entries contain duplicate diff_id values")
    return {"valid": True, "entry_count": len(entries), "enabled_count": enabled, "diff_ids": ids}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(load_json(args.manifest.resolve()))
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
