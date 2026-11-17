#!/usr/bin/env python3
"""Guarded runtime loader for session-local training-side Tensor replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ReplayError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class ReplayStore:
    """Load exact replay values after validating immutable runtime context."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayError(f"cannot read replay manifest: {exc}") from exc
        if self.manifest.get("schema_version") != 1:
            raise ReplayError("unsupported replay schema")
        if self.manifest.get("phase") != "prefill" or self.manifest.get("canonical_side") != "train":
            raise ReplayError("replay manifest must use training-side prefill values")
        entries = self.manifest.get("entries")
        if not isinstance(entries, list):
            raise ReplayError("replay entries must be a list")
        self.entries: dict[str, dict[str, Any]] = {}
        self.hits: dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("diff_id"), str):
                raise ReplayError("invalid replay entry")
            diff_id = entry["diff_id"]
            if diff_id in self.entries:
                raise ReplayError(f"duplicate replay entry: {diff_id}")
            self.entries[diff_id] = entry
            self.hits[diff_id] = 0

    def _entry(self, diff_id: str) -> dict[str, Any]:
        entry = self.entries.get(diff_id)
        if entry is None:
            raise ReplayError(f"unknown replay diff_id: {diff_id}")
        if entry.get("enabled") is not True:
            raise ReplayError(f"replay entry is disabled: {diff_id}")
        if entry.get("weights_only") is not True:
            raise ReplayError(f"unsafe replay entry: {diff_id}")
        return entry

    @staticmethod
    def _validate_context(expected: dict[str, Any], actual: dict[str, Any]) -> None:
        runtime_keys = (
            "model_digest",
            "config_digest",
            "sample_digest",
            "prompt_token_length",
            "token_slice",
            "layer",
            "symbol",
            "call_index",
        )
        for key in runtime_keys:
            if actual.get(key) != expected.get(key):
                raise ReplayError(
                    f"replay context mismatch for {key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
                )

    def replacement(self, diff_id: str, current: Any, runtime_context: dict[str, Any]):
        entry = self._entry(diff_id)
        expected_hits = entry.get("activation_gate", {}).get("expected_hits")
        if not isinstance(expected_hits, int) or expected_hits < 1:
            raise ReplayError(f"invalid expected_hits for {diff_id}")
        if self.hits[diff_id] >= expected_hits:
            raise ReplayError(f"replay entry activated too many times: {diff_id}")

        expected_context = entry.get("context")
        if not isinstance(expected_context, dict):
            raise ReplayError(f"missing context for {diff_id}")
        self._validate_context(expected_context, runtime_context)

        tensor_path = (self.root / str(entry.get("tensor_file", ""))).resolve()
        if not _within(tensor_path, self.root):
            raise ReplayError(f"tensor path escapes replay root: {tensor_path}")
        if not tensor_path.is_file():
            raise ReplayError(f"replay tensor is missing: {tensor_path}")
        if _sha256(tensor_path) != entry.get("sha256"):
            raise ReplayError(f"replay tensor hash mismatch: {diff_id}")

        try:
            import torch
        except ImportError as exc:
            raise ReplayError("PyTorch is required for Tensor replay") from exc
        try:
            replacement = torch.load(tensor_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise ReplayError("installed PyTorch lacks safe weights_only Tensor loading") from exc
        if not isinstance(replacement, torch.Tensor):
            raise ReplayError(f"replay payload is not a Tensor: {diff_id}")
        if not isinstance(current, torch.Tensor):
            raise ReplayError("current boundary value must be a Tensor; select the declared return field first")

        expected_shape = tuple(expected_context.get("shape", []))
        expected_dtype = str(expected_context.get("dtype"))
        if tuple(replacement.shape) != expected_shape or str(replacement.dtype) != expected_dtype:
            raise ReplayError(f"saved replay Tensor metadata mismatch: {diff_id}")
        if tuple(current.shape) != expected_shape or str(current.dtype) != expected_dtype:
            raise ReplayError(f"live boundary shape/dtype mismatch: {diff_id}")

        self.hits[diff_id] += 1
        return replacement.to(device=current.device)

    def assert_complete(self) -> None:
        failures = []
        for diff_id, entry in self.entries.items():
            if entry.get("enabled") is not True:
                continue
            expected = entry.get("activation_gate", {}).get("expected_hits")
            if self.hits.get(diff_id) != expected:
                failures.append(f"{diff_id}: expected {expected}, observed {self.hits.get(diff_id)}")
        if failures:
            raise ReplayError("replay activation count mismatch: " + "; ".join(failures))
