#!/usr/bin/env python3
"""Validate an immutable prefill discovery Difference Packet."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


DIFF_ID = re.compile(r"^D[0-9]{3,}$")
SHA256 = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
ROOT_CAUSES = {
    "Op path",
    "Input precision",
    "Config",
    "Missing step",
    "Framework bug",
    "Test setup mismatch",
    "Weight sync",
    "State transition",
    "Control plane",
}
CONFIDENCE = {"CONFIRMED", "SUPPORTED", "CANDIDATE"}
OBSERVATIONS = {"native", "conditional"}
BOUNDARY_KINDS = {"tensor", "weight_reload", "state_transition", "control_plane"}
REPLAY_STATUS = {"proposed", "ready", "active", "unsafe", "disabled", "resolved"}
REPAIR_STATUS = {
    "not_started",
    "queued",
    "agent_working",
    "plan_ready",
    "integrated_exact",
    "unresolved_replayed",
    "rejected",
}


class PacketError(ValueError):
    pass


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PacketError(f"cannot read JSON {path}: {exc}") from exc


def require_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise PacketError(f"{key} must be an object")
    return value


def require_string(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PacketError(f"{key} must be a non-empty string")
    return value.strip()


def require_bool(parent: dict[str, Any], key: str, expected: bool | None = None) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise PacketError(f"{key} must be a boolean")
    if expected is not None and value is not expected:
        raise PacketError(f"{key} must be {str(expected).lower()}")
    return value


def require_strings(parent: dict[str, Any], key: str, *, allow_empty: bool = False) -> list[str]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise PacketError(f"{key} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise PacketError(f"{key} may not be empty")
    return [item.strip() for item in value]


def validate(packet: Any) -> dict[str, Any]:
    if not isinstance(packet, dict):
        raise PacketError("packet root must be an object")
    schema_version = packet.get("schema_version")
    if schema_version not in {1, 2}:
        raise PacketError("schema_version must be 1 or 2")

    boundary_kind = "tensor"
    if schema_version == 2:
        boundary_kind = require_string(packet, "boundary_kind")
        if boundary_kind not in BOUNDARY_KINDS:
            raise PacketError("boundary_kind is unsupported")

    diff_id = require_string(packet, "diff_id")
    if not DIFF_ID.match(diff_id):
        raise PacketError("diff_id must match D followed by at least three digits")
    if packet.get("phase") != "prefill":
        raise PacketError("phase must be prefill")
    order = packet.get("execution_order")
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        raise PacketError("execution_order must be a non-negative integer")
    if require_string(packet, "observation") not in OBSERVATIONS:
        raise PacketError("observation must be native or conditional")
    require_string(packet, "semantic_boundary")

    upstream = require_strings(packet, "upstream_replay_ids", allow_empty=True)
    if any(not DIFF_ID.match(item) for item in upstream):
        raise PacketError("upstream_replay_ids contains an invalid diff ID")
    if diff_id in upstream or len(upstream) != len(set(upstream)):
        raise PacketError("upstream_replay_ids must be unique and may not contain diff_id")
    if packet["observation"] == "native" and upstream:
        raise PacketError("native observation may not depend on upstream replay IDs")
    if packet["observation"] == "conditional" and not upstream:
        raise PacketError("conditional observation requires upstream replay IDs")

    proof = require_dict(packet, "proof")
    if boundary_kind == "tensor":
        for key in (
            "aligned_prefix",
            "all_causal_inputs_exact",
            "same_shape",
            "same_dtype",
            "no_nonfinite",
            "output_nonzero",
        ):
            require_bool(proof, key, True)
        max_abs = proof.get("max_abs_diff")
        if (
            isinstance(max_abs, bool)
            or not isinstance(max_abs, (int, float))
            or not math.isfinite(max_abs)
            or max_abs <= 0
        ):
            raise PacketError("proof.max_abs_diff must be a positive finite number")
    else:
        for key in (
            "aligned_prefix",
            "same_model_config_sample",
            "expected_transition_recorded",
            "source_state_recorded",
            "target_state_recorded",
            "post_transition_state_mismatch",
            "no_nonfinite",
        ):
            require_bool(proof, key, True)
        observation = require_string(proof, "target_transition_observation")
        if observation not in {"observed", "missing", "failed", "stale"}:
            raise PacketError(
                "proof.target_transition_observation must be observed, missing, failed, or stale"
            )
        expected_digest = require_string(proof, "expected_state_digest")
        target_before_digest = require_string(proof, "target_before_state_digest")
        target_after_digest = require_string(proof, "target_after_state_digest")
        if any(
            not SHA256.match(value)
            for value in (expected_digest, target_before_digest, target_after_digest)
        ):
            raise PacketError("state proof digests must be full sha256 values")
        if expected_digest == target_after_digest:
            raise PacketError(
                "state proof requires expected_state_digest != target_after_state_digest"
            )
        require_string(proof, "impact_metric")
        impact = proof.get("impact_max_abs_diff")
        if (
            isinstance(impact, bool)
            or not isinstance(impact, (int, float))
            or not math.isfinite(impact)
            or impact <= 0
        ):
            raise PacketError(
                "proof.impact_max_abs_diff must be a positive finite number"
            )

    root_cause = require_dict(packet, "root_cause")
    if require_string(root_cause, "class") not in ROOT_CAUSES:
        raise PacketError("root_cause.class is unsupported")
    if require_string(root_cause, "confidence") not in CONFIDENCE:
        raise PacketError("root_cause.confidence is unsupported")

    source = require_dict(packet, "source")
    require_string(source, "train")
    require_string(source, "infer")

    evidence = require_dict(packet, "evidence")
    if boundary_kind == "tensor":
        for key in ("manifest", "compare_report", "train_stack", "infer_stack"):
            require_string(evidence, key)
        hashes = require_strings(evidence, "tensor_hashes")
        if any(not SHA256.match(item) for item in hashes):
            raise PacketError("evidence.tensor_hashes must contain full sha256 values")
    else:
        for key in ("state_manifest", "before_after_report", "event_log"):
            require_string(evidence, key)
        hashes = require_strings(evidence, "artifact_hashes")
        if any(not SHA256.match(item) for item in hashes):
            raise PacketError("evidence.artifact_hashes must contain full sha256 values")

    replay = require_dict(packet, "replay")
    require_string(replay, "recommended_boundary")
    require_strings(replay, "side_effects_to_cover")
    if require_string(replay, "status") not in REPLAY_STATUS:
        raise PacketError("replay.status is unsupported")

    repair = require_dict(packet, "repair")
    group_hint = repair.get("group_hint")
    if group_hint is not None and (not isinstance(group_hint, str) or not group_hint.strip()):
        raise PacketError("repair.group_hint must be null or a non-empty string")
    if require_string(repair, "status") not in REPAIR_STATUS:
        raise PacketError("repair.status is unsupported")

    return {
        "valid": True,
        "schema_version": schema_version,
        "diff_id": diff_id,
        "boundary_kind": boundary_kind,
        "observation": packet["observation"],
        "upstream_replay_count": len(upstream),
        "root_cause_confidence": root_cause["confidence"],
        "replay_status": replay["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(load_json(args.packet.resolve()))
    except PacketError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
