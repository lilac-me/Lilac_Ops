#!/usr/bin/env python3
"""Rank structural train/infer suspects without weakening exact Tensor truth gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from validate_semantic_mapping import (
    MAPPING_ID,
    SHA256,
    MappingError,
    load_json,
    sha256_file,
    validate as validate_mapping,
)


ROLES = {"input", "output", "parameter", "aux_state"}
EVIDENCE_KINDS = {"exact_tensor", "statistics", "metadata"}
SUMMARY_RELATIONS = {"same", "different", "unknown"}
SIDES = {"train", "infer"}
TRIAGE_ONLY_STATES = {
    "statistics_same",
    "statistics_difference",
    "metadata_only",
    "unknown",
}
DIFFERENCE_STATES = {
    "exact_difference",
    "statistics_difference",
    "shape_mismatch",
    "dtype_mismatch",
    "nonfinite",
}


class TriageError(ValueError):
    pass


def parse_expected_hashes(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        name, separator, digest = item.partition("=")
        if (
            not separator
            or not name.strip()
            or not SHA256.match(digest)
            or name.strip() in result
        ):
            raise TriageError(
                "--expect-input-hash must be unique NAME=sha256:<64 hex>"
            )
        result[name.strip()] = digest
    return result


def verify_expected_hashes(
    observed: dict[str, str], expected: dict[str, str]
) -> None:
    for name, digest in expected.items():
        current = observed.get(name)
        if current is None:
            raise TriageError(
                f"expected input hash {name!r} is missing from observations"
            )
        if current.lower() != digest.lower():
            raise TriageError(
                f"observations input hash {name!r} does not match the current case"
            )


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_string(parent: dict[str, Any], key: str, label: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TriageError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def validate_shape(value: Any, label: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in value
    ):
        raise TriageError(f"{label}.shape must be a list of non-negative integers")
    return value


def validate_statistics(value: Any, label: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TriageError(f"{label}.statistics must be an object")
    result: dict[str, float] = {}
    for key, number in value.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise TriageError(
                f"{label}.statistics must contain finite numeric values"
            )
        result[key.strip().lower()] = float(number)
    return result


def validate_side(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TriageError(f"{label} must be an object")
    present = value.get("present")
    if not isinstance(present, bool):
        raise TriageError(f"{label}.present must be a boolean")
    if not present:
        return {
            "present": False,
            "missing_reason": value.get("missing_reason"),
            "dtype": None,
            "shape": None,
            "statistics": {},
        }
    dtype = value.get("dtype")
    if dtype is not None and (not isinstance(dtype, str) or not dtype.strip()):
        raise TriageError(f"{label}.dtype must be a non-empty string or null")
    return {
        "present": True,
        "dtype": dtype.strip() if isinstance(dtype, str) else None,
        "shape": validate_shape(value.get("shape"), label),
        "statistics": validate_statistics(value.get("statistics"), label),
    }


def validate_exact_comparison(value: dict[str, Any], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {"evidence_kind": "exact_tensor"}
    for key in ("same_shape", "same_dtype", "no_nonfinite", "torch_equal"):
        item = value.get(key)
        if not isinstance(item, bool):
            raise TriageError(f"{label}.{key} must be a boolean")
        result[key] = item
    max_abs = value.get("max_abs_diff")
    if (
        isinstance(max_abs, bool)
        or not isinstance(max_abs, (int, float))
        or not math.isfinite(max_abs)
        or max_abs < 0
    ):
        raise TriageError(f"{label}.max_abs_diff must be a finite non-negative number")
    result["max_abs_diff"] = float(max_abs)
    basis = require_string(value, "basis", label)
    result["basis"] = basis
    if (
        result["same_shape"]
        and result["same_dtype"]
        and result["no_nonfinite"]
        and result["torch_equal"]
        and result["max_abs_diff"] != 0
    ):
        raise TriageError(f"{label} has torch_equal=true but max_abs_diff is nonzero")
    if (
        result["same_shape"]
        and result["same_dtype"]
        and result["no_nonfinite"]
        and not result["torch_equal"]
        and result["max_abs_diff"] == 0
    ):
        raise TriageError(
            f"{label} has torch_equal=false but max_abs_diff=0 under equal shape/dtype"
        )
    return result


def validate_comparison(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TriageError(f"{label} must be an object")
    kind = require_string(value, "evidence_kind", label)
    if kind not in EVIDENCE_KINDS:
        raise TriageError(f"{label}.evidence_kind must be one of {sorted(EVIDENCE_KINDS)}")
    if kind == "exact_tensor":
        return validate_exact_comparison(value, label)
    basis = require_string(value, "basis", label)
    result = {"evidence_kind": kind, "basis": basis}
    if kind == "statistics":
        relation = require_string(value, "summary_relation", label)
        if relation not in SUMMARY_RELATIONS:
            raise TriageError(
                f"{label}.summary_relation must be one of {sorted(SUMMARY_RELATIONS)}"
            )
        result["summary_relation"] = relation
    return result


def observation_state(
    train: dict[str, Any], infer: dict[str, Any], comparison: dict[str, Any]
) -> str:
    if not train["present"] and not infer["present"]:
        return "missing_both"
    if not train["present"]:
        return "missing_train"
    if not infer["present"]:
        return "missing_infer"

    train_shape, infer_shape = train["shape"], infer["shape"]
    if train_shape is not None and infer_shape is not None and train_shape != infer_shape:
        return "shape_mismatch"
    train_dtype, infer_dtype = train["dtype"], infer["dtype"]
    if train_dtype is not None and infer_dtype is not None and train_dtype != infer_dtype:
        return "dtype_mismatch"

    kind = comparison["evidence_kind"]
    if kind == "exact_tensor":
        if not comparison["same_shape"]:
            return "shape_mismatch"
        if not comparison["same_dtype"]:
            return "dtype_mismatch"
        if not comparison["no_nonfinite"]:
            return "nonfinite"
        if comparison["torch_equal"] and comparison["max_abs_diff"] == 0:
            return "exact_equal"
        return "exact_difference"
    if kind == "statistics":
        relation = comparison["summary_relation"]
        if relation == "same":
            return "statistics_same"
        if relation == "different":
            return "statistics_difference"
        return "unknown"
    return "metadata_only"


def validate_observations(
    value: Any, mapping_ids: set[str], expected_mapping_hash: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TriageError("observations root must be an object")
    if value.get("schema_version") != 1:
        raise TriageError("observations.schema_version must be 1")
    if value.get("phase") != "prefill":
        raise TriageError("observations.phase must be prefill")
    recorded_hash = require_string(value, "mapping_file_sha256", "observations")
    if not SHA256.match(recorded_hash):
        raise TriageError("observations.mapping_file_sha256 must be sha256:<64 hex>")
    if recorded_hash.lower() != expected_mapping_hash.lower():
        raise TriageError(
            "observations.mapping_file_sha256 does not match the current mapping file"
        )

    input_hashes = value.get("input_hashes")
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise TriageError("observations.input_hashes must be a non-empty object")
    for name, digest in input_hashes.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(digest, str)
            or not SHA256.match(digest)
        ):
            raise TriageError(
                "observations.input_hashes must map non-empty names to sha256:<64 hex>"
            )

    records = value.get("records")
    if not isinstance(records, list):
        raise TriageError("observations.records must be a list")
    normalized_records = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        label = f"observations.records[{index}]"
        if not isinstance(record, dict):
            raise TriageError(f"{label} must be an object")
        mapping_id = require_string(record, "mapping_id", label)
        if mapping_id not in mapping_ids:
            raise TriageError(f"{label}.mapping_id is not present in semantic mapping")
        role = require_string(record, "role", label)
        if role not in ROLES:
            raise TriageError(f"{label}.role must be one of {sorted(ROLES)}")
        identity = (mapping_id, role)
        if identity in seen:
            raise TriageError(f"duplicate observation for {mapping_id}/{role}")
        seen.add(identity)
        train = validate_side(record.get("train"), f"{label}.train")
        infer = validate_side(record.get("infer"), f"{label}.infer")
        comparison = validate_comparison(record.get("comparison"), f"{label}.comparison")
        normalized_records.append(
            {
                "mapping_id": mapping_id,
                "role": role,
                "train": train,
                "infer": infer,
                "comparison": comparison,
                "state": observation_state(train, infer, comparison),
            }
        )

    unmatched = value.get("unmatched_nodes", [])
    if not isinstance(unmatched, list):
        raise TriageError("observations.unmatched_nodes must be a list")
    normalized_unmatched = []
    for index, item in enumerate(unmatched):
        label = f"observations.unmatched_nodes[{index}]"
        if not isinstance(item, dict):
            raise TriageError(f"{label} must be an object")
        side = require_string(item, "side", label)
        if side not in SIDES:
            raise TriageError(f"{label}.side must be train or infer")
        order = item.get("execution_order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise TriageError(f"{label}.execution_order must be a non-negative integer")
        normalized_unmatched.append(
            {
                "side": side,
                "key": require_string(item, "key", label),
                "execution_order": order,
                "semantic_hint": item.get("semantic_hint", "unknown"),
                "source_evidence": require_string(item, "source_evidence", label),
            }
        )

    hints = value.get("structural_hints", [])
    if not isinstance(hints, list):
        raise TriageError("observations.structural_hints must be a list")
    normalized_hints = []
    hint_ids: set[str] = set()
    for index, hint in enumerate(hints):
        label = f"observations.structural_hints[{index}]"
        if not isinstance(hint, dict):
            raise TriageError(f"{label} must be an object")
        hint_id = require_string(hint, "hint_id", label)
        if not MAPPING_ID.match(hint_id) or hint_id in hint_ids:
            raise TriageError(f"{label}.hint_id is invalid or duplicated")
        hint_ids.add(hint_id)
        mapping_id = require_string(hint, "mapping_id", label)
        if mapping_id not in mapping_ids:
            raise TriageError(f"{label}.mapping_id is not present in semantic mapping")
        normalized_hints.append(
            {
                "hint_id": hint_id,
                "mapping_id": mapping_id,
                "kind": require_string(hint, "kind", label),
                "basis": require_string(hint, "basis", label),
            }
        )

    return {
        "input_hashes": input_hashes,
        "records": normalized_records,
        "unmatched_nodes": normalized_unmatched,
        "structural_hints": normalized_hints,
    }


def state_is_exact_input(state: str) -> bool:
    return state == "exact_equal"


def classify_candidate(
    roles: dict[str, dict[str, Any]],
    unmatched_train: list[dict[str, Any]],
    unmatched_infer: list[dict[str, Any]],
) -> tuple[str, list[str], int]:
    input_states = [
        roles[role]["state"] for role in ("input", "aux_state") if role in roles
    ]
    parameter_state = roles.get("parameter", {}).get("state", "missing")

    if unmatched_train or unmatched_infer:
        return (
            "missing_or_extra_op_between_exact_boundary",
            ["the unmatched node is only a lead until source and exact Tensor evidence confirm causality"],
            1,
        )
    if input_states and all(state_is_exact_input(state) for state in input_states):
        if parameter_state == "exact_equal":
            return (
                "in_module_impl_difference",
                ["operator path", "precision boundary", "effective configuration"],
                1,
            )
        if parameter_state in DIFFERENCE_STATES:
            return (
                "parameter_or_checkpoint_issue",
                ["weight conversion", "load path", "parameter layout or precision"],
                1,
            )
        return (
            "needs_parameter_evidence",
            ["module implementation difference", "parameter or checkpoint issue"],
            2,
        )
    if any(state in DIFFERENCE_STATES for state in input_states):
        return (
            "upstream_propagation",
            ["the current module may amplify an earlier difference"],
            2,
        )
    return (
        "needs_exact_input_evidence",
        ["in-module difference", "upstream propagation", "missing causal input"],
        3,
    )


def build_triage(
    mapping: dict[str, Any],
    observations: dict[str, Any],
    mapping_path: Path,
    observations_path: Path,
) -> dict[str, Any]:
    entries = sorted(
        mapping["mappings"], key=lambda item: (item["execution_order"], item["mapping_id"])
    )
    records_by_mapping: dict[str, dict[str, dict[str, Any]]] = {
        entry["mapping_id"]: {} for entry in entries
    }
    for record in observations["records"]:
        records_by_mapping[record["mapping_id"]][record["role"]] = record
    hints_by_mapping: dict[str, list[dict[str, Any]]] = {
        entry["mapping_id"]: [] for entry in entries
    }
    for hint in observations["structural_hints"]:
        hints_by_mapping[hint["mapping_id"]].append(hint)

    previous_exact: dict[str, Any] | None = None
    raw_candidates: list[dict[str, Any]] = []
    excluded_candidates: list[dict[str, Any]] = []
    exact_verification_required: list[dict[str, Any]] = []
    exact_boundaries: list[dict[str, Any]] = []
    missing_output_records: list[str] = []

    for entry in entries:
        mapping_id = entry["mapping_id"]
        order = entry["execution_order"]
        roles = records_by_mapping[mapping_id]
        output = roles.get("output")
        output_state = output["state"] if output else "missing_record"

        if output_state == "exact_equal":
            previous_exact = {
                "mapping_id": mapping_id,
                "semantic_node": entry["semantic_node"],
                "execution_order": order,
                "evidence_kind": "exact_tensor",
            }
            exact_boundaries.append(previous_exact)
            internal_differences = [
                {
                    "role": role,
                    "state": record["state"],
                }
                for role, record in roles.items()
                if role != "output" and record["state"] in DIFFERENCE_STATES
            ]
            if internal_differences:
                excluded_candidates.append(
                    {
                        "mapping_id": mapping_id,
                        "semantic_node": entry["semantic_node"],
                        "reason": (
                            "module output is exactly equal; internal observations are "
                            "non-blocking for declared output coverage"
                        ),
                        "internal_observations": internal_differences,
                    }
                )
            continue

        if output is None:
            missing_output_records.append(mapping_id)
        exact_verification_required.append(
            {
                "mapping_id": mapping_id,
                "semantic_node": entry["semantic_node"],
                "execution_order": order,
                "observed_state": output_state,
                "required_gate": (
                    "same_shape=true, same_dtype=true, no_nonfinite=true, "
                    "torch.equal=true, max_abs_diff=0"
                ),
                "reason": (
                    "formal first-divergence proof still required"
                    if output_state == "exact_difference"
                    else "no exact output equality proof is available"
                ),
            }
        )

        if output_state not in DIFFERENCE_STATES:
            continue

        lower_bound = previous_exact["execution_order"] if previous_exact else -1
        unproved_between = [
            {
                "mapping_id": earlier["mapping_id"],
                "semantic_node": earlier["semantic_node"],
                "execution_order": earlier["execution_order"],
                "output_state": records_by_mapping[earlier["mapping_id"]]
                .get("output", {})
                .get("state", "missing_record"),
            }
            for earlier in entries
            if lower_bound < earlier["execution_order"] < order
            and records_by_mapping[earlier["mapping_id"]]
            .get("output", {})
            .get("state")
            != "exact_equal"
        ]
        unmatched_train = [
            node
            for node in observations["unmatched_nodes"]
            if node["side"] == "train"
            and lower_bound < node["execution_order"] < order
        ]
        unmatched_infer = [
            node
            for node in observations["unmatched_nodes"]
            if node["side"] == "infer"
            and lower_bound < node["execution_order"] < order
        ]
        classification, alternatives, priority = classify_candidate(
            roles,
            unmatched_train if previous_exact else [],
            unmatched_infer if previous_exact else [],
        )
        if output_state == "exact_difference":
            evidence_strength = "exact_output_observation"
        elif output_state in {"shape_mismatch", "dtype_mismatch", "nonfinite"}:
            evidence_strength = "comparability_blocker"
            priority = min(priority, 2)
        else:
            evidence_strength = "summary_only"
            priority = max(priority, 3)

        raw_candidates.append(
            {
                "mapping_id": mapping_id,
                "semantic_node": entry["semantic_node"],
                "execution_order": order,
                "mapping_relation": entry["relation"],
                "priority_rank": priority,
                "observed_output_state": output_state,
                "evidence_strength": evidence_strength,
                "classification": classification,
                "alternative_hypotheses": alternatives,
                "previous_exact_boundary": previous_exact,
                "unproved_mappings_between": unproved_between,
                "previous_exact_boundary_is_not_exact_prefix": bool(unproved_between),
                "unmatched_train_between_boundary": unmatched_train,
                "unmatched_infer_between_boundary": unmatched_infer,
                "structural_hints": hints_by_mapping[mapping_id],
                "structural_false_positive_candidate": bool(
                    hints_by_mapping[mapping_id] or entry.get("structural_relation")
                ),
                "reason_kept": (
                    "output-level evidence is more important than internal-only evidence; "
                    "formal Module 4 proof remains mandatory"
                ),
                "formal_verification_required": True,
            }
        )

    raw_candidates.sort(
        key=lambda item: (
            item["priority_rank"],
            0 if item["evidence_strength"] == "exact_output_observation" else 1,
            item["execution_order"],
            item["mapping_id"],
        )
    )
    candidates = [
        {"candidate_id": f"C{index:03d}", **candidate}
        for index, candidate in enumerate(raw_candidates, start=1)
    ]
    status = "complete" if not missing_output_records else "partial"
    mapping_hash = sha256_file(mapping_path)
    observation_hash = sha256_file(observations_path)
    return {
        "schema_version": 1,
        "phase": "prefill",
        "truth_role": "triage_only",
        "status": status,
        "status_reason": (
            "every semantic mapping has an output observation"
            if status == "complete"
            else "some semantic mappings have no output observation"
        ),
        "may_create_difference_packet": False,
        "candidate_ranking_must_not_define_first_divergence": True,
        "exact_comparison_must_follow_semantic_execution_order": True,
        "source": {
            "semantic_mapping": str(mapping_path),
            "semantic_mapping_sha256": mapping_hash,
            "observations": str(observations_path),
            "observations_sha256": observation_hash,
            "input_hashes": observations["input_hashes"],
        },
        "coverage_order": [
            {
                "mapping_id": entry["mapping_id"],
                "semantic_node": entry["semantic_node"],
                "execution_order": entry["execution_order"],
            }
            for entry in entries
        ],
        "exact_boundaries_observed": exact_boundaries,
        "candidates": candidates,
        "structural_false_positive_candidates": [
            {
                "mapping_id": entry["mapping_id"],
                "semantic_node": entry["semantic_node"],
                "mapping_relation": entry["relation"],
                "structural_relation": entry.get("structural_relation"),
                "hints": hints_by_mapping[entry["mapping_id"]],
                "disposition": "inspect_mapping_before_exact_root_cause_claim",
            }
            for entry in entries
            if entry.get("structural_relation") or hints_by_mapping[entry["mapping_id"]]
        ],
        "excluded_candidates": excluded_candidates,
        "exact_verification_required": exact_verification_required,
        "missing_output_records": missing_output_records,
        "next_checks": [
            "walk coverage_order, not candidate rank, to prove the semantic first divergence",
            "inspect structural hints before forcing physical one-to-one mappings",
            "capture missing causal inputs, parameters, or auxiliary state narrowly",
            "create a Difference Packet only after the existing Module 4 exact gates pass",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expect-input-hash",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="bind observations to a current Module 2 artifact digest; repeat as needed",
    )
    args = parser.parse_args()
    mapping_path = args.mapping.resolve()
    observations_path = args.observations.resolve()
    try:
        mapping = load_json(mapping_path)
        validate_mapping(mapping)
        mapping_hash = sha256_file(mapping_path)
        observations_raw = load_json(observations_path)
        observations = validate_observations(
            observations_raw,
            {entry["mapping_id"] for entry in mapping["mappings"]},
            mapping_hash,
        )
        verify_expected_hashes(
            observations["input_hashes"],
            parse_expected_hashes(args.expect_input_hash),
        )
        result = build_triage(
            mapping, observations, mapping_path, observations_path
        )
        write_json(result, args.output.resolve())
    except (MappingError, TriageError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "valid": True,
                "status": result["status"],
                "candidate_count": len(result["candidates"]),
                "exact_verification_required": len(
                    result["exact_verification_required"]
                ),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
