#!/usr/bin/env python3
"""Validate one model-specific, reversible repair proposal for remote-ops."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ALLOWED_STATUSES = {
    "PATCH_PROPOSAL_READY",
    "CONFIG_ONLY_CHANGE",
    "INSUFFICIENT_EVIDENCE",
    "TRAIN_SIDE_OR_RESHARD_BUG",
    "DEPENDENCY_REQUIRED",
    "UNSUPPORTED_VERSION",
    "NO_SAFE_REVERSIBLE_FIX",
}
READY_STATUSES = {"PATCH_PROPOSAL_READY", "CONFIG_ONLY_CHANGE"}
ALLOWED_METHODS = {
    "config",
    "runtime_monkey_patch",
    "source_patch",
    "weight_or_reshard_patch",
}
ROOT_CAUSES = {
    "Op path",
    "Input precision",
    "Config",
    "Missing step",
    "Framework bug",
    "Weight sync",
    "State transition",
    "Control plane",
}
CONFIDENCE = {"CONFIRMED", "SUPPORTED"}
FAILURE_ROUTES = {
    "preflight_failure",
    "activation_failure",
    "no_effect",
    "target_worse",
    "downstream_difference",
    "e2e_regression",
}
PROBE_MODES = {
    "single_process_ab",
    "dual_process_replay",
    "in_situ_targeted",
    "not_applicable",
}
REPAIR_UNIT_ID = re.compile(r"^R[0-9]{3,}$")
DIFF_ID = re.compile(r"^D[0-9]{3,}$")
FORBIDDEN_MUTATIONS = re.compile(
    r"(?:git\s+reset\s+--hard|git\s+checkout\s+--|rm\s+-rf|pkill(?:\s|$))",
    re.IGNORECASE,
)


class PlanError(ValueError):
    pass


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"Cannot read fix plan {path}: {exc}") from exc


def require_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise PlanError(f"{key} must be an object")
    return value


def require_string(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{key} must be a non-empty string")
    if value.strip().lower() in {"unknown", "missing", "tbd", "todo"}:
        raise PlanError(f"{key} may not use an unknown placeholder")
    return value.strip()


def require_bool(parent: dict[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise PlanError(f"{key} must be a boolean")
    return value


def require_string_list(
    parent: dict[str, Any], key: str, *, allow_empty: bool = False
) -> list[str]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise PlanError(f"{key} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise PlanError(f"{key} may not be empty")
    return [item.strip() for item in value]


def validate_repo_relative_path(raw: str, label: str) -> None:
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or re.match(r"^[A-Za-z]:/", normalized):
        raise PlanError(f"{label} must be repository-relative: {raw}")
    if ".." in path.parts or ".git" in path.parts:
        raise PlanError(f"{label} escapes or mutates repository metadata: {raw}")
    if any(token in normalized for token in ("*", "?", "[", "]", "{")):
        raise PlanError(f"{label} may not contain globs: {raw}")


def reject_unsafe_steps(values: list[str], label: str) -> None:
    for value in values:
        if FORBIDDEN_MUTATIONS.search(value):
            raise PlanError(f"{label} contains a forbidden broad mutation: {value}")


def validate_analysis(plan: dict[str, Any]) -> dict[str, Any]:
    analysis = require_dict(plan, "analysis")
    if require_string(analysis, "status") != "FIRST_DIVERGENCE_FOUND":
        raise PlanError("ready fix plans require analysis.status=FIRST_DIVERGENCE_FOUND")
    root_cause = require_string(analysis, "root_cause_class")
    if root_cause not in ROOT_CAUSES:
        raise PlanError(f"unsupported root cause: {root_cause}")
    confidence = require_string(analysis, "root_cause_confidence")
    if confidence not in CONFIDENCE:
        raise PlanError("ready fix plans require CONFIRMED or SUPPORTED root cause")
    require_string(analysis, "first_divergence")
    require_string(analysis, "first_divergence_proof")
    evidence_files = require_string_list(analysis, "evidence_files")
    for index, path in enumerate(evidence_files):
        validate_repo_relative_path(path, f"analysis.evidence_files[{index}]")
    return analysis


def validate_generalization(plan: dict[str, Any]) -> None:
    generalization = require_dict(plan, "generalization")
    current_model = require_dict(generalization, "current_model")
    require_string(current_model, "family")
    require_string(current_model, "architecture")
    require_string(current_model, "revision")
    require_string_list(generalization, "architecture_evidence")
    require_string_list(generalization, "active_path_evidence")
    require_string(generalization, "historical_analogy")
    require_string_list(generalization, "applicability_limits")
    if not require_bool(generalization, "rederived_for_current_model"):
        raise PlanError("the repair must be re-derived for the current model")
    if require_bool(generalization, "copied_historical_patch"):
        raise PlanError("copying a historical/model-specific patch is forbidden")


def validate_exact_gate(change: dict[str, Any]) -> None:
    verify = require_dict(change, "targeted_verify_pair")
    manifest = require_string(verify, "manifest")
    validate_repo_relative_path(manifest, "targeted_verify_pair.manifest")
    require_string(verify, "pair_name")
    expected = require_dict(verify, "expected")
    for key in ("same_shape", "same_dtype", "torch_equal", "no_nonfinite"):
        if expected.get(key) is not True:
            raise PlanError(f"targeted_verify_pair.expected.{key} must be true")
    if expected.get("max_abs_diff") != 0:
        raise PlanError("targeted_verify_pair.expected.max_abs_diff must be 0")


def validate_micro_probe(change: dict[str, Any]) -> None:
    probe = require_dict(change, "micro_probe")
    mode = require_string(probe, "mode")
    if mode not in PROBE_MODES:
        raise PlanError(f"unsupported micro_probe.mode: {mode}")
    require_string(probe, "real_path_integration_gate")
    require_string_list(probe, "limitations")
    if mode == "not_applicable":
        require_string(probe, "reason")
        require_string(probe, "fallback_gate")
        return
    adapter = require_string(probe, "adapter")
    case_dir = require_string(probe, "case_dir")
    for label, value in (("adapter", adapter), ("case_dir", case_dir)):
        validate_repo_relative_path(value, f"micro_probe.{label}")
        if "references/fix-cases" in value.replace("\\", "/"):
            raise PlanError(f"micro_probe.{label} must stay session-local")
    require_string_list(probe, "captured_inputs")
    require_string_list(probe, "state_dependencies")
    require_string_list(probe, "semantic_transforms")
    for key in ("train_entry", "infer_baseline_entry", "infer_candidate_entry"):
        require_string(probe, key)
    reproduction = require_dict(probe, "reproduction_expected")
    for key in (
        "train_matches_train_dump",
        "infer_matches_infer_dump",
        "baseline_difference_present",
    ):
        if reproduction.get(key) is not True:
            raise PlanError(f"micro_probe.reproduction_expected.{key} must be true")
    candidate = require_dict(probe, "candidate_expected")
    for key in ("same_shape", "same_dtype", "torch_equal", "no_nonfinite"):
        if candidate.get(key) is not True:
            raise PlanError(f"micro_probe.candidate_expected.{key} must be true")
    if candidate.get("max_abs_diff") != 0:
        raise PlanError("micro_probe.candidate_expected.max_abs_diff must be 0")
    require_string(probe, "cleanup")
def validate_rollback(change: dict[str, Any]) -> None:
    rollback = require_dict(change, "rollback")
    require_string(rollback, "scope")
    steps = require_string_list(rollback, "steps")
    require_bool(rollback, "restart_required")
    reject_unsafe_steps(steps, "rollback.steps")


def validate_ready(plan: dict[str, Any], status: str) -> None:
    repair_unit_id = require_string(plan, "repair_unit_id")
    if not REPAIR_UNIT_ID.match(repair_unit_id):
        raise PlanError("repair_unit_id must match R followed by at least three digits")
    member_diff_ids = require_string_list(plan, "member_diff_ids")
    if any(not DIFF_ID.match(value) for value in member_diff_ids):
        raise PlanError("member_diff_ids must contain valid Difference Packet IDs")
    if len(member_diff_ids) != len(set(member_diff_ids)):
        raise PlanError("member_diff_ids must be unique")
    analysis = validate_analysis(plan)
    validate_generalization(plan)
    require_bool(plan, "requires_explicit_approval")
    change = require_dict(plan, "causal_change")
    require_string(change, "id")
    require_string(change, "pattern")
    require_string(change, "hypothesis")
    method = require_string(change, "method")
    if method not in ALLOWED_METHODS:
        raise PlanError(f"unsupported change method: {method}")
    if status == "CONFIG_ONLY_CHANGE" and method != "config":
        raise PlanError("CONFIG_ONLY_CHANGE requires method=config")
    if status == "PATCH_PROPOSAL_READY" and method == "config":
        raise PlanError("method=config must use CONFIG_ONLY_CHANGE")

    target = require_dict(change, "target")
    target_side = require_string(target, "side")
    if target_side not in {"infer", "train", "shared"}:
        raise PlanError("target.side must be infer, train, or shared")
    require_string(target, "repository")
    require_string(target, "baseline_revision")
    active_runtime_source = require_string(target, "active_runtime_source")
    validate_repo_relative_path(active_runtime_source, "target.active_runtime_source")
    dirty_paths = require_string_list(target, "preexisting_dirty_paths", allow_empty=True)
    for index, path in enumerate(dirty_paths):
        validate_repo_relative_path(path, f"target.preexisting_dirty_paths[{index}]")
    target_symbols = require_string_list(
        target, "symbols", allow_empty=(method == "config")
    )
    changed_paths = require_string_list(
        target, "changed_paths", allow_empty=(method == "config")
    )
    for index, path in enumerate(changed_paths):
        validate_repo_relative_path(path, f"target.changed_paths[{index}]")
    if method == "config":
        require_string_list(change, "config_keys")
    elif not target_symbols:
        raise PlanError("code changes require at least one active target symbol")

    if target_side != "infer":
        require_string_list(change, "train_side_exception_evidence")
        if plan.get("requires_explicit_approval") is not True:
            raise PlanError("train/shared changes require explicit approval")
        if analysis["root_cause_class"] not in {
            "Config",
            "Missing step",
            "Framework bug",
            "Weight sync",
            "State transition",
            "Control plane",
        }:
            raise PlanError(
                "train/shared changes require config/framework/state-transition evidence"
            )

    train_reference = require_dict(change, "train_reference")
    for key in ("repository", "revision", "path", "symbol", "semantic_boundary"):
        require_string(train_reference, key)
    validate_repo_relative_path(train_reference["path"], "train_reference.path")

    require_string(change, "mathematical_delta")
    require_string_list(change, "interface_invariants")
    require_string_list(change, "applicability_guards")
    require_string(change, "activation_probe")
    preflight = require_string_list(change, "preflight_checks")
    reject_unsafe_steps(preflight, "preflight_checks")
    validate_micro_probe(change)
    validate_exact_gate(change)
    require_string_list(change, "success_gates")

    failure_routes = require_dict(change, "failure_routes")
    missing_routes = FAILURE_ROUTES - set(failure_routes)
    if missing_routes:
        raise PlanError(
            "failure_routes is missing: " + ", ".join(sorted(missing_routes))
        )
    for key in FAILURE_ROUTES:
        require_string(failure_routes, key)

    validate_rollback(change)
    require_string(change, "performance_risk")
    require_string(change, "packaging_disposition")
    candidate_path = require_string(change, "case_candidate_path")
    validate_repo_relative_path(candidate_path, "case_candidate_path")
    if "references/fix-cases" in candidate_path.replace("\\", "/"):
        raise PlanError(
            "case_candidate_path must stay session-local until native target and final prefill verification"
        )
    if require_string(plan, "next_action") not in {
        "primary_agent_review",
        "remote_ops_apply_one_change",
    }:
        raise PlanError("ready plan next_action must hand off to the primary integrator")


def validate_dependency(plan: dict[str, Any]) -> None:
    dependency = require_dict(plan, "dependency")
    for key in ("name", "version_or_compatibility", "source", "reason"):
        require_string(dependency, key)
    require_string_list(dependency, "compatibility_evidence")
    if plan.get("requires_explicit_approval") is not True:
        raise PlanError("dependency installation requires explicit approval")


def validate_nonready(plan: dict[str, Any], status: str) -> None:
    require_string(plan, "reason")
    require_string_list(plan, "evidence_files")
    require_string(plan, "next_action")
    if status == "DEPENDENCY_REQUIRED":
        validate_dependency(plan)


def validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise PlanError("fix plan root must be an object")
    status = require_string(plan, "status")
    if status not in ALLOWED_STATUSES:
        raise PlanError(f"unsupported fix status: {status}")
    if status in READY_STATUSES:
        validate_ready(plan, status)
    else:
        validate_nonready(plan, status)
    return {
        "valid": True,
        "status": status,
        "repair_unit_id": plan.get("repair_unit_id") if status in READY_STATUSES else None,
        "causal_change_id": (
            plan.get("causal_change", {}).get("id") if status in READY_STATUSES else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a model-specific true-on-policy fix plan"
    )
    parser.add_argument("--plan", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = validate_plan(load_json(args.plan.resolve()))
    except PlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
