#!/usr/bin/env python3
"""Initialize or validate a resumable True On-Policy alignment state card."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "templates"
    / "state_card.json"
)
SHA256 = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
MODES = {"Full", "Local-analyze", "Plan-only"}
REGISTRY_STATUSES = {"open", "frozen", "reopened"}
TRUTH_STATUSES = {
    "tensor_prefill": {"exact", "nonzero", "not_run", "evidence_missing"},
    "weight_sync": {"exact", "mismatch", "not_run", "evidence_missing", "not_applicable"},
    "stateful_training": {"pass", "fail", "not_run", "evidence_missing"},
    "long_horizon_e2e": {"pass", "fail", "not_run", "evidence_missing"},
    "reduced_model_transfer": {"pass", "fail", "not_run", "evidence_missing", "not_applicable"},
    "requested_model_final": {"pass", "fail", "not_run", "evidence_missing"},
    "production_patch_hygiene": {"pass", "fail", "not_run", "evidence_missing"},
    "clean_runtime_activation": {"pass", "fail", "not_run", "evidence_missing"},
    "performance_regression": {"pass", "fail", "not_run", "evidence_missing", "not_applicable"},
}
GATE_TRUTH_EXPECTED = {
    "static_tensor_prefill": ("tensor_prefill", "exact"),
    "weight_sync": ("weight_sync", "exact"),
    "stateful_training": ("stateful_training", "pass"),
    "long_horizon_e2e": ("long_horizon_e2e", "pass"),
    "reduced_model_transfer": ("reduced_model_transfer", "pass"),
    "requested_model_final": ("requested_model_final", "pass"),
    "production_patch_hygiene": ("production_patch_hygiene", "pass"),
    "clean_runtime_activation": ("clean_runtime_activation", "pass"),
    "performance_regression": ("performance_regression", "pass"),
}


class StateCardError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateCardError(f"cannot read state card {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StateCardError("state card root must be an object")
    return value


def require_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise StateCardError(f"{key} must be an object")
    return value


def require_string(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise StateCardError(f"{key} must be a non-empty string")
    return value.strip()


def require_bool(parent: dict[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise StateCardError(f"{key} must be a boolean")
    return value


def require_strings(parent: dict[str, Any], key: str, *, allow_empty: bool = False) -> list[str]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise StateCardError(f"{key} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise StateCardError(f"{key} may not be empty")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise StateCardError(f"{key} must not contain duplicates")
    return normalized


def validate_optional_sha(parent: dict[str, Any], key: str) -> None:
    value = parent.get(key)
    if value is not None and (not isinstance(value, str) or not SHA256.match(value)):
        raise StateCardError(f"{key} must be null or a full sha256 value")


def validate(card: dict[str, Any]) -> dict[str, Any]:
    if card.get("schema_version") != 2:
        raise StateCardError("schema_version must be 2")
    session_id = require_string(card, "session_id")
    if card.get("mode") not in MODES:
        raise StateCardError("mode is invalid")

    objective = require_object(card, "objective")
    require_string(objective, "statement")
    require_string(objective, "requested_model")
    require_string(objective, "requested_scope")
    required_gates = require_strings(objective, "required_gates")
    unknown_gates = sorted(set(required_gates) - set(GATE_TRUTH_EXPECTED))
    if unknown_gates:
        raise StateCardError(
            "objective.required_gates contains unknown gates: " + ", ".join(unknown_gates)
        )
    require_string(objective, "stop_condition")
    success_metric = require_object(objective, "success_metric")
    require_string(success_metric, "name")
    require_string(success_metric, "target")
    historical_step = objective.get("historical_failure_step")
    if historical_step is not None and (
        isinstance(historical_step, bool)
        or not isinstance(historical_step, int)
        or historical_step < 0
    ):
        raise StateCardError("objective.historical_failure_step must be non-negative or null")

    authorization = require_object(card, "authorization")
    require_string(authorization, "record")
    authorization_confirmed = require_bool(authorization, "confirmed")

    environment = require_object(card, "environment")
    for key in (
        "execution_target",
        "working_directory",
        "training_backend",
        "inference_backend",
        "hardware_backend",
        "evidence_provider",
    ):
        require_string(environment, key)

    model = require_object(card, "model")
    require_string(model, "source_model")
    require_string(model, "immutable_diagnostic_model")
    for key in (
        "source_config_sha256",
        "effective_config_sha256",
        "checkpoint_manifest_sha256",
    ):
        validate_optional_sha(model, key)
    minimization = require_object(model, "minimization")
    if minimization.get("strategy") != "semantic_preserving":
        raise StateCardError("model.minimization.strategy must be semantic_preserving")
    coverage = minimization.get("coverage_matrix")
    if not isinstance(coverage, list):
        raise StateCardError("model.minimization.coverage_matrix must be a list")
    for index, entry in enumerate(coverage):
        if not isinstance(entry, dict):
            raise StateCardError(f"coverage_matrix[{index}] must be an object")
        require_string(entry, "semantic_family")
        require_bool(entry, "preserved")
        require_string(entry, "evidence")

    state = require_object(card, "state")
    module = state.get("active_module")
    if isinstance(module, bool) or not isinstance(module, int) or not 1 <= module <= 6:
        raise StateCardError("state.active_module must be an integer from 1 to 6")
    epoch = state.get("discovery_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise StateCardError("state.discovery_epoch must be a non-negative integer")
    if state.get("registry_status") not in REGISTRY_STATUSES:
        raise StateCardError("state.registry_status is invalid")
    task_status = state.get("task_status")
    if task_status not in {"open", "complete", "blocked"}:
        raise StateCardError("state.task_status is invalid")
    if state.get("registry_status") == "reopened":
        require_string(state, "reopen_reason")
    require_string(state, "next_gate")

    truth = require_object(card, "truth")
    truth_summary: dict[str, str] = {}
    for axis, statuses in TRUTH_STATUSES.items():
        record = require_object(truth, axis)
        status = record.get("status")
        if status not in statuses:
            raise StateCardError(f"truth.{axis}.status is invalid")
        evidence = require_strings(record, "evidence", allow_empty=True)
        if status not in {"not_run", "evidence_missing", "not_applicable"} and not evidence:
            raise StateCardError(
                f"truth.{axis}.status={status} requires non-empty evidence"
            )
        truth_summary[axis] = status

    runtime = require_object(card, "runtime")
    diagnostic_replay = require_bool(runtime, "diagnostic_replay_enabled")
    diagnostic_capture = require_bool(runtime, "diagnostic_capture_enabled")
    require_bool(runtime, "clean_process_started")
    for key in ("compile_cache", "prefix_cache", "task_queue", "weight_reload"):
        require_string(runtime, key)
    jobs = runtime.get("jobs")
    if not isinstance(jobs, list):
        raise StateCardError("runtime.jobs must be a list")

    artifacts = require_object(card, "artifacts")
    require_string(artifacts, "session_root")
    require_strings(artifacts, "retained", allow_empty=True)
    history = card.get("history")
    if not isinstance(history, list):
        raise StateCardError("history must be a list")
    event_ids: set[str] = set()
    for index, event in enumerate(history):
        label = f"history[{index}]"
        if not isinstance(event, dict):
            raise StateCardError(f"{label} must be an object")
        event_id = require_string(event, "event_id")
        if event_id in event_ids:
            raise StateCardError(f"{label}.event_id is duplicated")
        event_ids.add(event_id)
        require_string(event, "event")
        if event.get("status") not in {"running", "pass", "fail", "partial", "aborted"}:
            raise StateCardError(f"{label}.status is invalid")
        require_string(event, "started_at")
        if event.get("status") == "running":
            if event.get("ended_at") is not None or event.get("duration_seconds") is not None:
                raise StateCardError(f"{label} running event must not have end/duration")
        else:
            require_string(event, "ended_at")
            duration = event.get("duration_seconds")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or duration < 0
            ):
                raise StateCardError(f"{label}.duration_seconds must be non-negative")
        require_strings(event, "evidence", allow_empty=True)

    completion_ready = False
    if task_status == "complete":
        if not authorization_confirmed:
            raise StateCardError("complete task requires confirmed authorization record")
        for gate in required_gates:
            axis, expected = GATE_TRUTH_EXPECTED[gate]
            if truth_summary[axis] != expected:
                raise StateCardError(
                    f"complete task gate {gate} requires truth.{axis}.status={expected}"
                )
        if diagnostic_replay or diagnostic_capture:
            raise StateCardError("complete task requires diagnostic replay/capture disabled")
        if runtime.get("clean_process_started") is not True:
            raise StateCardError("complete task requires clean_process_started=true")
        crucial_values = [
            objective.get("statement"),
            objective.get("requested_model"),
            objective.get("requested_scope"),
            environment.get("execution_target"),
            environment.get("working_directory"),
            model.get("source_model"),
            model.get("immutable_diagnostic_model"),
            artifacts.get("session_root"),
        ]
        if any("待填写" in str(value) for value in crucial_values):
            raise StateCardError("complete task may not retain placeholder values")
        completion_ready = True

    return {
        "valid": True,
        "schema_version": 2,
        "session_id": session_id,
        "required_gates": required_gates,
        "registry_status": state["registry_status"],
        "task_status": task_status,
        "discovery_epoch": epoch,
        "truth": truth_summary,
        "diagnostic_runtime_active": diagnostic_replay or diagnostic_capture,
        "completion_ready": completion_ready,
    }


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--state", type=Path)
    mode.add_argument("--init", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        if args.init is not None:
            target = args.init.resolve()
            if target.exists() and not args.force:
                raise StateCardError(f"refusing to overwrite existing file: {target}")
            value = load_json(TEMPLATE)
            validate(value)
            write_text(target, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            result = {"valid": True, "initialized": str(target)}
        else:
            result = validate(load_json(args.state.resolve()))
    except StateCardError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
