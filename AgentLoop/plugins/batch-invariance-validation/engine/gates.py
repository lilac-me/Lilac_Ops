"""Schema, evidence, and TP-confirmation gates for workflow artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON object required")
    return value


def validate_schema(document: dict[str, Any], schema_path: str | Path) -> list[str]:
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema)
    return [f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in sorted(validator.iter_errors(document), key=lambda item: list(item.path))]


def validate_artifacts(result: dict[str, Any], run_dir: str | Path) -> list[str]:
    errors: list[str] = []
    root = Path(run_dir).resolve()
    for artifact in result.get("artifacts", []):
        relative = Path(str(artifact.get("path", "")))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"artifact escapes run directory: {relative}")
            continue
        if not candidate.is_file():
            errors.append(f"artifact missing: {relative}")
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != artifact.get("sha256"):
            errors.append(f"artifact hash mismatch: {relative}")
    return errors


def validate_tp_confirmation(config: dict[str, Any]) -> list[str]:
    confirmation = config.get("tp_confirmation", {})
    errors: list[str] = []
    if confirmation.get("status") != "confirmed":
        errors.append("BLOCKED_TP_PLAN_UNCONFIRMED: status must be confirmed")
    for key in ("source", "confirmed_utc", "user_response"):
        if not str(confirmation.get(key, "")).strip():
            errors.append(f"BLOCKED_TP_PLAN_UNCONFIRMED: {key} is required")
    return errors


def write_auto_planned_run_config(candidate: dict[str, Any], destination: str | Path,
                                  schema_path: str | Path, model_config: dict[str, Any],
                                  runnable_tps: list[int] | None = None,
                                  requested_tps: list[int] | None = None) -> None:
    """Derive TP from visible devices and atomically persist an auditable config."""
    from engine.tp_planner import plan_tp

    config = dict(candidate)
    topology = dict(config.get("device_topology", {}))
    visible = list(topology.get("visible_devices", []))
    planning = plan_tp(visible, model_config, runnable_tps=runnable_tps,
                       requested_tps=requested_tps)
    selected = dict(planning["selected_plan"])
    config["tp_plan"] = {key: selected[key] for key in (
        "phase1", "phase2_long_output", "phase2_long_input", "phase2_matrix", "phase3"
    )}
    topology.update(device_count=len(visible), parallelism={"tp": selected["phase1"]})
    config["device_topology"] = topology
    config["tp_planning"] = planning
    config["execution_policy"] = {
        "tp_planning": "auto_from_visible_devices", "environment_repair": "auto",
        "operator_repair": "auto_minimal_change", "blocked_recovery": "evidence_driven",
        "max_issue_recovery_rounds": 5, "max_stage_recovery_rounds": 10,
        "max_run_recovery_rounds": 20,
    }
    config["tp_confirmation"] = {
        "status": "confirmed", "source": "auto_planned",
        "confirmed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "user_response": f"auto-planned from {len(visible)} visible devices",
    }
    destination = Path(destination)
    if destination.is_file():
        previous = load_json(destination)
        comparable = dict(config)
        comparable["tp_confirmation"] = previous.get("tp_confirmation")
        comparable["tp_change_history"] = previous.get("tp_change_history", [])
        if comparable == previous:
            return
        config["tp_change_history"] = list(previous.get("tp_change_history", [])) + [{
            "changed_utc": config["tp_confirmation"]["confirmed_utc"],
            "reason": "automatic plan or execution configuration revision",
            "affected_stages": ["environment", "phase1", "phase2", "phase3"],
            "previous_plan": previous["tp_plan"], "new_plan": config["tp_plan"],
            "confirmation": config["tp_confirmation"],
        }]
    errors = validate_schema(config, schema_path) + validate_tp_confirmation(config) + validate_tp_capacity(config)
    if errors:
        raise ValueError("; ".join(errors))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def validate_tp_capacity(config: dict[str, Any]) -> list[str]:
    topology = config.get("device_topology", {})
    count = int(topology.get("device_count", 0) or 0)
    visible = topology.get("visible_devices", [])
    plan = config.get("tp_plan", {})
    requested = [plan.get("phase1"), plan.get("phase2_long_output"),
                 plan.get("phase2_long_input"), plan.get("phase3")]
    requested.extend(plan.get("phase2_matrix", []))
    errors: list[str] = []
    if len(visible) != count:
        errors.append("device_topology.device_count must equal visible_devices length")
    if any(isinstance(tp, int) and tp > count for tp in requested):
        errors.append("confirmed TP plan exceeds visible device count")
    return errors


def write_confirmed_run_config(candidate: dict[str, Any], destination: str | Path,
                               schema_path: str | Path, user_response: str,
                               source: str = "user_explicit") -> None:
    """Atomically write only after an explicit, non-empty confirmation reply."""
    if not user_response.strip():
        raise ValueError("BLOCKED_TP_PLAN_UNCONFIRMED: explicit user response required")
    config = dict(candidate)
    confirmation = {
        "status": "confirmed",
        "source": source,
        "confirmed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "user_response": user_response,
    }
    config["tp_confirmation"] = confirmation
    history = list(config.get("tp_change_history", []))
    if history and history[-1].get("new_plan") == config.get("tp_plan"):
        latest = dict(history[-1])
        latest["confirmation"] = confirmation
        history[-1] = latest
        config["tp_change_history"] = history
    errors = validate_schema(config, schema_path) + validate_tp_confirmation(config) + validate_tp_capacity(config)
    if errors:
        raise ValueError("; ".join(errors))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--document", required=True, type=Path)
    validate.add_argument("--schema", required=True, type=Path)
    validate.add_argument("--run-dir", type=Path)
    validate.add_argument("--verification", type=Path)
    confirm = sub.add_parser("confirm-tp")
    confirm.add_argument("--candidate", required=True, type=Path)
    confirm.add_argument("--output", required=True, type=Path)
    confirm.add_argument("--schema", required=True, type=Path)
    confirm.add_argument("--user-response", required=True)
    auto = sub.add_parser("auto-plan")
    auto.add_argument("--candidate", required=True, type=Path)
    auto.add_argument("--model-config", required=True, type=Path)
    auto.add_argument("--output", required=True, type=Path)
    auto.add_argument("--schema", required=True, type=Path)
    auto.add_argument("--runnable-tps", nargs="*", type=int)
    auto.add_argument("--requested-tps", nargs="+", type=int)
    args = parser.parse_args()
    if args.command == "confirm-tp":
        write_confirmed_run_config(load_json(args.candidate), args.output, args.schema, args.user_response)
        print(json.dumps({"status": "confirmed", "path": str(args.output)}, ensure_ascii=False))
        return 0
    if args.command == "auto-plan":
        model_config = load_json(args.model_config)
        write_auto_planned_run_config(load_json(args.candidate), args.output, args.schema,
                                      model_config, args.runnable_tps, args.requested_tps)
        print(json.dumps({"status": "auto_planned", "path": str(args.output)}, ensure_ascii=False))
        return 0
    document = load_json(args.document)
    errors = validate_schema(document, args.schema)
    if args.schema.name == "run_config.schema.json":
        errors.extend(validate_tp_confirmation(document))
        errors.extend(validate_tp_capacity(document))
    if args.run_dir:
        errors.extend(validate_artifacts(document, args.run_dir))
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "document": str(args.document),
        "schema": str(args.schema),
        "valid": not errors,
        "errors": errors,
    }
    if args.verification:
        args.verification.parent.mkdir(parents=True, exist_ok=True)
        args.verification.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
