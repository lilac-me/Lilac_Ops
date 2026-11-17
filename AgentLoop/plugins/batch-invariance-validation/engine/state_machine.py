"""YAML-driven state machine with append-only audit logging."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

TRANSITIONS_FILENAME = "state_transitions.jsonl"


def load_state_machine(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"State machine YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("state machine must be a YAML object")
    data["_source_path"] = str(path.resolve())
    return data


def state_specs(sm: dict[str, Any]) -> list[dict[str, Any]]:
    return list(sm.get("states", []))


def state_map(sm: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in state_specs(sm)}


def initial_state(sm: dict[str, Any]) -> str:
    return str(sm["pipeline"]["initial_state"])


def read_log(workspace: str | Path) -> list[dict[str, Any]]:
    path = Path(workspace) / TRANSITIONS_FILENAME
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid transition log line {number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid transition log line {number}: object required")
        entries.append(value)
    return entries


def append_log(workspace: str | Path, entry: dict[str, Any]) -> None:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / TRANSITIONS_FILENAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _entry_hash(entry: dict[str, Any]) -> str:
    payload = {key: value for key, value in entry.items() if key != "entry_hash"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_snapshot_hash(document: dict[str, Any]) -> str:
    encoded = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_state(workspace: str | Path, sm: dict[str, Any]) -> str:
    log = read_log(workspace)
    return str(log[-1]["to_state"]) if log else initial_state(sm)


def _value_at(context: dict[str, Any], dotted: str) -> Any:
    value: Any = context
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def eval_condition(condition: dict[str, Any], handoff: str, context: dict[str, Any]) -> bool:
    if condition.get("always") is True:
        return True
    if "handoff_match" in condition:
        return handoff.strip() == str(condition["handoff_match"]).strip()
    if "result_match" in condition:
        expected = condition["result_match"]
        return all(_value_at(context, key) == value for key, value in expected.items())
    if "all_of" in condition:
        return all(eval_condition(item, handoff, context) for item in condition["all_of"])
    if "any_of" in condition:
        return any(eval_condition(item, handoff, context) for item in condition["any_of"])
    return False


def select_transition(sm: dict[str, Any], from_state: str, handoff: str,
                      context: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    spec = state_map(sm).get(from_state)
    if spec is None:
        raise ValueError(f"unknown state: {from_state}")
    for index, transition in enumerate(spec.get("exit_transitions", [])):
        if eval_condition(transition.get("condition", {}), handoff, context or {}):
            return index, transition
    raise ValueError(f"no transition matched state {from_state}")


def iter_counts(log: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in log:
        if entry.get("rollback_kind") == "infra":
            continue
        state = str(entry.get("to_state", ""))
        counts[state] = counts.get(state, 0) + 1
    return counts


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_paths(sm: dict[str, Any]) -> tuple[Path, Path]:
    source = Path(str(sm.get("_source_path", "")))
    if not source.is_file():
        raise ValueError("state machine source path is unavailable")
    contracts = source.parent.parent / "contracts"
    return contracts / "run_config.schema.json", contracts / "stage_result.schema.json"


def _validate_runtime_input(workspace: Path, sm: dict[str, Any], source: str, handoff: str,
                            context: dict[str, Any], selected_index: int,
                            selected: dict[str, Any], config_override: dict[str, Any] | None = None) -> tuple[str, dict[str, str]]:
    from engine.gates import (load_json, validate_artifacts, validate_schema,
                              validate_tp_capacity, validate_tp_confirmation)

    run_schema, stage_schema = _contract_paths(sm)
    config_path = workspace / "run_config.json"
    requires_config = source not in {"tp_plan_gate", "blocked_tp_plan_unconfirmed"}
    if source == "tp_plan_gate" and handoff.strip() in {
            "handoff: TP_PLAN_CONFIRMED", "handoff: TP_PLAN_AUTO_READY"}:
        requires_config = True
    config: dict[str, Any] = config_override or {}
    if requires_config:
        if config_override is None and not config_path.is_file():
            raise ValueError("BLOCKED_TP_PLAN_UNCONFIRMED: confirmed run_config.json required")
        if config_override is None:
            config = load_json(config_path)
        errors = (validate_schema(config, run_schema) + validate_tp_confirmation(config)
                  + validate_tp_capacity(config))
        if errors:
            raise ValueError("; ".join(errors))

    spec = state_map(sm)[source]
    agent = str(spec.get("agent") or "")
    worker = bool(agent) and not agent.endswith(":orchestrator") and not spec.get("pause")
    artifact_hashes: dict[str, str] = {}
    if worker:
        if not context:
            raise ValueError(f"{source}: validated stage_result context is required")
        errors = validate_schema(context, stage_schema) + validate_artifacts(context, workspace)
        if context.get("state") != source:
            errors.append(f"stage_result.state must equal current state {source}")
        if context.get("handoff") != handoff.strip():
            errors.append("stage_result.handoff must equal transition handoff")
        if context.get("run_id") != config.get("run_id"):
            errors.append("stage_result.run_id must equal run_config.run_id")
        if selected.get("condition", {}).get("always") is True:
            errors.append(f"{source}: handoff is not explicitly declared")
        summary = context.get("config_summary", {})
        required_artifact_types = {
            "environment_admission": {"environment_snapshot"},
            "phase1_profiling": {"profiling_manifest"},
            "phase1_triage": {"operator_analysis"},
            "phase1_critical_path_review": {"critical_path_evidence"},
            "phase1_operator_10k": {"operator_10k_result"},
            "phase1_replacement": {"replacement_evidence"},
            "phase1_reprofile": {"reprofile_result"},
            "phase2_long_output": {"functional_result"},
            "phase2_long_input": {"functional_result"},
            "phase2_matrix": {"matrix_result"},
            "phase2_failure_diagnosis": {"diagnosis_evidence"},
            "phase3_math500": {"math500_manifest", "math500_round_results", "math500_raw_text_hashes"},
        }.get(source, set())
        present_types = {str(item.get("type")) for item in context.get("artifacts", [])}
        missing_types = required_artifact_types - present_types
        if missing_types:
            errors.append(f"{source}: missing required artifact types {sorted(missing_types)}")
        actual_tp = summary.get("tp")
        plan = config.get("tp_plan", {})
        expected_tp: Any = None
        if source.startswith("phase1_"):
            expected_tp = plan.get("phase1")
        elif source == "phase2_long_output":
            expected_tp = plan.get("phase2_long_output")
        elif source == "phase2_long_input":
            expected_tp = plan.get("phase2_long_input")
        elif source == "phase2_failure_diagnosis":
            expected_tp = config.get("tp_planning", {}).get("selected_plan", {}).get(
                "phase2_diagnosis", plan.get("phase2_matrix"))
        elif source == "phase2_matrix":
            expected_tp = plan.get("phase2_matrix")
        elif source == "phase3_math500":
            expected_tp = plan.get("phase3")
        elif source == "environment_admission":
            expected_tp = config.get("device_topology", {}).get("parallelism", {}).get("tp")
        if actual_tp is None:
            errors.append("stage_result.config_summary.tp is required")
        elif isinstance(expected_tp, list):
            actual_values = actual_tp if isinstance(actual_tp, list) else [actual_tp]
            if not set(actual_values) <= set(expected_tp):
                errors.append("stage_result TP is outside the confirmed matrix plan")
            if source == "phase2_matrix" and handoff.strip() == "handoff: MATRIX_PASS":
                if set(actual_values) != set(expected_tp):
                    errors.append("MATRIX_PASS must account for every confirmed matrix TP")
                tp_results = summary.get("tp_results", [])
                statuses = [item.get("status") for item in tp_results]
                if not statuses or "PASS" not in statuses:
                    errors.append("MATRIX_PASS requires at least one fully passing TP")
                if {item.get("tp") for item in tp_results} != set(expected_tp):
                    errors.append("MATRIX_PASS tp_results must cover every confirmed matrix TP")
                scenarios = summary.get("scenario_results", [])
                shapes = {(16, 100, 512), (32, 100, 512), (16, 1024, 2048), (32, 1024, 2048)}
                expected_scenarios = {(tp, *shape) for tp in expected_tp for shape in shapes}
                actual_scenarios = {(item.get("tp"), item.get("batch_size"), item.get("input_tokens"), item.get("output_tokens")) for item in scenarios}
                if actual_scenarios != expected_scenarios or any(item.get("rounds") != 3 for item in scenarios):
                    errors.append("MATRIX_PASS must cover every confirmed TP and both token scenarios at BS 16/32 for three rounds")
                passing_tps = {item.get("tp") for item in tp_results if item.get("status") == "PASS"}
                if any(item.get("status") != "PASS" for item in scenarios if item.get("tp") in passing_tps):
                    errors.append("every scenario for a passing TP must PASS")
        elif expected_tp is not None and actual_tp != expected_tp:
            errors.append(f"stage_result TP {actual_tp} does not match confirmed TP {expected_tp}")
        if source == "phase3_math500" and handoff.strip() == "handoff: MATH500_PASS":
            rounds = summary.get("round_results", [])
            if {item.get("round") for item in rounds} != set(range(1, 11)):
                errors.append("MATH500_PASS requires exactly rounds 1 through 10")
            if summary.get("accuracy_pass") is not True:
                errors.append("MATH500_PASS requires accuracy_pass=true")
            if len({item.get("score") for item in rounds}) != 1:
                errors.append("MATH500_PASS requires identical scores across all rounds")
            if len({item.get("raw_text_sha256") for item in rounds}) != 1:
                errors.append("MATH500_PASS requires identical per-sample raw text evidence across all rounds")
        target = str(selected.get("goto"))
        target_spec = state_map(sm).get(target, {})
        if context.get("outcome") == "BLOCKED":
            allowed_resume = target_spec.get("resume_states") or [target_spec.get("resume_state")]
            if handoff.strip() == "handoff: TP_PLAN_CHANGED":
                allowed_resume = ["tp_plan_gate"]
            if context.get("resume_state") not in allowed_resume:
                errors.append("stage_result.resume_state is not a declared recovery edge")
        if errors:
            raise ValueError("; ".join(errors))
        artifact_hashes = {str(item["path"]): str(item["sha256"]) for item in context.get("artifacts", [])}
    return sha256_file(config_path) if config_override is None and config_path.is_file() else "", artifact_hashes


def transition(workspace: str | Path, sm: dict[str, Any], handoff: str, *,
               from_state: str | None = None, context: dict[str, Any] | None = None,
               agent: str = "orchestrator", config_hash: str = "",
               artifact_hashes: dict[str, str] | None = None, dry_run: bool = False) -> dict[str, Any]:
    workspace = Path(workspace)
    actual_source = current_state(workspace, sm)
    if from_state is not None and from_state != actual_source:
        raise ValueError(f"from_state override {from_state} does not match current state {actual_source}")
    source = actual_source
    existing_log = read_log(workspace)
    if source == "recovery_controller":
        from engine.recovery import read_attempts, recovery_status
        from engine.workflow_runner import _blocked_context, RECOVERY_HANDOFFS
        attempts = read_attempts(workspace)
        consumed = sum(item.get("from_state") == "recovery_controller" for item in existing_log)
        if len(attempts) != consumed + 1:
            raise ValueError("recovery transition requires one new audited attempt")
        attempt = attempts[-1]
        resume, reason = _blocked_context(existing_log)
        if attempt.get("state") != resume or attempt.get("reason_code") != reason:
            raise ValueError("recovery attempt does not match blocked state and reason")
        exhausted = recovery_status(workspace, attempt["recovery_key"], attempt["stage"])["exhausted"]
        expected = "RECOVERY_EXHAUSTED" if exhausted and attempt["repair_result"] != "PASS" else RECOVERY_HANDOFFS.get(resume)
        if handoff.strip() != f"handoff: {expected}":
            raise ValueError("recovery handoff does not match audited attempt and budget")
    spawn_cap = int(sm.get("pipeline", {}).get("total_spawn_cap", 0) or 0)
    if spawn_cap and len(existing_log) >= spawn_cap:
        raise ValueError(f"total transition cap exceeded: {spawn_cap}")
    index, selected = select_transition(sm, source, handoff, context)
    spec = state_map(sm)[source]
    worker = bool(spec.get("agent")) and not str(spec.get("agent")).endswith(":orchestrator") and not spec.get("pause")
    actual_config_hash, actual_artifact_hashes = _validate_runtime_input(
        workspace, sm, source, handoff, context or {}, index, selected
    )
    previous_config_hash = str(existing_log[-1].get("config_hash", "")) if existing_log else ""
    confirmation_event = source == "tp_plan_gate" and handoff.strip() in {
        "handoff: TP_PLAN_CONFIRMED", "handoff: TP_PLAN_AUTO_READY"
    }
    if previous_config_hash and actual_config_hash != previous_config_hash and not confirmation_event:
        raise ValueError("run_config.json changed without a new TP confirmation gate")
    if previous_config_hash and actual_config_hash != previous_config_hash and confirmation_event:
        from engine.gates import load_json
        active_config = load_json(workspace / "run_config.json")
        history = active_config.get("tp_change_history", [])
        if not history:
            raise ValueError("changed TP plan requires tp_change_history and explicit reconfirmation")
        latest = history[-1]
        if latest.get("new_plan") != active_config.get("tp_plan"):
            raise ValueError("latest TP change history does not match the active TP plan")
        if latest.get("confirmation") != active_config.get("tp_confirmation"):
            raise ValueError("latest TP change history does not match the active confirmation")
    if config_hash and config_hash != actual_config_hash:
        raise ValueError("provided config hash does not match run_config.json")
    if artifact_hashes and artifact_hashes != actual_artifact_hashes:
        raise ValueError("provided artifact hashes do not match stage_result")
    target = str(selected["goto"])
    entry = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "from_state": source,
        "to_state": target,
        "agent": agent,
        "handoff": handoff.strip(),
        "matched_transition_index": index,
        "rationale": str(selected.get("rationale", "")),
        "config_hash": actual_config_hash,
        "artifact_hashes": actual_artifact_hashes,
        "iter_counts_snapshot": iter_counts(existing_log),
        "previous_entry_hash": existing_log[-1].get("entry_hash", "") if existing_log else "",
    }
    if context and worker:
        entry["stage_result"] = context
        from engine.gates import load_json
        entry["run_config_snapshot"] = load_json(workspace / "run_config.json")
        entry["run_config_snapshot_hash"] = _config_snapshot_hash(entry["run_config_snapshot"])
        entry["stage_result_hash"] = hashlib.sha256(
            json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    if context and context.get("outcome") == "BLOCKED":
        entry["blocked_resume_state"] = context.get("resume_state")
        entry["blocked_reason_code"] = context.get("reason_code")
    if spec.get("pause"):
        required_resume = existing_log[-1].get("blocked_resume_state") if existing_log else None
        if required_resume and target not in {required_resume, "recovery_controller"}:
            raise ValueError(f"blocked state must resume to {required_resume}, not {target}")
    catalog_key = handoff.strip().removeprefix("handoff: ")
    if target == "finalize":
        entry["pending_final_outcome"] = sm.get("handoff_catalog", {}).get(catalog_key, {}).get("outcome")
    if source == "finalize":
        pending = existing_log[-1].get("pending_final_outcome") if existing_log else None
        expected = "handoff: FINALIZE_FAILURE" if pending == "FAIL" else "handoff: FINALIZE_SUCCESS"
        if handoff.strip() != expected:
            raise ValueError(f"finalize must preserve pending outcome {pending}: expected {expected}")
    entry["entry_hash"] = _entry_hash(entry)
    if not dry_run:
        append_log(workspace, entry)
        progress = [
            "# Batch Invariance Validation Progress",
            "",
            f"- 当前状态：`{target}`",
            f"- 最后事件：`{handoff.strip()}`",
            f"- 更新时间：`{entry['ts']}`",
        ]
        if context:
            if context.get("reason_code"):
                progress.append(f"- 原因码：`{context['reason_code']}`")
            if context.get("resume_state"):
                progress.append(f"- 恢复状态：`{context['resume_state']}`")
        (workspace / "PROGRESS.md").write_text("\n".join(progress) + "\n", encoding="utf-8")
        verification = {
            "checked_utc": entry["ts"],
            "state": target,
            "transition_entry_hash": entry["entry_hash"],
            "schema_valid": True,
            "gate_valid": True,
            "evidence_valid": True,
            "valid": True,
            "errors": [],
        }
        (workspace / "verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return entry


def validate_definition(sm: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    specs = state_specs(sm)
    ids = [str(item.get("id", "")) for item in specs]
    known = set(ids)
    if len(ids) != len(known):
        errors.append("duplicate state id")
    if initial_state(sm) not in known:
        errors.append("initial state is not declared")
    graph: dict[str, set[str]] = {state: set() for state in known}
    for spec in specs:
        state = str(spec.get("id", ""))
        terminal = spec.get("terminal") is True or spec.get("kind") == "terminal"
        pause = spec.get("pause") is True or spec.get("kind") == "pause"
        transitions = list(spec.get("exit_transitions", []))
        if not terminal and not transitions:
            errors.append(f"{state}: non-terminal state has no transitions")
        if transitions:
            always = [i for i, item in enumerate(transitions) if item.get("condition", {}).get("always") is True]
            if not terminal and (not always or always[-1] != len(transitions) - 1):
                errors.append(f"{state}: final fail-closed always transition required")
        if pause and not any(str(item.get("goto")) != state for item in transitions):
            errors.append(f"{state}: pause state has no recovery transition")
        tokens: set[str] = set()
        for item in transitions:
            target = str(item.get("goto", ""))
            if target not in known:
                errors.append(f"{state}: unknown goto {target}")
            else:
                graph[state].add(target)
            token = item.get("condition", {}).get("handoff_match")
            if token and token in tokens:
                errors.append(f"{state}: duplicate handoff {token}")
            if token:
                tokens.add(str(token))
    reached: set[str] = set()
    stack = [initial_state(sm)]
    while stack:
        node = stack.pop()
        if node in reached:
            continue
        reached.add(node)
        stack.extend(graph.get(node, set()) - reached)
    for state in sorted(known - reached):
        errors.append(f"unreachable state: {state}")
    sinks = {
        str(spec["id"]) for spec in specs
        if spec.get("terminal") is True or spec.get("pause") is True
    }
    can_finish = set(sinks)
    changed = True
    while changed:
        changed = False
        for source, targets in graph.items():
            if source not in can_finish and targets & can_finish:
                can_finish.add(source)
                changed = True
    for state in sorted(known - can_finish):
        errors.append(f"state cannot reach pause or terminal: {state}")
    return errors


def verify_log(workspace: str | Path, sm: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    workspace = Path(workspace)
    expected = initial_state(sm)
    previous_hash = ""
    pending_final_outcome: str | None = None
    active_config_hash = ""
    log = read_log(workspace)
    spawn_cap = int(sm.get("pipeline", {}).get("total_spawn_cap", 0) or 0)
    if spawn_cap and len(log) > spawn_cap:
        violations.append(f"total transition cap exceeded: {len(log)} > {spawn_cap}")
    for index, entry in enumerate(log, 1):
        source = str(entry.get("from_state", ""))
        required_fields = {
            "ts", "from_state", "to_state", "agent", "handoff", "matched_transition_index",
            "config_hash", "artifact_hashes", "previous_entry_hash", "entry_hash",
        }
        missing = required_fields - set(entry)
        if missing:
            violations.append(f"line {index}: missing fields {sorted(missing)}")
        if entry.get("previous_entry_hash") != previous_hash:
            violations.append(f"line {index}: previous entry hash mismatch")
        if entry.get("entry_hash") != _entry_hash(entry):
            violations.append(f"line {index}: entry hash mismatch")
        if source != expected:
            violations.append(f"line {index}: expected from_state {expected}, got {source}")
        try:
            matched, selected = select_transition(sm, source, str(entry.get("handoff", "")))
            if str(selected.get("goto")) != str(entry.get("to_state")):
                violations.append(f"line {index}: handoff selects {selected.get('goto')}")
            if matched != entry.get("matched_transition_index"):
                violations.append(f"line {index}: transition index mismatch")
        except ValueError as exc:
            violations.append(f"line {index}: {exc}")
        if state_map(sm).get(source, {}).get("pause") and index > 1:
            required_resume = log[index - 2].get("blocked_resume_state")
            if required_resume and entry.get("to_state") not in {required_resume, "recovery_controller"}:
                violations.append(f"line {index}: blocked state must resume to {required_resume}")
        source_spec = state_map(sm).get(source, {})
        is_worker = bool(source_spec.get("agent")) and not str(source_spec.get("agent")).endswith(":orchestrator") and not source_spec.get("pause")
        if is_worker and ("stage_result" not in entry or "stage_result_hash" not in entry):
            violations.append(f"line {index}: worker transition missing stage result audit record")
        if "stage_result" in entry:
            actual_stage_hash = hashlib.sha256(json.dumps(
                entry["stage_result"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            if entry.get("stage_result_hash") != actual_stage_hash:
                violations.append(f"line {index}: stage result hash mismatch")
            result = entry["stage_result"]
            if result.get("state") != source or result.get("handoff") != entry.get("handoff"):
                violations.append(f"line {index}: stage result does not bind state and handoff")
            if result.get("outcome") == "BLOCKED":
                if entry.get("blocked_resume_state") != result.get("resume_state"):
                    violations.append(f"line {index}: blocked resume state audit mismatch")
                if entry.get("blocked_reason_code") != result.get("reason_code"):
                    violations.append(f"line {index}: blocked reason code audit mismatch")
            snapshot = entry.get("run_config_snapshot")
            if not isinstance(snapshot, dict):
                violations.append(f"line {index}: worker transition missing run config snapshot")
            else:
                if entry.get("run_config_snapshot_hash") != _config_snapshot_hash(snapshot):
                    violations.append(f"line {index}: run config snapshot hash mismatch")
                if entry.get("run_config_snapshot_hash") != entry.get("config_hash"):
                    violations.append(f"line {index}: run config snapshot does not match transition config")
                try:
                    replay_index, replay_selected = select_transition(sm, source, str(entry.get("handoff", "")), result)
                    _validate_runtime_input(workspace, sm, source, str(entry.get("handoff", "")),
                                            result, replay_index, replay_selected,
                                            config_override=snapshot)
                except ValueError as exc:
                    violations.append(f"line {index}: stage result replay failed: {exc}")
        config_hash = str(entry.get("config_hash", ""))
        confirmation_event = source == "tp_plan_gate" and entry.get("handoff") in {
            "handoff: TP_PLAN_CONFIRMED", "handoff: TP_PLAN_AUTO_READY"
        }
        if active_config_hash and config_hash != active_config_hash and not confirmation_event:
            violations.append(f"line {index}: config hash changed without TP confirmation")
        if config_hash:
            active_config_hash = config_hash
        for relative, expected_hash in entry.get("artifact_hashes", {}).items():
            artifact = (workspace / relative).resolve()
            try:
                artifact.relative_to(workspace.resolve())
            except ValueError:
                violations.append(f"line {index}: artifact escapes workspace: {relative}")
                continue
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                violations.append(f"line {index}: artifact hash mismatch: {relative}")
        if str(entry.get("to_state")) == "finalize":
            pending_final_outcome = entry.get("pending_final_outcome")
        if source == "finalize":
            required_handoff = "handoff: FINALIZE_FAILURE" if pending_final_outcome == "FAIL" else "handoff: FINALIZE_SUCCESS"
            if entry.get("handoff") != required_handoff:
                violations.append(f"line {index}: finalize outcome does not preserve {pending_final_outcome}")
        previous_hash = str(entry.get("entry_hash", ""))
        expected = str(entry.get("to_state", expected))
    if active_config_hash:
        config_path = workspace / "run_config.json"
        if not config_path.is_file() or sha256_file(config_path) != active_config_hash:
            violations.append("active run_config.json hash mismatch")
    return violations


def _load_context(path: str | None) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("current", "next", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--workspace", required=True, type=Path)
        command.add_argument("--yaml", required=True, type=Path)
        if name == "next":
            command.add_argument("--handoff", required=True)
            command.add_argument("--context")
            command.add_argument("--config", type=Path)
            command.add_argument("--run-dir", type=Path)
            command.add_argument("--agent", default="orchestrator")
            command.add_argument("--dry-run", action="store_true")
    definition = sub.add_parser("validate-definition")
    definition.add_argument("--yaml", required=True, type=Path)
    args = parser.parse_args()
    sm = load_state_machine(args.yaml)
    if args.command == "current":
        print(json.dumps({"current_state": current_state(args.workspace, sm)}, ensure_ascii=False))
        return 0
    if args.command == "next":
        context = _load_context(args.context)
        config_hash = sha256_file(args.config) if args.config else ""
        artifact_hashes: dict[str, str] = {}
        if args.run_dir:
            root = args.run_dir.resolve()
            for artifact in context.get("artifacts", []):
                path = (root / str(artifact.get("path", ""))).resolve()
                if path.is_file() and root in path.parents:
                    artifact_hashes[str(artifact["path"])] = sha256_file(path)
        result = transition(args.workspace, sm, args.handoff, context=context,
                            agent=args.agent, config_hash=config_hash,
                            artifact_hashes=artifact_hashes, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    errors = validate_definition(sm) if args.command == "validate-definition" else verify_log(args.workspace, sm)
    print(json.dumps({"valid": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
