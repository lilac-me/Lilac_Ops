"""Deterministic control surface for automatic dispatch and evidence-driven recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from engine.orchestrator import next_agent_for_state
from engine.recovery import record_attempt, recovery_status, write_exhausted_report
from engine.state_machine import current_state, load_state_machine, read_log, state_map, transition

RECOVERY_HANDOFFS = {
    "environment_admission": "RECOVERY_ENVIRONMENT",
    "phase1_profiling": "RECOVERY_PHASE1",
    "phase2_long_output": "RECOVERY_PHASE2_LONG_OUTPUT",
    "phase2_long_input": "RECOVERY_PHASE2_LONG_INPUT",
    "phase2_matrix": "RECOVERY_PHASE2_MATRIX",
    "phase2_failure_diagnosis": "RECOVERY_PHASE2_DIAGNOSIS",
    "phase3_math500": "RECOVERY_PHASE3",
}
AUTO_RECOVERY_STATES = {"blocked_environment", "blocked_phase1", "blocked_phase2", "blocked_phase3"}


def _blocked_context(log: list[dict[str, Any]]) -> tuple[str, str]:
    for entry in reversed(log):
        resume = entry.get("blocked_resume_state")
        reason = entry.get("blocked_reason_code")
        if resume and reason:
            return str(resume), str(reason)
    raise ValueError("recovery requires an audited blocked result")


def next_action(workspace: str | Path, sm: dict[str, Any], run_schema: str | Path) -> dict[str, Any]:
    workspace = Path(workspace)
    state = current_state(workspace, sm)
    if state in AUTO_RECOVERY_STATES:
        entry = transition(workspace, sm, "handoff: RECOVERY_START", agent="orchestrator")
        state = entry["to_state"]
    if state == "recovery_controller":
        resume, reason = _blocked_context(read_log(workspace))
        state_spec = state_map(sm)[state]
        return {"action": "recover", "state": state, "resume_state": resume,
                "reason_code": reason, "goal": state_spec["goal"],
                "allowed_actions": state_spec["allowed_actions"],
                "forbidden_actions": state_spec["forbidden_actions"],
                "attempt_limits": state_spec["attempt_limits"],
                "required_artifacts": state_spec["required_artifacts"]}
    agent = next_agent_for_state(sm, state, run_config=workspace / "run_config.json",
                                 run_config_schema=run_schema)
    return {"action": "terminal" if agent is None else "dispatch",
            "state": state, "agent": agent}


def complete_recovery(workspace: str | Path, sm: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(workspace)
    if current_state(workspace, sm) != "recovery_controller":
        raise ValueError("recovery attempt is only valid in recovery_controller")
    resume, _ = _blocked_context(read_log(workspace))
    recorded = record_attempt(workspace, attempt)
    status = recovery_status(workspace, recorded["recovery_key"], recorded["stage"])
    exhausted = status["exhausted"] and recorded.get("repair_result") != "PASS"
    report = str(write_exhausted_report(workspace, recorded["recovery_key"])) if exhausted else None
    handoff = "RECOVERY_EXHAUSTED" if exhausted else RECOVERY_HANDOFFS.get(resume)
    if handoff is None:
        raise ValueError(f"no recovery handoff for {resume}")
    entry = transition(workspace, sm, f"handoff: {handoff}", agent="orchestrator")
    return {"transition": entry, "recovery": status, "exhausted_report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["next", "complete-recovery"])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--yaml", required=True, type=Path)
    parser.add_argument("--run-schema", required=True, type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    sm = load_state_machine(args.yaml)
    if args.command == "complete-recovery":
        if args.attempt is None:
            parser.error("--attempt is required for complete-recovery")
        attempt = json.loads(args.attempt.read_text(encoding="utf-8"))
        result = complete_recovery(args.workspace, sm, attempt)
    else:
        result = next_action(args.workspace, sm, args.run_schema)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
