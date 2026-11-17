"""Fail-closed dispatch helpers used by the public orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.gates import load_json, validate_schema, validate_tp_confirmation
from engine.state_machine import state_map


def next_agent_for_state(sm: dict[str, Any], state: str, *, run_config: str | Path | None,
                         run_config_schema: str | Path) -> str | None:
    spec = state_map(sm).get(state)
    if spec is None:
        raise ValueError(f"unknown state: {state}")
    if spec.get("terminal") is True or spec.get("pause") is True:
        return None
    agent = spec.get("agent")
    if state in {"tp_plan_gate", "finalize"}:
        return str(agent) if agent else None
    if run_config is None or not Path(run_config).is_file():
        raise ValueError("BLOCKED_TP_PLAN_UNCONFIRMED: confirmed run_config.json required before dispatch")
    config = load_json(run_config)
    errors = validate_schema(config, run_config_schema) + validate_tp_confirmation(config)
    if errors:
        raise ValueError("; ".join(errors))
    return str(agent) if agent else None
