# State Executor
# ==============
# Orchestrator-facing Python API wrapping the state machine engine.
# Provides snapshot, next_state, record_transition, validate_log, and
# agent-type resolution for the main orchestration loop.
#
# This is the primary API that orchestrator scripts (any platform) call.
# It imports the state_machine module and adds convenience methods.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from engine import state_machine as _sm
from engine.agent_registry import load_agent_registry  # noqa: F401 (exported)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TERMINAL_STATES = ("done", "abort")
PAUSE_STATES = ("await_user_decision",)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TransitionDecision:
    """Result of state_executor.next_state(workspace, handoff)."""
    next_state: str
    matched_transition_index: int = -1
    rationale: str = ""
    from_state: str = ""
    handoff: str = ""
    rollback_kind: Optional[str] = None
    """None = no rollback, 'infra' = free re-entry (doesn't consume iter budget),
    'algorithm' = consumes budget (logged for audit)."""


@dataclass
class StateSnapshot:
    """Workspace state at a point in time. Used for orchestrator decisions."""
    op: str
    workspace: Path
    current_state: str
    is_terminal: bool
    is_pause: bool
    next_agent: str | None
    last_handoff: str
    iter_counts: dict[str, int]
    iter_caps: dict[str, int]
    log_length: int


# ---------------------------------------------------------------------------
# Agent mapping
# ---------------------------------------------------------------------------
# Map state → agent type. This is the DEFAULT mapping; plugins can override
# via the agent_registry or the plugins/<name>/agents/ tree.
STATE_TO_AGENT: dict[str, str | None] = {
    "architect": "inference-model-optimize:architect:architect",
    "developer": "inference-model-optimize:developer:developer",
    "tester": "inference-model-optimize:tester:tester",
    "reviewer": "inference-model-optimize:reviewer:reviewer",
    "optimizer": "inference-model-optimize:optimizer:optimizer",
    "finalize": None,   # orchestrator generates the final report itself
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_state_machine(yaml_path: str | Path) -> dict:
    """Load a pipeline YAML definition."""
    return _sm.load_state_machine(yaml_path)


def current_state(workspace: Path, sm: dict) -> str:
    """Get current state from state_transitions.jsonl tail."""
    return _sm.get_current_state(workspace, sm)


def next_agent(state: str) -> str | None:
    """Return the agent type owning `state`, or None for terminal/pause states."""
    if state in TERMINAL_STATES:
        return None
    return STATE_TO_AGENT.get(state)


def is_terminal(state: str) -> bool:
    """Return True if the state is terminal."""
    return state in TERMINAL_STATES


def is_pause(state: str) -> bool:
    """Return True if the state is a pause state (resumable)."""
    return state in PAUSE_STATES


def iter_count(workspace: Path, sm: dict, counter: str) -> int:
    """How many times agent type `counter` has been invoked for this workspace."""
    counts = _sm.iter_counts_from_log(workspace, sm)
    return counts.get(counter, 0)


def iter_cap(state: str, workspace: Path, sm: dict) -> int:
    """Read iter_cap for a state from YAML."""
    spec = _sm.get_state_spec(sm, state)
    if spec is None:
        return 999
    return int(spec.get("iter_cap", 999))


def at_iter_cap(workspace: Path, sm: dict, state: str) -> bool:
    """True if current iter count for state's counter equals or exceeds cap."""
    spec = _sm.get_state_spec(sm, state)
    if spec is None:
        return False
    counter = spec.get("iter_counter", state)
    cap = iter_cap(state, workspace, sm)
    counts = _sm.iter_counts_from_log(workspace, sm)
    return counts.get(counter, 0) >= cap


def snapshot(workspace: Path, sm: dict) -> StateSnapshot:
    """Read full workspace state for orchestrator decisions."""
    state = _sm.get_current_state(workspace, sm)
    counts = _sm.iter_counts_from_log(workspace, sm)
    caps = {}
    for s in sm.get("states", []) or sm.get("phase_o4_states", []):
        counter = s.get("iter_counter", s["id"])
        caps[counter] = int(s.get("iter_cap", 999))

    last_handoff = ""
    log = _sm.read_log(workspace)
    if log:
        last_handoff = log[-1].get("handoff", "")

    return StateSnapshot(
        op=workspace.name,
        workspace=workspace,
        current_state=state,
        is_terminal=is_terminal(state),
        is_pause=is_pause(state),
        next_agent=next_agent(state),
        last_handoff=last_handoff[:200],
        iter_counts=counts,
        iter_caps=caps,
        log_length=len(log),
    )


def next_state(
    workspace: Path,
    sm: dict,
    handoff: str,
    *,
    from_state: str | None = None,
    dry_run: bool = False,
    plugin: Any = None,
    runtime_kwargs: dict | None = None,
) -> TransitionDecision:
    """Compute next state given current state + handoff.

    If from_state is None, derives it from the log via current_state().
    If dry_run is False, appends to state_transitions.jsonl.
    """
    if from_state is None:
        from_state = current_state(workspace, sm)

    result = _sm.next_state(
        workspace, from_state, handoff,
        sm=sm, plugin=plugin, runtime_kwargs=runtime_kwargs,
    )

    if "error" in result:
        raise StateMachineError(result["error"])

    decision = TransitionDecision(
        next_state=result["next_state"],
        matched_transition_index=result.get("matched_transition_index", -1),
        rationale=result.get("rationale", ""),
        from_state=result.get("from_state", ""),
        handoff=handoff,
    )

    if not dry_run:
        record_transition(workspace, sm, decision)

    return decision


def record_transition(workspace: Path, sm: dict, decision: TransitionDecision) -> None:
    """Append a canonical state_transitions.jsonl entry."""
    counts = _sm.iter_counts_from_log(workspace, sm)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "from_state": decision.from_state,
        "to_state": decision.next_state,
        "handoff": decision.handoff,
        "matched_transition_index": decision.matched_transition_index,
        "rationale": decision.rationale,
        "iter_counts_snapshot": counts,
    }
    if decision.rollback_kind is not None:
        entry["rollback_kind"] = decision.rollback_kind
    _sm.append_log(workspace, entry)


def validate_log(workspace: Path, sm: dict) -> tuple[bool, list[str]]:
    """Sanity-check state_transitions.jsonl: every transition declared in YAML."""
    violations = _sm.verify_log(workspace, sm)
    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class StateMachineError(RuntimeError):
    """Raised when the state machine returns an unrecoverable error."""


# ---------------------------------------------------------------------------
# CLI for smoke-testing
# ---------------------------------------------------------------------------
def _main():
    import argparse

    ap = argparse.ArgumentParser(
        description="state_executor.py — orchestrator-side state machine wrapper"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("snapshot", help="Show full state snapshot for a workspace")
    sp.add_argument("--workspace", required=True, type=Path)
    sp.add_argument("--yaml", required=True, type=Path)

    sp = sub.add_parser("next", help="Dry-run state machine transition")
    sp.add_argument("--workspace", required=True, type=Path)
    sp.add_argument("--yaml", required=True, type=Path)
    sp.add_argument("--handoff", default="", help="Last handoff text")
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("validate", help="Validate state_transitions.jsonl")
    sp.add_argument("--workspace", required=True, type=Path)
    sp.add_argument("--yaml", required=True, type=Path)

    args = ap.parse_args()

    if args.cmd == "snapshot":
        sm = load_state_machine(args.yaml)
        snap = snapshot(args.workspace, sm)
        print(json.dumps({
            "op": snap.op,
            "current_state": snap.current_state,
            "is_terminal": snap.is_terminal,
            "is_pause": snap.is_pause,
            "next_agent": snap.next_agent,
            "last_handoff": snap.last_handoff,
            "iter_counts": snap.iter_counts,
            "iter_caps": snap.iter_caps,
            "log_length": snap.log_length,
        }, indent=2))

    elif args.cmd == "next":
        sm = load_state_machine(args.yaml)
        decision = next_state(args.workspace, sm, args.handoff, dry_run=args.dry_run)
        print(json.dumps({
            "from_state": decision.from_state,
            "to_state": decision.next_state,
            "matched_transition_index": decision.matched_transition_index,
            "rationale": decision.rationale,
            "next_agent": next_agent(decision.next_state),
            "is_terminal": is_terminal(decision.next_state),
        }, indent=2))

    elif args.cmd == "validate":
        sm = load_state_machine(args.yaml)
        ok, errors = validate_log(args.workspace, sm)
        if ok:
            print(json.dumps({"valid": True}))
        else:
            print(json.dumps({"valid": False, "errors": errors}, indent=2))
            sys.exit(2)


if __name__ == "__main__":
    import sys
    _main()
