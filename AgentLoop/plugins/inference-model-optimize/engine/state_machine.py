# State Machine Core
# ==================
# YAML-driven workflow state machine — loads pipeline definitions, evaluates
# transition conditions, maintains append-only transition logs, and supports
# critic-auditable replay verification.
#
# Design principles (from PR #609):
#   1. YAML is the single source of truth for all transitions.
#   2. Transition log is APPEND-ONLY — history is never rewritten.
#   3. Iteration counts are derived from the log (single source of truth).
#   4. Bootstrap: when the log is empty, infer current state from PROGRESS.md
#      handoff lines and immediately persist the inference.
#   5. Critic-verifiable: verify_log() replays the log against YAML and
#      checks every transition is declared + iter caps are respected.
#
# CLI Usage:
#   python -m engine.state_machine next \\
#       --workspace .workspace/run-001 \\
#       --yaml plugins/inference-model-optimize/workflows/optimize-pipeline.yaml \\
#       --handoff "→ orchestrator: done — 部署方案完成"
#
#   python -m engine.state_machine verify \\
#       --workspace .workspace/run-001 \\
#       --yaml plugins/inference-model-optimize/workflows/optimize-pipeline.yaml

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.fsm_conditions import eval_condition, extract_handoff

try:
    import yaml  # PyYAML
except ImportError:
    sys.stderr.write("state_machine: PyYAML required (pip install pyyaml)\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRANSITIONS_FILENAME = "state_transitions.jsonl"
TERMINAL_STATES = ("done", "abort")
PAUSE_STATES = ("await_user_decision",)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
def load_state_machine(yaml_path: str | Path) -> dict[str, Any]:
    """Load a pipeline YAML definition.

    Returns a dict with keys: pipeline (name, version, initial_state,
    total_spawn_cap), states (list of state dicts).
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"State machine YAML not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_state_spec(sm: dict, state_id: str) -> dict | None:
    """Find a state definition by id in the state machine."""
    for s in sm.get("states", []) or sm.get("phase_o4_states", []):
        if s["id"] == state_id:
            return s
    return None


def get_initial_state(sm: dict) -> str:
    """Return the initial state from the pipeline definition."""
    pipeline = sm.get("pipeline", {})
    return pipeline.get("initial_state", sm.get("phase_o4_initial_state", "await_worker"))


def get_total_spawn_cap(sm: dict) -> int:
    """Return the global spawn cap from the pipeline definition."""
    pipeline = sm.get("pipeline", {})
    return int(pipeline.get("total_spawn_cap", sm.get("phase_o4_total_spawn_cap", 20)))


# ---------------------------------------------------------------------------
# Transition log (append-only JSONL)
# ---------------------------------------------------------------------------
def log_path(ws: Path) -> Path:
    """Return the path to the transition log for a workspace."""
    return ws / TRANSITIONS_FILENAME


def read_log(ws: Path) -> list[dict]:
    """Read all transition entries from the append-only log."""
    p = log_path(ws)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def append_log(ws: Path, entry: dict) -> None:
    """Append a single transition entry to the log."""
    p = log_path(ws)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def iter_counts_from_log(ws: Path, sm: dict) -> dict[str, int]:
    """Count how many times each state has been entered.

    Entries with `rollback_kind == "infra"` are SKIPPED so that
    infrastructure rollbacks (SCP timeout, JSON parse failure, etc.)
    don't consume the iter_below_cap budget.
    """
    counts: dict[str, int] = {}
    log = read_log(ws)
    for entry in log:
        to_state = entry.get("to_state")
        if not to_state:
            continue
        # Infra rollbacks don't consume iteration budget
        if entry.get("rollback_kind") == "infra":
            continue
        spec = get_state_spec(sm, to_state)
        if spec is None:
            continue
        counter = spec.get("iter_counter", to_state)
        counts[counter] = counts.get(counter, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Workspace snapshot
# ---------------------------------------------------------------------------
def snapshot(ws: Path) -> dict[str, Any]:
    """Collect workspace state that transition conditions evaluate against.

    Reads verification.json, analysis.md, and other artifacts from the
    workspace directory.
    """
    import re as _re

    snap: dict[str, Any] = {
        "workspace": str(ws),
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    # verification.json
    vj = ws / "verification.json"
    if vj.exists():
        try:
            snap["verification"] = json.loads(vj.read_text(encoding="utf-8"))
        except Exception:
            snap["verification"] = None
    else:
        snap["verification"] = None

    # analysis.md (lowercased for substring matching)
    am = ws / "analysis.md"
    snap["analysis_md_lower"] = am.read_text(encoding="utf-8").lower() if am.exists() else ""

    # probe_report.md → classification
    pr = ws / "probe_report.md"
    snap["probe_classification"] = None
    snap["probe_report_has_actionable_fix"] = False
    if pr.exists():
        txt = pr.read_text(encoding="utf-8", errors="replace")
        cm = _re.search(
            r"(?im)^[\s\-*]*Type\s*:\s*\*{0,2}\s*(requirement|convention)\b", txt
        )
        snap["probe_classification"] = cm.group(1).lower() if cm else None
        m = _re.search(r"(?im)^#+\s*Recommendation\b.*?(?=^#+\s|\Z)", txt, _re.DOTALL)
        snap["probe_report_has_actionable_fix"] = bool(
            m and len(m.group(0).strip()) > 40
        )

    # user_decision.md → next_state target
    ud = ws / "user_decision.md"
    snap["user_decision_target"] = None
    if ud.exists():
        utxt = ud.read_text(encoding="utf-8", errors="replace")
        dm = _re.search(r"(?im)^[\s\-*]*next_state\s*:\s*([a-z_]+)\s*$", utxt)
        if dm:
            snap["user_decision_target"] = dm.group(1).lower()

    # op_taxonomy — domain plugins can extend this
    snap["op_taxonomy"] = {"class": "unknown", "complexity": "unknown"}

    return snap


# ---------------------------------------------------------------------------
# Current state inference
# ---------------------------------------------------------------------------
def get_current_state(ws: Path, sm: dict) -> str:
    """Determine the current state of a workspace.

    Order:
      1. Log non-empty → tail's to_state (canonical, no side effects).
      2. Log empty + PROGRESS.md has a recognizable handoff → bootstrap
         from initial state to the handoff target, persist synthetic
         transitions, return target.
      3. Log empty + no usable handoff → return initial state.
    """
    log = read_log(ws)
    if log:
        return log[-1]["to_state"]

    # Bootstrap from PROGRESS.md
    initial = get_initial_state(sm)
    handoff = _extract_handoff_from_progress(ws)
    if handoff:
        try:
            decision = next_state(ws, initial, handoff, sm=sm)
            target = decision.get("next_state")
            if target and target not in TERMINAL_STATES:
                _persist_bootstrap_chain(ws, initial, target, handoff, decision)
                return target
        except Exception:
            pass

    return initial


def _extract_handoff_from_progress(ws: Path) -> str | None:
    """Read PROGRESS.md, find the most recent canonical handoff line.

    Scopes search to lines AFTER the last 'EXIT' marker, or the last
    non-empty line if no EXIT marker exists.
    """
    progress = ws / "PROGRESS.md"
    if not progress.exists():
        return None
    try:
        text = progress.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    lines = text.splitlines()

    # Find the last EXIT marker
    scope_start = 0
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if s.startswith(("### EXIT", "## EXIT", "# EXIT")) or s in ("EXIT",):
            scope_start = i + 1
            break
    if scope_start == 0:
        # No EXIT marker — use only the last non-empty line
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                scope_start = i
                break
        else:
            return None

    # Collect canonical handoff lines
    prefixes = (
        "→ orchestrator:",
        "@aog-precision-probe",
        "@aog-kernel-optimizer",
        "@aog-fused-optimizer",
        "@aog-determinism-analyzer",
        "@aog-researcher",
        "handoff:",
    )
    candidates = []
    for line in lines[scope_start:]:
        s = line.strip()
        if not s:
            continue
        # Strip markdown decoration
        if s.startswith("- "):
            s = s[2:].strip()
        if s.startswith("**Exit handoff**:"):
            s = s.split(":", 1)[1].strip().strip("`")
        # Match canonical prefixes
        if any(s.startswith(p) for p in prefixes):
            candidates.append(s)

    return candidates[-1] if candidates else None


def _persist_bootstrap_chain(
    ws: Path,
    initial_state: str,
    target: str,
    handoff: str,
    decision: dict,
) -> None:
    """Write synthetic init→initial and initial→target transitions.

    Called when bootstrap successfully routes from initial state to a
    non-terminal target using a PROGRESS.md handoff.
    """
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    append_log(ws, {
        "ts": ts,
        "from_state": "init",
        "to_state": initial_state,
        "handoff": "",
        "matched_transition_index": -1,
        "rationale": "bootstrap: empty log, PROGRESS.md handoff present",
        "iter_counts_snapshot": {},
    })
    append_log(ws, {
        "ts": ts,
        "from_state": initial_state,
        "to_state": target,
        "handoff": handoff,
        "matched_transition_index": decision.get("matched_transition_index", -1),
        "rationale": "bootstrap: " + decision.get("rationale", "PROGRESS.md handoff routed to target"),
        "iter_counts_snapshot": {},
    })


# ---------------------------------------------------------------------------
# Next-state decision
# ---------------------------------------------------------------------------
def next_state(
    ws: Path,
    from_state: str,
    handoff: str,
    *,
    sm: dict | None = None,
    plugin: Any = None,
    runtime_kwargs: dict | None = None,
) -> dict:
    """Compute the next state given the current state and agent handoff.

    Args:
        ws: Workspace directory.
        from_state: Current state id (e.g. 'architect').
        handoff: Agent's handoff line (text).
        sm: Pre-loaded state machine dict (loads from YAML if None).
        plugin: Optional plugin instance for plugin_method conditions.
        runtime_kwargs: Optional per-invocation kwargs for plugin methods.

    Returns:
        dict with: next_state, matched_transition_index, rationale,
        from_state, handoff, iter_counts_snapshot.
        On error: dict with error, next_state=None.
    """
    if sm is None:
        raise ValueError("sm is required — load via load_state_machine() first")

    spec = get_state_spec(sm, from_state)
    if spec is None:
        return {
            "error": f"state '{from_state}' not found in YAML states",
            "next_state": None,
            "from_state": from_state,
        }

    snap = snapshot(ws)
    iter_counts = iter_counts_from_log(ws, sm)
    ctx = {
        "handoff": handoff,
        "snapshot": snap,
        "iter_counts": iter_counts,
        "ws": ws,
        "sm": sm,
        "plugin": plugin,
        "current_state": from_state,
        "runtime_kwargs": runtime_kwargs or {},
    }

    for idx, trans in enumerate(spec.get("exit_transitions", [])):
        cond = trans.get("condition", {})
        if eval_condition(cond, ctx):
            target = trans.get("goto")
            # Magic token: resolve from user_decision.md
            if target == "__from_user_decision__":
                target = snap.get("user_decision_target")
                if target is None:
                    return {
                        "error": "user_decision_target_in matched but target is None",
                        "next_state": None,
                        "from_state": from_state,
                    }
            return {
                "next_state": target,
                "matched_transition_index": idx,
                "rationale": trans.get("rationale", "matched condition"),
                "from_state": from_state,
                "handoff": handoff,
                "iter_counts_snapshot": dict(iter_counts),
            }

    # No transition matched — contract violation (states should have fallback)
    return {
        "error": f"no transition matched from state '{from_state}' (handoff={handoff!r})",
        "next_state": None,
        "from_state": from_state,
    }


def is_terminal(state: str) -> bool:
    """Return True if the state is a terminal state."""
    return state in TERMINAL_STATES


def is_pause(state: str) -> bool:
    """Return True if the state is a pause state (resumable, not terminal)."""
    return state in PAUSE_STATES


# ---------------------------------------------------------------------------
# Verify — replay log against YAML
# ---------------------------------------------------------------------------
def verify_log(ws: Path, sm: dict | None = None) -> list[str]:
    """Critic audit: replay the transition log and check every entry
    against the YAML declaration.

    Returns a list of violation strings. Empty list = clean.
    """
    if sm is None:
        raise ValueError("sm is required — load via load_state_machine() first")

    log = read_log(ws)
    violations: list[str] = []

    total_spawns = 0
    total_cap = get_total_spawn_cap(sm)
    running_iter_counts: dict[str, int] = {}

    valid_states = {s["id"] for s in (sm.get("states", []) or sm.get("phase_o4_states", []))}
    valid_states |= set(TERMINAL_STATES) | {"init"}

    for i, entry in enumerate(log):
        from_s = entry.get("from_state")
        to_s = entry.get("to_state")

        # Validate target state is declared
        if to_s not in valid_states:
            violations.append(
                f"line {i + 1}: to_state '{to_s}' not declared in YAML states"
            )
            continue

        # Validate transition is declared in from_state's exit_transitions
        from_spec = get_state_spec(sm, from_s) if from_s else None
        if from_spec is not None:
            declared = {t.get("goto") for t in from_spec.get("exit_transitions", [])}
            if to_s not in declared and from_s != "init":
                violations.append(
                    f"line {i + 1}: transition {from_s} → {to_s} not declared "
                    f"(allowed: {sorted(declared)})"
                )

            # Re-evaluate the actual handoff against the YAML conditions. A
            # destination being declared is not sufficient: the recorded
            # handoff must have selected that exact transition in order.
            if from_s != "init":
                replay_ctx = {
                    "handoff": entry.get("handoff", ""),
                    "snapshot": snapshot(ws),
                    "iter_counts": dict(running_iter_counts),
                    "ws": ws,
                    "sm": sm,
                    "plugin": None,
                    "current_state": from_s,
                    "runtime_kwargs": {},
                }
                replay_match = None
                for transition_index, transition in enumerate(
                    from_spec.get("exit_transitions", [])
                ):
                    if eval_condition(transition.get("condition", {}), replay_ctx):
                        replay_match = (transition_index, transition.get("goto"))
                        break

                if replay_match is None:
                    violations.append(
                        f"line {i + 1}: handoff does not match any YAML transition "
                        f"from '{from_s}'"
                    )
                elif replay_match[1] != to_s:
                    violations.append(
                        f"line {i + 1}: handoff selects '{replay_match[1]}' but log records '{to_s}'"
                    )
                recorded_index = entry.get("matched_transition_index")
                if (
                    recorded_index is not None
                    and replay_match is not None
                    and recorded_index != replay_match[0]
                ):
                    violations.append(
                        f"line {i + 1}: matched transition index {recorded_index} "
                        f"does not replay as {replay_match[0]}"
                    )

        # Iter cap check
        to_spec = get_state_spec(sm, to_s) if to_s else None
        if to_spec is not None:
            # Skip infra rollbacks from cap counting
            if entry.get("rollback_kind") != "infra":
                counter = to_spec.get("iter_counter", to_s)
                running_iter_counts[counter] = running_iter_counts.get(counter, 0) + 1
                cap = int(to_spec.get("iter_cap", 999))
                if running_iter_counts[counter] > cap:
                    violations.append(
                        f"line {i + 1}: iter cap exceeded for '{counter}': "
                        f"{running_iter_counts[counter]} > {cap}"
                    )

        total_spawns += 1
        if total_spawns > total_cap:
            violations.append(
                f"line {i + 1}: total spawn cap exceeded: {total_spawns} > {total_cap}"
            )

    return violations


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def _main():
    import argparse

    ap = argparse.ArgumentParser(
        description="State machine driver — YAML-driven workflow orchestration"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # next
    p_next = sub.add_parser("next", help="Decide next state given current state + handoff")
    p_next.add_argument("--workspace", required=True, type=Path)
    p_next.add_argument("--yaml", required=True, type=Path, help="Path to pipeline YAML")
    p_next.add_argument("--handoff", default="")
    p_next.add_argument("--current-state", help="Override auto-detected current state")
    p_next.add_argument("--dry-run", action="store_true", help="Don't append to log")

    # verify
    p_verify = sub.add_parser("verify", help="Critic-audit the transition log")
    p_verify.add_argument("--workspace", required=True, type=Path)
    p_verify.add_argument("--yaml", required=True, type=Path)

    # current
    p_cur = sub.add_parser("current", help="Show current state of a workspace")
    p_cur.add_argument("--workspace", required=True, type=Path)
    p_cur.add_argument("--yaml", required=True, type=Path)

    args = ap.parse_args()

    if args.cmd == "next":
        ws = args.workspace.resolve()
        if not ws.exists():
            print(json.dumps({"error": f"workspace not found: {ws}"}))
            sys.exit(2)

        sm = load_state_machine(args.yaml)
        current = args.current_state or get_current_state(ws, sm)
        result = next_state(ws, current, args.handoff, sm=sm)

        if "error" in result:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(2)

        if not args.dry_run:
            log_entry = {
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "from_state": result["from_state"],
                "to_state": result["next_state"],
                "handoff": args.handoff,
                "matched_transition_index": result["matched_transition_index"],
                "rationale": result["rationale"],
                "iter_counts_snapshot": result["iter_counts_snapshot"],
            }
            append_log(ws, log_entry)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.cmd == "verify":
        ws = args.workspace.resolve()
        if not ws.exists():
            print(f"workspace not found: {ws}")
            sys.exit(2)

        sm = load_state_machine(args.yaml)
        violations = verify_log(ws, sm)
        if not violations:
            sys.stderr.write(f"✓ state_machine.verify: {ws.name} log clean\n")
            sys.exit(0)

        sys.stderr.write(f"✗ state_machine.verify: {ws.name} has violations:\n")
        for v in violations:
            sys.stderr.write(f"  - {v}\n")
        sys.exit(2)

    elif args.cmd == "current":
        ws = args.workspace.resolve()
        sm = load_state_machine(args.yaml)
        cur = get_current_state(ws, sm)
        counts = iter_counts_from_log(ws, sm)
        print(json.dumps({
            "current_state": cur,
            "is_terminal": is_terminal(cur),
            "is_pause": is_pause(cur),
            "iter_counts": counts,
            "log_length": len(read_log(ws)),
        }, indent=2))


if __name__ == "__main__":
    _main()
