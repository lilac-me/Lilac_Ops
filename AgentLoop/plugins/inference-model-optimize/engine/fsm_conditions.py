# FSM Condition DSL Evaluator
# ============================
# Evaluates YAML transition-guard conditions against a runtime context.
# Supports nested all_of / any_of and single-primitive conditions.
#
# Design principle (from PR #609):
#   - Every condition is a PURE FUNCTION: context in → bool out.
#   - plugin_method is the extension point for paradigm-specific logic.
#   - Unknown primitives fail CLOSED (return False) — never silently match.
#
# Condition primitives supported (see eval_condition docstring for full list):
#   handoff_match       — handoff line starts with string
#   handoff_contains    — handoff line contains substring
#   always              — unconditional pass/fail
#   iter_below_cap      — iteration counter < cap
#   path_exists         — file exists in workspace
#   plugin_method       — delegate to plugin method
#   all_of / any_of     — logical composition

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helper: extract normalized handoff from agent output
# ---------------------------------------------------------------------------

def extract_handoff(raw_text: str) -> str:
    """Extract the handoff line from an agent's return text.

    Heuristic: find the last line containing a handoff marker
    ('→ orchestrator:', '@aog-', '→ orchestrator:') and return it stripped
    of markdown bold decoration.
    """
    if not raw_text:
        return ""

    # Canonical handoff prefixes (extend as new agent types are added)
    prefixes = (
        "→ orchestrator:",
        "handoff:",
        "@aog-",
    )

    lines = raw_text.splitlines()
    candidates = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Strip inline markdown bold (**) for robust matching
        s_norm = s.replace("**", "")
        if any(s_norm.lower().startswith(p.lower()) for p in prefixes):
            candidates.append(s_norm)

    return candidates[-1] if candidates else raw_text.splitlines()[-1].strip()[:500]


# ---------------------------------------------------------------------------
# Worker signal parser (shared with plugin_method dispatch)
# ---------------------------------------------------------------------------

def parse_worker_signal(handoff: str) -> str:
    """Parse a worker handoff line into a structured signal enum.

    Returns one of: 'done', 'partial_persist', 'abort',
    'structural_rewrite_needed', 'restart_needed', 'unknown'.
    """
    h = handoff.strip().replace("**", "")
    if "structural_rewrite_needed" in h:
        return "structural_rewrite_needed"
    if h.endswith("done") or " done" in h.lower() or "完成" in h:
        return "done"
    if "need_restart" in h.lower() or "需要重启" in h or "restart_params" in h:
        return "restart_needed"
    if "PARTIAL_PERSIST" in h or "partial_persist" in h.lower():
        return "partial_persist"
    if h.endswith("abort") or "abort" in h.lower():
        return "abort"
    return "unknown"


# ---------------------------------------------------------------------------
# Perf threshold resolver (stub — extended by domain plugins)
# ---------------------------------------------------------------------------

def _resolve_perf_threshold(workspace: Path | None, verification: dict | None) -> float:
    """Resolve the finalize/shippable performance floor.

    Stub implementation — domain plugins (e.g. inference-model-optimize)
    can monkey-patch this with their own resolution logic (e.g. reading
    perf-gate profiles or verification.json thresholds).

    Default: 1.0 (parity — only at-parity perf is shippable).
    """
    if verification and isinstance(verification, dict):
        perf = verification.get("performance", {}) or {}
        threshold = perf.get("threshold")
        if threshold is not None:
            try:
                return float(threshold)
            except (TypeError, ValueError):
                pass
    return 1.0


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------

def eval_condition(cond: Any, ctx: dict) -> bool:
    """Evaluate a condition DSL entry against a runtime context.

    Supports nested all_of / any_of and single-primitive conditions.

    ctx keys:
        handoff: str
            Worker handoff line (from extract_handoff).
        snapshot: dict
            Workspace state snapshot (verification.json, analysis.md, etc.).
        iter_counts: dict[str, int]
            Per-counter spawn counts.
        ws: pathlib.Path | None
            Workspace directory.
        sm: dict
            Loaded state machine YAML (for looking up iter caps, etc.).
        plugin: Any | None
            Active plugin instance (for plugin_method dispatch).

    Condition primitives:
        handoff_match: str
            True if handoff (normalized) starts with arg.
        handoff_contains: str
            True if arg is a substring of the normalized handoff.
        always: bool
            Unconditional pass (True) or fail (False).
        iter_below_cap: str
            True if iter_counts[counter_name] < the counter's iter_cap
            (looked up from sm.phase_o4_states).
        path_exists: str
            True if the file at ws/relpath exists.
        plugin_method: str | dict
            Dispatch to plugin.method(...). When plugin is None, returns False.
            String form: method_name is called with (op_class, complexity, worker_signal).
            Dict form: {method, args?, forward_kwargs?}.
        all_of: list
            True if ALL sub-conditions evaluate to True.
        any_of: list
            True if ANY sub-condition evaluates to True.
    """
    # Boolean passthrough
    if isinstance(cond, bool):
        return cond
    if not isinstance(cond, dict):
        return False

    # Logical composition
    if "all_of" in cond:
        return all(eval_condition(c, ctx) for c in cond["all_of"])
    if "any_of" in cond:
        return any(eval_condition(c, ctx) for c in cond["any_of"])

    # Single primitive — extract key and value
    (kind, arg), = cond.items()

    handoff = (ctx.get("handoff") or "").replace("**", "")
    snap = ctx.get("snapshot") or {}
    iter_counts = ctx.get("iter_counts") or {}
    ws: Path | None = ctx.get("ws")
    sm = ctx.get("sm") or {}

    # --- handoff_match ---
    if kind == "handoff_match":
        return handoff.startswith(str(arg))

    # --- handoff_contains ---
    if kind == "handoff_contains":
        return str(arg) in handoff

    # --- always ---
    if kind == "always":
        return bool(arg)

    # --- iter_below_cap ---
    if kind == "iter_below_cap":
        counter_name = str(arg)
        # Find cap from YAML
        cap = None
        for s in sm.get("states", []) or sm.get("phase_o4_states", []):
            if s.get("iter_counter") == counter_name:
                cap = s.get("iter_cap", 999)
                break
        if cap is None:
            # Fallback: look up state by id matching counter name
            for s in sm.get("states", []) or sm.get("phase_o4_states", []):
                if s.get("id") == counter_name:
                    cap = s.get("iter_cap", 999)
                    break
        if cap is None:
            return True  # unknown cap → permissive
        return iter_counts.get(counter_name, 0) < int(cap)

    # --- path_exists ---
    if kind == "path_exists":
        if ws is None:
            return False
        relpath = str(arg).replace("workspace/{op}/", "")
        return (ws / relpath).exists()

    # --- file_absent ---
    if kind == "file_absent":
        if ws is None:
            return True  # no workspace → assume absent
        relpath = str(arg).replace("workspace/{op}/", "")
        # Glob support: if path ends with *, check parent dir
        if relpath.endswith("*") or "*" in relpath:
            import fnmatch
            parent = ws / Path(relpath).parent
            pat = Path(relpath).name
            if not parent.exists():
                return True
            return not any(fnmatch.fnmatch(p.name, pat) for p in parent.iterdir())
        return not (ws / relpath).exists()

    # --- plugin_method ---
    if kind == "plugin_method":
        return _eval_plugin_method(arg, ctx)

    # --- user_decision_target_in ---
    if kind == "user_decision_target_in":
        target = snap.get("user_decision_target")
        return target in (arg or [])

    # Unknown primitive — fail closed
    import sys
    sys.stderr.write(
        f"state_machine: unknown condition primitive '{kind}'\n"
    )
    return False


# ---------------------------------------------------------------------------
# plugin_method dispatch
# ---------------------------------------------------------------------------

def _eval_plugin_method(arg: Any, ctx: dict) -> bool:
    """Evaluate a plugin_method condition.

    plugin_method lets domain-specific logic live in plugin Python code
    rather than in the YAML state machine. When the plugin is not available
    (e.g. ClCode Workflow JS runtime), the condition evaluates to False.

    Form (a) — string method name:
        plugin_method: "should_escalate"
        → calls plugin.should_escalate(op_class, complexity, worker_signal)
          with values resolved from the workspace snapshot.

    Form (b) — dict:
        plugin_method: {method: "should_escalate", args: [...], forward_kwargs: [...]}
        → same dispatch, with optional explicit args and kwarg forwarding.
    """
    plugin = ctx.get("plugin")
    if plugin is None:
        return False

    runtime_kwargs = ctx.get("runtime_kwargs") or {}
    handoff = ctx.get("handoff") or ""
    snap = ctx.get("snapshot") or {}
    ws = ctx.get("ws")

    # Resolve default args from snapshot (op_class, complexity, worker_signal)
    def _resolve_default_args():
        taxonomy = snap.get("op_taxonomy") or {}
        op_class = taxonomy.get("class", "unknown")
        op_complexity = taxonomy.get("complexity", "unknown")
        worker_signal = parse_worker_signal(handoff)
        if op_class == "unknown":
            return None  # Safe default — don't route when nothing is known
        return (op_class, op_complexity, worker_signal)

    forward_kwargs_names: list[str] = []

    if isinstance(arg, str):
        method_name = arg
        resolved = _resolve_default_args()
        if resolved is None:
            return False
        args = resolved
    elif isinstance(arg, dict):
        method_name = arg.get("method", "")
        if "args" in arg:
            args = tuple(arg.get("args", []))
        else:
            resolved = _resolve_default_args()
            if resolved is None:
                return False
            args = resolved
        forward_kwargs_names = list(arg.get("forward_kwargs", []) or [])
    else:
        return False

    method = getattr(plugin, method_name, None)
    if not callable(method):
        return False

    forwarded = {k: runtime_kwargs[k] for k in forward_kwargs_names if k in runtime_kwargs}

    try:
        # Try with workspace= kwarg first (graceful extension pattern)
        try:
            return bool(method(*args, workspace=ws, **forwarded))
        except TypeError as te:
            msg = str(te)
            if "workspace" in msg or "unexpected keyword" in msg:
                return bool(method(*args, **forwarded))
            raise
    except Exception:
        return False
