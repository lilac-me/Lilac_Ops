"""Minimal state machine engine for ops pipeline.

Enforces iteration caps strictly — optimize stage stops at cap or convergence.
API-compatible with MindSpeed-Agent engine/state_machine.py.

Usage:
    python -m plugins.ops.engine.state_machine next \
      --yaml plugins/ops/ops.yaml \
      --workspace outputs/<op> \
      --handoff "→ handoff: DESIGN_DONE"
"""

from __future__ import annotations

import json, sys, re
from pathlib import Path
from datetime import datetime
from typing import Optional


class StateMachine:
    """YAML-driven state tracker with hard iteration cap enforcement."""

    def __init__(self, yaml_path: str, workspace: str):
        self.yaml_path = Path(yaml_path)
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.log_path = self.workspace / "state_transitions.jsonl"
        self.definition = self._load_yaml()
        self.pipeline = self.definition.get("pipeline", [])
        self.caps = self.definition.get("iteration_caps", {})
        self.sm_cfg = self.definition.get("state_machine", {})
        self.handoff_pattern = self.sm_cfg.get(
            "handoff_pattern", r"→ handoff: (\S+)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_state(self) -> Optional[str]:
        if not self.log_path.exists():
            return None
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        if not lines:
            return None
        return json.loads(lines[-1]).get("to")

    def iteration_count(self, stage: str) -> int:
        if not self.log_path.exists():
            return 0
        count = 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if json.loads(line).get("to") == stage:
                    count += 1
        return count

    def next_state(self, handoff: str) -> dict:
        """Compute next state. Hard enforces iteration caps, convergence, and preconditions.

        Returns: {"stage": str, "reason": str, "cap_reached": bool, "force_stop": bool}
        """
        token = self._extract_token(handoff)
        current = self.current_state()

        # Find current in pipeline
        idx = -1
        for i, s in enumerate(self.pipeline):
            if s["stage"] == current:
                idx = i
                break

        if idx == -1:
            s = self.pipeline[0]["stage"]
            return self._check_preconditions(s, {"stage": s, "reason": "start",
                                                  "cap_reached": False, "force_stop": False})

        # ---- HARD CAP ----
        cap = self._resolve_cap(current)
        count = self.iteration_count(current)

        next_stage = current  # default: stay

        if count >= cap:
            if current == "optimize":
                return {"stage": "done", "reason": f"cap_reached ({count}/{cap})",
                        "cap_reached": True, "force_stop": True}
            elif idx + 1 < len(self.pipeline):
                next_stage = self.pipeline[idx + 1]["stage"]
            else:
                return {"stage": "done", "reason": "pipeline_complete",
                        "cap_reached": True, "force_stop": True}

        # Optimize convergence
        if current == "optimize" and token and token.startswith("OPTIMIZE_DONE"):
            return {"stage": "done", "reason": f"convergence_handoff: {token}",
                    "cap_reached": False, "force_stop": True}

        result = {"stage": next_stage, "reason": token or "continue",
                  "cap_reached": count >= cap, "force_stop": False}

        # ---- PRECONDITION CHECK ----
        return self._check_preconditions(next_stage, result)

    def _check_preconditions(self, stage: str, default_result: dict) -> dict:
        """Verify that required upstream files exist before entering a stage."""
        for s in self.pipeline:
            if s["stage"] != stage:
                continue
            pre = s.get("precondition", [])
            for check in pre:
                fname = check.get("file", "")
                src = check.get("source", "")
                fpath = self.workspace / fname
                if not fpath.exists():
                    return {
                        "stage": src,  # 回退到上游阶段
                        "reason": f"precondition_failed: {fname} not found (需要上游 {src} 阶段产出)",
                        "cap_reached": False,
                        "force_stop": True,
                        "missing_file": str(fpath),
                    }
            break
        return default_result

    def verify_gate(self, stage: str) -> dict:
        """Verify that a stage's exit gate files exist. Returns {passed: bool, missing: [...]}."""
        for s in self.pipeline:
            if s["stage"] != stage:
                continue
            gates = s.get("gate", [])
            missing = []
            for g in gates:
                # Non-file gates (e.g. anti_cheat) are semantic checks evaluated
                # by the implement agent, not file-presence checks — skip them
                # in this verifier so they don't raise KeyError on "file".
                if "file" not in g:
                    continue
                fpath = self.workspace / g["file"]
                if not fpath.exists():
                    missing.append(str(fpath))
                elif "min_bytes" in g and fpath.stat().st_size < g["min_bytes"]:
                    missing.append(f"{g['file']} (too small: {fpath.stat().st_size} < {g['min_bytes']} bytes)")
            return {"passed": len(missing) == 0, "missing": missing, "stage": stage}
        return {"passed": True, "missing": [], "stage": stage}

    def record_transition(self, from_state: str, to_state: str,
                          handoff: str, reason: str = "") -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "from": from_state or "start",
            "to": to_state,
            "handoff": handoff,
            "reason": reason,
            "iteration": {s["stage"]: self.iteration_count(s["stage"])
                          for s in self.pipeline},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def verify_log(self) -> list[str]:
        errors = []
        if not self.log_path.exists():
            return ["No log file"]
        stages = [s["stage"] for s in self.pipeline] + ["start", "done"]
        with open(self.log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("to") not in stages:
                    errors.append(f"Line {i}: unknown state '{entry.get('to')}'")
        return errors

    def state_summary(self) -> dict:
        """Return a readable state summary for agent context."""
        current = self.current_state() or "start"
        return {
            "current_stage": current,
            "iteration_counts": {s["stage"]: self.iteration_count(s["stage"])
                                 for s in self.pipeline},
            "caps": self.caps,
            "completed": current == "done",
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_yaml(self) -> dict:
        import yaml
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_cap(self, stage: str) -> int:
        """Resolve iteration cap for a stage, supporting per-operator-type caps."""
        caps = self.caps
        if stage in caps and isinstance(caps[stage], dict):
            # Per-type caps: {"VV": 3, "CV": 5, "default": 3}
            return caps[stage].get("default", 3)
        return caps.get(stage, 1) if isinstance(caps, dict) else 1

    def _extract_token(self, text: str) -> Optional[str]:
        m = re.search(self.handoff_pattern, text)
        return m.group(1) if m else None


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Ops State Machine Engine")
    sub = p.add_subparsers(dest="cmd")

    n = sub.add_parser("next")
    n.add_argument("--yaml", required=True)
    n.add_argument("--workspace", required=True)
    n.add_argument("--handoff", required=True)

    c = sub.add_parser("current")
    c.add_argument("--workspace", required=True)
    c.add_argument("--yaml", default="plugins/ops/ops.yaml")

    s = sub.add_parser("summary")
    s.add_argument("--yaml", required=True)
    s.add_argument("--workspace", required=True)

    v = sub.add_parser("verify")
    v.add_argument("--yaml", required=True)
    v.add_argument("--workspace", required=True)

    g = sub.add_parser("gate")
    g.add_argument("--yaml", required=True)
    g.add_argument("--workspace", required=True)
    g.add_argument("--stage", required=True)

    args = p.parse_args()

    if args.cmd == "gate":
        sm = StateMachine(args.yaml, args.workspace)
        result = sm.verify_gate(args.stage)
        print(json.dumps(result, ensure_ascii=False))
        if not result["passed"]:
            print(f"  ⛔ GATE FAILED for {args.stage}: {result['missing']}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "next":
        sm = StateMachine(args.yaml, args.workspace)
        cur = sm.current_state()

        # ---- EXIT GATE: verify current stage's outputs before accepting handoff ----
        if cur and cur != "start":
            gate = sm.verify_gate(cur)
            if not gate["passed"]:
                result = {"stage": cur, "reason": f"gate_failed: {gate['missing']}",
                          "cap_reached": False, "force_stop": True}
                print(json.dumps(result, ensure_ascii=False))
                print(f"  ⛔ GATE FAILED ({cur}): 产出缺失 {gate['missing']} — handoff 被拒绝", file=sys.stderr)
                sys.exit(1)

        result = sm.next_state(args.handoff)
        sm.record_transition(cur, result["stage"], args.handoff, result["reason"])
        print(json.dumps(result, ensure_ascii=False))
        if result.get("force_stop"):
            print(f"  ⛔ FORCE STOP: {result['reason']}", file=sys.stderr)

    elif args.cmd == "current":
        sm = StateMachine(args.yaml, args.workspace)
        s = sm.current_state()
        print(json.dumps({"state": s or "start"}, ensure_ascii=False))

    elif args.cmd == "summary":
        sm = StateMachine(args.yaml, args.workspace)
        print(json.dumps(sm.state_summary(), ensure_ascii=False, indent=2))

    elif args.cmd == "verify":
        sm = StateMachine(args.yaml, args.workspace)
        errors = sm.verify_log()
        if errors:
            for e in errors:
                print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print("OK: transition log is consistent")


if __name__ == "__main__":
    main()
