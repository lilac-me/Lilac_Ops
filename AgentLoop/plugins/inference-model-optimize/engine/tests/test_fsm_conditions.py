# Tests for engine/fsm_conditions.py — Condition DSL Evaluator
# ===============================================================

import pytest
from pathlib import Path
from engine.fsm_conditions import (
    eval_condition,
    extract_handoff,
    parse_worker_signal,
)


class TestExtractHandoff:
    def test_arrow_orchestrator(self):
        text = "some output\n→ orchestrator: done — precision PASS\nmore output"
        assert "→ orchestrator: done — precision PASS" in extract_handoff(text)

    def test_at_agent_prefix(self):
        text = "result\n@aog-kernel-optimizer: spawn complete\n"
        h = extract_handoff(text)
        assert "@aog-kernel-optimizer: spawn complete" in h

    def test_bold_stripping(self):
        text = "→ orchestrator: **done**, precision **PASS**"
        h = extract_handoff(text)
        assert "**" not in h
        assert "done" in h

    def test_empty_text(self):
        assert extract_handoff("") == ""

    def test_no_handoff_line(self):
        text = "just some random output\nwithout any handoff markers"
        h = extract_handoff(text)
        assert len(h) > 0  # Falls back to last line


class TestParseWorkerSignal:
    def test_done(self):
        assert parse_worker_signal("→ orchestrator: done") == "done"

    def test_done_chinese(self):
        assert parse_worker_signal("→ orchestrator: 优化完成，部署方案") == "done"

    def test_restart_needed(self):
        assert parse_worker_signal("→ orchestrator: 需要重启，candidate_restart_params") == "restart_needed"

    def test_abort(self):
        assert parse_worker_signal("→ orchestrator: abort — build failed") == "abort"

    def test_unknown(self):
        assert parse_worker_signal("some random text") == "unknown"


class TestEvalCondition:
    def _ctx(self, **overrides):
        ctx = {
            "handoff": "",
            "snapshot": {},
            "iter_counts": {},
            "ws": None,
            "sm": {},
            "plugin": None,
        }
        ctx.update(overrides)
        return ctx

    # --- always ---
    def test_always_true(self):
        assert eval_condition({"always": True}, self._ctx()) is True

    def test_always_false(self):
        assert eval_condition({"always": False}, self._ctx()) is False

    # --- handoff_match ---
    def test_handoff_match(self):
        ctx = self._ctx(handoff="→ orchestrator: done — PASS")
        assert eval_condition({"handoff_match": "→ orchestrator: done"}, ctx) is True

    def test_handoff_match_no_match(self):
        ctx = self._ctx(handoff="→ orchestrator: failed")
        assert eval_condition({"handoff_match": "→ orchestrator: done"}, ctx) is False

    # --- handoff_contains ---
    def test_handoff_contains(self):
        ctx = self._ctx(handoff="→ orchestrator: done — KO_PERF_PLATEAU")
        assert eval_condition({"handoff_contains": "KO_PERF_PLATEAU"}, ctx) is True

    def test_handoff_contains_bold(self):
        ctx = self._ctx(handoff="→ orchestrator: **done**, precision PASS")
        # ** is stripped before matching
        assert eval_condition({"handoff_contains": "done"}, ctx) is True

    def test_handoff_contains_no_match(self):
        ctx = self._ctx(handoff="→ orchestrator: failed")
        assert eval_condition({"handoff_contains": "PASSED"}, ctx) is False

    # --- iter_below_cap ---
    def test_iter_below_cap_under(self):
        sm = {
            "states": [
                {"id": "architect", "iter_counter": "architect", "iter_cap": 3},
            ]
        }
        ctx = self._ctx(sm=sm, iter_counts={"architect": 1})
        assert eval_condition({"iter_below_cap": "architect"}, ctx) is True

    def test_iter_below_cap_at_cap(self):
        sm = {
            "states": [
                {"id": "architect", "iter_counter": "architect", "iter_cap": 3},
            ]
        }
        ctx = self._ctx(sm=sm, iter_counts={"architect": 3})
        assert eval_condition({"iter_below_cap": "architect"}, ctx) is False

    def test_iter_below_cap_over_cap(self):
        sm = {
            "states": [
                {"id": "architect", "iter_counter": "architect", "iter_cap": 3},
            ]
        }
        ctx = self._ctx(sm=sm, iter_counts={"architect": 5})
        assert eval_condition({"iter_below_cap": "architect"}, ctx) is False

    def test_iter_below_cap_unknown_counter(self):
        ctx = self._ctx(sm={"states": []})
        # Unknown cap → permissive (returns True)
        assert eval_condition({"iter_below_cap": "nonexistent"}, ctx) is True

    # --- path_exists ---
    def test_path_exists_true(self, tmp_path):
        (tmp_path / "test_file.txt").write_text("hello")
        ctx = self._ctx(ws=tmp_path)
        assert eval_condition({"path_exists": "test_file.txt"}, ctx) is True

    def test_path_exists_false(self, tmp_path):
        ctx = self._ctx(ws=tmp_path)
        assert eval_condition({"path_exists": "nonexistent.txt"}, ctx) is False

    def test_path_exists_no_workspace(self):
        ctx = self._ctx(ws=None)
        assert eval_condition({"path_exists": "anything.txt"}, ctx) is False

    # --- all_of ---
    def test_all_of_all_true(self):
        ctx = self._ctx(handoff="→ orchestrator: done — PASS")
        cond = {
            "all_of": [
                {"handoff_contains": "done"},
                {"handoff_contains": "PASS"},
            ]
        }
        assert eval_condition(cond, ctx) is True

    def test_all_of_one_false(self):
        ctx = self._ctx(handoff="→ orchestrator: done — FAIL")
        cond = {
            "all_of": [
                {"handoff_contains": "done"},
                {"handoff_contains": "PASS"},
            ]
        }
        assert eval_condition(cond, ctx) is False

    # --- any_of ---
    def test_any_of_one_true(self):
        ctx = self._ctx(handoff="→ orchestrator: done — FAIL")
        cond = {
            "any_of": [
                {"handoff_contains": "PASS"},
                {"handoff_contains": "done"},
            ]
        }
        assert eval_condition(cond, ctx) is True

    def test_any_of_all_false(self):
        ctx = self._ctx(handoff="→ orchestrator: unknown")
        cond = {
            "any_of": [
                {"handoff_contains": "PASS"},
                {"handoff_contains": "done"},
            ]
        }
        assert eval_condition(cond, ctx) is False

    # --- boolean passthrough ---
    def test_boolean_true(self):
        assert eval_condition(True, self._ctx()) is True

    def test_boolean_false(self):
        assert eval_condition(False, self._ctx()) is False

    # --- unknown primitive (fail closed) ---
    def test_unknown_primitive(self):
        assert eval_condition({"unknown_op": "value"}, self._ctx()) is False

    # --- user_decision_target_in ---
    def test_user_decision_target_in(self):
        snap = {"user_decision_target": "developer"}
        ctx = self._ctx(snapshot=snap)
        assert eval_condition(
            {"user_decision_target_in": ["developer", "tester"]}, ctx
        ) is True

    def test_user_decision_target_not_in(self):
        snap = {"user_decision_target": "architect"}
        ctx = self._ctx(snapshot=snap)
        assert eval_condition(
            {"user_decision_target_in": ["developer", "tester"]}, ctx
        ) is False
