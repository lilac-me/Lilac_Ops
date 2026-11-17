# Tests for engine/state_machine.py — YAML-driven State Machine Core
# =====================================================================

import json
import pytest
from pathlib import Path
from engine.state_machine import (
    load_state_machine,
    get_state_spec,
    get_initial_state,
    get_current_state,
    next_state,
    read_log,
    append_log,
    iter_counts_from_log,
    verify_log,
    is_terminal,
    is_pause,
    TERMINAL_STATES,
    TRANSITIONS_FILENAME,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_yaml_path(tmp_path):
    """Create a minimal pipeline YAML for testing."""
    yaml_content = """pipeline:
  name: test-pipeline
  version: "1.0.0"
  initial_state: architect
  total_spawn_cap: 20

states:
  - id: architect
    agent: architect
    iter_cap: 3
    iter_counter: architect
    exit_transitions:
      - condition:
          handoff_contains: "部署方案"
        goto: developer
        rationale: "方案完成，进入部署"
      - condition:
          always: true
        goto: abort
        rationale: "兜底终止"

  - id: developer
    agent: developer
    iter_cap: 3
    iter_counter: developer
    exit_transitions:
      - condition:
          handoff_contains: "服务已启动"
        goto: tester
        rationale: "服务就绪"
      - condition:
          always: true
        goto: abort
        rationale: "兜底终止"

  - id: tester
    agent: tester
    iter_cap: 6
    iter_counter: tester
    exit_transitions:
      - condition:
          handoff_contains: "优化完成"
        goto: reviewer
      - condition:
          handoff_contains: "需要重启"
        goto: developer
        rationale: "重启服务"
      - condition:
          always: true
        goto: abort

  - id: reviewer
    agent: reviewer
    iter_cap: 5
    iter_counter: reviewer
    exit_transitions:
      - condition:
          all_of:
            - handoff_contains: "仍有优化空间"
            - iter_below_cap: tester
        goto: tester
      - condition:
          handoff_contains: "无需继续优化"
        goto: finalize
      - condition:
          always: true
        goto: abort

  - id: finalize
    agent: null
    iter_cap: 1
    iter_counter: finalize
    exit_transitions:
      - condition:
          always: true
        goto: done

  - id: done
    agent: null

  - id: abort
    agent: null
"""
    path = tmp_path / "test-pipeline.yaml"
    path.write_text(yaml_content, encoding="utf-8")
    return path


@pytest.fixture
def sm(sample_yaml_path):
    return load_state_machine(sample_yaml_path)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestLoadStateMachine:
    def test_load_returns_dict(self, sample_yaml_path):
        sm = load_state_machine(sample_yaml_path)
        assert isinstance(sm, dict)
        assert "pipeline" in sm
        assert "states" in sm

    def test_pipeline_metadata(self, sm):
        p = sm["pipeline"]
        assert p["name"] == "test-pipeline"
        assert p["initial_state"] == "architect"
        assert p["total_spawn_cap"] == 20

    def test_states_list(self, sm):
        states = sm["states"]
        assert len(states) == 7  # architect, developer, tester, reviewer, finalize, done, abort
        ids = {s["id"] for s in states}
        assert "architect" in ids
        assert "done" in ids
        assert "abort" in ids

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_state_machine(tmp_path / "nonexistent.yaml")


class TestGetStateSpec:
    def test_found(self, sm):
        spec = get_state_spec(sm, "architect")
        assert spec is not None
        assert spec["id"] == "architect"
        assert spec["iter_cap"] == 3
        assert len(spec["exit_transitions"]) == 2

    def test_not_found(self, sm):
        assert get_state_spec(sm, "nonexistent") is None


class TestGetInitialState:
    def test_from_pipeline(self, sm):
        assert get_initial_state(sm) == "architect"


# ---------------------------------------------------------------------------
# Transition log
# ---------------------------------------------------------------------------

class TestTransitionLog:
    def test_append_and_read(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        entry = {
            "ts": "2026-08-03T10:00:00Z",
            "from_state": "architect",
            "to_state": "developer",
            "handoff": "→ orchestrator: done — 部署方案完成",
            "rationale": "test",
        }
        append_log(ws, entry)
        log = read_log(ws)
        assert len(log) == 1
        assert log[0]["from_state"] == "architect"
        assert log[0]["to_state"] == "developer"

    def test_empty_log(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        assert read_log(ws) == []

    def test_append_is_additive(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        append_log(ws, {"from_state": "a", "to_state": "b"})
        append_log(ws, {"from_state": "b", "to_state": "c"})
        assert len(read_log(ws)) == 2

    def test_corrupt_line_skipped(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        log_file = ws / TRANSITIONS_FILENAME
        log_file.write_text('{"valid": "json"}\nnot json\n{"also": "valid"}\n', encoding="utf-8")
        log = read_log(ws)
        assert len(log) == 2  # corrupt line skipped


class TestIterCounts:
    def test_basic_count(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        append_log(ws, {"from_state": "init", "to_state": "architect"})
        append_log(ws, {"from_state": "architect", "to_state": "developer"})
        append_log(ws, {"from_state": "developer", "to_state": "tester"})
        append_log(ws, {"from_state": "reviewer", "to_state": "tester"})  # iter 2 for tester

        counts = iter_counts_from_log(ws, sm)
        assert counts.get("architect", 0) == 1
        assert counts.get("developer", 0) == 1
        assert counts.get("tester", 0) == 2

    def test_infra_rollback_skipped(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        append_log(ws, {"from_state": "architect", "to_state": "architect"})
        # Infra rollback — should be skipped
        append_log(ws, {"from_state": "architect", "to_state": "architect", "rollback_kind": "infra"})

        counts = iter_counts_from_log(ws, sm)
        assert counts.get("architect", 0) == 1  # infra entry skipped


# ---------------------------------------------------------------------------
# Current state inference
# ---------------------------------------------------------------------------

class TestGetCurrentState:
    def test_log_tail_wins(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        append_log(ws, {"from_state": "init", "to_state": "architect"})
        append_log(ws, {"from_state": "architect", "to_state": "developer"})
        assert get_current_state(ws, sm) == "developer"

    def test_empty_log_returns_initial(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        assert get_current_state(ws, sm) == "architect"


# ---------------------------------------------------------------------------
# Next state decision
# ---------------------------------------------------------------------------

class TestNextState:
    def test_matches_first_condition(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        result = next_state(ws, "architect", "→ orchestrator: done — 部署方案完成，TP=8", sm=sm)
        assert result["next_state"] == "developer"
        assert "方案完成" in result["rationale"]

    def test_fallback_always(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        result = next_state(ws, "architect", "some unexpected handoff", sm=sm)
        assert result["next_state"] == "abort"  # always: true fallback

    def test_tester_restart(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        result = next_state(ws, "tester", "→ orchestrator: 需要重启服务", sm=sm)
        assert result["next_state"] == "developer"

    def test_reviewer_iteration(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        # tester below cap → can iterate
        result = next_state(ws, "reviewer", "→ orchestrator: 仍有优化空间 — HAS_ACTIONABLE", sm=sm)
        assert result["next_state"] == "tester"

    def test_reviewer_no_actionable(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        result = next_state(ws, "reviewer", "→ orchestrator: 无需继续优化 — NO_MORE_ACTIONABLE", sm=sm)
        assert result["next_state"] == "finalize"

    def test_unknown_state(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        result = next_state(ws, "nonexistent", "anything", sm=sm)
        assert "error" in result
        assert result["next_state"] is None

    def test_matched_transition_index(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        result = next_state(ws, "architect", "→ orchestrator: 部署方案完成", sm=sm)
        assert result["matched_transition_index"] == 0  # first transition matched


# ---------------------------------------------------------------------------
# Terminal state checks
# ---------------------------------------------------------------------------

class TestTerminalAndPause:
    def test_terminal_states(self):
        assert is_terminal("done") is True
        assert is_terminal("abort") is True
        assert is_terminal("architect") is False

    def test_pause_states(self):
        assert is_pause("await_user_decision") is True
        assert is_pause("done") is False


# ---------------------------------------------------------------------------
# Verify log
# ---------------------------------------------------------------------------

class TestVerifyLog:
    def test_clean_log(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        # Valid transition: architect → developer (declared in exit_transitions)
        append_log(ws, {
            "from_state": "architect",
            "to_state": "developer",
            "handoff": "→ orchestrator: done — 部署方案完成",
            "matched_transition_index": 0,
        })
        violations = verify_log(ws, sm)
        assert len(violations) == 0

    def test_undeclared_transition(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        # Invalid: architect → tester (not declared)
        append_log(ws, {"from_state": "architect", "to_state": "tester"})
        violations = verify_log(ws, sm)
        assert len(violations) > 0
        assert any("not declared" in v for v in violations)

    def test_handoff_mismatch_detected(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        append_log(ws, {
            "from_state": "architect",
            "to_state": "developer",
            "handoff": "handoff: ARCHITECT_FAILED",
            "matched_transition_index": 0,
        })
        violations = verify_log(ws, sm)
        assert any("handoff selects 'abort'" in v for v in violations)

    def test_iter_cap_exceeded(self, tmp_path, sm):
        ws = tmp_path / "workspace"
        ws.mkdir()
        # architect iter_cap = 3, push 4 entries
        for _ in range(4):
            append_log(ws, {"from_state": "architect", "to_state": "developer"})
        violations = verify_log(ws, sm)
        assert any("iter cap exceeded" in v for v in violations)
