from __future__ import annotations

from pathlib import Path

import pytest

from engine.gates import load_json, write_auto_planned_run_config
from engine.recovery import record_attempt, recovery_key, recovery_status, write_exhausted_report
from engine.tp_planner import plan_tp

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
RUN_SCHEMA = PLUGIN_ROOT / "contracts" / "run_config.schema.json"


def test_tp_plan_uses_visible_cards_and_model_divisibility():
    plan = plan_tp(list(range(8)), {"num_attention_heads": 24, "num_key_value_heads": 8})
    assert plan["candidate_tps"] == [1, 2, 4, 8]
    assert plan["compatible_tps"] == [1, 2, 4, 8]
    assert plan["selected_plan"]["phase1"] == 8
    assert plan["selected_plan"]["phase2_matrix"] == [1, 2, 4]
    assert plan["selected_plan"]["phase2_diagnosis"] == [4, 2]


def test_tp_plan_filters_model_incompatible_and_unrunnable_values():
    plan = plan_tp(list(range(8)), {"num_attention_heads": 12, "num_key_value_heads": 4},
                   runnable_tps=[2, 4, 8])
    assert plan["compatible_tps"] == [2, 4]
    assert plan["selected_plan"]["phase1"] == 4


def test_explicit_tp_and_probe_status():
    model = {"num_attention_heads": 32, "num_key_value_heads": 8}
    plan = plan_tp(list(range(8)), model, requested_tps=[4])
    assert plan["selected_plan"]["phase1"] == 4
    assert plan["load_probe_status"] == "not_probed"
    with pytest.raises(ValueError, match="explicit TP"):
        plan_tp(list(range(4)), model, requested_tps=[8])
    with pytest.raises(ValueError, match="metadata"):
        plan_tp(list(range(8)), {})


def test_issue_identity_survives_config_revision():
    assert recovery_key("s", "r", "failure", "old") == recovery_key("s", "r", "failure", "new")


def _attempt(key: str, round_number: int, *, action: str, before: str = "a" * 64) -> dict:
    return {
        "recovery_key": key, "round": round_number, "stage": "environment",
        "state": "environment_admission", "reason_code": "BLOCKED_OPERATOR_INSTALLATION",
        "observations": [f"observation-{round_number}"],
        "root_cause_hypotheses": [f"hypothesis-{round_number}"],
        "selected_hypothesis": f"hypothesis-{round_number}",
        "planned_actions": [action], "commands": [["tool", action]],
        "verification": ["full environment preflight"],
        "reflection": f"reflection-{round_number}", "progress_assessment": "PROGRESSED",
        "repair_result": "BLOCKED",
        "before_evidence_sha256": before, "after_evidence_sha256": f"{round_number}" * 64,
    }


def test_recovery_rejects_mechanical_retry_and_stops_after_five(tmp_path):
    key = recovery_key("environment_admission", "BLOCKED_OPERATOR_INSTALLATION",
                       "missing-so", "c" * 64)
    record_attempt(tmp_path, _attempt(key, 1, action="build"))
    duplicate = _attempt(key, 2, action="build")
    duplicate["observations"] = ["same failure"]
    with pytest.raises(ValueError, match="mechanical retry"):
        record_attempt(tmp_path, duplicate)
    for round_number in range(2, 6):
        record_attempt(tmp_path, _attempt(key, round_number, action=f"repair-{round_number}"))
    assert recovery_status(tmp_path, key, "environment")["exhausted"] is True
    report = write_exhausted_report(tmp_path, key)
    assert "第 5 轮" in report.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="issue limit reached"):
        record_attempt(tmp_path, _attempt(key, 6, action="sixth"))


def test_auto_planned_config_is_schema_valid(tmp_path):
    candidate = {
        "schema_version": "1.0", "run_id": "auto-1",
        "model": {"path": "/models/qwen", "architecture": "Qwen"}, "chip": "A5",
        "device_topology": {"visible_devices": list(range(8)), "device_count": 1,
                            "parallelism": {"tp": 1}},
        "tp_plan": {"phase1": 1, "phase2_long_output": 1, "phase2_long_input": 1,
                    "phase2_matrix": [1], "phase3": 1},
        "tp_confirmation": {"status": "confirmed", "source": "default_confirmed",
                            "confirmed_utc": "2026-01-01T00:00:00Z", "user_response": "legacy"},
        "sampling": {"phase2_temperature": 0.6, "phase3_temperature": 0, "seed": 1234},
        "service": {"generation_config": "vllm", "enable_prefix_caching": False,
                    "enable_chunked_prefill": False},
        "scheduling": {"max_model_len": 22528, "max_num_batched_tokens": 22528,
                       "max_num_seqs": 64, "gpu_memory_utilization": 0.9,
                       "async_scheduling": False},
        "mutation_policy": "repair_if_unambiguous", "tp_change_history": [],
    }
    output = tmp_path / "run_config.json"
    write_auto_planned_run_config(candidate, output, RUN_SCHEMA,
                                  {"num_attention_heads": 32, "num_key_value_heads": 8})
    config = load_json(output)
    assert config["tp_confirmation"]["source"] == "auto_planned"
    assert config["tp_plan"]["phase1"] == 8
    assert config["execution_policy"]["max_issue_recovery_rounds"] == 5
    original = output.read_bytes()
    write_auto_planned_run_config(candidate, output, RUN_SCHEMA,
                                  {"num_attention_heads": 32, "num_key_value_heads": 8})
    assert output.read_bytes() == original
    write_auto_planned_run_config(candidate, output, RUN_SCHEMA,
                                  {"num_attention_heads": 32, "num_key_value_heads": 8}, requested_tps=[4])
    revised = load_json(output)
    assert revised["tp_change_history"][-1]["previous_plan"] == config["tp_plan"]
    assert revised["tp_change_history"][-1]["new_plan"] == revised["tp_plan"]
