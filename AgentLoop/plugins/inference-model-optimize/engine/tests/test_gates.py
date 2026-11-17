# Tests for engine/gates.py — Code-level gate validation (Python engine path)
# =============================================================================
# Rules must stay in sync with workflows/gates.js.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engine.gates import (
    GATE_MAP,
    validate_architect,
    validate_developer,
    validate_optimizer,
    validate_reviewer,
    validate_reviewer_composite,
    validate_state,
    validate_tester,
)

_ENGINE_DIR = Path(__file__).resolve().parent.parent  # engine/


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_tester_result():
    return {
        "baseline_metrics": {"output_tok_per_s": 100, "ttft_ms": 50, "tpot_ms": 10},
        "optimized_metrics": {"output_tok_per_s": 120, "ttft_ms": 45, "tpot_ms": 10},
        "comparison": {"output_tok_per_s_change_pct": 20},
        "comparison_report": "对比报告",
        "workload_description": "input 4096, output 32768, num-prompts=128, max-concurrency=128, temperature=0",
        "optimal_params": {"tp": 4, "dp": 2, "ep": 1},
        "topology_search": {
            "search_completed": True,
            "candidates": [
                {"tp": 8, "dp": 1, "ep": 1, "output_tok_per_s": 100, "notes": "基线"},
                {"tp": 4, "dp": 2, "ep": 1, "output_tok_per_s": 120, "notes": "更优"},
            ],
            "best_topology": {"tp": 4, "dp": 2, "ep": 1, "rationale": "吞吐更高"},
        },
    }


# ---------------------------------------------------------------------------
# Tester — topology search enforcement (the core of the fix)
# ---------------------------------------------------------------------------

class TestTesterTopologyGate:
    def test_missing_topology_search_rejected(self):
        r = _base_tester_result()
        del r["topology_search"]
        gate = validate_tester(r)
        assert not gate.passed
        assert any("topology_search" in f for f in gate.failures)

    def test_incomplete_search_rejected(self):
        r = _base_tester_result()
        r["topology_search"]["search_completed"] = False
        gate = validate_tester(r)
        assert not gate.passed
        assert any("search_completed" in f for f in gate.failures)

    def test_empty_candidates_rejected(self):
        r = _base_tester_result()
        r["topology_search"]["candidates"] = []
        gate = validate_tester(r)
        assert not gate.passed
        assert any("candidates" in f for f in gate.failures)

    def test_candidate_missing_ep_rejected(self):
        r = _base_tester_result()
        del r["topology_search"]["candidates"][0]["ep"]
        gate = validate_tester(r)
        assert not gate.passed
        assert any("candidates[0]" in f for f in gate.failures)

    def test_missing_best_topology_rejected(self):
        r = _base_tester_result()
        del r["topology_search"]["best_topology"]
        gate = validate_tester(r)
        assert not gate.passed
        assert any("best_topology" in f for f in gate.failures)

    def test_optimal_params_missing_ep_rejected(self):
        r = _base_tester_result()
        del r["optimal_params"]["ep"]
        gate = validate_tester(r)
        assert not gate.passed
        assert any("optimal_params 缺少 ep" in f for f in gate.failures)

    def test_complete_result_passes(self):
        gate = validate_tester(_base_tester_result())
        assert gate.passed, gate.failures


# ---------------------------------------------------------------------------
# Developer
# ---------------------------------------------------------------------------

class TestDeveloperGate:
    def _base(self):
        return {
            "service_url": "127.0.0.1",
            "port": 8000,
            "model_name": "m",
            "tp": 4,
            "dp": 2,
            "ep": 1,
            "health_check_passed": True,
            "models_endpoint_ok": True,
            "service_status": "running",
        }

    def test_passes(self):
        assert validate_developer(self._base()).passed

    def test_missing_ep_rejected(self):
        r = self._base()
        del r["ep"]
        gate = validate_developer(r)
        assert not gate.passed
        assert any("ep" in f for f in gate.failures)

    def test_service_not_running_rejected(self):
        r = self._base()
        r["service_status"] = "starting"
        gate = validate_developer(r)
        assert not gate.passed
        assert any("running" in f for f in gate.failures)


# ---------------------------------------------------------------------------
# Architect / Reviewer smoke coverage
# ---------------------------------------------------------------------------

class TestArchitectGate:
    def _base(self):
        return {
            "model_profile": {
                "architecture": "MoE",
                "num_layers": 40,
                "hidden_size": 5120,
                "num_parameters": "30B",
            },
            "memory_budget": {
                "model_weights_gb": 60,
                "kv_cache_gb": 10,
                "total_per_card_gb": 75,
            },
            "recommended_strategy": {"tp": 4, "dp": 2, "rationale": "x"},
            "hardware_requirements": {"min_npu_cards": 8, "recommended_config": "8x910B"},
        }

    def test_passes(self):
        assert validate_architect(self._base()).passed

    def test_missing_strategy_rejected(self):
        r = self._base()
        del r["recommended_strategy"]
        gate = validate_architect(r)
        assert not gate.passed
        assert any("recommended_strategy" in f for f in gate.failures)


class TestReviewerGate:
    def _base(self):
        return {
            "profiling_data_path": "/tmp/prof",
            "profiling_collected": True,
            "workload_description": "input 4096, output 32768, num-prompts=128, max-concurrency=128, temperature=0",
            "profiling_round": 1,
            "diagnosis": {"e2e_time_ms": 120, "compute_pct": 60, "communication_pct": 20, "idle_pct": 20},
            "bottleneck_classification": {"primary": "compute", "details": "x"},
            "operator_top_n": [{"rank": 1, "op_name": "mm", "time_ms": 30, "percentage": 40}],
            "optimization_suggestions": [{"id": "s1", "description": "x", "priority": "high", "category": "参数", "executable_by": "tester"}],
            "diagnosis_report": "x",
        }

    def test_missing_evidence_rejected(self):
        gate = validate_reviewer_composite(self._base())
        assert not gate.passed
        assert any("evidence_list" in f for f in gate.failures)

    def test_incomplete_evidence_rejected(self):
        r = self._base()
        r["evidence_list"] = [{"conclusion_id": "c1", "data_path": "/p", "raw_value": "1"}]
        gate = validate_reviewer_composite(r)
        assert not gate.passed
        assert any("verification_status" in f for f in gate.failures)

    def test_full_result_passes(self):
        r = self._base()
        r["evidence_list"] = [
            {"conclusion_id": "c1", "data_path": "/p", "raw_value": "1", "verification_status": "verified"}
        ]
        assert validate_reviewer_composite(r).passed

    def test_missing_workload_description_rejected(self):
        r = self._base()
        del r["workload_description"]
        gate = validate_reviewer(r)
        assert not gate.passed
        assert any("workload_description" in f for f in gate.failures)

    def test_workload_mismatch_rejected(self):
        r = self._base()
        gate = validate_reviewer(r, tester_workload="input 2048, output 16384, num-prompts=64")
        assert not gate.passed
        assert any("不一致" in f for f in gate.failures)

    def test_resample_round_without_resampled_rejected(self):
        r = self._base()
        r["profiling_round"] = 2
        gate = validate_reviewer(r)
        assert not gate.passed
        assert any("resampled=true" in f for f in gate.failures)

    def test_resample_round_with_comparison_passes(self):
        r = self._base()
        r["profiling_round"] = 2
        r["resampled"] = True
        r["previous_round_comparison"] = {
            "compute_pct_change": -5,
            "communication_pct_change": 0,
            "idle_pct_change": 5,
            "operator_top_n_changes": "mm 下降",
            "suggestion_verification": "生效",
        }
        assert validate_reviewer(r).passed

    def test_profiling_round_string_is_normalized(self):
        r = self._base()
        r["profiling_round"] = "2"
        gate = validate_reviewer(r)
        assert not gate.passed
        assert any("resampled=true" in f for f in gate.failures)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class TestOptimizerGate:
    def _base(self):
        return {
            "solutions": [
                {
                    "suggestion_id": "s1",
                    "approach": "启用 AOE 调优并生成配置",
                    "code_changes": "aoe_config.json",
                    "test_result": "有效：吞吐 +8%",
                    "evidence": "/tmp/logs/bench-after.log",
                },
            ],
            "optimization_metrics": {"before_output_tok_per_s": 100, "after_output_tok_per_s": 108, "change_pct": 8},
        }

    def test_passes(self):
        assert validate_optimizer(self._base()).passed

    def test_empty_solutions_rejected(self):
        r = self._base()
        r["solutions"] = []
        gate = validate_optimizer(r)
        assert not gate.passed
        assert any("solutions" in f for f in gate.failures)

    def test_missing_approach_rejected(self):
        r = self._base()
        del r["solutions"][0]["approach"]
        gate = validate_optimizer(r)
        assert not gate.passed
        assert any("approach" in f for f in gate.failures)


# ---------------------------------------------------------------------------
# validate_state dispatch
# ---------------------------------------------------------------------------

class TestValidateState:
    def test_unknown_state(self):
        gate = validate_state("nonexistent", {})
        assert not gate.passed
        assert any("未知状态" in f for f in gate.failures)

    def test_map_covers_all_states(self):
        assert set(GATE_MAP) == {"architect", "developer", "tester", "reviewer", "optimizer"}

    def test_dispatch_to_tester(self):
        gate = validate_state("tester", _base_tester_result())
        assert gate.passed

    def test_dispatch_to_optimizer(self):
        gate = validate_state(
            "optimizer",
            {"solutions": [{"suggestion_id": "s1", "approach": "x", "test_result": "有效"}]},
        )
        assert gate.passed

    def test_dispatch_reviewer_honors_tester_workload(self):
        r = {
            "profiling_data_path": "/tmp/prof",
            "profiling_collected": True,
            "workload_description": "A",
            "profiling_round": 1,
            "diagnosis": {"e2e_time_ms": 120, "compute_pct": 60, "communication_pct": 20, "idle_pct": 20},
            "bottleneck_classification": {"primary": "compute", "details": "x"},
            "operator_top_n": [{"rank": 1, "op_name": "mm", "time_ms": 30, "percentage": 40}],
            "optimization_suggestions": [{"id": "s1", "description": "x"}],
            "diagnosis_report": "x",
            "evidence_list": [{"conclusion_id": "c1", "data_path": "/p", "raw_value": "1", "verification_status": "verified"}],
        }
        gate = validate_state("reviewer", r, tester_workload="B")
        assert not gate.passed
        assert any("不一致" in f for f in gate.failures)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def _run_cli(self, state, payload):
        env = dict(os.environ)
        pythonpath = str(_ENGINE_DIR.parent)  # plugins/inference-model-optimize/
        env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return subprocess.run(
            [sys.executable, "-m", "engine.gates", "validate", state, "--stdin"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",  # Anaconda 注入 PYTHONIOENCODING=UTF-8，子进程输出 UTF-8
            env=env,
            cwd=str(_ENGINE_DIR.parent),
        )

    def test_cli_passes(self):
        proc = self._run_cli("tester", _base_tester_result())
        assert proc.returncode == 0, proc.stderr

    def test_cli_fails_on_missing_topology(self):
        bad = _base_tester_result()
        del bad["topology_search"]
        proc = self._run_cli("tester", bad)
        assert proc.returncode == 2
        assert "topology_search" in proc.stdout

    def test_cli_usage_error_without_input(self):
        env = dict(os.environ)
        pythonpath = str(_ENGINE_DIR.parent)
        env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.run(
            [sys.executable, "-m", "engine.gates", "validate", "tester"],
            input="",
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=str(_ENGINE_DIR.parent),
        )
        assert proc.returncode == 1
        assert "必须提供" in proc.stderr
