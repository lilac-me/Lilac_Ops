"""Unit tests for the ops benchmark scoring engine.

These lock down the *heuristics* (evidence-source preference, staleness guard,
required-shape coverage gate, anti-cheat, additive convergence) so a later tweak
cannot silently change published scores. Run from the repo root::

    python -m pytest benchmark/agents/ops/tests -q
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.agents.ops import metrics, scoring  # noqa: E402

# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

GOOD_KERNEL = """import triton
import triton.language as tl

@triton.jit
def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + o, tl.load(x_ptr + o, mask=o < n), mask=o < n)
"""

CHEAT_KERNEL = """import torch
import torch.nn.functional as F

def relu(x):
    return F.relu(x)
"""


def make_ws(tmp: pathlib.Path, *, passed=19, failed=0, kernel=GOOD_KERNEL,
            bench=None, speedup_md=None, sections=12) -> pathlib.Path:
    ws = tmp / "run"
    ws.mkdir(parents=True, exist_ok=True)
    # every mandatory deliverable is padded past its min-bytes gate so the
    # "good" workspace really is a full-marks workspace
    (ws / "test_result.md").write_text(
        f"# result\n\n{passed} passed, {failed} failed\nbit-exact\n" + "x" * 250,
        encoding="utf-8")
    (ws / "kernel.py").write_text(kernel, encoding="utf-8")
    (ws / "test.py").write_text("import pytest\n", encoding="utf-8")
    (ws / "design.md").write_text(
        "# design\n" + "".join(f"## {i}\n\nx\n" for i in range(sections)) + "x" * 500,
        encoding="utf-8")
    (ws / "bottleneck.md").write_text(
        "memory-bound bottleneck\n1.29 TB/s bandwidth\n"
        "P0 P1 P2 P3 grading\n" + "x" * 200,
        encoding="utf-8")
    # machine-readable convergence contract (D4 structured path)
    (ws / "bottleneck.json").write_text(json.dumps({
        "classification": "memory_bound",
        "bandwidth": {"measured": 1.29, "unit": "TB/s", "utilization_pct": 81},
        "suggestions": [{"id": "S1", "priority": "P1", "actionable": True}],
        "convergence": {"reason": "TARGET_MET", "judgement": "fp16 beats torch"},
    }), encoding="utf-8")
    (ws / "execution_log.md").write_text("x" * 200, encoding="utf-8")
    (ws / "delivery.md").write_text("x" * 250, encoding="utf-8")
    (ws / "env_snapshot.md").write_text("x" * 50, encoding="utf-8")
    (ws / "speedup.md").write_text(
        (speedup_md if speedup_md is not None else
         "| [4096, 4096] | fp16 | 0.02 | 0.02 | **1.10×** |\n") + "x" * 250,
        encoding="utf-8")
    if bench is not None:
        (ws / "bench_result.json").write_text(
            json.dumps(bench), encoding="utf-8")
    return ws


def bench_row(shape, speedup, dtype="torch.float16"):
    return {"shape": list(shape), "dtype": dtype, "speedup": speedup,
            "triton_us": 1.0, "torch_us": speedup}


# ---------------------------------------------------------------------------
# completion
# ---------------------------------------------------------------------------

def test_completion_all_pass(tmp_path):
    ws = make_ws(tmp_path)
    assert scoring.score_completion(metrics.extract(ws))["score"] == 100.0


def test_completion_partial(tmp_path):
    ws = make_ws(tmp_path, passed=8, failed=2)
    assert scoring.score_completion(metrics.extract(ws))["score"] == 80.0


def test_completion_missing_test_result(tmp_path):
    ws = make_ws(tmp_path)
    (ws / "test_result.md").unlink()
    assert scoring.score_completion(metrics.extract(ws))["score"] == 0.0


# ---------------------------------------------------------------------------
# evidence source: JSON preferred, stale guard, markdown fallback
# ---------------------------------------------------------------------------

def test_bench_json_preferred(tmp_path):
    ws = make_ws(tmp_path, bench=[bench_row((4096, 4096), 1.10)])
    rows, src = metrics.bench_rows(ws)
    assert src == "bench_result.json" and rows[0]["speedup"] == 1.10


def test_bench_json_stale_falls_back(tmp_path):
    ws = make_ws(tmp_path, bench=[bench_row((4096, 4096), 9.99)],
                 speedup_md="| [4096, 4096] | fp16 | 0.02 | 0.02 | **0.50×** |\n")
    # make bench_result.json older than kernel.py
    bj = ws / "bench_result.json"
    import os
    import time
    old = time.time() - 3600
    os.utime(bj, (old, old))
    rows, src = metrics.bench_rows(ws)
    assert "stale" in src
    assert rows and rows[0]["speedup"] == 0.50


def test_bench_markdown_fallback_handles_bold_and_at_shape(tmp_path):
    ws = make_ws(tmp_path, speedup_md=(
        "| [1024,1024]@[1024,1024] | fp16 | 22.9 | 14.1 | **0.61×** |\n"
        "| prose line | ~17× |\n"))
    rows, src = metrics.bench_rows(ws)
    assert src == "speedup.md"
    assert rows[0]["shape"] == (1024, 1024)      # A shape only, not all four dims
    assert rows[0]["speedup"] == 0.61
    assert len(rows) == 1                        # prose row rejected


# ---------------------------------------------------------------------------
# performance + scalability gate
# ---------------------------------------------------------------------------

def test_performance_parity_is_100(tmp_path):
    ws = make_ws(tmp_path, bench=[bench_row((4096, 4096), 1.0)])
    assert scoring.score_performance(metrics.extract(ws))["score"] == 100.0


def test_performance_beating_torch_caps_at_100(tmp_path):
    ws = make_ws(tmp_path, bench=[bench_row((4096, 4096), 3.4)])
    assert scoring.score_performance(metrics.extract(ws))["score"] == 100.0


def test_performance_conservative_within_deciding_shape(tmp_path):
    """Largest shape with two dtypes -> the WORST dtype decides."""
    ws = make_ws(tmp_path, bench=[bench_row((4096, 4096), 0.51, "torch.float16"),
                                  bench_row((4096, 4096), 0.80, "torch.float32")])
    ev = metrics.extract(ws)
    assert scoring.score_performance(ev)["deciding_speedup"] == 0.51


def test_coverage_gate_penalises_uncovered_required_shapes(tmp_path):
    """Fast only on a toy shape => gate scales the score down (attention case)."""
    case = {"required_shapes": [[1, 8, 64, 64], [1, 8, 128, 64], [1, 8, 256, 64]]}
    ws = make_ws(tmp_path, bench=[bench_row((1, 8, 64, 64), 2.49)])
    ev = metrics.extract(ws, case)
    perf = scoring.score_performance(ev)
    assert perf["base_speedup_score"] == 100.0
    assert round(perf["coverage_gate"], 4) == round(1 / 3, 4)
    assert round(perf["score"], 2) == round(100 / 3, 2)


def test_coverage_gate_disabled_without_required_shapes(tmp_path):
    ws = make_ws(tmp_path, bench=[bench_row((4096, 4096), 0.5)])
    perf = scoring.score_performance(metrics.extract(ws, {}))
    assert perf["coverage_gate"] == 1.0 and perf["score"] == 50.0


def test_coverage_gate_full_credit(tmp_path):
    case = {"required_shapes": [[1024, 1024], [4096, 4096]]}
    ws = make_ws(tmp_path, bench=[bench_row((1024, 1024), 0.61),
                                 bench_row((4096, 4096), 0.51)])
    perf = scoring.score_performance(metrics.extract(ws, case))
    assert perf["coverage_gate"] == 1.0 and perf["score"] == 51.0


# ---------------------------------------------------------------------------
# engineering / anti-cheat
# ---------------------------------------------------------------------------

def test_engineering_full(tmp_path):
    ws = make_ws(tmp_path)
    assert scoring.score_engineering(metrics.extract(ws))["score"] == 100.0


def test_anti_cheat_flags_torch_wrapper(tmp_path):
    ws = make_ws(tmp_path, kernel=CHEAT_KERNEL)
    ac = metrics.anti_cheat(ws)
    assert ac["pass"] is False
    eng = scoring.score_engineering(metrics.extract(ws))
    assert eng["score"] == 75.0 and eng["anti_cheat_pass"] is False


def test_anti_cheat_ignores_docstring_mention(tmp_path):
    ws = make_ws(tmp_path, kernel=(
        '"""Reference: torch.nn.functional.relu(x) — a docstring mention."""\n'
        + GOOD_KERNEL))
    assert metrics.anti_cheat(ws)["pass"] is True


def test_engineering_penalises_missing_deliverables(tmp_path):
    ws = make_ws(tmp_path)
    (ws / "speedup.md").unlink()
    assert scoring.score_engineering(metrics.extract(ws))["score"] < 100.0


# ---------------------------------------------------------------------------
# convergence / efficiency / aggregate
# ---------------------------------------------------------------------------

def test_convergence_structured_full_marks(tmp_path):
    ws = make_ws(tmp_path)
    (ws / "state_transitions.jsonl").write_text(
        json.dumps({"from": "optimize", "to": "done",
                    "handoff": "→ handoff: OPTIMIZE_DONE: TARGET_MET",
                    "iteration": {"design": 1, "implement": 1, "optimize": 1}}) + "\n",
        encoding="utf-8")
    conv = scoring.score_convergence(metrics.extract(ws))
    assert conv["score"] == 100.0
    assert conv["source"] == "bottleneck.json"


def test_convergence_structured_partial(tmp_path):
    """A vague structured file scores below full marks (e.g. no grading)."""
    ws = make_ws(tmp_path)
    (ws / "bottleneck.json").write_text(json.dumps({
        "classification": "made_up_class",
        "suggestions": [],
        "convergence": {"judgement": "done"},
    }), encoding="utf-8")
    conv = scoring.score_convergence(metrics.extract(ws))
    # invalid class 0 + no bandwidth 0 + no suggestions 0 + judgement 15
    # + not-done 0 + no reason 0
    assert conv["score"] == 15.0
    assert conv["source"] == "bottleneck.json"


def test_convergence_falls_back_to_keywords_and_flags_it(tmp_path):
    ws = make_ws(tmp_path)
    (ws / "bottleneck.json").unlink()
    (ws / "state_transitions.jsonl").write_text(
        json.dumps({"from": "optimize", "to": "done",
                    "handoff": "→ handoff: OPTIMIZE_DONE: NO_IMPROVEMENT"}) + "\n",
        encoding="utf-8")
    conv = scoring.score_convergence(metrics.extract(ws))
    assert "keyword-fallback" in conv["source"]
    # class 30 + bandwidth 10 + grading 20 + done 10 + honest reason 15 = 85
    # (the fixture has no "收敛" judgement wording -> 15 lost, which is exactly
    # the wording sensitivity the structured contract removes)
    assert conv["score"] == 85.0


def test_convergence_missing_evidence_is_zero(tmp_path):
    ws = make_ws(tmp_path)
    (ws / "bottleneck.json").unlink()
    (ws / "bottleneck.md").write_text("nothing useful\n", encoding="utf-8")
    assert scoring.score_convergence(metrics.extract(ws))["score"] == 0.0


def test_efficiency_within_budget_is_100(tmp_path):
    ws = make_ws(tmp_path)
    (ws / "run_meta.json").write_text(json.dumps(
        {"total_tokens": 100_000, "wall_ms": 1_000_000}), encoding="utf-8")
    ev = metrics.extract(ws)
    assert scoring.score_efficiency(ev, "easy")["score"] == 100.0


def test_efficiency_over_budget_penalised(tmp_path):
    ws = make_ws(tmp_path)
    # easy budget is 300k tokens (V7 calibration) -> 600k is 2x over
    (ws / "run_meta.json").write_text(json.dumps(
        {"total_tokens": 600_000, "wall_ms": 1_000_000}), encoding="utf-8")
    ev = metrics.extract(ws)
    # tokens 50 (300k/600k) x .5 + wall 100 x .3 + iter 100 x .2 = 75
    assert scoring.score_efficiency(ev, "easy")["score"] == pytest.approx(75.0, abs=0.1)
    assert scoring.score_efficiency(ev, "easy")["components"]["tokens"] == 50.0


def test_efficiency_without_cost_evidence_is_none_not_100(tmp_path):
    """Unmeasured must not read as perfect.

    ``outputs/relu`` is a reference run with no token/wall capture. The iteration
    component is always computable, so before this guard the dimension
    renormalised to a full 100 — reporting "perfect efficiency" for a run nothing
    ever measured.
    """
    ws = make_ws(tmp_path)
    ev = metrics.extract(ws)          # no run_meta.json at all
    got = scoring.score_efficiency(ev, "easy")
    assert got["score"] is None
    assert got["tokens_captured"] is False
    assert "not measurable" in got["note"]


def test_efficiency_partial_evidence_still_scores(tmp_path):
    """Wall clock alone is real evidence and must still score."""
    ws = make_ws(tmp_path)
    (ws / "run_meta.json").write_text(json.dumps(
        {"wall_ms": 1_000_000}), encoding="utf-8")
    ev = metrics.extract(ws)
    got = scoring.score_efficiency(ev, "easy")
    assert got["score"] is not None
    assert "tokens" not in got["components"]


def test_convergence_classification_checked_against_measured_expectation(tmp_path):
    """A well-formed but *wrong* classification must not score full marks.

    D4 used to award its 30 classification points for any class in the enum, i.e.
    it graded the format of the answer. rmsnorm is the concrete case: our own run
    got 2.9x by restructuring the reduction, so calling it ``bandwidth_bound``
    ("at the HBM wall, nothing to do") is factually false, not just a synonym.
    """
    ws = make_ws(tmp_path)
    case = {"id": "rmsnorm", "bottleneck_classes": ["reduction_bound"]}

    def conv_with(cls):
        (ws / "bottleneck.json").write_text(json.dumps({
            "classification": cls,
            "bandwidth": {"measured": 1.6, "unit": "TB/s", "utilization_pct": 100},
            "suggestions": [{"id": "S1", "priority": "P0", "actionable": True}],
            "convergence": {"reason": "NO_IMPROVEMENT", "judgement": "3N 访存已到 HBM"},
        }), encoding="utf-8")
        return scoring.score_convergence(metrics.extract(ws), case)

    right = conv_with("reduction_bound")
    assert right["classification_ok"] is True
    assert right["score"] == 90.0          # 30+10+20+15+15, no state log -> no +10

    wrong = conv_with("bandwidth_bound")
    assert wrong["classification_ok"] is False
    assert wrong["score"] == 60.0          # the 30 class points are withheld
    assert "WRONG for this case" in wrong["classification_note"]
    assert wrong["classification_expected"] == ["reduction_bound"]


def test_convergence_synonyms_all_accepted(tmp_path):
    """Two classes that mean the same physical limit both score full marks."""
    ws = make_ws(tmp_path)
    case = {"id": "gelu", "bottleneck_classes": ["memory_bound", "bandwidth_bound"]}
    for cls in ("memory_bound", "bandwidth_bound"):
        (ws / "bottleneck.json").write_text(json.dumps({
            "classification": cls,
            "bandwidth": {"measured": 1.29, "unit": "TB/s", "utilization_pct": 81},
            "suggestions": [{"id": "S1", "priority": "P2", "actionable": True}],
            "convergence": {"reason": "TARGET_MET",
                            "judgement": "fp16 全 shape 不慢于 torch"},
        }), encoding="utf-8")
        got = scoring.score_convergence(metrics.extract(ws), case)
        assert got["classification_ok"] is True, cls
        assert got["score"] == 90.0


def test_convergence_without_expectation_falls_back_and_flags(tmp_path):
    """Cases we have not profiled must not be scored as if verified."""
    ws = make_ws(tmp_path)
    (ws / "bottleneck.json").write_text(json.dumps({
        "classification": "memory_bound",
        "bandwidth": {"measured": 1.0, "unit": "TB/s"},
        "suggestions": [{"id": "S1", "priority": "P1", "actionable": True}],
        "convergence": {"reason": "TARGET_MET", "judgement": "ok"},
    }), encoding="utf-8")
    got = scoring.score_convergence(metrics.extract(ws), {"id": "unprofiled-case"})
    assert got["classification_ok"] is None
    assert got["score"] == 90.0
    assert "no bottleneck_classes" in got["classification_note"]


def test_convergence_hardware_limit_must_be_backed_by_its_own_bandwidth(tmp_path):
    """Claiming "at the hardware limit" while reporting 34% of HBM is a false claim.

    ``ops.yaml`` itself defines the hardware limit as HBM utilisation >= 95%, so the
    claim is checkable against the run's **own** ``bandwidth.utilization_pct`` with no
    external data: a run at 34% is contradicting its own evidence. The rule was added
    after a run was observed writing ``reason: HARDWARE_LIMIT`` next to a reported
    0.54 TB/s.
    """
    ws = make_ws(tmp_path)

    def conv(util, reason):
        (ws / "bottleneck.json").write_text(json.dumps({
            "classification": "reduction_bound",
            "bandwidth": {"measured": 0.54, "unit": "TB/s", "utilization_pct": util},
            "suggestions": [{"id": "S1", "priority": "P0", "actionable": True}],
            "convergence": {"reason": reason, "judgement": "v2 用 Cube 归约"},
        }), encoding="utf-8")
        return scoring.score_convergence(metrics.extract(ws),
                                         {"id": "rmsnorm",
                                          "bottleneck_classes": ["reduction_bound"]})

    contradicted = conv(34, "HARDWARE_LIMIT")
    assert contradicted["convergence_reason_ok"] is False
    assert contradicted["score"] == 75.0        # 90 minus the 15 honest-reason points
    assert "contradicts its own evidence" in contradicted["convergence_reason_note"]

    honest = conv(100, "HARDWARE_LIMIT")
    assert honest["convergence_reason_ok"] is True
    assert honest["score"] == 90.0

    # a reason that makes no bandwidth claim is unaffected
    assert conv(34, "CAP")["score"] == 90.0
    # and an unmeasured utilisation cannot be used to convict anyone
    # 30 (in-enum, case declares no expectation) + 0 bw + 20 + 15 + 0 done + 15
    (ws / "bottleneck.json").write_text(json.dumps({
        "classification": "reduction_bound",
        "suggestions": [{"id": "S1", "priority": "P0", "actionable": True}],
        "convergence": {"reason": "HARDWARE_LIMIT", "judgement": "x"},
    }), encoding="utf-8")
    unmeasured = scoring.score_convergence(metrics.extract(ws), {"id": "rmsnorm"})
    assert unmeasured["convergence_reason_ok"] is True
    assert unmeasured["score"] == 80.0


def test_aggregate_shape_and_completed_flag(tmp_path):
    ws = make_ws(tmp_path, bench=[bench_row((4096, 4096), 1.10)])
    (ws / "run_meta.json").write_text(json.dumps(
        {"harness": "dsh", "model": "m", "total_tokens": 1000,
         "wall_ms": 1000}), encoding="utf-8")
    res = scoring.score(ws, {"id": "case-x", "difficulty": "easy",
                             "required_shapes": [[4096, 4096]]})
    assert res["domain"] == "ops" and res["case"] == "case-x"
    assert res["completed"] is True
    assert set(res["dimensions"]) == {"completion", "performance",
                                      "engineering", "convergence", "efficiency"}
    per_1m = res["effect_score"] / 1000 * 1e6
    assert res["effect_per_1m_tokens"] == pytest.approx(per_1m, rel=1e-3)
