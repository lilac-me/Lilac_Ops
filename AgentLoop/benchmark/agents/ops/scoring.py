"""Five-dimension scoring for the ops-agent benchmark.

Turns the raw evidence extracted by :mod:`metrics` into a comparable score that
answers the benchmark's two core questions:

1. **What is each agent's capability baseline?** — a per-run ``effect_score``
   decomposed into four quality dimensions (completion / performance /
   engineering / convergence), comparable across models and harnesses because
   the case and the scoring formula are identical.

2. **When two runs both complete, which completion is better?** — the answer is
   split into *effect* (did it do the task well) and *cost* (tokens / wall
   clock / iterations). A run is "better" when it achieves **higher effect**;
   at equal effect, the run with **lower cost** (and therefore higher
   ``effect_per_token``) wins.

Dimensions (each 0–100):

    D1 completion      — did it produce a correct operator? (L1-L5 pass rate)
    D2 performance     — how close is speedup to the torch reference parity?
    D3 engineering     — are all deliverables present + Triton-native + a real design?
    D4 convergence     — did optimisation converge honestly, with evidence?
    D5 efficiency      — did it stay within the case's cost budget?

``effect_score`` aggregates D1..D4 (quality). **Do not collapse effect and cost
into a single number** — that hides exactly the trade-off the benchmark exists
to measure. Cost and ``effect_per_token`` are reported alongside, never folded
in.

A run is *completed* iff D1 == 100 (every L1-L5 test passed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from . import metrics
from .console import force_utf8_stdout

# Weights for the effect score (quality dimensions only; cost is separate).
W_EFFECT = {
    "completion": 0.35,
    "performance": 0.30,
    "engineering": 0.20,
    "convergence": 0.15,
}

# --- enums for the structured bottleneck.json convergence contract (D4) ---
CONV_CLASSES = {
    "memory_bound", "compute_bound", "launch_bound", "reduction_bound",
    "bandwidth_bound", "attention_bound",
}
CONV_PRIORITIES = {"P0", "P1", "P2", "P3"}
CONV_REASONS = {
    "NO_SUGGESTIONS", "HARDWARE_LIMIT", "TARGET_MET", "NO_IMPROVEMENT", "CAP",
}

# ``convergence.reason = HARDWARE_LIMIT`` claims "no further gain is physically
# available". The ops pipeline defines that state itself — ops.yaml's convergence
# table says "HBM 利用率 >= 95% (硬件极限)" — so the claim is checkable against the
# run's *own* reported bandwidth utilisation. A run that reports 34% of HBM and
# then claims a hardware limit contradicts itself, and that is a factual error in
# exactly the way a wrong classification is, not a matter of wording.
HARDWARE_LIMIT_MIN_UTIL = 95.0

# Per-difficulty cost budgets, **calibrated on observed agent runs** (V7).
#
# The first version reused the ops-design *human* workload table (easy ~1h,
# medium ~3h, hard ~8h). Measured agent runs showed that table is mis-scaled for
# agents: an agent finishes a "hard" operator in ~21-28 min wall clock, and the
# observed token spread is 31k-507k (median ~118k; high values are inflated by
# turn-level context, see run_meta notes). Budgets below are the observed range
# plus headroom, so the efficiency dimension separates *outliers* rather than
# marking every normal run as over budget.
TIER = {
    "easy":    {"budget_wall_ms": 3_600_000,  "budget_tokens": 300_000},
    "medium":  {"budget_wall_ms": 3_600_000,  "budget_tokens": 250_000},
    "hard":    {"budget_wall_ms": 5_400_000,  "budget_tokens": 400_000},
    "complex": {"budget_wall_ms": 10_800_000, "budget_tokens": 800_000},
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------

def score_completion(ev: dict) -> dict[str, Any]:
    t = ev["test"]
    if not t.get("has_test_result"):
        return {"score": 0.0, "passed": None, "failed": None,
                "bit_exact": False, "note": "test_result.md missing"}
    passed, failed = t.get("passed"), t.get("failed")
    if passed is None or failed is None:
        return {"score": 0.0, "passed": passed, "failed": failed,
                "bit_exact": t.get("bit_exact"), "note": "pass/fail counts unparseable"}
    total = passed + failed
    score = 100.0 * passed / total if total else 0.0
    return {"score": _round(score), "passed": passed, "failed": failed,
            "bit_exact": t.get("bit_exact"),
            "note": "bit-exact" if (t.get("bit_exact") and failed == 0) else None}


def score_performance(ev: dict) -> dict[str, Any]:
    """Performance = parity with torch, **gated by required-shape coverage**.

    ``base`` = 100 × clamp(deciding_speedup, 0, 1): parity with the torch vendor
    kernel (speedup 1.0) is the honest bar; beating it caps at 100.

    ``gate`` = fraction of the case's ``required_shapes`` the run actually
    produced. This is the **scalability gate**: a kernel that is fast only on toy
    shapes (e.g. attention at S=64 while S=128/256 UB-overflow) must not score as
    if it solved the case. No ``required_shapes`` declared => gate = 1.0.
    """
    sp = ev.get("speedup", {})
    spd = sp.get("deciding")
    cov = sp.get("coverage") or {"gate": 1.0}
    gate = float(cov.get("gate", 1.0))
    if spd is None:
        return {"score": 0.0, "deciding_speedup": None, "coverage_gate": gate,
                "coverage": cov, "source": sp.get("source"),
                "note": f"no speedup evidence (source={sp.get('source')})"}
    base = 100.0 * _clamp(spd, 0.0, 1.0)
    # V4: surface the repeat-measurement spread so a noisy number is visible
    stds = [r.get("speedup_std") for r in (sp.get("rows") or [])
            if isinstance(r.get("speedup_std"), (int, float))]
    runs = next((r.get("runs") for r in (sp.get("rows") or [])
                 if r.get("runs")), None)
    return {
        "score": _round(base * gate),
        "deciding_speedup": _round(spd, 4),
        "base_speedup_score": _round(base),
        "coverage_gate": _round(gate, 4),
        "coverage": cov,
        "source": sp.get("source"),
        "runs": runs,
        "worst_speedup_std": _round(max(stds), 4) if stds else None,
        "note": "100*clamp(deciding_speedup) x required-shape coverage gate",
    }


def score_engineering(ev: dict) -> dict[str, Any]:
    files = ev["deliverables"]
    present = sum(1 for f in files if f["present"])
    ratio = present / len(files) if files else 0.0

    dq = ev["design"]
    sections = dq.get("sections", 0)
    # 8 of the 12 design sections is an "adequate" design; more is rewarded.
    design_score = 100.0 * _clamp(sections / 8.0)

    ac = ev["anti_cheat"]
    anti_cheat_score = 100.0 if ac.get("pass") else 0.0

    score = 0.50 * 100.0 * ratio + 0.25 * anti_cheat_score + 0.25 * design_score
    return {
        "score": _round(score),
        "files_present": f"{present}/{len(files)}",
        "anti_cheat_pass": ac.get("pass"),
        "anti_cheat_reasons": ac.get("reasons", []),
        "design_sections": sections,
        "note": "files 50% + anti-cheat 25% + design quality 25%",
    }


def score_convergence(ev: dict, case: Optional[dict] = None) -> dict[str, Any]:
    """Convergence = honest, evidence-backed optimisation stop (0-100).

    Prefers the **structured** ``bottleneck.json`` contract; the keyword
    heuristic on ``bottleneck.md``/``opt_history.md`` remains only as a fallback
    and is flagged, because it scores wording rather than substance.

    ``classification`` is checked against the case's ``bottleneck_classes`` when
    the manifest declares one. Scoring only enum *membership* would repeat the
    exact flaw the structured contract exists to remove: it would grade the
    format of the answer instead of whether the answer is right. The expectation
    comes from our own profiling of that case, and lists synonyms that describe
    the same physical limit, so a correct answer is never punished for wording.
    """
    c = ev["convergence"]
    bs = ev.get("bottleneck_structured") or {}
    b = ev["bottleneck"]

    if bs.get("present"):
        pts = 0.0
        cls = bs.get("classification")
        expected = list((case or {}).get("bottleneck_classes") or [])
        if cls not in CONV_CLASSES:
            class_note = f"not a known class: {cls!r}"
            class_ok = False
        elif not expected:
            # No measured expectation for this case: fall back to membership and
            # say so, rather than silently implying the class was verified.
            pts += 30.0
            class_ok = None
            class_note = "in-enum only (case declares no bottleneck_classes)"
        elif cls in expected:
            pts += 30.0
            class_ok = True
            class_note = f"matches the case's measured bottleneck {expected}"
        else:
            class_ok = False
            class_note = (f"WRONG for this case: {cls!r} not in measured set "
                          f"{expected}")

        bw = bs.get("bandwidth") or {}
        if bw.get("measured") is not None and bw.get("unit"):
            pts += 10.0
        sug = bs.get("suggestions") or []
        if sug and all(s.get("priority") in CONV_PRIORITIES for s in sug):
            pts += 20.0
        if (bs.get("convergence") or {}).get("judgement"):
            pts += 15.0
        if c.get("reached_done"):
            pts += 10.0
        reason = (bs.get("convergence") or {}).get("reason")
        reason_ok = None
        reason_note = None
        if reason in CONV_REASONS:
            util = bw.get("utilization_pct")
            if (reason == "HARDWARE_LIMIT" and isinstance(util, (int, float))
                    and util < HARDWARE_LIMIT_MIN_UTIL):
                reason_ok = False
                reason_note = (f"contradicts its own evidence: HARDWARE_LIMIT claimed "
                               f"at {util}% of HBM (< {HARDWARE_LIMIT_MIN_UTIL}% is "
                               "what the pipeline itself defines as the "
                               "hardware limit)")
            else:
                pts += 15.0
                reason_ok = True
        return {
            "score": _round(pts),
            "source": "bottleneck.json",
            "classification": cls,
            "classification_ok": class_ok,
            "classification_expected": expected or None,
            "classification_note": class_note,
            "suggestions": len(sug),
            "convergence_reason": reason,
            "convergence_reason_ok": reason_ok,
            "convergence_reason_note": reason_note,
            "reached_done": c.get("reached_done"),
            "note": "structured: 30 class(checked against measured expectation) "
                    "+ 10 bw + 20 grading + 15 judgement + 10 done "
                    "+ 15 honest-reason(self-consistency checked)",
        }

    # ---- fallback: keyword heuristic (wording-sensitive, flagged) ----
    pts = 0.0
    if b.get("classification"):
        pts += 30.0
    if b.get("has_bandwidth"):
        pts += 10.0
    if b.get("has_suggestion_grading"):
        pts += 20.0
    if c.get("has_judgement"):
        pts += 15.0
    if c.get("reached_done"):
        pts += 10.0
    if c.get("convergence_reason") in CONV_REASONS:
        pts += 15.0
    return {
        "score": _round(pts),
        "source": "keyword-fallback (no bottleneck.json)",
        "classification": b.get("classification"),
        "convergence_reason": c.get("convergence_reason"),
        "reached_done": c.get("reached_done"),
        "note": "fallback keyword heuristic - emit bottleneck.json for a "
                "structure-based score",
    }


def score_efficiency(ev: dict, tier: str) -> dict[str, Any]:
    cfg = TIER.get(tier, TIER["easy"])
    cost = ev["cost"]
    parts: dict[str, float] = {}
    weights = {"tokens": 0.5, "wall": 0.3, "iter": 0.2}

    tokens = cost.get("total_tokens")
    wall = cost.get("wall_ms")
    iters = cost.get("iterations") or {}

    # "Unmeasured" is not "perfect". The iteration component is always computable
    # (a run with no state log has 0 wasteful iterations), so without this guard a
    # run with *no cost evidence at all* would renormalise to a perfect 100 — which
    # is how `outputs/relu` (a reference run with no token capture) silently scored
    # full marks on a dimension nothing measured.
    if not tokens and not wall:
        return {
            "score": None,
            "components": {},
            "budget": cfg,
            "tokens_captured": False,
            "note": "no cost evidence (no tokens and no wall clock): efficiency is "
                    "not measurable and must not score as perfect",
        }

    if tokens:
        parts["tokens"] = 100.0 * _clamp(cfg["budget_tokens"] / tokens)

    if wall:
        parts["wall"] = 100.0 * _clamp(cfg["budget_wall_ms"] / wall)

    # Iteration efficiency: minimal iterations (<= per-stage caps) are ideal.
    optimize = iters.get("optimize", 0)
    design = iters.get("design", 0)
    implement = iters.get("implement", 0)
    wasteful = max(0, design - 1) + max(0, implement - 1) + max(0, optimize - 3)
    parts["iter"] = 100.0 * _clamp(1.0 - wasteful / 5.0)

    w = sum(weights[k] for k in parts)
    score = sum(parts[k] * weights[k] for k in parts) / w if w else None

    return {
        "score": _round(score),
        "components": {k: _round(v) for k, v in parts.items()},
        "budget": cfg,
        "tokens_captured": cost.get("tokens_captured", False),
        "note": "tokens/wall/iter renormalised over available inputs",
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def score(workspace: Path, case: Optional[dict] = None) -> dict[str, Any]:
    """Score one run workspace. ``case`` is the matching manifest entry."""
    case = case or {}
    tier = case.get("difficulty", "easy")
    ev = metrics.extract(workspace, case)

    dims = {
        "completion": score_completion(ev),
        "performance": score_performance(ev),
        "engineering": score_engineering(ev),
        "convergence": score_convergence(ev, case),
        "efficiency": score_efficiency(ev, tier),
    }

    # Effect = quality dimensions only (D1-D4). Missing evidence = 0.
    effect = 0.0
    for name, w in W_EFFECT.items():
        d = dims[name].get("score")
        effect += w * (d if d is not None else 0.0)
    effect = _round(effect)

    cost = ev["cost"]
    completed = dims["completion"]["score"] == 100.0
    effect_per_mtok = None
    if cost.get("total_tokens"):
        effect_per_mtok = _round(effect / cost["total_tokens"] * 1_000_000, 4)

    return {
        "domain": "ops",
        "case": case.get("id", workspace.name),
        "difficulty": tier,
        "harness": cost.get("harness"),
        "model": cost.get("model"),
        "provider": cost.get("provider"),
        "completed": completed,
        "dimensions": {k: v["score"] for k, v in dims.items()},
        "dimension_detail": dims,
        "effect_score": effect,
        "cost": {
            "total_tokens": cost.get("total_tokens"),
            "input_tokens": cost.get("input_tokens"),
            "output_tokens": cost.get("output_tokens"),
            "cache_read_tokens": cost.get("cache_read_tokens"),
            "wall_ms": cost.get("wall_ms"),
            "iterations": cost.get("iterations"),
            "reach_stage": cost.get("reach_stage"),
        },
        "effect_per_1m_tokens": effect_per_mtok,
        "evidence": {
            "test": ev["test"],
            "speedup": ev["speedup"],
            "bottleneck": ev["bottleneck"],
            "convergence": ev["convergence"],
            "anti_cheat": ev["anti_cheat"],
        },
    }


# ---------------------------------------------------------------------------
# CLI: score a single workspace
# ---------------------------------------------------------------------------

def main():
    import argparse
    force_utf8_stdout()

    p = argparse.ArgumentParser(description="Score an ops-agent run workspace")
    p.add_argument("workspace")
    p.add_argument("--case", default=None, help="manifest case id (relu, gelu, ...)")
    p.add_argument("--difficulty", default="easy", choices=sorted(TIER))
    p.add_argument("--out", default=None, help="write result.json to this path")
    args = p.parse_args()

    case = {"id": args.case or Path(args.workspace).name,
            "difficulty": args.difficulty}
    result = score(Path(args.workspace), case)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n[scoring] written -> {args.out}")


if __name__ == "__main__":
    main()
