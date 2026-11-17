"""Weight-sensitivity analysis for the ops effect score (V3).

The effect weights (completion .35 / performance .30 / engineering .20 /
convergence .15) were chosen by judgement, not fitted. Before trusting the
ranking they produce, check whether it survives plausible alternatives: if a
case's rank collapses under a mild reweighting, the ranking is a weighting
artefact rather than a capability signal.

Scoped to the **ops** domain, the only domain currently benchmarked.

Usage::

    python -m benchmark.agents.ops.sensitivity            # default results dir
    python -m benchmark.agents.ops.sensitivity --results benchmark/results/ops
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .console import force_utf8_stdout

EFFECT_DIMS = ["completion", "performance", "engineering", "convergence"]

SCHEMES: dict[str, dict[str, float]] = {
    "baseline": {"completion": 0.35, "performance": 0.30,
                 "engineering": 0.20, "convergence": 0.15},
    "equal": {"completion": 0.25, "performance": 0.25,
              "engineering": 0.25, "convergence": 0.25},
    "completion_first": {"completion": 0.50, "performance": 0.25,
                         "engineering": 0.15, "convergence": 0.10},
    "performance_first": {"completion": 0.25, "performance": 0.45,
                          "engineering": 0.15, "convergence": 0.15},
    "convergence_first": {"completion": 0.30, "performance": 0.25,
                          "engineering": 0.15, "convergence": 0.30},
    "engineering_first": {"completion": 0.30, "performance": 0.25,
                          "engineering": 0.30, "convergence": 0.15},
}


def load_ops_results(results_dir: Path) -> list[dict[str, Any]]:
    """Recursively find every ops result.

    Recursive (not ``*/result.json``) on purpose: one case holds many runs, so the
    layout is ``results/ops/<case>/<harness>__<model>/result.json``. Filtering by the
    result's own ``domain`` field instead of by path depth also means a layout change
    cannot silently make this tool report "no results".
    """
    out = []
    for p in sorted(results_dir.glob("**/result.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("domain") != "ops":
            continue
        out.append(d)
    return out


def effect_under(dims: dict[str, float], weights: dict[str, float]) -> float:
    return round(sum(weights[k] * (dims.get(k) or 0.0) for k in weights), 2)


def analyse(results: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, dict[str, float]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, w in SCHEMES.items():
        eff = {r["case"]: effect_under(r["dimensions"], w) for r in results}
        table[name] = eff
        order = sorted(eff, key=lambda c: (-eff[c], c))
        ranks[name] = {c: i + 1 for i, c in enumerate(order)}

    cases = [r["case"] for r in results]
    moves = {}
    for c in cases:
        rs = [ranks[n][c] for n in SCHEMES]
        moves[c] = {"min": min(rs), "max": max(rs), "range": max(rs) - min(rs),
                    "baseline_rank": ranks["baseline"][c]}
    return {"table": table, "ranks": ranks, "moves": moves, "cases": cases}


def render(res: dict[str, Any]) -> str:
    schemes = list(SCHEMES)
    lines = ["# ops 效果分权重敏感性（V3）", "",
             "> 权重 0.35/0.30/0.20/0.15 是人为选定、未标定。"
             "下表检查换权重后排名是否翻转。", ""]
    hdr = "| case | " + " | ".join(schemes) + " | 排名区间 |"
    lines.append(hdr)
    n_cols = hdr.count("|") - 1
    lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")
    for c in res["cases"]:
        cells = [f"{res['table'][s][c]:.1f}" for s in schemes]
        m = res["moves"][c]
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {m['min']}–{m['max']} |")
    lines += ["", "## 排名变化", "",
              "| case | baseline 排名 | 排名区间 | 是否翻转 |",
              "| --- | --- | --- | --- |"]
    for c in res["cases"]:
        m = res["moves"][c]
        lines.append(f"| {c} | {m['baseline_rank']} | {m['min']}–{m['max']} | "
                     f"{'⚠️ 是' if m['range'] >= 2 else '否'} |")
    flipped = [c for c in res["cases"] if res["moves"][c]["range"] >= 2]
    tail = f"（{', '.join(flipped)}）" if flipped else "——排名对权重稳健。"
    lines += ["", f"**结论**：{len(flipped)}/{len(res['cases'])} 个 case 的排名"
                  f"随权重变动 ≥2 位{tail}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmark/results/ops")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = load_ops_results(Path(args.results))
    if not results:
        raise SystemExit(f"no ops result.json under {args.results}")
    md = render(analyse(results))
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"[sensitivity] written -> {args.out}")


if __name__ == "__main__":
    main()
