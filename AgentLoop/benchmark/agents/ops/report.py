"""Aggregate many domain ``result.json`` files into one cross-run comparison view.

The report is the benchmark's deliverable for "不同模型和不同 harness 之间可以对比"
and "同样都是完成的情况下，怎么才是完成更好". Result files share a uniform top-level
schema (``domain``, ``case``, ``difficulty``, ``harness``, ``model``, ``completed``,
``effect_score``, ``cost``, ``effect_per_1m_tokens``); per-domain ``dimensions`` are
rendered compactly so every agent domain lands on the same table.

Ranking rule (documented, not a hidden blend):

    1. ``completed`` first           — an incomplete run never outranks a complete one;
    2. higher ``effect_score`` next  — "效果更好" is the primary quality signal;
    3. lower token cost last         — "token 更少" is the tie-breaker at equal effect.

Usage::

    python -m benchmark.agents.ops.report \\
        --results benchmark/results --out benchmark/results/_index.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .console import force_utf8_stdout

# Synthetic harness of a gold-standard baseline replayed through the scorer
# (``reference__gold-standard``). It is a *reference* run, not "the same case run
# in a second harness", so it must not count as cross-harness coverage.
REFERENCE_HARNESSES = frozenset({"reference"})


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("**/result.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(r: dict[str, Any]) -> tuple:
        completed = 1 if r.get("completed") else 0
        effect = r.get("effect_score") or 0.0
        tokens = r.get("cost", {}).get("total_tokens")
        token_key = tokens if tokens is not None else float("inf")
        return (completed, effect, -token_key)

    return sorted(results, key=key, reverse=True)


def _fmt(x: Any, nd: int = 2, default: str = "—") -> str:
    if x is None:
        return default
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _dims(r: dict[str, Any]) -> str:
    d = r.get("dimensions") or {}
    if not d:
        return "—"
    return " ".join(f"{k}={_fmt(v, 0)}" for k, v in d.items())


def coverage_note(results: list[dict[str, Any]]) -> list[str]:
    """Markdown stating what the generated table actually covers.

    The table is built from whatever run directories happen to exist, so it has to
    say which harness/model combinations those are. With a single *evaluated*
    combination it is that combination's capability baseline and **not** a
    cross-harness / cross-model comparison — the reading error this note exists to
    prevent (review: "harness 基本都是 dsh，cc/codex 怎么补测").
    """
    combo = Counter(f"{_fmt(r.get('harness'))}/{_fmt(r.get('model'))}"
                    for r in results)
    detail = "、".join(f"{k} ×{n}" for k, n in sorted(combo.items()))
    evaluated = sorted(k for k in combo
                       if k.split("/")[0] not in REFERENCE_HARNESSES)
    head = f"> 覆盖范围：{len(results)} 个 run —— harness/model：{detail}。"
    if not evaluated:
        body = "> 表中只有 `reference` 回灌基线，**没有被测组合**。"
    elif len(evaluated) == 1:
        harness = evaluated[0].split("/")[0]
        body = (f"> `reference` 是黄金回灌基线，不是「换个 harness 跑一遍」，"
                f"因此**被测组合只有 1 个（{evaluated[0]}）**："
                "本表是**该组合**的能力基线，**不是**跨 harness / 跨模型对比——"
                "Claude Code / Codex / 其它模型的补测尚未进行"
                f"（{len(results)} 条结果全部由 `{harness}` 产出）；"
                "补测就是换环境再跑一遍同一 case；"
                "协议见 `benchmark/README.md`。")
    else:
        body = (f"> 除 `reference` 回灌基线外**被测组合有 {len(evaluated)} 个**；"
                "本表可用作跨 harness / 跨模型对比，"
                "继续补测仍是再跑一遍、自动进同一张表。")
    return [head, body]


def render_table(results: list[dict[str, Any]]) -> list[str]:
    cols = ["#", "域", "harness", "model", "case", "难度",
            "完成", "效果分", "维度分", "token", "耗时(min)", "效果/百万token"]
    # Separator pipes are spaced on both sides, matching the rest of the repo
    # (markdownlint MD060 flags the compact `|---|---|` form).
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = ["| " + " | ".join(cols) + " |", sep]

    for i, r in enumerate(rank(results), 1):
        cost = r.get("cost", {})
        wall_min = None if cost.get("wall_ms") is None else cost["wall_ms"] / 60000.0
        cells = [str(i), _fmt(r.get("domain")), _fmt(r.get("harness")),
                 _fmt(r.get("model")), _fmt(r.get("case")), _fmt(r.get("difficulty")),
                 "✅" if r.get("completed") else "❌",
                 f"**{_fmt(r.get('effect_score'))}**", _dims(r),
                 _fmt(cost.get("total_tokens"), 0), _fmt(wall_min, 1),
                 _fmt(r.get("effect_per_1m_tokens"), 4)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_markdown(results: list[dict[str, Any]]) -> str:
    if not results:
        return "# agent benchmark\n\n_No results yet._\n"

    lines: list[str] = ["# MindSpeed-Agent 能力 benchmark — 对比报告", ""]
    lines.append("> 排名规则：**完成** > **效果分** > **token 更少**。"
                 "效果分 = 各维度质量加权；"
                 "成本（token/耗时/迭代）单独呈现，不与效果混淆。")
    lines.append(">")
    lines.append("> 每一行是一个 {域 × case × harness × model} 组合，"
                 "所以**换 harness 或换模型就是再跑一遍、结果自动进同一张表**——"
                 "对比能力靠 `harness` / `model` 两列，不需要另一套跑法或另一张榜。")
    lines.append("")
    lines += render_table(results)
    lines += ["", *coverage_note(results)]
    lines += ["", "_按以上排名规则排序。维度分 = 该领域各质量维度得分（0-100）。_"]
    return "\n".join(lines) + "\n"


def main() -> None:
    force_utf8_stdout()

    p = argparse.ArgumentParser(description="aggregate agent benchmark results")
    p.add_argument("--results", default="benchmark/results")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = load_results(Path(args.results))
    md = render_markdown(results)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"[report] written -> {out}")


if __name__ == "__main__":
    main()
