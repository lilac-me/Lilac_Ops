"""Ops-benchmark runner: case management + run-meta harness adapter + scoring.

The runner does **not** spawn agents. Each harness (Claude Code / Codex / DSH)
runs the ops-dev pipeline in its own native agent mechanism; the runner only:

  * lists / resolves cases from ``testcases/manifest.yaml``;
  * writes the standard ``run_meta.json`` cost-metadata file into a run workspace
    (the harness adapter contract — see ``benchmark/README.md``);
  * scores a finished workspace via :mod:`scoring`;
  * aggregates many ``result.json`` files into one comparison view (:mod:`report`).

Usage::

    python -m benchmark.agents.ops.runner list-cases
    python -m benchmark.agents.ops.runner record-meta outputs/relu \\
        --harness dsh --model deepseek-v4 --input-tokens 12345 \\
        --output-tokens 6789 --wall-ms 2136000
    python -m benchmark.agents.ops.runner score outputs/relu --case relu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from . import scoring
from .console import force_utf8_stdout

MANIFEST = Path(__file__).parent / "testcases" / "manifest.yaml"
# benchmark/agents/ops -> benchmark/results : the root of the result tree, i.e. the
# same directory `report.py --results` globs. The domain is a subdirectory under it,
# so a result can never land outside the tree the aggregator reads.
RESULTS = Path(__file__).resolve().parents[2] / "results"


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    if yaml is None:
        sys.exit("PyYAML is required (pip install pyyaml)")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def case_by_id(case_id: str) -> Optional[dict[str, Any]]:
    data = load_manifest()
    for c in data.get("cases", []):
        if c.get("id") == case_id:
            return c
    return None


def cmd_list_cases() -> None:
    data = load_manifest()
    cases = data.get("cases", [])
    from collections import Counter
    tiers = Counter(c.get("difficulty", "?") for c in cases)
    print(f"suite={data.get('suite')} total={data.get('total')} declared={len(cases)}")
    print("by difficulty:", dict(tiers))
    print()
    for c in cases:
        print(f"  {c['id']:<28} {c['difficulty']:<9} {c['kernel_type']:<3} "
              f"{c.get('reference', '')}")


def cmd_record_meta(args: argparse.Namespace) -> None:
    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    it, ot = args.input_tokens, args.output_tokens
    if it is not None and ot is not None:
        total = it + ot
    else:
        total = it if it is not None else ot  # may be None -> tokens uncaptured
    meta: dict[str, Any] = {
        "harness": args.harness,
        "model": args.model,
        "provider": args.provider,
        "case": args.case,
        "input_tokens": it,
        "output_tokens": ot,
        "total_tokens": total,
        "wall_ms": args.wall_ms,
        "tool_calls": args.tool_calls,
        "spawns": args.spawns,
        "note": args.note,
    }
    (ws / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[runner] wrote {ws / 'run_meta.json'}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def default_result_path(result: dict, case: dict | None, ws: Path) -> Path:
    """``results/<domain>/<case>/<harness>__<model>/result.json``.

    The run id goes in a **directory**, not the filename. Two reasons:

    * one case holds many runs — same case under a second harness or model must not
      overwrite the first;
    * ``report.py`` finds results with a single ``**/result.json`` glob, so anything
      named otherwise is silently absent from the comparison report. An earlier
      default named the file ``<harness>__<model>.json``, which dodged that glob —
      i.e. using the default produced a result the report never showed.
    """
    domain = result.get("domain") or "ops"
    case_id = (case or {}).get("id") or ws.name
    harness = result.get("harness") or "unknown"
    model = result.get("model") or "unknown"
    return RESULTS / domain / case_id / f"{harness}__{model}" / "result.json"


def cmd_score(args: argparse.Namespace) -> None:
    ws = Path(args.workspace)
    case = None
    if args.case:
        case = case_by_id(args.case)
        if case is None:
            sys.exit(f"unknown case id: {args.case}")
    result = scoring.score(ws, case)
    out = Path(args.out) if args.out else default_result_path(result, case, ws)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[runner] result -> {out}")


def main() -> None:
    force_utf8_stdout()

    p = argparse.ArgumentParser(description="ops-agent benchmark runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-cases")

    s_meta = sub.add_parser("record-meta", help="write run_meta.json (harness adapter)")
    s_meta.add_argument("workspace")
    s_meta.add_argument("--harness", required=True,
                        choices=["claude", "codex", "dsh", "manual"])
    s_meta.add_argument("--model", required=True)
    s_meta.add_argument("--provider", default=None)
    s_meta.add_argument("--case", default=None)
    s_meta.add_argument("--input-tokens", type=int, default=None)
    s_meta.add_argument("--output-tokens", type=int, default=None)
    s_meta.add_argument("--wall-ms", type=int, default=None)
    s_meta.add_argument("--tool-calls", type=int, default=None)
    s_meta.add_argument("--spawns", type=int, default=None)
    s_meta.add_argument("--note", default=None)

    s_score = sub.add_parser("score", help="score a finished workspace")
    s_score.add_argument("workspace")
    s_score.add_argument("--case", default=None)
    s_score.add_argument("--out", default=None)

    args = p.parse_args()
    if args.cmd == "list-cases":
        cmd_list_cases()
    elif args.cmd == "record-meta":
        cmd_record_meta(args)
    elif args.cmd == "score":
        cmd_score(args)


if __name__ == "__main__":
    main()
