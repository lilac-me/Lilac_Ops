"""DSH harness token adapter — turn-level token accounting from a DSH session log.

Serves requirement #2 of the benchmark ("精准衡量整体效果 … token 更少还是效果更好"):
token cost is measured at the harness level and this adapter turns the harness's
native session log into the standard ``run_meta.json`` fields the scorer reads.

How DSH records usage
---------------------
DSH appends a zstd-compressed JSON Lines session log (``session.jsonl.zstd``).
Relevant record types:

* ``turn/start`` / ``turn/end``      — delimit one human turn (one benchmark case
                                       run maps naturally to one turn);
* ``assistant/message``              — carries ``data.usage`` =
                                       {inputTokens, outputTokens, cacheReadTokens}.

This adapter segments assistant-message usage by turn and emits:

* ``input_tokens``       — sum of inputTokens across the turn's assistant messages;
* ``output_tokens``      — sum of outputTokens (generation cost);
* ``cache_read_tokens``  — context re-reads (prompt cache), reported separately:
                           it is NOT generation cost, so the scorer's efficiency
                           uses ``total_tokens = input + output`` only.

Other harnesses (Claude Code ``usage`` API, Codex ``subtask`` stats) can implement
the same ``run_meta.json`` contract — this file is the reference implementation.

Usage::

    python -m benchmark.agents.ops.harness_adapters.dsh_tokens list
    python -m benchmark.agents.ops.harness_adapters.dsh_tokens extract \\
        --workspace outputs/gelu --turn 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from ..console import force_utf8_stdout


def _session_path() -> str:
    p = os.environ.get("DSH_SESSION_JSONL")
    if p:
        return p
    # Fallback: rebuild from DSH_HOME + DSH_SESSION_ID + cwd-derived dir is not
    # reliably reconstructible, so require the env var.
    sys.exit("DSH_SESSION_JSONL is not set in this environment")


def _load_lines(path: str) -> list[dict]:
    raw = Path(path).read_bytes()
    try:
        import zstandard
    except ImportError:
        sys.exit("zstandard is required (pip install zstandard)")
    reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw))
    out = []
    for line in reader.read().decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def per_turn_usage(lines: list[dict]) -> list[dict[str, Any]]:
    """Return one dict per turn.

    Keys: ``turn``, ``input``, ``output``, ``cache``, ``messages``, ``t0_ms``,
    ``t1_ms``.
    """
    turns: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    for o in lines:
        t = o.get("type")
        ts = o.get("time")   # epoch ms in DSH turn/start records
        if t == "turn/start":
            cur = {"turn": (o.get("data") or {}).get("turn"),
                   "input": 0, "output": 0, "cache": 0, "messages": 0,
                   "t0_ms": ts}
            turns.append(cur)
        elif t == "assistant/message" and cur is not None:
            u = (o.get("data") or {}).get("usage")
            if isinstance(u, dict):
                cur["input"] += u.get("inputTokens") or 0
                cur["output"] += u.get("outputTokens") or 0
                cur["cache"] += u.get("cacheReadTokens") or 0
                cur["messages"] += 1
        elif t == "turn/end" and cur is not None:
            if ts is not None:
                cur["t1_ms"] = ts
    return turns


def _render_turn(tr: dict[str, Any]) -> str:
    total = tr["input"] + tr["output"]
    wall = None
    if tr.get("t0_ms") and tr.get("t1_ms"):
        wall = (tr["t1_ms"] - tr["t0_ms"]) / 60000.0
    return (f"turn {tr['turn']}: input={tr['input']} output={tr['output']} "
            f"total={total} cache={tr['cache']} msgs={tr['messages']} "
            f"wall_min={wall if wall is None else round(wall, 1)}")


def cmd_list(lines: list[dict]) -> None:
    turns = per_turn_usage(lines)
    if not turns:
        print("no turns found")
        return
    print(f"session turns: {len(turns)}")
    for tr in turns:
        print("  " + _render_turn(tr))


def cmd_extract(lines: list[dict], args: argparse.Namespace) -> None:
    turns = per_turn_usage(lines)
    turn_idx = args.turn
    tr = next((t for t in turns if str(t.get("turn")) == str(turn_idx)), None)
    if tr is None:
        # --turn may be given as a 1-based index into the turn list
        with contextlib.suppress(ValueError, IndexError):
            tr = turns[int(turn_idx) - 1]
    if tr is None:
        sys.exit(f"turn {turn_idx} not found; available: "
                 f"{[t['turn'] for t in turns]}")
    wall_ms = None
    if tr.get("t0_ms") and tr.get("t1_ms"):
        wall_ms = tr["t1_ms"] - tr["t0_ms"]

    fields = {
        "input_tokens": tr["input"],
        "output_tokens": tr["output"],
        "total_tokens": tr["input"] + tr["output"],
        "cache_read_tokens": tr["cache"],
    }
    if wall_ms is not None:
        fields["wall_ms"] = wall_ms

    if args.workspace:
        meta_path = Path(args.workspace) / "run_meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(fields)
        meta.setdefault("note", "")
        meta["note"] += " [token fields injected by harness_adapters/dsh_tokens.py]"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"[dsh_tokens] updated {meta_path}")
    print(_render_turn(tr))
    print(json.dumps(fields, ensure_ascii=False, indent=2))


def main() -> None:
    force_utf8_stdout()

    p = argparse.ArgumentParser(description="DSH harness token adapter")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list per-turn token usage")
    s_ext = sub.add_parser("extract", help="merge a turn's tokens into run_meta.json")
    s_ext.add_argument("--turn", required=True, help="turn id (or 1-based index)")
    s_ext.add_argument("--workspace", default=None,
                       help="workspace whose run_meta.json will be updated")
    args = p.parse_args()

    lines = _load_lines(_session_path())
    if args.cmd == "list":
        cmd_list(lines)
    else:
        cmd_extract(lines, args)


if __name__ == "__main__":
    main()
