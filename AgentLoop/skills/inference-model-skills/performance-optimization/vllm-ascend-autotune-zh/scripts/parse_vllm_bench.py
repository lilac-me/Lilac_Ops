#!/usr/bin/env python3
"""Parse vLLM bench serve logs and rank valid results by output throughput."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


PATTERNS = {
    "successful_requests": r"Successful requests:\s+(\d+)",
    "failed_requests": r"Failed requests:\s+(\d+)",
    "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "total_input_tokens": r"Total input tokens:\s+(\d+)",
    "total_generated_tokens": r"Total generated tokens:\s+(\d+)",
    "output_token_throughput": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_token_throughput": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="Log files or glob patterns")
    p.add_argument("--expected-prompts", type=int)
    p.add_argument("--expected-output-len", type=int)
    p.add_argument("--jsonl", action="store_true", help="Emit JSONL instead of table")
    return p.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def coerce(value: str):
    return float(value) if "." in value else int(value)


def parse_file(path: Path, args: argparse.Namespace) -> dict:
    text = path.read_text(errors="replace")
    row = {"path": str(path)}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text)
        row[key] = coerce(m.group(1)) if m else None

    reasons = []
    if row["output_token_throughput"] is None:
        reasons.append("missing throughput")
    if args.expected_prompts is not None and row["successful_requests"] != args.expected_prompts:
        reasons.append(f"successful_requests != {args.expected_prompts}")
    if row["failed_requests"] not in (0, None):
        reasons.append("failed_requests > 0")
    if args.expected_prompts is not None and args.expected_output_len is not None:
        expected_tokens = args.expected_prompts * args.expected_output_len
        if row["total_generated_tokens"] != expected_tokens:
            reasons.append(f"total_generated_tokens != {expected_tokens}")

    # Logs with Bad Request or startup errors may still contain a zero summary; mark invalid.
    if "Bad Request" in text:
        reasons.append("Bad Request")
    if "ERROR: vLLM server exited during startup" in text:
        reasons.append("server startup failure")

    row["valid"] = not reasons
    row["invalid_reasons"] = reasons
    return row


def main() -> None:
    args = parse_args()
    rows = [parse_file(p, args) for p in expand_paths(args.paths) if p.exists()]
    rows.sort(key=lambda r: (bool(r["valid"]), r["output_token_throughput"] or -1), reverse=True)

    if args.jsonl:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return

    print("valid\toutput_tok_s\tsuccess\tfailed\tgen_tokens\tpath\treason")
    for r in rows:
        print(
            f"{r['valid']}\t{r['output_token_throughput']}\t"
            f"{r['successful_requests']}\t{r['failed_requests']}\t"
            f"{r['total_generated_tokens']}\t{r['path']}\t"
            f"{'; '.join(r['invalid_reasons'])}"
        )


if __name__ == "__main__":
    main()
