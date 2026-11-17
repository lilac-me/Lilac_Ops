#!/usr/bin/env python3
"""Detect unsupported request preemption or KV-capacity risk in vLLM logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


CAPACITY_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.IGNORECASE)
STATS_RE = re.compile(
    r"Running:\s*(\d+)\s*reqs,\s*Waiting:\s*(\d+)\s*reqs.*?"
    r"GPU KV cache usage:\s*([\d.]+)%",
    re.IGNORECASE,
)
REQUIREMENT_MARKER = "BATCH_INVARIANCE_KV_REQUIREMENT:"
UNSUPPORTED_MARKER = "BATCH_INVARIANCE_EVICTION_UNSUPPORTED:"


def _parse_marker_json(line: str, marker: str) -> dict[str, Any] | None:
    if marker not in line:
        return None
    raw = line.split(marker, 1)[1].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    log_path = Path(args.log).resolve()
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    capacity_matches = CAPACITY_RE.findall(text)
    capacity_tokens = (
        int(capacity_matches[-1].replace(",", "")) if capacity_matches else None
    )

    requirement: dict[str, Any] | None = None
    marker_unsupported = False
    for line in lines:
        parsed = _parse_marker_json(line, REQUIREMENT_MARKER)
        if parsed is not None:
            requirement = parsed
        if UNSUPPORTED_MARKER in line:
            marker_unsupported = True

    required_tokens = None
    requirement_source = None
    if requirement is not None:
        raw_required = requirement.get("required_kv_tokens")
        if isinstance(raw_required, int) and raw_required > 0:
            required_tokens = raw_required
            requirement_source = "tokenized_prompts"
        raw_capacity = requirement.get("capacity_kv_tokens")
        if capacity_tokens is None and isinstance(raw_capacity, int) and raw_capacity > 0:
            capacity_tokens = raw_capacity

    if required_tokens is None and args.max_sequence_length is not None:
        block_size = args.block_size
        per_request = math.ceil(args.max_sequence_length / block_size) * block_size
        required_tokens = args.expected_batch_size * per_request
        requirement_source = "configured_max_sequence_length"

    samples: list[dict[str, Any]] = []
    max_usage = 0.0
    max_waiting = 0
    full_batch_seen = False
    initial_queueing_observed = False
    static_batch_transition = False
    transition_evidence: dict[str, Any] | None = None

    for line_number, line in enumerate(lines, start=1):
        stats_match = STATS_RE.search(line)
        if not stats_match:
            continue
        running = int(stats_match.group(1))
        waiting = int(stats_match.group(2))
        usage = float(stats_match.group(3))
        max_usage = max(max_usage, usage)
        max_waiting = max(max_waiting, waiting)
        sample = {
            "line": line_number,
            "running": running,
            "waiting": waiting,
            "kv_cache_usage_percent": usage,
        }
        samples.append(sample)

        if args.static_batch and not full_batch_seen and waiting > 0:
            initial_queueing_observed = True

        if running == args.expected_batch_size and waiting == 0:
            full_batch_seen = True
        elif (
            args.static_batch
            and full_batch_seen
            and running < args.expected_batch_size
            and waiting > 0
        ):
            static_batch_transition = True
            if transition_evidence is None:
                transition_evidence = sample

    capacity_risk = (
        capacity_tokens is not None
        and required_tokens is not None
        and required_tokens > capacity_tokens
    )
    eviction_unsupported = any(
        (
            marker_unsupported,
            capacity_risk,
            static_batch_transition,
        )
    )

    reasons: list[str] = []
    if marker_unsupported:
        reasons.append("test emitted BATCH_INVARIANCE_EVICTION_UNSUPPORTED")
    if capacity_risk:
        reasons.append(
            f"required KV tokens {required_tokens} exceed capacity {capacity_tokens}"
        )
    if static_batch_transition:
        reasons.append(
            "static batch moved from fully running to running below BS with waiting requests"
        )

    evidence_complete = capacity_tokens is not None and required_tokens is not None
    if eviction_unsupported:
        status = "EVICTION_UNSUPPORTED"
        decision = "BLOCKED_EVICTION_UNSUPPORTED"
    elif not samples:
        status = "INCOMPLETE"
        decision = "BLOCKED_SCHEDULER_EVIDENCE"
        reasons.append("no periodic scheduler statistics were found")
    elif not evidence_complete:
        status = "INCOMPLETE"
        decision = "BLOCKED_SCHEDULER_EVIDENCE"
        reasons.append("KV capacity or required-token evidence is missing")
    else:
        status = "PASS"
        decision = "NO_EVICTION_OBSERVED"

    return {
        "status": status,
        "decision": decision,
        "log": str(log_path),
        "static_batch": args.static_batch,
        "expected_batch_size": args.expected_batch_size,
        "capacity_kv_tokens": capacity_tokens,
        "required_kv_tokens": required_tokens,
        "requirement_source": requirement_source,
        "capacity_headroom_tokens": (
            capacity_tokens - required_tokens
            if capacity_tokens is not None and required_tokens is not None
            else None
        ),
        "capacity_risk": capacity_risk,
        "full_batch_seen": full_batch_seen,
        "initial_queueing_observed": initial_queueing_observed,
        "static_batch_preemption_transition": static_batch_transition,
        "transition_evidence": transition_evidence,
        "max_kv_cache_usage_percent": max_usage if samples else None,
        "scheduler_sample_count": len(samples),
        "max_waiting_requests": max_waiting if samples else None,
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="vLLM/pytest log path")
    parser.add_argument("--expected-batch-size", required=True, type=int)
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Enable Running=BS -> Running<BS/Waiting>0 transition detection.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        help="Fallback per-request length when the log has no requirement marker.",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument(
        "--tsv",
        action="store_true",
        help="Print a compact TSV summary instead of JSON.",
    )
    args = parser.parse_args()
    if args.expected_batch_size <= 0:
        parser.error("--expected-batch-size must be positive")
    if args.max_sequence_length is not None and args.max_sequence_length <= 0:
        parser.error("--max-sequence-length must be positive")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    report_path = Path(args.report).resolve()
    try:
        result = analyze(args)
    except OSError as exc:
        result = {
            "status": "ERROR",
            "decision": "BLOCKED_SCHEDULER_EVIDENCE",
            "error": str(exc),
            "log": str(Path(args.log).resolve()),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.tsv:
        reason = "; ".join(result.get("reasons", [])).replace("\t", " ")
        values = (
            result.get("status", "ERROR"),
            result.get("capacity_kv_tokens"),
            result.get("required_kv_tokens"),
            result.get("max_kv_cache_usage_percent"),
            reason,
        )
        print("\t".join("<none>" if value is None else str(value) for value in values))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))

    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
