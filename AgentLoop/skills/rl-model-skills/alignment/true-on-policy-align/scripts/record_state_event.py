#!/usr/bin/env python3
"""Atomically start or finish a timed event in a validated alignment state card."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import validate_state_card


class EventError(ValueError):
    pass


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EventError(f"invalid ISO timestamp: {value}") from exc


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def start_event(
    card: dict[str, Any],
    *,
    event_id: str,
    event: str,
    module: int | None,
    run_id: str | None,
    evidence: list[str],
    next_gate: str | None,
) -> dict[str, Any]:
    if any(item.get("event_id") == event_id for item in card["history"] if isinstance(item, dict)):
        raise EventError(f"event_id already exists: {event_id}")
    started_at = timestamp()
    entry = {
        "event_id": event_id,
        "event": event,
        "status": "running",
        "module": module,
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": None,
        "duration_seconds": None,
        "evidence": evidence,
    }
    card["history"].append(entry)
    card["updated_at"] = started_at
    if module is not None:
        card["state"]["active_module"] = module
    if next_gate:
        card["state"]["next_gate"] = next_gate
    return entry


def finish_event(
    card: dict[str, Any],
    *,
    event_id: str,
    status: str,
    evidence: list[str],
    next_gate: str | None,
) -> dict[str, Any]:
    matches = [
        item
        for item in card["history"]
        if isinstance(item, dict) and item.get("event_id") == event_id
    ]
    if len(matches) != 1:
        raise EventError(f"expected exactly one event_id: {event_id}")
    entry = matches[0]
    if entry.get("status") != "running" or entry.get("ended_at") is not None:
        raise EventError(f"event is not running: {event_id}")
    ended_at = timestamp()
    duration = (parse_timestamp(ended_at) - parse_timestamp(entry["started_at"])).total_seconds()
    combined_evidence = list(dict.fromkeys([*entry.get("evidence", []), *evidence]))
    entry.update(
        status=status,
        ended_at=ended_at,
        duration_seconds=max(0.0, duration),
        evidence=combined_evidence,
    )
    card["updated_at"] = ended_at
    if next_gate:
        card["state"]["next_gate"] = next_gate
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--state", type=Path, required=True)
    start.add_argument("--event-id", required=True)
    start.add_argument("--event", required=True)
    start.add_argument("--module", type=int)
    start.add_argument("--run-id")
    start.add_argument("--evidence", action="append", default=[])
    start.add_argument("--next-gate")
    finish = subparsers.add_parser("finish")
    finish.add_argument("--state", type=Path, required=True)
    finish.add_argument("--event-id", required=True)
    finish.add_argument("--status", choices=("pass", "fail", "partial", "aborted"), required=True)
    finish.add_argument("--evidence", action="append", default=[])
    finish.add_argument("--next-gate")
    args = parser.parse_args()
    state_path = args.state.resolve()
    try:
        card = validate_state_card.load_json(state_path)
        validate_state_card.validate(card)
        if args.action == "start":
            entry = start_event(
                card,
                event_id=args.event_id,
                event=args.event,
                module=args.module,
                run_id=args.run_id,
                evidence=args.evidence,
                next_gate=args.next_gate,
            )
        else:
            entry = finish_event(
                card,
                event_id=args.event_id,
                status=args.status,
                evidence=args.evidence,
                next_gate=args.next_gate,
            )
        validate_state_card.validate(card)
        atomic_write(state_path, card)
    except (OSError, EventError, validate_state_card.StateCardError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"valid": True, "event": entry}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
