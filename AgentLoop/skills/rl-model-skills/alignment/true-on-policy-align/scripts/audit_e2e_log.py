#!/usr/bin/env python3
"""Strictly audit a long-horizon alignment metric and emit a first-failure trigger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, TextIO


FLOAT = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf(?:inity)?|nan)"
DEFAULT_STEP_PATTERN = r"(?<![A-Za-z0-9_])(?:global_)?step[\"']?\s*[:=/]\s*(\d+)"


class AuditError(ValueError):
    pass


def metric_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9_/])[\"']?{re.escape(name)}[\"']?\s*[:=]\s*({FLOAT})",
        re.IGNORECASE,
    )


def parse_number(raw: str) -> float:
    normalized = raw.lower().replace("infinity", "inf")
    return float(normalized)


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def missing_between(steps: list[int], limit: int = 100) -> list[int]:
    missing: list[int] = []
    for previous, current in zip(steps, steps[1:]):
        if current <= previous + 1:
            continue
        take = min(current - previous - 1, limit - len(missing))
        missing.extend(range(previous + 1, previous + 1 + take))
        if len(missing) >= limit:
            break
    return missing


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def first_failure_record(
    *,
    kind: str,
    metric: str,
    value: float,
    step: int | None,
    line_number: int,
    line: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "metric": metric,
        "value": value if math.isfinite(value) else str(value),
        "step": step,
        "line_number": line_number,
        "line_sha256": "sha256:"
        + hashlib.sha256(line.encode("utf-8", errors="replace")).hexdigest(),
        "safe_excerpt": f"{metric}={value}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def read_following(
    handle: TextIO,
    *,
    follow_pid: int | None,
    idle_timeout: float | None,
    poll_seconds: float,
) -> tuple[str | None, bool]:
    line = handle.readline()
    if line:
        return line, False
    if follow_pid is None or not process_alive(follow_pid):
        return None, True
    deadline = (
        time.monotonic() + idle_timeout
        if idle_timeout is not None and idle_timeout > 0
        else None
    )
    while True:
        time.sleep(poll_seconds)
        line = handle.readline()
        if line:
            return line, False
        if not process_alive(follow_pid):
            return None, True
        if deadline is not None and time.monotonic() >= deadline:
            raise AuditError(
                "followed process is still alive but the log heartbeat timed out; "
                "this is not treated as end-of-run"
            )


def audit(
    log_path: Path,
    *,
    zero_metrics: list[str],
    valid_metrics: list[str],
    expected_steps: int,
    expected_first_step: int | None = None,
    historical_failure_step: int | None = None,
    zero_atol: float = 0.0,
    step_pattern: str = DEFAULT_STEP_PATTERN,
    allow_identical_duplicates: bool = False,
    follow_pid: int | None = None,
    idle_timeout: float | None = None,
    poll_seconds: float = 0.25,
    trigger_path: Path | None = None,
) -> dict[str, Any]:
    if expected_steps <= 0:
        raise AuditError("expected_steps must be positive")
    if idle_timeout is not None and idle_timeout < 0:
        raise AuditError("idle_timeout must be non-negative or null")
    if poll_seconds <= 0:
        raise AuditError("poll_seconds must be positive")
    if zero_atol < 0 or not math.isfinite(zero_atol):
        raise AuditError("zero_atol must be finite and non-negative")
    if not zero_metrics:
        raise AuditError("at least one zero metric is required")
    names = zero_metrics + valid_metrics
    if len(names) != len(set(names)):
        raise AuditError("metric names must be unique")
    try:
        step_re = re.compile(step_pattern, re.IGNORECASE)
    except re.error as exc:
        raise AuditError(f"invalid step regex: {exc}") from exc
    patterns = {name: metric_pattern(name) for name in names}
    samples: dict[str, dict[int, float]] = {name: {} for name in names}
    violations: list[dict[str, Any]] = []
    first_nonzero: dict[str, Any] | None = None
    first_violation: dict[str, Any] | None = None
    current_step: int | None = None
    line_number = 0
    digest = hashlib.sha256()

    try:
        handle = log_path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AuditError(f"cannot open log {log_path}: {exc}") from exc

    with handle:
        stopped_at_required_horizon = False
        while True:
            line, ended = read_following(
                handle,
                follow_pid=follow_pid,
                idle_timeout=idle_timeout,
                poll_seconds=poll_seconds,
            )
            if line is None:
                if ended:
                    break
                continue
            line_number += 1
            digest.update(line.encode("utf-8", errors="replace"))
            step_match = step_re.search(line)
            if step_match:
                current_step = int(step_match.group(1))
            for name, pattern in patterns.items():
                for match in pattern.finditer(line):
                    value = parse_number(match.group(1))
                    if current_step is None:
                        record = first_failure_record(
                            kind="metric_without_step",
                            metric=name,
                            value=value,
                            step=None,
                            line_number=line_number,
                            line=line,
                        )
                        violations.append(record)
                        first_violation = first_violation or record
                        continue
                    previous = samples[name].get(current_step)
                    if previous is not None and not (
                        allow_identical_duplicates and previous == value
                    ):
                        record = first_failure_record(
                            kind="duplicate_metric_step",
                            metric=name,
                            value=value,
                            step=current_step,
                            line_number=line_number,
                            line=line,
                        )
                        violations.append(record)
                        first_violation = first_violation or record
                    samples[name][current_step] = value
                    expected = 0.0 if name in zero_metrics else 1.0
                    bad_kind = None
                    if not math.isfinite(value):
                        bad_kind = "nonfinite"
                    elif name in zero_metrics and abs(value) > zero_atol:
                        bad_kind = "nonzero"
                    elif name in valid_metrics and value != expected:
                        bad_kind = "invalid_validity_metric"
                    if bad_kind:
                        record = first_failure_record(
                            kind=bad_kind,
                            metric=name,
                            value=value,
                            step=current_step,
                            line_number=line_number,
                            line=line,
                        )
                        violations.append(record)
                        first_violation = first_violation or record
                        if bad_kind == "nonzero" and first_nonzero is None:
                            first_nonzero = record
                        if trigger_path is not None and not trigger_path.exists():
                            atomic_json(
                                trigger_path,
                                {
                                    "schema_version": 1,
                                    "truth_role": "first_failure_trigger",
                                    "log": str(log_path),
                                    "failure": record,
                                    "requested_capture": [
                                        "runtime state fingerprint",
                                        "weight reload fingerprint",
                                        "nearest semantic Tensor boundaries",
                                        "operator call-context fingerprint",
                                    ],
                                    "capture_contract": {
                                        "prearmed_required": True,
                                        "trigger_occurs_after_metric_log": True,
                                        "already_lost_tensor_recovery_claimed": False,
                                    },
                                },
                            )
            if follow_pid is not None:
                complete_metrics = set(samples[names[0]])
                for metric_name in names[1:]:
                    complete_metrics &= set(samples[metric_name])
                if expected_first_step is not None:
                    required_steps = set(
                        range(expected_first_step, expected_first_step + expected_steps)
                    )
                    horizon_complete = required_steps.issubset(complete_metrics)
                else:
                    ordered_complete = sorted(complete_metrics)
                    horizon_complete = (
                        len(ordered_complete) >= expected_steps
                        and not missing_between(ordered_complete, limit=1)
                    )
                historical_complete = (
                    historical_failure_step is None
                    or (
                        complete_metrics
                        and max(complete_metrics) >= historical_failure_step
                    )
                )
                if horizon_complete and historical_complete:
                    stopped_at_required_horizon = True
                    break

    common_steps = set(samples[names[0]])
    for name in names[1:]:
        common_steps &= set(samples[name])
    ordered_steps = sorted(common_steps)
    missing_steps: list[int] = []
    if expected_first_step is not None:
        required = set(range(expected_first_step, expected_first_step + expected_steps))
        missing_steps = sorted(required - common_steps)
        complete = not missing_steps
    else:
        missing_steps = missing_between(ordered_steps)
        complete = len(common_steps) >= expected_steps and not missing_steps
    max_step = max(ordered_steps) if ordered_steps else None
    crossed_historical = (
        None
        if historical_failure_step is None
        else max_step is not None and max_step >= historical_failure_step
    )
    if historical_failure_step is not None and crossed_historical is not True:
        violations.append(
            {
                "kind": "historical_failure_step_not_crossed",
                "historical_failure_step": historical_failure_step,
                "max_observed_step": max_step,
            }
        )
    if not complete:
        violations.append(
            {
                "kind": "incomplete_run",
                "expected_steps": expected_steps,
                "observed_steps": len(common_steps),
                "missing_steps": missing_steps[:100],
            }
        )

    metric_summary: dict[str, Any] = {}
    for name in names:
        values = list(samples[name].values())
        finite_values = [value for value in values if math.isfinite(value)]
        metric_summary[name] = {
            "sample_count": len(values),
            "finite_count": len(finite_values),
            "nonfinite_count": len(values) - len(finite_values),
            "nonzero_count": (
                sum(abs(value) > zero_atol for value in finite_values)
                if name in zero_metrics
                else None
            ),
            "max_abs": (
                max((abs(value) for value in finite_values), default=None)
                if name in zero_metrics
                else None
            ),
            "min": min(finite_values, default=None),
            "max": max(finite_values, default=None),
        }

    return {
        "schema_version": 1,
        "truth_role": "long_horizon_e2e_gate",
        "status": "pass" if complete and not violations else "fail",
        "complete": complete,
        "expected_steps": expected_steps,
        "observed_steps": len(common_steps),
        "first_observed_step": ordered_steps[0] if ordered_steps else None,
        "last_observed_step": max_step,
        "historical_failure_step": historical_failure_step,
        "crossed_historical_failure_step": crossed_historical,
        "zero_atol": zero_atol,
        "metrics": metric_summary,
        "first_nonzero": first_nonzero,
        "first_violation": first_violation,
        "violation_count": len(violations),
        "violations": violations[:100],
        "log_sha256": "sha256:" + digest.hexdigest(),
        "log_sha256_scope": (
            "audited_prefix_through_required_horizon"
            if stopped_at_required_horizon
            else "complete_bytes_read"
        ),
        "stopped_at_required_horizon": stopped_at_required_horizon,
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="metric that must be exactly zero; repeatable",
    )
    parser.add_argument(
        "--valid-metric",
        action="append",
        default=[],
        help="metric that must equal 1; repeatable",
    )
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-first-step", type=int)
    parser.add_argument("--historical-failure-step", type=int)
    parser.add_argument("--zero-atol", type=float, default=0.0)
    parser.add_argument("--step-regex", default=DEFAULT_STEP_PATTERN)
    parser.add_argument("--allow-identical-duplicates", action="store_true")
    parser.add_argument("--follow-pid", type=int)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        help=(
            "optional live-log heartbeat timeout in seconds; 0 disables it. "
            "Timeout is an audit error, never successful EOF"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--trigger", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = args.metrics or ["training/rollout_probs_diff_mean"]
    try:
        result = audit(
            args.log.resolve(),
            zero_metrics=metrics,
            valid_metrics=args.valid_metric,
            expected_steps=args.expected_steps,
            expected_first_step=args.expected_first_step,
            historical_failure_step=args.historical_failure_step,
            zero_atol=args.zero_atol,
            step_pattern=args.step_regex,
            allow_identical_duplicates=args.allow_identical_duplicates,
            follow_pid=args.follow_pid,
            idle_timeout=args.idle_timeout or None,
            poll_seconds=args.poll_seconds,
            trigger_path=args.trigger.resolve() if args.trigger else None,
        )
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
