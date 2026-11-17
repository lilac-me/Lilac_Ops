#!/usr/bin/env python3
"""Compare adjacent MindSpeed-LLM fusion experiments and apply a performance gate."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ITERATION_RE = re.compile(r"\biteration\s+(\d+)\s*/\s*(\d+)")
TIME_RE = re.compile(
    r"\belapsed time per iteration \(ms\):\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)
ALLOCATED_RE = re.compile(
    r"\bmax_memory_allocated\(GB\):\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)
RESERVED_RE = re.compile(
    r"\bmax_memory_reserved\(GB\):\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)


class ComparisonBlocked(ValueError):
    """Inputs cannot form a valid adjacent-run comparison."""


def finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--warmup-iterations", type=non_negative_int, default=1)
    parser.add_argument("--min-speedup-percent", required=True, type=finite_float)
    parser.add_argument("--report-name", default="fused_op_performance_validation.json")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ComparisonBlocked(f"missing file: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonBlocked(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonBlocked(f"JSON root must be an object: {path}")
    return value


def validate_run(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "validation_report.json"
    log_path = run_dir / "training.log"
    report = read_json(report_path)
    training_result = report.get("training_result")
    if (
        report.get("mode") != "run"
        or report.get("stage") != "completed"
        or report.get("passed") is not True
        or not isinstance(training_result, dict)
        or training_result.get("exit_code") != 0
    ):
        raise ComparisonBlocked(f"remote training did not complete successfully: {report_path}")

    plan = report.get("plan")
    execution = plan.get("execution") if isinstance(plan, dict) else None
    training_script = execution.get("script") if isinstance(execution, dict) else None
    if not isinstance(training_script, str) or not training_script:
        raise ComparisonBlocked(f"training script is missing from: {report_path}")
    if not log_path.is_file():
        raise ComparisonBlocked(f"missing file: {log_path}")
    return {
        "run_id": report.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "validation_report": str(report_path),
        "training_log": str(log_path),
        "training_script": training_script,
    }


def extract_number(pattern: re.Pattern[str], line: str, field: str, iteration: int) -> float:
    match = pattern.search(line)
    if match is None:
        raise ComparisonBlocked(f"iteration {iteration} is missing {field}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ComparisonBlocked(f"iteration {iteration} has non-finite {field}")
    return value


def parse_log(path: Path) -> tuple[int, dict[int, dict[str, float]]]:
    records: dict[int, dict[str, float]] = {}
    total_iterations: int | None = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ComparisonBlocked(f"cannot read log {path}: {exc}") from exc

    for line in lines:
        match = ITERATION_RE.search(line)
        if match is None:
            continue
        iteration = int(match.group(1))
        current_total = int(match.group(2))
        if total_iterations is None:
            total_iterations = current_total
        elif total_iterations != current_total:
            raise ComparisonBlocked(f"inconsistent total iterations in {path}")
        if iteration in records:
            raise ComparisonBlocked(f"duplicate iteration {iteration} in {path}")
        records[iteration] = {
            "elapsed_time_ms": extract_number(TIME_RE, line, "elapsed time", iteration),
            "max_memory_allocated_gb": extract_number(
                ALLOCATED_RE, line, "max allocated memory", iteration
            ),
            "max_memory_reserved_gb": extract_number(
                RESERVED_RE, line, "max reserved memory", iteration
            ),
        }

    if total_iterations is None or not records:
        raise ComparisonBlocked(f"no iteration metrics found in {path}")
    return total_iterations, records


def summarize(
    records: dict[int, dict[str, float]], measured_iterations: list[int]
) -> dict[str, float | int]:
    times = [records[index]["elapsed_time_ms"] for index in measured_iterations]
    return {
        "measured_iterations": len(measured_iterations),
        "mean_step_time_ms": statistics.fmean(times),
        "median_step_time_ms": statistics.median(times),
        "max_memory_allocated_gb": max(
            records[index]["max_memory_allocated_gb"] for index in measured_iterations
        ),
        "max_memory_reserved_gb": max(
            records[index]["max_memory_reserved_gb"] for index in measured_iterations
        ),
    }


def compare(
    baseline_total: int,
    baseline_records: dict[int, dict[str, float]],
    candidate_total: int,
    candidate_records: dict[int, dict[str, float]],
    warmup_iterations: int,
) -> dict[str, Any]:
    if baseline_total != candidate_total:
        raise ComparisonBlocked(
            f"total iterations differ: baseline={baseline_total}, candidate={candidate_total}"
        )
    baseline_set = set(baseline_records)
    candidate_set = set(candidate_records)
    if baseline_set != candidate_set:
        raise ComparisonBlocked(
            "iteration sets differ: "
            f"candidate missing={sorted(baseline_set - candidate_set)}, "
            f"unexpected={sorted(candidate_set - baseline_set)}"
        )

    iterations = sorted(baseline_set)
    measured_iterations = iterations[warmup_iterations:]
    if not measured_iterations:
        raise ComparisonBlocked("warmup removes every recorded iteration")

    baseline_summary = summarize(baseline_records, measured_iterations)
    candidate_summary = summarize(candidate_records, measured_iterations)
    baseline_mean = float(baseline_summary["mean_step_time_ms"])
    candidate_mean = float(candidate_summary["mean_step_time_ms"])
    if baseline_mean <= 0 or candidate_mean <= 0:
        raise ComparisonBlocked("mean step time must be positive")

    return {
        "total_iterations": baseline_total,
        "compared_iterations": iterations,
        "warmup_iterations_excluded": iterations[:warmup_iterations],
        "baseline_metrics": baseline_summary,
        "candidate_metrics": candidate_summary,
        "delta": {
            "mean_step_time_ms": candidate_mean - baseline_mean,
            "speedup_percent": (baseline_mean - candidate_mean) / baseline_mean * 100,
            "max_memory_allocated_gb": (
                float(candidate_summary["max_memory_allocated_gb"])
                - float(baseline_summary["max_memory_allocated_gb"])
            ),
            "max_memory_reserved_gb": (
                float(candidate_summary["max_memory_reserved_gb"])
                - float(baseline_summary["max_memory_reserved_gb"])
            ),
        },
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    baseline_dir = args.baseline_run.expanduser().resolve()
    candidate_dir = args.candidate_run.expanduser().resolve()
    if baseline_dir == candidate_dir:
        print("baseline and candidate must be different directories", file=sys.stderr)
        return 2
    if Path(args.report_name).name != args.report_name or not args.report_name.endswith(".json"):
        print("--report-name must be a .json file name without path separators", file=sys.stderr)
        return 2
    if not args.operator.strip():
        print("--operator must not be empty", file=sys.stderr)
        return 2

    report_path = candidate_dir / args.report_name
    report: dict[str, Any] = {
        "schema_version": 2,
        "workflow": "mindspeed-llm-fsdp2-fused-op-integration",
        "operator": args.operator,
        "status": "blocked",
        "gate_passed": False,
        "thresholds": {
            "min_speedup_percent": args.min_speedup_percent,
        },
        "baseline": {"run_dir": str(baseline_dir)},
        "candidate": {"run_dir": str(candidate_dir)},
        "comparison": None,
        "errors": [],
        "artifacts": {"report": str(report_path)},
    }

    try:
        baseline = validate_run(baseline_dir)
        candidate = validate_run(candidate_dir)
        report["baseline"] = baseline
        report["candidate"] = candidate
        if baseline["training_script"] != candidate["training_script"]:
            raise ComparisonBlocked(
                "training scripts differ: "
                f"baseline={baseline['training_script']}, "
                f"candidate={candidate['training_script']}"
            )
        baseline_total, baseline_records = parse_log(Path(baseline["training_log"]))
        candidate_total, candidate_records = parse_log(Path(candidate["training_log"]))
        comparison = compare(
            baseline_total,
            baseline_records,
            candidate_total,
            candidate_records,
            args.warmup_iterations,
        )
        report["comparison"] = comparison
    except ComparisonBlocked as exc:
        report["errors"] = [str(exc)]
        if candidate_dir.is_dir():
            write_report(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 3

    performance_passed = (
        comparison["delta"]["speedup_percent"] >= args.min_speedup_percent
    )
    if performance_passed:
        report["status"] = "success"
        report["gate_passed"] = True
        exit_code = 0
    else:
        report["status"] = "failed"
        report["errors"].append(
            f"speedup {comparison['delta']['speedup_percent']:.12g}% is below "
            f"{args.min_speedup_percent:.12g}%"
        )
        exit_code = 4

    write_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
