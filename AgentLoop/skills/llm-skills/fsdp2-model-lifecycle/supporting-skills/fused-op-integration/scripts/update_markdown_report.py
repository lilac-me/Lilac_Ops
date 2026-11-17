#!/usr/bin/env python3
"""Create or append one row to the standard fused-op Markdown report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPORT_NAME = "fused_op_integration_report.md"
HEADER = """# FSDP2 融合算子接入报告

| 步骤 | 模型 | 融合算子 | 修改说明 | 修改文件 | 运行对比 | 单步耗时（基线 → 候选） | 性能变化 | Loss 精度 | 结构/语义/Logits | 结论 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""


class ReportInputError(ValueError):
    """Input cannot produce a valid report row."""


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--step", required=True, type=positive_int)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--change-description", required=True)
    parser.add_argument("--modified-file", action="append", default=[])
    parser.add_argument("--performance-report", required=True, type=Path)
    parser.add_argument("--accuracy-report", required=True, type=Path)
    parser.add_argument("--correctness-report", required=True, type=Path)
    return parser.parse_args(argv)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportInputError(f"missing file: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportInputError(f"JSON root must be an object: {path}")
    return value


def resolve_report_dir(args: argparse.Namespace) -> Path:
    if args.report_dir is not None:
        return args.report_dir.expanduser().resolve()
    if args.config is None:
        raise ReportInputError("provide --report-dir or --config")
    config_path = args.config.expanduser().resolve()
    config = read_object(config_path)
    report_dir = config.get("report_dir")
    if not isinstance(report_dir, str) or not report_dir.strip():
        raise ReportInputError(f"report_dir is missing from: {config_path}")
    configured = Path(report_dir).expanduser()
    if not configured.is_absolute():
        configured = config_path.parent / configured
    return configured.resolve()


def finite_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReportInputError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ReportInputError(f"{field} must be finite")
    return parsed


def markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace(
        "\n", "<br>"
    )


def run_name(run: Any) -> str:
    if not isinstance(run, dict):
        return "—"
    run_id = run.get("run_id")
    if isinstance(run_id, str) and run_id:
        return run_id
    run_dir = run.get("run_dir")
    if isinstance(run_dir, str) and run_dir:
        return Path(run_dir).name
    return "—"


def report_status(report: dict[str, Any], report_name: str) -> str:
    status = report.get("status")
    if status not in ("success", "failed", "blocked"):
        raise ReportInputError(f"{report_name} has invalid status")
    return status


def report_errors(report: dict[str, Any]) -> str:
    errors = report.get("errors")
    return "; ".join(str(item) for item in errors) if isinstance(errors, list) else ""


def build_performance_metrics(report: dict[str, Any]) -> tuple[str, str, str]:
    status = report_status(report, "performance report")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        if status != "blocked":
            raise ReportInputError("performance report is missing comparison metrics")
        return "—", "—", status

    baseline = comparison.get("baseline_metrics")
    candidate = comparison.get("candidate_metrics")
    delta = comparison.get("delta")
    if not all(isinstance(item, dict) for item in (baseline, candidate, delta)):
        raise ReportInputError("performance report has incomplete comparison metrics")

    baseline_time = finite_number(baseline.get("mean_step_time_ms"), "baseline mean step time")
    candidate_time = finite_number(candidate.get("mean_step_time_ms"), "candidate mean step time")
    speedup = finite_number(delta.get("speedup_percent"), "speedup percent")

    step_time = f"{baseline_time:.2f} → {candidate_time:.2f} ms"
    if speedup >= 0:
        performance = f"+{speedup:.2f}%（加速）"
    else:
        performance = f"{speedup:.2f}%（回退）"
    return step_time, performance, status


def build_accuracy_metrics(report: dict[str, Any]) -> tuple[str, str]:
    if report.get("workflow") != "mindspeed-llm-loss-accuracy-validation":
        raise ReportInputError("accuracy report has unexpected workflow")
    status = report_status(report, "accuracy report")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        if status != "blocked":
            raise ReportInputError("accuracy report is missing comparison metrics")
        error_text = report_errors(report)
        accuracy = f"blocked：{error_text}" if error_text else "blocked"
        return accuracy, status

    threshold = report.get("threshold")
    if not isinstance(threshold, dict):
        raise ReportInputError("accuracy report is missing threshold")
    mean_error = finite_number(
        comparison.get("mean_relative_absolute_error_percent"),
        "mean relative absolute error percent",
    )
    max_error = finite_number(
        comparison.get("max_relative_absolute_error_percent"),
        "max relative absolute error percent",
    )
    max_allowed = finite_number(
        threshold.get("max_relative_error_percent"),
        "max relative error percent",
    )
    accuracy = (
        f"MeanRelAbs={mean_error:.6g}%；MaxRelAbs={max_error:.6g}%；"
        f"门限={max_allowed:.6g}%；{'通过' if status == 'success' else '失败'}"
    )
    return accuracy, status


def build_correctness_metrics(report: dict[str, Any]) -> tuple[str, str]:
    if report.get("workflow") != "mindspeed-llm-fsdp2-fusion-correctness":
        raise ReportInputError("correctness report has unexpected workflow")
    status = report_status(report, "correctness report")
    gates = report.get("gates")
    if not isinstance(gates, dict):
        raise ReportInputError("correctness report is missing gates")
    required = (
        "requested_active",
        "structure_signature",
        "semantic_probes",
        "logits_precision",
    )
    values: list[str] = []
    for name in required:
        value = gates.get(name)
        if value not in ("passed", "failed", "blocked"):
            raise ReportInputError(f"correctness gate {name} has invalid status")
        values.append(f"{name}={value}")
    derived = "blocked" if "blocked" in gates.values() else (
        "success" if all(value == "passed" for value in gates.values()) else "failed"
    )
    if status != derived:
        raise ReportInputError("correctness report status does not match gates")
    return "; ".join(values), status


def existing_steps(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    steps: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        first_cell = line.split("|", 2)[1].strip()
        if first_cell.isdigit():
            steps.add(int(first_cell))
    return steps


def append_row(path: Path, row: list[str], step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if step in existing_steps(path):
        raise ReportInputError(f"step {step} already exists in: {path}")
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(markdown_cell(cell) for cell in row) + " |\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        report_dir = resolve_report_dir(args)
        performance_path = args.performance_report.expanduser().resolve()
        accuracy_path = args.accuracy_report.expanduser().resolve()
        correctness_path = args.correctness_report.expanduser().resolve()
        performance_report = read_object(performance_path)
        accuracy_report = read_object(accuracy_path)
        correctness_report = read_object(correctness_path)
        step_time, performance, performance_status = build_performance_metrics(
            performance_report
        )
        accuracy, accuracy_status = build_accuracy_metrics(accuracy_report)
        correctness, correctness_status = build_correctness_metrics(correctness_report)
        if "blocked" in (performance_status, accuracy_status, correctness_status):
            derived_decision = "blocked"
        elif all(
            status == "success"
            for status in (performance_status, accuracy_status, correctness_status)
        ):
            derived_decision = "accepted"
        else:
            derived_decision = "rejected"
        files = "<br>".join(f"`{path}`" for path in args.modified_file) or "—"
        runs = (
            f"{run_name(performance_report.get('baseline'))} → "
            f"{run_name(performance_report.get('candidate'))}"
        )
        report_path = report_dir / REPORT_NAME
        append_row(
            report_path,
            [
                str(args.step),
                args.model,
                args.operator,
                args.change_description,
                files,
                runs,
                step_time,
                performance,
                accuracy,
                correctness,
                derived_decision,
            ],
            args.step,
        )
    except ReportInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
