#!/usr/bin/env python3
"""比较 MindSpeed-LLM 基线与候选训练的逐 iteration loss，并执行平均相对绝对误差门禁。"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SKILL_ROOT.parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_BASELINE_RUN = "remote-validation-0001"
DEFAULT_MAX_RELATIVE_ERROR_PERCENT = 1.0
DEFAULT_REPORT_NAME = "loss_accuracy_validation.json"
RUN_NAME_RE = re.compile(r"^remote-validation-\d{4,}$")
LOSS_RE = re.compile(
    r"\biteration\s+(\d+)\s*/\s*(\d+).*?\blm loss:\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)


class GateBlocked(ValueError):
    """输入产物不足以形成合法精度比较。"""


def parse_non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是数值") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("必须是有限的非负数")
    return parsed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--baseline-run",
        help=(
            "基线运行名称或目录；省略时使用 remote-validation-0001。"
            "名称相对 --output-dir 解析，目录路径直接解析"
        ),
    )
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument(
        "--max-relative-error-percent",
        default=DEFAULT_MAX_RELATIVE_ERROR_PERCENT,
        type=parse_non_negative_float,
        help="允许的平均相对绝对误差百分比；默认 1，即不超过 1%%",
    )
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    return parser.parse_args(argv)


def validate_run_name(value: str, field: str) -> str:
    if not RUN_NAME_RE.fullmatch(value):
        raise GateBlocked(f"{field} 必须符合 remote-validation-<至少四位数字>")
    return value


def resolve_baseline_run(value: str | None, output_root: Path) -> tuple[str, Path]:
    selector = value or DEFAULT_BASELINE_RUN
    if RUN_NAME_RE.fullmatch(selector):
        return selector, output_root / selector

    run_dir = Path(selector).expanduser().resolve()
    run_id = validate_run_name(run_dir.name, "baseline-run 目录名")
    return run_id, run_dir


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateBlocked(f"缺少文件：{path}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise GateBlocked(f"无法读取 JSON：{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateBlocked(f"JSON 顶层必须是 object：{path}")
    return payload


def require_successful_run(run_dir: Path, expected_run_id: str) -> dict[str, Any]:
    report_path = run_dir / "validation_report.json"
    report = load_json(report_path)
    if report.get("run_id") != expected_run_id:
        raise GateBlocked(
            f"报告 run_id 与目录不一致：expected {expected_run_id}, got {report.get('run_id')!r}"
        )
    training_result = report.get("training_result")
    if (
        report.get("mode") != "run"
        or report.get("stage") != "completed"
        or report.get("passed") is not True
        or not isinstance(training_result, dict)
        or training_result.get("exit_code") != 0
    ):
        raise GateBlocked(f"远端训练未成功完成：{report_path}")
    plan = report.get("plan")
    execution = plan.get("execution") if isinstance(plan, dict) else None
    script = execution.get("script") if isinstance(execution, dict) else None
    if not isinstance(script, str) or not script:
        raise GateBlocked(f"报告中缺少训练脚本路径：{report_path}")
    return {
        "run_id": expected_run_id,
        "run_dir": str(run_dir),
        "validation_report": str(report_path),
        "training_log": str(run_dir / "training.log"),
        "training_script": script,
    }


def extract_loss_series(log_path: Path) -> tuple[int, dict[int, float]]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError as exc:
        raise GateBlocked(f"缺少文件：{log_path}") from exc
    except OSError as exc:
        raise GateBlocked(f"无法读取训练日志：{log_path}: {exc}") from exc

    total_steps: int | None = None
    losses: dict[int, float] = {}
    for line in lines:
        match = LOSS_RE.search(line)
        if match is None:
            continue
        iteration = int(match.group(1))
        current_total = int(match.group(2))
        loss = float(match.group(3))
        if not math.isfinite(loss):
            raise GateBlocked(f"iteration {iteration} 的 lm loss 不是有限数值：{log_path}")
        if total_steps is None:
            total_steps = current_total
        elif total_steps != current_total:
            raise GateBlocked(f"日志中的总训练步数不一致：{log_path}")
        if iteration in losses:
            raise GateBlocked(f"日志中 iteration {iteration} 出现重复 loss：{log_path}")
        losses[iteration] = loss

    if total_steps is None or not losses:
        raise GateBlocked(f"日志中未找到 iteration/lm loss：{log_path}")
    return total_steps, losses


def compare_series(
    baseline_total: int,
    baseline_losses: dict[int, float],
    candidate_total: int,
    candidate_losses: dict[int, float],
) -> dict[str, Any]:
    compared_total = min(baseline_total, candidate_total)
    expected_iterations = set(range(1, compared_total + 1))
    baseline_iterations = {
        iteration for iteration in baseline_losses if iteration <= compared_total
    }
    candidate_iterations = {
        iteration for iteration in candidate_losses if iteration <= compared_total
    }
    if baseline_iterations != expected_iterations or candidate_iterations != expected_iterations:
        baseline_missing = sorted(expected_iterations - baseline_iterations)
        candidate_missing = sorted(expected_iterations - candidate_iterations)
        raise GateBlocked(
            "共同训练区间的 iteration 不完整："
            f"baseline missing={baseline_missing}, candidate missing={candidate_missing}"
        )

    differences = []
    for iteration in range(1, compared_total + 1):
        baseline_loss = baseline_losses[iteration]
        candidate_loss = candidate_losses[iteration]
        if baseline_loss == 0:
            raise GateBlocked(
                f"iteration {iteration} 的 baseline loss 为 0，无法计算相对误差"
            )
        relative_absolute_error = abs((baseline_loss - candidate_loss) / baseline_loss)
        differences.append(
            {
                "iteration": iteration,
                "baseline_loss": baseline_loss,
                "candidate_loss": candidate_loss,
                "relative_absolute_error": relative_absolute_error,
                "relative_absolute_error_percent": relative_absolute_error * 100,
            }
        )
    mean_relative_absolute_error = (
        sum(item["relative_absolute_error"] for item in differences) / len(differences)
    )
    max_relative_absolute_error = max(
        item["relative_absolute_error"] for item in differences
    )
    return {
        "compared_points": len(differences),
        "compared_iteration_start": 1,
        "compared_iteration_end": compared_total,
        "baseline_total_training_steps": baseline_total,
        "candidate_total_training_steps": candidate_total,
        "mean_relative_absolute_error": mean_relative_absolute_error,
        "mean_relative_absolute_error_percent": mean_relative_absolute_error * 100,
        "max_relative_absolute_error": max_relative_absolute_error,
        "max_relative_absolute_error_percent": max_relative_absolute_error * 100,
        "per_iteration": differences,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    output_root = Path(args.output_dir).expanduser().resolve()
    if Path(args.report_name).name != args.report_name or not args.report_name.endswith(".json"):
        print("--report-name 必须是不含路径分隔符的 .json 文件名", file=sys.stderr)
        return 2

    try:
        baseline_run, baseline_dir = resolve_baseline_run(args.baseline_run, output_root)
        candidate_run = validate_run_name(args.candidate_run, "candidate-run")
        candidate_dir = output_root / candidate_run
        if baseline_dir == candidate_dir:
            raise GateBlocked("候选运行不能与基线运行相同")
    except GateBlocked as exc:
        print(str(exc), file=sys.stderr)
        return 2

    report_path = candidate_dir / args.report_name
    report: dict[str, Any] = {
        "schema_version": 3,
        "workflow": "mindspeed-llm-loss-accuracy-validation",
        "status": "blocked",
        "gate_passed": False,
        "threshold": {
            "metric": "loss_mean_relative_absolute_error",
            "max_relative_error_percent": args.max_relative_error_percent,
        },
        "baseline": {
            "run_id": baseline_run,
            "run_dir": str(baseline_dir),
        },
        "candidate": {
            "run_id": candidate_run,
            "run_dir": str(candidate_dir),
        },
        "comparison": None,
        "errors": [],
        "artifacts": {"report": str(report_path)},
    }

    try:
        baseline = require_successful_run(baseline_dir, baseline_run)
        candidate = require_successful_run(candidate_dir, candidate_run)
        report["baseline"] = baseline
        report["candidate"] = candidate
        if baseline["training_script"] != candidate["training_script"]:
            raise GateBlocked(
                "训练脚本路径不一致："
                f"baseline={baseline['training_script']}, candidate={candidate['training_script']}"
            )
        baseline_total, baseline_losses = extract_loss_series(Path(baseline["training_log"]))
        candidate_total, candidate_losses = extract_loss_series(Path(candidate["training_log"]))
        comparison = compare_series(
            baseline_total,
            baseline_losses,
            candidate_total,
            candidate_losses,
        )
        report["comparison"] = comparison
    except GateBlocked as exc:
        report["errors"] = [str(exc)]
        if candidate_dir.is_dir():
            write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 3

    if (
        comparison["mean_relative_absolute_error_percent"]
        <= args.max_relative_error_percent
    ):
        report["status"] = "success"
        report["gate_passed"] = True
        exit_code = 0
    else:
        report["status"] = "failed"
        report["errors"] = [
            "loss 平均相对绝对误差 "
            f"{comparison['mean_relative_absolute_error_percent']:.12g}% "
            f"超过阈值 {args.max_relative_error_percent:.12g}%"
        ]
        exit_code = 4
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
