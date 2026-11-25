"""整网对照指标：同工作负载、同采样窗口，逐步比较耗时与 loss。"""
from __future__ import annotations

import math
import statistics


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数值")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} 非法: {value}")
    return value


def validate_metrics(data, run_id, phase, workload_id, min_steps):
    for key, expected in (("run_id", run_id), ("phase", phase),
                          ("workload_id", workload_id)):
        if data.get(key) != expected:
            raise ValueError(f"指标 {key} 不匹配本轮: {data.get(key)!r}")
    if data.get("profiling_enabled") is not False:
        raise ValueError("整网计时必须关闭 profiler；采集单独执行一次")
    samples = data.get("samples", [])
    if len(samples) < min_steps:
        raise ValueError(f"稳态样本不足: {len(samples)} < {min_steps}")
    steps = []
    for sample in samples:
        step = sample.get("step")
        if type(step) is not int or step < 0:
            raise ValueError("step 必须是非负整数")
        steps.append(step)
        number(sample.get("step_time_ms"), "step_time_ms", positive=True)
        number(sample.get("loss"), "loss")
        if "tokens" in sample:
            number(sample["tokens"], "tokens", positive=True)
    if steps != sorted(set(steps)):
        raise ValueError("step 必须唯一且递增；多 rank 日志须先选择同一统计 rank")
    if any("tokens" in s for s in samples) and not all("tokens" in s for s in samples):
        raise ValueError("tokens 必须全部提供或全部省略")
    return data


def aggregate(data):
    samples = data["samples"]
    times = [s["step_time_ms"] for s in samples]
    return {
        "steps": len(samples), "mean_step_ms": statistics.mean(times),
        "median_step_ms": statistics.median(times),
        "tokens_per_second": (sum(s["tokens"] for s in samples) / (sum(times) / 1000)
                              if all("tokens" in s for s in samples) else None),
    }


def compare(baseline, candidate, loss_atol, loss_rtol, max_step_regression_pct,
            min_step_improvement_pct=1.0):
    threshold = number(min_step_improvement_pct, "min_step_improvement_pct", positive=True)
    if threshold >= 100:
        raise ValueError("min_step_improvement_pct 必须小于 100")
    a, b = baseline["samples"], candidate["samples"]
    if [s["step"] for s in a] != [s["step"] for s in b]:
        raise ValueError("基线与优化后的 step 窗口不一致")
    if [s.get("tokens") for s in a] != [s.get("tokens") for s in b]:
        raise ValueError("基线与优化后的每步 token 工作量不一致")
    errors = [abs(x["loss"] - y["loss"]) for x, y in zip(a, b)]
    accuracy = all(err <= loss_atol + loss_rtol * abs(x["loss"])
                   for x, err in zip(a, errors))
    base, opt = aggregate(baseline), aggregate(candidate)
    change = (opt["mean_step_ms"] / base["mean_step_ms"] - 1) * 100
    improvement = -change
    median_improvement = (1 - opt["median_step_ms"] / base["median_step_ms"]) * 100
    gain_passed = (improvement >= threshold or math.isclose(improvement, threshold, rel_tol=1e-9, abs_tol=1e-9)) and median_improvement > 0
    nonregression = change <= max_step_regression_pct
    return {
        "baseline": base, "optimized": opt,
        "step_speedup": base["mean_step_ms"] / opt["mean_step_ms"],
        "step_time_change_pct": change,
        "throughput_change_pct": ((opt["tokens_per_second"] / base["tokens_per_second"] - 1) * 100
                                  if base["tokens_per_second"] else None),
        "accuracy_passed": accuracy, "max_loss_abs_error": max(errors),
        "performance_nonregression": nonregression,
        "mean_step_improvement_pct": improvement,
        "median_step_improvement_pct": median_improvement,
        "min_step_improvement_pct": threshold,
        "e2e_gain_passed": gain_passed,
        "acceptance_reason": ("ACCURACY_FAILED" if not accuracy else
                              "PERFORMANCE_REGRESSION" if not nonregression else
                              "NO_E2E_GAIN" if not gain_passed else "E2E_GAIN"),
        "accepted": accuracy and nonregression and gain_passed,
    }
