# Gate Validation (Python engine path)
# ====================================
# Code-level hard validation of agent structured outputs.
# Mirrors workflows/gates.js (JS Workflow path) so both execution paths
# enforce identical rules — including the mandatory TP/DP/EP topology
# search checks for the tester phase.
#
# The orchestrator calls the CLI after each agent returns; on gate
# failure the failures are appended to the retry prompt.
#
# Usage:
#   python -m engine.gates validate tester --json-file result.json
#   python -m engine.gates validate tester --stdin < result.json
#
# Exit codes: 0 = passed, 2 = gate failed, 1 = usage/parse error.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GateResult:
    """Validation outcome: passed flag + human-readable failure list."""

    passed: bool
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validators (mirror workflows/gates.js)
# ---------------------------------------------------------------------------

def validate_architect(result: Any) -> GateResult:
    failures: list[str] = []
    if not isinstance(result, dict):
        return GateResult(False, ["结果不是 JSON 对象"])

    mp = result.get("model_profile")
    if not mp:
        failures.append("缺失 model_profile")
    else:
        if not mp.get("architecture"):
            failures.append("model_profile 缺少 architecture（架构类型）")
        if mp.get("num_layers") is None:
            failures.append("model_profile 缺少 num_layers（层数）")
        if mp.get("hidden_size") is None:
            failures.append("model_profile 缺少 hidden_size（隐藏层维度）")
        if not mp.get("num_parameters"):
            failures.append("model_profile 缺少 num_parameters（参数量）")

    mb = result.get("memory_budget")
    if not mb:
        failures.append("缺失 memory_budget")
    else:
        if mb.get("model_weights_gb") is None:
            failures.append("memory_budget 缺少 model_weights_gb")
        if mb.get("kv_cache_gb") is None:
            failures.append("memory_budget 缺少 kv_cache_gb")
        if mb.get("total_per_card_gb") is None:
            failures.append("memory_budget 缺少 total_per_card_gb")

    rs = result.get("recommended_strategy")
    if not rs:
        failures.append("缺失 recommended_strategy")
    else:
        if rs.get("tp") is None:
            failures.append("recommended_strategy 缺少 tp")
        if rs.get("dp") is None:
            failures.append("recommended_strategy 缺少 dp")
        if not rs.get("rationale"):
            failures.append("recommended_strategy 缺少 rationale")

    hr = result.get("hardware_requirements")
    if not hr:
        failures.append("缺失 hardware_requirements")
    else:
        if hr.get("min_npu_cards") is None:
            failures.append("hardware_requirements 缺少 min_npu_cards")
        if not hr.get("recommended_config"):
            failures.append("hardware_requirements 缺少 recommended_config")

    return GateResult(len(failures) == 0, failures)


def validate_developer(result: Any) -> GateResult:
    failures: list[str] = []
    if not isinstance(result, dict):
        return GateResult(False, ["结果不是 JSON 对象"])

    if not result.get("service_url"):
        failures.append("缺少 service_url")
    if result.get("port") is None:
        failures.append("缺少 port")
    if not result.get("model_name"):
        failures.append("缺少 model_name")
    if result.get("tp") is None:
        failures.append("缺少 tp")
    if result.get("dp") is None:
        failures.append("缺少 dp")
    if result.get("ep") is None:
        failures.append("缺少 ep（未启用专家并行时应填 1）")
    if result.get("health_check_passed") is not True:
        failures.append("/health 端点不可达或非 200")
    if result.get("models_endpoint_ok") is not True:
        failures.append("/v1/models 端点不可达或未返回目标模型")
    if result.get("service_status") != "running":
        failures.append("服务状态非 running: " + str(result.get("service_status") or "unknown"))

    return GateResult(len(failures) == 0, failures)


def validate_tester(result: Any) -> GateResult:
    failures: list[str] = []
    if not isinstance(result, dict):
        return GateResult(False, ["结果不是 JSON 对象"])

    bm = result.get("baseline_metrics")
    if not bm:
        failures.append("缺失 baseline_metrics")
    else:
        if bm.get("output_tok_per_s") is None:
            failures.append("baseline_metrics 缺少 output_tok_per_s")
        if bm.get("ttft_ms") is None:
            failures.append("baseline_metrics 缺少 ttft_ms")
        if bm.get("tpot_ms") is None:
            failures.append("baseline_metrics 缺少 tpot_ms")

    om = result.get("optimized_metrics")
    if not om:
        failures.append("缺失 optimized_metrics")
    else:
        if om.get("output_tok_per_s") is None:
            failures.append("optimized_metrics 缺少 output_tok_per_s")
        if om.get("ttft_ms") is None:
            failures.append("optimized_metrics 缺少 ttft_ms")
        if om.get("tpot_ms") is None:
            failures.append("optimized_metrics 缺少 tpot_ms")

    comparison = result.get("comparison")
    if not comparison:
        failures.append("缺失 comparison")
    elif comparison.get("output_tok_per_s_change_pct") is None:
        failures.append("comparison 缺少 output_tok_per_s_change_pct")

    op = result.get("optimal_params")
    if not op:
        failures.append("optimal_params 为空或缺失")
    else:
        if op.get("tp") is None:
            failures.append("optimal_params 缺少 tp")
        if op.get("dp") is None:
            failures.append("optimal_params 缺少 dp")
        if op.get("ep") is None:
            failures.append("optimal_params 缺少 ep（未启用专家并行时应为 1）")

    # 拓扑搜索是强制第一阶段：未完成 TP/DP/EP 搜索不得交接 OPTIMIZATION_READY
    ts = result.get("topology_search")
    if not ts:
        failures.append("缺失 topology_search（TP/DP/EP 拓扑搜索是强制第一阶段，必须完成并输出结果）")
    else:
        if ts.get("search_completed") is not True:
            failures.append("topology_search.search_completed 不为 true（TP/DP/EP 拓扑搜索未完成）")
        candidates = ts.get("candidates")
        if not candidates:
            failures.append("topology_search.candidates 为空（必须至少包含一个已测试的拓扑候选）")
        else:
            for i, c in enumerate(candidates):
                if c.get("tp") is None or c.get("dp") is None or c.get("ep") is None:
                    failures.append(f"topology_search.candidates[{i}] 缺少 tp/dp/ep")
                if c.get("output_tok_per_s") is None:
                    failures.append(f"topology_search.candidates[{i}] 缺少 output_tok_per_s")
        bt = ts.get("best_topology")
        if not bt:
            failures.append("topology_search.best_topology 缺失")
        else:
            if bt.get("tp") is None:
                failures.append("best_topology 缺少 tp")
            if bt.get("dp") is None:
                failures.append("best_topology 缺少 dp")
            if bt.get("ep") is None:
                failures.append("best_topology 缺少 ep")

    if not result.get("workload_description"):
        failures.append("缺少 workload_description（固定 workload 声明，必须与 profiling 采集一致）")
    if not result.get("comparison_report"):
        failures.append("缺失 comparison_report")

    return GateResult(len(failures) == 0, failures)


def validate_reviewer(result: Any, tester_workload: str | None = None) -> GateResult:
    """Structural reviewer gate (mirrors validateReviewer in gates.js).

    Args:
        tester_workload: tester 的 workload_description（跨文件一致性对比，可选）。
    """
    failures: list[str] = []
    if not isinstance(result, dict):
        return GateResult(False, ["结果不是 JSON 对象"])

    if not result.get("profiling_data_path"):
        failures.append("缺少 profiling_data_path")
    if result.get("profiling_collected") is not True:
        failures.append("profiling 数据未成功采集")
    if not result.get("workload_description"):
        failures.append("缺少 workload_description（实际采集负载声明，必须与 tester 固定 workload 一致）")
    elif tester_workload and result["workload_description"] != tester_workload:
        failures.append(
            f"采集负载与 tester 固定 workload 不一致：reviewer=\"{result['workload_description']}\" vs tester=\"{tester_workload}\""
        )

    # 回采强制：profiling_round >= 2 时必须 resampled=true 且有上轮对比
    pr_raw = result.get("profiling_round")
    try:
        pr = None if pr_raw is None else int(pr_raw)
    except (TypeError, ValueError):
        pr = None
    if pr is None:
        failures.append("缺少 profiling_round（1=首次采集，2+=回采轮次）")
    elif pr >= 2:
        if result.get("resampled") is not True:
            failures.append(f"profiling_round={pr}（再入轮次）必须 resampled=true（重新采集），禁止用旧数据收尾")
        pc = result.get("previous_round_comparison")
        if not pc or not pc.get("suggestion_verification"):
            failures.append("再入轮次缺少 previous_round_comparison（上轮 vs 本轮对比 + 建议验证结论）")

    d = result.get("diagnosis")
    if not d:
        failures.append("缺失 diagnosis")
    else:
        if d.get("e2e_time_ms") is None:
            failures.append("diagnosis 缺少 e2e_time_ms")
        if d.get("compute_pct") is None:
            failures.append("diagnosis 缺少 compute_pct")
        if d.get("communication_pct") is None:
            failures.append("diagnosis 缺少 communication_pct")
        if d.get("idle_pct") is None:
            failures.append("diagnosis 缺少 idle_pct")

    bc = result.get("bottleneck_classification")
    if not bc:
        failures.append("缺失 bottleneck_classification")
    else:
        if not bc.get("primary"):
            failures.append("bottleneck_classification 缺少 primary")
        if not bc.get("details"):
            failures.append("bottleneck_classification 缺少 details")

    if not result.get("operator_top_n"):
        failures.append("operator_top_n 为空或缺失")

    if not result.get("optimization_suggestions"):
        failures.append("optimization_suggestions 为空或缺失")

    if not result.get("diagnosis_report"):
        failures.append("缺失 diagnosis_report")

    return GateResult(len(failures) == 0, failures)


def validate_reviewer_evidence(result: Any) -> GateResult:
    """Evidence gate: every quantified conclusion needs data path + raw value."""
    failures: list[str] = []
    evidence = result.get("evidence_list") if isinstance(result, dict) else None
    if not evidence:
        failures.append("evidence_list 为空（每条量化结论需提供数据路径、表名/字段名、原始数值）")
        return GateResult(False, failures)

    for ev in evidence:
        cid = ev.get("conclusion_id")
        if not cid:
            failures.append("evidence 条目缺少 conclusion_id")
        if not ev.get("data_path"):
            failures.append(f"evidence[{cid or '?'}] 缺少 data_path")
        if not ev.get("raw_value"):
            failures.append(f"evidence[{cid or '?'}] 缺少 raw_value")
        if not ev.get("verification_status"):
            failures.append(f"evidence[{cid or '?'}] 缺少 verification_status")

    return GateResult(len(failures) == 0, failures)


def validate_reviewer_composite(result: Any, tester_workload: str | None = None) -> GateResult:
    """Reviewer gate = structural checks, then evidence checks."""
    structural = validate_reviewer(result, tester_workload=tester_workload)
    if not structural.passed:
        return structural
    return validate_reviewer_evidence(result)


def validate_optimizer(result: Any) -> GateResult:
    """Optimizer gate: every attempted suggestion needs plan + verification."""
    failures: list[str] = []
    if not isinstance(result, dict):
        return GateResult(False, ["结果不是 JSON 对象"])

    solutions = result.get("solutions")
    if not solutions:
        failures.append('solutions 为空（至少一条建议必须完成"方案+验证"闭环）')
    else:
        for i, s in enumerate(solutions):
            if not s.get("suggestion_id"):
                failures.append(f"solutions[{i}] 缺少 suggestion_id")
            if not s.get("approach"):
                failures.append(f"solutions[{i}] 缺少 approach（方案思路）")
            if not s.get("test_result"):
                failures.append(f"solutions[{i}] 缺少 test_result（验证结果）")

    return GateResult(len(failures) == 0, failures)


# Composite gate map: state → validator function
GATE_MAP: dict[str, Callable[..., GateResult]] = {
    "architect": validate_architect,
    "developer": validate_developer,
    "tester": validate_tester,
    "reviewer": validate_reviewer_composite,
    "optimizer": validate_optimizer,
}


def validate_state(state: str, result: Any, tester_workload: str | None = None) -> GateResult:
    """Run the gate validator for `state` against a parsed result object."""
    validator = GATE_MAP.get(state)
    if validator is None:
        return GateResult(False, [f"未知状态 '{state}'，可选: " + ", ".join(sorted(GATE_MAP))])
    if state == "reviewer":
        return validator(result, tester_workload=tester_workload)
    return validator(result)


def _load_result(args: argparse.Namespace) -> tuple[Any | None, str | None]:
    """Load result JSON from --json-file or --stdin.

    Uses utf-8-sig decoding so PowerShell/cmd piping (which may prepend a
    UTF-8 BOM) does not break parsing.
    """
    if args.json_file:
        try:
            with open(args.json_file, encoding="utf-8-sig") as f:
                return json.load(f), None
        except json.JSONDecodeError as e:
            return None, f"JSON 解析失败: {e}"
        except OSError as e:
            return None, f"读取文件失败: {e}"
    if args.stdin:
        try:
            data = sys.stdin.buffer.read().decode("utf-8-sig")
            return json.loads(data), None
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return None, f"stdin JSON 解析失败: {e}"
    return None, "必须提供 --json-file 或 --stdin 输入"


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="engine.gates — 各阶段 Agent 结构化输出的代码级硬校验"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="校验某个阶段的 Agent 输出")
    p_validate.add_argument("state", choices=sorted(GATE_MAP), help="阶段名")
    p_validate.add_argument("--json-file", type=str, help="结果 JSON 文件路径")
    p_validate.add_argument("--stdin", action="store_true", help="从 stdin 读取结果 JSON")
    p_validate.add_argument("--tester-workload", type=str, default=None,
                            help="tester 的 workload_description（reviewer 门禁跨文件一致性对比；可选）")

    args = ap.parse_args()

    if args.cmd == "validate":
        result, err = _load_result(args)
        if err is not None:
            print(err, file=sys.stderr)
            sys.exit(1)

        gate = validate_state(args.state, result, tester_workload=args.tester_workload)
        if gate.passed:
            print(f"gate {args.state}: ✅ 门禁校验通过")
            sys.exit(0)

        print(f"gate {args.state}: ❌ 门禁校验失败（{len(gate.failures)} 项）:")
        for f_ in gate.failures:
            print("  - " + f_)
        sys.exit(2)


if __name__ == "__main__":
    _main()
