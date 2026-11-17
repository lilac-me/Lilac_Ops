// Gate validation functions
// ==========================
// Extracted from optimize-pipeline.workflow.js.
// Each validator checks that the agent's structured output has all required fields.
// Returns { passed: bool, failures: string[] }.

import { ARCHITECT_SCHEMA, DEVELOPER_SCHEMA, TESTER_SCHEMA, REVIEWER_SCHEMA } from './schemas.js';

export function validateArchitect(result) {
  const failures = [];
  if (!result.model_profile) {
    failures.push('缺失 model_profile');
  } else {
    const mp = result.model_profile;
    if (!mp.architecture) failures.push('model_profile 缺少 architecture（架构类型）');
    if (!mp.num_layers) failures.push('model_profile 缺少 num_layers（层数）');
    if (!mp.hidden_size) failures.push('model_profile 缺少 hidden_size（隐藏层维度）');
    if (!mp.num_parameters) failures.push('model_profile 缺少 num_parameters（参数量）');
  }
  if (!result.memory_budget) {
    failures.push('缺失 memory_budget');
  } else {
    const mb = result.memory_budget;
    if (mb.model_weights_gb == null) failures.push('memory_budget 缺少 model_weights_gb');
    if (mb.kv_cache_gb == null) failures.push('memory_budget 缺少 kv_cache_gb');
    if (mb.total_per_card_gb == null) failures.push('memory_budget 缺少 total_per_card_gb');
  }
  if (!result.recommended_strategy) {
    failures.push('缺失 recommended_strategy');
  } else {
    const rs = result.recommended_strategy;
    if (rs.tp == null) failures.push('recommended_strategy 缺少 tp');
    if (rs.dp == null) failures.push('recommended_strategy 缺少 dp');
    if (!rs.rationale) failures.push('recommended_strategy 缺少 rationale');
  }
  if (!result.hardware_requirements) {
    failures.push('缺失 hardware_requirements');
  } else {
    const hr = result.hardware_requirements;
    if (hr.min_npu_cards == null) failures.push('hardware_requirements 缺少 min_npu_cards');
    if (!hr.recommended_config) failures.push('hardware_requirements 缺少 recommended_config');
  }
  return { passed: failures.length === 0, failures };
}

export function validateDeveloper(result) {
  const failures = [];
  if (!result.service_url) failures.push('缺少 service_url');
  if (result.port == null) failures.push('缺少 port');
  if (!result.model_name) failures.push('缺少 model_name');
  if (result.tp == null) failures.push('缺少 tp');
  if (result.dp == null) failures.push('缺少 dp');
  if (result.ep == null) failures.push('缺少 ep（未启用专家并行时应填 1）');
  if (result.health_check_passed !== true) failures.push('/health 端点不可达或非 200');
  if (result.models_endpoint_ok !== true) failures.push('/v1/models 端点不可达或未返回目标模型');
  if (result.service_status !== 'running') failures.push('服务状态非 running: ' + (result.service_status || 'unknown'));
  return { passed: failures.length === 0, failures };
}

export function validateTester(result) {
  const failures = [];
  if (!result.baseline_metrics) {
    failures.push('缺失 baseline_metrics');
  } else {
    const bm = result.baseline_metrics;
    if (bm.output_tok_per_s == null) failures.push('baseline_metrics 缺少 output_tok_per_s');
    if (bm.ttft_ms == null) failures.push('baseline_metrics 缺少 ttft_ms');
    if (bm.tpot_ms == null) failures.push('baseline_metrics 缺少 tpot_ms');
  }
  if (!result.optimized_metrics) {
    failures.push('缺失 optimized_metrics');
  } else {
    const om = result.optimized_metrics;
    if (om.output_tok_per_s == null) failures.push('optimized_metrics 缺少 output_tok_per_s');
    if (om.ttft_ms == null) failures.push('optimized_metrics 缺少 ttft_ms');
    if (om.tpot_ms == null) failures.push('optimized_metrics 缺少 tpot_ms');
  }
  if (!result.comparison) {
    failures.push('缺失 comparison');
  } else {
    if (result.comparison.output_tok_per_s_change_pct == null) {
      failures.push('comparison 缺少 output_tok_per_s_change_pct');
    }
  }
  if (!result.optimal_params || Object.keys(result.optimal_params).length === 0) {
    failures.push('optimal_params 为空或缺失');
  } else {
    if (result.optimal_params.tp == null) failures.push('optimal_params 缺少 tp');
    if (result.optimal_params.dp == null) failures.push('optimal_params 缺少 dp');
    if (result.optimal_params.ep == null) failures.push('optimal_params 缺少 ep（未启用专家并行时应为 1）');
  }
  // 拓扑搜索是强制第一阶段：未完成 TP/DP/EP 搜索不得交接 OPTIMIZATION_READY
  if (!result.topology_search) {
    failures.push('缺失 topology_search（TP/DP/EP 拓扑搜索是强制第一阶段，必须完成并输出结果）');
  } else {
    const ts = result.topology_search;
    if (ts.search_completed !== true) {
      failures.push('topology_search.search_completed 不为 true（TP/DP/EP 拓扑搜索未完成）');
    }
    if (!ts.candidates || ts.candidates.length === 0) {
      failures.push('topology_search.candidates 为空（必须至少包含一个已测试的拓扑候选）');
    } else {
      ts.candidates.forEach((c, i) => {
        if (c.tp == null || c.dp == null || c.ep == null) {
          failures.push(`topology_search.candidates[${i}] 缺少 tp/dp/ep`);
        }
        if (c.output_tok_per_s == null) {
          failures.push(`topology_search.candidates[${i}] 缺少 output_tok_per_s`);
        }
      });
    }
    if (!ts.best_topology) {
      failures.push('topology_search.best_topology 缺失');
    } else {
      if (ts.best_topology.tp == null) failures.push('best_topology 缺少 tp');
      if (ts.best_topology.dp == null) failures.push('best_topology 缺少 dp');
      if (ts.best_topology.ep == null) failures.push('best_topology 缺少 ep');
    }
  }
  if (!result.workload_description) failures.push('缺少 workload_description（固定 workload 声明，必须与 profiling 采集一致）');
  if (!result.comparison_report) failures.push('缺失 comparison_report');
  return { passed: failures.length === 0, failures };
}

export function validateReviewer(result, ctx = {}) {
  const failures = [];
  const testerWorkload = ctx.testerWorkload; // tester 的 workload_description（可选传入）
  if (!result.profiling_data_path) failures.push('缺少 profiling_data_path');
  if (result.profiling_collected !== true) failures.push('profiling 数据未成功采集');
  if (!result.workload_description) {
    failures.push('缺少 workload_description（实际采集负载声明，必须与 tester 固定 workload 一致）');
  } else if (testerWorkload && result.workload_description !== testerWorkload) {
    failures.push(`采集负载与 tester 固定 workload 不一致：reviewer="${result.workload_description}" vs tester="${testerWorkload}"`);
  }
  // 回采强制：profiling_round >= 2 时必须 resampled=true 且有上轮对比
  const roundRaw = result.profiling_round;
  const round = (roundRaw == null || roundRaw === '') ? NaN : Number(roundRaw);
  if (isNaN(round)) {
    failures.push('缺少 profiling_round（1=首次采集，2+=回采轮次）');
  } else if (round >= 2) {
    if (result.resampled !== true) {
      failures.push(`profiling_round=${round}（再入轮次）必须 resampled=true（重新采集），禁止用旧数据收尾`);
    }
    if (!result.previous_round_comparison || !result.previous_round_comparison.suggestion_verification) {
      failures.push('再入轮次缺少 previous_round_comparison（上轮 vs 本轮对比 + 建议验证结论）');
    }
  }
  if (!result.diagnosis) {
    failures.push('缺失 diagnosis');
  } else {
    const d = result.diagnosis;
    if (d.e2e_time_ms == null) failures.push('diagnosis 缺少 e2e_time_ms');
    if (d.compute_pct == null) failures.push('diagnosis 缺少 compute_pct');
    if (d.communication_pct == null) failures.push('diagnosis 缺少 communication_pct');
    if (d.idle_pct == null) failures.push('diagnosis 缺少 idle_pct');
  }
  if (!result.bottleneck_classification) {
    failures.push('缺失 bottleneck_classification');
  } else {
    if (!result.bottleneck_classification.primary) failures.push('bottleneck_classification 缺少 primary');
    if (!result.bottleneck_classification.details) failures.push('bottleneck_classification 缺少 details');
  }
  if (!result.operator_top_n || result.operator_top_n.length === 0) {
    failures.push('operator_top_n 为空或缺失');
  }
  if (!result.optimization_suggestions || result.optimization_suggestions.length === 0) {
    failures.push('optimization_suggestions 为空或缺失');
  }
  if (!result.diagnosis_report) failures.push('缺失 diagnosis_report');
  return { passed: failures.length === 0, failures };
}

export function validateReviewerEvidence(result) {
  const failures = [];
  if (!result.evidence_list || result.evidence_list.length === 0) {
    failures.push('evidence_list 为空（每条量化结论需提供数据路径、表名/字段名、原始数值）');
    return { passed: false, failures };
  }
  for (const ev of result.evidence_list) {
    if (!ev.conclusion_id) failures.push('evidence 条目缺少 conclusion_id');
    if (!ev.data_path) failures.push(`evidence[${ev.conclusion_id || '?'}] 缺少 data_path`);
    if (!ev.raw_value) failures.push(`evidence[${ev.conclusion_id || '?'}] 缺少 raw_value`);
    if (!ev.verification_status) failures.push(`evidence[${ev.conclusion_id || '?'}] 缺少 verification_status`);
  }
  return { passed: failures.length === 0, failures };
}

export function validateOptimizer(result) {
  const failures = [];
  if (!result.solutions || result.solutions.length === 0) {
    failures.push('solutions 为空（至少一条建议必须完成"方案+验证"闭环）');
  } else {
    result.solutions.forEach((s, i) => {
      if (!s.suggestion_id) failures.push(`solutions[${i}] 缺少 suggestion_id`);
      if (!s.approach) failures.push(`solutions[${i}] 缺少 approach（方案思路）`);
      if (!s.test_result) failures.push(`solutions[${i}] 缺少 test_result（验证结果）`);
    });
  }
  return { passed: failures.length === 0, failures };
}

// Composite gate map: state → validator function
export const GATE_MAP = {
  architect: validateArchitect,
  developer: validateDeveloper,
  tester: validateTester,
  optimizer: validateOptimizer,
  reviewer(result, ctx) {
    const structural = validateReviewer(result, ctx);
    if (!structural.passed) return structural;
    return validateReviewerEvidence(result);
  },
};
