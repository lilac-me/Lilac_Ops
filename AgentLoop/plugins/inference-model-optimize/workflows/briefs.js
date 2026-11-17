// Agent Brief / Prompt Templates
// ================================
// Each function builds the prompt for a specific agent phase,
// incorporating prior outputs, iter counts, and user args.
// Extracted from optimize-pipeline.workflow.js main() prompt builders.

/**
 * Build architect prompt from user args.
 * @param {object} args - orchestrator's args (model_path, scenario, npu_cards, etc.)
 * @param {object} prior - prior phase outputs (empty for architect)
 * @param {object} counts - iter counts (empty for architect)
 * @returns {string}
 */
export function architectBrief(args, prior, counts) {
  return [
    '对以下模型进行部署方案设计：',
    `- 模型路径或 config.json：${args.model_path || '未指定'}`,
    `- 场景需求：${args.scenario || '未指定'}`,
    `- 可用 NPU 卡数：${args.npu_cards || '未指定'}`,
    `- 单卡显存：${args.memory_per_card || '64'} GB`,
    `- 推理框架：${args.framework || 'vLLM'}`,
    `- 精度：${args.precision || 'BF16'}`,
    '',
    '请分析模型结构、计算显存预算、推荐并行策略（TP/DP），产出结构化部署方案。',
  ].join('\n');
}

/**
 * Build developer prompt incorporating architect's deployment plan.
 */
export function developerBrief(args, prior, counts) {
  const strategy = (prior.architect && prior.architect.recommended_strategy)
    || { tp: args.tp || 1, dp: args.dp || 1 };

  return [
    '执行 vLLM Ascend 推理服务部署：',
    `- 模型路径：${args.model_path || '未指定'}`,
    `- TP=${strategy.tp}, DP=${strategy.dp}, EP=${strategy.ep || 1}`,
    `- CANN 路径：${args.cann_path || '/usr/local/Ascend'}`,
    `- 单卡显存：${args.memory_per_card || '64'} GB`,
    '- gpu-memory-utilization：保守设置 0.80-0.85',
    '',
    '请先确认环境就绪，然后启动推理服务，验证 /health 和 /v1/models 端点可达。',
  ].join('\n');
}

/**
 * Build tester prompt from developer's service info.
 */
export function testerBrief(args, prior, counts, iterationLabel, extraContext) {
  const dev = prior.developer || {};
  const prevTester = prior.tester || {};

  // 续接上一轮拓扑搜索进度（重启后 tester 是新上下文，必须显式带回已测候选）
  const prevTopo = prevTester.topology_search || {};
  const topoContext = (prevTopo.candidates && prevTopo.candidates.length > 0)
    ? [
        '--- 拓扑搜索进度续接 ---',
        '上一轮已测试的拓扑候选（必须累计保留，不要重测已完成的有效候选）：',
        JSON.stringify(prevTopo.candidates),
        prevTopo.best_topology ? `当前阶段最优拓扑：${JSON.stringify(prevTopo.best_topology)}` : '',
      ].filter(Boolean).join('\n')
    : '';

  return [
    '对以下推理服务执行自动寻优：',
    `- 服务地址：${dev.service_url || args.service_url}:${dev.port || args.port || 8000}`,
    `- 模型名：${dev.model_name || args.model_name || '未指定'}`,
    `- 当前服务 TP=${dev.tp || args.tp || 1}, DP=${dev.dp || args.dp || 1}, EP=${dev.ep || args.ep || 1}（这是起点配置，不是固定约束）`,
    `- 可用 NPU 卡数：${args.npu_cards || '未指定'}`,
    `- 优化目标：${args.optimization_goal || 'throughput'}`,
    `- CANN 路径：${args.cann_path || '/usr/local/Ascend'}`,
    args.target_throughput ? `- 目标吞吐：${args.target_throughput} tok/s` : '',
    args.max_autotune_minutes ? `- 最长寻优时间：${args.max_autotune_minutes} 分钟` : '',
    '- 固定 workload：采用相同请求数、并发数、输入/输出长度、采样参数和预热规则',
    '',
    '【强制第一阶段：TP/DP/EP 拓扑搜索】',
    '调用 vllm-ascend-autotune-zh 技能寻优时，必须先完成并行策略拓扑搜索，再调其它参数：',
    '1. 在卡数约束内枚举全部合法的 TP/DP/EP 组合（卡数 >= 2 时至少实际测试 2 种不同拓扑；MoE 模型必须测试 --enable-expert-parallel 开/关）；',
    '2. 对每个候选拓扑逐一跑 benchmark，在固定 workload 下比较 Output token throughput、TTFT、TPOT 和有效性；',
    '3. 更换拓扑需要重启服务：输出 handoff: NEED_RESTART，并把目标拓扑（tp/dp/ep 及所需启动参数）写入 candidate_restart_params；',
    '4. 从重启后的服务拿到新拓扑数据后，继续测试剩余候选，直到候选穷尽或确定阶段最优拓扑；',
    '5. 拓扑搜索未完成（search_completed=false）时，禁止输出 handoff: OPTIMIZATION_READY；',
    '6. 仅当卡数约束下只有一种合法拓扑时，允许只测一个候选，但必须在 candidates[0].notes 与 best_topology.rationale 中写明理由。',
    '',
    '拓扑搜索完成后，再按 scheduler、PCP/DCP、KVCacheOffload、显存、compilation、特性开关的顺序继续寻优其它参数。',
    '',
    '【错误分析与自愈（先救后换）】',
    '任何候选报错时，禁止直接跳到下一个候选：先读取 server log / benchmark log / SMA 日志定位根因，',
    '判定可修复性（参数取值不当、生成参数、端口冲突、拼写错误 → 可修复；特性不支持、需重编译/量化、设备故障 → 不可修复），',
    '可修复则修复后重试同一候选（最多 2 次修复尝试），不可修复或修复耗尽才标记负样本并写明原因，再进入下一个候选。',
    topoContext,
    iterationLabel || '',
    extraContext || '',
  ].filter(Boolean).join('\n');
}

/**
 * Build tester restart prompt (tester → developer restart cycle).
 */
export function developerRestartBrief(args, prior, counts) {
  const dev = prior.developer || {};
  const testerOutput = prior.tester || {};
  const cp = testerOutput.candidate_restart_params || {};
  return [
    '使用以下参数重启 vLLM Ascend 推理服务：',
    `- 服务地址：${dev.service_url}:${dev.port}`,
    `- 模型名：${dev.model_name}`,
    `- 候选参数：${JSON.stringify(cp)}`,
    `- 拓扑切换：TP=${cp.tp || dev.tp}, DP=${cp.dp || dev.dp}, EP=${cp.ep || dev.ep || 1}`,
    `- CANN 路径：${args.cann_path || '/usr/local/Ascend'}`,
    '',
    '这是 tester 拓扑搜索或参数寻优触发的重启，必须严格按候选参数启动（含 TP/DP/EP 变更），',
    '并在返回结果中填写实际生效的 tp/dp/ep。请重启服务并验证 /health 和 /v1/models 端点。',
  ].join('\n');
}

/**
 * Build reviewer prompt from tester's optimal config.
 */
export function reviewerBrief(args, prior, counts, iterationLabel, extraContext) {
  const dev = prior.developer || {};
  const testerOutput = prior.tester || {};
  const optOutput = prior.optimizer || {};
  const prevReviewer = prior.reviewer || {};
  // 解析轮次：iterationLabel 形如 "(profiling_round=2)"；缺省按首次采集=1
  const roundMatch = /profiling_round=(\d+)/.exec(iterationLabel || '');
  const profilingRound = roundMatch ? parseInt(roundMatch[1], 10) : 1;
  const isResample = profilingRound >= 2;
  const attempted = (optOutput.solutions && optOutput.solutions.length > 0)
    ? [
        '--- 上一轮 optimizer 已实现并验证的方案（不要再原样重复给出，基于结果提出新方向或确认收敛）---',
        JSON.stringify(optOutput.solutions),
      ].join('\n')
    : '';
  // 上轮 reviewer 诊断注入（供 previous_round_comparison 对比上轮 vs 本轮）
  const prevRoundContext = (prevReviewer.diagnosis || prevReviewer.bottleneck_classification || prevReviewer.operator_top_n)
    ? [
        '--- 上一轮 reviewer 诊断（用于 previous_round_comparison 对比）---',
        prevReviewer.diagnosis ? `上轮 E2E 耗时/瓶颈构成：${JSON.stringify(prevReviewer.diagnosis)}` : '',
        prevReviewer.bottleneck_classification ? `上轮瓶颈分类：${JSON.stringify(prevReviewer.bottleneck_classification)}` : '',
        prevReviewer.operator_top_n ? `上轮算子 Top-N：${JSON.stringify(prevReviewer.operator_top_n)}` : '',
      ].filter(Boolean).join('\n')
    : '';
  return [
    '对最优配置的推理服务进行 profiling 诊断：',
    `- 服务地址：${dev.service_url}:${dev.port}`,
    `- 模型名：${dev.model_name}`,
    `- 当前最优 TP=${dev.tp}, DP=${dev.dp}`,
    `- CANN 路径：${args.cann_path || '/usr/local/Ascend'}`,
    testerOutput.optimal_params
      ? `- 最优参数组合：${JSON.stringify(testerOutput.optimal_params)}`
      : '',
    '',
    '请采集 torch-profiler 数据（L1 级别，middle 阶段），使用 msprof-analyze 进行多维度分析，',
    '产出结构化诊断报告（E2E 耗时、瓶颈三分类、算子 Top-N、优化建议）。',
    '每条量化结论必须提供数据路径、实际表名/字段名、查询方法、原始数值。',
    '',
    '【profiling 轮次与负载约束】',
    `- 本轮为第 ${profilingRound} 轮采集${isResample ? '（回采轮次）' : '（首次采集）'}：profiling_round 必须填 ${profilingRound}，resampled 必须填 ${isResample}。`,
    '- workload_description 必须如实声明本次实际采集负载（请求数/并发/输入输出长度），且必须与 tester 的固定 workload 完全一致，禁止用小负载替代采集。',
    isResample ? '- 回采轮次必须输出 previous_round_comparison：逐项对比上轮与本轮 compute/communication/idle 占比、算子 Top-N 变化，并对上轮建议给出 suggestion_verification 验证结论（生效/无效/负收益及依据）。' : '',
    '',
    'handoff 判定规则（不要自己判断迭代上限，是否继续由状态机决定）：',
    '- 存在至少一条可执行的建议（参数/环境变量/算子融合/AOE/下发/通信/拓扑调整等任何 optimizer 能分析方案、',
    '  生成代码并验证的方向，且给出具体 description + verification_method）→ REVIEWER_CONTINUE',
    '- 建议全部被尝试且无效、或优化已收敛 → REVIEWER_DONE',
    '',
    prevRoundContext,
    attempted,
  ].filter(Boolean).join('\n');
}

/**
 * Build optimizer prompt from reviewer's suggestions.
 */
export function optimizerBrief(args, prior, counts, iterationLabel, extraContext) {
  const dev = prior.developer || {};
  const reviewerOutput = prior.reviewer || {};
  const suggestions = reviewerOutput.optimization_suggestions || [];
  const suggestionLines = suggestions.length > 0
    ? suggestions.map((s, i) =>
        `- [${s.id || 's' + i}] ${s.description} (类别: ${s.category || '未标注'}, ` +
        `executable_by: ${s.executable_by || '未标注'}, 优先级: ${s.priority || '未标注'}, ` +
        `预期收益: ${s.expected_benefit || '未标注'}, 验证方法: ${s.verification_method || '未给出'})`
      ).join('\n')
    : '';
  return [
    '对 reviewer 的优化建议逐一思考方案、生成方案代码、测试验证优化是否有效：',
    `- 服务地址：${dev.service_url}:${dev.port}`,
    `- 模型名：${dev.model_name}`,
    `- 当前服务 TP=${dev.tp || 1}, DP=${dev.dp || 1}, EP=${dev.ep || 1}`,
    `- CANN 路径：${args.cann_path || '/usr/local/Ascend'}`,
    '',
    'reviewer 诊断建议：',
    suggestionLines || '（无建议，直接输出 handoff: OPTIMIZER_FAILED 并说明原因）',
    '',
    '工作方式（每条建议都必须完成"方案 → 代码 → 验证"闭环）：',
    '1. 分析方案：定位目标（代码/配置/环境变量/启动参数），设计具体修改步骤；',
    '2. 生成代码/配置：实际写 patch、脚本或配置变更，记录修改位置与回滚方式；',
    '3. 测试验证：必要时用 vllm-ascend-serving 重启服务，用 vllm-ascend-benchmark 在固定 workload 下对比优化前后指标；',
    '4. 报错时按"先救后换"规则读日志修复重试（最多 2 次），无法实现或验证无效的逐条说明原因；',
    '5. 至少一条建议完成实现+验证 → handoff: OPTIMIZER_DONE；全部无法实现 → handoff: OPTIMIZER_FAILED。',
    iterationLabel || '',
    extraContext || '',
  ].filter(Boolean).join('\n');
}

/**
 * Build finalize/report prompt.
 */
export function finalizeBrief(args, prior, counts) {
  return [
    '汇总所有阶段产出，生成最终优化报告：',
    '',
    '## 部署方案',
    prior.architect ? JSON.stringify(prior.architect.recommended_strategy) : '无',
    '',
    '## 服务信息',
    prior.developer ? `${prior.developer.service_url}:${prior.developer.port} (${prior.developer.model_name})` : '无',
    '',
    '## 性能优化',
    prior.tester ? JSON.stringify({
      baseline: prior.tester.baseline_metrics,
      optimized: prior.tester.optimized_metrics,
      improvement: prior.tester.comparison,
    }) : '无',
    '',
    '## 诊断结论',
    prior.reviewer ? JSON.stringify({
      diagnosis: prior.reviewer.diagnosis,
      bottleneck: prior.reviewer.bottleneck_classification,
      suggestions: prior.reviewer.optimization_suggestions,
    }) : '无',
  ].join('\n');
}

// Map state → brief builder function
export const BRIEF_MAP = {
  architect: architectBrief,
  developer: developerBrief,
  developerRestartBrief,
  tester: testerBrief,
  reviewer: reviewerBrief,
  optimizer: optimizerBrief,
  finalize: finalizeBrief,
};
