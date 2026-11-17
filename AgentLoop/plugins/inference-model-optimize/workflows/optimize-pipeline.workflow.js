// Inference Model Optimize Pipeline Workflow
// ============================================
// State-machine-driven 4-phase optimization pipeline.
//
// Architecture (v2.0 — cross-platform):
//   1. YAML state machine (optimize-pipeline.yaml) — single source of truth
//   2. JS config (state-machine-config.js) — compiled runtime equivalent
//   3. This workflow.js — thin orchestration glue (~200 lines)
//   4. schemas.js — JSON schemas for structured output
//   5. gates.js — gate validation functions
//   6. briefs.js — agent prompt templates
//
// Platforms: Claude Code (via Workflow tool), Codex
// Plugin: inference-model-optimize
// Entry: orchestrator AGENT.md → Workflow({scriptPath, args})

import { STATE_MACHINE, nextState, initIterCounts } from './state-machine-config.js';
import { SCHEMA_MAP } from './schemas.js';
import { GATE_MAP } from './gates.js';
import { BRIEF_MAP } from './briefs.js';

export const meta = {
  name: 'inference-model-optimize-pipeline',
  description: '推理模型优化流水线 — 状态机驱动四阶段编排，门禁校验 + 迭代闭环',
  phases: [
    { title: 'Orchestration Loop', detail: '状态机驱动的编排循环 — architect → developer → tester → reviewer' },
    { title: 'Final Report', detail: '汇总所有阶段产出，输出最终优化报告' },
  ],
};

// ============================================================================
// Handoff extraction
// ============================================================================

/**
 * Extract the canonical handoff line from an agent's return value.
 * Accepts a structured object (reads its `handoff` field first — JSON
 * stringification would escape newlines and break line-based matching)
 * or a plain-text string (looks for "→ orchestrator:" / "handoff:" / "@aog-"
 * prefixed lines).
 */
function extractHandoff(text) {
  if (!text) return '';
  if (typeof text === 'object') {
    var h = text && text.handoff;
    if (h) return String(h).trim().replace(/\*\*/g, '');
    text = JSON.stringify(text);
  }
  var lines = String(text).split('\n');
  var prefixes = ['→ orchestrator:', 'handoff:', '@aog-'];
  var candidates = [];
  for (var i = 0; i < lines.length; i++) {
    var s = lines[i].trim().replace(/\*\*/g, '');
    if (!s) continue;
    for (var j = 0; j < prefixes.length; j++) {
      if (s.indexOf(prefixes[j]) === 0) {
        candidates.push(s);
        break;
      }
    }
  }
  return candidates.length > 0 ? candidates[candidates.length - 1] : '';
}

// ============================================================================
// Phase runner with gate validation + retry
// ============================================================================

/**
 * Run a single phase: spawn agent, validate output, retry on gate failure.
 *
 * @param {string} stateId - Current state id (e.g. 'architect')
 * @param {string} agentType - Qualified agent type
 * @param {string} prompt - Full prompt text
 * @param {function} gateValidator - Gate validation function
 * @param {number} maxRetries - Max retries (default 2 = 3 total attempts)
 * @returns {object|null} Agent result or null on exhaustion
 */
async function runPhase(stateId, agentType, prompt, gateValidator, maxRetries) {
  maxRetries = maxRetries || 2;
  var schema = SCHEMA_MAP[stateId];

  for (var attempt = 0; attempt <= maxRetries; attempt++) {
    var label = attempt === 0
      ? agentType
      : agentType + '-retry-' + attempt;

    log('Phase ' + stateId + ': 启动 ' + agentType +
        (attempt > 0 ? ' (重试 ' + attempt + '/' + maxRetries + ')' : ''));

    var result = await agent(prompt, {
      label: label,
      phase: stateId.charAt(0).toUpperCase() + stateId.slice(1),
      agentType: agentType,
      schema: schema,
    });

    if (!result) {
      log('Phase ' + stateId + ': ' + agentType + ' 返回 null（agent 失败或被跳过）');
      if (attempt < maxRetries) continue;
      return null;
    }

    var validation = gateValidator(result);
    if (validation.passed) {
      log('Phase ' + stateId + ': ✅ 门禁校验通过');
      return result;
    }

    log('Phase ' + stateId + ': ❌ 门禁校验失败: ' + validation.failures.join('; '));

    if (attempt < maxRetries) {
      prompt = prompt + '\n\n前一次输出未通过门禁校验，以下问题需要修正：\n' +
        validation.failures.map(function (f) { return '- ' + f; }).join('\n') +
        '\n\n请根据上述反馈修正输出，确保所有必填字段完整且准确。';
    }
  }

  log('Phase ' + stateId + ': 🛑 ' + agentType + ' 已达最大重试次数，流水线终止');
  return null;
}

// ============================================================================
// Main pipeline — state machine driven
// ============================================================================

async function main() {
  // --------------------------------------------------------------------------
  // Extract args from orchestrator (backward-compatible)
  // --------------------------------------------------------------------------
  args = args || {};
  var modelPath = args.model_path || '未指定';
  var scenario = args.scenario || '未指定';
  var npuCards = args.npu_cards || '未指定';
  var cannPath = args.cann_path || '/usr/local/Ascend';
  var memoryPerCard = args.memory_per_card || '64';
  var framework = args.framework || 'vLLM';
  var precision = args.precision || 'BF16';
  var optimizationGoal = args.optimization_goal || 'throughput';
  var targetThroughput = args.target_throughput || null;
  var maxAutotuneMinutes = args.max_autotune_minutes || null;
  var entryPhase = args.entry_phase || 1;

  log('流水线启动: 模型=' + modelPath + ', 场景=' + scenario +
      ', NPU卡数=' + npuCards + ', 目标=' + optimizationGoal +
      ', 入口阶段=' + entryPhase);

  // --------------------------------------------------------------------------
  // State accumulators
  // --------------------------------------------------------------------------
  var priorOutputs = {};      // { stateId: structuredOutput }
  var transitionLog = [];     // append-only transition history
  var iterCounts = initIterCounts();
  var currentState = STATE_MACHINE.initial;
  var iterationRound = 0;
  var attemptedSuggestions = {};  // reviewer suggestion dedup

  // --------------------------------------------------------------------------
  // Mid-pipeline entry: synthesize prior outputs from args
  // --------------------------------------------------------------------------
  if (entryPhase >= 3 && args.service_url) {
    priorOutputs.developer = {
      service_url: args.service_url,
      port: args.port || 8000,
      model_name: args.model_name || modelPath,
      tp: args.tp || 1,
      dp: args.dp || 1,
      ep: args.ep || 1,
      health_check_passed: true,
      models_endpoint_ok: true,
      service_status: 'running',
    };
    log('Phase 2 (skip): 使用已有服务 ' + priorOutputs.developer.service_url +
        ':' + priorOutputs.developer.port);
    currentState = 'tester';
    // Synthesize bootstrap transition
    transitionLog.push({
      fromState: 'developer', toState: 'tester',
      handoff: '合成入口: entry_phase=' + entryPhase,
      rationale: '从已有服务进入',
    });
    iterCounts.developer = 1;
  } else if (entryPhase >= 2 && args.model_path) {
    currentState = 'developer';
  }

  // --------------------------------------------------------------------------
  // Main orchestration loop — state machine driven
  // --------------------------------------------------------------------------
  var maxIterations = STATE_MACHINE.totalSpawnCap || 20;
  var spawnCount = 0;

  while (spawnCount < maxIterations) {
    // Check terminal
    if (STATE_MACHINE.terminal.indexOf(currentState) !== -1) {
      log('流水线到达终端状态: ' + currentState);
      break;
    }

    var stateSpec = STATE_MACHINE.states[currentState];
    if (!stateSpec) {
      log('未知状态: ' + currentState + '，终止流水线');
      break;
    }

    var agentType = stateSpec.agent;

    // finalize state: orchestrator generates the report, no agent spawn needed
    if (currentState === 'finalize' || !agentType) {
      log('进入最终报告阶段');
      break;
    }

    // Check if we exceeded iter cap for this state
    if (iterCounts[stateSpec.iterCounter] >= stateSpec.iterCap) {
      log('状态 ' + currentState + ' 已达迭代上限 (' +
          iterCounts[stateSpec.iterCounter] + '/' + stateSpec.iterCap + ')，强制转为 abort');
      transitionLog.push({
        fromState: currentState, toState: 'abort',
        handoff: 'iter_cap_exceeded',
        rationale: '迭代上限触发: ' + stateSpec.iterCounter +
                   '=' + iterCounts[stateSpec.iterCounter] +
                   ' >= ' + stateSpec.iterCap,
      });
      break;
    }

    // Build prompt from brief template
    var briefFn = BRIEF_MAP[currentState];
    var prompt;
    if (currentState === 'developer' && priorOutputs.tester &&
        priorOutputs.tester.candidate_restart_params &&
        Object.keys(priorOutputs.tester.candidate_restart_params).length > 0) {
      // Restart path: tester → developer (topology switch or param restart)
      prompt = BRIEF_MAP.developerRestartBrief
        ? BRIEF_MAP.developerRestartBrief(args, priorOutputs, iterCounts)
        : prompt;
    } else if (currentState === 'optimizer' && priorOutputs.reviewer) {
      // Iteration path: reviewer → optimizer with suggestions
      var iterContext = buildIterationContext(priorOutputs.reviewer, attemptedSuggestions, iterationRound);
      prompt = briefFn
        ? briefFn(args, priorOutputs, iterCounts, '(迭代' + iterationRound + ')', iterContext)
        : prompt;
    } else if (currentState === 'reviewer') {
      // reviewer 需要轮次信息以正确填写 profiling_round（1=首次，2+=回采）与 resampled
      var reviewerRound = iterationRound + 1;
      prompt = briefFn
        ? briefFn(args, priorOutputs, iterCounts, '(profiling_round=' + reviewerRound + ')', '')
        : prompt;
    } else {
      prompt = briefFn
        ? briefFn(args, priorOutputs, iterCounts, '', '')
        : '请执行 ' + currentState + ' 阶段的任务。';
    }

    if (!prompt) {
      prompt = '请执行 ' + currentState + ' 阶段的任务。';
    }

    // Spawn agent
    spawnCount++;
    log('[' + spawnCount + '/' + maxIterations + '] 启动: ' + agentType +
        ' (state=' + currentState + ', iter=' + (iterCounts[stateSpec.iterCounter] || 0) + '/' + stateSpec.iterCap + ')');

    var gateValidator = GATE_MAP[currentState] || (function () { return { passed: true, failures: [] }; });
    // reviewer 门禁需要 tester 的 workload_description 做跨文件一致性对比：
    // 从 tester 结构化输出提取并注入 ctx，避免 reviewer 用小负载替代采集。
    if (currentState === 'reviewer' && priorOutputs.tester && priorOutputs.tester.workload_description) {
      var reviewerGate = GATE_MAP.reviewer;
      var testerWorkload = priorOutputs.tester.workload_description;
      gateValidator = function (r) { return reviewerGate(r, { testerWorkload: testerWorkload }); };
    }
    var result = await runPhase(currentState, agentType, prompt, gateValidator, 2);

    if (!result) {
      // Phase exhausted — transition to abort
      log(currentState + ' 阶段失败，终止流水线');
      transitionLog.push({
        fromState: currentState, toState: 'abort',
        handoff: 'phase_exhausted',
        rationale: currentState + ' 阶段重试耗尽',
      });
      break;
    }

    // Store output
    priorOutputs[currentState] = result;
    iterCounts[stateSpec.iterCounter] = (iterCounts[stateSpec.iterCounter] || 0) + 1;

    // Extract handoff + compute next state
    var handoff = extractHandoff(result);
    var decision = nextState(currentState, handoff, iterCounts);

    if (decision.error) {
      log('状态机错误: ' + decision.error + '，终止流水线');
      break;
    }

    // Record transition
    var entry = {
      fromState: currentState,
      toState: decision.nextState,
      handoff: handoff,
      matchedTransitionIndex: decision.matchedTransitionIndex,
      rationale: decision.rationale,
      iterCountsSnapshot: Object.assign({}, iterCounts),
    };
    transitionLog.push(entry);

    log('转移: ' + currentState + ' → ' + decision.nextState +
        ' (' + decision.rationale + ')');

    // Handle tester → developer restart cycle
    if (currentState === 'tester' && decision.nextState === 'developer') {
      // Mark that we're in a restart cycle — the next iteration will use the restart brief
      log('tester 请求重启服务，回到 developer 阶段');
    }

    // Handle reviewer → optimizer iteration
    if (currentState === 'reviewer' && decision.nextState === 'optimizer') {
      iterationRound++;
      log('进入迭代轮次 ' + iterationRound);
    }

    currentState = decision.nextState;
  }

  // --------------------------------------------------------------------------
  // Final Report
  // --------------------------------------------------------------------------
  phase('Final Report');

  var report = buildFinalReport(priorOutputs, transitionLog, iterationRound);
  log('🎯 流水线完成: ' + iterationRound + ' 轮迭代, ' +
      '最终吞吐=' + (report.performance ? report.performance.optimized_tok_per_s : '?'));

  return report;
}

// ============================================================================
// Iteration context builder
// ============================================================================

function buildIterationContext(reviewerOutput, attemptedSuggestions, round) {
  if (!reviewerOutput || !reviewerOutput.optimization_suggestions) return '';

  var suggestions = reviewerOutput.optimization_suggestions;
  var actionable = [];

  for (var i = 0; i < suggestions.length; i++) {
    var s = suggestions[i];
    // Filter: not attempted + has desc + verify method.
    // 不再按类别排除：算子融合/AOE/下发/通信类建议同样交给 optimizer 分析方案并尝试，
    // 只有物理上无法执行的（硬件采购等）由 optimizer 在报告中说明原因。
    if (attemptedSuggestions[s.id]) continue;
    if (!s.description || !s.verification_method) continue;
    actionable.push(s);
  }

  if (actionable.length === 0) return '';

  var ctx = '\n\n--- 迭代轮次 ' + round + ' ---\n' +
    'reviewer 诊断发现以下优化方向，请对每条建议先分析可行方案（如算子融合 → AOE/融合开关，下发瓶颈 → 任务队列/缓存环境变量），再尝试执行并验证；无法执行的说明原因：\n';

  for (var j = 0; j < actionable.length; j++) {
    var a = actionable[j];
    attemptedSuggestions[a.id] = true;
    ctx += '- [' + a.id + '] ' + a.description +
      ' (优先级: ' + a.priority + ', 预期收益: ' + a.expected_benefit + ')\n';
  }

  ctx += '\n验证方法：' + actionable.map(function (a) { return a.verification_method; }).join('; ');
  return ctx;
}

// ============================================================================
// Final report builder
// ============================================================================

function buildFinalReport(priorOutputs, transitionLog, iterations) {
  var arch = priorOutputs.architect;
  var dev = priorOutputs.developer;
  var tester = priorOutputs.tester;
  var reviewer = priorOutputs.reviewer;
  var optimizer = priorOutputs.optimizer;

  return {
    status: 'completed',
    iterations: iterations,
    total_spawns: transitionLog.length,
    transition_log: transitionLog.map(function (e) {
      return {
        from: e.fromState,
        to: e.toState,
        handoff: (e.handoff || '').substring(0, 100),
        rationale: e.rationale,
      };
    }),
    deployment_plan: arch ? {
      model_profile: arch.model_profile,
      recommended_strategy: arch.recommended_strategy,
      hardware_requirements: arch.hardware_requirements,
    } : null,
    service_info: dev ? {
      address: dev.service_url + ':' + dev.port,
      model_name: dev.model_name,
      tp: dev.tp,
      dp: dev.dp,
    } : null,
    performance: tester ? {
      baseline_tok_per_s: tester.baseline_metrics && tester.baseline_metrics.output_tok_per_s,
      optimized_tok_per_s: tester.optimized_metrics && tester.optimized_metrics.output_tok_per_s,
      improvement_pct: tester.comparison && tester.comparison.output_tok_per_s_change_pct,
      optimal_params: tester.optimal_params,
    } : null,
    diagnosis: reviewer ? {
      e2e_time_ms: reviewer.diagnosis && reviewer.diagnosis.e2e_time_ms,
      bottleneck: reviewer.bottleneck_classification,
      operator_top_n: reviewer.operator_top_n,
      suggestion_count: reviewer.optimization_suggestions ? reviewer.optimization_suggestions.length : 0,
    } : null,
    optimization: optimizer ? {
      solutions: optimizer.solutions,
      metrics: optimizer.optimization_metrics,
    } : null,
  };
}

// Entry point
return await main();
