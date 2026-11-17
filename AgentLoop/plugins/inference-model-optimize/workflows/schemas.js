// JSON Schemas for structured agent outputs
// ============================================
// Extracted from optimize-pipeline.workflow.js (Phase 1-4 schemas).
// Used by the workflow to enforce structured output from each agent phase.

export const ARCHITECT_SCHEMA = {
  type: 'object',
  properties: {
    model_profile: {
      type: 'object',
      properties: {
        architecture: { type: 'string', description: '模型架构类型，如 MHA / GQA / MLA / GDN / CSA+HCA / MoE' },
        num_layers: { type: 'number', description: 'Transformer 层数' },
        hidden_size: { type: 'number', description: '隐藏层维度' },
        num_attention_heads: { type: 'number', description: '注意力头数' },
        num_kv_heads: { type: 'number', description: 'KV 头数（GQA 场景下与注意力头数不同）' },
        num_parameters: { type: 'string', description: '参数量，如 "72B"、"13B"' },
        weight_size_gb: { type: 'number', description: '权重体积（GB）' },
      },
      required: ['architecture', 'num_layers', 'hidden_size', 'num_parameters'],
    },
    memory_budget: {
      type: 'object',
      properties: {
        model_weights_gb: { type: 'number' },
        kv_cache_gb: { type: 'number' },
        activation_gb: { type: 'number' },
        engine_overhead_gb: { type: 'number' },
        total_per_card_gb: { type: 'number' },
        notes: { type: 'string' },
      },
      required: ['model_weights_gb', 'kv_cache_gb', 'total_per_card_gb'],
    },
    recommended_strategy: {
      type: 'object',
      properties: {
        tp: { type: 'number' },
        dp: { type: 'number' },
        ep: { type: 'number' },
        rationale: { type: 'string' },
      },
      required: ['tp', 'dp', 'rationale'],
    },
    hardware_requirements: {
      type: 'object',
      properties: {
        min_npu_cards: { type: 'number' },
        recommended_config: { type: 'string' },
      },
      required: ['min_npu_cards', 'recommended_config'],
    },
    risk_warnings: { type: 'array', items: { type: 'string' } },
    handoff: { type: 'string', description: '状态机协议 token，如 "handoff: ARCHITECT_READY" 或 "handoff: ARCHITECT_FAILED"' },
  },
  required: ['model_profile', 'memory_budget', 'recommended_strategy', 'hardware_requirements', 'handoff'],
};

export const DEVELOPER_SCHEMA = {
  type: 'object',
  properties: {
    service_url: { type: 'string' },
    port: { type: 'number' },
    model_name: { type: 'string' },
    tp: { type: 'number' },
    dp: { type: 'number' },
    ep: { type: 'number', description: '专家并行度；未启用专家并行时为 1' },
    health_check_passed: { type: 'boolean' },
    models_endpoint_ok: { type: 'boolean' },
    service_status: { type: 'string' },
    handoff: { type: 'string', description: '状态机协议 token，如 "handoff: SERVICE_READY" 或 "handoff: SERVICE_FAILED"' },
    errors: { type: 'array', items: { type: 'string' } },
  },
  required: ['service_url', 'port', 'model_name', 'tp', 'dp', 'ep', 'health_check_passed', 'models_endpoint_ok', 'service_status', 'handoff'],
};

export const TESTER_SCHEMA = {
  type: 'object',
  properties: {
    baseline_metrics: {
      type: 'object',
      properties: {
        output_tok_per_s: { type: 'number' },
        ttft_ms: { type: 'number' },
        tpot_ms: { type: 'number' },
        latency_p50_ms: { type: 'number' },
        latency_p99_ms: { type: 'number' },
      },
      required: ['output_tok_per_s', 'ttft_ms', 'tpot_ms'],
    },
    optimized_metrics: {
      type: 'object',
      properties: {
        output_tok_per_s: { type: 'number' },
        ttft_ms: { type: 'number' },
        tpot_ms: { type: 'number' },
        latency_p50_ms: { type: 'number' },
        latency_p99_ms: { type: 'number' },
      },
      required: ['output_tok_per_s', 'ttft_ms', 'tpot_ms'],
    },
    comparison: {
      type: 'object',
      properties: {
        output_tok_per_s_change_pct: { type: 'number' },
        ttft_change_pct: { type: 'number' },
        tpot_change_pct: { type: 'number' },
        peak_concurrency_change_pct: { type: 'number' },
      },
      required: ['output_tok_per_s_change_pct'],
    },
    // 拓扑搜索是寻优的强制第一阶段：未完成 TP/DP/EP 比较不得交接 OPTIMIZATION_READY。
    topology_search: {
      type: 'object',
      properties: {
        search_completed: { type: 'boolean', description: '是否已完成 TP/DP/EP 拓扑搜索；false 时不允许交接 OPTIMIZATION_READY' },
        total_cards: { type: 'number', description: '本次拓扑搜索受约束的总卡数' },
        candidates: {
          type: 'array',
          description: '已测试的全部拓扑候选（含上一轮通过重启已测过的，必须累计）',
          minItems: 1,
          items: {
            type: 'object',
            properties: {
              tp: { type: 'number' },
              dp: { type: 'number' },
              ep: { type: 'number' },
              output_tok_per_s: { type: 'number' },
              ttft_ms: { type: 'number' },
              tpot_ms: { type: 'number' },
              notes: { type: 'string', description: '有效/无效、启动失败或未测原因' },
            },
            required: ['tp', 'dp', 'ep', 'output_tok_per_s', 'notes'],
          },
        },
        best_topology: {
          type: 'object',
          properties: {
            tp: { type: 'number' },
            dp: { type: 'number' },
            ep: { type: 'number' },
            rationale: { type: 'string' },
          },
          required: ['tp', 'dp', 'ep', 'rationale'],
        },
      },
      required: ['search_completed', 'candidates', 'best_topology'],
    },
    optimal_params: {
      type: 'object',
      properties: {
        tp: { type: 'number' },
        dp: { type: 'number' },
        ep: { type: 'number', description: '专家并行度；未启用专家并行时为 1' },
      },
      required: ['tp', 'dp', 'ep'],
    },
    candidate_restart_params: {
      type: 'object',
      description: '需要 developer 重启服务时填写，至少包含 tp/dp/ep',
      properties: {
        tp: { type: 'number' },
        dp: { type: 'number' },
        ep: { type: 'number' },
      },
    },
    target_achieved: { type: 'boolean' },
    workload_description: { type: 'string' },
    comparison_report: { type: 'string' },
    handoff: { type: 'string', description: '状态机协议 token，如 "handoff: NEED_RESTART" 或 "handoff: OPTIMIZATION_READY"' },
    errors: { type: 'array', items: { type: 'string' } },
  },
  required: ['baseline_metrics', 'optimized_metrics', 'comparison', 'topology_search', 'optimal_params', 'target_achieved', 'comparison_report', 'workload_description', 'handoff'],
};

export const REVIEWER_SCHEMA = {
  type: 'object',
  properties: {
    profiling_data_path: { type: 'string' },
    profiling_collected: { type: 'boolean' },
    profiling_round: { type: 'number', description: '采集轮次：1=首次，2+=回采轮次（每次从 tester 回来后递增）' },
    resampled: { type: 'boolean', description: '本轮是否重新采集 profiling；再入轮次必须为 true' },
    workload_description: { type: 'string', description: '实际采集负载声明（请求数/并发/输入输出长度），必须与 tester 的 workload_description 完全一致' },
    previous_round_comparison: {
      type: 'object',
      description: '再入轮次（profiling_round>=2）必填：上轮 vs 本轮的瓶颈构成对比与建议验证结论',
      properties: {
        compute_pct_change: { type: 'number' },
        communication_pct_change: { type: 'number' },
        idle_pct_change: { type: 'number' },
        operator_top_n_changes: { type: 'string' },
        suggestion_verification: { type: 'string', description: '上一轮建议的实际验证结论（生效/无效/负收益及依据）' },
      },
    },
    diagnosis: {
      type: 'object',
      properties: {
        e2e_time_ms: { type: 'number' },
        compute_pct: { type: 'number' },
        communication_pct: { type: 'number' },
        idle_pct: { type: 'number' },
      },
      required: ['e2e_time_ms', 'compute_pct', 'communication_pct', 'idle_pct'],
    },
    bottleneck_classification: {
      type: 'object',
      properties: {
        primary: { type: 'string', description: 'compute / communication / dispatch / memory' },
        details: { type: 'string' },
      },
      required: ['primary', 'details'],
    },
    operator_top_n: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          rank: { type: 'number' },
          op_name: { type: 'string' },
          time_ms: { type: 'number' },
          percentage: { type: 'number' },
        },
        required: ['rank', 'op_name', 'time_ms', 'percentage'],
      },
    },
    optimization_suggestions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          description: { type: 'string' },
          priority: { type: 'string' },
          expected_benefit: { type: 'string' },
          category: { type: 'string' },
          executable_by: { type: 'string' },
          risk: { type: 'string' },
          estimated_time: { type: 'string' },
          verification_method: { type: 'string' },
        },
        required: ['id', 'description', 'priority', 'category', 'executable_by'],
      },
    },
    evidence_list: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          conclusion_id: { type: 'string' },
          data_path: { type: 'string' },
          table_name: { type: 'string' },
          field_name: { type: 'string' },
          raw_value: { type: 'string' },
          query_or_method: { type: 'string' },
          verification_status: { type: 'string' },
        },
        required: ['conclusion_id', 'data_path', 'raw_value', 'verification_status'],
      },
    },
    diagnosis_report: { type: 'string' },
    handoff: { type: 'string', description: '状态机协议 token，如 "handoff: REVIEWER_CONTINUE" 或 "handoff: REVIEWER_DONE"' },
    errors: { type: 'array', items: { type: 'string' } },
  },
  required: ['profiling_data_path', 'profiling_collected', 'profiling_round', 'workload_description', 'diagnosis', 'bottleneck_classification', 'operator_top_n', 'optimization_suggestions', 'diagnosis_report', 'handoff'],
};

export const OPTIMIZER_SCHEMA = {
  type: 'object',
  properties: {
    solutions: {
      type: 'array',
      description: '对每条 reviewer 建议的方案、代码变更与验证结果（至少一条完成实现+验证）',
      minItems: 1,
      items: {
        type: 'object',
        properties: {
          suggestion_id: { type: 'string' },
          approach: { type: 'string', description: '模型自己分析的优化方案思路与具体步骤' },
          code_changes: { type: 'string', description: '生成的代码/配置变更：文件路径、修改内容、回滚方式' },
          test_result: { type: 'string', description: '验证结果：有效 / 无效 / 无法实现（附原因）' },
          evidence: { type: 'string', description: 'benchmark 数值对比或日志路径' },
        },
        required: ['suggestion_id', 'approach', 'test_result'],
      },
    },
    optimization_metrics: {
      type: 'object',
      properties: {
        before_output_tok_per_s: { type: 'number' },
        after_output_tok_per_s: { type: 'number' },
        change_pct: { type: 'number' },
      },
    },
    handoff: { type: 'string', description: '状态机协议 token，如 "handoff: OPTIMIZER_DONE" 或 "handoff: OPTIMIZER_FAILED"' },
    errors: { type: 'array', items: { type: 'string' } },
  },
  required: ['solutions', 'handoff'],
};

// Map state id → schema
export const SCHEMA_MAP = {
  architect: ARCHITECT_SCHEMA,
  developer: DEVELOPER_SCHEMA,
  tester: TESTER_SCHEMA,
  reviewer: REVIEWER_SCHEMA,
  optimizer: OPTIMIZER_SCHEMA,
};
