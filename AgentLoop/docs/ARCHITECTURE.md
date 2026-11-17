# Inference-Model-Optimize-Agent 架构文档

## 概述

MindSpeed-Agent 是面向 MindSpeed / Ascend NPU 平台的 Claude Code 与 Codex 插件集合，遵循 **Plugin → Agent → Skill** 三层架构。本文档以 `inference-model-optimize` 插件为核心，详细描述推理模型优化流水线的完整架构，并补充双平台 `ecosystem-code-repository-npu-migration` 子插件的接入边界。

---

## 仓库三层架构

```text
┌──────────────────────────────────────────────────┐
│  Plugin 层（应用编排）                              │
│  .claude-plugin/plugin.json  — 注册入口            │
│  agents/   — Agent 角色定义 (agent.yaml + AGENT.md) │
│  workflows/ — 流水线定义 (YAML + JS + schemas)      │
│  skills/   — symlink → 实际 skill 目录              │
├──────────────────────────────────────────────────┤
│  Agent 层（角色执行）                                │
│  每个 Agent 承担单一角色职责                          │
│  将用户需求分解为 Skill 调用链                        │
│  输出结构化 JSON + handoff token                    │
├──────────────────────────────────────────────────┤
│  Skill 层（知识能力）                                │
│  SKILL.md 定义领域知识与工程模板                      │
│  最小知识单元，可被多个 Agent 复用                     │
└──────────────────────────────────────────────────┘
```

### 跨平台支持

| 平台 | 插件清单 | 指令文件 | Workflow 引擎 |
| ------ | --------- | --------- | -------------- |
| Claude Code | `.claude-plugin/plugin.json` | `CLAUDE.md` | Workflow JS（内嵌 JS 状态机） |
| Codex | `.codex-plugin/plugin.json` | `AGENTS.md` | Python CLI 直调 |
| DSH（DeepSeek Harness） | `.dsh/skills/`（投影） | `AGENTS.md` + `CLAUDE.md`（双加载） | 复用 Python 引擎 + subagent |

> **注意**：`CLAUDE.md` 和 `AGENTS.md` 分别是 Claude Code 和 Codex 的项目级指令文件，内容镜像但平台专属字段不同。根目录 `.claude-plugin/marketplace.json` 是**两平台共用的市场清单**（已在 codex-cli 0.146.0 上验证原生读取该路径）。`inference-model-optimize` 与 `ops` 没有 `.codex-plugin/plugin.json`，Codex 侧可见但不可安装。

### ecosystem-code-repository-npu-migration 子插件（Claude Code 与 Codex 双平台）

`plugins/ecosystem-code-repository-npu-migration/` 是双平台独立插件包，同时持有 `.claude-plugin/plugin.json`（skills 显式数组）与 `.codex-plugin/plugin.json` 两份清单，版本由 `set_plugin_version.py` 统一同步。8 个 skill 的正本位于仓库根 `skills/ecosystem-code-repository-npu-migration/`（按 model-adaptation / model-analysis / deployment / common 分组），`plugins/ecosystem-code-repository-npu-migration/skills/` 下是由 install 脚本生成的同名软链接/junction 视图——该扁平视图维持两份清单的 skill 布局、`EXPECTED_SKILLS` 校验和 `bundle.lock.json` 的形状。引用资料、脚本、测试和 `bundle.lock.json` 保留在插件目录内。它不复用 `inference-model-optimize` 的 YAML/JavaScript 状态机。

安装入口为：

```bash
claude plugin marketplace add .
claude plugin install ecosystem-code-repository-npu-migration@MindSpeed-Agent

codex plugin marketplace add .
codex plugin add ecosystem-code-repository-npu-migration@MindSpeed-Agent
```

安装后通过 `$ecosystem-code-repository-npu-migration:goal` 启动仓库级迁移；主 Agent 自适应调用其余 specialist skill，直接命令和日志作为证据，`.npu-agent/migration_progress.md` 与 `.npu-agent/compatibility.json` 仅作为轻量工作笔记。完整的目录布局、路径解析规则（`<plugin-root>` 锚点）、skill 协作与发布校验说明见 [ecosystem-code-repository-npu-migration 架构文档](ecosystem-code-repository-npu-migration.md)。

---

## inference-model-optimize 流水线

### 流水线概览

四阶段严格顺序流水线，带 tester ↔ reviewer 迭代闭环：

```text
architect (Phase 1) → developer (Phase 2) → tester (Phase 3) → reviewer (Phase 4)
                                                    ↑              │
                                                    └── 迭代闭环 ──┘ (最多 3 轮)

所有路径最终汇聚到 finalize → done，或异常退出到 abort
```

### 完整状态转移图

```text
                    ┌──────────┐
                    │   init   │ (流水线入口)
                    └────┬─────┘
                         │
              ┌──────────▼──────────┐
              │    architect        │ iter_cap=3
              │  方案设计 Agent      │
              └────┬─────┬─────┬────┘
                   │     │     │
    ARCHITECT_READY│  FAILED  always
                   │     │     │
         ┌─────────▼┐    └─────┤
         │ developer│          │
         │ 环境部署  │          │
         └──┬───┬───┘          │
            │   │              │
  SERVICE_READY│ FAILED/always │
            │   │              │
    ┌───────▼┐  │              │
    │ tester │◄─┘              │
    │ 自动寻优│                 │
    └─┬───┬──┘                │
      │   │                   │
 NEED_RESTART                  │
      │   │                   │
      │   ├─── OPTIMIZATION_READY ──┐
      │   │    + iter_below_cap      │
      │   │    (reviewer)            │
      │   │                         │
      │   ├─── OPTIMIZATION_READY ──┤
      │   │    (reviewer at cap)     │
      │   │                         │
┌─────┘   │                         │
│  ┌──────▼──────┐                  │
│  │  reviewer   │ iter_cap=5       │
│  │  性能诊断    │                  │
│  └──┬────┬─────┘                  │
│     │    │                        │
│  CONTINUE │                        │
│  + tester │  REVIEWER_DONE        │
│  below_cap│  / CONTINUE           │
│     │     │  (tester at cap)      │
│     │     └───────────┐           │
│     │                 │           │
│  ┌──┘                 │           │
│  │              ┌─────▼─────┐     │
│  │              │ finalize  │◄────┘
│  │              │ 最终报告   │
│  │              └─────┬─────┘
│  │                    │ always
│  │              ┌─────▼─────┐
│  │              │    done   │
│  │              │ 正常完成   │
│  │              └───────────┘
│  │
└──┘ (tester→developer 重启循环)
```

---

## Plugin 层

### 目录结构

```text
plugins/inference-model-optimize/
├── .claude-plugin/
│   └── plugin.json              # Claude Code 插件清单
├── agents/
│   ├── orchestrator/
│   │   ├── agent.yaml           # 跨平台 Agent 定义
│   │   └── AGENT.md             # Claude Code Agent 指令
│   ├── architect/
│   │   ├── agent.yaml
│   │   └── AGENT.md
│   ├── developer/
│   │   ├── agent.yaml
│   │   └── AGENT.md
│   ├── tester/
│   │   ├── agent.yaml
│   │   └── AGENT.md
│   └── reviewer/
│       ├── agent.yaml
│       └── AGENT.md
├── workflows/
│   ├── optimize-pipeline.yaml          # 状态机 YAML（单一真相源）
│   ├── optimize-pipeline.workflow.js   # Claude Code Workflow JS 运行时
│   ├── state-machine-config.js         # YAML 编译出的 JS 状态机
│   ├── schemas.js                      # 结构化输出 JSON Schema
│   ├── gates.js                        # 门禁校验函数
│   └── briefs.js                       # Prompt 模板
└── skills/                             # symlink → ../../skills/...
    ├── model-analysis
    ├── deployment
    └── performance-optimization
```

### plugin.json

```json
{
  "name": "inference-model-optimize",
  "version": "0.5.0",
  "skills": [
    "./skills/model-analysis",
    "./skills/deployment",
    "./skills/performance-optimization"
  ]
}
```

插件通过 `.claude-plugin/plugin.json` 声明自身及其依赖的 skill 集合。Claude Code 在加载插件时自动将 `skills/` 下的 symlink 目标注册为可用 skill。

---

## Agent 层

### Agent 角色定义

每个 Agent 由两个文件定义：

| 文件 | 用途 |
| ------ | ------ |
| `agent.yaml` | 跨平台元数据 — name、description、skills、tools、schema，以及 `claude_code:` / `codex:` 平台配置 |
| `AGENT.md` | Claude Code 原生 Agent 指令 — YAML frontmatter + 详细的中文执行指引 |

### 五个 Agent

#### 1. orchestrator（编排器）

- **角色**：唯一入口 Agent，状态机引擎执行者
- **工具**：`Agent, Read, Bash, Skill`
- **职责**：
  - 收集用户参数（model_path、scenario、npu_cards 等 17 个字段）
  - 判断入口阶段（从 architect 开始或从中途继续）
  - 亲自充当状态机引擎，循环调用子 Agent
  - 每次状态转移调用 `engine.state_machine next` 而非凭记忆决定
  - 门禁校验 + 重试（最多 2 次）
  - 汇总最终报告（finalize 阶段）
- **委派边界**：单点任务（仅显存计算、仅 benchmark）直接委派对应 Agent，不走流水线

#### 2. architect（方案设计）

- **角色**：流水线起点，主控分析 Agent
- **Skill**：`vllm-memory-calculator`
- **工具**：`Read, Bash, Skill`
- **输入**：用户提供的 model_path、scenario、npu_cards、precision 等
- **输出** (ARCHITECT_SCHEMA)：
  - `model_profile`：架构类型、层数、hidden_size、参数量
  - `memory_budget`：权重/KV Cache/激活/引擎开销（GB）
  - `recommended_strategy`：TP × DP × EP × PP、总卡数、rationale
  - `hardware_requirements`：最少 NPU 卡数、推荐配置
  - `risk_warnings`：风险提示
- **Handoff**：`handoff: ARCHITECT_READY` / `handoff: ARCHITECT_FAILED`
- **迭代上限**：3 次

#### 3. developer（环境部署）

- **角色**：执行 Agent，将方案落地为运行中的推理服务
- **Skill**：`environment-install`、`vllm-ascend-serving`
- **工具**：`Read, Bash, Skill`
- **输入**：architect 的 recommended_strategy（TP、DP）、model_path
- **输出** (DEVELOPER_SCHEMA)：
  - `service_url`、`port`、`model_name`、`tp`、`dp`
  - `health_check_passed`、`models_endpoint_ok`、`service_status`
- **Handoff**：`handoff: SERVICE_READY` / `handoff: SERVICE_FAILED`
- **迭代上限**：4 次
- **特殊路径**：tester 可通过 `candidate_restart_params` 触发 developer 重启服务

#### 4. tester（性能优化/自动寻优）

- **角色**：在 developer 之后、reviewer 之前执行；先寻优后分析
- **Skill**：`vllm-ascend-benchmark`、`vllm-ascend-autotune-zh`
- **工具**：`Read, Bash, Skill`
- **输入**：developer 的服务地址、TP/DP、优化目标
- **输出** (TESTER_SCHEMA)：
  - `baseline_metrics`：基线吞吐、TTFT、TPOT、P50/P99 延迟
  - `optimized_metrics`：优化后指标
  - `comparison`：各维度变化百分比
  - `optimal_params`：最优参数组合
  - `candidate_restart_params`：需重启服务的候选参数
  - `target_achieved`：是否达到目标
  - `comparison_report`：对比分析文本
- **Handoff**：`handoff: NEED_RESTART`（回 developer）/ `handoff: OPTIMIZATION_READY`（进 reviewer 或 finalize）
- **迭代上限**：8 次

#### 5. reviewer（性能诊断）

- **角色**：对 tester 的最优配置采集 profiling、深度瓶颈分析
- **Skill**：`ascend-profiling-collection`、`ascend-msprof-analyze-cli`、`ascend-perf-tuning`
- **工具**：`Read, Bash, Skill`
- **输入**：tester 的最优参数配置
- **输出** (REVIEWER_SCHEMA)：
  - `profiling_data_path`、`profiling_collected`
  - `diagnosis`：E2E 耗时、compute/communication/idle 占比
  - `bottleneck_classification`：compute / communication / dispatch / memory
  - `operator_top_n`：耗时 Top-N 算子及占比
  - `optimization_suggestions`：可执行优化建议（id、priority、executable_by、verification_method）
  - `evidence_list`：每条量化结论的数据来源、表名/字段名、原始数值
  - `diagnosis_report`：诊断报告文本
- **Handoff**：`handoff: REVIEWER_CONTINUE`（回 tester 迭代）/ `handoff: REVIEWER_DONE`（进 finalize）
- **迭代上限**：5 次

### agent.yaml 结构

```yaml
name: architect
description: ...
skills: [vllm-memory-calculator]
tools: [read, bash, skill]
schema: architect_schema

claude_code:
  model: sonnet
  extra_tools: [Read, Bash, Skill]

codex:
  model: claude-sonnet-5
  context: fork
```

`claude_code:` 和 `codex:` 分别定义平台特定的配置（模型、上下文模式等），由 `engine/agent_registry.py` 在运行时解析。

---

## Skill 层

### 技能路由

`skills/inference-model-skills/SKILL.md` 是推理领域的**路由索引**，按用户意图指引到对应的子目录：

- 部署和服务管理 → `deployment/`
- 模型与显存分析 → `model-analysis/`
- benchmark、寻优和 profiling → `performance-optimization/`

### 技能映射

| Agent | Skills | 说明 |
| ------- | -------- | ------ |
| architect | `vllm-memory-calculator` | 显存计算器，分析模型结构与显存预算 |
| developer | `environment-install` | 深度学习环境（Conda、CANN、vLLM）安装 |
| developer | `vllm-ascend-serving` | vLLM Ascend 推理服务管理（启停、健康检查） |
| tester | `vllm-ascend-benchmark` | 推理 benchmark 测试（吞吐、延迟、并发） |
| tester | `vllm-ascend-autotune-zh` | 自动参数寻优（决策驱动 + 网格搜索） |
| reviewer | `ascend-profiling-collection` | torch-profiler 数据采集（L1 级别） |
| reviewer | `ascend-msprof-analyze-cli` | msprof 分析工具（瓶颈识别、算子排序） |
| reviewer | `ascend-perf-tuning` | 性能调优建议生成 |

### Skill 目录结构

```text
skills/inference-model-skills/
├── model-analysis/
│   └── vllm-memory-calculator/
│       └── SKILL.md
├── deployment/
│   ├── environment-install/
│   │   └── SKILL.md
│   └── vllm-ascend-serving/
│       └── SKILL.md
└── performance-optimization/
    ├── vllm-ascend-benchmark/
    │   └── SKILL.md
    ├── vllm-ascend-autotune-zh/
    │   └── SKILL.md
    ├── ascend-profiling-collection/
    │   └── SKILL.md
    ├── ascend-msprof-analyze-cli/
    │   └── SKILL.md
    └── ascend-perf-tuning/
        └── SKILL.md
```

每个 SKILL.md 包含 YAML frontmatter（name、description、domain、function、tags）+ Markdown 正文（中文操作指引、命令示例）。

---

## 状态机引擎

### 设计原则

1. **YAML 是单一真相源**：所有状态转移定义在 `optimize-pipeline.yaml` 中
2. **转移日志只追加**：`state_transitions.jsonl` 永不重写，历史可审计
3. **迭代计数从日志推导**：不做带外计数，日志是唯一的计数来源
4. **Bootstrap 恢复**：空日志 + PROGRESS.md 中有 handoff → 自动推断当前状态
5. **Critic 可验证**：`verify_log()` 可重放日志，检查每条转移都在 YAML 中声明

### 核心模块

```text
engine/
├── state_machine.py        # 状态机核心
│   ├── load_state_machine()    — 加载 YAML
│   ├── next_state()            — 计算下一状态
│   ├── get_current_state()     — 日志尾部推断当前状态
│   ├── read_log() / append_log() — 只追加日志 (JSONL)
│   ├── verify_log()            — Critic 审计重放
│   └── _main()                 — CLI: next / verify / current
├── state_executor.py       # Orchestrator API 包装
│   ├── snapshot()              — 工作空间完整快照
│   ├── next_state()            — 转移 + 自动记录
│   ├── validate_log()          — 日志校验
│   └── STATE_TO_AGENT          — state → agent type 映射表
├── fsm_conditions.py       # 条件 DSL 求值器
│   ├── eval_condition()        — 条件原语求值
│   ├── extract_handoff()       — 从文本提取 handoff 行
│   └── parse_worker_signal()   — 解析 worker 信号枚举
├── agent_registry.py       # Agent 发现与注册
│   ├── discover_agents()       — 扫描 agent.yaml
│   ├── load_agent_registry()   — 按平台解析配置
│   └── load_agent_spec()       — 加载单个 Agent 定义
└── backends/
    ├── base.py              # AbstractBackend 接口 + AgentResult
    ├── claude_code.py       # Claude Code 后端 + YAML→JS 编译器
    └── codex.py             # Codex 后端（stub）
```

### 条件 DSL 原语

YAML 中的 `exit_transitions` 使用下述条件原语，由 `eval_condition()` 求值：

| 原语 | 类型 | 说明 |
| ------ | ------ | ------ |
| `handoff_match` | `string` | handoff 行以指定字符串开始 |
| `handoff_contains` | `string` | handoff 行包含指定子串 |
| `always` | `bool` | 无条件通过/失败 |
| `iter_below_cap` | `string` | 指定计数器的迭代次数 < 上限 |
| `path_exists` | `string` | 工作空间中指定文件存在 |
| `file_absent` | `string` | 工作空间中指定文件不存在（支持 glob） |
| `plugin_method` | `string\|dict` | 委托给插件方法判定 |
| `user_decision_target_in` | `list` | user_decision.md 中的目标状态在列表中 |
| `all_of` | `list` | 所有子条件同时满足 |
| `any_of` | `list` | 任一子条件满足 |

### 转移匹配规则

`exit_transitions` **从上到下顺序匹配**，取第一条条件为真的转移。每个状态必须有一个兜底分支（`always: true`），防止无转移可匹配。

### 迭代上限机制

| 计数器 | 上限 | 说明 |
| -------- | ------ | ------ |
| architect | 3 | architect 最多 3 次 |
| developer | 4 | developer 最多 4 次 |
| tester | 8 | tester 最多 8 次 |
| reviewer | 5 | reviewer 最多 5 次 |
| 总 spawn | 20 | 流水线全局上限 |

迭代次数从 `state_transitions.jsonl` 中自动统计。打上 `rollback_kind: "infra"` 标记的记录不计入迭代配额（用于 SCP 超时等基础设施回滚）。

### CLI 命令

```bash
# 计算下一状态
python -m engine.state_machine next \
  --workspace .workspace/run-001 \
  --yaml plugins/inference-model-optimize/workflows/optimize-pipeline.yaml \
  --handoff "handoff: ARCHITECT_READY"

# Critic 审计重放
python -m engine.state_machine verify \
  --workspace .workspace/run-001 \
  --yaml plugins/inference-model-optimize/workflows/optimize-pipeline.yaml

# 查看当前状态
python -m engine.state_machine current \
  --workspace .workspace/run-001 \
  --yaml plugins/inference-model-optimize/workflows/optimize-pipeline.yaml
```

---

## Workflow 运行时

### Claude Code 路径

Claude Code 下，orchestrator Agent 通过 `Agent` 工具直接调用子 Agent，不使用 Workflow JS。orchestrator 亲自执行状态机循环（见 AGENT.md 第三步），每次调用 Agent 后：

1. 提取 handoff token
2. 调用 `engine.state_machine next`（Python CLI）得出下一状态
3. 记录转移日志
4. 进入下一状态

```text
orchestrator (AGENT.md)
  └─ Agent('architect', prompt, schema=ARCHITECT_SCHEMA)
  └─ 门禁校验 (gates.js 等价逻辑)
  └─ python -m engine.state_machine next --handoff "handoff: ARCHITECT_READY"
  └─ Agent('developer', prompt, schema=DEVELOPER_SCHEMA)
  └─ ...
  └─ 最终报告
```

备用路径：`optimize-pipeline.workflow.js` 可直接在 Workflow 工具中执行，提供完整的 JS 状态机实现。

### Codex 路径

Codex 下，orchestrator 通过 Python CLI 驱动状态机引擎：

```text
用户 → $mindspeed-agent:inference-model-optimize
  → orchestrator (AGENTS.md)
  → python -m engine.state_machine next
  → Codex task/subtask 系统
```

### YAML → JS 编译

`engine/backends/claude_code.py` 提供编译命令，将 YAML 状态机编译为 JS 模块：

```bash
python -m engine.backends.claude_code compile \
  --yaml plugins/inference-model-optimize/workflows/optimize-pipeline.yaml \
  --out plugins/inference-model-optimize/workflows/state-machine-config.js
```

生成的 `state-machine-config.js` 导出 `STATE_MACHINE` 常量，供 Workflow JS 使用。

---

## 门禁校验

每个 Agent 产出必须通过门禁校验，否则重试（最多 2 次）。校验规则定义在 `gates.js` 和 orchestrator 的 AGENT.md 中：

### architect 门禁

- `model_profile` 必须包含 architecture、num_layers、hidden_size、num_parameters
- `memory_budget` 必须包含 model_weights_gb、kv_cache_gb、total_per_card_gb
- `recommended_strategy` 必须包含 tp、dp、rationale
- `hardware_requirements` 必须包含 min_npu_cards、recommended_config

### developer 门禁

- 所有字段非空
- `health_check_passed` === true（`/health` 端点返回 200）
- `models_endpoint_ok` === true（`/v1/models` 返回目标模型）
- `service_status` === "running"

### tester 门禁

- baseline_metrics 和 optimized_metrics 必须包含 output_tok_per_s、ttft_ms、tpot_ms
- comparison 必须包含 output_tok_per_s_change_pct
- optimal_params 非空
- comparison_report 非空

### reviewer 门禁

- 结构门禁：profiling_data_path、profiling_collected、diagnosis（e2e_time_ms + compute/communication/idle 占比）、bottleneck_classification、operator_top_n、optimization_suggestions、diagnosis_report 全部非空
- 证据门禁：`evidence_list` 非空，每条包含 conclusion_id、data_path、raw_value、verification_status

---

## Handoff 协议

Agent 之间通过精确的 handoff token 交接。每个 Agent 在输出末尾必须输出 YAML 规定的精确 token：

| Agent | Handoff Token | 目标状态 | 说明 |
| ------- | -------------- | --------- | ------ |
| architect | `handoff: ARCHITECT_READY` | developer | 方案完成 |
| architect | `handoff: ARCHITECT_FAILED` | abort | 方案失败 |
| developer | `handoff: SERVICE_READY` | tester | 服务就绪 |
| developer | `handoff: SERVICE_FAILED` | abort | 部署失败 |
| tester | `handoff: NEED_RESTART` | developer | 需重启服务 |
| tester | `handoff: OPTIMIZATION_READY` | reviewer/finalize | 寻优完成 |
| reviewer | `handoff: REVIEWER_CONTINUE` | tester/finalize | 有优化建议 |
| reviewer | `handoff: REVIEWER_DONE` | finalize | 诊断完成 |

Handoff 提取逻辑（`extract_handoff()`）在文本中搜索以 `→ orchestrator:`、`handoff:` 或 `@aog-` 开头的行，取最后一条匹配。

---

## 数据流

```text
用户参数
  │
  ▼
orchestrator (收集参数，args 对象)
  │
  ▼
architect
  │ model_profile, memory_budget, recommended_strategy, hardware_requirements
  ▼
developer
  │ service_url, port, model_name, tp, dp, health_check_passed
  ▼
tester
  │ baseline_metrics, optimized_metrics, comparison, optimal_params
  │ (可能产生 candidate_restart_params → 回到 developer)
  ▼
reviewer
  │ profiling_data_path, diagnosis, bottleneck_classification,
  │ operator_top_n, optimization_suggestions, evidence_list
  │ (suggestions.executable_by === 'tester' → 回到 tester 迭代)
  ▼
finalize → 最终报告 (JSON)
  │ model_profile, service_info, performance (baseline/optimized/improvement),
  │ diagnosis (bottleneck, top_n ops, suggestions)
```

### 跨阶段上下文传递

每个阶段产出存入 `priorOutputs`，后续阶段通过 brief 模板获取：

- `developer` 读取 `priorOutputs.architect.recommended_strategy`
- `tester` 读取 `priorOutputs.developer.service_url` 等
- `reviewer` 读取 `priorOutputs.developer` + `priorOutputs.tester.optimal_params`
- `finalize` 汇总所有 `priorOutputs`

### 迭代去重

tester→reviewer→tester 迭代时，`attemptedSuggestions` 记录已尝试的 reviewer 建议 ID，防止重复执行。过滤规则：

- `executable_by` 必须为 `"tester"` 或 `"developer"`
- 排除 category 为 `"算子优化"`、`"硬件升级"`、`"软件升级"` 的非 actionrable 建议

---

## 平台后端

### AbstractBackend 接口

```python
class AbstractBackend(ABC):
    def platform_name(self) -> str: ...
    def available_agents(self) -> list[str]: ...
    def spawn_agent(self, agent_type, prompt, schema, workspace, label) -> AgentResult: ...
```

### ClaudeCodeBackend

- `platform_name`: `"claude-code"`
- Agent 生成由 Workflow JS 运行时处理，Python 端不支持直接生成
- 提供 YAML → JS 编译功能，生成 `state-machine-config.js`

### CodexBackend

- `platform_name`: `"codex"`
- Agent 生成通过 Codex 的 task/subtask 系统实现（当前为 stub）
- Agent 配置从 `agent.yaml` 的 `codex:` 段读取

---

## Workspace 结构

每次流水线运行对应一个工作空间目录：

```text
.workspace/run-001/
├── state_transitions.jsonl     # 只追加转移日志（每行一个 JSON）
├── PROGRESS.md                 # orchestrator 维护的进度（含 handoff 行）
├── verification.json           # 可选 — 门禁校验快照
├── analysis.md                 # 可选 — 分析文档
└── (Agent 产生的其他文件...)
```

`state_transitions.jsonl` 的每行格式：

```json
{
  "ts": "2026-08-06T10:30:00Z",
  "from_state": "architect",
  "to_state": "developer",
  "handoff": "handoff: ARCHITECT_READY",
  "matched_transition_index": 0,
  "rationale": "方案设计完成，交付结构化部署方案，进入环境部署阶段",
  "iter_counts_snapshot": {"architect": 1}
}
```

---

## 关键设计决策

### 1. YAML 作为单一真相源

状态转移不在 Agent 指令中硬编码，而是从 YAML 读取。orchestrator 每步都调用 `engine.state_machine next`（而非凭记忆）决定下一状态。这确保了跨平台一致性：Claude Code 和 Codex 共享同一份 YAML。

### 2. 只追加日志 + 可审计

`state_transitions.jsonl` 只追加、不重写。`verify_log()` 可独立重放日志验证每步转移的合法性，支持 Critic 审计。

### 3. 结构化输出 + 门禁校验

每个 Agent 通过 JSON Schema 强制输出结构化数据。orchestrator 对每个阶段产出执行门禁校验，失败即通过追加反馈到 prompt 的方式重试（最多 2 次），确保进入下一阶段的数据完整性。

### 4. tester ↔ reviewer 迭代闭环

核心优化策略是"先寻优后分析"——tester 先找到阶段最优参数，reviewer 再对最优配置采 profiling。当 reviewer 发现可执行优化建议时回到 tester，形成迭代闭环（最多 3 轮），确保每次诊断的都是真正的算子和硬件瓶颈。

### 5. 入口可恢复

流水线支持从任意阶段进入。orchestrator 根据用户已有产出（已有部署方案 → 从 developer 开始；已有运行服务 → 从 tester 开始）灵活路由。

---

## 已知问题

### 版本号不一致

`plugins/inference-model-optimize/.claude-plugin/plugin.json` 中的版本为 `0.5.0`，但 `.claude-plugin/marketplace.json` 中注册的 inference-model-optimize 版本为 `0.4.0`。两者应保持一致，更新版本时需同步修改两个文件。

### state-machine-config.js 缺少导出

`optimize-pipeline.workflow.js` 第 17 行从 `state-machine-config.js` 导入 `{ STATE_MACHINE, nextState, initIterCounts }`，但 `state-machine-config.js`（由 `engine/backends/claude_code.py compile` 生成）仅导出 `STATE_MACHINE`，未导出 `nextState` 和 `initIterCounts`。这导致 Workflow JS 路径下这两个函数为 `undefined`。

**影响范围**：仅影响直接使用 `optimize-pipeline.workflow.js` 的 Workflow 工具调用路径。orchestrator 通过 Agent 工具直接调用子 Agent 的路径不受影响（该路径使用 Python CLI）。

**修复方向**：更新 YAML→JS 编译器，在 `state-machine-config.js` 中同时生成 `nextState()` 和 `initIterCounts()` 函数。

### CodexBackend 为 Stub

`engine/backends/codex.py` 中的 `spawn_agent()` 方法当前为 stub 实现，不实际调用 Codex API。待 Codex Python SDK 就绪后需替换为真实调用。

---

## 文件索引

```text
MindSpeed-Agent/
├── CLAUDE.md                           # 项目开发规范
├── README.md                           # 项目总览
├── docs/
│   └── ARCHITECTURE.md                 # 本文档
│
├── plugins/
│   └── inference-model-optimize/
│       ├── .claude-plugin/plugin.json  # 插件清单
│       ├── agents/
│       │   ├── orchestrator/           # 编排器 Agent
│       │   ├── architect/              # 方案设计 Agent
│       │   ├── developer/              # 环境部署 Agent
│       │   ├── tester/                 # 性能优化 Agent
│       │   └── reviewer/               # 性能诊断 Agent
│       ├── workflows/
│       │   ├── optimize-pipeline.yaml  # ★ 状态机单一真相源
│       │   ├── optimize-pipeline.workflow.js
│       │   ├── state-machine-config.js # YAML 编译产物
│       │   ├── schemas.js              # 4 阶段 JSON Schema
│       │   ├── gates.js                # 门禁校验
│       │   └── briefs.js               # Prompt 模板
│       └── skills/                     # symlink → skills/
│
├── engine/
│   ├── __init__.py
│   ├── state_machine.py               # ★ 状态机核心
│   ├── state_executor.py              # Orchestrator API
│   ├── fsm_conditions.py              # 条件 DSL
│   ├── agent_registry.py              # Agent 注册
│   └── backends/
│       ├── base.py                    # 抽象接口
│       ├── claude_code.py             # Claude Code + YAML→JS 编译器
│       └── codex.py                   # Codex 后端
│
├── skills/
│   └── inference-model-skills/
│       ├── model-analysis/vllm-memory-calculator/
│       ├── deployment/environment-install/
│       ├── deployment/vllm-ascend-serving/
│       └── performance-optimization/
│           ├── vllm-ascend-benchmark/
│           ├── vllm-ascend-autotune-zh/
│           ├── ascend-profiling-collection/
│           ├── ascend-msprof-analyze-cli/
│           └── ascend-perf-tuning/
│
├── .claude-plugin/                    # 根插件清单（Claude Code）
├── .codex-plugin/                     # 根插件清单（Codex）
├── install.sh                         # Linux/macOS 安装脚本
├── install-codex.sh                   # Codex 专用安装
└── install.bat                        # Windows 安装脚本
```
