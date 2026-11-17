---
name: ecosystem-routing
description: "This skill should be used when the user asks which MindSpeed community repository or plugin applies to their request, mentions a repository name or model nickname (mindspeed-llm / ModelLink / vllm-ascend / MindSpeed-Bridge / triton-ascend / torch-npu ...), or wants a natural-language request routed into the correct repo → plugin → agent → skill chain. It disambiguates the repository deterministically through catalog/repos.yaml aliases, detects the task dimension, assembles the candidate chain, checks whether the target on-demand unit plugin is installed, and asks the user to confirm when the intent is ambiguous. Typical triggering phrasings include Chinese requests such as: \"我想优化 mindspeed-llm 的模型\", \"MindSpeed-LLM 怎么加新模型\", \"ModelLink 训练卡住\", \"把 Qwen 适配到 MindSpeed-Bridge\", \"vllm-ascend 怎么部署/压测/调优\", \"生成一个 triton 算子\", \"这个问题该找哪个仓库或插件\"."
version: 1.0.0
domain: common
function: common
tags:
  - routing
  - knowledge-graph
  - repository
  - catalog
  - plugin
author: mindspeed-agent
---

# 生态路由：一句话 → 代码仓 → 插件 → Skill

本 skill 是 MindSpeed 生态知识图谱的**路由入口**（生态级；领域内路由见
`inference-model-skills` 等同级路由 skill，本 skill 负责跨仓/跨插件消歧与链
路组装）。对完全不懂生态的新手，把用户一句话映射为可执行候选链：

```text
repo（社区代码仓） → plugin（MindSpeed-Agent 插件） → agent → skill
```

## 触发场景速查（命中任一类即执行本 skill 路由）

- 用户提到社区仓/模型名但没指明 skill：MindSpeed、MindSpeed-LLM、ModelLink、
  MindSpeed-MM、MindSpeed-Bridge、Megatron、vLLM、vllm-ascend、torch-npu、
  CANN、triton-ascend 等；
- 用户问"用哪个仓库/插件做 XX"、"怎么优化/迁移/适配/部署/验证 XX 模型"；
- 需要区分训练框架与推理引擎（如 MindSpeed-LLM vs vllm-ascend）；
- 用户是完全新手，话术口语化、信息不全（只说模型名 + 意图）。

> 若模型因故未自动加载本 skill，会话级兜底已由根指令
> （AGENTS.md / CLAUDE.md「生态图谱路由」节）触发：先路由、后执行。

## 数据源（先读，勿凭记忆）

| 文件 | 内容 |
| --- | --- |
| `catalog/repos.yaml` | 社区代码仓注册表：id、aliases、kind、backends、urls、related |
| `catalog/plugins.yaml` | 本仓插件注册表：repo_affinity（插件处理哪些仓）、skills、workflow |
| `catalog/task-kinds.yaml` | 任务维度枚举 + trigger 词（性能优化/迁移/精度/部署/跑测/算子/资料…） |
| `catalog/graph.json` | generator 生成的统一索引（节点 + 边），可替代逐文件读取 |
| `catalog/intent-examples.yaml` | 已验证金句（回归样例），可参考其措辞模式 |

> catalog 文件是**事实源**；新增仓/改别名时先改 `repos.yaml`，不要在本 skill
> 内硬编码仓库清单。

## 路由流程（确定性优先，R1–R5）

### R1 仓库消歧

把用户话术与 `repos.yaml` 的 `id` + `aliases` 匹配（不区分大小写，忽略空格/
连字符/下划线差异；长别名优先，命中后不再被其前缀短词重复命中）：

- 命中 **1 个**仓 → 确定目标仓，进入 R2；
- 命中 **0 个** → 用话术中的模型名（如 Qwen3、DeepSeek-V3、GLM-4）、技术词
  （mcore / fsdp2 / 推理 / 算子）做语义候选 top-N，进入 R4 反问；
- 命中 **多个**（如 MindSpeed-LLM vs MindSpeed 框架 vs 推理 vLLM）→ 进入 R4 反问
  区分（列出每项一行解释），**绝不静默猜**。

口语/曾用名对照（以 repos.yaml 为准的常见例子）：

| 用户可能说 | 消歧到 |
| --- | --- |
| mindspeed-llm / MindSpeed-LLM / ModelLink | `mindspeed-llm` |
| MindSpeed-Bridge / megatron-bridge | `mindspeed-bridge` |
| vllm-ascend / vLLM 昇腾 | `vllm-ascend`（推理，非训练） |
| MindSpeed 加速库（仅"训练"无 llm 后缀时确认） | `mindspeed` |
| triton / Triton-Ascend | `triton-ascend`（算子场景） |
| MindSpeed-MM / 多模态 | `mindspeed-mm` |

### R2 任务维度

从话术抽 trigger 词映射到 `task-kinds.yaml`（perf-optimize / migrate / align /
deploy / benchmark / diagnose / ops-dev / doc / validate）。多义（如只说了
"优化"）时不要自选，列出该仓可用的任务候选，进入 R4。

### R3 链路组装

目标仓 + 任务维度 → `plugins.yaml` 的 `repo_affinity` ∩ TaskKind 能力列 →
输出候选链 `repo → plugin → agent → skill`，并附**一行中文解释**说明依据
（哪个仓、为什么是优化/迁移/…、将由哪个插件执行）。

### R3.1 执行形态判定（plugin 是否带状态机）

`catalog/plugins.yaml` 每个插件声明 `execution`：

- `execution: state-machine`（如 `inference-model-optimize` / `ops` / `doc-agent` /
  `veomni-agent`）：插件自带 workflow 状态机（`workflow` 路径 + `entry_agent` +
  `agents`）。此时**路由终点不是单个 skill**，而是插件 workflow 入口
  `entry_agent`（如 `inference-model-optimize` 的 `orchestrator`）：
  候选链为 `repo → plugin → workflow(entry_agent) → agents → skills(知识底座)`，
  由插件按 `workflows/*.yaml` / `engine/state_machine.py` 编排执行；插件登记的
  `skills` 只是各阶段 agent 按需加载的知识，不作为路由终点。
- `execution: skills`（缺省）：纯技能聚合，路由终点为单个 skill（原 R3 语义）。

**单点任务例外**：若用户意图只是单一原子能力（仅显存估算、仅起/停服务、仅一次
benchmark、仅采集/分析 profiling、仅单个算子），即使命中 state-machine 插件也
**直接委派对应 skill/agent**，不强行拉起完整流水线（与 orchestrator AGENT.md
「委派边界」一致）。端到端优化（部署→寻优→诊断→实施）才进入 workflow。

### R3.2 状态机插件 entry 解析

命中 `state-machine` 插件且确需完整流水线时，从 `catalog/plugins.yaml` 读
`workflow`、`entry_agent`、`agents`，并按平台差异执行（见下）；graph.json 的
`edges.plugin_workflows` / `edges.plugin_agents` 提供同一关系。

### R3.5 安装就绪检查（按需安装）

本仓库采用 **core + 领域 unit 按需安装**（见 `docs/install-on-demand-design.md`）：
`ecosystem-routing` 属于 core（默认已装），领域技能按 unit 分发
（`mindspeed-llm` / `inference-model-optimize` / `ops` / `doc-agent`）。R3 命中
unit 后，先检查该 unit 是否已装：

- `python tools/install-unit.py status` → 查看各 unit 状态（core 恒装，on-demand
  看 `install.default_installed`）；
- 目标 unit **已装** → 进入其执行链：
  - `execution: state-machine`（R3.1）→ 进入插件 workflow 入口（`entry_agent`，
    见 R3.2 与「平台差异」），不落到单 skill；
  - `execution: skills` → 直接进入该插件登记的 skill；
- 目标 unit **未装** → 向用户给出该 unit 的按需安装命令（读
  `catalog/plugins.yaml` 的 `install.install_hint`，勿硬编码），例如：

  | 平台 | 安装 mindspeed-llm |
  | --- | --- |
  | Claude Code | `claude plugin install mindspeed-llm@MindSpeed-Agent` |
  | Codex | `codex plugin add mindspeed-llm@MindSpeed-Agent` |
  | DSH | `python tools/install-unit.py dsh mindspeed-llm` |

  执行后提示：**Claude Code / Codex / DSH 均需新开会话**方可加载新装 skills
  （平台不保证会话内热加载）；待用户确认新会话就绪后再继续任务，或把目标
  请求原样保留给下一个会话执行。

### R4 反问确认

- 候选唯一（且目标 unit 已装）→ 按执行形态进入：state-machine 插件进入
  workflow 入口，skills 插件进入对应 skill（R3.1/R3.5）；单点任务按 R3.1 例外
  直接委派对应 skill/agent；
- 候选不唯一 → 向用户给出"多选一 + 每项一行解释"的确认问题。金句示例：
  `"我想优化 mindspeed-llm 的模型"` → 反问：
  1. 训练侧性能 / FSDP2 融合算子优化 → `mindspeed-llm-fsdp2-main-control` 族
  2. 开源模型迁入该仓（MCore/FSDP2）→ `mindspeed-llm-mcore-open-model-migration` 等
  3. 精度对齐 → `align-mindspeed-transformers-precision` 等

  用户选定后回到 **R3.5** 做安装就绪检查再进入执行。

  推理场景示例：`"优化 qwen3-4b 推理模型吞吐（昇腾 vLLM 服务）"` → 反问她希望
  完整流水线（部署→寻优→profiling→实施，`inference-model-optimize` orchestrator
  状态机）还是单点动作（仅 benchmark / 仅 autotune / 仅诊断）；用户选完整流水线
  后进入该插件 workflow 入口。

### R5 兜底

图谱无命中 → 退化为各平台原生语义匹配（读 SKILL.md `description`），并把本次
话术记录为"待标注意图"（供人工补进 `catalog/intent-examples.yaml`，勿自行写入
未经确认的金句）。

## 平台差异

- **Claude Code / DSH / Codex**：本 skill 属于 core，三平台共用同一份正文与
  catalog，无需重复实现；领域 unit 安装命令见 §R3.5；
- 路由完成后调用目标插件（如 `inference-model-optimize`、`ops`、`doc-agent`）
  或目标 skill 时，遵循其自身的状态机/流水线（engine/state_machine.py、
  workflows/*.yaml），本 skill 只负责路由与安装就绪检查，不代行其流程；
- **state-machine 插件的平台入口**（R3.1/R3.2 命中后）：
  - `inference-model-optimize`：入口 agent = `orchestrator`
    （`plugins/inference-model-optimize/agents/orchestrator/AGENT.md`），它亲自充当
    状态机引擎，按 `workflows/optimize-pipeline.yaml` 逐个委派 architect →
    developer → tester → reviewer/optimizer；对应 skill（benchmark/autotune/
    profiling）由各阶段 agent 按需加载；
  - `ops`：入口 agent = `ops-design-agent`（`plugins/ops/agents/ops-design-agent/`），
    按 `plugins/ops/ops.yaml`（design → implement → optimize）执行；
  - `doc-agent`：入口 agent = `doc-orchestrator`
    （`plugins/doc-agent/agents/doc-orchestrator/`），按
    `plugins/doc-agent/engine/doc_pipeline.yaml` 执行；
  - `veomni-agent`：入口 agent = `veomni-orchestrator`
    （`plugins/veomni-agent/agents/veomni-orchestrator/`），按
    `plugins/veomni-agent/veomni.yaml` 执行。
  各插件入口 agent 的平台调度方式见其自身 AGENT.md（Claude Code 用 Agent 工具 /
  Codex 用 task/subtask / DSH 用 subagent）。

## 校验与维护

- 改 catalog 后运行：`python catalog/generator.py && python catalog/lint_catalog.py`
  确保别名无冲突、skill targets-repo 引用合法、金句断言通过；
- CI 侧由 `ci/guard.py`（`ci/checks/knowledge_graph.py`）兜底。
