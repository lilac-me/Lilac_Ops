---
name: doc-orchestrator
description: >-
  This skill should be used when the user asks to generate, review, update,
  or check documentation in a repository. It provides the engine CLI usage,
  handoff tokens, workspace layout, and depth decisions that drive all
  documentation workflows.
version: 1.0.0
domain: doc
function: common
tags:
  - doc
  - orchestration
  - workflow
author: fighting_zhen
user-invocable: true
---

# 资料工作流编排（doc-orchestrator）

## 功能概述

资料任务编排主入口：场景识别、状态机驱动、深度决策与修复循环节奏。流程求值由引擎完成，本 Skill 只提供引擎使用方法与调度规则。

## 引擎使用方法（CLI）

工作流开始时先按 `references/platform_runtime.md` 解析插件根绝对路径并记为 `<DOC_AGENT_ROOT>`。Claude Code 使用宿主注入的 `CLAUDE_PLUGIN_ROOT`；Codex 从已加载 Skill 的来源路径反推，不假设存在 `CODEX_PLUGIN_ROOT`。每个文档或审查分组必须先创建独立 run；同一仓库的不同任务不得共用 ledger、issue 或报告文件。

```bash
# 创建隔离 run（不传 --run-id 时由引擎生成 12 位 ID）
python "<DOC_AGENT_ROOT>/engine/state_machine.py" new \
  --workspace .doc-work/<repo> --doc <doc>
```

随后使用命令输出的 `<run-workspace>`，或使用同一个 `--workspace .doc-work/<repo> --run-id <run-id>` 组合推进。引擎命令如下：

```bash
# 推进：以某产物就绪为条件进入下一状态
python "<DOC_AGENT_ROOT>/engine/state_machine.py" next \
  --workspace <run-workspace> --yaml "<DOC_AGENT_ROOT>/engine/doc_pipeline.yaml" \
  --handoff "handoff: CONTEXT_READY"

# 查询：当前状态与可转移
python "<DOC_AGENT_ROOT>/engine/state_machine.py" current --workspace <run-workspace> --yaml "<DOC_AGENT_ROOT>/engine/doc_pipeline.yaml"

# 校验：重放 ledger.jsonl，验证每条转移都在状态机中声明
python "<DOC_AGENT_ROOT>/engine/state_machine.py" verify --workspace <run-workspace> --yaml "<DOC_AGENT_ROOT>/engine/doc_pipeline.yaml"
```

## Handoff token 表

引擎取 Agent 输出末尾最后一条 `handoff: XXX` 行作为转移条件：

| 状态 | 出边 | token |
| ------ | ------ | ------- |
| init | → context | `handoff: CONTEXT_READY` |
| init | → plan | `handoff: PLAN_READY` |
| context | → design | `handoff: DESIGN_READY` |
| design | → confirm | `handoff: DESIGN_DONE` |
| confirm | → write | `handoff: USER_APPROVED` |
| confirm | → design | `handoff: USER_REJECT` |
| write | → review | `handoff: REVIEW_READY` |
| plan | → review | `handoff: REVIEW_READY` |
| review | → fix | `handoff: NEED_FIX` |
| review | → finalize | `handoff: PASSED` |
| review | → finalize | `handoff: MANUAL_INTERVENTION` |
| fix | → review | `handoff: FIXED` |
| fix | → finalize | `handoff: MANUAL_INTERVENTION` |
| finalize | → done | `handoff: DONE` |

## 场景识别

| 用户输入特征 | 判定场景 | 流程 |
| ------------- | --------- | ------ |
| 新建/生成/创建 + 文档主题 | 场景 A（新文档） | init → context → design → confirm（人工确认 H2 清单）→ write → review ⇄ fix(≤3) → finalize → done |
| 审视/检查/审查 + 仓库名 | 场景 B（审视已有文档） | init → plan → review ⇄ fix(≤3) → finalize → done |

- 场景 A：context 产物 context_report.json → design 产物 design_plan.json（H2 章节清单）→ **confirm 人工确认** → write 产物 draft_{doc}.md + figure_requests.json → review 产物 quality_report.json。
- 场景 B：plan 阶段完成文档清单与亲和分组（按内容相关性聚类：主题/对象/术语/文档类型，交叉引用同组，每组 3~8 个，组内 >8 拆组）；逐组进入 review。
- **全局范围约束**：`docs/en/**` 出域（由专门翻译团队负责）——上下文收集、文档清单、lint 与语义审查均不纳入英文树内容。

## 人工确认环节（confirm，仅场景 A）

1. design 产出 design_plan.json 后，doc-orchestrator 向用户展示 H2 章节清单（required_sections / optional_sections），**等待用户确认，不自动流转**。
2. 用户确认 → 写 `design_approval.md`（确认后的章节清单 + 用户确认说明 + 时间戳）→ 推进 `handoff: USER_APPROVED`。
3. 用户提出修改 → 交 doc-designer 按意见修订 design_plan.json → 再次展示并等待确认；或 `handoff: USER_REJECT` 回 design 重设计。
4. 门控强制：`USER_APPROVED` 转移要求 `design_approval.md` 存在且 design_plan.json 契约校验通过——未确认无法进入 write。
5. 场景 B（审视存量文档）不涉及 confirm。

## depth 决策

| 深度 | 审查内容 | 适用 |
| ------ | --------- | ------ |
| L2 | lint + 5 reviewer 全对抗（correctness-code / correctness-semantic / completeness / clarity / accessibility） | 新文档（默认） |
| L1 | lint + correctness-code + completeness 两个 reviewer | 小修改、单章更新 |
| L0 | 仅 lint | 纯格式/链接修正 |

- 新文档默认 L2；小修改/存量文档默认 L0/L1；用户可显式覆盖 depth。

## workspace 布局

工作产物统一在 `.doc-work/<repo>/`，不污染目标仓：

```text
.doc-work/<repo>/
└── <run-id>/
    ├── run.json
    ├── context_report.json
    ├── design_plan.json
    ├── design_approval.md      # 人工确认记录（confirm 阶段产出）
    ├── draft_{doc}.md
    ├── figure_requests.json
    ├── lint_report.json   # 当前文档版本的确定性 lint 完整结果
    ├── review_report.json # depth 要求的 reviewer 完成证明
    ├── code_verify.json   # L1/L2 代码块逐块验证结果
    ├── quality_report.json
    ├── issues.jsonl      # 语义 issue 生命周期记录（只追加）
    └── ledger.jsonl      # 阶段账本（只追加）
```

多文档任务按文档或审查分组分别创建 run，避免单例产物互相覆盖。

## 修复循环节奏

- review ⇄ fix 循环计数由引擎从 ledger.jsonl 推导（<3 轮），不在 prompt 带外计数。
- 每轮：review 未达标 → `handoff: NEED_FIX` → doc-updater 单轮修复 → `handoff: FIXED` → 重审。
- 通过 → `handoff: PASSED` → finalize → `handoff: DONE`；第 3 轮仍不通过 → `handoff: MANUAL_INTERVENTION`，输出遗留问题清单供人工处理。
- L1/L2 代码验证出现 `manual` 或 `fail` 时不得发送 `PASSED`；质量报告为 `manual_intervention` 时直接发送 `MANUAL_INTERVENTION`。
- 共享事实基线：context_report.json 一次性生成，不因单个 Agent 局部发现反复重建；补充信息经变更记录追加。

## 平台调度

- Claude Code 使用 Agent 工具调度插件角色。
- Codex 读取 `<DOC_AGENT_ROOT>/agents/<role>/agent.md`，使用宿主原生子任务/子 Agent 能力调度；若当前宿主没有该能力，则由编排者按相同契约顺序执行。
- DSH 读取 `<DOC_AGENT_ROOT>/agents/<role>/agent.md`，使用 `subagent` 工具调度角色；引擎 CLI（`next`/`current`/`verify`）原样复用，门控、轮次、评分仍由引擎计算。
- reviewer 之间保持独立，不互相传递发现；所有平台都由引擎计算门控、轮次和评分。
- 根目录解析和平台差异详见 `references/platform_runtime.md`。

## 错误处理

- 引擎 CLI 失败：检查 workspace 路径与 token 拼写，用 `current` 复位后重试一次。
- 上下文收集失败：重试 ≤3 次并扩大搜索范围，仍失败标记缺失项并询问用户。
- 单组审查超时：标记"审查不完整"，不阻塞其他组。
- 中断恢复：用 `current` 查询当前状态从断点继续，不重新开始整个流程。
