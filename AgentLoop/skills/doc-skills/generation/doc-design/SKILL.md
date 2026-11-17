---
name: doc-design
description: >-
  This skill should be used when the user asks to design documentation for
  a repository. It derives the repository-level document portfolio from
  the manifest (required documents plus capability-driven conditional and
  user-requested optional ones), then outputs per-document design plans
  with freely proposed section lists.
version: 1.0.0
domain: doc
function: generation
tags:
  - doc
  - design
  - portfolio
author: fighting_zhen
user-invocable: false
---

# 资料设计（doc-design）

## 功能概述

把资料需求转化为"仓库资料组合 + 单篇设计决策"，输出 design_plan.json。**约束模型：仓库级必备资料清单**（`<DOC_AGENT_ROOT>/templates/repo_manifest.yaml`）——只约束"仓库需要哪些资料"，不约束每份资料的内部目录结构；章节清单由设计者按目标仓场景自拟。`<DOC_AGENT_ROOT>` 由 doc-orchestrator 在工作流开始时解析并传递。

## 使用方式

由 doc-orchestrator 编排调用（场景 A design 阶段），或由 doc-designer Agent 调用。

## 输入

- context_report.json：仓库形态（模型仓/加速库/适配器仓）、仓库能力（是否承载模型内容、是否提供工具链）、文档主题、目标受众、参考来源（PR/Issue/已有文档）。
- 用户输入诉求：可选资料的增删（`--include/--exclude` 语义）、**目标仓性质**（昇腾自研开源仓 / 参与业界主流开源社区的三方仓——引导用户提供，不默认臆测）。

## 操作步骤

1. 判定仓库形态（模型仓/加速库/适配器仓）与目标仓性质：昇腾自研开源仓，或参与业界主流开源社区的三方仓（引导用户输入确认）。
   - **仅当目标仓为参与开源三方社区的三方仓时**，才判定协作模式 `ascend_dir` / `inline`；
   - 昇腾自研开源仓不涉及协作模式，`contribution_mode` 记 `none`。
2. 从 `templates/repo_manifest.yaml` 计算仓库资料组合：
   - 必备（required）：全部适用；`when: repo_has_models` / `when: repo_has_toolchain` 按仓库能力启用；
   - 可选（optional）：默认不强制；按用户输入诉求动态增删；
   - 组合判定细则见 `references/portfolio_guide.md`。
3. 对组合中的每份资料输出单篇设计决策（design_plan.json）：
   - `type_id` = manifest 条目 id（D-XXX）；
   - `required_sections` / `optional_sections` = 设计者自拟的章节清单（参考同仓同类资料惯例与目标受众，无骨架约束）；
   - 登记插图基线（no/section/purpose/status 等字段，见 `contracts/design_plan.json`）。
4. 输出：每份资料在独立 run 中生成一个 design_plan.json；组合清单随上下文交接，不在同一 run 中覆盖多个文档的单例产物。

## 输出格式

`design_plan.json`，Schema 见 `contracts/design_plan.json`；多文档任务每文档单独创建 run、各自产出一份。

## 错误处理

- 仓库能力无法判定：依据仓库形态（模型仓→repo_has_models）确认，仍不确定时询问用户。
- 目标仓性质（昇腾自研开源仓 / 参与开源三方社区）无法判定：引导用户提供输入，不默认按三方仓处理。
- 用户诉求与 manifest 冲突：以用户诉求为准并记录理由（可选资料的增删是使用者的权利）。
- repo_manifest.yaml 缺失：报告并回退。

## 参考信息

- `references/portfolio_guide.md`：仓库形态与能力判定、必备/条件/可选资料组合规则、三方仓协作模式
