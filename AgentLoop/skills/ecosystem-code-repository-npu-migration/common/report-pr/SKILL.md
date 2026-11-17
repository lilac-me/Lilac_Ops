---
name: report-pr
description: This skill should be used when the user asks to turn current Ascend NPU migration commands, logs, import provenance, diffs, semantic coverage outputs and review findings into conservative engineering reports or PR-ready text. Use after implementation or validation when results must be summarized without changing repository documentation or inventing support claims.
version: 0.16.0
domain: ecosystem-code-repository-npu-migration
function: common
tags:
  - ascend
  - npu
  - migration
  - pytorch
author: addsubmuldiv
---

# NPU 迁移报告

## 角色

根据当前任务生成有证据支撑的报告或 PR 描述。直接命令和日志是一级证据；不要求也不接受 controller 产物。

## 使用场景

当用户在环境设置、源码改动或运行时验证后需要内部报告、审查摘要、验证矩阵或 PR 文本时使用。

## 工作流程

1. 检查当前 checkout 和 diff。收集精确解释器、命令、cwd、退出码、有界日志、指标/输出、导入路径和明确未测试范围。
2. 在可用时阅读 feature-model、本地资产、迁移进度、兼容性或既有报告文件；将其中笔记与当前 checkout 和命令进行比对。
3. 分开记录环境就绪、静态/导入检查、组件 runtime、仓库原生 smoke 和代表性覆盖。
4. 将每个请求的 workload 分类为 supported、partial、blocked、excluded 或 unverified，并说明每项成功检查证明了什么、不能证明什么。
5. 对仓库级目标闭环，分别报告任务处置（`COMPLETE`、`IN_PROGRESS` 或 `STOPPED`）和迁移结果（`SUPPORTED`、`PARTIAL` 或 `BLOCKED`）。`COMPLETE` 可以与具体 blocker 共存，绝不能升级迁移结果。
6. 使用 `templates/` 提供请求的格式。只有在用户要求文件或文件确实有用时，才写入 `.npu-agent/reports/`；否则直接返回文本。

## 边界

不要修改目标源码或文档、重跑迁移命令、创建缺失证据、更新迁移工作笔记、从笔记内容推断 PASS，或从更窄的 smoke 声称已覆盖分布式/task family。

## 完成条件

返回请求的报告或 PR 文本，并包含改动文件及理由、精确的成功/失败命令、限制和复现步骤。报告目标闭环时，同时包含两个最终维度，并明确指出缺少的真实 NPU 证据。

## 资源

- [reporting_policy.md](references/reporting_policy.md)：结论和证据规则。
- `templates/`：报告和 PR 的结构模板。
- `examples/`：脱敏措辞示例，绝不是运行时证据。
