---
# ============================================================
# SKILL.md 模板 — 复制此文件到 skills/<your-skill-name>/ 目录
# ============================================================
#
# YAML frontmatter 部分：定义 skill 的元信息
# 字段分为必填（✅）、推荐（⭐）、可选（🔧）三类
# ------------------------------------------------------------

# ✅ 必填：skill 唯一标识符，kebab-case，与目录名完全一致
name: your-skill-name

# ✅ 必填：英文，第三人称，≤1024 字符
# 格式：以 "This skill should be used when the user asks to..." 开头
# 写清楚用户会在什么场景/说什么话触发此 skill，越具体越好
description: >-
  This skill should be used when the user asks to [具体做什么].
  [补充典型触发场景和关键词].

# ⭐ 推荐：语义版本号（semver）
version: 1.0.0

# ⭐ 推荐：领域（一级分类），从以下领域中选择
# llm-model          — LLM 模型迁移调优
# mm-model           — 多模态模型迁移调优
# rl-model           — 强化学习模型迁移调优
# inference-model    — 推理模型部署优化
# ops                — 算子开发调优
# common             — 通用基础工具
domain: llm-model

# ⭐ 推荐：功能类别（二级分类），从以下选择
# deployment              — 部署类（环境搭建、服务部署）
# model-analysis          — 模型分析类（性能采集、瓶颈分析、模型结构分析）
# model-adaptation        — 模型适配类（GPU→NPU 迁移适配、格式转换）
# alignment               — 精度对齐类（GPU→NPU 精度对比）
# performance-optimization — 性能优化类（量化压缩、算子融合、混合精度）
# community-operations    — 社区运营类（案例分享、最佳实践沉淀）
# common                  — 通用类（跨领域公共能力）
function: performance-optimization

# ⭐ 推荐：搜索/过滤标签，最多 5 个
tags:
  - mindspeed
  - ascend
  - training

# ⭐ 推荐：作者标识（GitHub 用户名或团队名）
author: your-github-username

# ------------------------------------------------------------
# 🔧 可选字段（如不需要，删除或注释掉）
# ------------------------------------------------------------

# 🔧 model：指定使用哪个模型
# model: claude-sonnet-4-20250514

# 🔧 agent：指定 agent 类型
# agent: Explore

# 🔧 context：独立的执行上下文
# context: fork

# 🔧 user-invocable：是否允许用户直接 / 调用，默认 true
# user-invocable: true

# 🔧 allowed-tools：限制可使用的工具
# allowed-tools: "Read, Bash(git:*)"

# 🔧 license：skill 的许可证
# license: MIT

# 🔧 compatibility：兼容性声明，≤500 字符
# compatibility: MindSpeed >= 1.0.0, CANN >= 8.0

# 🔧 metadata：任意扩展元数据
# metadata:
#   zh-description: "当用户请求 [中文补充描述] 时使用此 skill。"
#   framework: "PyTorch"

# 🔧 hooks：工具调用钩子
# hooks:
#   PreToolUse:
#     - matcher: "Bash"
#       hooks:
#         - type: "command"
#           command: 'echo "running command"'
---

# [Skill 名称]

<!--
  正文部分：使用简体中文编写
  建议章节结构（可按需调整）：
-->

## 功能概述

简要说明该 skill 的用途、解决的问题和适用场景。

## 使用方式

在 Claude Code 中，可通过以下方式调用：

- 直接输入 `/mindspeed-agent:<skill-name>`
- 或由 Claude 根据上下文自动匹配调用

## 操作步骤

### 步骤一：准备工作

说明 skill 执行前需要满足的条件、需要的信息或环境要求。

### 步骤二：核心操作

具体的操作指令。给 Claude 明确的、可执行的步骤，而非模糊的建议。

- 使用清晰的列表或编号步骤
- 包含代码示例或命令模板
- 说明不同条件下的分支逻辑

### 步骤三：验证结果

如何确认操作已成功完成？预期的输出是什么？

## 注意事项

- 常见错误及处理方式
- 权限或环境依赖
- 与 MindSpeed/Ascend/CANN 版本的兼容性

## 参考信息

- 相关文档链接
- API 参考
- 同类 skill 的关系说明
