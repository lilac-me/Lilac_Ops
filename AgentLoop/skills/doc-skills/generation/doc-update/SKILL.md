---
name: doc-update
description: >-
  This skill should be used when the user asks to fix, update, or repair a
  documentation file according to review feedback. It applies open issues
  from quality_report.json by severity, respects collaboration-mode modification
  boundaries, and performs a single repair round without controlling the
  review loop.
version: 1.0.0
domain: doc
function: generation
tags:
  - doc
  - update
  - fix
author: fighting_zhen
user-invocable: false
---

# 资料更新（doc-update）

## 功能概述

按 issue 账本的 open 条目逐项修复文档，遵守协作模式修改边界，只做单轮修复；循环控制归引擎与 doc-orchestrator。

## 使用方式

由 doc-updater Agent 调用，接收"quality_report.json 当前 open issues + 目标文档 + 协作模式"；issues.jsonl 仅用于语义问题生命周期。

## 输入

- quality_report.json（合并 lint 与语义审查后的 open 条目：rule_id、severity、location、suggestion）
- issues.jsonl（仅当修复语义问题时用于追加生命周期状态）
- 目标文档路径
- 协作模式与修改边界：自研仓（`contribution_mode: none`，在本仓范围内修改中文资料——绝大多数位于 docs/zh/，examples/tests/docker 等位置可能存在零星文件，同属可修改范围）；三方仓（ascend_dir 只在约定目录内修改 / inline 不修改上游仓共享文件）

## 操作步骤

1. 筛选 status: open 条目，按 critical → major → minor 排序。
2. 逐项修复：
   - critical：立即修复，修复后必须复核。
   - major：逐项修复；语义类标注修复依据（代码引用或实测验证）。
   - minor：批量修复，可由单次更新覆盖。
3. 定位：按 (rule_id, location.file, location.line/anchor) 定位原文，只改问题相关段落，不做无关重构。
4. 修改纪律：不改变标题结构与已确认章节清单；不引入新占位符；插图占位与 figure_requests.json 同步更新。代码块注释与正文同等对待，不得整条删除；**注释确有错误时按问题条目修正内容**（修正依据引用代码行为/参数语义），不因"注释有错"而剥离注释。
5. 遵守协作模式修改边界：自研仓（none）在本仓范围内修改中文资料（绝大多数位于 docs/zh/，examples/tests/docker 等位置的零星文件同属可修改范围），不越界到他仓；三方仓 ascend_dir 只在约定目录内修改、inline 不修改上游仓共享文件；越界问题改为输出建议。
6. 单轮修复完成即交接 `handoff: FIXED`；验证由 doc-quality-check 重审执行，本 Skill 不运行 lint/评分脚本。

语义问题的修复结果应由具备 Bash 权限的编排层通过引擎追加生命周期状态（不直接改写 `issues.jsonl`）；lint 问题不追加状态，由下一轮 lint 是否重新发现决定：

```bash
python "<DOC_AGENT_ROOT>/engine/state_machine.py" issue-status \
  --workspace <run-workspace> --rule-id <rule-id> --file <file> \
  [--line <line> | --anchor <anchor>] --status fixed
```

修复完成追加 `fixed`；重审确认后追加 `verified`，最终交付可追加 `closed`；未修复或重审重新发现的问题追加 `open`。

## 错误处理

- 条目信息不足（缺位置/建议）：标记"无法定位"，不猜测修改。
- 修复引入新问题：回退该处修改并重新处理。
- 越界修改需求：输出建议交用户/对应仓库 owner。

## 参考信息

- `references/fix_strategy.md`：修复策略细则（等级处理、修复依据、修改边界、验证）
