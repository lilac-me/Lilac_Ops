---
name: doc-generate
description: >-
  This skill should be used when the user asks to write or generate a
  document from a design plan. It fills required sections chapter by
  chapter with evidence gathered from the target repository, forbids
  content placeholders, keeps code comments as first-class documentation
  content, and hands off illustration requests to humans (no AI-generated
  images).
version: 1.0.0
domain: doc
function: generation
tags:
  - doc
  - generate
  - writing
author: fighting_zhen
user-invocable: false
---

# 资料生成（doc-generate）

## 功能概述

按 design_plan.json 的粗粒度标题从目标仓实测取证逐章生成文档。章节内部内容按目标仓场景灵活撰写，不受模板细节约束。

## 使用方式

由 doc-writer Agent 调用（场景 A write 阶段，此时 design_plan.json 已经用户人工确认）。

## 输入

- design_plan.json（**用户已确认**的 required_sections / optional_sections / figure_baseline；确认记录见 design_approval.md）
- context_report.json（共享事实基线）

## 操作步骤

1. 按 design_plan.json 的 required_sections（用户已确认章节清单）建立 H2 章节框架；optional_sections 按仓取舍并记录依据；**不得擅自增删 H2 标题**。
2. 逐章撰写：从目标仓实测取证（代码/脚本/配置/记录），不照抄其他仓库文档。
3. 代码块纪律：命令可复制执行；占位符 `<...>` 给出实际值或说明；`#`、`//` 注释属于资料内容，保留并纳入审查。
4. 需要插图处插入统一占位标记：`![图 {文档}-{n}：{用途}（待人工插图）](figures/{文档}/{n}.png)`；不生成、不编造图片。
5. 以 design_plan.figure_baseline 为基线产出 figure_requests.json（no/section/purpose/elements/form/alt_text/status，不得增删图号），正文占位与清单一一对应。
6. 链接处理：同仓相对链接优先；跨仓链接遵循版本分支规则；新文档登记 README 对应入口章节。
7. 自检：设计者自拟章节均已覆盖、无 TODO/内容占位符（插图占位除外且已登记）、术语统一。
8. 输出 draft_{doc}.md + figure_requests.json，文件路径交接给审查阶段。

## 输出格式

`draft_{doc}.md`（文档本身）+ `figure_requests.json`，Schema 见 `contracts/figure_requests.json`。

## 错误处理

- 信息源缺失：标记缺失项，请求补充上下文。
- 无法验证的命令/版本：标注"需实测验证"，禁止编造。
- 插图/截图无法验证：登记为诉求并标注"待人工插图"，禁止编造或 AI 生成。

## 参考信息

- `references/writing_guidelines.md`：通用写作规范（章节层级、表格、链接、命名、语言、插图规范）
