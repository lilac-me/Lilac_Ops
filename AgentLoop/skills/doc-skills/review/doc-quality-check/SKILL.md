---
name: doc-quality-check
description: >-
  This skill should be used when the user asks to assess documentation
  quality across four dimensions (correctness, completeness, clarity,
  accessibility). It runs the deterministic lint layer first, dispatches
  reviewers by review depth, applies voting adjudication bound to
  rule_ids, and produces the quality report via the engine.
version: 1.0.0
domain: doc
function: review
tags:
  - doc
  - quality
  - review
author: fighting_zhen
user-invocable: false
---

# 资料质量检查（doc-quality-check）

## 功能概述

两层审查：确定性 lint 层（脚本，永远先行）+ 语义审查层（LLM，按深度调度）。lint 产出 lint_report.json，reviewer 的完成状态写入 review_report.json、语义 issue 写入 issues.jsonl；引擎只在全部前置产物完整且与当前文档版本一致时生成 quality_report.json。

## 使用方式

由 doc-quality-check Agent 调用；用户直接调用时需提供文档或仓库范围。

## 输入

- 待审文档路径、design_plan.json、depth（L0/L1/L2）
- rules/rules.yaml（规则目录：rule_id + 维度 + 类型等级 + 扣分权重 + 主责维度）

## 评分模型（机器计算以引擎 score 为准）

```text
总分 = correctness×40% + completeness×25% + clarity×20% + accessibility×15%
维度最低分：correctness 90 / completeness 80 / clarity 60 / accessibility 60
通过：总分 ≥ 80 且各维度 ≥ 最低分（verdict == "pass"）
```

> 数值沿用既有评分模型口径；实际计算以 `<DOC_AGENT_ROOT>/engine/state_machine.py score` 输出为准（`<DOC_AGENT_ROOT>` 的解析方法见 doc-orchestrator 的 `references/platform_runtime.md`，阈值见 rules/rules.yaml 的 scoring 段）。

## 操作步骤

1. 运行 lint 层：`python "<DOC_AGENT_ROOT>/linters/doc_lint.py" lint --repo <目标仓> --doc <文档路径> --manifest "<DOC_AGENT_ROOT>/templates/repo_manifest.yaml" --capabilities <models,toolchain 按仓库能力> [--include D-EVAL] [--exclude D-ACKS] [--self-developed] --rules "<DOC_AGENT_ROOT>/rules/rules.yaml" --depth <L0|L1|L2> --out <run-workspace>/lint_report.json`。退出码 0/1 均为完整执行，2 或 failed 报告必须停止评分。
2. 按 depth 调度 reviewer：L0 不调度；L1 调度 correctness-code + completeness；L2 调度 5 个 reviewer 全对抗，互不知晓结果。
3. 收集 reviewer 以最终消息回传的 issue 清单；将 depth 对应的 required/completed reviewer 和当前文档 SHA-256 写入 review_report.json。缺失必需 reviewer 时报告为 failed，不得评分。
4. 聚合：同一问题按 (rule_id, location.file, location.line) 判同（line 缺失用 anchor）。
5. 投票裁决：≥3 位独立发现 → critical、2 位 → major、1 位 → minor（专长维度可升级）；类型等级取高——类型等级为 critical 的规则命中即 critical，不因投票少降级。
6. 仅将语义 reviewer 仍发现的问题追加 `status: open`；lint issue 由 lint_report.json 直接消费，不重复入账。上一轮语义问题修复后通过 `issue-status` 追加 verified。L1/L2 的 code_verify 必须包含当前文档 SHA-256，并以真实文件路径和 fenced code block 起始行逐块对应，evidence 非空。
7. 生成质量报告：`python "<DOC_AGENT_ROOT>/engine/state_machine.py" score --workspace <run-workspace> --rules "<DOC_AGENT_ROOT>/rules/rules.yaml" --depth <L0|L1|L2> --doc <文档路径>`。lint/review/code 前置产物缺失、损坏、未完成、depth 不符或文档哈希过期时，score 必须失败且不生成 quality_report.json。
8. 回传结论：pass → `handoff: PASSED`（引擎 PASSED 门控要求全部 code_verify 块为 pass，L0 豁免）；manual_intervention → `handoff: MANUAL_INTERVENTION`；fail → `handoff: NEED_FIX` 交 doc-updater。

## 纪律

- lint 永远先行；确定性 issue 不重复让 reviewer 语义判断。
- reviewer 只以最终消息回传清单，不写文件；本 Skill 统一写账本。
- 语义发现必须映射 rule_id；无匹配规则的发现进规则提案流程（rules/README.md），不计分。
- 跨维度去重：非主责维度命中同一对象只记录不扣分（主责见 rules.yaml）。

## 错误处理

- lint 脚本失败：检查路径与参数重试一次，仍失败保留 failed lint_report 并回报，不评分。
- reviewer 结果缺失：重试 1 次，仍缺失写 failed review_report 并回报，不评分。
- 目标文档不可读：标记"无法评估"并回报。

## 参考信息

- `references/scoring.md`：评分模型、投票裁决、等级取高、跨维度去重
- `references/dimension_guides.md`：四维检查项与判定细则（映射 rule_id）
- `references/cases.md`：四维典型判例
