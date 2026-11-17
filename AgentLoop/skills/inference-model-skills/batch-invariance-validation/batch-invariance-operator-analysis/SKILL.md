---
name: batch-invariance-operator-analysis
description: This skill should be used when the user asks to collect profiling operators, compare batch invariance allowlists and blocklists, validate critical operators, or assess an unambiguous replacement.
version: 1.1.0
domain: inference-model
function: batch-invariance-validation
tags: [profiling, operator, batch-invariance]
author: mindspeed-agent
targets-repo:
  - vllm-ascend
---

# 阶段一：算子采集与白名单分析

调用算子分析脚本时传入 `--environment-report <运行目录>/environment/preflight.json`，先验证上游报告的产物哈希，再消费其规范芯片分类；不得重新猜测芯片。A2 分类保持不变，仅白名单分区复用 A3。

按 [Agent 调用资料](references/profiling-collection.md) 采集合规 profiling，并逐 rank 保存 `op_statistic.csv`。正式请求包含 logprob、调试或统计支路时返回 `INVALID_PROFILING_CONFIG` 并重采；缺少统计文件时返回 `BLOCKED_MISSING_OP_STATISTIC`。

对每个统计文件调用 [算子分析脚本](scripts/analyze_op_statistic.py)，依据 [算子白黑名单](references/batch_invariant_ops.md) 和 [分诊规则](references/profiling-triage.md) 先查白名单、再查强制黑名单。关键路径全部在白名单时通过；非关键路径必须保存 shape、dtype、调用栈和排除依据；关键路径命中黑名单时保留 `FAIL_BLACKLIST` 并检查实际代码替换路径；其他关键路径算子进入单算子验证。

按 [单算子验证规则](references/operator-batch-invariance.md) 还原真实 API、属性、shape、dtype、layout、推理阶段、TP 本地形状和批次轴。计算算子使用 [计算测试入口](scripts/operator_validation/computation_ops/test_compute_ops_batch_invariance.py)，通信算子使用 [AllReduce 测试入口](scripts/operator_validation/communication_ops/test_allreduce_batch_invariance.py)。每个算子必须完成 10,000 条有效 NPU 用例且全部按位一致；OOM、接口、环境异常或数量不足进入 `blocked_phase1`。

失败时按 [小算子替换资料](references/small_operator_allowlist.md) 检查实际代码。恢复状态的目标是消除已证实的算子差异；允许基于调用栈、实现位置和失败用例选择最小隔离修改，随后重采 profiling 并重跑 10,000 条验证。每轮保存假设、动作、脚本、差异、重采、重验和反思；相同问题最多五轮，且不得重复证据与动作均未变化的机械重试。耗尽后才返回 `BLOCKED_RECOVERY_EXHAUSTED`。输出统一 `stage_result.json`，产物必须带相对路径、类型和 SHA-256。
