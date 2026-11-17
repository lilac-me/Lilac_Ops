---
name: profile-bottleneck-analyzer
description: >-
  This skill should be used when the user provides an Ascend NPU profiler archive
  (profile.zip or extracted dir) and wants to identify bottleneck-level custom
  Triton operators for optimization. It parses kernel_details.csv and
  step_trace_time.csv, aggregates per-operator time share across steady-state
  steps, runs a two-tier Triton classification (builtin CANN prefixes excluded;
  strong vs weak/needs-human-review), and emits a ranked candidates.json for
  downstream per-operator performance tuning. Typical triggers include "find the
  Triton bottleneck ops in this profile", "which kernel should I optimize first",
  or "rank the hot custom kernels for tuning".
version: 1.0.0
domain: ops
function: triton
tags:
  - profiling
  - bottleneck
  - triton
  - analysis
  - candidates
author: mindspeed-agent
compatibility: MindSpeed >= 0.1.0, CANN >= 9.1.0
targets-repo:
  - triton-ascend
---

# Profile 瓶颈 Triton 算子识别（S2 前端）

## 功能概述

从 profiler 产物中识别出**瓶颈级（占稳态 step 总耗时 ≥ 阈值）的自定义 Triton 算子**，
产出排序后的 `candidates.json`，供下游 S3 源装配 + S4 单算子性能调优使用。

**定位：只做算子级归因，不做端到端验收（见设计文稿 §9 不变量）。**

## 输入

| 输入 | 说明 |
| --- | --- |
| profile 产物 | Ascend profiler 的 `profile.zip` **或**已解压目录；内部必须有 `kernel_details.csv`，`step_trace_time.csv` 可选 |
| config.json（可选） | 覆盖阈值 / 两档判定规则（`bottleneck_threshold` / `builtin_prefixes` / `strong_patterns` / `semi_builtin_prefixes`） |
| `--max-shapes` | 每个 candidate 保留的 Input Shapes 去重样本数（默认 5） |

## 用法

```bash
python skills/ops-skills/triton/profile-bottleneck-analyzer/scripts/parse_profile.py \
    <profile.zip|dir> -o <outdir> [--config config.json] [--max-shapes N]
```

> 不写死 zip 文件名/内层路径：脚本内部用 **glob** 递归定位 `kernel_details.csv` / `step_trace_time.csv`。

## 处理流程

1. **定位 CSV**：glob 递归找到 `kernel_details.csv` / `step_trace_time.csv`。
2. **稳态 step 集**：取 `step_trace_time.csv` 的 `Step` 列；缺失时回退为 `kernel_details.csv`
   去重后的 `Step Id` 集（设计 §6.2 回落）。
3. **op 类型级聚合**：每 step 内按 kernel `Name` 求和 `Duration(us)`；
   `op占比 = op_total / step_total`；**跨 step 先算占比再平均**（缺场均按 0 计入）。
4. **两档 Triton 判定**（`is_triton`，config 可配）：
   - **排除**：命中内建前缀（`aclnn` / `Hccl` / `hcom` / `aicpu` / `Atc` 等）。
   - **strong**：非内建前缀且命中 `_kernel$` / `kernel` / `triton` 等特征 → 可进 S4。
   - **weak**：非内建但无强特征、或命中 `semi_builtin` 前缀 → 「待人工核定」，**不改判、不自动进 S4**。
5. **阈值过滤**：`mean_op_ratio ≥ bottleneck_threshold`（默认 3%）才入候选，按占比降序。
6. **产出** `candidates.json`。

## 产出 `candidates.json`

```jsonc
{
  "profile_source": "...",
  "csvs": { "kernel_details": "...", "step_trace_time": "..." },
  "steady_steps": ["1", "2", ...],
  "summary": { "n_steady_steps": N, "n_strong": N, "n_weak": N },
  "note": "weak 归入待人工核定，不改判，不自动进 S4。",
  "candidates": [
    {
      "name": "causal_conv1d_fwd_kernel",   // kernel 名
      "tier": "strong",                      // strong | weak
      "mean_op_ratio": 0.038,                // 跨 step 平均 op 占比
      "coverage": 1.0,                       // 在多少稳态 step 中出现
      "total_duration_us": 12345.0,
      "n_samples_shapes": 1,
      "shapes": [ { "shape": "...", "dtype": "...", "core": "AI_CORE" } ],
      "per_step": [ { "step": "1", "op_total_us": ..., "step_total_us": ..., "ratio": ... } ]
    }
  ]
}
```

## 判定规则说明

- 两档判定的语义主观性是本工作流**唯一主观点**（设计 §6.4）：规则全在 config 可改。
- 同名可能既有 triton 也有 aclnn 实现路径 → 一律保守归 weak（不改判），交人工核定。
- 详细列口径见 `references/schema.md`。

## 下一步

- candidates 的 `tier=strong` → 交 `triton-op-source-assembler`（S3）装配源 → S4 性能调优。
- `tier=weak` → 只写进 batch_report「待人工核定」栏，给建议不改判。
