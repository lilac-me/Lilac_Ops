---
name: triton-op-source-assembler
description: >-
  This skill should be used when the user provides a mapping from profiled kernel
  names to their operator source files and wants those sources assembled into the
  input contract expected by the Triton performance-tuning pipeline. It accepts a
  kernel-name-to-source map, validates that each named source exists and matches
  the candidate, packages each candidate into Mode A (torch Model .py) or Mode B
  (@triton.jit kernel .py + optional .pt) assembly, and marks candidates without a
  valid source as SKIPPED_NO_SOURCE instead of inventing one. Typical triggers
  include "assemble the sources for the profile bottleneck candidates" or
  "wire my kernel sources into the tuning workflow".
version: 1.0.0
domain: ops
function: triton
tags:
  - source
  - assembly
  - triton
  - contract
  - validate
author: mindspeed-agent
compatibility: MindSpeed >= 0.1.0, CANN >= 9.1.0, Triton >= 3.2.2
targets-repo:
  - triton-ascend
---

# 源装配（S3 前置依赖）

## 功能概述

把用户提供的「profiled kernel 名 → 源文件」映射，转成下游**性能调优模块**的输入合约，并做存在性/对应性校验。

**源装配是本工作流的全链 #1 前置依赖**（设计 §10）：用户源 = 用户真实训练在跑的 kernel。
无源 → 该 candidate **不进 S4**，标注 `SKIPPED_NO_SOURCE`，agent 明示「该算子无源、无法做优化」，
**不猜测、不硬造、不自动补齐**。

## 输入

| 输入 | 说明 |
| --- | --- |
| candidates（来自 S2） | 含 `name` / `tier` / `shapes`（profiled shape 作为 S4 主测试点） |
| source.map.yaml | 候选 kernel 名 → 源路径 的映射；或用户直接给出源文件 |
| 用户源 | Mode A：torch Model `.py`；Mode B：`@triton.jit` kernel `.py`（+ 可选 `.pt`） |

`source.map.yaml` 语义（可给整批或单个 candidate）：

```yaml
candidates:
  causal_conv1d_fwd_kernel:
    mode: B                 # A=torch Model 封装；B=@triton.jit kernel
    source: path/to/causal_conv1d_fwd.py
    weights: path/to/dump.pt   # 可选，与 profiled Input Shapes 对齐的真实 IO/权重 dump
```

## Mode A / Mode B 输入合约

| 模式 | 源形态 | 要求 |
| --- | --- | --- |
| **A（torch Model）** | `Model` 类 + `get_inputs` / `get_init_inputs` | 可独立实例化并 forward，用于 verifier 的参考实现与基准 |
| **B（GPU kernel）** | `@triton.jit` kernel `.py` | 可独立 launch；可选 `.pt` 含 `input_data` / `gpu_output`；可选 `gpu_perf.csv` |

> 两种模式都必须能用 **profiled shape(+dtype+dispatch)** 独立运行——这是 S4 的唯一测试点（设计 §2）。

## 处理流程

1. 读 `source.map.yaml`，逐一匹配 `candidates.json` 中的 `name`。
2. **存在性**：源文件路径存在；缺失/不可读 → 该 candidate 标 `SKIPPED_NO_SOURCE`。
3. **对应性**：校验源的函数/类与候选 kernel 名可对应（直觉核对 + 必要的 import 冒烟），
   核对失败 → `SKIPPED_NO_SOURCE`，不臆造。
4. **装配**：为每个合格 candidate 生成 `assembly/<name>/` 输入包，含：
   - 源文件（或指向其的引用）
   - `test_shapes.json`：profiled shape(+dtype+dispatch) 作为主测试点
   - 源说明（mode / 依赖 / 运行前提）
5. 产出装配清单，供 S4 逐 candidate 消费。

## 产出

```text
assembly/
  causal_conv1d_fwd_kernel/
    source.py（或引用）
    test_shapes.json        # profiled shape 主测试点
    assembly.json           # mode / 依赖 / SKIPPED 原因（如有）
  chunk_kda_bwd_kernel_wy_dqkg_fused/...
bottleneck-none-source.md    # 所有 SKIPPED_NO_SOURCE 的清单与原因
```

## 判定即纪律

- **无源 = 无法做优化**。绝不替用户补源、绝不臆造 golden。
- golden 生成：仅对**用户已给出且可隔离**的 kernel，才按需从 profile dump IO tensor 用 CPU fp64 参考重算；
  融合/无源 kernel 不硬造，走 SKIPPED。
- 装配后进入 S4 的，必须拥有 profiled shape 对应的真实运行源。
