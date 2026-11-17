---
name: veomni-mfu-calculation
description: >-
  This skill should be used when the user asks to calculate or verify the Model
  FLOPs Utilization (MFU) of a VeOmni training task for Dense, MoE, VLM, Omni or
  DiT models: derive model FLOPs from the VeOmni FLOPs Counter, compute achieved
  FLOPs/s from global batch, valid tokens, sequence length and step time,
  determine cluster peak from chip model, precision and device count, handle
  padding vs valid tokens and MoE routing assumptions, or cross-check MFU
  reported in training logs.
version: 0.1.0
domain: common
function: performance-optimization
tags:
  - veomni
  - mfu
  - flops
  - dense
  - moe
author: mindspeed-agent
license: MIT
compatibility: "VeOmni >= 0.1"
---

# VeOmni MFU 计算与校验

## 目的

计算和校验 VeOmni 训练任务的模型浮点运算利用率（MFU），口径可追溯、假设显式化。

## 计算公式

```text
MFU = 实际模型 FLOPs/s ÷ 集群理论峰值 FLOPs/s
```

## 计算流程

1. **输入收集**
   - 模型配置（层数、隐藏维度、vocab、MoE 专家参数）、训练配置（global batch、序列长度）、运行日志（step time）
   - 记录 VeOmni source revision
2. **模型理论 FLOPs**
   - **优先复用 VeOmni 已有 FLOPs Counter** 计算每 token FLOPs
   - 按模型类型套用结构公式（见 `references/flops-formulas.md`）：
     Dense / MoE（区分总参数与激活参数 + 路由假设）/ VLM（视觉塔 + 语言塔）/ Omni / DiT
3. **实际 FLOPs/s**
   - `有效 token 数 = global batch × 非_PADDING token 均值`（区分有效 token 与 Padding token，记录统计口径）
   - `实际 FLOPs/s = 每 token FLOPs × 有效 token 数 ÷ step time`
4. **集群理论峰值**
   - 根据芯片型号、计算精度（fp16/bf16）查得单卡峰值，乘以设备数量
   - 峰值数据来源必须记录（官方规格书/官方文档），不得凭记忆填写
5. **MFU 计算**：单卡与集群分别计算。
6. **日志校验**
   - 与训练日志中的 MFU 对比；不一致时检查公式、配置与统计口径（有效 token、设备数、dtype）找出差异原因
7. **输出**：`mfu_report.json`；面向人的 MFU 结论由最终交付件统一汇总。

## 停止条件（硬约束）

**缺少以下任一项时，不使用估算值代替，输出缺失项并停止计算：**

- 模型 FLOPs 公式（或 VeOmni FLOPs Counter 不支持该模型结构）
- 有效吞吐（step time 或有效 token 数缺失）
- 芯片理论峰值算力数据

## 输出契约

`mfu_report.json`（Schema：`benchmarks/veomni/schemas/mfu_report.json`）：

```json
{
  "model": "string",
  "model_type": "dense | moe | vlm | omni | dit",
  "formula": "MFU = achieved_flops_per_s / cluster_peak_flops_per_s",
  "inputs": {
    "global_batch": 0, "seq_len": 0, "valid_tokens_per_step": 0,
    "padding_ratio": 0.0, "step_time_ms": 0.0, "device_count": 0, "precision": "bf16"
  },
  "model_flops_per_token": {"value": 0.0, "source": "veomni_flops_counter | formula", "assumptions": ["string"]},
  "moe": {"total_params": "string", "active_params": "string", "routing_assumption": "string"},
  "achieved_flops_per_s": 0.0,
  "cluster_peak_flops_per_s": {"value": 0.0, "per_device": 0.0, "source": "string"},
  "mfu": {"single_device": 0.0, "cluster": 0.0},
  "log_cross_check": {"reported_mfu": 0.0, "match": true, "delta_reason": null},
  "missing_inputs": [],
  "conclusion": "string"
}
```

## 门禁与约束

- 放行条件：FLOPs、有效吞吐、芯片峰值、设备数量和计算口径**完整且可追溯**。
- `missing_inputs` 非空时 MFU 字段必须为 null，结论写明缺失项。
- MoE 必须记录总参数量、激活参数量和专家路由假设（如 top-k 平均负载）。
