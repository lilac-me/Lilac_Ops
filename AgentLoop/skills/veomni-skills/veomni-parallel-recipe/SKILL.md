---
name: veomni-parallel-recipe
description: >-
  This skill should be used when the user asks to design a model-centric
  distributed training recipe for VeOmni: analyze layer/param/sequence/expert
  counts and memory needs, design FSDP2, Ulysses SP, EP or combined schemes,
  check DeviceMesh and world size constraints, set recomputation, dtype, micro
  batch and global batch, estimate memory and communication overhead, compare
  candidate configs, or verify single-card vs distributed consistency.
version: 0.1.0
domain: common
function: performance-optimization
tags:
  - veomni
  - fsdp2
  - sequence-parallel
  - expert-parallel
  - devicemesh
author: mindspeed-agent
license: MIT
compatibility: "VeOmni >= 0.1"
---

# VeOmni 并行策略设计

## 目的

为 VeOmni 训练任务设计模型中心的分布式方案（FSDP2 / Ulysses SP / EP 及组合），输出可执行配置与比较分析。

## 设计流程

1. **输入分析**
   - 模型：层数、隐藏维度、参数量（MoE 记录专家数与激活参数）、序列长度
   - 硬件：NPU 卡数（world size）、单卡显存、互联拓扑
   - 训练：dtype（fp16/bf16/fp32）、目标 global batch、上下文长度
2. **显存估算**
   - 参数显存 = 参数量 × dtype 字节数（训练再乘 2 存主权重，fp16 训练时）
   - 优化器显存（AdamW：m + v，fp32 计）
   - 激活显存：层数 × 序列长度 × batch × hidden × 系数（考虑重计算则大幅下降）
   - 梯度与通信缓冲
3. **策略选择**
   - FSDP2：参数/梯度/优化器状态按 DP 维度分片，首选通用手段
   - Ulysses SP：序列维度切分 attention，长序列（如 ≥ 32K）收益明显
   - EP：MoE 专家切分，专家数需能被 EP 度整除
   - 组合约束：**world size 必须能被各并行度整除**；DeviceMesh 各维度乘积 = world size
4. **候选配置生成**：至少 2-3 组候选（如 FSDP2-only / FSDP2+SP / FSDP2+EP），分别估算显存与通信量。
5. **训练参数设置**：重计算策略、micro batch、global batch（= micro batch × DP × 梯度累积）、dtype。
6. **一致性验证**：单卡（无分片）与所选分布式配置各跑 toy 训练若干 step，比较 loss 曲线一致。
7. **输出**：`parallel_recipe.yaml`（可直接使用的配置，包含候选比较与选择依据）；面向人的说明由最终交付件统一汇总。

## 输出契约

`parallel_recipe.yaml`（Schema：`benchmarks/veomni/schemas/parallel_recipe.json`）：

```yaml
model: qwen3-moe          # 模型名
world_size: 8
device_mesh:              # 各维度乘积必须等于 world_size
  dp: 2
  sp: 2
  ep: 2
fsdp2:
  enabled: true
  sharding_strategy: "full"
  recomputation: "full"   # none | full | selective
dtype: "bf16"
batch:
  micro_batch: 1
  global_batch: 8
  gradient_accumulation: 4
consistency_check:
  executed: true
  single_card_loss: 0.0
  distributed_loss: 0.0
  consistent: true
```

## 门禁与约束

- 放行条件：DeviceMesh 与 world size 约束通过、候选配置已比较、单卡与分布式一致性验证通过（或明确记录未执行原因）。
- 显存与通信量为估算值时，必须在 `parallel_recipe.yaml` 的说明字段中标注“估算”，实测值以 performance-tuning 阶段为准。
- 不实际发起多机训练；多卡验证仅限用户已授权的环境。
