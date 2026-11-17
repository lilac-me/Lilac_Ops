# MFU 计算公式参考（Dense / MoE / VLM / Omni / DiT）

以下公式为标准 Transformer FLOPs 估算（乘加算 2 次 FLOP）。**优先使用 VeOmni FLOPs Counter**；Counter 不支持时才使用公式，并在报告中标注 `source: formula` 与假设。

## Dense

```text
每 token FLOPs ≈ 2 × N_active + 4 × L × d × s_frozen_terms
简化（忽略 logits 头可另计）：
  前向 ≈ 2 × N_active
  前向+反向 ≈ 6 × N_active        （反向约为前向 2 倍）
```

- `N_active`：激活参数量（embedding 与 tied head 是否计入需声明）
- 注意：attention 的序列相关项 `2 × L × d × s` 量级在长序列下不可忽略，Counter 结果优先

## MoE

```text
每 token FLOPs ≈ 6 × (N_attn + N_shared + k × N_expert_per_token)
```

- `N_attn`：attention 参数；`N_shared`：共享专家/稠密层参数
- `k`：top-k 路由专家数；`N_expert_per_token`：单个专家参数量
- **路由假设必须记录**：top-k 均匀负载（标准口径）；如做专家容量限制（capacity factor），FLOPs 按实际 dispatch 统计
- 报告同时记录 `total_params`（用于显存分析）与 `active_params`（用于 FLOPs），两者不可混用

## VLM

```text
每样本 FLOPs = 视觉塔 FLOPs（token 数 × 每 token FLOPs）+ 语言塔 FLOPs
```

- 视觉 token 数按分辨率/patch 大小计算，随图像数量变化
- 有效 token 统计需覆盖视觉 + 文本两部分

## Omni

```text
每样本 FLOPs = Σ 各模态流 FLOPs + 融合/生成模块 FLOPs
```

- 各模态分别按其结构（文本 Transformer / 音频 codec-Transformer / 视觉塔）套用对应公式
- 记录各模态 token 数与采样率假设

## DiT

```text
每样本 FLOPs ≈ 6 × N × latent_tokens × 采样步数（训练时为单步去噪）
```

- `latent_tokens`：VAE latent 序列长度（分辨率/时长决定）
- 扩散训练每 step 随机取一个时间步，FLOPs 与单步前反向成正比

## 集群理论峰值

```text
集群峰值 FLOPs/s = 单卡峰值（dtype）× 设备数量
```

- 单卡峰值取官方规格书数据（如 昇腾 910B bf16 规格值），来源写入 `cluster_peak_flops_per_s.source`
- dtype 混用时按实际计算精度选择峰值，禁止用 fp16 峰值乘 bf16 吞吐

## 统计口径检查清单

1. 有效 token 数是否排除 padding？padding 比例是否记录？
2. global batch 是否已含梯度累积与 DP 维度？
3. step time 是否为剔除 warmup 后的稳定均值？
4. 前向 or 前向+反向口径与日志报告口径是否一致（6×N vs 2×N）？
5. 设备数量与实际参与训练的卡数一致？
