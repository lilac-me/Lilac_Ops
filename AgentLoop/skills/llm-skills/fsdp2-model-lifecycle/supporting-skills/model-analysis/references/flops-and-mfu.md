# FLOPs 与 MFU 分析

## 目录

- 必须回答
- FLOPs 基本口径
- 常见模块 FLOPs
- 特殊结构推导规则
- 前向、反向和训练 FLOPs
- 整网 FLOPs 组合
- MFU 公式与输出模板
- FLOPs 校验
- 执行 FLOPs、有效 FLOPs 与推理 Head 口径

## 必须回答

1. 模型各模块理论 FLOPs；
2. 单层 FLOPs；
3. 整网前向 FLOPs；
4. 训练单步 FLOPs；
5. MoE Active FLOPs 与 Total Parameter 的区别；
6. 稀疏 Attention、线性 Attention、Indexer、MTP 等增量 FLOPs；
7. MFU 计算公式；
8. 峰值算力、卡数、单步时间、Token 数和精度口径；
9. 给出至少一个可代入配置的公式或计算表。

---

## FLOPs 基本口径

默认：

```text
一次乘法 + 一次加法 = 2 FLOPs，关注源码 `forward` 实现中所有与 Cube 计算相关的运算过程。
```

若用户或 Profiler 使用其他口径，必须同时给出转换关系。

必须区分：

- 参数量；
- 激活参数量；
- 模型计算速率；
- Token Throughput。

推荐术语：

```text
单卡平均模型计算速率（TFLOP/s）
```

不要使用容易与 `tokens/s` 混淆的“计算吞吐”。

## 常见模块 FLOPs

### Linear

```text
输入：[B, S, Din]
权重：[Dout, Din]
F_linear = 2 × B × S × Din × Dout
```

### Embedding

Embedding Lookup 通常不计主模型 FLOPs，单独标注为内存访问。

### RMSNorm

Norm 计算通常不计入主模型 FLOPs，无需呈现。

### SwiGLU MLP

```text
gate_proj: H → F
up_proj:   H → F
down_proj: F → H
F_swiglu_linear = 6 × B × S × H × F
```

激活与逐元素乘法单独列出。

### Dense Attention

```text
F_QK = 2 × B × N × S1 × S2 × Dh
F_AV = 2 × B × N × S1 × S2 × Dv
```

必须注明源码所使用的 Mask 场景，并根据实际实现判断有效计算量系数。只有 Kernel 或理论口径确实跳过无效区域时，`F_QK + F_AV` 才乘对应系数；例如 causal 下三角有效区域可近似取 0.5，不能仅因为存在 causal mask 就固定乘 0.5。

### Sparse Attention

```text
F_QK_sparse = 2 × B × N × S1 × K × Dh
F_AV_sparse = 2 × B × N × S1 × K × Dv
```

必须注明源码所使用的 Mask 场景，并根据实际实现判断有效计算量系数。只有 Kernel 或理论口径确实跳过无效区域时，`F_QK + F_AV` 才乘对应系数；例如 causal 下三角有效区域可近似取 0.5，不能仅因为存在 causal mask 就固定乘 0.5。

### MoE

```text
F_router = 2 × B × S × H × E
F_routed = TopK × F_single_expert
F_shared = Nshared × F_single_expert
```

整网 MFU 应使用 Active FLOPs，而不是全部 Expert FLOPs。

### Conv

```text
F_conv = 2 × B × Lout × Cout × (Cin/groups) × Kernel
F_depthwise = 2 × B × Lout × C × Kernel
```

### TopK、Softmax、激活与 Mask

必须单列为精确公式、近似公式或“未计入主 FLOPs”，不能在不同模型之间改变口径而不说明。

## 特殊结构推导规则

### MLA

不得直接套用普通 MHA 公式。必须从实际源码拆出 Q-A、Q-B、KV-A、KV-B、RoPE、Attention Score、Value 聚合和 Output Projection。

### Linear Attention/KDA

必须从实现拆出：

- Q/K/V Projection；
- Short Conv；
- Forget/Input/Output Gate；
- Chunk 内矩阵；
- Recurrent State 读写；
- Output Norm；
- Output Projection。

若 Kernel 代码不可见，必须给出上界、下界、估算公式和未知项。

### Indexer

至少计算：

- Index Query Projection；
- Index Key Projection；
- Compression；
- Query Score；

### MTP

计算 MTP Layer、MTP Head、Additional Loss、主模型共享部分，以及训练阶段是否始终启用。

### 重计算

重计算不影响理论 FLOPs 计算公式；理论 FLOPs 只考虑模型公式推导结果，不计入实际执行时因重计算产生的额外计算量。

## 前向、反向和训练 FLOPs

对于标准 MatMul/Linear，可使用：

```text
Backward ≈ 2 × Forward
Training ≈ 3 × Forward
```

但必须注明这是针对主要矩阵乘法的常用近似。

## 整网 FLOPs 组合

```text
F_model_forward
= Σ F_attention_type_i
+ Σ F_ffn_type_i
+ F_lm_head
+ F_auxiliary
```

训练单步：

```text
F_step
= F_model_train_per_token × global_tokens_per_step
+ F_auxiliary_loss
```

```text
global_tokens_per_step
= global_batch_size × sequence_length
```

如果使用 Packing，应按实际参与计算 Token 数统计。不要使用 `loss_mask` 中有效 Loss Token 数替代实际模型计算 Token 数，除非模型确实跳过无效 Token 的计算。

## MFU 公式

```text
MFU
= F_step
  / (step_time × device_count × peak_flops_per_device)
```

单卡平均模型计算速率：

```text
Model TFLOP/s per Device
= F_step / step_time / device_count / 1e12
```

```text
MFU
= Model TFLOP/s per Device / Peak TFLOP/s per Device
```

必须明确：

- 峰值对应 BF16、FP16、FP8、MXFP8 或其他精度；
- `step_time` 是端到端时间；
- Pipeline Bubble、Data Loading 是否包含；
- 是否为单卡平均；
- 理论 FLOPs 不计入重计算带来的额外执行量；辅助 Loss、MTP 是否计入必须明确；

## MFU 输出模板

| 项目 | 符号 | 数值/公式 | 口径 |
|---|---|---|---|
| 单 Token 训练 FLOPs | `F_token` | ... | Active FLOPs |
| 每步 Token 数 | `N_token` | ... | 实际计算 Token |
| 每步 FLOPs | `F_step` | ... | 理论 FLOPs，不计重计算额外执行量 |
| 单步耗时 | `t_step` | ... | 端到端 |
| 卡数 | `N_device` | ... | 训练卡 |
| 单卡峰值 | `P_peak` | ... | BF16/FP8 |
| 单卡模型计算速率 | — | ... TFLOP/s | 非 tokens/s |
| MFU | — | ...% | 端到端 |

## FLOPs 校验

至少使用两种方式交叉验证：

1. 公式逐模块求和；
2. Profiler 算子 FLOPs；
3. 小 Shape 实际算子统计；
4. 其他开源实现公式。

出现差异时必须定位到 Attention、Embedding/LM Head、MoE Router、Shared Expert、MTP、Causal 三角、Sparse TopK、激活/Norm，以及 Profiler 是否包含重计算等实际执行量。

## 执行 FLOPs、有效 FLOPs 与推理 Head 口径

对存在 Mask、稀疏选择、Block Padding 或仅最后 Token Head 的模型，必须至少区分：

1. **源码执行口径**：按实际算子在 Mask 前后、Padding 后和 Kernel Block 对齐后的计算量；
2. **算法有效口径**：只计算因果有效区、真实 TopK 或非 Padding 元素；
3. **训练等价口径**：全 Token LM Head、主要 MatMul Forward+Backward；
4. **参考推理口径**：Prefill/Decode 实际是否只计算最后 Token Head、是否复用 Cache。

强制规则：

- 源码在完整 QK/Indexer Score 之后才施加 Causal Mask 时，源码执行口径不得乘 0.5；
- Kernel 对 TopK、Head 或序列宽度按 Block 补齐时，逻辑宽度与执行宽度分别报告；
- 主 Forward 只取最后 Token logits 时，Prefill 总 FLOPs 使用 `S×Body + 1×LMHead`，训练使用 `S×(Body+LMHead)`；
- MTP、DSpark、Speculative Decoder、Auxiliary Loss 和 Reward Head 单列，不得默认混入主模型；
- 至少给短、中、目标最大长度三个数值示例；模型只支持单一长度时说明原因；
- MFU 示例没有真实硬件输入时必须标记“纯代入示例”，不得写成实测。

---
