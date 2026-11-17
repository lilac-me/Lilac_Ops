# Batch Invariant 算子白名单

命令示例以本 Skill 根目录（与 `SKILL.md` 同级）为工作目录。

本文件列出当前 skill 在分析 Ascend profiling op_statistic.csv 时视为 batch invariant 或可接受的算子名称。该白名单用于 triage 缩小嫌疑范围，不等价于这些算子在所有 shape、dtype、layout、TP 配置下都已完成独立证明。

`scripts/analyze_op_statistic.py` 会先根据芯片类型选择本文件中的分区，再解析对应分区表格 `算子名` 列中的反引号内容。

## 强制黑名单

以下五个基础算子名直接判定为不满足 batch invariance，优先级高于 A2/A3 和 A5 白名单；只有名称完全匹配时才命中黑名单：

| 基础算子名 | 黑名单规则 |
| ------ | ------ |
| `SoftmaxV2` | 精确匹配 `SoftmaxV2` |
| `MatMulV2` | 精确匹配 `MatMulV2` |
| `FusedInferAttentionScore` | 精确匹配 `FusedInferAttentionScore` |
| `AddRmsNorm` | 精确匹配 `AddRmsNorm` |
| `ReduceSum` | 精确匹配 `ReduceSum` |

名称已明确列在当前芯片白名单中的 BatchInvariant 算子不命中黑名单。不得据此把任意基础算子名直接拼接 `BatchInvariant` 作为替换方案；profiling 名称、模型或 vLLM-Ascend 源代码 API、注册入口和芯片实现可能并非一一对应。黑名单只检查白名单外算子；命中后先保留 `FAIL_BLACKLIST` 证据并跳过基础算子的单算子验证，再检查实际代码路径。恢复状态允许根据替换实现、参数语义、芯片可用性和修改位置选择最小修改；替换后重新 profiling，并对实际替换路径完成 10,000 条验证。相同问题五轮不能闭环才请求用户介入。

## A3 / A2

A2 暂时复用 A3 白名单。

| 类别 | 算子名 | 说明 |
| ------ | -------- | ------ |
| Batch invariant custom op | `FusedInferAttentionScoreBatchInvariant` | attention score batch invariant 变体 |
| Batch invariant custom op | `SoftmaxV2BatchInvariant` | softmax batch invariant 变体 |
| Batch invariant custom op | `MatMulV2BatchInvariant` | matmul batch invariant 变体 |
| Batch invariant custom op | `AddRmsNormBatchInvariant` | add rms norm batch invariant 变体 |
| Batch invariant custom op | `ReduceSumBatchInvariant` | reduce sum batch invariant 变体 |
| Batch invariant custom op | `MatMulV3BatchInvariant` | matmul batch invariant 变体 |
| Tensor shape / data movement | `ScatterPaKvCache` | paged attention KV cache scatter 路径 |
| Tensor shape / data movement | `ScatterElementsV2` | scatter elements 路径 |
| Tensor shape / data movement | `Transpose` | shape 或数据搬移相关 |
| Tensor shape / data movement | `Tile` | shape 或数据搬移相关 |
| Tensor shape / data movement | `Slice` | shape 或数据搬移相关 |
| Tensor shape / data movement | `ReshapeAndCacheNdKernel` | KV cache reshape/cache 路径 |
| Tensor shape / data movement | `BroadcastTo` | shape 广播 |
| Tensor shape / data movement | `GatherV3` | gather 路径 |
| Tensor shape / data movement | `GatherV2` | gather 路径 |
| Tensor shape / data movement | `Index` | index 路径 |
| Tensor shape / data movement | `LinearIndex` | 线性索引/散射辅助路径 |
| Tensor shape / data movement | `Fill` | fill 路径 |
| Compute / elementwise | `SwiGlu` | 激活计算 |
| Compute / elementwise | `Sub` | elementwise |
| Compute / elementwise | `RealDiv` | elementwise |
| Compute / elementwise | `Neg` | elementwise |
| Compute / elementwise | `Mul` | elementwise |
| Compute / elementwise | `MaskedFill` | mask 填充 |
| Compute / elementwise | `LogicalNot` | logical |
| Compute / elementwise | `LogicalAnd` | logical |
| Compute / elementwise | `Log` | elementwise |
| Compute / elementwise | `Less` | compare |
| Compute / elementwise | `GreaterEqual` | compare |
| Compute / elementwise | `Cast` | dtype 转换 |
| Compute / elementwise | `Add` | elementwise |
| Compute / elementwise | `Abs` | elementwise |
| Compute / elementwise | `Exp` | elementwise |
| Reduction / selection | `ReduceMax` | reduction |
| Reduction / selection | `ArgMaxV2` | selection |
| Norm / projection | `RmsNorm` | normalization |
| MoE | `MoeTokenUnpermute` | MoE token restore |
| MoE | `MoeInitRoutingCustom` | MoE routing |
| MoE | `MoeInitRoutingV3` | MoE routing 初始化 |
| MoE | `MoeGatingTopK` | MoE gating top-k |
| MoE | `GroupedMatmul` | expert grouped matmul |
| Random / sampling | `DSARandomUniform` | 采样随机数路径，需结合 seed 和采样阶段判断 |
| vLLM / custom kernel | `_triton_rope` | rotary embedding kernel |
| vLLM / custom kernel | `_compute_slot_mapping_kernel` | slot mapping kernel |
| Communication | `allreduceAicpuKernel` | 通信相关，单独出现不代表失败 |
| Communication | `allgatherAicpuKernel` | 通信相关，单独出现不代表失败 |
| Communication | `hcom_allReduce` | HCCL 通信相关 |
| Communication | `hcom_allGather` | HCCL 通信相关 |

## A5

A5 指昇腾 950 系列。A5 与 A2/A3 使用相同的强制黑名单和白名单外算子处理规则：命中强制黑名单时先输出并保留 `FAIL_BLACKLIST`，再按实际代码和 A5 已安装实现确认替换；不得用 A2/A3 的同名实现外推。无法一次确认时进入五轮受限恢复。白名单外且黑名单外时输出 `REVIEW_REQUIRED`，完成关键路径分析和必要的 10,000 条单算子验证后才能进入 Phase 2。

| 类别 | 算子名 | 说明 |
| ------ | -------- | ------ |
| Batch invariant custom op | `MatMulV3BatchInvariant` | matmul batch invariant 变体 |
| Batch invariant custom op | `FusedInferAttentionScoreBatchInvariant` | attention score batch invariant 变体 |
| Batch invariant custom op | `ReduceSumBatchInvariant` | reduce sum batch invariant 变体 |
| Compute / elementwise | `GroupedMatmul` | expert grouped matmul |
| Compute / elementwise | `RmsNorm` | normalization |
| Compute / elementwise | `Slice` | slice 路径 |
| Compute / elementwise | `Add` | elementwise |
| Compute / elementwise | `SwiGlu` | 激活计算 |
| Compute / elementwise | `RealDiv` | elementwise |
| Compute / elementwise | `Cast` | dtype 转换 |
| Compute / elementwise | `Fill` | fill 路径 |
| Compute / elementwise | `Sub` | elementwise |
| MoE | `MoeInitRoutingV3` | MoE routing 初始化 |
| MoE | `MoeGatingTopK` | MoE gating top-k |
| MoE | `MoeFinalizeRoutingV2` | MoE routing 完成 |
| Tensor shape / data movement | `ScatterPaKvCache` | paged attention KV cache scatter 路径 |
| Tensor shape / data movement | `Index` | index 路径 |
| Tensor shape / data movement | `Tile` | shape 或数据搬移相关 |
| Tensor shape / data movement | `Embedding` | embedding 查表路径 |
| Tensor shape / data movement | `GatherV2` | gather 路径 |
| Tensor shape / data movement | `BroadcastTo` | shape 广播 |
| Tensor shape / data movement | `TensorMove` | 张量搬移路径 |
| Tensor shape / data movement | `ScatterElementsV2` | scatter elements 路径 |
| Reduction / selection | `ArgMaxV2` | selection |
| vLLM / custom kernel | `_triton_rope` | rotary embedding kernel |
| vLLM / custom kernel | `_compute_slot_mapping_kernel` | slot mapping kernel |
