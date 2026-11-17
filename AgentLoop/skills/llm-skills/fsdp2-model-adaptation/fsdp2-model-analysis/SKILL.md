---
name: fsdp2-model-analysis
description: 单独生成中文 FSDP2 模型分析报告。分析真实 config 和 modeling，输出详细结构图、逐 module shape、参数量、理论显存、FLOPS、通信量、NPU 算子支持度，并在 TP=1、单卡 64GiB 条件下推导最优 FSDP2/EP/E-FSDP/CP/PP 切分和集群规模。只做分析与报告，不做模型代码适配。
targets-repo:
  - mindspeed-llm
---

# FSDP2 模型分析

## 功能概述

本 Skill 用于单独生成模型分析报告，不修改模型代码。报告必须是中文，必须包含图片形式的模型结构流程图，并逐 module 展示 shape、显存、计算量、通信量、昇腾 NPU 算子支持度和推荐并行切分。不要在报告中写模型适配清单、启动脚本、MFU 接入步骤或开发 TODO。

当用户同时要求“模型适配 + 分析报告”时，继续使用 `fsdp2-model-adaptation`；当用户只要求“模型分析、报告、切分建议、理论显存/计算/通信量”时，使用本 Skill。

## 工作流

### 步骤一：收集模型输入

优先读取用户提供的本地文件、仓库 modeling/configuration 文件、`config.json`、HuggingFace 链接或论文结构说明。若用户只给模型名或 URL，先获取真实源码和配置，再分析。

必须提取以下字段；缺失字段要在报告中标记为“按假设估算”，并写清假设值。报告正文中的变量命名必须优先使用可读的 `snake_case` 长名字，不要只用 `B/S/H/E/K` 这类单字母变量；如为公式紧凑性保留数学符号，必须同时给出长变量名映射。

| 类别 | 字段 |
| ------ | ------ |
| 基础结构 | `model_type`、`architectures`、`vocab_size`、`hidden_size`、`num_hidden_layers`、`max_position_embeddings` |
| Attention | `num_attention_heads`、`num_key_value_heads`、`head_dim`、RoPE 参数、attention 类型、sliding window、MLA/CSA/HCA/稀疏 attention 参数 |
| MLP/MoE | dense intermediate size、expert intermediate size、专家总数、每 token 激活专家数 `top_k`、shared expert、router 结构、hash/TopK 路由 |
| 训练假设 | micro batch、global batch、sequence length、dtype、是否使用 bf16/fp16、optimizer、是否启用 fp32 master weight、是否 checkpoint/recompute |
| 并行候选 | 可用 GPU 数范围、节点内卡数、网络层级、允许的 CP/EP/FSDP/E-FSDP/PP 范围；默认暂不考虑 TP，固定 `tensor_parallel_size = 1`，若用户明确要求 TP 再枚举 TP |
| 算子环境 | 目标昇腾产品、CANN/驱动/PyTorch/torch_npu 版本、训练或推理、目标 dtype、确定性要求；缺失时只做文档级分析并标记“环境待确认” |

如果模型配置同时存在 legacy 字段和新字段，必须优先还原真实 checkpoint 的 per-layer schedule。例如 DeepSeek-V4/DeepSeek-V4-Flash 必须读取 `compress_ratios`、`compress_rate_csa`、`compress_rate_hca`、`layer_types`、`mlp_layer_types`、`num_hash_layers`：

- `compress_ratio = 0` 表示 `sliding_attention`，不是 CSA，也不是 HCA。
- `compress_ratio = 4` 表示 `compressed_sparse_attention` / CSA。
- `compress_ratio = 128` 表示 `heavily_compressed_attention` / HCA。
- 若 `compress_ratios` 长度大于 `num_hidden_layers`，报告中说明原始长度，并按实际 decoder 构建逻辑截取前 `num_hidden_layers` 层计算。
- `num_hash_layers` 或 `mlp_layer_types == "hash_moe"` 必须转化为逐层 MLP/MoE schedule。不要把前几层 Hash-MoE 写成普通 TopK-MoE。
- 报告中必须列出前若干层的 `layer index -> attention 类型 -> MLP/MoE 类型` 对照表，尤其要把前两层、Hash-MoE 层和第一个 TopK-MoE 层写清楚。

### 步骤二：先做整体占用与最优切分推荐

报告必须先给整体理论占用、切分候选和推荐切分，再进入逐 module 细节。不要一开始就展开 module 级长表，否则读者很难先判断模型需要多少卡。

默认先不考虑 TP 切分：

- 固定 `tensor_parallel_size = 1`。
- 不枚举 TP 候选，不把参数、激活或通信量按 TP 切分。
- 不使用 `hidden_size / tensor_parallel_size`、`attention_head_count / tensor_parallel_size` 作为主公式。
- 如某模型因为单卡矩阵过大理论上需要 TP，先在“风险与后续分析”中说明，不把 TP 放入本轮最优切分。

整体分析必须包含：

| 小节 | 内容 |
| ------ | ------ |
| 整体参数占用 | 按模块大类汇总 embedding、attention、dense/shared MLP、router、routed experts、norm、LM head 的参数量和参数/梯度/优化器/master weight 显存 |
| 整体激活占用 | 按训练假设给出非专家激活、attention 中间量、MoE dispatch/combine 激活、checkpoint/recompute 后激活峰值 |
| 整体通信量 | 汇总 FSDP 参数 all-gather、梯度 reduce-scatter、EP dispatch/combine、E-FSDP expert shard 通信、CP 通信 |
| 候选切分 | 枚举 CP、EP、E-FSDP、FSDP、PP；`tensor_parallel_size` 固定为 1 |
| 推荐切分 | 先满足 64GiB 单卡预算，再选最小 GPU 数；同卡数下选理论通信量更低、整除关系更简单的候选 |
| 集群规模 | 输出理论 GPU 数；若未提供节点规格，默认按 8 卡/节点折算并标注为假设 |

整体占用需要先给一个总览表：

| 项目 | 未切分总量 | 推荐切分后每卡峰值 | 主要切分因子 | 说明 |
| ------ | ------------ | -------------------- | -------------- | ------ |
| 参数 + 梯度 + 优化器 + master weight | GiB | GiB/card | `fsdp_shard_size`、`expert_parallel_size`、`expert_fsdp_shard_size` | TP 固定为 1 |
| 激活 | GiB | GiB/card | `context_parallel_size`、`expert_parallel_size`、recompute | FSDP 不切非专家激活 |
| 通信 buffer | GiB 或 GB/step | GiB/card 或 GB/step | FSDP、EP、E-FSDP、CP | 理论估算 |
| 峰值 | GiB | GiB/card | `max(forward_prefetch, backward, optimizer_step)` | 必须小于 `device_memory_budget_gib` |

### 步骤三：拆解 module 结构

按模型真实 forward 路径拆解，不套用 Llama/Qwen 通用模板。特殊结构必须单独列出，例如 MLA、CSA、HCA、MTP、shared expert、grouped output projection、sliding window、hash MoE、稀疏 attention。存在 per-layer schedule 时，必须按实际层类型拆行，不能只写“Attention”或“Router”。

报告必须包含 module 级 shape 表。每一行都要写完整维度，不能只写“hidden states”或“tokens”。对会改变 token 形态的模块，必须额外写“shape 变化过程”，例如 `input_ids[int64, micro_batch_size, sequence_length] -> token_embedding[bf16, micro_batch_size, sequence_length, hidden_size]`。

| Module | 输入 shape | 参数 shape | 输出 shape | 说明 |
| -------- | ------------ | ------------ | ------------ | ------ |
| Embedding | `input_ids[int64, micro_batch_size, sequence_length]` | `embedding_weight[vocab_size, hidden_size]` | `hidden_states[activation_dtype, micro_batch_size, sequence_length, hidden_size]` | 查表默认不计 matmul FLOPS |
| RMSNorm | `hidden_states[activation_dtype, micro_batch_size, sequence_length, hidden_size]` | `norm_weight[hidden_size]` | `normalized_states[activation_dtype, micro_batch_size, sequence_length, hidden_size]` | 统计激活与参数显存 |
| Q/K/V 或压缩注意力 | 写真实 shape | 写真实权重 shape | 写真实输出 shape | MLA/CSA/HCA 分支要拆开 |
| Sliding Attention | `query_states/key_states/value_states` 的真实 head shape | 无 compressor 参数 | `local_kv_axis[activation_dtype, micro_batch_size, attention_head_count, <=sliding_window, attention_head_dim]` | `compress_ratio=0` 时必须单独写，不要归到 CSA/HCA |
| CSA/HCA Compressor | 写真实输入 shape | 写 compressor/indexer 权重 shape | 写 compressed KV、index scores 的完整 shape | CSA 和 HCA 必须分开，按真实压缩比例计算 token 数 |
| HashRouter | `flat_tokens[activation_dtype, token_count, hidden_size]`、`input_ids[int64, micro_batch_size, sequence_length]` | `router_weight[routed_expert_count, hidden_size]`、`tid2eid[vocab_size, experts_per_token]` | `hash_router_logits[token_count, routed_expert_count]`、`hash_indices[token_count, experts_per_token]`、`hash_weights[token_count, experts_per_token]` | `tid2eid[input_ids]` 固定选择 experts；learned score 只对选中 experts gather 权重 |
| TopKRouter | `flat_tokens[activation_dtype, token_count, hidden_size]` | `router_weight[routed_expert_count, hidden_size]` | `router_logits[token_count, routed_expert_count]`、`topk_indices[token_count, experts_per_token]`、`topk_weights[token_count, experts_per_token]` | `token_count = micro_batch_size * sequence_length / context_parallel_size` |
| Expert | `expert_input[activation_dtype, dispatched_token_count, hidden_size]` | `gate_up_weight[local_expert_count, 2 * expert_intermediate_size, hidden_size]`、`down_weight[local_expert_count, hidden_size, expert_intermediate_size]` | `expert_output[activation_dtype, dispatched_token_count, hidden_size]` | 只按激活 top-k 专家计算 FLOPS |
| LM Head | `final_hidden_states[activation_dtype, micro_batch_size, sequence_length, hidden_size]` | `lm_head_weight[vocab_size, hidden_size]` | `logits[activation_dtype, micro_batch_size, sequence_length, vocab_size]` | tie weight 时说明参数复用 |

### 步骤四：绘制结构流程图

必须生成图片文件，而不只是在报告中放 Mermaid 源码。优先输出 SVG；若环境支持 Mermaid/Graphviz，也可渲染 PNG。

流程图必须展示整体框架、block 内部计算过程，以及每个节点的完整输入/输出维度。节点文本必须采用下面这种格式：`模块名\n输入: tensor_name[dtype, dim...]\n输出: tensor_name[dtype, dim...]`。边上如果发生 reshape、flatten、split、concat、dispatch、gather、all-gather、reduce-scatter，也必须标注变化后的 shape。

```text
Input ids[int64, micro_batch_size, sequence_length]
  -> Embedding: output hidden_states[activation_dtype, micro_batch_size, sequence_length, hidden_size]
  -> Decoder Block x decoder_layer_count
  -> Final Norm: output final_hidden_states[activation_dtype, micro_batch_size, sequence_length, hidden_size]
  -> LM Head: output logits[activation_dtype, micro_batch_size, sequence_length, vocab_size]
Decoder Block:
  block_input[activation_dtype, micro_batch_size, sequence_length, hidden_size]
    -> Attention Norm: same shape
    -> Q/K/V 或压缩注意力: q/k/v 或 compressed_kv 的完整 head 维度
    -> RoPE: rotary 部分维度和 non-rotary 部分维度
    -> Attention Scores: [micro_batch_size, attention_head_count, query_length, key_value_length]
    -> Softmax -> AV -> O Projection: [micro_batch_size, sequence_length, hidden_size] -> [micro_batch_size, sequence_length, hidden_size]
    -> Residual
    -> FFN Norm
    -> HashRouter 或 TopKRouter:
       HashRouter: input_ids -> tid2eid[vocab_size, experts_per_token] -> hash_indices[token_count, experts_per_token]
       HashRouter: flat_tokens -> router_logits[token_count, routed_expert_count] -> gather hash_weights[token_count, experts_per_token]
       TopKRouter: flat_tokens -> router_logits[token_count, routed_expert_count] -> topk_indices/topk_weights[token_count, experts_per_token]
    -> Dispatch: [token_count, hidden_size] -> [dispatched_token_count, hidden_size]
    -> Expert gate/up/down: [dispatched_token_count, hidden_size] -> [dispatched_token_count, hidden_size]
    -> Combine: [token_count, hidden_size]
    -> Shared Expert 或 Dense MLP
    -> Residual: block_output[activation_dtype, micro_batch_size, sequence_length, hidden_size]
```

要求：

- 每类 block 都要画出来，不能只画一个“Attention + MLP”黑盒。
- 每个节点都要写清输入 tensor 名、dtype、完整维度、输出 tensor 名、dtype、完整维度。
- 每一次 token shape 变化都要写出来：`input_ids`、embedding 后、flatten 为 `token_count`、EP dispatch 后、combine 后、还原为 `[micro_batch_size, sequence_length, hidden_size]`。
- 对 MoE 模型必须画出 Router、TopK、Dispatch、Expert、Combine。
- 对 Hash-MoE 模型必须单独画出 `input_ids -> tid2eid -> hash_indices` 的固定专家选择路径，以及 `flat_tokens -> router_logits -> gather selected weights` 的 learned score 路径。不能只画普通 TopK。
- 对压缩/稀疏 attention 必须画出压缩 KV、索引/检索、局部窗口或全局压缩路径。
- 对 `compress_ratio=0` 的层必须画出 sliding attention/local window 路径，并明确这些层没有 CSA/HCA compressor。
- 图片文件与报告放在同一目录，报告使用相对路径引用图片。

### 步骤五：定义变量和常数

报告公式前必须先列变量表。所有公式里的变量、单位、常数来源都要解释，不能出现来源不明的硬编码数字。变量名必须使用描述性长名字；公式中可以在括号里给一个短符号别名，但正文和表格必须以长名字为主。

推荐变量：

| 长变量名 | 可选短符号 | 含义 |
| ---------- | ------------ | ------ |
| `micro_batch_size` | `B` | 每个 data parallel rank 的 micro batch size |
| `sequence_length` | `S` | 每条样本的 token 长度 |
| `token_count` | `T` | 当前公式中的 token 数，必须说明是全局、每 rank、还是 CP 切分后；常见为 `micro_batch_size * sequence_length / context_parallel_size` |
| `hidden_size` | `H` | Transformer hidden 维度 |
| `decoder_layer_count` | `L` | decoder block 层数 |
| `attention_head_count` | `A` | attention query head 总数 |
| `key_value_head_count` | `A_kv` | key/value head 总数，GQA/MLA 模型必须单独列出 |
| `attention_head_dim` | `D` | 每个 attention head 的维度 |
| `vocab_size` | `V` | 词表大小 |
| `routed_expert_count` | `E` | routed expert 总数 |
| `experts_per_token` | `K` | 每个 token 激活的专家数 |
| `expert_intermediate_size` | `I_expert` | 单个专家 FFN 中间维度 |
| `dense_intermediate_size` | `I_dense` | dense MLP 或 shared expert 中间维度 |
| `tensor_parallel_size` | `TP` | Tensor Parallel 规模；默认固定为 1，本轮切分推荐暂不考虑 TP |
| `context_parallel_size` | `CP` | Context Parallel 规模 |
| `expert_parallel_size` | `EP` | Expert Parallel 规模 |
| `fsdp_shard_size` | `FSDP` | 非专家参数 FSDP shard 规模 |
| `expert_fsdp_shard_size` | `E_FSDP` | 专家参数 FSDP shard 规模 |
| `pipeline_parallel_size` | `PP` | Pipeline Parallel 规模 |
| `parameter_bytes` | `b_param` | 参数 dtype 字节数，bf16/fp16 为 2 |
| `gradient_bytes` | `b_grad` | 梯度 dtype 字节数，通常为 2 |
| `optimizer_state_bytes` | `b_optim` | optimizer state 字节数，Adam fp32 m/v 为 8 |
| `master_weight_bytes` | `b_master` | fp32 master weight 字节数，启用时为 4，否则为 0 |
| `device_memory_capacity_gib` | `M_card` | 单卡物理显存，默认 64GiB |
| `device_memory_budget_gib` | `M_budget` | 单卡可用训练预算，默认 `0.9 * 64GiB = 57.6GiB`，为通信 buffer、碎片和 runtime 预留 10% |

训练线性层 FLOPS 常用系数 `6`，必须说明：forward matmul、backward input grad、backward weight grad 共三次矩阵乘，每次 multiply-add 计 `2` FLOPs，因此 `3*2=6`。

### 步骤六：逐 module 计算理论显存

必须分别计算未切分显存和切分后每 rank 显存。单位同时给出 GiB 和公式。

每个 module 至少拆成：

```text
parameter_memory = parameter_count * parameter_bytes / parameter_shard_factor
gradient_memory = parameter_count * gradient_bytes / gradient_shard_factor
optimizer_memory = parameter_count * optimizer_state_bytes / optimizer_shard_factor
master_weight_memory = parameter_count * master_weight_bytes / master_weight_shard_factor
activation_memory = activation_element_count * activation_bytes * activation_save_factor / activation_shard_factor
```

切分规则：

- 非专家参数按 `fsdp_shard_size` 切分；本轮默认 `tensor_parallel_size = 1`，暂不按 TP 切分。
- 专家参数按 `EP * E_FSDP` 切分；如果每 rank 只持有 `E/EP` 个专家，参数量先除 `EP`，再按 `E_FSDP` shard。
- Router、Norm、Embedding、LM Head、Attention、Dense MLP、Shared Expert 分别按真实并行策略计算。
- 专家激活按 EP token 分布估算：均匀估算为 `token_count * experts_per_token / expert_parallel_size`，并额外给出负载不均衡系数 `expert_load_imbalance_factor` 后的保守值 `expert_load_imbalance_factor * token_count * experts_per_token / expert_parallel_size`。
- 非专家激活不按 FSDP 切分；TP/CP 会改变 hidden 或 sequence 维度时才切分。
- CP 切分 sequence 维：每 rank token 近似 `T/CP`，但 attention KV cache 或跨 CP 通信 buffer 要单独列出。
- Recompute/checkpoint 会减少保存激活但增加计算量，必须明确是否假设启用。

### 步骤七：逐 module 计算训练 FLOPS

报告必须按 module 给公式和代入过程，不只给总数。

通用公式：

| Module | 训练 FLOPS |
| -------- | ------------ |
| Linear `input_dim x output_dim` | `6 * token_count * input_dim * output_dim` |
| QK attention | `4 * micro_batch_size * attention_head_count * query_length * key_value_length * attention_head_dim`，训练时按 forward/backward 展开；若统一估算可写为 `12 * attention_pair_count * attention_head_count * attention_head_dim` |
| AV attention | 同 QK |
| Router | `6 * token_count * hidden_size * routed_expert_count` |
| Active Experts | `6 * token_count * experts_per_token * (hidden_size * 2 * expert_intermediate_size + expert_intermediate_size * hidden_size)` |
| Shared Expert | `6 * token_count * (hidden_size * 2 * dense_intermediate_size + dense_intermediate_size * hidden_size)` |
| LM Head | `6 * token_count * hidden_size * vocab_size`，tie weight 只影响参数显存，不自动取消计算 |

如果模型使用 sliding/CSA/HCA/MLA/稀疏 attention，必须按真实 pair 数或压缩 token 数计算，不能用 `S^2` 一把梭。

存在 per-layer schedule 时，FLOPS 必须使用真实层数，而不是写死默认层型。例如 DeepSeek-V4 Flash 的前两层 `compress_ratio=0` 只计算 sliding window attention pair 和普通 attention projection，不计算 CSA/HCA compressor；HashRouter 的 `tid2eid[input_ids]` 是查表，不按 matmul FLOPS 计，但其 learned `router_weight` score matmul 仍按 `6 * token_count * hidden_size * routed_expert_count` 计算。

### 步骤八：逐 module 计算通信量

通信量必须区分理论通信和 profiler 实测；没有实测时写“理论估算”。

至少包含：

- FSDP 参数 all-gather：每层前向需要聚合被 shard 参数，量级约 `module_parameter_count * parameter_bytes * (fsdp_shard_size - 1) / fsdp_shard_size`。
- FSDP 梯度 reduce-scatter：反向梯度切分，量级约 `module_parameter_count * gradient_bytes * (fsdp_shard_size - 1) / fsdp_shard_size`。
- TP 通信：默认不计算，标注 `tensor_parallel_size = 1`。只有用户明确要求 TP 分析时，才增加列并行/行并行 linear 的 all-reduce、reduce-scatter 或 all-gather。
- CP 通信：attention 需要跨 sequence shard 的 KV 或 ring attention 通信，按 KV shape 与层数估算。
- EP dispatch/combine：发送 token hidden 到 expert rank，理论量约 `2 * token_count * experts_per_token * hidden_size * activation_bytes * (expert_parallel_size - 1) / expert_parallel_size`，再乘负载不均衡系数。
- E-FSDP expert shard 通信：专家参数 all-gather 与梯度 reduce-scatter，按 expert 参数量除 EP 后估算。
- Pipeline parallel 如果使用，要列 micro-batch 激活发送量和 bubble 风险。

### 步骤九：分析 NPU 算子支持度

用户只需提供 modeling 源码、权重路径和可运行 NPU 容器；默认从权重路径读取 config/tokenizer，并在容器中探测产品型号、`torch`、`torch_npu`、MindSpeed、自定义算子、驱动和 CANN 版本。不要要求用户手工列算子清单。

#### 9.1 文档基线

以用户指定 CANN 版本为准。未指定时默认参考 [CANN 社区版 9.1.0-beta.3 算子库 API 入口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/aolapi/operatorlist_00001.html)，并明确它只是文档基线，不代表服务器已安装该版本。该入口按 Math、NN、CV、Transformer aclnn 接口分类，并另列 Ascend IR 与框架原生 IR 清单；LLM 至少检查 Math、NN、Transformer，涉及图模式或框架下沉时再检查 IR 清单。

官方入口明确：未声明支持的产品、dtype、数据格式、维度等场景不保证调用效果；昇腾虚拟化实例当前不支持融合算子。因此同名 API 存在不等于当前模型签名可直接使用。

#### 9.2 提取真实算子签名

从真实 `forward`、helper/autograd function、loss 和 FSDP2/EP 通信路径提取算子。优先结合静态调用链与最小输入的 `torch.fx`/`torch.export`/profiler 图；无法 trace 时标记静态分析及漏算风险。以“语义 + 阶段 + dtype + layout + 关键 shape/属性”生成唯一签名，不能只列 module 类名。

至少覆盖 Embedding/Gather、RMSNorm/LayerNorm、MatMul/BatchMatMul、Add/Mul、SwiGLU/SiLU/GELU、RoPE、mask、Softmax、Attention 正反向、TopK/Sort、Gather/Scatter/Index、MoE route/permute/grouped matmul/combine、cross entropy 和实际 collective。CSA/HCA/MLA/Hash-MoE 必须同时拆出底层 primitive 与期望融合算子。

#### 9.3 官方证据链

按以下顺序查证，并保存直接证据 URL、版本与查询日期：

1. 目标版本昇腾社区单算子详情页：核对训练/推理、正反向、产品、dtype、format/layout、维度、shape、属性、确定性及配套接口。
2. 匹配版本的 CANN 官方 GitCode 仓标签/分支，如 `cann/ops-transformer`、`cann/ops-nn`、`cann/ops-math` 的 `op_api_list.md` 和单算子文档。禁止用 `master` 或高版本证明旧版本支持。
3. 匹配版本的 torch_npu 官方文档/源码：确认 aclnn API 有 PyTorch binding、autograd 注册和公开入口。
4. 用户目标环境中的符号检查、最小执行与 profiler。页面不可达时标记“待确认”，不能写“不支持”。

推理 API 不能证明训练支持；有独立 Grad API 时必须分别列正向和反向。通信融合还要核对 HCCL 通信域、rank 组织、Dispatch/Combine 配套接口和版本一致性。

#### 9.4 Python/torch_npu 接口支持分析

CANN/aclnn 算子支持和 Python 层可调用性必须分开分析。CANN 算子详情页中的 torch_npu/PyTorch 调用示例是 Python 候选入口的重要证据，但它只对应文档版本，不能证明用户安装的 `torch_npu` 版本已经暴露该入口，也不能证明 autograd、目标 shape 或目标 kernel 可用。

对每个算子签名必须完成以下检查：

1. 从目标版本 CANN 详情页提取“PyTorch框架调用”“torch_npu调用”“调用示例”等内容，记录原样 Python 符号、模块路径和示例 URL。入口可能是标准 `torch`/`torch.nn.functional`、`torch_npu` 公开函数或 `torch.ops` 注册算子；只能记录文档或目标环境真实存在的形式，禁止凭 API 名猜 Python 函数名。
2. 在目标环境记录 `torch.__version__`、`torch_npu.__version__`，检查该符号是否可导入、是否 callable，并通过官方 torch_npu 文档/匹配版本源码确认函数签名。`hasattr`、`dir()` 或 import 成功只记为 `IMPORTABLE`。
3. 建立 aclnn 参数到 Python 参数的逐项映射，包括输入、输出、可选参数、dtype、layout、shape、属性默认值、in-place/out 语义和返回值数量。文档示例与 Python 签名不一致时，以目标安装版本为准并标记版本差异。
4. 分析训练可用性：检查 Python 入口是否注册 autograd，是否有独立 backward/Grad API，以及 `requires_grad=True` 时所需输入是否产生正确 shape/dtype 的梯度。只有 forward 成功时只能写“仅前向认证”，不能证明训练支持。
5. 使用报告中的真实 dtype、layout 和代表性 shape 执行 Python 最小样例；依次验证输出 shape、dtype、有限值和数值对照。训练算子必须再执行 loss.backward 并检查必需梯度。
6. 条件允许时用 profiler 确认 Python 调用实际下沉到预期 aclnn/kernel。标准 PyTorch API 可能经 NPU dispatch 下沉，即使没有同名 `torch_npu` 函数也可以实测支持，但必须给 profiler 或 dispatch 证据，不能仅按函数名称推断。
7. Python binding 缺失时，区分“CANN 有能力但 Python 未接入”和“CANN 本身不支持”：前者给出 C++/aclnn extension、`torch.library` 注册或已有框架封装建议；后者再考虑基础算子分解或自定义算子。

每个签名分别使用 9.5 的 S0-S5 与证据等级评价两层，并给出端到端结论：

| 判定层 | 必填内容 |
| -------- | ---------- |
| CANN/aclnn 层 | aclnn API、目标 CANN 版本、产品、dtype/layout/shape、正反向、状态/证据、官方详情页 |
| Python/torch_npu 层 | 完整 Python 符号、函数签名或调用片段、torch/torch_npu 版本、参数映射、autograd、状态/证据、文档或源码 URL |
| 端到端模型层 | 实际 modeling 调用点、fallback、最终状态、是否计入直接支持率/可执行覆盖率 |

聚合规则：端到端状态不能高于两层中证据更弱的一层。典型情况包括：

- `CANN S2 / Python S0`：底层文档支持，但 Python 入口未确认，端到端为 `S0-待确认`。
- `CANN S2 / Python S3`：底层规格匹配，但 Python binding 有版本或 autograd 条件，端到端为 `S3-条件支持`。
- `CANN S1 / Python S1`：Python 入口在目标环境完成相应 forward，训练场景还完成 backward，端到端才可判 `S1-实测支持`。
- `CANN S5 / Python 可导入`：即使 Python 符号存在，只要目标算子语义或约束不满足，端到端仍为 `S5-需适配或自定义算子`。

#### 9.5 支持状态与证据等级

状态码回答“该算子签名是否受支持、以什么方式支持”，证据等级回答“本轮验证实际走到了哪一步”。两者是两个独立维度，必须组合记录，不能把 `EXECUTED` 本身当作支持结论。

| 状态 | 中文含义 | 判定条件 | 报告结论 |
| ------ | ---------- | ---------- | ---------- |
| `S0-待确认` | 当前证据不足 | 环境信息不足、页面不可达、binding 未核实或无法可靠提取签名 | 不得计入直接支持率或可执行覆盖率，列入待确认项 |
| `S1-实测支持` | 目标环境已验证可用 | 推理场景完成目标 forward；训练场景完成目标 forward/backward。产品、版本、dtype、layout、shape 与报告一致 | 可作为当前验证范围内的直接支持证据 |
| `S2-文档直接支持` | 官方规格明确支持，但本机未实测 | 目标版本官方文档满足全部约束，且已确认 torch_npu/aclnn 调用入口 | 可计入文档直接支持率，必须注明“尚未实测” |
| `S3-条件支持` | 满足附加条件后可能可用 | API 存在，但产品、版本、dtype、layout、shape、反向、确定性或配套接口仍有限制 | 不能笼统写“支持”，必须逐项列出前置条件和未满足条件 |
| `S4-可分解支持` | 无对应融合算子，但基础算子可实现等价语义 | 已确认可由基础算子链等价实现 | 必须列分解链、性能代价、数值或语义风险；只有分解链实测通过后才计入可执行覆盖率 |
| `S5-需适配或自定义算子` | 当前没有满足模型语义的直接实现 | 官方清单无匹配实现，或产品/版本/shape/语义约束明确不满足 | 属于阻塞或性能缺口；给出 eager fallback、自定义算子或模型改写建议，不得计入支持率 |

证据等级必须使用下表，不允许只写模糊的“已验证”：

| 证据等级 | 含义 | 能证明什么 | 不能证明什么 |
| ---------- | ------ | ------------ | -------------- |
| `STATIC` | 仅完成源码、文档或调用链映射 | 已定位候选 API、约束和调用位置 | 不能证明可导入、可执行或命中目标 kernel |
| `IMPORTABLE` | Python/C++ 入口可导入或符号存在 | 当前环境暴露了调用入口 | 不能证明参数约束满足、数值正确、反向可用或 kernel 真正执行 |
| `EXECUTED` | 已按报告 shape/dtype 实际调用并记录结果 | 能证明这次调用成功，或能稳定复现明确失败 | 不自动代表支持；是否支持仍由 S0-S5 决定，也不能证明命中了预期融合 kernel |
| `PROFILED` | profiler 已确认实际 kernel、通信或融合路径 | 能证明执行路径和目标 kernel 命中 | 仍只覆盖本次产品、版本、shape、dtype 与阶段 |

报告中的组合状态必须按以下方式解释：

- `S1-实测支持 / EXECUTED`：目标验证范围执行成功；训练结论必须包含 backward，只有 forward 时明确写“仅前向认证”。
- `S1-实测支持 / PROFILED`：不仅执行成功，而且 profiler 已确认目标 kernel；这是最强的本轮运行证据。
- `S2-文档直接支持 / STATIC`：官方规格匹配，但当前服务器没有运行证据。
- `S3-条件支持 / IMPORTABLE`：入口存在，但仍有版本、shape、dtype、反向或配套条件未验证。
- `S4-可分解支持 / EXECUTED`：融合 API 不匹配，但等价基础算子链已经运行通过。
- `S5-需适配或自定义算子 / EXECUTED`：已经实际尝试并稳定复现不兼容或阻塞；`EXECUTED` 表示失败证据已复现，不表示支持。
- `S0-待确认 / STATIC`：只有静态线索，信息不足，不能给出支持结论。

`S1` 至少要求 `EXECUTED`；`hasattr` 或 import 成功不能判为 `S1`。找不到公开 torch_npu binding 但 aclnn API 存在时，最多为 `S3`，并说明需要 C++/aclnn 接入。

每份模型分析报告必须在“算子支持度摘要”和“逐 module/逐算子明细”之前完整输出上述两张说明表，并在正文第一次出现状态时使用完整名称，例如 `S4-可分解支持 / EXECUTED`，不能只写读者无法理解的 `S4/EXECUTED`。报告还必须明确：状态码是支持结论，证据等级是验证深度。CANN 层和 Python 层都使用这套定义，但必须分列，不能用一个状态同时代替两层结论。

用户提供 NPU 环境时，对训练关键 `S2/S3/S4` 项使用报告中的真实 dtype、layout 和代表 shape 运行最小样例。记录命令、产品、版本、shape、结果、异常摘要和确定性。大 shape 无法直接测试时，可做约束等价的小 shape，但原 shape 必须保持“未实测”。

#### 9.6 报告与覆盖率

先输出 9.5 定义的状态说明表和证据等级说明表，再给整体摘要与逐 module 明细：

| 指标 | 计算方式 |
| ------ | ---------- |
| CANN 唯一签名直接支持率 | `CANN 层 (S1 + S2 的唯一签名数) / 全部唯一签名数` |
| Python 接口直接支持率 | `Python 层 (S1 + S2 的唯一签名数) / 全部唯一签名数` |
| 端到端唯一签名直接支持率 | `CANN 与 Python 两层均为 S1/S2 的唯一签名数 / 全部唯一签名数` |
| 调用点直接支持率 | `(S1 + S2 的调用点数) / 全部调用点数` |
| 可执行覆盖率 | `(S1 + S2 + 已验证可分解的 S4) / 全部唯一签名数`，不得包含 S0 |
| 融合算子覆盖率 | `约束匹配的直接支持融合候选数 / 模型期望融合候选数` |
| FLOPS 加权覆盖率 | 可选；仅在 FLOPS 可可靠归属时计算并说明权重口径 |

| Module/源码位置 | 模型算子签名 | 阶段 | 完整输入/输出 shape | CANN API 与状态/证据 | Python 符号与状态/证据 | torch/torch_npu/CANN 版本 | Python参数映射/autograd | dtype/layout/关键约束 | 端到端状态 | fallback/建议 | 证据 URL |
| ------------------- | -------------- | ------ | --------------------- | ----------------------- | -------------------------- | -------------------------- | ------------------------- | ------------------------- | ------------ | --------------- | ---------- |

同一 module 的不同签名必须拆行。若 CANN 文档给出 Python 示例，证据栏必须链接到包含该示例的详情页，Python 列必须写出具体调用形式，不能只写“torch_npu 支持”。报告还要汇总正确性阻塞项、缺失融合导致的性能缺口和全部待确认项；建议按 `P0 正确性阻塞 -> P1 训练关键融合 -> P2 常规性能` 排序，但不要写成模型代码适配清单。

### 步骤十：枚举并选择最优切分

不要固定写 `FSDP=16, EP=8`。必须根据理论显存自动枚举候选并给出最优切分和理论集群规模。本轮默认暂不考虑 TP，固定 `tensor_parallel_size = 1`。

默认硬件假设：

- 单卡显存 `M_card = 64GiB`。
- 可用训练预算 `M_budget = 57.6GiB`，为 runtime、通信 buffer、内存碎片预留 10%。如果用户给了显存利用率或保留比例，使用用户值。
- 节点内卡数未知时，以总卡数为主要输出；若用户给节点规格，再换算节点数。

候选枚举规则：

1. 枚举 `context_parallel_size, expert_parallel_size, fsdp_shard_size, expert_fsdp_shard_size, pipeline_parallel_size` 的合法组合；固定 `tensor_parallel_size = 1`。
2. 约束 `routed_expert_count % expert_parallel_size == 0`、`sequence_length % context_parallel_size == 0` 或说明 padding、`fsdp_shard_size >= 1`、`expert_fsdp_shard_size >= 1`。
3. 总卡数 `gpu_count = context_parallel_size * pipeline_parallel_size * data_parallel_group_size`；若 `fsdp_shard_size` 是 data-parallel shard size，则 `data_parallel_group_size >= fsdp_shard_size`；专家总切分要满足 `data_parallel_group_size` 中的 EP/E-FSDP 组织方式，并在报告中写明 group 关系。
4. 对每个候选计算每 rank 峰值显存：`M_rank_peak = max(M_forward_prefetch, M_backward, M_optimizer_step)`，至少包含参数、梯度、优化器、master weight、激活、通信 buffer。
5. 可行条件：`M_rank_peak <= M_budget`，且所有 module shape 与切分约束合法。

优化目标按顺序执行：

1. 首先选择满足 64GiB 单卡预算的最小 `gpu_count`。
2. 若多个候选卡数相同，选择理论通信量更低的组合。
3. 若通信量接近，优先选择更简单的组合：较小 TP、较小 CP、EP 能整除专家数、FSDP 组内规模不过大。
4. 若用户给出吞吐、MFU 或网络拓扑目标，再把目标写入排序规则。

报告必须输出“候选切分表”：

| 候选 | TP | CP | EP | E-FSDP | FSDP | PP | 总卡数 | 每卡峰值显存 | 理论通信量/step | 是否可行 | 原因 |
| ------ | ---- | ---- | ---- | -------- | ------ | ---- | -------- | -------------- | ------------------ | ---------- | ------ |
| 示例 | 1 | context_parallel_size | expert_parallel_size | expert_fsdp_shard_size | fsdp_shard_size | pipeline_parallel_size | gpu_count | GiB/card | GB/step | 是/否 | TP 固定为 1 |

并输出“推荐切分”：

```text
推荐：TP=1, CP=y, EP=z, E-FSDP=a, FSDP=b, PP=c
理论需要 GPU：N 张
按每节点 8 卡折算：ceil(N/8) 个节点（若用户未提供节点规格，标为假设）
推荐理由：满足 57.6GiB 单卡预算；在最小卡数候选中通信量最低；专家数/头数/序列长度均可整除。
风险：理论估算未包含 allocator 碎片、框架临时 buffer、数据加载峰值和实际负载不均衡。
```

### 步骤十一：报告结构

报告 Markdown 按以下顺序输出：

1. 摘要：模型、关键规模、整体峰值显存、推荐切分、理论 GPU 数。
2. 变量与硬件假设：包括 64GiB 单卡预算、57.6GiB 可用预算、dtype、batch、seq、optimizer，并声明 `tensor_parallel_size = 1`。
3. 整体理论占用总览：参数/梯度/优化器/master weight、激活、通信 buffer、峰值。
4. 切分候选枚举与最优切分：先给候选表，再给推荐切分和理论集群规模。
5. 模型结构流程图：引用同目录图片，图中必须包含每个 module 的完整输入/输出维度。
6. 模型结构分析：逐 block 描述真实 forward 和 token shape 变化。
7. Module 级 shape 表。
8. Module 级理论显存：未切分与推荐切分后每 rank。
9. Module 级训练 FLOPS：公式、变量解释、代入过程。
10. Module 级通信量：FSDP/CP/EP/E-FSDP；TP 固定为 1，默认不计算 TP 通信。
11. NPU 算子支持度：先说明 S0-S5 状态和 STATIC/IMPORTABLE/EXECUTED/PROFILED 证据等级，再分别给 CANN/aclnn 层与 Python/torch_npu 层的接口、版本、参数映射、autograd、状态和证据，最后给端到端覆盖率、fallback、阻塞与性能缺口。
12. 风险与边界：TP 暂不考虑、文档支持不等于目标环境实测、理论估算未包含 profiler 实测、allocator 碎片和实际负载不均衡。

报告中不要出现：

- 模型适配清单。
- 启动命令或训练脚本说明。
- MFU 接入步骤。
- “后续 TODO” 式开发任务。

### 步骤十二：验证

生成报告后做最小验证：

- 检查 Markdown 图片链接存在且可打开。
- 检查所有公式变量都在变量表中定义。
- 检查报告先给整体占用、候选切分和推荐切分，再进入逐 module 分析。
- 检查 `tensor_parallel_size` 固定为 1，且没有把 TP 作为候选参与最优切分。
- 检查候选切分表至少包含一个不可行候选和一个推荐候选；如果没有可行候选，必须明确“64GiB 单卡预算下最小枚举范围内不可行”，并扩大枚举或说明需要更大集群。
- 检查推荐切分不是硬编码值，而是由候选表推导出来。
- 检查报告没有写入模型适配清单、启动脚本、MFU 接入步骤或开发 TODO。
- 检查每个 forward 关键 module 都有算子签名，训练路径覆盖 backward/loss，FSDP2/EP 路径覆盖实际 collective。
- 检查每条 `S1/S2/S3` 都有目标版本官方证据；`S1` 另有执行记录；没有把页面不可达写成不支持。
- 检查产品、CANN/torch_npu 版本、dtype、layout 或 shape 未知时没有误判为 `S1/S2`。
- 检查覆盖率分母与状态集合明确，`S0` 未计入可执行覆盖率，推理 API 未被当作训练反向证据。
- 检查没有把 `STATIC/IMPORTABLE` 误写成 `EXECUTED/PROFILED`。
- 检查报告在算子摘要和明细前完整解释了 S0-S5 与四种证据等级，并明确状态码与证据等级是两个独立维度。
- 检查没有出现未解释的 `S1/S4/S5` 或 `S5/EXECUTED` 缩写；`S5/EXECUTED` 必须说明是已复现失败，而不是已支持。
- 检查每个关键签名分别给出 CANN/aclnn 与 Python/torch_npu 两层状态；没有因为底层 API 存在就直接判定 Python 可用。
- 检查 CANN 详情页给出的 Python 调用方式已落到具体符号、版本、参数映射和 autograd 结论；import/`hasattr` 没有被当成执行证据。
- 检查端到端直接支持率只统计 CANN 与 Python 两层都满足直接支持条件的签名。

## 输出要求

最终回复用中文，简要说明：

- 新增或更新的报告路径。
- 结构图图片路径。
- 推荐切分和理论 GPU 数。
- 昇腾 NPU 算子支持摘要、阻塞项数量和采用的 CANN 文档/实测环境版本。
- 验证结果。
- 哪些内容是理论估算，哪些没有经过 profiler 或真实训练验证。
