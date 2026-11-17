---
name: vllm-memory-calculator
description: "This skill should be used when the user asks to analyze a new LLM or multimodal model for Ascend NPU deployment — including architecture detection, parameter counting, KV cache estimation, single-layer operator DFG analysis, vLLM Ascend incremental operator identification (P0/P1/P2), weight classification (Expert/Replicated/TP-Sharded), quant×parallel memory matrix, and parallel strategy search (TP/DP/EP/PP) with rank topology. Also generates a self-contained HTML analysis report. Covers MHA, GQA, MLA, GDN, CSA+HCA, MoE (standard + LatentMoE), MSA/DSA sparse attention, and multimodal vision tower. Trigger on phrases like \"0day适配\", \"模型分析报告\", \"Ascend NPU适配\", \"算子分析\", \"KVCache估算\", \"模型结构分析\", \"并行策略\", \"需要几张卡\", \"显存放不下\", \"并行下显存估算\", \"版本差异\", or when pointing to a config.json directory or HuggingFace URL."
version: 4.0.0
domain: inference-model
function: model-analysis
tags:
  - vllm
  - memory
  - parallelism
  - deployment
  - kvcache
  - model-analysis
  - architecture-detection
  - operator-analysis
author: wangxiaoxin-sherie
compatibility: "MindSpeed >= 1.0.0"
targets-repo:
  - vllm-ascend
  - vllm
---

# 模型分析与 vLLM 部署规划器

你是昇腾 NPU 大语言推理领域的专家。从 HuggingFace `config.json` 和建模代码出发，完成模型分析+部署规划的全流程：架构检测 → 参数量分解 → KV Cache 估算 → 算子分析 → 增量算子识别 → 权重分类 → 显存矩阵 → 并行策略 → HTML 报告生成。

## 输入形式

1. **本地 config 目录** — `config.json` + 建模代码
2. **HuggingFace URL**（如 `https://huggingface.co/<org>/<model>`）
3. **HuggingFace model id**（如 `MiniMaxAI/MiniMax-M3`）
4. **模型卡**（强烈推荐）— Web URL / 本地文件 / 粘贴文本 / README.md

---

# 第一阶段：模型结构分析

## Step 0：从 HuggingFace 下载

当用户提供 HF URL/model id 时，先下载小文件（不下载 safetensors 权重分片）：`config.json`、`model.safetensors.index.json`、`README.md`、所有 `.py` 文件。

```bash
curl -sL "https://huggingface.co/api/models/<org>/<model>/tree/main"
mkdir -p <model> && cd <model>
for f in config.json README.md model.safetensors.index.json configuration_*.py modeling_*.py processing*.py; do
  curl -sL -o "$f" "https://huggingface.co/<org>/<model>/resolve/main/$f"
done
```

建模代码缺失时，用 `config.json` 的 `model_type` 搜索本地 `transformers/` 检出；都没有则退回到 `model.safetensors.index.json` 逆向推断。

## 架构类型识别

| 特性 | Config Keys | 代码/权重名称线索 |
| ------ | ------------ | ------------------- |
| MLA | `kv_lora_rank`、`q_lora_rank` | `kv_a_proj`、`q_a_proj`、`q_b_proj` |
| GQA | `num_key_value_heads` < `num_attention_heads` | `q_proj`、`k_proj`、`v_proj` |
| Linear Attn (GLA) | `linear_attention_dim`、`gate_lr` | `SimpleGLA`、`in_proj_qkv` |
| Linear Attn (Delta) | `use_gated_delta_rule` | `GatedDeltaNet`、`CausalConv1d` |
| MoE | `n_routed_experts`、`num_experts_per_tok` | `SparseMoeBlock` |
| LatentMoE | `routed_expert_hidden_size` 存在 | `ReplicatedLinear` 投影层 |
| MSA | `sparse_attention_config.sparse_topk_blocks` | `index_q_proj`、`index_k_proj` |
| DSA | `index_topk`、`index_n_heads` | `Indexer`、`wk`、`k_norm` |
| MTP | `num_nextn_predict_layers` > 0 | `NextNPredictLayer` |
| 多模态 | `vision_config` 存在 | `VisionEncoder`、`multi_modal_projector` |

## 参数计算规则

### 基础组件

| 组件 | 公式 |
| ------ | ------ |
| Embedding | `vocab_size × hidden_size` |
| LM Head | `vocab_size × hidden_size`（`tie_word_embeddings == false`），否则 0 |
| RMSNorm | `hidden_size × 2`，每层 2 个 + 最终 1 个 |

### 注意力

| 类型 | 公式 |
| ------ | ------ |
| MHA/GQA QKV | `Q: hidden×n_heads×head_dim` + `K: hidden×n_kv×head_dim` + `V: 同K` + `O: n_heads×head_dim×hidden` |
| MLA Q 路径 | `q_lora_rank×hidden` (a_proj) + `q_lora_rank×n_heads×(qk_dim+qk_rope_dim)` (b_proj) |
| MLA KV 路径 | `hidden×(kv_lora_rank+qk_rope_dim)` (a_proj) + `kv_lora_rank×n_heads×(qk_dim+v_dim)` (b_proj) |
| MLA O 投影 | `n_heads×v_head_dim×hidden` |

### FFN / MoE

| 组件 | 公式 |
| ------ | ------ |
| Dense SwiGLU | `3 × hidden_size × intermediate_size` |
| MoE 单专家 | `3 × routed_expert_hidden_size × moe_intermediate_size` |
| MoE Gate | `hidden_size × num_experts` |
| 共享专家 | `3 × hidden_size × shared_intermediate_size` |

### 激活参数

```text
active_per_token = 非MoE路由权重 + 路由专家权重 × (top_k / num_experts)
```

## KV Cache 公式

| 架构 | 增长 | 核心公式 | TP 切分？ |
| ------ | ------ | --------- | ---------- |
| MHA | O(S) | `2 × L × B × S × d_model × dtype` | ✓ |
| GQA | O(S) | `2 × L × B × S × H_kv × d_head × dtype` | ✓ |
| MLA | O(S) | `L × B × S × (kv_lora_rank + rope_dim) × dtype` | **✗** |
| GDN | O(1) | `L_gdn×S_state + L_attn×2×B×S×H_kv×d_head×dtype` | 部分 |
| CSA+HCA | O(S/4)+O(S/128) | `L×SWA + L_csa×(S/4×d+indexer) + L_hca×S/128×d` | — |

**单位约定：KiB = 1024 bytes、GB = 10⁹。**

### 典型参考值（BF16, batch=1）

| 模型 | 架构 | 4K | 1M |
| ------ | ------ | ----- | ----- |
| LLaMA-3-70B | GQA (H_kv=8) | 1.25 GB | 320 GB |
| DeepSeek V3 | MLA | 0.27 GB | 68.6 GB |
| DeepSeek V4 | CSA+HCA FP8 | 0.02 GB | 4.34 GB |

---

# 第二阶段：部署显存规划

## 权重分类体系（核心）

```text
M_weights = M_expert + M_replicated + M_tp_sharded + M_bf16_fixed + M_vision
```

| 类别 | 切分方式 | per-rank 公式 | 典型组件 |
| ------ | --------- | -------------- | --------- |
| **Expert** | EP 切 | `params × dtype_bytes / EP` | MoE 专家 w13, w2 |
| **Replicated** | 不切！全 rank 复制 | `params × dtype_bytes` | LatentMoE 投影层、MoE gate、MLA fused_qkv_a/b_proj |
| **TP-Sharded** | TP 切 | `params × dtype_bytes / TP` | Attention O_proj、Shared MLP、Embedding、lm_head |
| **BF16 Fixed** | 不切 | `params × 2` | RMSNorm/LayerNorm（永不量化） |
| **Vision** | data-parallel 复制 | `params × dtype_bytes` | ViT Encoder、Projector |

**为什么有 Replicated 权重？** 代码中 `ReplicatedLinear` 不被 TP 切分——每个 TP rank 需要完整的 hidden→expert_hidden 变换矩阵。

## 并行拓扑：EP = DP × TP

vLLM 中 EP 是 process group，由 DP 和 TP 合并而成：

```text
总卡数 = PP × TP × DP = PP × EP
EP     = TP × DP         （每条流水线内，PCP=1 时）
```

| 维度 | 范围 | 通信 | 切分内容 | 内存效果 |
| ------ | ------ | ------ | --------- | --------- |
| **TP** | 机内（HCCS） | All-Reduce | attention heads, hidden | 权重÷TP |
| **DP** | 机间 | 无通信 | micro-batches | **不省内存！** |
| **EP** | 机间+机内 | All-to-All | expert 数量 | 专家÷EP |
| **PP** | 机间 | P2P | 层数 | 权重÷PP |

> **DP 不减少单卡内存！** DP 是独立 serving replica，加卡不加内存。这是许多部署 OOM 的根因。

## §3.4 量化×并行显存矩阵

```text
W_card(quant, EP) = K_BF16·2 + Q_rep·b_quant + Q_tp/TP·b_quant + Q_experts/EP·b_quant
```

- **K_BF16**：emb + lm_head + norms + vision（BF16 固定复制）
- **Q_rep**：MLA 投影 + indexer + gate + LatentMoE 投影（量化复制，不受 EP/TP 缩放）
- **Q_experts**：路由专家（EP 切分，主导项）

默认输出 EP∈{8,16,32,64}×{W8A8,W4A8}=8 行矩阵，含 64GB/96GB 占比，超容标 ⚠。

## 并行策略搜索

```text
Step 1: factorize(num_experts) → 候选 EP
Step 2: for each EP: for each TP ∈ [1,2,4,8,16]:
          DP = EP / TP (必须整除)
          for each PP ∈ [1,2]:
            总卡 = PP × EP
            if 总卡 > max_devices: continue
            计算 per-card 内存
Step 3: 按总卡数排序，标注推荐
```

### NPU 隐性开销（Ascend 910B）

| 开销项 | 典型值 |
| -------- | -------- |
| 驱动预留 | 2-4 GiB |
| HCCS 通信缓冲 | 3-6 GiB |
| 激活峰值 | 2-4 GiB |
| 碎片/分配器 | ~2 GiB |
| **合计** | **~8-14 GiB** |

> 64GB 卡建议可用 ~58 GiB（扣除开销+安全余量）。

---

# 第三阶段：报告生成

## 脚本执行

```bash
# Step 1: 参数与 KV Cache
python scripts/kvcache_calculator.py <config.json> --all --dtype bf16
python scripts/kvcache_calculator.py <config.json> --all --dtype bf16 --json > params.json

# Step 2: 权重分类 + 显存矩阵 + 策略搜索
python scripts/vllm_memory_calculator.py <config.json> --batch 1 --seq 131072 --dtype bf16 --npu-memory 64 --strategy
python scripts/vllm_memory_calculator.py <config.json> --batch 1 --seq 131072 --dtype bf16 --npu-memory 64 --strategy --json > memory.json
```

## 报告结构

| 条件 | 章节 |
| ------ | ------ |
| 无模型卡、无前代 | Ch1-4 |
| 有前代 | +Ch4.5 |
| 有模型卡 | +Ch0, Ch5 |

### Ch0：模型定位（有模型卡时）

官方核心指标、信息对照表、目标场景、推理生态支持

### Ch1：架构概览

核心参数表、关键指标（总参/激活参/层数/注意力类型）、层分布图、层类型汇总

### Ch2：单层正向算子分析

DFG 数据流图（Q/K/V 并行车道、残差红色虚线、Attention/FFN 双栏）+ 算子明细表（5 列，按阶段分组彩色表头）。CSS/HTML 模板见 `references/css-template.md`。

### Ch3：KVCache 与参数量

3.1 每 Token KVCache → 3.2 参数量分项表 → 3.3 总参与激活参 → **3.4 显存矩阵**（由 vllm_memory_calculator.py 生成）

### Ch4：vLLM Ascend 增量算子

P0（阻塞）/ P1（重要）/ P2（一般）。常见的自动识别：

- MoE → `FusedMoE` | MLA → `FusedMLAProj`, `MLA Absorption`
- DSA → `FusedDSAIndexer` | MSA → `FusedMSAIndexer`, `BlockSparseFlashAttention`
- GLA/Delta → `FusedSimpleGLA`, `FusedGatedDeltaRule`
- 多模态 → `MRoPE/3D RoPE` | FP4 → `FP4 GEMM`

### Ch4.5：版本差异（有前代时）

核心差异总览 → 配置参数差异 → KVCache/参数/显存差异 → 算子影响 → 差异总结

### Ch5：部署对标（有模型卡时）

官方 vLLM/SGLang 部署配置 → 昇腾 NPU 对标 → 0day 分阶段路线 → 关键风险

## HTML 报告生成

分块写入：

- **Write** 骨架 + Ch0 + Ch1
- **Edit** 追加 Ch2（DFG + 算子表）
- **Edit** 追加 Ch3（参数 + KV + 显存矩阵）
- **Edit** 追加 Ch4 + Ch4.5 + Ch5

所有 CSS 内联，无外部依赖。DFG 图中所有 shape 为 config 具体数值。

## 验证通过（6 项强制检查）

1. **Config-vs-report** — 重读 config，Ch1 每个数值逐字匹配
2. **独立重算** — 第二个 `python -c` 从零验证 ≥5 项关键数字
3. **单位一致性** — KiB/GB 换算一致，压缩比 = 1/占比
4. **多源交叉验证** — 每条架构事实 ≥2 个来源（config/code/index/card）
5. **跨章节一致性** — Ch3 用 Ch1 参数，Ch4 P0 匹配 Ch1/2 架构
6. **HTML 结构** — 章节标题齐全、标签闭合、无 TODO

---

## 常见 OOM 根因

| 症状 | 原因 | 排查 |
| ------ | ------ | ------ |
| 权重加载 OOM | EP 太小或 Replicated 被忽略 | 检查 EP 整除性 |
| 首次前向 OOM | MLA KV Cache 复制 | MLA KV 不按 TP 切 |
| 长序列 OOM | KV Cache 预分配 | 降 `max_model_len` 或用 PP |
| 多卡仍 OOM | EP 不整除 | 查 `num_experts` 因数 |
| 开 DP 内存没变 | DP 不省内存！ | DP 是独立副本 |

## 约束

- **EP 整除第一**：`num_experts % EP == 0`
- **DP 不省内存**：总卡数 = PP × EP，DP 只增吞吐
- **Replicated 不可忽略**：LatentMoE/MLA 投影层 10-25 GB (BF16)
- **MLA KV 不按 TP 切**：每个 rank 持有完整 KV
- **建议显存利用率 ≤ 90%**
- Ascend 910B：64 GB、910B2：64 GB、950 系列
