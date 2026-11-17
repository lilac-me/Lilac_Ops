# 工作流与数据合同

## 目录

1. 输入 payload
2. logits payload
3. 比较报告
4. 架构与执行奇偶性
5. 生产器实现
6. 阶段和能力矩阵

## 1. 输入 payload

使用单个生成器产生两个框架共用的输入。建议字段：

```python
{
    "format_version": 1,
    "input_ids": torch.int64,        # [B, S], CPU
    "attention_mask": torch.bool,    # [B, S], CPU
    "input_sha256": str,
    "metadata": {
        "tokenizer_path": str,
        "tokenizer_commit": str,
        "padding_side": "right",
        "special_tokens": dict,
        "prompts": list[str],
    },
}
```

hash 必须基于规范化后的 `input_ids`、`attention_mask`、shape 和 dtype 计算。
比较器同时核验顶层输入与嵌入在两份 logits payload 中的输入。

第一阶段支持无 padding 或右 padding。左 padding、packed sequence、cache 和生成
应作为独立合同，不与最小基准混合。

## 2. logits payload

```python
{
    "format_version": 1,
    "framework": "transformers" | "mindspeed-mcore",
    "logits": torch.float32,          # [B, S, V], CPU
    "input_ids": torch.int64,
    "attention_mask": torch.bool,
    "input_sha256": str,
    "metadata": {
        "model_identity": str,
        "checkpoint_identity": str,
        "code_commit": str,
        "compute_dtype": str,
        "raw_vocab_size": int,
        "saved_vocab_size": int,
        "logits_layout": "BSV",
        "execution_flags": dict,
        "parallel_layout": dict,
        "writer_rank": int,
    },
}
```

保存前要求：

- logits 有限，shape 与输入 batch/sequence 相同；
- 保存为 CPU FP32；
- 仅裁剪已知的 MCore padded vocab；
- 不做 softmax、clamp、NaN 替换或归一化；
- 记录 producer 版本、checkpoint 和全部影响数值的执行 flag。

大 tensor 按 token 分块搬运：

1. 预分配 CPU FP32 `[B,S,V]`；
2. 将连续 token chunk 以源 dtype 搬到 CPU；
3. 在 CPU 上转 FP32；
4. 写入预分配 tensor；
5. 记录 chunk 大小和传输策略。

## 3. 比较报告

比较器必须在模型初始化之外运行，并保持 CPU 可执行。检查顺序：

1. payload schema；
2. 输入 hash 和内容；
3. `[B,S,V]` shape；
4. 有限性和近零向量；
5. `attention_mask == 1` 的有效位置；
6. 逐 token 全词表余弦；
7. 聚合主指标和诊断指标。

默认报告至少包含：

```text
passed
primary_metric
threshold
input_sha256
input_shape
logits_shape
valid_token_count
average/minimum/median/maximum_token_cosine
first_valid_token_cosine
last_valid_token_cosines
flatten_cosine
mean_abs/max_abs/rms_error
top1_match_rate/topK_mean_overlap
reference/candidate metadata
```

主判定严格使用 `average_token_cosine >= threshold`。绝对误差没有跨模型统一阈值。

## 4. 架构与执行奇偶性

实现前建立表格：

| 合同 | Transformers | MCore | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| embedding/LM head tied | | | | |
| norm 类型/位置/epsilon | | | | |
| residual 顺序 | | | | |
| attention scale | | | | |
| QKV packing/GQA | | | | |
| RoPE width/layout/scaling | | | | |
| mask shape/value | | | | |
| MoE router/top-k | | | | |
| MLA/DSA 层模式 | | | | |
| MTP 所有权 | | | | |
| logits scale/trim | | | | |

条件层再建立：

```text
global layer -> HF type/keys -> MCore type/keys
             -> PP rank/local layer -> checkpoint shard
```

`num-layer-list` 下使用前缀和，不能用均匀 PP 公式。运行时构造、转换和导出都要
证明使用同一层身份。

## 5. 生产器实现

优先调用通用工具的 hook：

- model provider；
- custom forward；
- output selector；
- logits layout。

模型专用适配器只实现配置修正、预期 checkpoint 键、运行时结构、forward 差异、
诊断模块和元数据。不得因为返回类型不同就复制完整 dumper。

所有 rank 参与 PP schedule；只有最后 pipeline stage 的指定 writer 保存。
TP 词表切片先 gather，再保存完整 V。

Transformers 加载必须检查 missing、unexpected、mismatched 和新初始化参数。
API 返回的诊断容器可能是 list/tuple/set/frozenset；set 类容器应排序为稳定列表，
但不得删除错误条目。

## 6. 阶段与能力矩阵

阶段名称不等于 kernel 覆盖。明确区分：

| 阶段 | 语义特性 | 执行路径 | 必需 preflight | 输入条件 |
| --- | --- | --- | --- | --- |
| direct | 基础架构 | eager/direct | tensor 合同 | 短序列 |
| sparse-semantic | 稀疏选择语义 | direct attention + sparse mask | mask/index | `S > topk` |
| fused-sparse | 融合稀疏 kernel | 真实融合算子 | target-runtime operator | `S > topk` |
| absorb | MLA 吸收 | absorbed path | 维度/布局恢复 | 模型条件 |
| long-context | RoPE 长上下文 | 指定 kernel | RoPE 行为回归 | 跨缩放边界 |
| cache/generation | 增量语义 | KV cache | cache layout | 独立输入 |

每阶段保存直接前驱、flag、输入 hash、producer 合同和认证状态。阶段严格通过仅表示
该阶段及已认证前驱链通过，不自动覆盖未执行的优化路径。
