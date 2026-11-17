# FSDP2 融合算子接入指南

只读取与当前候选算子对应的章节。以下内容用于检查接口与风险，不是可以直接复制的固定代码模板。

## 共通检查

### 找到真实执行路径

同时检查：

- FSDP2 模型注册和实际 `modeling` 文件。
- Transformers patch manager 是否替换类、方法或函数。
- 命令行参数如何进入 `get_args()` 或模型 config。
- 公共实现 `mindspeed_llm/fsdp2/models/common/fusions.py`。
- `successful-modeling-examples.md` 列出的成功 modeling 示例。
- 有训练与精度验证证据的同模型 patch 和其他 FSDP2 实现。

不要引用先前接入失败的 fused modeling 文件，也不要因为文件名、开关或 fused API 存在就把实现视为成功。参考实现只证明一种模型和一种环境下的做法；逐项比较 Transformers 版本、函数签名、tensor layout 和 mask 约定。

### 设计开关和 fallback

- 使用仓库已有参数名，不为同一能力增加第二个开关。
- 在最窄且稳定的函数边界切换 fused/eager。
- 开关关闭时保持原路径、返回类型和副作用不变。
- 缺少 `torch_npu`、MindSpeed op 或必需依赖时给出可理解的错误。
- 对已知不支持的并行组合显式拒绝。

### 验证完整训练语义

至少覆盖 forward、backward、参数梯度、checkpoint 加载和随机初始化。融合算子的输出接近原算子只是必要条件，不代表训练正确。

## MoE Grouped GEMM

### 它改变了什么

普通 MoE 通常遍历本轮命中的 expert，对每个 expert 分别执行小矩阵乘法。Grouped GEMM 先按 expert id 排列 token，形成 `tokens_per_expert`，再用一次 grouped 调用提交多个不同尺寸的矩阵乘法，最后把结果恢复到原 token 顺序并乘路由权重。收益主要来自减少 Python/launch 开销并提高 NPU 利用率。

### 必查接口

- `top_k_index` 是 expert id；排序依据就是 expert id，不是重新给 expert 编号。
- permute 输出顺序、`tokens_per_expert` 和 stacked expert weights 的 expert 维顺序必须一致。
- 检查 gate/up 的拼接顺序、激活函数、down projection 和路由权重乘法位置。
- 检查 GMM 接口的 `trans_b`、输入 dtype、输出 dtype 和空 expert 行为。
- 检查 unpermute 是否正确合并一个 token 被 top-k 多个 expert 处理后的结果。

### 权重布局

Grouped GEMM 通常需要带 expert 维的三维权重，但具体轴顺序必须从当前调用的 GMM 接口和 `view/transpose` 代码推导，不能凭名称猜测。对每组权重写出：

```text
parameter shape -> runtime view/transpose -> GMM expected shape
```

分别检查：

- 随机初始化创建的 Parameter shape 和初始化分布。
- HF checkpoint 中每个 expert 的独立参数如何映射或堆叠。
- `state_dict` 保存后能否恢复到相同运行布局。
- `train_from_scratch=True` 是否绕过 checkpoint 转换。即使不需转换，也仍需验证初始化语义和 RNG 可比性。

权重转换脚本、conversion mapping 和运行时 GMM 属于同一功能链，但职责不同：前者把 checkpoint 变成运行时所需布局，后者执行计算。

### 当前边界

在未实现 token dispatcher 和 expert partition 前，显式限制 `ep_size == 1`。不要把单卡/无 EP 的 permute 逻辑宣传为 EP 支持。

确认底层 op 有训练所需的 Autograd 实现。若只注册了 forward，参考 `mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py` 中 `GmmFunction` 的显式 `torch.autograd.Function` forward/backward 结构，但要按当前 GMM API 重新核对。

## Fused RMSNorm

确认融合实现与原实现使用相同：

- epsilon。
- 归一化维度。
- weight 的 shape 和 dtype。
- 输入与输出 dtype 策略。

优先在 RMSNorm 类的 `forward` 开头按 `use_fused_rmsnorm` 切换到公共 `fused_rmsnorm_forward`。关闭开关时保留原始 float32 方差计算路径。

参考 `qwen3/qwen3_moe.py` 的 patch 注册方式，或 `gpt_oss/modeling_gpt_oss.py` 中 `GptOssRMSNorm.forward` 的类内分支方式；根据目标模型真实加载机制选择，不要混用。

## Fused RoPE

确认：

- q/k、cos/sin 广播维和 `unsqueeze_dim`。
- interleaved 与 rotate-half 约定。
- position id、sequence 维和缓存行为。
- 输入输出 layout 与 dtype。

优先复用公共 `apply_rotary_pos_emb`，并保留原实现 fallback。不要只测试 shape；RoPE 广播错误可能正常运行但改变 loss。

参考 `qwen3/qwen3_moe.py` 对 Transformers 函数的 patch 注册，或 `gpt_oss/modeling_gpt_oss.py` 中 `GptOssRotaryEmbedding.forward` 与 `apply_rotary_pos_emb` 的配套实现。

## Flash Attention

### 接口检查

- Q/K/V layout：例如 `BNSD` 或 `TND`。
- `head_num`、GQA/MQA 的 KV repeat 责任。
- scale 和 dropout/keep probability。
- padding mask、causal mask、mask dtype 与 True/False 极性。
- 定长与变长序列、`actual_seq_qlen` 和 `actual_seq_kvlen`。
- 输出是否需要 transpose/reshape/contiguous。
- 当前 CANN/torch_npu 对 `sparse_mode` 的精确定义。

### Causal 陷阱

Transformers 的上游 causal-mask 构造可能在某个 attention 后端下返回 `None`，意思是“后端负责 causal”，而不是“执行双向注意力”。直接把 `None` 传给 NPU fused attention 可能训练继续执行，但 loss 已错误。

不要硬编码一个只适合单一序列长度的全局 mask。若确需缓存：

- 根据当前 query/key 长度切片或扩容。
- 按 device 管理缓存。
- 处理 decode、padding 和变长序列。
- 用小输入和完整训练 loss 双重验证 causal 语义。

Flash Attention 在特定 4K shape 上可能比 eager/SDPA 慢。只有实测达到显式性能门禁才保留。

参考 `gpt_oss/modeling_gpt_oss.py` 中 `flash_attention_forward` 与 `GptOssAttention.forward` 的接口连接方式，同时重新验证目标模型的 causal、sliding-window、GQA、mask 和序列长度语义。

## 已验证的 Qwen3-MoE 经验

在一次固定 4K、30 iteration 的 FSDP2 实验中，逐项结果为：

| 组合 | 去除首步后的平均单步耗时 |
| --- | ---: |
| origin | 8014.63 ms |
| + Grouped GEMM | 1593.80 ms |
| + fused RMSNorm | 1406.90 ms |
| + fused RoPE | 1365.51 ms |
| + corrected Flash Attention | 1395.51 ms |

这是一个案例，不是跨模型结论。它说明：

- Grouped GEMM 可能是 MoE 的主收益项。
- 多个小收益需要逐项测量才能归因。
- Flash Attention 即使数值正确，也可能在当前 shape 上回退约 2.2%。
- 第一次 Flash 尝试因遗漏 causal 语义而能运行但 loss 错误；“不报错”不是正确性证明。
