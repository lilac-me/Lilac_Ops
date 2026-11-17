# 已成功接入的 modeling 示例

优先读取下面列出的不可变 commit，不从当前分支的同名文件推断“已验证”。不在 Skill 中复制完整源码，避免示例与仓库实现发生版本漂移。只读取与当前候选算子相关的完整函数。

## Qwen3-MoE

纯 eager 接入检查点：

```text
9885cfc3c83331632dbda869e459a2ab45434f43 — mindspeed_llm/fsdp2/models/qwen3/modeling_qwen3_moe.py
```

最终逐项融合检查点：

```text
6d82b5e403416eec9b775e60dbbf47d1f454a25b — mindspeed_llm/fsdp2/models/qwen3/modeling_qwen3_moe.py
```

适合参考：

- `Qwen3MoeAttention.forward`：Q/K projection 后保留 `q_norm`/`k_norm`，再应用 RoPE。
- `Qwen3MoeExperts.ep_forward`：eager/fused 在同一三维专家参数布局上切换，不替换模块或参数结构。
- `Qwen3MoeRMSNorm.forward`：保留 FP32 计算语义并回到输入 dtype。
- `flash_attention_forward`：纯 causal 条件、mask 和 `sparse_mode` 的配套实现。

使用 `git show <commit>:<path>` 读取。commit 不存在时将参考标记为 unavailable，不得静默退回当前分支的 `qwen3_moe.py` patch 文件。`9885cfc3c83331632dbda869e459a2ab45434f43` 用于阶段二结构参考；`6d82b5e403416eec9b775e60dbbf47d1f454a25b` 仅用于阶段五对应融合函数参考，不能反向证明阶段二应提前包含融合。

## GPT-OSS

路径：

```text
mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py
```

适合参考：

- `GptOssRMSNorm.forward`：`use_fused_rmsnorm` 开关及 eager fallback。
- `GmmFunction` 和 `npu_group_gemm`：显式 `torch.autograd.Function` forward/backward。
- `GptOssFusedExperts`、`GptOssTopKRouter` 和 `GptOssMLP`：Grouped GEMM 专家计算与开关选择。
- `GptOssRotaryEmbedding.forward` 和 `apply_rotary_pos_emb`：fused RoPE 的 cos/sin 生成、广播和 fallback。
- `flash_attention_forward` 和 `GptOssAttention.forward`：NPU Flash Attention 的调用边界和 eager fallback。

## 使用原则

- 新接入使用本地完整 modeling 和类内分支；历史 patch 文件只能用于诊断，不能作为新适配模板。
- 只迁移当前候选算子所需的最小逻辑，不整段复制 modeling 文件。
- 重新推导目标模型的 shape、dtype、layout、mask、权重和 checkpoint 语义。
- 示例中的固定值、并行假设和 Transformers 兼容代码不自动适用于其他模型。
- 即使来自成功示例，每次接入仍必须执行本 Skill 的相邻远端训练与精度验证。
- 不再使用先前失败的 fused modeling 文件作为参考来源。
