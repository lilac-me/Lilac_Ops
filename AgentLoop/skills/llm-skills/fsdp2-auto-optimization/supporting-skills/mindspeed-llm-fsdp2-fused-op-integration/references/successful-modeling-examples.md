# 已成功接入的 modeling 示例

优先从当前 MindSpeed-LLM 仓库读取下面两个源文件，不在 Skill 中复制完整源码，避免示例与仓库实现发生版本漂移。只读取与当前候选算子相关的类和函数。

## Qwen3-MoE

路径：

```text
mindspeed_llm/fsdp2/models/qwen3/qwen3_moe.py
```

适合参考：

- `Qwen3MoEForCausalLM.register_patches`：根据已有参数开关注册 Grouped GEMM、fused RMSNorm 和 fused RoPE。
- `Qwen3MoeExperts`：专家权重布局、token permute/unpermute、`tokens_per_expert` 和两次 `Ops.gmm`。
- `Qwen3MoeSparseFusedMoeBlock`：router、top-k 权重和 fused experts 的连接方式。
- `qwen3_moe_model_forward`：模型 forward patch 的组织方式。

该文件没有提供应当直接复制到其他模型的通用 Flash Attention 实现。只把它用于 Grouped GEMM、RMSNorm、RoPE 和 patch 注册参考。

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

- 先确认目标 modeling 文件的真实加载和 patch 机制，再选择类内分支或 patch 注册方式。
- 只迁移当前候选算子所需的最小逻辑，不整段复制 modeling 文件。
- 重新推导目标模型的 shape、dtype、layout、mask、权重和 checkpoint 语义。
- 示例中的固定值、并行假设和 Transformers 兼容代码不自动适用于其他模型。
- 即使来自成功示例，每次接入仍必须执行本 Skill 的相邻远端训练与精度验证。
- 不再使用先前失败的 fused modeling 文件作为参考来源。
