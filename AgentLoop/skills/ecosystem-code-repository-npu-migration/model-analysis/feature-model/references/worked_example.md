# 功能模型完整示例

## 仓库范围

```text
tests/test_npu_device.py
examples/sft_lora_tiny.py
examples/sft_lora_large.py
examples/sft_full.py
examples/grpo_vllm.py
```

源码和文档检查显示三个用户可见的训练功能：

- 通过仓库原生 trainer 执行 LoRA SFT；
- 通过同一 trainer 执行全参数 SFT；
- rollout 路径使用 vLLM 的 GRPO。

设备测试隔离了共享的 `device_selection` 组件能力。

## 分类

两个 LoRA 脚本只有模型和数据集大小不同。它们的 trainer、update strategy、单设备 topology、precision 和非 packed 数据路径相同，因此组成 `sft-lora-single`。

`sft_full.py` 使用不同的 optimizer 和参数更新路径，因此即使同样属于 SFT，也组成 `sft-full-single`。

`grpo_vllm.py` 增加生成、rollout/update 数据流和 vLLM runtime，因此组成 `grpo-vllm-single`。

## 代表项选择

- 为 `sft-lora-single` 选择 `sft_lora_tiny.py`，将大资产变体 deferred 到该代表项。
- 为 `sft-full-single` 选择 `sft_full.py`。
- 为 `grpo-vllm-single` 选择 `grpo_vllm.py`。
- 将 `test_npu_device.py` 保留为单独的低成本 capability check。

这样可以在不隐藏任何 entry 的情况下，将 4 个 E2E entry 减少为 3 个 E2E 代表项。完整的配套输出位于 `../examples/feature_model.json` 和 `../examples/coverage_plan.json`。
