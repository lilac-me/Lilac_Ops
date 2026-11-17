# Operator 与语义模式

## 不支持的 operator 与自定义 kernel

识别信号包括 unsupported-op 消息、kernel 缺失、`CUDAExtension`、`.cu`/`nvcc`、Triton CUDA 假设、fused library，以及无条件导入 CUDA 的第三方软件包。

检查数学操作、输入/输出 shape、layout、dtype、autograd、mask 语义、caller 频率和现有 fallback。选择替代方案前查询当前 TorchNPU API 和 operator 支持文档。

按以下顺序优先选择：已有的设备无关仓库路径、文档化的 PyTorch/TorchNPU operator，或可与参考实现比较语义的小型纯 PyTorch 实现。保留优化后的 CUDA 路径。没有明确需求和用户授权，不要添加 `triton-ascend` 或其他项目依赖。

先使用明确容差，将隔离的 operator 与可信 CPU 或仓库参考实现比较，再验证 forward/backward 和模型集成。仅导入或编译成功不能证明数值或梯度正确性。

## FlashAttention 与 fused attention

信号包括 `flash_attn`、unpadding helper、Q/K/V packed layout、`cu_seqlens`、causal 对齐和自定义 cache helper。不要仅因为存在 NPU fused-attention symbol 就替换 CUDA FlashAttention 调用。

确认当前 TorchNPU 签名，并映射实际 tensor layout、head dimension、dtype、scale、causal mask、sparse mode 和返回值。dense 与 variable-length 路径需要分别分析。保留 caller 会消费的辅助输出或 autograd 行为。

分别验证 dense 和 varlen 场景；在支持时包含不等长 query/key、forward/backward 对齐和真实模型路径。一次 dense forward 不能证明 packed、cached 或训练行为。

## Packing、varlen 与 padding-free 执行

检查 `cu_seqlens`、sample-boundary metadata、attention mask、position ID、cache layout、padding 移除和恢复。验证 metadata 在 batching、sharding、repeat、reorder 和分布式 dispatch 过程中保持对齐。

验证顺序是隔离 helper 或 kernel、forward/backward、数学比较，然后是模型集成。在可行时加入 negative control：损坏的边界 metadata 应改变结果或明确失败。普通 padded test 不能证明 packed 正确性。

## 量化与 CPU 回退

信号包括 bitsandbytes、GPTQ、AWQ、exllama、torchao、int4/int8 CUDA kernel 和 quantized matmul API。除非用户请求，quantization 默认不在范围内，但检测到的路径必须明确分类。

在范围内时，识别精确的 quantization math、calibration/state format、kernel backend 和 serialization contract。只有已安装版本支持文档化的 NPU 路径时才使用它，并以合适容差将输出与可信参考比较。

CPU fallback 可能有助于诊断，但不是核心操作已验证 NPU 支持的证据。如果它保留，报告精确路径和语义/性能风险；绝不要用成功退出码掩盖它。
