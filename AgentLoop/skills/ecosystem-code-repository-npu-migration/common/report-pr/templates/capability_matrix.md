# 能力矩阵

| 能力 | 状态 | 证据 | 命令或来源 | 剩余风险 |
| --- | --- | --- | --- | --- |
| 功能模型清单 | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |
| 基础设备迁移 | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |
| AMP 与 dtype | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |
| 分布式/HCCL | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |
| 仓库原生多卡训练 | UNKNOWN_NEEDS_DOC_OR_SMOKE | 仅通用 HCCL smoke 不足以证明 | | |
| Checkpoint 与 optimizer state | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |
| 自定义 kernel 与 compiler | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |
| Packing/varlen/padding-free | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |
| Quantization | NOT_APPLICABLE | 默认排除，除非用户请求 | | |
| TensorFlow/TFLite | NOT_APPLICABLE | 默认排除，除非用户请求 | | |
| CPU fallback 路径 | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |
| 核心缺口闭环 | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |
| 性能适配 | UNKNOWN_NEEDS_DOC_OR_SMOKE | | | |

状态值：`SUPPORTED_VERIFIED`、`SUPPORTED_DOC_ONLY`、`SUPPORTED_WITH_NPU_PATH`、`CUDA_ONLY_OR_BLOCKED`、`UNKNOWN_NEEDS_DOC_OR_SMOKE`、`NOT_APPLICABLE`。
