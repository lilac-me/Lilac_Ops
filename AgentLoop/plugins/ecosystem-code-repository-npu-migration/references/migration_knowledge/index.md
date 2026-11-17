# 迁移知识索引

初始仓库适配和选择修复假设前，先使用这个索引。使用当前仓库的具体 API、exception、framework、task family、backend 和 topology 搜索。只阅读与当前调用链匹配的主题。

## 主题路由

| 信号或搜索词 | 阅读 |
| --- | --- |
| `torch_npu`、`.cuda()`、`torch.cuda`、设备选择、AMP、dtype、stream、event、memory、DataParallel、软件包漂移 | [backend_runtime.md](backend_runtime.md) |
| NCCL、HCCL、Gloo、process group、rank、collective、Ray、vLLM、可见性、placement、权重同步 | [distributed_serving.md](distributed_serving.md) |
| FlashAttention、fused attention、Triton、CUDA extension、不支持的 operator、packing、varlen、mask、position ID、quantization | [operators_semantics.md](operators_semantics.md) |
| checkpoint、state dict、optimizer、RNG、dataloader resume、Hydra、YAML、shell wrapper、launcher、CLI override | [checkpoint_launchers.md](checkpoint_launchers.md) |

失败跨越边界时组合多个主题。例如，启动 Ray actor 并在权重传输期间 hang 的 GRPO launcher，需要同时阅读 launcher 和 distributed-serving 主题。

## 查询方法

1. 提取精确的本地信号：最深层 exception、失败 symbol、导入模块路径、task family、runtime、topology 和调用链。
2. 使用这些信号通过 `rg -n -i` 搜索当前目录，并阅读匹配主题的章节。
3. 根据当前源码、调用方、版本保护和已安装软件包来源确认模式。措辞相似不等于根因相同。
4. API 签名、兼容性说明、operator 支持结论或 framework 行为可能随版本变化时，在编辑前刷新当前官方来源。
5. 应用一个有边界的改动，先验证受影响层，再验证原始仓库目标。知识只建议检查位置；是否有效由执行决定。

## 官方来源顺序

优先选择与已安装或请求发布版本匹配的来源：

1. [TorchNPU 官方仓库](https://github.com/Ascend/pytorch)，以及其发布说明、维护分支和版本化文档。
2. [昇腾 PyTorch 安装文档](https://ascend.github.io/docs/sources/pytorch/install.html)，用于当前 CANN、Python、torch 和 torch_npu 兼容性。
3. 版本化的 `hiascend.com` TorchNPU API、operator 支持、迁移和 HCCL 文档。
4. 上游 PyTorch 文档中的通用 PyTorch 语义。
5. 调用链中精确 framework 的官方文档和维护源码，例如 Ray、vLLM-Ascend、Transformers、DeepSpeed 或 Megatron-LM。

网络来源影响补丁时，在当前任务证据中保留精确 URL、访问日期、版本上下文和使用的结论。文档不能替代本地导入来源或真实 NPU 执行。

## 决策边界

- 初始适配期间，只有范围内 caller 实际到达共享问题且 backend 感知改动有边界时，才修复它。
- entry 专属的资产、命令、规模和不确定 operator 改动，推迟到对应目标准备好或失败时处理。
- 保留 CUDA/GPU 和 CPU 行为。绝不把缺失的 NPU 实现变成静默 CPU 路径或弱化断言。
- 通过的导入或 helper 检查只作为该层证据，不作为仓库支持证明。
