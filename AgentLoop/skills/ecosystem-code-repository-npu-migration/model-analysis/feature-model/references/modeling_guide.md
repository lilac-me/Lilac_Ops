# 仓库功能建模指南

## 从仓库行为出发

使用仓库及其用户熟悉的语言描述功能。训练仓库可能提供 SFT、偏好优化、RL、评测、推理、导出、checkpoint 转换和服务；库则可能提供 operator、模型系列、分布式原语或集成能力。

结合每个 wrapper 背后的实现检查文档。目录扫描或命令 grep 只能用于初步定位：它会遗漏软件包内测试和配置驱动的 launcher，还经常把 helper 误判为可运行 entrypoint。

## 区分功能与组件能力

- **feature** 是用户可见的 workload 或行为，例如 LoRA SFT、带 vLLM rollout 的 GRPO、评测或 checkpoint resume。
- **capability** 是可以由专项测试隔离验证的低层前置能力，例如设备选择、attention、optimizer update、checkpoint I/O、分布式 collective、packing/varlen 或自定义 kernel。

使用 `requires_capabilities` 在 feature 上记录组件前置条件。这是规划依赖，不是任一侧可用的证据。

## 分组前检查 NPU 敏感维度

至少比较当前仓库实际存在的以下维度：

- task 或 algorithm；
- update strategy：full、LoRA、adapter、QLoRA；
- runtime 和 launcher：repository-native、Transformers、Accelerate、DeepSpeed、Megatron、torchrun、vLLM；
- model kind：dense、MoE、多模态、encoder/decoder 变体；
- topology 和并行方式：single device、DDP、FSDP、ZeRO、TP、PP、CP、EP、SP；
- precision 和 optimizer path；
- packing、varlen、mask、position ID 和 padding 行为；
- checkpoint save/load/resume 或 weight transfer；
- fused attention、custom operator、compiler path、quantization、offload 和外部 runtime。

两个 entry 的预期行为和这些维度的相关值都相同，才可以共享一个分组。当源码检查表明 backend path 不变时，资产大小、数据集名称、日志位置或训练步数可以不同而无需新建分组。

如果重要维度未知，应保留不确定性，在解决前保持 entry 分开。

## 选择代表项

优先选择仓库维护的 entry，并满足：

1. 能到达真实代码路径，而不仅是解析配置；
2. 使用现有且较小的模型和数据集资产；
3. 需要更少设备和可选依赖；
4. 能产生有界的语义结果，例如数值有限的 loss、optimizer update、生成、collective 或 checkpoint 往返；
5. 易于诊断和重跑。

如果较便宜的 entry 会移除分组需要验证的行为，就不要选择它。对于分布式分组，单设备 fallback 不是代表项；对于 vLLM 集成分组，仅使用 Transformers 的生成脚本不是代表项。

将所有可行的迁移相关 unit/component test 放入 `capability_checks`。不要为了减少运行时间，使用示例式等价分组跳过不同的 unit test。代表性分组中只放仓库示例、可执行脚本和文档启动命令。

## 将规划与证据分开

`feature_model.json` 和 `coverage_plan.json` 描述预期覆盖范围，不记录执行结果。后续工作将真实命令证据与计划中的 feature 和代表项 ID 比对，不会把这些文件改造成 workflow 状态机。
