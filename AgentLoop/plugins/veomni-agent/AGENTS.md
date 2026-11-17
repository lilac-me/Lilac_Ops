# VeOmni Agent Plugin — 迁移闭环

> YAML 状态机：`plugins/veomni-agent/veomni.yaml`
> 阶段产物 Schema：`benchmarks/veomni/schemas/`

## 定位

VeOmni Agent 支持在昇腾 NPU 上完成完整模型迁移：环境检查、模型分析、代码适配、训练验证、并行策略、精度对齐、MFU 计算和性能调优。实际安装、下载、改代码和训练均需遵守用户授权范围。

## 流程

```text
环境检查 → 模型分析 → 代码适配 → 功能验证 → 并行策略 → 精度验证 → MFU → 性能调优 → 最终交付
```

| Agent | 职责 |
| ------ | ------ |
| veomni-orchestrator | 调度门禁与回退，汇总最终交付件 |
| veomni-adaptation-agent | 环境检查、模型分析、dataset/dataloader/model/config 适配和并行策略 |
| veomni-validation-agent | 功能/训练验证、精度对齐和回归 |
| veomni-performance-agent | MFU、Profiling、性能优化与精度复测 |

## 产物策略

- 阶段 JSON/YAML（如 `model_inventory.json`、`patch_manifest.json`、`accuracy_report.json`、`performance_report.json`）用于自动门禁、Agent 交接和运行证据；最终还必须生成内部验收契约 `acceptance_report.json`。
- 面向人的最终报告**只允许**以下两份 Markdown，写入 `outputs/{model}/`：
  1. `delivery_coverage_report.md`
  2. `migration_process_record.md`

交付件覆盖报告至少涵盖 `dataset`、`dataloader`、`model`、`config`，以及适用的 tokenizer/processor、Trainer/loss、权重/checkpoint、注册入口、并行、训练、精度和性能。每项必须带有源端、目标端、代码改动和验证证据。

过程记录必须包含输入版本、执行命令、环境、数据、日志/结构化产物路径、关键决策、回退和未执行原因。`COVERED` 只能说明具备文件与运行证据；未运行项目必须标记 `NOT_RUN`、`BLOCKED` 或 `NEEDS_HUMAN`。

## 约束

1. generated modeling 文件只能通过 Patchgen 生成。
2. 功能失败回到适配，精度失败定位首个差异，性能优化发生精度回退则回到精度阶段。
3. 不自动执行未经授权的环境安装、模型下载或多机训练。
4. 不生成环境、分析、精度、性能或验收等阶段 Markdown 报告；相关内容统一汇总至两份最终交付件。
