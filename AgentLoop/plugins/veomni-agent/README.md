# VeOmni Agent Plugin

VeOmni 模型迁移插件（对应 [RFC #15](https://gitcode.com/Ascend/MindSpeed-Agent/issues/15)）提供昇腾 NPU 上从代码适配到训练、精度和性能验证的完整闭环。

## 安装

在 MindSpeed-Agent 仓库根目录执行安装脚本；脚本会注册本仓库为 marketplace 并安装 `veomni-agent`：

```bash
# Claude Code
bash install.sh claude-code

# Codex
bash install.sh codex
```

Windows CMD 使用：

```cmd
install.bat claude-code
install.bat codex
```

安装后可在 Claude Code 或 Codex 中直接描述模型、VeOmni 工作副本、目标 NPU 环境，以及是否授权安装依赖、下载模型和启动单卡/多卡训练。

## 能力

- 适配并修改 dataset、dataloader、model、config、tokenizer/processor、Trainer、checkpoint 与注册入口。
- 在已授权环境中运行加载、前向、反向、最小训练、并行训练、精度对齐、Profiling 和性能优化。
- 通过 JSON/YAML 阶段门禁记录代码版本、命令、日志与测试结果；性能优化后自动要求精度复测。

## 最终报告仅两份

中间 JSON/YAML 是机器可验证的执行证据。最终阶段还会生成 `acceptance_report.json`，用于兼容 RFC 的自动验收、阶段状态、回归范围和复现命令；它不是额外的人工报告。每个任务对用户只输出以下两份 Markdown：

1. `outputs/{model}/delivery_coverage_report.md`：交付件覆盖、模块映射、代码改动、训练/精度/性能状态、风险与下一步。必含 dataset、dataloader、model、config。
2. `outputs/{model}/migration_process_record.md`：实际执行步骤、环境与版本、命令、日志/产物路径、决策、失败与回退。

## 快速开始

```text
将 <模型> 接入 VeOmni：允许修改代码，并在 <目标 NPU 环境> 运行最小训练、精度校验和性能调优。
最终只输出交付件覆盖报告与过程记录，重点展示 dataset、dataloader、model、config。
```

交付报告结构可通过以下命令检查：

```bash
python benchmarks/veomni/quick_migration/check_outputs.py \
  outputs/<model>/delivery_coverage_report.md \
  outputs/<model>/migration_process_record.md
```

## 版本

| 版本 | 说明 |
| ------ | ------ |
| 0.2.1 | 增加 Flux DiT SFT 的配置、权重加载、批量前向和图像条件训练前置清单。 |
| 0.2.0 | 保留完整迁移、训练、精度和性能闭环；最终报告收敛为两份 Markdown。 |
| 0.1.0 | 首期端到端接入闭环。 |
