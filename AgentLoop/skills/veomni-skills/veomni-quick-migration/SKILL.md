---
name: veomni-quick-migration
description: >-
  This skill should be used when the user asks to migrate a model into VeOmni
  with code adaptation, training validation, precision alignment, or performance
  tuning, and wants the final human-facing output limited to a delivery coverage
  report and a process record covering dataset, dataloader, model, config, and
  other applicable modules.
version: 1.1.0
domain: common
function: model-adaptation
tags:
  - veomni
  - migration
  - coverage
  - dataset
  - dataloader
author: mindspeed-agent
license: MIT
compatibility: "VeOmni >= 0.1"
---

# VeOmni 迁移交付汇总

## 目的

本 Skill 用于完整迁移闭环的最终交付阶段，不替代代码适配、训练、精度和性能 Skill。前序阶段可以修改代码并在获得授权后运行训练、精度校验和性能调优；其 JSON/YAML 产物是可追溯证据。最终只生成两份面向人的 Markdown。

## 汇总步骤

1. 读取环境、模型分析、Patch、功能训练、并行、Golden、精度、MFU 和性能阶段的 JSON/YAML，以及实际代码 diff、命令和日志。
2. 对 `dataset`、`dataloader`、`model`、`config` 做强制覆盖；按模型实际情况纳入 tokenizer/processor、trainer/loss、weights/checkpoint、registry/entrypoint、parallel、training、precision、performance。
3. 每个模块均记录源端证据、VeOmni 目标、代码改动、执行证据和状态。状态使用 `COVERED`、`PARTIAL`、`NOT_REQUIRED`、`BLOCKED`、`UNKNOWN`；运行类步骤可额外标记 `PASS`、`FAIL`、`NOT_RUN`。
4. 未运行训练、精度或性能时，保留原始阻塞原因和授权状态，禁止推断通过。
5. Flux DiT SFT 场景还须记录 `dp_shard_size`、`global_batch_size`、`vae_subfolder`、`architectures`、本地权重加载、列表 `forward` 和图像 resize 的实际改动与验证证据。
6. 先写 `acceptance_report.json`（阶段状态、回归、复现命令和结论），再写 `delivery_coverage_report.md` 和 `migration_process_record.md`，然后运行交付件校验器。

## 交付件一：覆盖报告

写入 `outputs/{model}/delivery_coverage_report.md`。

```markdown
# VeOmni 迁移交付件覆盖报告

## 范围与结论

- 源仓库与版本：<路径 / 分支或 commit>
- VeOmni 仓库与版本：<路径 / 分支或 commit>
- 代码改动：<commit、diff 或文件列表>
- 总结：<覆盖与验证结论；未运行项必须单独说明>

## 模块覆盖明细

| 模块 | 源端证据 | VeOmni 目标 | 代码改动 | 训练/验证证据 | 状态 | 缺口与下一步 |
|---|---|---|---|---|---|---|
| dataset | `path: symbol` | `path: symbol` | <改动> | <样本加载日志> | COVERED | 无 |
| dataloader | `path: symbol` | `path: symbol` | <改动> | <batch/collate 日志> | PARTIAL | <下一步> |
| model | `path: symbol` | `path: symbol` | <Patchgen/注册改动> | <forward/训练/精度证据> | COVERED | 无 |
| config | `path: key` | `path: key` | <改动> | <实际运行命令> | COVERED | 无 |

## 训练、精度与性能

| 项目 | 执行状态 | 结构化证据 | 结论 |
|---|---|---|---|
| 最小训练 | PASS / FAIL / NOT_RUN | `functional_report.json` | <结论> |
| 精度校验 | PASS / FAIL / NOT_RUN | `accuracy_report.json` | <结论> |
| 性能调优 | PASS / FAIL / NOT_RUN | `performance_report.json` | <结论> |

## 风险与待确认事项

1. <风险、影响和下一步>
```

## 交付件二：过程记录

写入 `outputs/{model}/migration_process_record.md`。过程必须记录实际发生的改动和执行，不可将计划写成结果。

```markdown
# VeOmni 迁移过程记录

## 输入与可追溯信息

| 项目 | 内容 |
|---|---|
| 需求 | <原始需求摘要> |
| 源仓库 / VeOmni 版本 | <路径、分支或 commit> |
| 环境 | <NPU、CANN、TorchNPU、Python 等> |
| 数据与权重 | <版本、位置或访问限制> |

## 执行记录

| 序号 | 阶段 | 实际操作/代码改动 | 命令或日志证据 | 结果 |
|---|---|---|---|---|
| 1 | 环境检查 | <实际操作> | `environment_report.json` | PASS / BLOCKED |
| 2 | 模型分析 | <实际操作> | `model_inventory.json` | PASS |
| 3 | 代码适配 | <实际改动> | `patch_manifest.json` | PASS / FAIL |
| 4 | 训练验证 | <实际命令> | `functional_report.json` | PASS / FAIL / NOT_RUN |
| 5 | 精度校验 | <实际命令> | `accuracy_report.json` | PASS / FAIL / NOT_RUN |
| 6 | 性能调优 | <实际命令与改动> | `performance_report.json` | PASS / FAIL / NOT_RUN |

## 关键决策、失败与回退

1. <决策、依据和结果>

## 未决项与授权边界

1. <缺失信息、阻塞原因或需要授权的动作>
```

## 自检

```bash
python benchmarks/veomni/quick_migration/check_outputs.py \
  outputs/{model}/delivery_coverage_report.md \
  outputs/{model}/migration_process_record.md
```

校验只检查最终 Markdown 的结构；训练、精度和性能是否通过以阶段 JSON/YAML 和日志为准。
