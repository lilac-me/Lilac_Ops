---
name: bridge-migration-workflow
description: This skill should be used when the user asks to migrate an open-source Hugging Face model to MindSpeed-Bridge or bring up a new model end to end, orchestrate the staged migration workflow by sequentially calling the bridge-model-adaptation, bridge-precision-alignment, and bridge-feature-validation skills, maintain the migration_process workspace with inputs, stage status, and deliverables, or resume or re-initialize an existing migration task.
license: MIT
targets-repo:
  - mindspeed-bridge
---

# MindSpeed-Bridge 模型迁移工作流

把这个 Skill 作为 MindSpeed-Bridge 模型开箱/迁移的统一入口：编排整个模型迁移工作流，只管理公共信息、阶段状态与总结报告；代码适配、精度对齐、基础功能验证等具体工作交给对应 skill 闭环完成。

## Skill 编排关系

```text
                       bridge-migration-workflow
        ┌───────────────────────┼─────────────────────────┐
        ▼                       ▼                         ▼
bridge-model-adaptation ────> bridge-precision-alignment ────> bridge-feature-validation
        │                       │                         │
        └───────────────────────┴─────────────────────────┘
                                │
                        bridge-remote-train-run
```

`$bridge-migration-workflow` 作为编排入口，按工作流顺序依次调用 `$bridge-model-adaptation`、`$bridge-precision-alignment`、`$bridge-feature-validation` 三个阶段 skill，前一阶段完成后才进入下一阶段，逐阶段闭环后汇总产物，输出端到端验证报告。
`$bridge-remote-train-run` 为上述三个阶段 skill 提供在远程 NPU 环境上拉起脚本的能力，只回传执行结果，不参与问题定位。

**各Skill职责**：

| skill | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `$bridge-migration-workflow` | 编排整个迁移工作流，收集输入、阶段状态、输出报告 | 目标模型、开源权重、可用NPU环境 | 迁移报告 |
| `$bridge-model-adaptation` | 模型迁移代码适配 | 目标模型，开源权重，可用NPU环境 | 适配后的模型代码，正常训练日志及可复现步骤 |
| `$bridge-precision-alignment` | 与transformer HF模型完成精度对齐 | 可正常拉起训练的模型代码 | 精度对齐结果及可复现步骤 |
| `$bridge-feature-validation` | 验证TP/PP/EP等切分特性，重计算、数据pack等基础功能，输出全层性能脚本 | 精度对齐的模型代码 | 特性验证结果及可复现步骤 |
| `$bridge-remote-train-run` | 连接远程NPU环境，拉起训练 | 可用NPU环境，训练脚本 | 训练日志，训练结果 |

**核心约定**：

- `$bridge-migration-workflow` 始终拥有整体流程控制权，但不介入其他 skill 的内部流程，也不代替其他 skill 完成代码适配、精度对齐、问题修复等工作。
- `$bridge-migration-workflow`不直接使用`$bridge-remote-train-run`拉起训练代替其他skill的内部验证。
- `$bridge-remote-train-run`为`$bridge-model-adaptation`、`$bridge-precision-alignment`、`$bridge-feature-validation`只提供远程训练能力，只把错误或者成功信息收集给所调用的skill，不参与定位。
- 每个skill是一个自验证阶段，进入后按其`SKILL.md`完成内部闭环，结束后才返回`$bridge-migration-workflow`。
- `$migration_process`信任各个skill的验证结果，只核实验证结果是否真实存在，而不是重复拉起训练验证其结果。

## 建立工作区

每个迁移任务必须维护一个 `migration_process/<model>` 文件夹用作整个流程的工作区（默认在任务工作目录）。
它是整个流程的单一事实来源，保证进度固化、可追踪，会话中断或上下文清空后可按合同恢复，不从头开始：

```text
migration_process/<model>/
├── status.md            # 刷新当前任务进展，阶段状态，文档模板见`references/status-template.md`.
├── report.md            # 输出模型开箱或模型迁移报告，文档模板见`references/report-template.md`.
├── inputs/              # 输入：任务目标，目标模型，config.json，可用NPU环境
├── result/              # 转测交付：精度对齐结论与原始日志precision/、功能验证结论与原始日志logs/
├── model-adaptation/    # model-adaptation的工作区，由`$bridge-model-adaptation`管理
├── precision-alignment/ # precision-alignment的工作区，由`$bridge-precision-alignment`管理
├── feature-validation/  # feature-validation的工作区，由`$bridge-feature-validation`管理
└── remote-train-run/    # remote-train-run的工作区，由`$bridge-remote-train-run`管理
```

若migration_process已存在，必须向用户确认是重新初始化迁移任务，还是根据已存在的migration_process进度继续迁移任务。
**初始化**：
  根据上述文件结构创建migration_process工作区，并在status.md中标记当前任务进展为“初始化”。
  若已存在migration_process，整个文件夹删除后重新创建。

**继续迁移任务**：
  检测migration_process中status.md的当前任务进展，按当前任务进展继续迁移任务。
  根据当前进展校验已完成任务的完整性，若不完整，则提示用户重新初始化迁移任务。
    - 输入收集完整
    - 模型适配完整
    - 精度对齐完整
    - 基础功能验证完整
  若status.md不存在，或者status.md中当前任务进展为“初始化”，则认为是重新初始化迁移任务。

## 工作流

### 1. 收集输入

必须向用户收集需要迁移的目标模型名称。
必须收集目标模型的开源`config.json`。
必须收集目标模型的开源实现。
必须收集可用的NPU环境，且环境能通过ssh免密登录。
必须确认NPU环境上有无开源权重路径。
必须确认NPU环境上有无可用的数据集。
必须确认本地代码路径和NPU环境上的代码路径，用于同步代码和拉起训练。

上述信息除了模型的开源实现和config.json可以从网络上获取，其他信息都必须由用户提供，不要自行推导。
所有输入信息放到工作区inputs目录下，如果有权重路径和数据集路径确保NPU环境上存在，不要下载到inputs。
收集完成后刷新status.md，标记当前任务进展为“已收集输入”，并在status.md中记录所有收集到的输入信息。

### 2. 阶段一：模型适配

使用子Agent或者子会话调用`$bridge-model-adaptation` SKILL 完成模型代码适配，向`$bridge-model-adaptation`提供工作区和所有的输入。
子Agent完成模型代码适配后，返回主Agent，主Agent刷新status.md，标记当前任务进展。

模型适配完成的判定条件：
  `$bridge-model-adaptation`的工作区内存在`summary.md`，且`summary.md`中记录了模型适配的方案、可复现步骤和结果，结果核实有效。
否则认为`$bridge-model-adaptation`适配失败，终止模型迁移流程。

确认`$bridge-model-adaptation`完成后，刷新status.md，释放创建的子Agent或者子会话。

### 3. 阶段二：精度对齐

使用子Agent或者子会话调用`$bridge-precision-alignment` SKILL 完成精度对齐，向`$bridge-precision-alignment`提供工作区和所有输入。
子Agent完成精度对齐后，返回主Agent，主Agent刷新status.md，标记当前任务进展。

精度对齐完成的判定条件:
  `$bridge-precision-alignment`的工作区内存在`summary.md`，且`summary.md`中记录了精度对齐遇到的问题和解决方法，精度对齐的可复现步骤和结果，结果核实有效。
否则认为`$bridge-precision-alignment`精度对齐失败，终止模型迁移流程。

确认`$bridge-precision-alignment`精度对齐任务完成后，刷新status.md，释放相应的子Agent或者子会话。

### 4. 阶段三：基础功能验证

使用子Agent或者子会话调用`$bridge-feature-validation`完成模型基础功能的开发与验证，向`$bridge-feature-validation`提供工作区和所有的输入。
子Agent完成模型基础功能开发和验证后，返回主Agent，主Agent刷新status.md，标记当前任务进展。

基础功能开发和验证完成的判定调教：
  `$bridge-feature-validation`的工作区内存在`summary.md`，且`summary.md`中记录了存在的问题和解决方法，以及基础功能验证的可复现步骤和结果，结果核实有效。
否则认为`$bridge-feature-validation`基础功能验证失败，终止模型迁移流程。

确认`$bridge-feature-validation`完成基础功能开发和验证后，刷新status.md，释放创建的子Agent或者子会话。

### 5. 汇总报告

将各阶段最终迭代产物拷贝（不移动）到 `migration_process/result/`，按照文档模板`references/report-template.md`输出最终的模型开箱或模型迁移报告。
