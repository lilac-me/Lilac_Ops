---
name: bridge-feature-validation
description: This skill should be used when the user asks to validate the basic functional features of a MindSpeed-Bridge model migration, such as fixed-length and packed data, recomputation, and TP/PP/EP/CP parallel splits, generate scenario validation scripts and launch them on a remote NPU environment with bridge-remote-train-run, or produce a single-node full-layer performance script and training log.
license: MIT
targets-repo:
  - mindspeed-bridge
---

# MindSpeed-Bridge 基础特性验证

本 skill 用于确保目标模型支持以下基本功能特性：

1. 数据处理：

  - 支持定长
  - 支持pack

1. 支持重计算
2. 并行切分：

  - 支持TP切分
  - 支持PP切分
  - 支持EP切分
  - 支持CP切分

并提供单机全层的性能脚本和训练日志。

## 输入与输出

输入：

- 目标开源模型
- 目标开源模型的MindSpeed-Bridge实现
- 可用的远端NPU环境

输出：

- 各个功能特性的验证脚本和验证结果
- 单机全层的性能脚本和2000步验证结果

## 工作区

在指定的位置建立`feature-validation`工作区，默认在任务工作目录，
它是整个流程的单一事实来源，用于固化特性验证的中间产物和迭代过程，保证可追踪、可复现。工作区结构如下：

```text
feature-validation/
├── inputs/               # 输入：开源模型名称、config.json、MindSpeed-Bridge实现
├── data-validation/      # 定长/pack 数据验证
│    ├── train.sh         # 验证脚本
│    ├── train.log        # 训练日志
│    └── status.md        # 特性开发及验证结果
├── recompute-validation/ # 结构同data-validation
├── <xxx>-validation/     # 结构同data-validation
├── performance/          # 单机全层以小验大
│    ├── train.sh         # 单机全层的性能脚本
│    └── train.log        # 性能脚本的训练日志
└── summary.md            # 总结，模板见 `references/summary-template.md`
```

如果是由`$bridge-migration-workflow`创建的工作区，`feature-validation`已包含在`$bridge-migration-workflow`的工作区目录中。且不再单独维护`inputs/`，而是使用公共的`migration_process/inputs/`。

## 核心约定

- 必须使用`$bridge-remote-train-run`拉起脚本
- 除性能脚本外，其余脚本GBS在约束条件下尽可能小，减少验证成本
- 优先保证功能跑通，精度问题无明确要求仅作观测，在结果中标注
- 性能脚本必须开启全层，单机跑不下时可以减少MoE专家数，还跑不下则标注无单机全层性能脚本
- 拉起脚本前，必须确保TP/EP/PP必须满足并行度约束

## 验证范围与预期

| 特性 | 特性使能方法 | 目标要求 | 观测内容 |
| --- | --- | --- | --- |
| 数据-定长 | 默认配置 | 跑通无报错 | 无其他观测内容 |
| 数据-pack | `enable_offline_packing`为True或`enable_in_batch_packing`为True | 跑通无报错 | 无其他观测内容 |
| 重计算 | `recompute_granularity`不为null | 跑通无报错 | 预期显存降低、性能变慢、loss 精度对齐 |
| 并行-TP | TP > 1 | 跑通无报错 | 预期显存降低、性能变慢、loss 精度对齐 |
| 并行-PP | PP > 1 | 跑通无报错 | 预期显存降低、性能变慢、loss 精度对齐 |
| 并行-EP | EP > 1 | 跑通无报错 | 预期显存降低、性能不确定、loss 精度对齐 |
| CP | CP > 1 | 跑通无报错 | 无其他观测内容 |

根据不同的特性验证场景，参考 MindSpeed-Bridge 中的examples下的脚本生成验证脚本，通过`$bridge-remote-train-run`拉起训练。
以上特性并非完全冲突，优先跑通一版默认场景作为baseline，如（TP为1、PP为1、EP为1、CP为1、开启重计算、数据定长）。
每个特性实验只与baseline有一个条件不同。

显存、性能、loss精度的观测方法参考MindSpeed-Bridge st_run的比较逻辑，用 `tests/test_tools/acquire_json.py` 把日志解析为 JSON（指标：lm loss、grad norm、time info、memo info、mtp loss）。

验证结果根据工作区文件结构要求，刷新在对应的 `<xxx>-validation` 中，每次通过`$bridge-remote-train-run`拉起训练验证时，必须覆盖目录中的train.sh和收集到的train.log，status.md中必须记录每次遇到的报错与解决方法。

## 单机全层性能脚本

探索能够单机跑通的全层性能脚本，可以减MoE专家随机初始化权重。

## 总结报告

参考`references/summary-template.md`完成报告总结。
