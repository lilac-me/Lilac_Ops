---
name: bridge-precision-alignment
description: This skill should be used when the user asks to verify or debug numerical precision alignment between a Hugging Face Transformers model and its MindSpeed-Bridge implementation, generate and run the compare.sh script with the compare_hf_and_megatron tool, judge alignment by next-token match and cosine similarity thresholds, or diagnose precision mismatches using debug hooks.
license: MIT
targets-repo:
  - mindspeed-bridge
---

# MindSpeed-Bridge 精度对齐

本 skill 基于MindSpeed-Bridge的 compare 工具完成目标模型与Transformer模型的精度对齐。
compare工具提供了验证和Debug能力。

compare工具路径：MindSpeed-Bridge/mindspeed_bridge/examples/conversion/compare_hf_and_megatron

## 输入与输出

输入：

- 需要完成精度对齐的模型名称
- 模型的开源HF权重（可选）
- 模型的开源实现和MindSpeed-Bridge实现
- 可用 NPU 环境（用于 `$bridge-remote-train-run` 拉起脚本）。

输出：

- compare.sh compare脚本
- compare 工具的验证结果

## 工作区

在指定的位置建立`precision-alignment`工作区，默认在任务工作目录，
它是整个流程的单一事实来源，用于固化精度对齐的中间产物和迭代过程，保证可追踪、可复现。工作区结构如下：

```text
precision-alignment/
├── inputs/              # 输入：开源模型名称、开源的模型权重路径
├── iter-NNN/            # 每次迭代的工作区，NNN为迭代号
│    ├── *.log           # 迭代中返回的训练日志
│    ├── hf_debug_*.jsonl         # hook返回的hf模型各模块输入、输出和权重
│    ├── megatron_debug_*.jsonl   # hook返回的megatron模型各模块输入、输出和权重
│    └── analysis.md     # 分析
├── compare.sh           # 用于精度对齐的脚本
└── summary.md           # 总结，包含精度验证结果，存在的精度问题和解决方法, 模板见 `references/summary-template.md`。
```

如果是由`$bridge-migration-workflow`创建的工作区，`precision-alignment`已包含在`$bridge-migration-workflow`的工作区目录中。且不再单独维护`inputs/`，而是使用公共的`migration_process/inputs/`。

- 必须使用`$bridge-remote-train-run`拉起`compare.sh`脚本
- 每次使用`$bridge-remote-train-run`拉起脚本视为一次迭代，迭代号从`001`严格递增，不允许覆盖已有迭代，收集拉起的脚本和返回的训练日志到`iter-NNN/`中

## 精度对齐验证

阅读 MindSpeed-Bridge/mindspeed_bridge/examples/conversion/compare_hf_and_megatron/README.md 了解compare工具的使用方法，编写compare.sh脚本。

- 必须使用最小验证，不开启TP/PP/CP切分，使用最小卡数对齐精度(--nproc-per-node)
- 单卡MoE专家数不能超过128，比如当前模型专家总数为256，则至少需要2卡，EP为2
- 修改`config.json`减层加载HF模型对齐精度，精度对齐后恢复`config.json`
- 不加载megatron权重，只加载同一个hf权重，无hf权重时使用`--roundtrip_hf`选项导出hf权重对比

通过`$bridge-remote-train-run`拉起`compare.sh`脚本，收集返回的日志放到`iter-NNN/`中。

- 每次使用`$bridge-remote-train-run`拉起脚本视为一次迭代，迭代号从`001`严格递增，不允许覆盖已有迭代。

日志中观察到`=== COMPARISON COMPLETE ===`表明compare执行完成，以下条件全部满足认为精度对齐

- `Next Token Match:` 后为True, 表示Next Token匹配
- `Output_Cosine_Similarity:` 后`Max`、`Min`、`Mean`列的余弦相似度结果必须全部大于0.98
- `Cosine similarity:` 后的结果必须大于98%

以上条件有一个不满足，都认为精度没有对齐，需要进入精度问题的定位流程，直至上述条件均满足。

## 精度问题定位

compare 工具开启`--enable_debug_hooks`，输出迭代一步中所有前反向过程中module的输入、输出、权重，用于定位。

- 必须使用`$bridge-remote-train-run`拉起`compare.sh`脚本，拉起脚本前确保已清除上一次生成的hf_debug_*.jsonl和megatron_debug_*.jsonl文件。
- 拉起脚本后，需要收集远端的hf_debug_*.jsonl和megatron_debug_*.jsonl到`iter-NNN`中用于分析定位
- 多卡场景会生成多个rank的.jsonl，只需要对比一个相同rank的jsonl

结合模型实现代码，按照网络顺序对比第一个出现明显精度差异的module，分析对比transformer实现差异。
将分析结果刷新至`iter-NNN/analysis.md`中，需要包含精度差异、分析存在的精度问题，解决方法。

修复精度问题后继续验证精度，直至精度对齐。

## 总结

完成精度对齐后，参考 references/summary-template.md 完成精度对齐报告。
