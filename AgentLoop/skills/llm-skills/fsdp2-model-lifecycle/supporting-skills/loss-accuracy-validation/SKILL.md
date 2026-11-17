---
name: mindspeed-llm-loss-accuracy-validation
description: 以 MindSpeed-LLM 首次远端训练 remote-validation-0001 的逐 iteration lm loss 为默认精度基线，也支持显式指定其他基线运行目录；将训练步数相同或不同的两份 loss 日志按共同 iteration 区间严格对齐，默认要求平均相对绝对误差不超过 1%，并生成结构化百分比精度门禁报告。适用于代码或训练实现修改后，需要按 abs((baseline-candidate)/baseline) 验证 loss 相对偏差、决定自动调优工作流是否允许结束的场景。
---

# MindSpeed-LLM Loss 精度门禁

将这个 Skill 作为远端训练后的本地精度一致性检查器。只读取远端验证产物、比较 loss 并生成报告；不要修改训练代码、配置、日志或远端验证报告。

## 核心约定

- 作为精度门禁使用时，调用方必须保证基线与候选都是加载真实权重的训练运行（权重不完整时两侧共用同一减层 checkpoint view，并在报告中声明减层局部 A/B 证据范围）；随机初始化训练的 loss 比较不构成精度门禁证据。本工具只执行比较，不验证权重来源，权重来源由主控流程负责。
- 未提供 `--baseline-run` 时使用 `remote-validation-0001` 作为默认首次训练基线。
- 显式提供 `--baseline-run` 时，接受 `remote-validation-NNNN` 运行名称或该运行的目录路径。运行名称相对 `--output-dir` 解析，目录路径直接解析。
- 要求基线和候选目录都包含成功完成的 `validation_report.json` 与完整 `training.log`。
- 按 iteration 编号严格对齐 `lm loss`；允许两侧总训练步数不同，并比较 `1..min(baseline_total, candidate_total)` 共同训练区间。
- 要求共同训练区间内两侧的每个 iteration 都存在，且训练脚本路径一致；较长日志超出共同区间的尾部 iteration 不参与计算。
- 对每个 iteration 计算 `relative_error = abs((baseline_loss - candidate_loss) / baseline_loss)`，再计算全部对齐 iteration 的平均相对绝对误差。
- 默认门限为平均相对绝对误差小于或等于 `1%`。
- 命令行门限使用百分比。用户或总控 Skill 可以显式传入 `--max-relative-error-percent` 覆盖默认值，例如 `0.5` 表示门限为 `0.5%`。
- baseline loss 为 `0` 时不添加隐式 epsilon，直接将比较标记为 `blocked`。
- 将报告写入候选运行目录，不修改基线目录。
- 只将门禁通过解释为当前精度验证阶段 `success`。总控 Skill 还需检查其他 required gates，才能结束完整工作流。

## 工作流

1. 确认本地输出根目录和候选 `remote-validation-<四位数字>` 目录。
2. 解析基线：优先使用显式 `--baseline-run`，否则使用 `remote-validation-0001`；确认候选目录与基线目录不同。
3. 确认两次训练均成功完成且训练脚本路径一致。
4. 从完整日志提取每个 iteration 的 `lm loss`。
5. 以两侧总训练步数的较小值确定共同训练区间，严格检查该区间内的 iteration 完整性。
6. 仅在共同训练区间计算逐 iteration 相对绝对误差、平均相对绝对误差和最大相对绝对误差，并记录两侧总步数与实际比较范围。
7. 将 `loss_accuracy_validation.json` 写入候选目录，并把结果返回总控流程。

## 脚本入口

```bash
python3 scripts/compare_loss.py \
  --candidate-run remote-validation-0002
```

上述命令使用默认 `1%` 门限。用户或总控 Skill 覆盖门限时显式传入，例如：

```bash
python3 scripts/compare_loss.py \
  --candidate-run remote-validation-0002 \
  --max-relative-error-percent 0.5
```

显式指定基线目录：

```bash
python3 scripts/compare_loss.py \
  --baseline-run /path/to/output/remote-validation-0010 \
  --candidate-run remote-validation-0014 \
  --output-dir /path/to/output
```

默认从当前 Skill 集合根目录的 `output/` 读取；使用 `--output-dir` 指定远端验证 Skill 的自定义输出根目录。`--baseline-run remote-validation-0010` 也会解析为该输出根目录下的同名运行。

## 结果语义

- `success`：两侧数据完整且平均相对绝对误差百分比不超过阈值，`gate_passed: true`。
- `failed`：比较已完成但平均相对绝对误差百分比超过阈值，`gate_passed: false`。
- `blocked`：训练未成功、文件缺失、日志无法解析、训练脚本不一致或共同训练区间内缺失 iteration，不能形成合法比较。

退出码：`0` 表示门禁通过；`2` 表示参数非法；`3` 表示比较被阻塞；`4` 表示平均相对绝对误差门禁失败。

## 文件职责

- 使用 `scripts/compare_loss.py` 确定性提取 loss、计算平均相对绝对误差并生成报告。
- 使用 `references/loss_accuracy_report_schema.json` 定义门禁报告的输出接口合同，供总控 Skill 稳定解析。
