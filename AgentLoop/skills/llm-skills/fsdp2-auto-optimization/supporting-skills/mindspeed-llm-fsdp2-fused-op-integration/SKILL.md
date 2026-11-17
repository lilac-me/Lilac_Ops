---
name: mindspeed-llm-fsdp2-fused-op-integration
description: 为 MindSpeed-LLM 的 FSDP2 模型逐个分析、接入和验证可用融合算子，包括 MoE Grouped GEMM、RMSNorm、RoPE 和 Flash Attention。适用于修改原始 modeling 文件或 Transformers 补丁实现、复用仓库已有融合实现、调用远端训练、用独立 loss 精度 Skill 执行精度门禁、比较相邻实验性能，并把每一步结果写入统一 Markdown 报告的任务。
targets-repo:
  - mindspeed-llm
---

# MindSpeed-LLM FSDP2 融合算子接入

把这个 Skill 作为融合算子接入和逐项实验的完整闭环。依靠模型分析目标代码和仓库参考实现，自行判断当前模型可以接入哪些融合算子；一次只接入一个，验证完成后再处理下一个。

## 边界

- 只处理 MindSpeed-LLM 的 FSDP2 模型路径、必要的参数转换映射和紧邻注册代码。
- 不处理 EP、CP、切分策略、优化器、数据集或训练超参数。
- 调用 `$mindspeed-llm-remote-validation` 完成同步、NPU 门禁和远端训练，不在本 Skill 中实现 SSH。
- 必须调用 `$mindspeed-llm-loss-accuracy-validation` 执行 loss 精度门禁；不要在本 Skill 或其脚本中重复实现 loss 精度计算。
- 使用 `scripts/compare_fusion_runs.py` 比较相邻实验的性能和显存；该脚本不读取 loss，也不提供精度结论。
- 不自动提交、丢弃或覆盖用户已有 Git 修改。
- 训练成功、精度通过和性能提升是三个独立条件；融合算子不一定值得保留。

## 开始前确定

能从仓库和训练产物可靠推断的内容直接读取，只向用户询问无法安全推断的关键输入：

- MindSpeed-LLM 仓库、目标模型和真实 modeling 执行入口。
- 训练脚本、远端验证配置、初始化或权重加载方式及并行拓扑。
- `min_speedup_percent` 性能门禁。缺少性能门禁时可以试运行，但不能宣告接入成功。
- loss 精度门禁默认使用平均相对绝对误差 `1%`，由 `$mindspeed-llm-loss-accuracy-validation` 执行。
- 报告目录。优先使用用户在对话中指定的目录；否则读取 `config/report_config.json` 或用户提供的同格式配置。

默认配置把报告写入当前 Skill 集合根目录的 `output/fused-op-report`。对话中指定的报告目录覆盖配置文件；需要固定其他目录时直接修改配置。

## 工作流

### 1. 理解当前模型

1. 检查目标仓库的 `git status`、当前分支和最近提交，保护用户已有修改。
2. 沿真实运行入口追踪模型选择、参数注入和 forward 调用。
3. 阅读 `references/successful-modeling-examples.md`，优先参考其中明确列出的已成功接入 modeling 文件，再搜索 Transformers patch、`models/common/fusions` 和其他已验证的 FSDP2 实现。
4. 不要把文件名含 `fused`、能导入或能完成 forward 当作成功证据。未经精度和训练验证的实验性 fused 文件不得作为接入参考。
5. 阅读 `references/operator-integration-guide.md`，结合目标模型结构判断可接入的融合算子及合理顺序。不要仅按固定清单机械接入。

### 2. 建立基线

1. 固定训练脚本、随机种子、数据、步数、batch、序列长度、精度、并行拓扑和软件环境。
2. 使用 `$mindspeed-llm-remote-validation` 运行未接入融合算子的原始代码，确保 `remote-validation-0001` 是 loss 精度基线。
3. 后续性能比较仍使用当前保留方案对应的相邻运行作为性能基线。
4. 确认基线训练成功，日志包含逐 iteration 的耗时、loss 和显存。

### 3. 逐个接入并验证

对每个可接入的候选算子重复：

1. 只修改当前算子，并保留受已有开关控制的原始 fallback。
2. 检查输入输出、shape、dtype、device、layout、mask、权重布局和 forward/backward。
3. 运行语法、导入和可执行的本地检查，确认 diff 没有混入下一个算子。
4. 调用 `$mindspeed-llm-remote-validation` 运行 candidate。
5. 阅读完整 `training.log`，检查异常、NaN/Inf、Autograd warning、OOM、编译回退和非预期 fallback；确认 fused 分支确实执行。
6. 使用 `scripts/compare_fusion_runs.py` 比较当前性能 baseline 和 candidate，只执行性能门禁。
7. 调用 `$mindspeed-llm-loss-accuracy-validation`，默认省略 `--baseline-run` 以使用 `remote-validation-0001`；需要其他基线时显式传入其运行名称或目录。以当前 candidate 为候选运行，执行 `1%` 平均相对绝对误差门禁。
8. 分别读取 `fused_op_performance_validation.json` 和 `loss_accuracy_validation.json`；不得从性能报告推导精度结论。
9. 精度和性能门禁都通过时保留本算子，并把 candidate 作为下一算子的性能 baseline。
10. 精度或性能未通过时记录失败结果，不把该 candidate 作为下一步 baseline。没有用户明确授权时，不自动丢弃实验代码。
11. 立即把本步骤追加到统一 Markdown 报告，不等所有算子完成后再写。

无法证明关键语义、远端训练失败或依赖超出本 Skill 边界时，把该步骤记为 `blocked`，并继续分析不依赖它的其他候选。

## 性能和精度验证

先比较相邻实验的性能和显存：

```bash
python3 scripts/compare_fusion_runs.py \
  --baseline-run /path/to/remote-validation-NNNN \
  --candidate-run /path/to/remote-validation-NNNN \
  --operator fused_rotary_pos_emb \
  --warmup-iterations 1 \
  --min-speedup-percent 0
```

再调用 `$mindspeed-llm-loss-accuracy-validation` 执行精度门禁：

```bash
python3 skills/llm-skills/fsdp2-auto-optimization/supporting-skills/mindspeed-llm-loss-accuracy-validation/scripts/compare_loss.py \
  --output-dir /path/to/output \
  --candidate-run remote-validation-NNNN \
  --max-relative-error-percent 1
```

省略 `--baseline-run` 时精度 Skill 默认使用 `remote-validation-0001`。需要覆盖时增加 `--baseline-run /path/to/remote-validation-NNNN`，也可传入相对 `--output-dir` 的运行名称。不要在本 Skill 中另算 loss 差异。性能门禁值必须来自用户或上游任务；示例数值不是默认建议。短训练结果接近噪声范围时应重复运行，不通过挑选最快一次接受算子。

## 统一 Markdown 报告

报告文件固定命名为 `fused_op_integration_report.md`，写入指定报告目录。使用 `scripts/update_markdown_report.py` 创建或追加行，不手工改变列名或顺序。

未在对话中指定报告目录时，默认写入当前 Skill 集合根目录的 `output/fused-op-report`。它与远端验证产物共用同一个 `output/` 根目录，默认布局为：

```text
fsdp2-auto-optimization/
└── output/
    ├── remote-validation-0001/
    ├── remote-validation-0002/
    └── fused-op-report/
        └── fused_op_integration_report.md
```

默认路径由 `config/report_config.json` 配置；需要改为固定的其他目录时修改该配置，对话中显式提供的目录仍然优先。

```bash
python3 scripts/update_markdown_report.py \
  --report-dir /path/to/report-dir \
  --model qwen3_moe \
  --step 1 \
  --operator fused_rotary_pos_emb \
  --change-description "增加 fused RoPE 分支并保留原实现 fallback" \
  --modified-file mindspeed_llm/fsdp2/models/qwen3/qwen3_moe.py \
  --performance-report /path/to/remote-validation-NNNN/fused_op_performance_validation.json \
  --accuracy-report /path/to/remote-validation-NNNN/loss_accuracy_validation.json
```

也可以用 `--config /path/to/report_config.json` 提供报告目录；`--report-dir` 优先。

统一表格列为：

| 步骤 | 模型 | 融合算子 | 修改说明 | 修改文件 | 运行对比 | 单步耗时（基线 → 候选） | 性能变化 | 精度验证 | 结论 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |

每个尝试都写一行，包括性能回退、精度失败和 blocked。修改说明要简洁说明做了什么；修改文件必须给出具体路径。精度列读取独立精度报告，至少包含平均相对绝对误差百分比、最大相对绝对误差百分比、门限和通过/失败结论。

## 结束条件

当模型分析得到的可接入候选都已经验证或明确 blocked 后结束，输出：

- 最终保留的融合算子。
- 未保留或 blocked 的算子及原因。
- 当前最佳远端运行目录。
- 统一 Markdown 报告的绝对路径。
- 尚未解决的 warning 和兼容性风险。

## 特别警戒

- Flash Attention 的 `attention_mask=None` 不代表无需 causal 语义。确认 mask 极性、shape、长度和 `sparse_mode`。
- Grouped GEMM 必须匹配 token 分组、三维专家权重和 checkpoint 转换语义；shape 正确不等于权重语义正确。
- NPU GMM 出现缺少 Autograd key 的 warning 时，必须确认 backward 和梯度正确性，不能仅凭 loss 正常下降接受。
- RMSNorm 和 RoPE 也会产生小数值差异，仍需执行精度门禁。
