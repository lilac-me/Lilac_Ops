---
name: mindspeed-llm-fsdp2-fused-op-integration
description: 为 MindSpeed-LLM 的 FSDP2 本地 modeling 逐个或按用户明确授权的原子组合接入并验证融合算子，包括 MoE Grouped GEMM、RMSNorm、RoPE 和 Flash Attention。适用于复用已验证实现、执行结构与语义门禁、远端训练、loss/logits 精度和性能显存 A/B，并形成可追溯报告的任务。
---

# MindSpeed-LLM FSDP2 融合算子接入

把这个 Skill 作为融合算子接入和实验的完整闭环。依靠模型分析、目标代码和不可变的已验证参考，自行判断当前模型可以接入哪些融合算子。默认一次只接入一个；只有用户明确要求时才使用不可拆分的原子组合模式。

## 边界

- 只处理 MindSpeed-LLM 的 FSDP2 本地 modeling 路径、必要的参数转换映射和紧邻注册代码；不新增 wrapper + monkey patch 主路径。
- 不处理 EP、CP、切分策略、优化器、数据集或训练超参数。
- 调用 `$mindspeed-llm-remote-validation` 完成同步、NPU 门禁和远端训练，不在本 Skill 中实现 SSH。
- 必须调用 `$mindspeed-llm-loss-accuracy-validation` 执行 loss 精度门禁；不要在本 Skill 或其脚本中重复实现 loss 精度计算。
- 使用 `scripts/compare_fusion_runs.py` 比较相邻实验的性能和显存；该脚本不读取 loss，也不提供精度结论。
- 不自动提交、丢弃或覆盖用户已有 Git 修改。
- 训练成功、精度通过和性能提升是三个独立条件；融合算子不一定值得保留。
- 测试粒度不等于接受或提交粒度。即使原子组合只产生一个结论和一个提交，每个成员仍要通过本地结构与语义 probe。

## 开始前确定

能从仓库和训练产物可靠推断的内容直接读取，只向用户询问无法安全推断的关键输入：

- MindSpeed-LLM 仓库、目标模型和真实 modeling 执行入口。
- 训练脚本、远端验证配置、初始化或权重加载方式及并行拓扑。
- `min_speedup_percent` 性能门禁。缺少性能门禁时可以试运行，但不能宣告接入成功。
- loss 精度门禁默认使用平均相对绝对误差 `1%`，由 `$mindspeed-llm-loss-accuracy-validation` 执行；基线与候选都必须是加载真实权重的训练运行（权重不完整时两侧共用同一减层 checkpoint view，证据范围记为减层局部 A/B），随机初始化训练的 loss 比较不构成精度门禁证据。
- 报告目录。优先使用用户在对话中指定的目录；否则读取 `config/report_config.json` 或用户提供的同格式配置。
- 执行模式：默认 `sequential`；只有用户明确指定不可拆分组合时使用 `atomic-combined`，并记录组合成员和禁止的自动诊断动作。

默认配置把报告写入当前 Skill 集合根目录的 `output/fused-op-report`。对话中指定的报告目录覆盖配置文件；需要固定其他目录时直接修改配置。

## 工作流

### 1. 理解当前模型

1. 检查目标仓库的 `git status`、当前分支和最近提交，保护用户已有修改。
2. 沿真实运行入口追踪模型选择、参数注入和 forward 调用。
3. 阅读 `references/successful-modeling-examples.md`，优先用其中不可变 commit 上的完整 modeling，再搜索 `models/common/fusions` 和其他有训练与精度证据的 FSDP2 实现。当前分支上的同名文件不能仅凭名称当作成功参考。
4. 不要把文件名含 `fused`、能导入或能完成 forward 当作成功证据。未经精度和训练验证的实验性 fused 文件不得作为接入参考。
5. 阅读 `references/operator-integration-guide.md`，结合目标模型结构判断可接入的融合算子及合理顺序。不要仅按固定清单机械接入。
6. 参考已有实现时必须整函数对照（含返回前的 transpose/reshape/contiguous 和布局约定），不得凭片段或印象改写。
7. 完整读取 `references/correctness-gates.md`，先建立结构签名和 forward 语义不变量，再写候选代码。
8. 对改变数值语义的算子（包括 attention、RMSNorm、RoPE、MoE/EP 路径），训练 loss 门禁不足以免除语义验证：随机初始化初期 loss 对语义错误不敏感，必须补真实权重减层 logits 对齐（复用 `$fsdp2-precision-alignment`）。

### 2. 建立基线

1. 固定训练脚本、随机种子、数据、步数、batch、序列长度、精度、并行拓扑和软件环境。
2. 确认输入 commit 是通过 eager logits 门禁的纯接入检查点；若阶段二已混入本模型专属 fused 分支、patch、参数重排或 fused conversion mapping，返回模型接入阶段修正，不以“开关关闭”放行。
3. 使用 `$mindspeed-llm-remote-validation` 运行未接入融合算子的原始代码，确保 `remote-validation-0001` 是 loss 精度基线；该基线必须加载真实权重（权重不完整时使用减层 checkpoint view）。
4. 性能/显存基线默认实例化原始完整层数；真实权重不完整时可明确使用随机初始化，但随机初始化运行只能作为性能/显存基线，不能作为 loss 精度基线。1/2 层 checkpoint view 承担真实权重 logits 验收与 loss 精度基线。若用户明确采用减层性能 A/B，必须记录证据范围且候选保持完全相同层数。
5. 后续性能比较仍使用当前保留方案对应的相邻运行作为性能基线。
6. 确认基线训练成功，日志包含实际层范围、逐 iteration 耗时、loss 和显存。

### 3. 逐个接入并验证

对每个可接入的候选算子重复：

1. 只修改当前算子，并保留受已有开关控制的原始 fallback。
2. 比较候选与 eager 检查点的结构签名。module class、参数名/shape/dtype、tied weight、state_dict key 和专家轴布局必须一致；若算子必须改变这些内容，它不是“只切换计算实现”的融合候选，必须重新建立适配基线，不能继续做相邻性能归因。
3. 按 `correctness-gates.md` 检查输入输出、调用顺序、shape、dtype、device、layout、mask、权重布局、forward、input gradient 和 parameter gradient。
4. 运行语法、导入和可执行的本地检查，确认 diff 没有混入下一个算子。
5. 调用 `$mindspeed-llm-remote-validation` 运行 candidate。
6. 阅读完整 `training.log`，检查异常、NaN/Inf、Autograd warning、OOM、编译回退和非预期 fallback；同时记录每个开关的 requested/active 证据，不能用 YAML 值代替真实执行证据。
7. 使用 `scripts/compare_fusion_runs.py` 比较当前性能 baseline 和 candidate，只执行性能门禁。
8. 调用 `$mindspeed-llm-loss-accuracy-validation`，基线使用真实权重 loss 基线运行（权重不完整时为减层 view 基线），显式传入其运行名称或目录；以当前 candidate（同样加载真实权重、相同层数配置）为候选运行，执行 `1%` 平均相对绝对误差门禁。
9. 执行真实权重 1/2 层完整词表 logits 对齐；权重不足以覆盖减层所需分片时标记 blocked，不用随机初始化 loss 替代。
10. 分别读取性能、显存、loss 和 logits 报告；不得互相推导结论。
11. 全部门禁通过时保留本算子，并把 candidate 作为下一算子的性能 baseline。
12. 任一门禁未通过时记录失败结果，不把该 candidate 作为下一步 baseline。没有用户明确授权时，不自动丢弃实验代码。
13. 立即把本步骤追加到统一 Markdown 报告，不等所有算子完成后再写。

无法证明关键语义、远端训练失败或依赖超出本 Skill 边界时，把该步骤记为 `blocked`，并继续分析不依赖它的其他候选。

### 4. 原子组合模式

用户明确要求 `atomic-combined` 时：

1. 一次性实现指定成员，不进行逐算子训练、性能比较、接受判断或提交。
2. 对每个成员分别执行本地结构与语义 probe，并在一个组合审计中记录；不得因此声称测得单算子贡献。
3. 训练、loss、性能和显存只与融合全关基线比较，其中 loss 基线必须是真实权重训练运行；真实执行记录必须证明所有成员 active。
4. 对组合执行真实权重 1/2 层完整词表 logits 对齐。
5. 任一成员或整体门禁失败即拒绝整个组合，保存完整 diff 和所有报告；除非用户另行授权，不自动切换逐算子测试或二分。
6. 全部通过后才向主控返回一个可提交的组合检查点。

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
python3 skills/fsdp2-model-lifecycle/supporting-skills/loss-accuracy-validation/scripts/compare_loss.py \
  --output-dir /path/to/output \
  --baseline-run /path/to/real-weight-baseline-run \
  --candidate-run remote-validation-NNNN \
  --max-relative-error-percent 1
```

loss 精度门禁的基线与候选都必须是加载真实权重的训练运行（权重不完整时两侧共用同一减层 checkpoint view）；随机初始化训练的 loss 比较不构成精度门禁证据。基线运行与候选不共用一个输出根目录时，按目录路径显式传入 `--baseline-run`。不要在本 Skill 中另算 loss 差异。性能门禁值必须来自用户或上游任务；示例数值不是默认建议。短训练结果接近噪声范围时应重复运行，不通过挑选最快一次接受算子。

## 统一 Markdown 报告

报告文件固定命名为 `fused_op_integration_report.md`，写入指定报告目录。使用 `scripts/update_markdown_report.py` 创建或追加行，不手工改变列名或顺序。

未在对话中指定报告目录时，默认写入当前 Skill 集合根目录的 `output/fused-op-report`。它与远端验证产物共用同一个 `output/` 根目录，默认布局为：

```text
fsdp2-model-lifecycle/
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
  --modified-file mindspeed_llm/fsdp2/models/qwen3/modeling_qwen3_moe.py \
  --performance-report /path/to/remote-validation-NNNN/fused_op_performance_validation.json \
  --accuracy-report /path/to/remote-validation-NNNN/loss_accuracy_validation.json \
  --correctness-report /path/to/candidate/fusion_correctness_validation.json
```

也可以用 `--config /path/to/report_config.json` 提供报告目录；`--report-dir` 优先。

统一表格列为：

| 步骤 | 模型 | 融合算子 | 修改说明 | 修改文件 | 运行对比 | 单步耗时（基线 → 候选） | 性能变化 | Loss 精度 | 结构/语义/Logits | 结论 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

每个尝试都写一行，包括性能回退、精度失败和 blocked。修改说明要简洁说明做了什么；修改文件必须给出具体路径。Loss 列读取独立 loss 报告；正确性列读取符合 `references/fusion_correctness_gate_schema.json` 的独立报告，包含 requested/active、结构签名、语义 probe 和真实权重 logits 门禁。

## 结束条件

当模型分析得到的可接入候选都已经验证或明确 blocked 后结束，输出：

- 最终保留的融合算子。
- 未保留或 blocked 的算子及原因。
- 当前最佳远端运行目录。
- 统一 Markdown 报告的绝对路径。
- 尚未解决的 warning 和兼容性风险。

## 特别警戒

- Flash Attention 的 `attention_mask=None` 不代表无需 causal 语义。默认只有 NPU、纯 causal、无 padding、无 sliding-window 且 kernel 的 layout/GQA/mask 合同均满足时才启用，否则必须走 eager fallback；只有目标 kernel 对额外语义已有明确实现和专项 probe 证据时才能扩大支持范围。确认 mask 极性、shape、长度和 `sparse_mode`。
- Grouped GEMM 必须匹配 token 分组、三维专家权重和 checkpoint 转换语义；融合开关不得替换 module class 或把既有三维专家参数改成另一种布局。shape 正确不等于权重语义正确。
- NPU GMM 出现缺少 Autograd key 的 warning 时，必须确认 backward 和梯度正确性，不能仅凭 loss 正常下降接受。
- RMSNorm 必须保留目标模型的原生 dtype 策略；对 Qwen3-MoE，要保留 FP32 归一化/权重计算再回到输入 dtype，直接把 BF16 输入和权重交给 fused kernel 不视为已证明等价。
- Attention 必须保留 Q/K projection 后的 `q_norm`/`k_norm` 与 RoPE 顺序；融合 forward 不能因为手写 QKV 路径跳过它们。
- RoPE 也会产生小数值差异，仍需执行精度门禁。
