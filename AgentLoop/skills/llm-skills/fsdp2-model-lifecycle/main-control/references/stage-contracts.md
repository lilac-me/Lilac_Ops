# 阶段结果合同

每个阶段向主控返回并写入 manifest：

- `stage`：稳定的阶段标识。
- `status`：`success`、`no_improvement`、`failed` 或 `blocked`。
- `input_commit`：阶段开始时的完整 Git SHA。
- `output_commit`：通过门禁后由主控创建的完整 Git SHA；未提交时为 `null`。
- `modified_files`：本阶段实际修改的仓库相对路径。
- `gates`：必需门禁名称及 `passed`、`failed`、`blocked` 状态。
- `artifacts`：报告、日志、JSON、差异清单和其他阶段产物的绝对路径。
- `summary`：简短结论，不复制完整日志。
- `error`：失败或阻塞原因；成功时为 `null`。

## 必需阶段

| 阶段 | 必需门禁 |
| --- | --- |
| `pre_adaptation_analysis` | 原始源码版本、输入假设、`model_analysis.md`、`model_analysis.json`、NPU 缺口与并行建议 |
| `model_adaptation` | 分析产物版本一致、纯 eager 边界审计、静态检查、注册、并行路径、FLOPS、启动配置、差异清单、结构签名、forward 语义不变量矩阵 |
| `eager_precision` | 接入差异已处置、融合全关、原生 Reference、真实 FSDP2 Candidate、真实权重完整加载、减层（默认 2 层）完整词表 cosine |
| `training_baseline` | preflight、NPU 空闲、训练层范围证据、forward、backward、optimizer、有限 loss；完整架构或 reduced-layer 证据范围明确 |
| `fused_operator` | 修改范围、requested/active、结构签名一致、成员语义 probe、训练成功、forward/backward/optimizer、性能、显存、loss、真实权重减层 logits |
| `final_precision` | 干净工作树、确定 Git commit、最终执行标签、requested/active、真实 FSDP2、真实权重减层（默认 2 层）完整词表 cosine |

主控不得根据自然语言推测门禁通过。缺少必需字段或 artifact 时，将阶段标记为 `blocked`。

## 融合阶段模式

- 对 `full` 和 `optimize-only` 新运行，主控必须在对话中取得用户对 `fusion_validation_strategy` 的明确选择并写入 manifest；不得从候选列表或提交要求推断。`adapt-only` 记录 `not-applicable`；旧 manifest 缺失该字段时，`resume` 必须在进入融合阶段前询问。
- `sequential`：每个 `fused_operator` 记录只包含一个候选，性能与上一个已接受运行比较，loss 默认与融合全关基线比较。
- `atomic-combined`：一个 `fused_operator` 记录包含全部成员；只与融合全关基线比较，只能整体接受或拒绝。仍须为每个成员提供本地结构/语义 probe，但不得报告单算子性能贡献。

## 训练证据范围

- 完整架构性能/loss：日志证明实际实例化并训练配置中的全部层；权重不完整时允许明确记录的随机初始化。
- 减层真实权重 logits：默认 1 层 smoke、2 层验收，保留 checkpoint 中完整专家；只能证明被覆盖层的真实权重数值语义。
- reduced-layer 训练：只形成相同层数、相同拓扑候选之间的局部 A/B，不得写成完整模型训练或完整模型性能结论。
