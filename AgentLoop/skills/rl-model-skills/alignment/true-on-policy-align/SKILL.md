---
name: true-on-policy-align
description: >-
  This skill should be used when the user asks to prove or repair deterministic
  prefill differences between training and inference or rollout paths, diagnose
  true on-policy Tensor or logprob drift, locate the first semantic divergence,
  use guarded Tensor replay, or investigate late-onset weight and runtime-state
  transitions. Use a lighter existing-dump workflow when only a report is needed.
  Decode Tensor and TP-sharded collective alignment are out of scope.
version: 1.0.0
domain: rl-model
function: alignment
tags:
  - true-on-policy
  - train-inference-alignment
  - prefill
  - tensor-replay
  - rollout
license: MIT
---

# True On-Policy 训推一致性对齐

在任意训练后端与推理或 rollout 后端之间，对声明的 **prefill** 数学边界进行
精确对齐。严格区分差异发现与生产修复：

1. 使用训练侧 Tensor 对已证明的差异边界做受保护 replay，发现相互独立的语义差异；
2. 冻结 Discovery Registry；
3. 运行时支持有界并行 worker 时，可让隔离的 repair worker 研究独立 Repair Unit；
4. 由主执行者按依赖顺序串行集成候选，并通过真实目标运行路径复验。

执行对齐任务时不得修改本 Skill。只在用户授权范围内修改目标工作树。

## 与仓库现有 Skill 的边界

- `align-mindspeed-transformers-precision` 和 `bridge-precision-alignment` 主要验证
  Hugging Face 与 MindSpeed 实现之间的模型迁移精度、logits 和 cosine 门禁；
- 本 Skill 处理训练路径与推理或 rollout 路径之间的 exact Tensor 首差异、受保护
  replay、权重或运行状态转换以及原始拓扑复验；
- 若任务只要求根据已有 dump 生成轻量根因报告，不进入完整 discovery、repair 和
  真实运行验证，应选择更轻量的已有工件分析流程。

## 先锁定最终目标

在缩减模型、拓扑或响应长度前，将用户要求的模型、端到端范围、成功指标、历史
失败 step 和最终停止条件写入机器可读的 `state_card.json`。单层、单 token、单 step
并将 tensor parallelism 降至 TP=1 的 case 只是中间诊断门禁，不得被静默改写为
最终目标。

使用以下命令初始化和校验状态卡：

```bash
python scripts/validate_state_card.py --init alignment-session/state_card.json
python scripts/validate_state_card.py --state alignment-session/state_card.json
```

长期远端任务还应按需阅读
[运行可靠性说明](references/operational-reliability.md)。

## 范围与真值标签

- 数学发现范围是确定性、单样本 **prefill** 的显式语义边界；
- 构造最小 case 时，在后端允许的情况下将 TP 降到 1，以排除 TP collective 和
  reassembly 对非 collective 数学的干扰。这只是诊断最小化步骤，不是 Skill 的
  整体拓扑限制；修复后必须回到用户要求的原始拓扑完成迁移与最终验证；
- 不假设固定框架、推理引擎、加速器或 dump 工具。采集前记录训练后端、推理后端、
  硬件或运行时后端以及证据提供者，并按
  [后端适配契约](references/backend-adapters.md) 选择或实现 adapter；
- 不进入 decode Tensor 对齐。长程训练可以作为 E2E 或状态回归证据，但不能证明
  decode 数学一致；
- 不分析 TP-sharded collective 或 TP reassembly。若根因只存在于 collective 行为，
  明确报告为超出范围；
- 不实现 `PROMPTS_ONLY`。训练输入可能包含 prompt 和 response，而 rollout prefill
  只包含 prompt；比较与 replay 前必须记录 token provenance，并通过 manifest 中的
  `slice` 或 `select` 显式取 prompt；
- 在 logits 或 logprob 边界单独检查 next-token causal shift，避免 prompt slice
  在因果位移后出现 off-by-one；
- 可选结构预分析只能加速语义映射、缩小重采范围和解释候选，不得成为入口条件，
  也不得削弱 exact 门禁或替代原始 pair-manifest 路径。需要时阅读
  [结构预分析](references/structural-triage.md)。

只输出以下三种 prefill 结果之一：

| 结果 | 含义 |
| --- | --- |
| `prefill_native_exact` | 所有 replay 关闭后，声明的 prefill 覆盖范围严格相等 |
| `prefill_exact_after_unresolved_replay` | 只有启用已声明的未解决 replay 边界时，下游覆盖范围严格相等 |
| `prefill_not_exact` | 仍存在可修复或未分类差异 |

用户声明的 E2E 差异指标，例如 `training/rollout_probs_diff_mean == 0`，属于回归或
影响证据，不能代替 prefill Tensor 证明。非零 E2E 指标在没有新边界证据时不会推翻
已经证明的 prefill 边界；若它与静态 exact 结果冲突，则必须进入
[状态验证](references/stateful-validation.md)。

分别跟踪 Tensor、权重同步、状态型训练、长程、迁移或最终模型、干净运行时和性能
结果。`prefill_native_exact` 不等于 `end_to_end_alignment_verified`。

## 模块路由

| 模块 | 文件 | 退出证据 |
| --- | --- | --- |
| 1. 远端就绪 | [remote-connect.md](remote-connect.md) | 已确认目标、授权、环境、smoke 日志和安全清理范围 |
| 2. 最小证据 case | [dump-case.md](dump-case.md) | 不可变诊断模型、语义覆盖矩阵、最小样本和可比较工件 |
| 3. 编排 | [remote-ops.md](remote-ops.md) | 带版本的 discovery epoch、repair queue 和状态/长程验证矩阵 |
| 4. 差异发现 | [analysis.md](analysis.md) | 精确首差异或已证明的权重、状态、控制面边界，以及 Difference Packet v2 |
| 5. 修复策略 | [fix-patterns.md](fix-patterns.md) | 每个 Repair Unit 对应一个通过校验的修复计划 |
| 6. 报告与打包 | [packaging.md](packaging.md) | JSON、终端 TXT、Markdown、单文件 HTML、patch 和脱敏案例工件 |

只读取当前模块及其直接需要的 reference。通过 handoff 记录传递状态，不得凭记忆
重建未知事实。

## 一次性确认授权

远端操作前一次列出所有可预见问题并记录明确答复，不要为已经授权的事项反复打断：

```text
AUTH_SSH_INSPECT=yes|no
AUTH_START_MINIMAL_JOB=yes|no
AUTH_STOP_RECORDED_JOB=yes|no
AUTH_EDIT_ALIGNMENT_WORKTREES=yes|no
AUTH_CREATE_ISOLATED_REPAIR_WORKTREES=yes|no
AUTH_DIAGNOSTIC_TENSOR_REPLAY=yes|no
AUTH_PARALLEL_REPAIR_PLANNING=yes|no
AUTH_INSTALL_MISSING_PACKAGES=yes|no
AUTH_CLEAN_APPROVED_DUMP_ROOT=yes|no
AUTH_CLEAN_SESSION_TEMP=yes|no
```

同时确认 SSH 目标、已准备的环境或容器、工作目录、命令、日志位置、资源所有权规则、
工件根目录、backend adapter，以及凭据是否需要临时 askpass。不得把凭据写入 Skill、
工作区、报告或 casebook。

## 可恢复状态卡

使用 `references/templates/state_card.json`，不要用自由格式笔记。状态卡记录不可变
目标、授权、环境与后端轴、源模型和有效模型指纹、语义保持的最小化矩阵、discovery
epoch、多轴真值、运行时或 cache/reload 状态、受管作业、保留工件和下一门禁。

未知事实必须保持为 `unknown`、`not_run`、`evidence_missing` 或 `null`，不得填入
看似合理的默认值。保留状态历史，使后续执行者可以从证据恢复任务。

## 三阶段闭环

### 阶段 A：Discovery Sweep

模块 3 循环调用模块 4：

```text
ENSURE_COMPARABLE_DUMPS
  -> OPTIONAL_STRUCTURAL_TRIAGE
  -> FIND_FIRST_DIVERGENCE
  -> FREEZE_DIFFERENCE_PACKET
  -> BUILD_REPLAY_ENTRY
  -> VERIFY_REPLAY_BOUNDARY_EXACT
  -> APPEND_REPLAY_CHAIN
  -> RECOLLECT_MINIMAL_PREFILL
  -> FIND_NEXT_DIVERGENCE
  -> FREEZE_DISCOVERY_REGISTRY
```

Discovery 期间不得应用改变模型行为的生产修复，只能使用会话内 capture 和诊断
replay。只有在声明的上游 replay chain 生效后，所有因果输入都严格相等，才能把
一个差异判定为独立差异。

结构预分析 fail-soft，真值门禁 fail-closed。预分析元数据缺失或校验失败时，记录
原因并继续原始 exact pair-manifest 流程。候选排名不能定义首差异；正式比较必须
沿已证明的语义执行顺序进行。

以训练侧 Tensor 作为 replay 标准值。注入前校验模型、样本和配置 digest、phase、
layer/op/call index、shape、dtype、token slice、文件 hash 和副作用覆盖范围。使用
后端原生且不会执行代码的安全加载器。内置 PyTorch adapter 只允许通过
`torch.load(..., weights_only=True)` 加载当前会话创建的 Tensor；其他后端必须
提供等价 adapter 后才能 replay。

若内部 native 或 fused 算子无法安全 replay，将边界提升到最近的 Python 可见语义
模块，并完整覆盖其输出和因果副作用。不得跨越不安全 replay 边界推断下游结论。

### 阶段 B：Repair Waves

Discovery Registry 冻结后，按根因、活动 symbol 和数学差异合并重复项，形成 Repair
Units。只有运行时支持有界并行 worker 且用户预先授权时才并行研究，否则顺序处理。

- 主执行者拥有远端作业、canonical worktree、registry 和集成权；
- 每个 repair worker 只接收一个不可变 Repair Unit 和原始证据引用；
- worker 在隔离 worktree 中修改，或只返回 patch，不得触碰 canonical worktree；
- worker 不得启停共享设备作业、清理工件、合并代码或声明真实路径成功；
- 主执行者按依赖顺序串行集成。已解决边界关闭 replay，未解决边界保留 replay，
  然后重新运行目标 prefill 门禁。

需要时阅读[并行 worker 编排](references/agent-orchestration.md)和
[Tensor replay](references/tensor-replay.md)。

### 阶段 C：状态、迁移与最终验证

当差异延迟出现、静态 exact 与 E2E 冲突，或权重、worker、queue、cache 发生转换时，
执行[状态转换矩阵](references/stateful-validation.md)。使用严格长程 auditor 和
first-failure trigger，不要等运行结束后才保存首个非零状态。

启动前预置有界 backend capture。auditor 只能在指标写入日志后发出 trigger，无法
恢复已经丢弃的 Tensor；未预置 capture 时，trigger 只能标记确定性复跑点，Tensor
真值必须保持 `evidence_missing`。

按以下层级验证：静态 Tensor → 最小化状态或长程 → 语义代表性缩减模型 → 用户要求的
模型和运行时 → 干净进程与性能。每次运行，包括失败和未完成运行，都写入
`validation_matrix`。只有最后一个必需层级通过后才能关闭用户目标。

## 修复策略

- 默认让推理侧复现训练侧数学；
- 每个 Repair Unit 只做一个因果修改，保持 API、device、dtype、layout、scheduler
  和 no-backward 约束；
- 每个生产修复使用独立且窄范围的 feature gate 或 alignment profile。诊断 replay、
  capture、强制 response 和 fingerprint 使用单独控制，不得共用生产 gate；
- 只尝试有证据支持的方案。没有新证据时，将边界标记为 `unresolved_replayed` 并
  继续，不得无限循环；
- Attention 或 FA 可以保留为未解决 replay 边界。固定工件候选或替换训练算子不算
  真实修复，除非新的活动 worker prefill dump 证明原生边界严格相等；
- 不得根据 mock、缩减统计量、`allclose` 或独立 probe 宣称原生对齐。

## Exact 门禁

每对 Tensor 在声明的语义转换后必须同时满足：

```text
same_shape=true
same_dtype=true
no_nonfinite=true
torch.equal=true
max_abs_diff=0
```

每次 replay 还必须证明 replay 已激活且紧邻的下游输入严格相等。每个修复都必须经过
probe 复现、候选 exact、活动 worker 激活、窄范围真实路径重采和持续下游比较。

长程通过还要求 step 覆盖完整、无缺失或重复指标、无非有限值、无非零值，并在已知
历史失败 step 后继续运行。最终生产验证要求新进程、新权重加载、关闭 replay 和
capture、生产 alignment profile 激活证据，以及用户要求的性能门禁。

## 报告真值与格式

始终同时生成：

1. 一页式摘要，展示所有差异、状态和最终范围；
2. 详细报告，包含证据、代码路径或差异片段、拒绝方案、replay 依赖、验证和限制。

四种视图来自同一份结构化事实源：

- `alignment_report.json`：机器可读事实源；
- `alignment_summary.txt`：Linux 终端摘要；
- `alignment_report.md`：仓库详细报告；
- `alignment_report.html`：响应式、可离线打开的单文件页面。

每次新任务先通过以下模板初始化事实源，再只填入本次证据：

```bash
python scripts/generate_alignment_report.py \
  --init-source alignment-session/alignment_report_source.zh-CN.json
```

模板位于 `references/templates/alignment_report_source.zh-CN.json`。不得手写新 schema，
也不得复制旧案例作为起点。面向用户的标题、标签、状态、解释和案例叙述使用简体中文；
JSON key 和 enum 保持稳定英文。TXT 视图不超过 96 显示列，Markdown 和 HTML 应保证
长路径、代码和详情在桌面及窄屏下均可读。

报告记录 discovery、repair worker、集成、设备运行和总墙钟时间；区分 native 与
conditional 结果；不得发布原始 Tensor、完整 dump、prompt、凭据、主机、私有路径
或其他秘密。

存在多次 E2E 测量时填写 `metric_progression`，记录每次 run 的范围、step 和累计
Difference ID。标准训练、最小单层和完整 rollout 的指标不得拼成一条可直接比较的
曲线，也不得把 E2E 零值升级为 prefill exact。

新的状态型任务必须填写 `acceptance_contract`、`verification_axes` 和
`validation_matrix`。已解决差异同时保留 `discovery_max_abs_diff` 与
`verification_max_abs_diff`，不得用零覆盖原始非零证据。只有所有必需门禁都有
passing validation row 时，才能设置 `acceptance_contract.status=pass`。

## 清理与安全

- 只停止已记录的作业进程组，或用户明确确认的精确 PID；
- 不使用宽泛 `pkill`、无范围 `/tmp` glob 或破坏性 git reset/checkout；
- 只删除已授权的 `DUMP_ROOT/train`、`DUMP_ROOT/infer` 和已记录会话临时目录；
- 保留 discovery、repair、最终脚本、报告、patch 所需的最新证据，以及用户已有的
  dirty changes；
- 发布前通过 `scripts/validate_production_patch.py` 审计生产 patch，并保留 JSON
  结果；生产 patch 不得包含诊断 replay、capture、forced-response、fingerprint
  代码或 Tensor payload。
