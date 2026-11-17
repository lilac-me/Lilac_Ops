---
name: mindspeed-llm-fsdp2-model-lifecycle
description: 统一编排 MindSpeed-LLM FSDP2 原始模型分析、模型接入、原生与 FSDP2 logits 精度对齐、远端训练基线和逐项融合算子调优，并在隔离 Git worktree 中为通过门禁的阶段创建可恢复提交。适用于从 HuggingFace causal language model 开始，一次完成 FSDP2 迁移、验证和性能优化，或从既有接入/优化检查点恢复执行的任务。
---

# MindSpeed-LLM FSDP2 模型生命周期主控

将本 Skill 作为全流程唯一主控。主控负责隔离工作区、阶段编排、Git 版本和统一报告；领域实现与验证结论由对应子 Skill 负责。

## 子 Skill

按需完整读取并调用：

1. `$llm-model-analysis`：在接入前分析原始模型并生成 Markdown 报告与结构化 JSON；本开发目录中的副本来自 MindSpeed-Agent 顶层 `skills/llm-skills/model-analysis/`，不使用 `fsdp2-model-adaptation` 内嵌分析 Skill。
2. `$fsdp2-model-adaptation`：消费模型分析结论，实现模型、注册、并行计划、FLOPS/MFU 和启动配置。
3. `$fsdp2-precision-alignment`：比较原生 Transformers 与真实 MindSpeed-LLM FSDP2 logits。
4. `$mindspeed-llm-remote-validation`：同步隔离 worktree，执行远端预检和训练并回收日志。
5. `$mindspeed-llm-fsdp2-fused-op-integration`：逐项接入和验证融合算子。
6. `$mindspeed-llm-loss-accuracy-validation`：执行训练 loss 精度门禁。

不要调用 `$mindspeed-llm-fsdp2-main-control`，避免两个主控同时管理 worktree、提交和报告。

## 执行模式

- `full`：原始模型分析、模型接入、精度验收、训练基线和自动优化全部执行；默认模式。
- `adapt-only`：完成原始模型分析、模型接入和 eager FSDP2 精度验收后停止。
- `optimize-only`：从用户指定的已验收 commit 开始建立训练基线并执行优化。
- `resume`：读取已有 manifest、统一报告和 Git 状态，从最后一个成功检查点继续。

## 开始前收集

**收集方式（任何 agent 运行环境通用，不依赖 hook 或特定提问工具）：**

从用户提供的任一锚点开始执行只读自动发现；先推断仓库、模型、源码、环境、脚本和数据，只追问无法可靠推断且会影响执行的歧义项，不要求用户预先填写或逐项回答完整清单。已由证据确认的信息不得重复索要；发现路径类型错误或证据冲突时应明确指出。

自动发现只生成本次运行的候选输入，不能代替用户授权。发现完成后必须向用户展示一份脱敏的确认摘要，覆盖下方内部完整性清单中的已解析项、采用的默认值、未决项和阻塞项，并请用户确认或更正。用户更正后影响摘要内容时，必须重新展示更新后的摘要。

只有取得用户对确认摘要的明确同意后，才可以初始化 lifecycle run、创建 worktree、生成 manifest、同步代码、启动训练或修改目标仓库。环境收集和等待确认期间不得执行这些操作；若用户只要求环境发现，则输出脱敏后的候选输入、未决项和阻塞项后停止。

执行环境信息（服务器地址与账号、Conda 环境、CANN 环境及 PYTHONPATH 等）即使已通过只读自动发现在既有私有配置（如 `*.local.json`）中找到，也必须在对话中向用户显式确认后才能使用；配置文件的存在不等于使用授权。每次 lifecycle run（包括 `resume`）首次执行远端同步、预检或训练前，必须在确认摘要或单独提问中取得用户对这些项的明确同意；run 中途这些项发生变化或换用其他配置时，必须重新确认。

对于包含融合优化的 `full` 或 `optimize-only`，`fusion_validation_strategy` 是必须由用户明确选择的对话项，不得从算子数量、单一最终提交等描述中自行推断。若初始请求尚未明确，初始化前必须请用户二选一：`sequential`（逐个接入、逐个验证与接受，便于归因和定位，推荐）或 `atomic-combined`（全部指定算子作为一个不可拆分候选，只做一次组合训练/性能/精度验收，不归因单算子，任一失败则整体拒绝）。选择原子组合仍须逐算子执行本地结构和语义 probe。`adapt-only` 记为 `not-applicable`；`resume` 复用 manifest 中已记录的选择，缺失时在进入融合阶段前补问。

**内部完整性清单（不要在启动时整份抛给用户）：**

- 目标 MindSpeed-LLM 仓库、原始分支和允许修改路径。
- 模型准确名称、原生 modeling/configuration 来源和完整权重目录；权重完整性必须明确声明（全量 / 仅部分分片——部分分片走减层验证路线 / 缺失）。
- Reference 与 Candidate Python/Conda 环境、FSDP2 YAML 或生成它所需的并行约束。
- 本地或远端执行环境、NPU 数量、远端私有配置路径和训练数据/脚本。
- logits cosine、loss 相对误差和最低性能提升门限；loss 平均相对绝对误差默认门限为 `<= 1%`，用户可以在提示词或确认摘要中覆盖。
- 融合验证策略：`sequential` 或 `atomic-combined`；包含优化的运行必须取得用户明确选择。
- 输出目录，以及是否允许恢复本次隔离 worktree 中未通过的候选修改。

可以从 fixture JSON 读取这些输入，但读取结果必须进入确认摘要并由用户确认。密码、私钥、token 和真实 host 不得写入 fixture、报告或仓库。

## 初始化

只有在自动发现完成、确认摘要已展示且用户明确同意后，才执行初始化。`resume` 遇到没有输入确认记录的旧 manifest 时，也必须先展示恢复摘要并取得确认。

1. 检查用户确认的目标仓库、分支、`HEAD` 和工作区状态，不处理用户已有修改。
2. 从原始 `HEAD` 创建专用分支和独立 Git worktree。所有模型修改只发生在该 worktree。
3. 生成唯一 `run_id`。找到同时包含 `skills/`、`fixtures/` 和 `output/` 的开发根目录，默认运行目录为其中的 `output/fsdp2-lifecycle-<run_id>/`。
4. 创建 `run_manifest.json` 和 `fsdp2_lifecycle_report.md`，格式遵循 `references/run_manifest_schema.json` 与 `references/stage-contracts.md`。
5. 记录已确认的输入路径、环境标识、执行模式、门限、策略和确认状态，不复制敏感配置内容。新运行还要记录执行模型/推理档位（运行环境未暴露时记为 `unavailable`，不得猜测）、实际读取的 Skill/关键 reference 的 SHA256，以及采用的成功参考 commit；旧运行的指标和结论只能作为环境线索，不能写成本次证据。自动发现得到的旧运行路径、配置或 commit 只有在确认摘要中经用户同意后，才能写入本次 manifest。
6. 向远端验证、loss 比较、融合报告和精度对齐显式传入本次运行的子目录，不依赖各子 Skill 的独立默认输出路径。

## 阶段一：接入前原始模型分析

在修改目标仓库前调用 `$llm-model-analysis`，只以用户明确确认的原始模型源码及其对应的不可变版本为主证据，传入原生 modeling/configuration、真实 config、权重或权重索引、训练假设、可用并行维度、设备显存和昇腾软件版本。

- 按 `$llm-model-analysis` 的固定输出契约生成 `model_analysis.md` 和 `model_analysis.json`；结构与 Shape 流程图写入 Markdown，不额外要求图片文件。
- 明确模型结构、逐模块 Tensor Shape、参数与显存、FLOPS/MFU、通信量、NPU 支持缺口及并行切分建议。
- 本阶段不得修改目标 MindSpeed-LLM worktree，`modified_files` 必须为空，`output_commit` 必须为 `null`。
- 两个固定交付文件不完整、关键输入不足或状态为 `blocked` 时停止，不进入模型接入。

将产物写入本次运行的 `01-model-analysis/<model-name>/`，后续阶段只通过 manifest 中记录的绝对路径消费产物。

## 阶段二：模型接入

调用 `$fsdp2-model-adaptation`，传入 worktree、原生源码、权重、模型 ID、并行约束、接入前 `model_analysis.md`、`model_analysis.json` 和输出目录。

- 只建立正确、可运行的 eager/reference baseline。阶段二的产物必须能在完全没有本模型专属性能融合实现的情况下成立。
- 以分析报告中的模块结构、Shape、NPU 缺口、并行建议和 FLOPS 口径作为接入设计输入。
- 不在本阶段新增或启用以性能为目的的融合算子：不得新增本模型专属 fused 分支、kernel 调用、patch 注册、为 fused 路径服务的参数重排或 conversion mapping。EP eager dispatcher 合同属于基础接入，但不得借此提前加入 fused dispatcher 或 Grouped GEMM。
- 完成 modeling、模型注册、参数声明、并行模块路径、FLOPS/MFU、YAML 和启动脚本。
- 执行 Python、Shell、YAML、导入和 FLOPS 注册静态检查，并对 `input_commit..working_tree` 做接入边界审计。新增 `npu_fusion_attention`、`npu_rms_norm`、fused RoPE、Grouped GEMM/GMM、融合开关或 `MindSpeedPatchesManager` 注册命中时默认阻塞；只有证明是未修改的上游兼容声明而非可执行融合路径时才可在审计报告中解释放行。
- 将实现与接入前分析不一致的项目写入差异清单，并附源码证据；不得静默改写分析结论。
- 生成 `adaptation_boundary_audit`、参数/模块结构签名和 forward 语义不变量矩阵；缺少任一产物不得进入阶段三。

阶段失败或阻塞时停止，不进入接入后一致性与精度验收。

## 阶段三：接入后一致性检查与 eager FSDP2 精度验收

先将适配后 modeling、模型注册、并行模块路径、FLOPS estimator 和启动配置与接入前分析报告逐项核对。差异必须具有源码证据和明确处置状态。对容易“能运行但语义错误”的路径，必须逐项核对调用顺序，例如 Q/K projection 后的 per-head norm、RoPE、KV repeat、mask 与输出布局，不能只比较类名和 shape。

- 若差异只是 FSDP2 封装、并行计划或 eager fallback 的实现细节，在差异清单中解释，不重复运行完整模型分析。
- 若接入过程中发现原始 config、模型结构或执行路径的分析结论有误，或适配实现有意改变了模型语义，必须再次调用 `$llm-model-analysis`，将结果写入独立的 `03-post-adaptation-analysis/<model-name>/`，不得覆盖接入前报告与 JSON。
- 存在未解释的结构、Shape、参数、FLOPS 或模块路径差异时，将阶段标记为 `blocked`。

一致性检查通过后，复用 `$mindspeed-llm-remote-validation` 的同步、环境预检、NPU 门禁和产物回收能力，在远端调用 `$fsdp2-precision-alignment`：

1. Reference 使用用户提供的原生 Transformers modeling。
2. Candidate 必须经过 `train_fsdp2.py` 参数体系、`ModelFactory`、并行引擎、FSDP2/EP 包装和 `Trainer._compute_loss`。
3. 单机环境统一使用 `--layer-counts 1,2`：1 层作为冒烟/隔离层（启动开销最小的链路自检，2 层失败时用于二分结构问题与跨层问题），2 层作为验收层（跨层交互的最小覆盖）。统一使用 4096 个全有效 token。显存受限时按精度 Skill 的降级策略执行：OOM 先用 `--expert-count` 减专家（可被 EP 整除且 ≥ top-k），仍不足才减到 1 层；减专家必须在报告中显式记录为验收偏差。
4. 所有可选融合算子保持关闭，`execution_label` 使用 `eager`。

本地不要求安装 NPU 运行环境。按 precision Skill 的脚本合同只同步 8 个运行文件到远端仓库的 `.validation-runtime/precision-alignment/`，不包含测试和环境安装工具。远端只调用模型仓库内的 `run_complete_alignment.sh`；该 launcher 再调用唯一公共入口 `run_alignment.py`。使用一条 `run_validation.py --mode run` 命令完成同步、启动和结果回收。

只有减层正式门禁通过（默认 2 层）、权重完整加载且真实 FSDP2 证据成立时，才创建模型接入 checkpoint commit。

## 阶段四：远端训练基线

使用模型接入 checkpoint 调用 `$mindspeed-llm-remote-validation`，显式传入隔离 worktree 作为 `--repo-root`。

- 先执行 `plan` 和 `preflight`，确认本地路径、远端仓库、Conda、CANN、训练脚本和 NPU 状态。
- 再执行未启用可选融合算子的短训练，保存为本次运行的 `remote-validation-0001`。
- 固定代码、权重、数据、随机种子、步数、batch、序列长度、dtype 和并行拓扑。
- 要求 forward、backward、optimizer step、loss 和日志均有效。
- 性能/显存基线默认使用原始 `num_hidden_layers` 和完整模型结构；权重不完整时可以使用明确声明的同源随机初始化。日志必须证明实际实例化并参与训练的层范围。阶段三的 1/2 层 checkpoint view 只用于真实权重 logits 验收，不能据此宣称完整模型训练性能。
- loss 精度基线必须使用加载真实权重的训练运行：权重完整时使用完整架构；权重不完整时使用减层 checkpoint view（默认 2 层、保留完整专家数）建立真实权重 loss 基线，证据范围仅为相同减层配置的局部 A/B，不得外推为完整模型 loss 结论。随机初始化训练的 loss 只作观察值，不构成 loss 精度门禁证据。因此阶段四通常产生两个基线：完整架构性能基线和真实权重 loss 基线。
- 用户明确要求减层训练或完整架构受资源阻塞时，可以生成 reduced-layer smoke/局部 A/B，但报告必须标明验证层数，后续候选只能与完全相同的减层基线比较，且不得外推为完整模型收益。

基线训练失败时停止，不执行优化。

## 阶段五：融合优化

调用 `$mindspeed-llm-fsdp2-fused-op-integration`，并读取其 `references/correctness-gates.md`。严格执行 manifest 中由用户明确确认的 `fusion_validation_strategy`；字段缺失或与当前请求冲突时阻塞并询问，不得自行选择。测试粒度、接受粒度和提交粒度是三个独立概念，不能因为用户要求单一提交就跳过必要的算子级语义检查。

### 默认：逐项模式

对每个候选算子：

1. 从最后一个已接受 commit 开始，一次只修改一个算子。
2. 执行结构签名比较、算子语义 probe、本地静态检查和远端训练，并记录 requested/active。
3. 与上一个已接受运行比较性能和显存，另执行真实权重减层完整词表 logits。
4. 调用 `$mindspeed-llm-loss-accuracy-validation` 比较训练 loss：基线与候选都必须是加载真实权重的训练运行（权重不完整时两侧使用同一减层 checkpoint view，证据范围记为减层局部 A/B），并使用平均相对绝对误差 `<= 1%` 门限；用户已明确指定其他门限时按确认后的值执行。随机初始化训练的 loss 比较不得作为该门禁证据。
5. 结构、语义、active、训练、性能、显存、loss 和 logits 门禁全部通过后，创建独立 checkpoint commit。
6. 未通过时记录日志、报告和 diff；只有已获得用户的限定授权，才恢复本次候选在隔离 worktree 中产生的修改。

一个候选失败不代表其他互不依赖候选必须失败，但 worktree 未恢复到可信检查点时不得继续。

### 用户覆盖：原子组合模式

只有用户明确要求时，才把多个融合视为一个不可拆分候选：

1. 从阶段四可信 eager commit 一次性实现组合；不做逐算子训练、性能归因或独立提交。
2. 仍须对组合中的每个算子执行本地结构与语义 probe；这不是独立性能实验，也不产生单算子接受结论。
3. 组合的 loss 只与阶段四真实权重全关 loss 基线比较（权重不完整时同为减层 view 局部 A/B），性能/显存与完整架构全关基线比较；报告明确“组合整体 A/B，未测量单算子贡献”。
4. 任一算子的 requested/active、结构、forward/backward、loss、性能、显存或真实权重 logits 门禁失败，整个组合 rejected；不得选择性保留，也不得自动切换逐算子或二分诊断。
5. 全部门禁通过后才创建一个组合 checkpoint commit。

## 阶段六：最终组合验收

所有优化结束后，在干净工作树上通过 `$mindspeed-llm-remote-validation` 在远端调用 `$fsdp2-precision-alignment` 验证最终组合，使用清晰的 `execution_label` 记录实际开启的融合和并行路径。最终报告必须绑定执行时的确定 Git commit；`-dirty`、未提交 diff 或标签与实际 active 路径不一致时阻塞。

- 最终完整 logits 门禁通过：保留优化 commits。
- 最终门禁失败：结果为 `failed`，保留报告和提交链供定位；未经授权不改写历史。
- 没有任何算子达到性能门限：结果为 `no_improvement`，模型接入 checkpoint 仍然有效。

## 执行纪律（来自实际运行的教训）

- 诊断与长命令不得用 `| tail` 等管道收尾——会吞掉真实退出码。一律重定向日志文件并显式检查退出码。
- 定位数值/精度回归时先跑“全关对照组”确认基线健康，再逐个开变量二分；若用户明确要求原子组合且禁止二分，则只记录失败并保存证据，不自动扩大诊断范围。
- 每次远端运行前显式核对 YAML/flag 的当前状态；上一轮的临时改动（dispatcher、开关）必须在下一轮前复位，否则二分结论会被残留状态污染。
- 权重等大输入在阶段一启动时就并行开始同步，不要等到精度阶段才发现缺失。
- 任何"疑似挂死"先查远端真实状态（进程、npu-smi、日志更新时间），确认无进展后再等待；依赖 remote-validation 的静默看门狗兜底，不要无限等待。

## Git 与报告规则

- 只有本主控可以创建提交；子 Skill 不得提交、切换分支或改写历史。
- 显式暂存已核验文件，不使用 `git add -A`。
- 不自动 push、merge、rebase、amend 或删除用户工作区。
- 每个阶段按 `references/stage-contracts.md` 写入结构化结果并立即更新统一 Markdown 报告。
- 恢复时以 manifest、统一报告和 Git 检查点三者的一致状态为准；不一致时标记 `blocked`。

## 结束结果

- `success`：模型接入和最终必需门禁通过，报告最终 commit 与最佳训练运行。
- `no_improvement`：模型接入通过，但没有值得保留的性能优化。
- `failed`：执行或门禁失败，报告最早失败阶段和可恢复检查点。
- `blocked`：输入、路径、环境、权重或状态不足以形成可信结论。

最终只向用户汇总：状态、模型接入 commit、优化 commits、关键精度/性能指标、报告与日志路径、未验证项和下一步；不复制完整训练日志。
