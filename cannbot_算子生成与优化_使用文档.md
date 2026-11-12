# CANNBot 算子生成与优化插件使用手册（Triton / Ascend C）


适用范围：本文只介绍 CANNBot 中与算子生成、算子优化相关的 4 个插件，不代表 CANNBot 只有这 4 个插件。完整清单以 cannbot 仓的 `plugins-official/`、`plugins-community/` 目录与 `docs/community-plugins.md` 为准，该清单可能滞后于当前目录。

代码仓链接：https://gitcode.com/cann/cannbot

依据：`plugins-official/triton-op-generator/quickstart.md`、`plugins-community/tilelang2ascendc-ops-generator/quickstart.md`、`plugins-community/ops-perf-evolution/quickstart.md`、`plugins-community/triton-auto-evolve/README.md`，以及各插件的 `AGENTS.md`。

注：社区插件目录名是 `ops-perf-evolution`，不是 `ascendc-ops-perf-evolution`。

## 0. 插件选择

### 0.1 按输入和输出选

| 输入 | 输出 | 插件 | 说明 |
|---|---|---|---|
| PyTorch 参考实现，可选再带 GPU Triton kernel | Triton-Ascend 算子，含精度验证和性能优化 | `triton-op-generator` | 单次任务内完成生成到优化 |
| PyTorch 模型 + 测试用例 | 新建 Ascend C 算子 | `tilelang2ascendc-ops-generator` | 新算子开发，简单/复杂两条路径 |
| 已有 Ascend C 内核，或 ops-nn/cv/math/transformer/omni 仓中的算子 | 更高的性能 | `ops-perf-evolution` | 对已有 kernel 做多轮并行进化 |
| 已有 Triton-Ascend 算子，且已能跑通 | 更高的加速比 | `triton-auto-evolve` | 跨轮调度，成本高于前三个 |

### 0.2 按 DSL 和新写/已有划分

| | 新生成 | 已有代码优化 |
|---|---|---|
| Triton-Ascend | `triton-op-generator` | `triton-op-generator`（Phase 4 单任务内扫描）<br>`triton-auto-evolve`（跨轮进化，成本更高） |
| Ascend C | `tilelang2ascendc-ops-generator` | `ops-perf-evolution` |

`triton-op-generator` 在单次任务内跑完 Phase 4（33 个优化点、IR 多轮、simulator 采集）即交付；`triton-auto-evolve` 由主 agent 跨轮调子 CLI，轮数由 `max_rounds` 决定，token 与 NPU 时间成本明显更高。

### 0.3 总览矩阵

| 生成算子的语言 | 生成场景 | CANNBot 插件 | 支持输入 | 支持框架 |
| --- | --- | --- | --- | --- |
| AscendC | A2/A3 生成 | [tilelang2ascendc-ops-generator](https://gitcode.com/cann/cannbot/tree/master/plugins-community/tilelang2ascendc-ops-generator) | triton/torch/cuda | 参考实现 Claude Code、OpenCode、Trae、Cursor、Copilot、CodeArts |
| AscendC | A2/A3 性能优化 | ops-perf-evolution | AscendC 算子参考实现 | Claude Code、OpenCode、Trae、Cursor
| AscendC | A2/A3 迁移 A5 | [ascendc-port-orchestrator](https://gitcode.com/cann/cannbot/tree/master/plugins-community/ascendc-port-orchestrator) | ascendc/triton/cuda 参考实现  | Claude Code、OpenCode、Cursor、Copilot
| AscendC | A5 生成、训练反向生成 | [ascendc-port-orchestrator](https://gitcode.com/cann/cannbot/tree/master/plugins-community/ascendc-port-orchestrator) | 算子描述/数学公式/输入输出规格 | Claude Code、OpenCode、Codex、DSH、Trae
| AscendC | A5 性能优化 | ops-perf-evolution | ascendc/triton/cuda 算子参考实现 | Claude Code、 OpenCode、Trae、Cursor
| Triton | A2/A3 生成 | [triton-op-generator](https://gitcode.com/cann/cannbot/tree/master/plugins-official/triton-op-generator) | torch/GPU Triton Kernel 参考实现 | Claude Code、OpenCode、Trae、Cursor、Codex、Copilot、CodeArts
| Triton | A2/A3 性能优化 | ops-perf-evolution | triton 参考实现 | Claude Code、OpenCode、Trae、Cursor
| Triton | A2/A3 迁移 A5 | [triton-op-generator](https://gitcode.com/cann/cannbot/tree/master/plugins-official/triton-op-generator) | triton 参考实现 | Claude Code、OpenCode、Trae、Cursor、Codex、Copilot、CodeArts
| Triton | A5 生成 | [triton-optimizer](https://gitcode.com/cann/cannbot/tree/master/plugins-community/triton-optimizer) | Triton Ascend NPU 算子/Torch 参考实现 | Claude Code、OpenCode、Trae、Cursor、Copilot
| Catlass | A2/A3/A5 生成 | [catlass-op-generator](https://gitcode.com/cann/cannbot/tree/master/plugins-official/catlass-op-generator) | 数学 contract/算子需求/可审查参考实现 | Claude Code、OpenCode、Trae、Cursor、Codex、Copilot、CodeArts
| PyPTO | A2/A3/A5 生成 | [pypto-op-orchestrator](https://gitcode.com/cann/cannbot/tree/master/plugins-official/pypto-op-orchestrator) | 算子公式/Shape/dtype/属性 | Claude Code、OpenCode、Trae、Cursor、Codex、Copilot、CodeArts
| TileLang | A2/A3/A5 生成 | [tilelang-op-orchestrator](https://gitcode.com/cann/cannbot/tree/master/plugins-official/tilelang-op-orchestrator) | 算子公式/输入输出规格/PyTorch API 示例 | Claude Code、OpenCode、Trae、Cursor、Codex、Copilot、CodeArts

## 1. 安装与前置

### 1.1 安装入口

四个插件的安装方式不统一，以各自 quickstart 为准。

| 插件 | 一键安装 | 源码安装 | 支持工具 |
|---|---|---|---|
| triton-op-generator | `npx @cannbot-plugin/cannbot@latest install triton-op-generator --tool claude` | `bash plugins-official/triton-op-generator/init.sh project claude` | claude / opencode / codex / trae / dsh |
| tilelang2ascendc-ops-generator | `/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git`<br>`/plugin install tilelang2ascendc-ops-generator@cannbot` | `bash init.sh project claude`（或 `global claude`） | claude / opencode / trae / cursor / copilot / codearts |
| ops-perf-evolution | `/plugin marketplace add https://gitcode.com/cann/cannbot.git`<br>`/plugin install ops-perf-evolution@cannbot` | `bash init.sh project claude`（或 `global claude`） | claude |
| triton-auto-evolve | 无 marketplace 入口 | `bash init.sh project claude` | claude |

两个 marketplace 指向不同的仓（`cannbot-skills` 与 `cannbot`），这是上游发布现状。装不上时先确认插件实际在哪个仓。

项目级（`project`）适合多项目并存、各自隔离配置；全局级（`global`）适合单项目长期使用。tilelang2ascendc 与 ops-perf-evolution 两种都支持，triton-op-generator 与 triton-auto-evolve 以项目级为主。

### 1.2 前置依赖

| 项 | triton-op-generator | tilelang2ascendc | ops-perf-evolution | triton-auto-evolve |
|---|---|---|---|---|
| CANN Toolkit | ≥ 9.0.0 | ≥ 9.0.0（建议） | ≥ 9.0.0（建议） | 同目标算子环境 |
| NPU | 910 / 950 PR | 910 / 950 PR | 910 / 950 PR | 同目标算子环境 |
| Python / PyTorch | ≥3.8 / ≥2.0，加 torch_npu、triton-ascend | ≥3.8 / ≥2.0 | 无 | 无 |
| 其它 | Node.js ≥18、jq（缺失时 hook fail-open） | tilelang-ascend | Claude Code | Claude Code |
| 知识库 | 安装时自动克隆 cannbot-knowledge | 由 `CANNBOT_KNOWLEDGE_ROOT` 解析 | 必须手动配置，否则 exit 3 | 自带 `template/` 经验模板 |

### 1.3 知识库配置

`triton-op-generator` 安装时自动浅克隆 cannbot-knowledge 到 `<项目>/.cannbot/dependencies/triton-op-generator/cannbot-knowledge`。离线时可用 `CANNBOT_SKIP_DEPENDENCY_REPOS=1` 跳过，此时按骨架模式运行，没有知识增强。

`tilelang2ascendc-ops-generator` 的知识仓根由 `CANNBOT_KNOWLEDGE_ROOT` 或 CANNBot 的 `knowledge.env` 解析。检索走 `knowledge-query`，沉淀走 `ops-knowledge-ingest`。

`ops-perf-evolution` 的策略 Playbook 正文托管在知识仓，未配置时直接以 exit 3 报错，不会静默跳过。配置方式任选其一：

1. `export CANNBOT_KNOWLEDGE_ROOT=<cannbot-knowledge 仓路径>`
2. 项目根或任一父级建 `.cannbot/knowledge.env`，写入 `CANNBOT_KNOWLEDGE_ROOT=<路径>`
3. cannbot-knowledge 与 cannbot-skills 在同一父目录下检出，零配置自动命中

自检：`python3 .claude/skills/knowledge-query/scripts/knowledge_query.py discover` 应输出 selected root。

## 2. triton-op-generator：Triton 算子生成与优化

### 2.1 适用场景

适用于从 PyTorch 算子或 GPU Triton kernel 出发，在 Ascend 上生成 Triton 实现，并在一次任务内拿到精度通过、性能尽量高的结果。

算子已经是能跑的 Triton-Ascend 代码、只要求继续提性能时，用 `triton-auto-evolve`。

### 2.2 输入模式

判定优先级从高到低。同时提供 torch 标杆和 GPU kernel 时走 Mode A：torch 作精度基准更可靠，GPU kernel 只作参考实现。

| 模式 | 输入 | 说明 |
|---|---|---|
| Mode A（标准） | PyTorch 参考 `.py` | 单 case（只有 `get_inputs()`）时自动扩展为至少 5 种 shape；多 case（含 `get_input_groups()` 和同名同目录 `.json`）原样透传 |
| Mode A 优先级 1 | torch 标杆 + GPU Triton kernel，可选 `gpu_perf.csv` | GPU kernel 复制为 `gpu_kernel_ref.py` 供草图与编码参考；有 `gpu_perf.csv` 时报告中附 Ascend 与 GPU 的延迟对比 |
| Mode B（仅 GPU kernel） | GPU kernel `.py` + 同名 `.pt`，须含 `input_data`，可选 `gpu_output` | `.pt` 无 `gpu_output` 时由 agent 手写等价 PyTorch 参考；无法翻译则报错，需补 `gpu_output` |

示例：

```
生成一个 Triton-Ascend 框架的 softmax 算子实现，ASCEND_RT_VISIBLE_DEVICES=1
```

### 2.3 流程

| Phase | 内容 | 主要产物 |
|---|---|---|
| 0 | 参数确认、arch 检测、输入模式判定、KB 自举 | 工作目录 |
| 1 | 任务构建、基线冻结 | `{op_name}.py`（可选 `.json`）、`.baseline_anchor.json` |
| 2 | 类别判定、读 Layer 1 约束卡、`triton-op-designer` 出草图、合规检查 | `precheck.json`、`sketch.txt` |
| 3 | `triton-op-coding` 生成、AST 预检查、架构符合性判定、verify、Conductor 分析，循环 | `iter_N/generated_code.py`、`verify_result.json`、`perf_result.json` |
| 4 | `triton-latency-optimizer` 扫描 33 个优化点，加 IR 多轮、simulator 诊断、Block Size 候选扫描 | `opt_iter_N/optimized_code.py` |
| 5 | 报告与 summary | `report.md`、`summary.json` |
| 6 | 会话导出 | `session.jsonl`、`session.md` |
| 7 | 经验归档到 KB 约束卡与 runbooks | 约束卡追加、`index.md` 同步 |

流程中有 6 处约束需要注意：

- 基线冻结。Phase 1 末尾调用 `freeze_baseline.py` 写 sha256 锚。此后修改 `{op_name}.py` 会让 verify/benchmark 以 exit 4 拒绝运行，按 C 类终止。锚文件不可重入，重跑 freeze 会 exit 1。源 benchmark 有 bug 时也不允许改，只能在 `report.md` 标 `baseline_buggy: true` 后失败退出。
- Layer 1 约束门。草图与类别约束卡冲突时按 A 类错误退回重做，最多 2 次。
- 3.2b 架构符合性判定。代码未按 sketch 的 kernel 或路径实现即判 A 类，即使精度更高也不放行，连续 3 次终止。
- 4.6 退出前置门。`scan_complete` 必须为 true，由编排器按扫描游标推导，不采信 skill 自述。latency-optimizer 的优化点耗尽且未达 target 时，必须先调用 simulator 采集，才允许进入终局判定。
- simulator 判据。`MMAD` 实测占比超过 50% 才算硬件极限，其余信号映射回优化点编号（Cube 空等对应分阶段解耦，标量降级对应避免标量降级，访存 bound 对应 pass 合并）。
- 报告口径。`passed_cases` 取自 `verify_result.json`；`perf_result.json` 中的 pass 只表示 benchmark 进程没有崩溃，与精度无关。

### 2.4 产物

```
triton_ascend_output/op_{op_index}_{op_name}_{YYYYMMDD_HHMM}_{rid}/
├── {op_name}.py              任务描述，冻结后只读
├── {op_name}.json            多 case 模式
├── sketch.txt                算法草图
├── output/
│   ├── iter_{n}/             Phase 3 每轮
│   ├── opt_iter_{n}/         Phase 4 每轮；IR 轮为 opt_iter_{n}_ir_{k}，Block Size 轮为 opt_iter_{n}_bs_{BLOCK}
│   ├── generated_code.py
│   └── optimized_code.py
├── {op_name}_generated.py    最终代码
├── report.md / summary.json
└── session.jsonl / session.md
```

### 2.5 常用参数

- 目标加速比：默认 `5`，取自插件 `config.json` 的 `target_speedup`；项目根的 `triton_op_generator.config.json` 可覆盖。该值只影响 `target_reached` 判定与终止条件，不改变流程走向。
- 多 shape：torch 标杆是单 case 时自动扩到至少 5 种 shape；已有 `get_input_groups()` 的多 case 源不允许裁剪成单 case。
- 设备：在对话中指定，如 `ASCEND_RT_VISIBLE_DEVICES=1`。`npu_preflight` 默认要求最小空闲 49152 MB。

## 3. tilelang2ascendc-ops-generator：Ascend C 新算子生成

### 3.1 适用场景

从 PyTorch 模型和测试用例出发新建 Ascend C kernel（直调工程），要求能编译、能过精度，并有性能分析结果。

已有 AscendC kernel 只想提性能时，用 `ops-perf-evolution`。

### 3.2 命令

```
生成ascendC算子，npu=0，算子描述文件为 <op_file 路径>，输出到 <output_dir>
```

参数为 `npu`、`op_file`（算子模型文件）、`output_dir`。

### 3.3 双路径路由

| 路径 | 算子类型 | Skill 链 |
|---|---|---|
| 简单算子 | Index / IndexPut / Gather / Scatter / Nonzero / RepeatInterleave / EmbeddingDenseBackward | `ops-direct-invoke`：Architect 设计 → Developer 渐进式开发 → Reviewer 代码审查（100 分制，含修复循环） |
| 复杂算子 | Elementwise、激活（ReLU/Sigmoid/SiLU/GELU/SwiGLU/Add/Sub/Mul/Div）、Attention（FlashAttention/SparseAttention/GQA）、MatMul 变体、Norm 变体（RMSNorm/LayerNorm）、Sort/TopK、Pooling、多输入融合 | `tilelang2ascend-tilelang-designer` 做 Block 与 Tile 两级设计 → `tilelang2ascend-translator` 转译为 AscendC |

简单路径产出 `docs/DESIGN.md`、`PLAN.md`、`WALKTHROUGH.md`、`REVIEW.md`；复杂路径产出 `design/block_level/`、`design/tile_level/`、`design/PERF_DESIGN.md`、`model_new_tilelang.py`，最终得到 `model_new_ascendc.py`。

### 3.4 流程

```
0 参数确认与算子分类
1 环境准备、kernel 工程初始化、算子注册
2 Case 精简，保留不超过 10 个代表性用例
3 设计表达（简单算子走架构设计与串讲，复杂算子走 TileLang 设计，含 Step 4 性能迭代门禁）
4 AscendC 生成与验证（简单算子走开发与审查，复杂算子走转译与退化检测）
5 性能分析，ops-profiling --quick，只测量归档
6 全量验证，恢复全量用例
7 trace 记录与知识演进，写入共享 KB 的 runbooks/
```

### 3.5 迭代上限

| 计数器 | 上限 | 超限处理 |
|---|---|---|
| 简单路径 REVIEW 修复循环 | 3 轮 | 暂停上报 |
| A 类（编译或运行时错误）`a_retry` | 5 次 | 暂停上报 |
| D 类（精度不匹配）`d_retry` | 12 次（D1 7 次 + D2 5 次） | 暂停，请求人工介入；编排 agent 不得自行改代码 |
| TileLang 退化检测 | 连续 3 次 | 自动终止 |
| TileLang 性能迭代 `p_retry` | 3 轮 | 必须产出 `final_report.md`（含上限分析）后继续，不阻塞流程 |

TileLang 用于设计表达，不作为 correctness gate。TileLang 验证失败但设计意图正确时，可记录原因后跳过，直接进入 AscendC 转译。TileLang 的性能迭代则是强制门禁，达标线为 geomean ≥ 0.6x，除非有对照实验证明是框架底层不支持。

### 3.6 产物

```
{output_dir}/
├── model.py                                 只读
├── <op_name>.json 与 .bak
├── docs/{DESIGN,PLAN,WALKTHROUGH,REVIEW}.md  简单路径
├── design/{block_level,tile_level}/          复杂路径
├── kernel/                                   AscendC 实现
├── model_new_tilelang.py / model_new_ascendc.py
├── performance.json
└── trace.md
```

### 3.7 批量性能测试

多个已生成算子批量对比时用内置脚本，支持单 NPU 串行或多 NPU 并行：

```bash
bash .claude/skills/ops-profiling/scripts/msprof_profile_run.sh --batch \
    --base-dir <多个算子输出子目录的根目录> \
    --max-jobs 7 \
    --device-start 1
```

前置条件是每个算子目录下已有 `model.py`、`model_new_ascendc.py` 和测试用例文件。`--warm-up` 默认 3。

## 4. ops-perf-evolution：Ascend C 已有算子性能进化

### 4.1 适用场景

输入为可编译、能过精度的 AscendC 内核，或 ops 仓库中的算子，目标是在保持精度的前提下提高性能。该插件不新建算子。

### 4.2 两条路径

| 输入类型 | 判据 | Subagent |
|---|---|---|
| Ops 仓库路径 | ops-nn / ops-cv / ops-math / ops-transformer / omni-ops 仓中的算子路径，含 `op_kernel/`、`op_host/` 与构建脚本 | `ops-evo` |
| 基线内核模式 | 其它现有 AscendC 内核目录，含 `model.py`、`model_new_ascendc.py`、`kernel/` | `lingxi-evo`（默认） |

### 4.3 命令示例

```
对 ops-nn 仓的算子进行进化优化，npu=0，算子名称为 softmax，
算子路径为 <ops 仓内的算子目录>，目标加速比 2x，进化轮数 3，并行数 5
```

```
对基线内核进行进化优化，npu=0，基线内核路径为 <基线内核目录>，
目标加速比 3x，进化轮数 3，并行数 5
```

### 4.4 工作流

```
0 参数确认与路径判断
1 环境检测与设备绑定
2 基线性能评估（ops-profiling / ascendc-ops-evaluation）
3 世界模型初始化（ascendc-evolution-world-model）
4 多轮进化：策略选择 → 并行生成变体 → 构建与评估 → 证据积累
5 最佳方案确定与泛化验证
6 生成 evolution-report_*.html
```

策略选择由世界模型驱动，依据是已积累的决策树与评测数据，不是随机重试；`stagnation_count` 与 `open_questions` 记录停滞原因。

### 4.5 参数与行为

| 参数 | 默认 | 说明 |
|---|---|---|
| 目标加速比 | 用户指定 | 基线或某轮变体达标时提前结束 |
| `max_rounds`（进化轮数） | 2 | 轮间串行，轮内并行 |
| `parallel_num`（并行数） | 3 | 同一轮内的候选变体数 |

变体全部生成在 `{output_dir}/`，原基线目录保持只读。Ops 仓库路径连续构建失败 3 次后暂停并上报，由人检查环境或仓库状态。

### 4.6 产物与查看位置

```
output/<op_name>_<mode>_<timestamp>/
├── evolution-report_*.html    可视化总报告
├── state.json                 当前进度
├── world_model.json           决策树与评测数据
├── shared/
└── round_N/parallel_K/
    ├── kernel/                该变体的 AscendC 代码
    ├── evaluation_results.json  compile / precision / speedup
    └── profiling/             msprof 数据
```

| 要查的内容 | 位置 |
|---|---|
| 整体效果 | `evolution-report_*.html` |
| 最优变体代码 | `world_model.json` 中 best_score 节点的 `solution_ref`，对应 `round_N/parallel_K/kernel/` |
| 每个变体的精度与性能 | `round_N/parallel_K/evaluation_results.json` |
| 当前进度 | `cat state.json` |
| 停滞原因 | `world_model.json` 的 `stagnation_count` 与 `open_questions` |

### 4.7 使用建议

quickstart 的 Tips 中有两条：

1. 进化轮数内未达标时，可以改用定向话术继续，例如「分析性能瓶颈，列出当前最优变体三个最严重的性能劣化因素」「针对劣化最严重的因素，在最优变体的基础上再优化 1 轮」。人工看过代码后也可以直接指路：「基于最优变体，修正 xxxx 的实现问题，再进化一轮」。
2. 进化期间的性能数字只作参考。优化过程中会精简测试用例，最终性能需用全量用例复测：
   - 交互方式：「采用 ops-profiling --quick 模式重新采集一遍性能」
   - 手动方式：
     ```bash
     ASCEND_RT_VISIBLE_DEVICES=5 bash .claude/skills/ops-profiling/scripts/msprof_profile_run.sh \
         --quick --output-dir=<kernel 目录>
     ```

## 5. triton-auto-evolve：Triton 已有算子自动进化

### 5.1 适用场景

算子已经是 Triton-Ascend 实现且精度能过，需要把加速比推到 `target_speedup`。插件的价值在于跨轮次复用历史奖励与上下文隔离，代价是 token 消耗较大。

### 5.2 架构

```
主 Orchestrator Agent       只做跨轮调度：选方向、选 baseline、停止判定、最终汇总
  └─ 每轮启动 1 个子 Claude Code CLI，隔离上下文，执行 Phase 0-8
        └─ gene-fusion 模式在子 agent 内部再并行派发 3 条 route 子 CLI
```

主 agent 不生成代码、不跑 verify 与 benchmark、不写 attempts.md；子 agent 不做跨轮判定，不读历史 `round_index.json`；C1/C2 判定只在主 agent。

### 5.3 命令

```
请优化 XX 目录的 Triton Ascend 算子，从 baseline 开始按多轮迭代优化
```

新任务时主 agent 会用 `AskUserQuestion` 让用户从 `config.json` 中 `enabled=true` 的算法（`gene-fusion`、`mcts`、`naive`）里选一个，结果写入 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json`，不回写 config。

### 5.4 三种算法

| 算法 | 机制 | 适用 |
|---|---|---|
| `gene-fusion` | Round 1 做计算原子分解与自动融合分析，并行派发多条 route 构建初始基因池；Round 2 起做基因选择、交叉、变异，并由 profiling 驱动新增基因 | 算子可拆成多个可融合的原子，如 GEMM+激活、多输入融合 |
| `mcts` | 每个 `opt-round-*` 作为树节点，用 UCT 与 Progressive Widening 决定从哪个历史轮扩展 | 需要结构化探索，用历史奖励指导 baseline 选择 |
| `naive` | 按 `direction_yield`、兼容性、recency 选 baseline，方向由 LLM 自主探索 | 默认策略，不确定时先用它 |

### 5.5 配置

插件根的 `config.json` 是全局默认，目标算子项目根的同名 `config.json` 会覆盖它。

```json
{
  "target_speedup": 5,
  "max_rounds": 20,
  "initial_worker_timeout": 7200,
  "max_worker_timeout": 28800,
  "npu_device_id": null,
  "npu_idle": {"aicore": 1.0, "aivector": 1.0, "npu_util": 1.0, "hbm": 10.0},
  "algorithms": {
    "gene-fusion": {"enabled": true, "params": {"max_routes": 3, "route_timeout": 3600, "...": "..."}},
    "mcts": {"enabled": true, "params": {"c_uct": 1.414, "c_pw": 2.0, "...": "..."}},
    "naive": {"enabled": true, "params": {}}
  }
}
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `target_speedup` | 5 | 某轮 `best_speedup` 达到即满足 C1，可停止 |
| `max_rounds` | 20 | 非 idle round 数达到即满足 C2，强制停止 |
| `initial_worker_timeout` | 7200 s | 单个子 CLI 初始超时；某轮超时则下一轮翻倍 |
| `max_worker_timeout` | 28800 s | 超时翻倍的上限 |
| `npu_device_id` | null | 为 null 时由空闲检测自动选卡 |
| `npu_idle.*` | 1.0 / 1.0 / 1.0 / 10.0 | 空闲阈值；可用 `TRITON_NPU_IDLE_MAX_AICORE`、`TRITON_NPU_IDLE_MAX_AIVECTOR`、`TRITON_NPU_IDLE_MAX_NPU_UTIL`、`TRITON_NPU_IDLE_MAX_HBM` 覆盖，`TRITON_SKIP_NPU_IDLE_CHECK=1` 可跳过检测 |

### 5.6 停止条件与成本

停止只有两条途径：C1 达标，或 C2 达到 `max_rounds`。

`naive` 策略不再主动建议停止（`can_recommend_stop=false`）。目标未达成时会一直跑到 `max_rounds`，启动前应按 `max_rounds × 单轮耗时` 估算 token 与 NPU 成本。

gene-fusion 单轮包含 3 条路线 × 3 次内部迭代，单轮耗时数小时属正常。

### 5.7 产物

```
triton_ascend_output/{op_name}-{algorithm}-{run_tag}/
├── opt-round-N/                         每轮产物
├── .strategy_state/{op_name}_mcts_tree.json   mcts 树状态
└── {op_name}_final.py                   全局最优
```

另有 `.triton-agent/state-*.json`（会话状态，用于恢复）、`round_index.json`、`opt-note.md`、`summary.json`。

### 5.8 template 目录

插件 `template/` 下按类别放置 `attention.md`、`normalization.md`、`quantization.md` 等文件，`init.sh` 会复制（不是软链）到项目的 `.claude/template/`，作为子 agent 的优化约束与经验参考。修改源码 `template/` 后需重跑 `init.sh` 才会同步；运行期间 Orchestrator 追加的经验只写入项目副本，不回写插件源码。

## 6. 组合使用

### 链路 A：GPU kernel 落到 Ascend Triton

```
GPU Triton kernel（.py + .pt）
  → triton-op-generator（Mode B），Phase 3 出可运行版本，Phase 4 扫 33 个优化点
  → 未达 target_speedup 且已能稳定运行
  → triton-auto-evolve，baseline 取上一步的 optimized_code.py
```

分界点：先用 `triton-op-generator` 拿到精度全过加一次系统优化，这一步是单任务内循环，成本较低；确认还有空间时再付 `triton-auto-evolve` 的跨轮成本。

### 链路 B：Ascend C 从无到有再到高性能

```
PyTorch model + cases
  → tilelang2ascendc-ops-generator，Phase 0-7，产出 model_new_ascendc.py 与 kernel/
  → performance.json 未达目标
  → ops-perf-evolution（基线内核模式，输入上一步的 output_dir）
```

顺序不能颠倒。`ops-perf-evolution` 不做新算子设计，只对已有 kernel 做变异。

### 链路 C：批量执行

| 插件 | 批量方式 |
|---|---|
| triton-op-generator | `run_benchmark_triton.sh --npu-list`，并行子进程；session 定位用工作目录路径精确过滤，不用 `ls -t` |
| tilelang2ascendc | `msprof_profile_run.sh --batch --base-dir ... --max-jobs 7 --device-start 1` |
| ops-perf-evolution | 支持多 NPU 并行批量进化任务，自动分配与汇总 |

一个任务对应一个工作目录或 `output_dir`，不要多个任务共用同一目录。

## 7. 排障

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 知识库克隆失败，或报告标注 `kb_unavailable` | 断网或代理不通 | 用 `CANNBOT_SKIP_DEPENDENCY_REPOS=1` 安装，以骨架模式运行 |
| ops-perf-evolution 启动即 exit 3 | 未配置知识仓 | 按 §1.3 配置 `CANNBOT_KNOWLEDGE_ROOT` |
| verify 或 benchmark exit 3 | 未执行 freeze | Phase 1 流程问题，终止重跑 |
| verify 或 benchmark exit 4 | 基线被改动 | 回滚 `{op_name}.py` 到冻结时状态 |
| benchmark exit 2，stderr 含 `[L1 闸门]` | 越过了 verify 闸门 | 按 verify 失败处理，读 `verify_result.json` 的 failures |
| 大批 AST 校验假失败 | `--kernel-path`、`--scripts-dir` 传了相对路径 | 改绝对路径 |
| Ascend C 精度不过（D 类） | kernel 数值与参考不一致 | 由 subagent 走 D 类流程，最多 12 轮；编排 agent 不得自行改代码 |
| TileLang 验证失败 | 设计表达问题或编译限制 | 有对照实验证据时可记录原因跳过，直接转译 AscendC |
| 进化停滞 | 轮数不足或方向未变 | 按 §4.7 的定向话术继续进化 |
| triton-auto-evolve 长时间不结束 | `naive` 不主动停止，未达 target 会跑满 `max_rounds` | 属预期行为，启动前按 `max_rounds` 估算成本 |

## 8. 上游文档口径差异

四份文档与对应 `AGENTS.md` 存在少量不一致，使用时以实际安装版本为准。

| 项 | 差异 |
|---|---|
| triton-op-generator 的 Phase 编号 | quickstart 写 Phase 0-7，`AGENTS.md` 正文列 Phase 0-6 并把 Phase 7 单列在末尾，实际为 0-7 |
| Phase 3 迭代上限 | `AGENTS.md` 状态变量写 `max_iterations = 10`，约束表写 5 次 |
| Phase 4 迭代上限 | 状态变量写 `max_opt_iterations = 50`，约束表写「优化点个数 + 1」 |
| IR 优化点编号 | `triton-latency-optimizer` 的 SKILL.md 定为 31，`AGENTS.md` 4.1 有一处写「命中点为 32（IR 分析）」 |
| ops-perf-evolution 目录名 | 常被误称为 `ascendc-ops-perf-evolution` |
| marketplace 源 | tilelang2ascendc 用 `cannbot-skills` 仓，ops-perf-evolution 用 `cannbot` 仓 |

## 附录：四个插件速查

| | triton-op-generator | tilelang2ascendc | ops-perf-evolution | triton-auto-evolve |
|---|---|---|---|---|
| DSL | Triton-Ascend | Ascend C | Ascend C | Triton-Ascend |
| 生成新算子 | 是 | 是 | 否 | 否 |
| 优化已有算子 | 是，单任务内 | 是，一轮 TileLang 性能迭代 | 是，多轮进化 | 是，多轮进化 |
| 输入 | torch `.py`，或 GPU kernel 加 `.pt` | `model.py` 与 cases | 基线 kernel 目录或 ops 仓路径 | 已有 Triton 算子目录 |
| 轮数上限 | Phase 3 与 Phase 4 内部循环 | A 类 5 次 / D 类 12 次 / REVIEW 3 轮 | `max_rounds`，默认 2 | `max_rounds`，默认 20 |
| 最终交付物 | `{op_name}_generated.py` 与 report、summary | `model_new_ascendc.py`、`kernel/`、`trace.md` | `evolution-report_*.html` | `{op_name}_final.py` |
| 知识闭环 | Phase 7 写 KB 约束卡 | Phase 7 写 KB runbooks | 读取策略 Playbook | `.claude/template/*.md` |
