# AgentLoop Training Optimization

本分支是独立的大模型训练系统优化工程，支持 Megatron 与 FSDP/FSDP2（不绑定具体 FSDP 框架）。算子优化是第一项落地能力，不是整个工程的边界。

## 快速开始（推荐）

和 CANNBot 一样采用项目级 init + Agent 对话入口。默认 Claude Code，支持 Codex、OpenCode 的项目 Skill 目录。不修改全局配置，不覆盖项目已有同名 Skill，不自动拉取 CANNBot 或安装 NPU 环境。

```bash
# 在已经具备 CANNBot 依赖的模型项目中执行
cd /path/to/training-project
bash /path/to/ops/AgentLoop/init.sh project claude
# 或指定项目目录：
# bash /path/to/ops/AgentLoop/init.sh project codex /path/to/model
claude
```

在 Agent 中输入：

> 使用 agentloop-train-opt，基于当前项目的训练启动脚本，采集 Level1 Profiling，识别热点，按真实实现语言调用 CANNBot Triton/Ascend C 优化算子，验证精度和性能，并报告整网收益。

Agent 负责按 Skill 编排内部脚本；用户无需手工拼接多个 Python 命令。实际启动参数、真实源码与 NPU 环境不能猜测。双插件仍需在相同 Agent 环境中提前按上游方式安装；init 仅提示项目本地依赖检测结果，不能验证 marketplace/global 安装或 NPU 可用性。现有 Triton 训练链路需要真实 latency Index.md（配置 latency_index）。当前生成/融合的自动集成与推理完整回归编排仍未闭合，不承诺一句话已能全自动完成所有优化。高级 Python 入口保留用于调试。

初始化脚本的本地 smoke test：`bash scripts/test_init.sh`。

## 当前开发优先级：P0 算子闭环

训练与推理分别维护独立工程，但第一阶段验收目标一致：真实 Profiling → 热点/融合机会 → 原始实现语言路由 → CANNBot Triton / Ascend C → 真实算子性能与精度验证 → 隔离集成 → 整网回归与回滚。优先跑通原生 Triton 热点、Ascend C/CANN 原生热点、同语言 Vector 融合及 MatMul 的实际形态优化。没有真实 NPU 联合测试与重复回归，不宣称稳定。其他系统优化只保留规划，不在 P0 扩展。

## 能力规划

| 优化领域 | 当前状态 |
| --- | --- |
| 训练拉起、固定 Step Profiling、性能/精度基线 | 已有接口，待实机验收 |
| Vector 融合机会、MatMul MFU、Triton / Ascend C 路由 | 已有分析与任务生成，融合/Ascend C 自动集成待实现 |
| 已有 Triton 算子优化、补丁集成与整网回归 | 已有编排，待实机验收 |
| 通信与计算通信掩盖（TP/DP/PP/EP/FSDP） | 规划 |
| 并行策略、流水线调度、显存及重计算 | 规划 |
| 集群拓扑、跨节点带宽、负载不均、规模扩展与快恢复 | 规划 |

训练入口负责固定训练负载、采集与验收；后续按优化领域扩展独立 Skill，不把通信、集群等任务强行交给算子生成器。训练与推理分别维护独立分支，不共享运行时或整网验收入口。本分支不包含推理服务、量化部署、KVCache 池化、PD 调度等推理专项能力。

## 目录

- `plugins/ops/engine/model_hotspot.py`：整网单批编排，prepare/tune/finish/run。
- `plugins/ops/engine/model_metrics.py`、`training_benchmark.py`：训练指标与验收。
- `plugins/ops/agents/profile-driven-opt-agent/`：Agent 入口定义。
- `skills/ops-skills/triton/profile-bottleneck-analyzer/`：热点识别。
- `skills/ops-skills/triton/triton-op-source-assembler/`：真实源码装配。
- `skills/ops-skills/triton/profile-driven-triton-opt/`：调优编排、产物契约、可选重构与整网接入文档。
- `skills/ops-skills/vendored-triton-op/`：CANNBot 调优依赖（本地安装，不提交生成的副本）。
- `scripts/vendor-triton-op-generator.sh`：上游安装与版本记录。

## 1. 安装

在具备 Python 3、Git、Node.js、实际 NPU 训练环境及 Agent CLI 的环境中：

```bash
cd AgentLoop
# 将 3 个本地 skill 和 6 个上游调优 skill 链接到实际训练仓的 Agent skill 目录
bash install.sh /path/to/model claude
# Codex 可改为 codex；也可只安装依赖，不投影：bash install.sh
```

安装器默认使用已有完整 vendor；不存在时调用原 vendor 脚本。可用 CANN_BOT_TAG、CANN_BOT_ROOT、CANN_BOT_SKILLS 固定上游版本。vendor 安装生成的版本信息见 VENDORED_TRITON.json；仓库已有清单仅表示历史快照，并不表示依赖已安装。若需要知识库，运行调优 Agent 前将 CANNBOT_KNOWLEDGE_ROOT 指向 cannbot-knowledge 的 knowledge 目录（见 vendor README）。安装不会覆盖目标仓已有同名 skill；冲突时请人工确认版本。

## 2. 配置训练

新增统一 Megatron / FSDP 训练接入，默认 20 step、前 10 step 训练预热、第 11 step profiler 预热、第 12–14 step 正式采集。FSDP 采用公共 TrainingHook，不区分 VeOmni / TorchTitan；Megatron 可使用原生 profiler 或同一个 Hook。详细接入见 [训练后端说明](skills/ops-skills/triton/profile-driven-triton-opt/docs/training-backend.md)，配置示例：[FSDP](skills/ops-skills/triton/profile-driven-triton-opt/docs/model-pass.fsdp.example.json) / [Megatron](skills/ops-skills/triton/profile-driven-triton-opt/docs/model-pass.megatron.example.json)。

以下为旧版分阶段命令配置（仍支持）：

复制 [model-pass.example.json](skills/ops-skills/triton/profile-driven-triton-opt/docs/model-pass.example.json) 至模型自己的配置目录，填写实际路径、profile_command、benchmark_command、source_map、workload。若使用自动 CLI 模式还必须配置 tune_command（支持 {request} 和 {prompt}）。

训练脚本负责真实开启/关闭框架 profiler，固定 checkpoint、数据、随机种子、并行策略、batch 和测量窗口。多 rank 产物使用 profile_subdir 指定统计 rank。source_map 需要提供真实源码，未匹配则跳过。具体环境变量及指标格式见 [整网单批接入说明](skills/ops-skills/triton/profile-driven-triton-opt/docs/model-hotspot-pass.md)。

## 3. 执行

```bash
python3 plugins/ops/engine/model_hotspot.py run --config /path/to/model-pass.json --out ./outputs
```

或分阶段：

```bash
python3 plugins/ops/engine/model_hotspot.py prepare --config /path/to/model-pass.json --out ./outputs
# 使用输出的 RUN_DIR，逐个让 Agent 执行 operators/op_*/prompt.txt
# 如果已配置 tune_command，也可执行：
python3 plugins/ops/engine/model_hotspot.py tune --run ./outputs/model_opt_xxx
python3 plugins/ops/engine/model_hotspot.py finish --run ./outputs/model_opt_xxx
```

已有 profile.zip 时可直接让 Agent 使用 profile-driven-triton-opt 的算子级入口；该入口不做整网回归。

## 4. 验收

热点按稳态 kernel 累计耗时占比排序（默认 3%，不是墙钟关键路径占比）。仅 strong 且有真实源码的算子进入调优。单算子默认目标 2x，真实用户源码为性能分母，官方 verifier/benchmark 为证据。合格补丁先试应用，再统一回归，训练异常、loss 超差或性能退化时回滚。整网只采集一次热点，无外层多轮寻优和 MFU 目标。

输出 model_report.md / model_report.json 及独立算子任务目录。未实测不填写整网收益。

## 5. 测试

```bash
python3 -m unittest discover -s plugins/ops/engine/tests -p 'test_*.py'
python3 -m pytest -q skills/ops-skills/triton/profile-bottleneck-analyzer/tests
# 安装 vendor 后可额外执行可选重构契约测试：
python3 skills/ops-skills/triton/profile-driven-triton-opt/scripts/test_restructure_contract.py
```

前两项为本地逻辑/合成数据测试，不代表真实 NPU 端到端性能验收。

新增训练采集与优化分流：Level1 / PipeUtilization / Text / record_shapes；MatMul MFU 与连续 Vector 融合候选见 [采集与分流契约](skills/ops-skills/triton/profile-driven-triton-opt/docs/training-backend.md#cannbot-profiling-采集契约与优化分流)。

算子优化不限定 Triton：按实际 dispatch 路由至 Triton 或 Ascend C，详见 [算子实现语言路由与融合原则](skills/ops-skills/triton/profile-driven-triton-opt/docs/operator-routing.md)。现有 Triton 自动集成路径保留，Ascend C 与融合生成任务单独交付并待图级验收。

## CANNBot 双插件执行入口（P0）

先使用对应的 Profiling 分析/prepare 生成任务，核实真实 dispatch、源实现、reference、输入形态以及融合数据依赖，再明确选取任务执行。安装真实的 CANNBot `triton-op-generator` / `tilelang2ascendc-ops-generator`，按当前上游版本配置 Agent CLI；执行器不猜测 CLI 参数，也不将退出码当作验收。

```json
{
  "commands": {
    "triton-op-generator": ["/path/to/your-triton-agent-wrapper", "{request}", "{prompt_file}"],
    "tilelang2ascendc-ops-generator": ["/path/to/your-ascendc-agent-wrapper", "{request}", "{prompt_file}"]
  },
  "timeout_seconds": 7200
}
```

上面的 wrapper 路径是待替换示例，非仓库自带命令。可使用真实 Agent CLI 的 argv；支持 {request}、{prompt}、{prompt_file}、{plugin}、{output_dir}、{model_root}。在任务目录执行，所以 Agent 需在该执行环境可发现所需插件。应使用隔离工作环境/只读原始模型源码，不授权任务直接修改原模型。

```bash
python3 plugins/ops/engine/dispatch_operator_tasks.py \
  --run /path/to/run --config /path/to/cannbot-cli.json --task task_000
```

任务来源：`generation_requests/task_*/request.json`。可重复 --task 逐个串行执行。每次执行生成独立 dispatch_*/command.log、dispatch.json，成功退出仅标记 CLI_COMPLETED_UNVERIFIED；仍需独立校验精度、性能及集成契约。该入口不会应用补丁，不证明实际插件执行正确，也不会自动回归。
