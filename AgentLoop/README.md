# AgentLoop Inference Optimization

独立的大模型推理系统优化工程，首个后端为 vLLM-Ascend。算子仅是一个优化场景；不依赖训练分支、训练 Hook、Megatron/FSDP 或 Loss/Step Time 验收。

## 快速开始（推荐）

实机测试步骤及排障：[实机测试教程](docs/real-machine-test.md)。 正式整网性能验收优先采用 AISBench，现有 vLLM benchmark 用于采集与冒烟；自动结果解析与回归编排待实现。

和 CANNBot 一样采用项目级 init + Agent 对话入口。默认 Claude Code，支持 Codex、OpenCode 的项目 Skill 目录。不修改全局配置，不覆盖项目已有同名 Skill，不自动拉取 CANNBot 或安装 NPU 环境。

```bash
# 在已经具备 CANNBot 依赖的模型项目中执行
cd /path/to/vllm-ascend
bash /path/to/ops/AgentLoop/init.sh project claude
# 或指定项目目录：
# bash /path/to/ops/AgentLoop/init.sh project codex /path/to/model
claude
```

在 Agent 中输入：

> 使用 agentloop-infer-opt，基于当前项目的vLLM-Ascend 服务启动脚本，采集 Level1 Profiling，识别热点，按真实实现语言调用 CANNBot Triton/Ascend C 优化算子，验证精度和性能，并报告整网收益。

Agent 负责按 Skill 编排内部脚本；用户无需手工拼接多个 Python 命令。实际启动参数、真实源码与 NPU 环境不能猜测。双插件仍需在相同 Agent 环境中提前按上游方式安装；init 仅提示项目本地依赖检测结果，不能验证 marketplace/global 安装或 NPU 可用性。现有 Triton 训练链路需要真实 latency Index.md（配置 latency_index）。当前生成/融合的自动集成与推理完整回归编排仍未闭合，不承诺一句话已能全自动完成所有优化。高级 Python 入口保留用于调试。

初始化脚本的本地 smoke test：`bash scripts/test_init.sh`。

## 当前开发优先级：P0 算子闭环

训练与推理分别维护独立工程，但第一阶段验收目标一致：真实 Profiling → 热点/融合机会 → 原始实现语言路由 → CANNBot Triton / Ascend C → 真实算子性能与精度验证 → 隔离集成 → 整网回归与回滚。优先跑通原生 Triton 热点、Ascend C/CANN 原生热点、同语言 Vector 融合及 MatMul 的实际形态优化。没有真实 NPU 联合测试与重复回归，不宣称稳定。其他系统优化只保留规划，不在 P0 扩展。

## 能力规划

| 场景 | 当前状态 |
| --- | --- |
| vLLM-Ascend 服务启动/停止/健康检查 | 从原 main 迁入 |
| 在线压测、参数寻优、分段 profiling | 从原 main 迁入，待实机核验 |
| 单 rank profile 数据门禁、热点与语言路由 | 已提供独立分析入口，未实机验证 |
| Vector 融合、MatMul MFU/Latency、Triton/Ascend C | 分析与任务描述；已提供显式 CLI 调度入口，待实机验证及自动集成 |
| 量化（W8A8/W4A8/FP8 等）、精度验收 | 规划 |
| KVCache/权重/资源池化、Attention/Cache 访存 | 规划 |
| PD 分离、跨实例 KV 传输与调度 | 规划 |
| 并行、通信、图模式、连续批处理与多流 | 规划，已有参数寻优基础 |

CANNBot model-infer-optimize 负责推理领域专项优化；triton-op-generator 与 tilelang2ascendc-ops-generator 负责算子生成/优化。AgentLoop 负责推理实验编排和服务回归。上游当前目录尚未直接核验；不声称已完成该插件执行接入。

## 入口

所有命令从 AgentLoop 根执行。先阅读对应 Skill 并执行 --help 核对实际环境、参数和版本。迁移的脚本支持 SSH docker/direct/conda，需用户提供真实服务环境与工作负载，不自动操作生产服务。

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py --help
python3 skills/inference-model-skills/performance-optimization/vllm-ascend-benchmark/scripts/bench_run.py --help
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py --help
```

压测固定模型、版本、拓扑、图模式、量化、输入输出长度、并发/到达率及缓存策略。正式 baseline 与 regression 均关闭 profiling。采集完成后把远端真实 profile 同步到分析环境；每个 rank 独立执行：

```bash
python3 plugins/ops/engine/infer_profile_analyze.py \
  --profile /path/to/copied_rank_profile \
  --rank 0 --phase unknown \
  --model-root /path/to/vllm-ascend \
  --out ./outputs/run_001_rank0
```

若已有真实 scheduler/worker 阶段证据，可以增加 --phase prefill/decode/mixed 和 --phase-evidence /path/to/evidence。分析器不自动证明阶段标签。使用 --source-map 传入真实 kernel→源码映射。生成 profile_contract.json、candidates.json、opportunities.json、operator_routes.json 及 requests/，分析入口不自动启动优化插件或替换模型；下方可显式执行已核定的任务。

采集默认开启 stack；必须另在实际 vLLM-Ascend worker 中确认 Level1 + PipeUtilization + record_shapes + Text。缺字段则失败，不伪造。单 rank 分析不是整服务关键路径分析。完整服务启动—采集—优化—回归的统一编排仍待建设。

## 设计与扩展

[推理系统设计](docs/inference-design.md)；[优化场景与验收](docs/optimization-domains.md)。独立推理分支不保留训练入口、训练配置、训练测试或训练 Agent。仅复用可移植分析与实现语言路由。安装 CANNBot 插件遵循实际上游版本，不使用训练专用 vendor 安装器。

## 本地测试（尚未执行）

```bash
python3 -m pytest -q skills/ops-skills/triton/profile-bottleneck-analyzer/tests
python3 -m unittest discover -s skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/tests
```

这些是逻辑/合成测试，不能替代真实 NPU 服务验证。

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

任务来源：`requests/task_*/request.json`。可重复 --task 逐个串行执行。每次执行生成独立 dispatch_*/command.log、dispatch.json，成功退出仅标记 CLI_COMPLETED_UNVERIFIED；仍需独立校验精度、性能及集成契约。该入口不会应用补丁，不证明实际插件执行正确，也不会自动回归。
