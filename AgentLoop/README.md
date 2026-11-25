# AgentLoop Inference Optimization

独立的大模型推理系统优化工程，首个后端为 vLLM-Ascend。算子仅是一个优化场景；不依赖训练分支、训练 Hook、Megatron/FSDP 或 Loss/Step Time 验收。

## 能力规划

| 场景 | 当前状态 |
| --- | --- |
| vLLM-Ascend 服务启动/停止/健康检查 | 从原 main 迁入 |
| 在线压测、参数寻优、分段 profiling | 从原 main 迁入，待实机核验 |
| 单 rank profile 数据门禁、热点与语言路由 | 已提供独立分析入口，未实机验证 |
| Vector 融合、MatMul MFU/Latency、Triton/Ascend C | 分析与任务描述，待自动执行/集成 |
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

若已有真实 scheduler/worker 阶段证据，可以增加 --phase prefill/decode/mixed 和 --phase-evidence /path/to/evidence。分析器不自动证明阶段标签。使用 --source-map 传入真实 kernel→源码映射。生成 profile_contract.json、candidates.json、opportunities.json、operator_routes.json 及 requests/，不自动启动优化插件或替换模型。

采集默认开启 stack；必须另在实际 vLLM-Ascend worker 中确认 Level1 + PipeUtilization + record_shapes + Text。缺字段则失败，不伪造。单 rank 分析不是整服务关键路径分析。完整服务启动—采集—优化—回归的统一编排仍待建设。

## 设计与扩展

[推理系统设计](docs/inference-design.md)；[优化场景与验收](docs/optimization-domains.md)。独立推理分支不保留训练入口、训练配置、训练测试或训练 Agent。仅复用可移植分析与实现语言路由。安装 CANNBot 插件遵循实际上游版本，不使用训练专用 vendor 安装器。

## 本地测试（尚未执行）

```bash
python3 -m pytest -q skills/ops-skills/triton/profile-bottleneck-analyzer/tests
python3 -m unittest discover -s skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/tests
```

这些是逻辑/合成测试，不能替代真实 NPU 服务验证。
