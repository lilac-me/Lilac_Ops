# AgentLoop 推理端到端性能优化设计（vLLM-Ascend，方案初版）

本分支从训练精简分支派生，保留可共享的 Profiling 解析、算子语言路由与 CANNBot 接入经验。当前仅提交设计，不声称已实现推理执行器或上游插件自动集成。训练分支与 main 不修改。

## 1. 上游能力与复用边界

CANNBot model-infer-optimize 公开资料显示两条流程：基础流程从模型适配、并行、KVCache/FA、融合、量化、图模式建立 baseline；SOTA 探索流程在 baseline 上进行场景构建、profiling、组件拆解、候选方案、实施及验收。原子 skills 包括 model-infer-profiling、model-infer-perf-breakdown、model-infer-fusion、model-infer-kvcache、model-infer-parallel-analysis/impl、model-infer-graph-mode、model-infer-multi-stream、model-infer-prefetch 等。

直接访问当前 cannbot master 下的插件目录及 workflow 未成功；上述信息来自当前公开功能清单与迁移前公开文档，不代表已核对最新源文件。落地前必须在实际 checkout 中核对入口、输入输出契约、版本及安装方式。

参考：
- https://gitcode.com/cann/cannbot/tree/master/plugins-official/model-infer-optimize
- https://gitcode.com/cann/cannbot-skills
- https://gitcode.com/LiJianhao2/cannbot-skills-2026-10-01-002-contrib/blob/master/plugins-official/model-infer-optimize/AGENTS.md
- https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html
- https://docs.vllm.ai/en/stable/benchmarking/cli/

原 main 还已有 plugins/inference-model-optimize/workflows/optimize-pipeline.yaml 以及 vllm-ascend-serving、vllm-ascend-benchmark、ascend-profiling-collection 等 Skill。先复核依赖与脚本契约，选择性移入服务管理、压测及采集能力；不直接移入整套多 Agent FSM。CANNBot 负责领域分析与实现，AgentLoop 作为唯一外层编排 owner，不嵌套两套全局优化循环。

## 2. 第一版范围

仅支持已跑通的 vLLM-Ascend 服务（单机、固定 TP/DP/EP、固定模型版本与精度）；默认不改并行、量化、图模式和模型结构。原生 CANN 与 Triton 热点采用训练分支一致的实现语言原则；第一版按固定候选单批优化、整服务一次回归，后续才考虑多轮 SOTA、PD 分离、系统级调度等。

流程：启动原服务 → 健康及精度基线 → 无 profiler 压测 → 独立 profiling 场景 → 热点/组件分析 → 按目标和实现语言路由 → CANNBot 生成/优化 → 独立候选精度与性能验证 → 隔离集成 → 重启相同服务 → 同 workload 回归 → 保留或回滚。

## 3. vLLM-Ascend 接入

建议 InferenceBackend 提供 start/health/warmup/benchmark/start_profile/stop_profile/collect/stop，首个实现 VllmAscendBackend。服务启动使用用户原始 argv 与固定环境；由版本适配层核对 --profiler-config、/start_profile、/stop_profile、解析方式。当前上游文档的 vLLM 新版本采用 --profiler-config，旧版环境变量仅在版本确认后使用。不得默认打开面向公网的 profiler API；实验服务限本机或受控网络。

profiling 不使用训练的 optimizer step：以请求阶段、worker/model execution iteration 及真实 scheduler 元数据为边界。独立采集 Prefill（首次/分块）、Decode（稳定生成窗口）和需要时 Mixed；head/middle/tail 是采集时序而非真实阶段标签，chunked prefill、continuous batching、spec decode 下必须从实际请求和调度事件确认。建议预热至服务加载、图捕获、算子编译完成，再发固定 workload；decode 初始可采 10–20 个 worker iteration（需按实际输出长度、batch 和版本配置校准）。

采集契约：Level1 + Text + PipeUtilization，record_shapes=True、with_stack=True；CPU+NPU，按需关闭 memory/modules。vLLM profiler CLI 可启用采集，不保证全部 torch_npu 实验参数实际生效：检查安装版本的 worker profiler 构造路径；需要时使用最小适配，不改全局 torch_npu。必须保留 kernel_details.csv、trace_view.json、可用 operator_details.csv、step_trace_time.csv（若版本或场景不提供则使用真实 worker iteration 证据）、rank/role、实际 tokens/batch/phase、流水指标和源码关联证据。每个 rank 独立检查，不能沿用训练单 rank 唯一 CSV 检查。禁止把缺失字段填成零或伪造阶段标签。profile 测量不作为服务 benchmark。

## 4. 优化路由

| 计算/系统类型 | 优先目标 | 工具 |
| --- | --- | --- |
| Prefill MatMul/GMM | 实际形态 MFU、latency | Triton 或 Ascend C 对应插件 |
| Decode MatMul/GMM | latency、有效带宽、小 M 效率；仅在计算受限时以 MFU 为主 | 同上 |
| 连续 Vector | 同语言合法融合、中间访存及 launch 降低 | 同上 |
| Attention/KVCache | KV 访存、layout、paged attention、实际长度、latency | 先复用 model-infer-kvcache / 现有 fused kernel，缺失时按语言生成 |
| Host/调度/通信 | critical-path、排队、bubble、Overlap | model-infer 专项 Skill；不强行派给算子生成器 |

实际 Triton 使用 triton-op-generator；确认 Ascend C 或 CANN 原生 compute dispatch，优先 tilelang2ascendc-ops-generator（内建源码不可见则基于真实 reference 生成等价实现）；混合语言默认拆分，跨语言须另行证明收益及可行性；未知实现先回查 dispatch。Core 类型不等于语言。相邻 Vector 不等于合法融合，仍要检查数据依赖、别名、外部消费者、layout、KV cache 原地写入、量化尺度、图捕获兼容和跨请求隔离。只优化当前生产形态，不以极限大 M 的 MFU 代替真实小 M 性能。算子性能分母为真实原始实现/完整序列，精度 reference 单独维护。

## 5. 基线与验收

固定 checkpoint/tokenizer/model revision、vLLM/vLLM-Ascend/torch_npu/CANN 版本、设备、并行、quant/graph/KV cache、scheduler、采样配置、请求集、真实 ISL/OSL、到达率与并发、随机种子、prefix cache 冷暖策略。服务预热后至少 3 次无 profiler 压测作为建议默认值；保存逐请求及汇总，比较相同场景的中位数及尾延迟，验证成功数和实际 token 数，不允许通过少生成 token 或改变 workload 获得虚假收益。

Prefill：TTFT、输入 tokens/s；Decode：TPOT/ITL、输出 tokens/s；混合服务：吞吐、p50/p95/p99 TTFT/TPOT/E2E、request goodput（按用户 SLO）、失败率、HBM/KV cache。固定精度对照集：greedy token/logit 或 logprob（按约定容差）、任务准确率（需要时）；涉及 cache/attention 或 fusion 时做变长、不同 batch、跨请求隔离和图模式回归。没有输出长度或可比较的精度证据不接受。不能以单算子收益代替端到端验收。补丁在隔离 worktree/独立环境构建，保留 enable 开关与原配置，重启加载后验证，失败回滚。

## 6. 开发计划（未实现）

1. 在实际 cannbot checkout 核对 model-infer-optimize SOTA workflow 和 atomic skills 契约，检查其是否支持 vLLM-Ascend 服务而不只适配独立 PyTorch infer.sh。
2. 从 main 选择性复用 serving/benchmark/profiling 脚本，形成 vLLM adapter + 统一 workload/metric contract。
3. 通用化 profile contract、热点分析及 operator_routing；给推理增加 phase/rank/shape 权重和组件归因，保持训练接口兼容。
4. 复用 CANNBot model-infer-profiling/perf-breakdown 与双算子插件；补全 Ascend C 与 fusion 的实际执行、前向/状态精度、图级集成门禁。
5. 选已可运行的 vLLM-Ascend 模型执行 CPU 模拟测试及 NPU 实机验证。第一版不自动开展生产服务改造。

建议代码布局：plugins/ops/engine/inference_backend/{base.py,vllm_ascend.py,metrics.py}、infer_hotspot.py；公共 analyzer/router 继续复用；具体文件及调用待核对上游后再实现。
