# AgentLoop Training Optimization

只处理大模型训练系统优化（Megatron/FSDP），当前落地场景是 Profiling 驱动的算子优化。通信、并行、调度、显存和集群是后续独立优化领域，不属于算子生成任务。不要接入推理服务或推理工作流。

- 整网任务：先读 README.md 和 skills/ops-skills/triton/profile-driven-triton-opt/docs/model-hotspot-pass.md，使用 plugins/ops/engine/model_hotspot.py run 或 prepare → 逐个 prompt.txt → finish。
- 已有 Profiling：使用 profile-driven-triton-opt 和 analyzer/source assembler。先核对 Level1/Shape/PipeUtilization 产物与真实 dispatch。
- 语言路由：先读 skills/ops-skills/triton/profile-driven-triton-opt/docs/operator-routing.md；Triton 现有算子使用 CANNBot verifier、latency/simulator/MFU；Ascend C 或同语言融合使用独立 generation_requests，需确认真实 reference 和前反向精度，不能走旧 Triton 单文件集成契约。
- 弱候选、混合/未知实现先核实；不要伪造源码、benchmark、精度或性能结果；仅修改隔离副本。
- 整网只在 finish 统一集成与回归，失败回滚。保持原有门禁和测试；无自动二次热点采集。
