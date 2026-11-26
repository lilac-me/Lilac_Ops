# AgentLoop Training Optimization

## 强制上层优化原则

所有优化任务先读取 `skills/agentloop-optimization-principles/SKILL.md`（agentloop-optimization-principles），在选择算子、调用 CANNBot 之前完成真实关键路径诊断、端到端收益预告及唯一优化单元决策，按其模板记录 `optimization-decision.md`。缺关键路径证据不得把 kernel 累计耗时当整网占比；只有整网实测正收益且精度/保护指标通过才保留。该规则优先于领域 Skill 的默认候选排序和局部性能目标；不新增独立用户操作入口。

只处理大模型训练系统优化（Megatron/FSDP），当前落地场景是 Profiling 驱动的算子优化。通信、并行、调度、显存和集群是后续独立优化领域，不属于算子生成任务。不要接入推理服务或推理工作流。

- 整网任务：先读 README.md 和 skills/ops-skills/triton/profile-driven-triton-opt/docs/model-hotspot-pass.md，使用 plugins/ops/engine/model_hotspot.py run 或 prepare → 逐个 prompt.txt → finish。
- 已有 Profiling：使用 profile-driven-triton-opt 和 analyzer/source assembler。先核对 Level1/Shape/PipeUtilization 产物与真实 dispatch。
- 语言路由：先读 skills/ops-skills/triton/profile-driven-triton-opt/docs/operator-routing.md；Triton 现有算子使用 CANNBot verifier、latency/simulator/MFU；Ascend C 或同语言融合使用独立 generation_requests，需确认真实 reference 和前反向精度，不能走旧 Triton 单文件集成契约。
- 弱候选、混合/未知实现先核实；不要伪造源码、benchmark、精度或性能结果；仅修改隔离副本。
- 整网只在 finish 统一集成与回归，失败回滚。保持原有门禁和测试；无自动二次热点采集。


P0 只建设算子优化闭环。其他优化领域仅保留规划，待真实 NPU 联合验证稳定后再扩展。已核定的双插件任务通过 plugins/ops/engine/dispatch_operator_tasks.py 显式执行；CLI 成功不等于精度/性能验收，禁止自动应用未经验证的产物。


## 端到端收益硬门禁

单算子收益不是交付目标；只有同真实 workload 下整网主指标出现超过噪声且达到预设最低门槛的正收益，精度及保护指标同时通过，才计入有效优化。无整网收益、退化或未验证均不得作为优化成功交付。详见 [端到端收益验收](docs/e2e-acceptance.md)。
