---
name: agentloop-optimization-principles
description: Mandatory system-level decision policy for AgentLoop optimization. Load before selecting any hotspot or invoking CANNBot. Analyze the actual critical path and overlap, forecast end-to-end gain, select one optimization unit, and accept only measured end-to-end benefit without precision or guardrail regression. Applies to training, inference and future system optimization domains.
---

# AgentLoop 优化总原则

**目标是实际端到端收益，而不是单算子加速、MFU、Kernel 数量或优化次数。** 本 Skill 是训练、推理和后续专项优化的上层决策规则；领域 Skill 决定具体工具和实现。当前 P0 仅做算子优化，不因此扩大范围。

## 1. 固定真实问题与验收目标（先于优化）

- 固定当前已保留的模型/服务版本、checkpoint/权重、实际输入与输出工作量、硬件/拓扑、软件版本、精度、并行和运行配置。明确定义一个整网主指标、精度约束及保护指标。
- 训练优先完整 Step Time / 实际 Token 吞吐；推理优先 AISBench 的 Output Token Throughput 或 TTFT/TPOT/Goodput，按真实场景选定主目标并锁定 SLO、尾延迟、请求成功率、精度和资源约束。
- 无 profiler 的真实整网 benchmark 建立 baseline；预热一致，记录噪声/波动；在看到优化结果前确定最小可接受正收益和保护指标容差。初筛可单次，正式稳定性需至少三组独立前后对照。不要事后修改验收口径。
- 单独采集 Level1/Text/PipeUtilization/record_shapes 及 trace。确认采集窗口代表实际生产形态，profiling 运行不能代替 baseline。

## 2. 热点不等于关键路径（先诊断再动代码）

从 kernel_details.csv 获得真实 kernel/name、调用次数、shape/dtype、累计时间和流水指标；从 trace_view.json、实际调度/Host-Device 关联、stream/event 依赖及同步点分析端到端关键路径。

- 识别真正阻塞下游完成的算子/子图，考虑多 stream 重叠、计算与通信掩盖、Host launch/等待、PP stage 或慢 rank、FSDP prefetch、同步边界，以及推理 scheduler、batch、Prefill/Decode/Mixed 等。
- 同一时段的重叠仅说明可能被掩盖，不单独证明因果关系；同一 stream 相邻也不证明数据依赖。需要真实依赖、同步等待、关键完成时间和 rank/phase 证据。关键路径可能随优化转移到其他 rank/stream。
- 将候选分为 CRITICAL（证据支持关键路径）、PARTIAL（仅部分暴露）、HIDDEN（证据支持无可观测关键路径贡献）、UNKNOWN（不足以判断）。注明证据文件、rank/phase、step/iteration、时间区间与判断依据。HIDDEN 本轮不优化；UNKNOWN 先补证据，可进行明确标记的有限诊断实验，不宣称收益或自动排序为关键路径。
- 不允许以 kernel 累计耗时/全部 kernel 耗时之和作为整网时间占比；不同 stream、rank 的重叠不能重复计算。单 rank trace 无法证明跨 rank 整网关键路径时，应降低置信度而非假定其它 rank 相同。

## 3. 收益预告（调用 CANNBot 前必须记录）

设 T 为同一实际 workload 的原端到端时间；p 为选定算子可缩短的有效关键路径耗时占比（不是 profiler kernel ratio）；s 为预估单算子加速比（原耗时 / 新耗时）。

在其它执行时间及依赖关系不变的简化模型下：

```text
T_new_est = T * (1 - p + p/s)
time_reduction_est = p * (1 - 1/s)
e2e_speedup_est = 1 / (1 - p + p/s)
```

吞吐提升仅在实际工作量固定、吞吐与该时间口径互为倒数时，可估为 e2e_speedup_est - 1；不能直接套用到服务 TTFT/TPOT/尾延迟、动态 batch 或 Goodput。推理按真实请求阶段和调度瓶颈分别估算，不能把 device step 缩短等同请求延迟缩短。

- p 与 s 必须注明来源/假设。优先使用真实 trace 的暴露时间、依赖关系和可比单算子测量；必要时给保守/乐观范围，未知填 unknown，不编造数字。部分 overlap 时暴露贡献按真实依赖推断，不简单相减；完全掩盖时不能假定有收益。
- p 是该简化模型下的耗时降低上界；只有在固定依赖/工作量、其它开销不变的假设下成立。若可证明该上界仍低于预设收益门槛，跳过候选，不浪费 CANNBot 迭代资源。
- 结合收益潜力、置信度、优化成本、风险和真实源码/参考实现可得性排序；选择最值得做的一个，而非单纯最大累计 kernel 耗时或最大单算子 speedup。范围跨越门槛时标记不确定并优先低成本验证。
- 预告是预测，不是实测验收。模型中存在调度变化、关键路径切换、缓存和资源竞争时，预测可能失效。

例（纯示意）：若实际关键路径暴露占比 10%、预计单算子加速 2x，则整网时间预计下降 5%、同工作量理论吞吐提升约 5.26%；若实际完全被掩盖，则当前模型下无直接时间收益。不要将示例数值当作真实测量。

## 4. 每轮仅选择一个优化单元

在调用 CANNBot 或执行优化之前，将决策记录为本轮 `optimization-decision.md`（见 [轻量记录模板](references/decision-record.md)）。记录 baseline、主指标/门槛、候选排序、trace 证据、收益预估及选定对象；保留其它候选为 DEFERRED。已有分析阶段可以列出多个候选，但不能一次修改多个独立算子。

- 一个优化单元 = 一个真实算子，或一个数据流合法且不可拆分的融合子图；核实 source/dispatch、真实 shape/dtype/stride/layout 与独立 reference。
- 已有 Triton → CANNBot triton-op-generator；已有 Ascend C/CANN 原生计算 → tilelang2ascendc-ops-generator（内建源不可见则等价替代，不冒称改原内建源码）；混合/未知先核实或拆分。Vector 考虑融合，MatMul 根据真实形态判断 MFU/Latency，不能强制所有算子同一目标。
- AgentLoop 只负责选择、交付和整网验收；CANNBot 在该单元内部自主多轮分析、优化、精度与性能验证，按其退出条件交付最佳合格版本。无需每次单算子迭代都重跑整网，也不以 2x/固定 MFU 为最终保留条件。
- 性能基线是真实原算子/完整原序列，不得用慢 reference 冒充原实现；融合需验证真实依赖、别名、外部消费者、训练前反向或推理状态/KVCache/跨请求隔离等约束。

## 5. 整网验收才决定保留

只将选定优化单元接回隔离模型/服务，关闭 profiling，在同一 workload、基线版本及约束下实测端到端。确认新实现实际执行，而不是 fallback/旧编译缓存；核对精度、主指标及保护指标。

- E2E_GAIN：真实正收益达到预设门槛且超出噪声，精度与保护指标通过，才保留。
- NO_E2E_GAIN：无收益或低于门槛；回滚/不合入，保留诊断证据。
- ACCURACY_FAILED / PERFORMANCE_REGRESSION：精度或保护指标失败；回滚。
- NOT_VERIFIED：没有完整真实整网对照、工作量不一致、未证明新实现执行或证据不足；不计入收益、不合入。

报告必须分开记录单算子实测、整网预测、整网实测与最终状态。下一个优化单元以最新保留版本重新建立 baseline 和关键路径分析，不重复归因旧收益，不将多个算子的孤立预测相加。系统优化未来也遵守相同规则，届时以一个可归因的方案为优化单元，不强行套用算子工具。

当前 Skill 是 Agent 决策规范，不是自动关键路径解析器或收益预测引擎。执行前必须读取并遵守；现有脚本无法自动证明所有 trace 依赖、预测及推理验收，缺证据应如实标记，不声称自动门禁已全部实现。
