# AgentLoop Inference Optimization

只处理推理系统（首个 vLLM-Ascend 后端）。先读 README.md、docs/inference-design.md、docs/optimization-domains.md。不要调用训练 Step、Loss、Megatron/FSDP 或训练补丁集成流程。

服务、benchmark、profiling、参数寻优使用本分支迁入的对应 Skill。固定工作负载，采集和性能测试分离；明确 Prefill/Decode/Mixed 的真实调度证据，不把 head/middle/tail 当成阶段。逐 rank 校验 Level1/Shape/PipeUtilization。按原始实现语言路由算子，融合需真实数据流、KVCache、图模式及精度检查。量化、池化、PD、并行/调度等任务路由到相应推理领域，不强制转成算子任务。任何未完成的上游接入必须如实报告，不自动改生产服务。


P0 只建设算子优化闭环。其他优化领域仅保留规划，待真实 NPU 联合验证稳定后再扩展。已核定的双插件任务通过 plugins/ops/engine/dispatch_operator_tasks.py 显式执行；CLI 成功不等于精度/性能验收，禁止自动应用未经验证的产物。


## 端到端收益硬门禁

单算子收益不是交付目标；只有同真实 workload 下整网主指标出现超过噪声且达到预设最低门槛的正收益，精度及保护指标同时通过，才计入有效优化。无整网收益、退化或未验证均不得作为优化成功交付。正式验收使用 AISBench。详见 [端到端收益验收](docs/e2e-acceptance.md)。
