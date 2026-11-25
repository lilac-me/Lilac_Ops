# AgentLoop Inference Optimization

只处理推理系统（首个 vLLM-Ascend 后端）。先读 README.md、docs/inference-design.md、docs/optimization-domains.md。不要调用训练 Step、Loss、Megatron/FSDP 或训练补丁集成流程。

服务、benchmark、profiling、参数寻优使用本分支迁入的对应 Skill。固定工作负载，采集和性能测试分离；明确 Prefill/Decode/Mixed 的真实调度证据，不把 head/middle/tail 当成阶段。逐 rank 校验 Level1/Shape/PipeUtilization。按原始实现语言路由算子，融合需真实数据流、KVCache、图模式及精度检查。量化、池化、PD、并行/调度等任务路由到相应推理领域，不强制转成算子任务。任何未完成的上游接入必须如实报告，不自动改生产服务。


P0 只建设算子优化闭环。其他优化领域仅保留规划，待真实 NPU 联合验证稳定后再扩展。已核定的双插件任务通过 plugins/ops/engine/dispatch_operator_tasks.py 显式执行；CLI 成功不等于精度/性能验收，禁止自动应用未经验证的产物。


## 端到端收益硬门禁

单算子收益不是交付目标；只有同真实 workload 下整网主指标出现超过噪声且达到预设最低门槛的正收益，精度及保护指标同时通过，才计入有效优化。无整网收益、退化或未验证均不得作为优化成功交付。正式验收使用 AISBench。详见 [端到端收益验收](docs/e2e-acceptance.md)。


## 单算子事务（当前执行原则）

每轮从当前已保留的服务版本出发，只选一个热点算子或一个不可拆分的融合算子组。CANNBot 自主多轮分析、代码修改、精度与性能测量，按其内层退出条件交付最佳合格实现；不要求每次单算子迭代都重启整网。然后只集成这个算子，在相同 AISBench workload 下进行整网对照。只要主指标真实正收益超过预设噪声门槛、精度及保护指标不劣化，就保留该版本；不强求固定单算子加速比或 MFU。失败或无收益则回滚。下一个算子必须从最新保留版本重新建立 baseline，不能同时混合多算子改动。显式 dispatch 入口每个 analysis run 只允许选择一个 task；重试仅可针对同一个 task。直接通过 Agent 调用插件时也须遵守此原则。当前服务级自动集成、AISBench 自动解析/回归仍未实现，先在隔离环境实测验收。