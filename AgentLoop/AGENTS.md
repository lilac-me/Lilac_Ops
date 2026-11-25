# AgentLoop Inference Optimization

只处理推理系统（首个 vLLM-Ascend 后端）。先读 README.md、docs/inference-design.md、docs/optimization-domains.md。不要调用训练 Step、Loss、Megatron/FSDP 或训练补丁集成流程。

服务、benchmark、profiling、参数寻优使用本分支迁入的对应 Skill。固定工作负载，采集和性能测试分离；明确 Prefill/Decode/Mixed 的真实调度证据，不把 head/middle/tail 当成阶段。逐 rank 校验 Level1/Shape/PipeUtilization。按原始实现语言路由算子，融合需真实数据流、KVCache、图模式及精度检查。量化、池化、PD、并行/调度等任务路由到相应推理领域，不强制转成算子任务。任何未完成的上游接入必须如实报告，不自动改生产服务。
