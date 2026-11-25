---
name: agentloop-infer-opt
description: Optimize hot operators of a vLLM-Ascend inference model using profiling and CANNBot Triton/Ascend C skills.
---

# 推理算子优化

1. 从本 SKILL.md 的实际源路径定位 AgentLoop 根（本文件位于 skills/agentloop-infer-opt/）；阅读 README.md 和 docs/inference-design.md。
2. 复用 vllm-ascend-serving、vllm-ascend-benchmark、ascend-profiling-collection。固定模型、版本、并行、图模式、量化、真实 ISL/OSL、并发和请求集，先无 profiler 压测，再独立采集。目标环境、连接授权及必要参数必须明确。核对实际 worker profiler 的 Level1、Text、PipeUtilization、record_shapes、with_stack；缺失不能进入优化。
3. 把实际 rank 产物同步到分析环境，使用 infer_profile_analyze.py；真实调度证据支持 Prefill/Decode/Mixed 标签，否则 unknown。Prefill MatMul 偏 MFU，Decode 小 M 偏 latency/带宽，按真实形态决策。
4. 读取 requests/task_*/request.json 和 prompt.txt，当前 Agent 调用对应 CANNBot 插件，或可选 dispatch_operator_tasks.py 调度实际 Agent CLI。先核实源码/reference、动态 shape、KVCache 原地更新、跨请求隔离与图模式；不强制量化或 PD 改造。
5. 真实算子前向/状态精度与性能验收，生成隔离的可审查补丁；整服务自动补丁集成与回归编排尚未完成。使用相同 workload 的无 profiler 压测评估 TTFT/TPOT/throughput 和精度，不将 CLI 退出码或单算子收益当作服务验收。

本 Skill 是用户唯一推荐的交互入口；不要要求用户逐条运行内部 Python 命令。先解析用户给出的真实项目、模型、启动命令、工作负载、NPU 环境及优化目标，缺失的必要信息才询问。只处理 P0 算子优化，不扩展其他系统优化。安装的 CANNBot 插件必须能被当前 Agent 发现；若缺少，说明需要先按对应上游 init.sh 安装。保持原始算子实现语言，Triton 使用 triton-op-generator，Ascend C/CANN Native 使用 tilelang2ascendc-ops-generator；未知或混合语言先核实或拆分，连续 Vector 融合须确认真实数据依赖。MatMul 按真实形态分析 MFU，性能基线必须是原实际实现或完整原序列，不能使用不等价的参考实现。不得编造结果或自动修改生产源码。


## 正式推理验收（AISBench）

优先按 docs/real-machine-test.md 使用 AISBench 流式服务压测，统一 baseline/regression 的真实请求、token、并发、采样、缓存和无 profiler 环境。保存逐请求 CSV 与整网 JSON，对比 TTFT/TPOT/ITL/E2E、整服务输出 Token 吞吐及请求失败数。现有 vLLM benchmark 用于采集负载及冒烟，不与 AISBench 混作前后对照。尚无 AISBench 自动解析及整网回归执行器，必须如实报告实际验收状态。


## 端到端收益硬门禁

单算子收益不是交付目标；只有同真实 workload 下整网主指标出现超过噪声且达到预设最低门槛的正收益，精度及保护指标同时通过，才计入有效优化。无整网收益、退化或未验证均不得作为优化成功交付。正式验收使用 AISBench。详见 AgentLoop 根目录 docs/e2e-acceptance.md。
