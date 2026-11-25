---
name: agentloop-train-opt
description: Optimize hot operators of a Megatron or FSDP training model using profiling and CANNBot Triton/Ascend C skills.
---

# 训练算子优化

1. 从本 SKILL.md 的实际源路径定位 AgentLoop 根（本文件位于 skills/agentloop-train-opt/）；读取 README.md、训练后端及 operator-routing 说明。
2. 使用用户真实训练命令，优先复用原生 profiler；需要时对接 TrainingHook。默认 20 个完整训练 step，10 个预热，1 个 profiler warmup，3 个 active；Level1、Text、PipeUtilization、record_shapes、with_stack。固定 checkpoint/数据/seed/并行/精度。
3. Agent 负责构建 model-pass 配置并执行 model_hotspot.py prepare；真实源码映射和 profiler 导出不能猜测。若已有 profile 可走算子级分析。现有 Triton 调优须配置实际安装的 triton-latency-optimizer/references/Index.md 的绝对路径（latency_index）；若需原 vendor 依赖，使用旧 install.sh 显式安装，不自动克隆。
4. 读取本轮 operators/*/prompt.txt，调用已安装 CANNBot Skill 执行；已有 Triton 路径通过 tune 或当前 Agent 完成。已确认 Ascend C/融合任务读取 generation_requests/task_*/request.json 与 prompt.txt，当前 Agent 直接调用相应插件，或可选 dispatch_operator_tasks.py；不得伪造 upstream CLI。
5. 真实前向/反向精度与性能验收；原 Triton 单文件合格补丁走 finish，统一整网对照并回滚。Ascend C/融合多文件集成门禁尚未完成，单独交付待验收补丁，不宣称整网优化成功。

本 Skill 是用户唯一推荐的交互入口；不要要求用户逐条运行内部 Python 命令。先解析用户给出的真实项目、模型、启动命令、工作负载、NPU 环境及优化目标，缺失的必要信息才询问。只处理 P0 算子优化，不扩展其他系统优化。安装的 CANNBot 插件必须能被当前 Agent 发现；若缺少，说明需要先按对应上游 init.sh 安装。保持原始算子实现语言，Triton 使用 triton-op-generator，Ascend C/CANN Native 使用 tilelang2ascendc-ops-generator；未知或混合语言先核实或拆分，连续 Vector 融合须确认真实数据依赖。MatMul 按真实形态分析 MFU，性能基线必须是原实际实现或完整原序列，不能使用不等价的参考实现。不得编造结果或自动修改生产源码。



## 端到端收益硬门禁

单算子收益不是交付目标；只有同真实 workload 下整网主指标出现超过噪声且达到预设最低门槛的正收益，精度及保护指标同时通过，才计入有效优化。无整网收益、退化或未验证均不得作为优化成功交付。详见 AgentLoop 根目录 docs/e2e-acceptance.md。
