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


## 单算子事务（当前执行原则）

每轮从当前已保留的模型版本出发，只选一个热点算子或一个不可拆分的融合算子组。AgentLoop 不干预 CANNBot 内部每一次代码迭代；由 CANNBot 自主执行分析、改写、精度与性能测量，按其退出条件交付最佳合格实现。整网只对该实现做独立对照；无需达到固定的单算子 2×/MFU 目标，只要满足单算子交付契约且实际整网收益超过预设噪声门槛、精度不劣化即可保留。失败或无收益回滚；下一候选必须以最新保留版本重新建立基线，禁止把多个算子的收益混在一起归因。第一版不自动循环所有候选。训练配置可选 target_operator（精确候选名称；融合用 operator_routes.json 的完整 name）；不指定则选择第一个可自动调优的 Triton 候选，其余标记 DEFERRED。生成/融合任务需要明确选定，当前仍需独立集成与回归，不能走现有 Triton finish 门禁。