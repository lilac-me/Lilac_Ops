# AgentLoop 训练算子优化实机教程

分支：agentloop-train-opt-lite。测试对象：已可运行的 Megatron 或 FSDP/FSDP2 模型。

## 0. 实机测试目标

第一轮只验证算子优化 P0 链路，不同时修改并行、量化、池化、PD、集群等系统策略。建议每个框架先选择一个已能正常运行的模型、一个有真实源码的热点 Triton 算子，再逐步验证 Ascend C/CANN 原生热点、Vector 融合及 MatMul 优化。先用少量可复现负载打通流程，再扩大范围。

开始前确认：
- 独占或空闲 NPU、可运行的原模型/权重与数据、已知有效的启动脚本、稳定的 CANN/torch/torch_npu 及实际框架版本。
- 当前 Agent（Claude Code / Codex / OpenCode）可正常运行；在同一项目/Agent 环境已按 CANNBot 上游文档安装 triton-op-generator 与 tilelang2ascendc-ops-generator。AgentLoop init 仅注册本地 Skill，不代装 CANNBot。
- 在独立测试项目或 worktree 操作，保留原始源码、版本、配置和 checkpoint；不要修改生产服务或覆盖共同输入。
- 网络、SSH、模型与输出目录实际可访问；日志中不要记录密码、Token 等凭据。

检查环境（在实际训练或推理环境内执行）：

```bash
npu-smi info
python3 -c 'import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__); print(torch.npu.is_available())'
git status --short
```

CANNBot 上游入口：[Triton](https://gitcode.com/cann/cannbot/tree/master/plugins-official/triton-op-generator)、[Ascend C](https://gitcode.com/cann/cannbot-skills/blob/master/plugins-community/tilelang2ascendc-ops-generator/quickstart.md)。按实际 checkout 的 init.sh --help 核对安装命令；不猜测上游 CLI 参数。确认 Agent 能发现插件及需要的 Skill；Triton 训练自动集成还需要实际 latency optimizer Index.md。

## 1. 初始化（推荐）

```bash
git clone --single-branch -b agentloop-train-opt-lite https://github.com/lilac-me/ops.git AgentLoop-Train
cd AgentLoop-Train/AgentLoop
bash scripts/test_init.sh
bash init.sh project claude /path/to/model
cd /path/to/model
claude
```

Codex / OpenCode 将 claude 替换为 codex / opencode。/path/to/model 替换为实际训练代码仓（不是权重目录）。启动 Agent 前先装好两个 CANNBot 插件。若现有项目 CANNBot 安装已完整，直接复用，不重复下载。

在 Agent 中输入以下任务（替换真实路径和后端）：

> 使用 agentloop-train-opt。训练后端为 FSDP（或 Megatron），使用当前项目的 bash scripts/train.sh 启动。先确认真实训练参数、原始 checkpoint、数据和设备；采集 Level1 + PipeUtilization + record_shapes 的 Profiling，默认运行 20 个完整 Step，前 10 个训练预热，第 11 个 profiler 预热，第 12–14 个采集。只选择一个真实源码明确的热点 Triton 算子做第一次优化；按原实现语言调用 CANNBot，验证单算子前反向精度和真实性能，合格后回接训练并对比 Step Time、Loss、Throughput；失败回滚。将结果及日志留在本轮独立目录。先展示实际参数和当前缺失项，禁止伪造数据。

## 2. 训练采集与对照

| 阶段 | 默认窗口 | 说明 |
| --- | --- | --- |
| Profiling | 20 步；12–14 active | Level1/Text/PipeUtilization、CPU+NPU、record_shapes=True、with_stack=True |
| Baseline | 20 步；11–20 统计 | 关闭 profiler，使用真实 step wall time |
| CANNBot | 独立算子任务 | 整网训练已退出并释放资源 |
| Regression | 20 步；11–20 统计 | 相同 checkpoint、数据、seed、精度、拓扑、实际 tokens |

固定窗口不代表一定稳态，JIT/编译较慢时扩大预热及 max_steps。FSDP Hook 在完整 optimizer iteration 结束调用一次，不能每个 microbatch 调用。Megatron native 用实际训练脚本消费 AGENTLOOP 环境变量。详细接入：[training-backend.md](../skills/ops-skills/triton/profile-driven-triton-opt/docs/training-backend.md)。

采集通过门禁后，检查 kernel_details.csv、step_trace_time.csv、trace_view.json、profile_contract.json 及 candidates/opportunities/operator_routes。若缺 shape/dtype、流水或时间线字段，先修采集，不继续优化。

## 3. 手动排障入口（通常由 Agent 执行）

以下从 AgentLoop 根执行，先复制对应模板到模型项目，并填写绝对路径、真实训练 argv、workload、source_map、实际 latency_index 和需要的 tune_command。旧 config 与新 training 配置不可混用。模板位于 skills/ops-skills/triton/profile-driven-triton-opt/docs/model-pass.{fsdp,megatron}.example.json。

```bash
python3 plugins/ops/engine/model_hotspot.py prepare \
  --config /path/to/model-pass.json --out ./outputs
```

prepare 完成后记录打印的 RUN_DIR。核对 operators/op_*/request.json、prompt.txt，原始 Triton 算子由当前 Agent 调用对应 CANNBot verifier/optimizer；配置实际 tune_command 后也可运行：

```bash
python3 plugins/ops/engine/model_hotspot.py tune --run /path/to/RUN_DIR
python3 plugins/ops/engine/model_hotspot.py finish --run /path/to/RUN_DIR
```

finish 只接受符合原有 Triton 单文件契约的真实收益补丁，并重新训练进行整网回归。不能把失败任务或仅 CLI 成功当成优化完成。

Ascend C/CANN native 与融合任务位于 generation_requests/task_*/；先人工核对真实 reference、源码、数据依赖，按 prompt 调用对应 CANNBot 插件。若已配置实际 Agent CLI，亦可运行：
```bash
python3 plugins/ops/engine/dispatch_operator_tasks.py \
  --run /path/to/RUN_DIR --config /path/to/cannbot-cli.json --task task_000
```
CLI 配置见 README 的双插件执行入口。当前 Ascend C 和融合的自动工程集成门禁未完成：可先实测独立算子，接回隔离模型后手工运行同 workload 回归；不得用旧 finish 契约硬套，也不得宣称自动完成。

## 4. 首轮验收与记录

| 项目 | 通过条件 |
| --- | --- |
| 安装 | init smoke 通过；Agent 可发现入口与实际 CANNBot plugins |
| 采集 | 指定 rank 的真实 Level1 产物通过门禁 |
| 路由 | 真实 Triton / Ascend C 路径证据明确 |
| 算子 | 真实 shape、dtype、stride、前反向精度通过，原实现作为性能分母 |
| MatMul | 同形态 FLOPs/latency/峰值口径；流水占比不是 MFU |
| 融合 | 真实生产者消费者、外部消费者、alias、autograd 等合法；原序列性能对照 |
| 整网 | 相同采样窗口、loss 容差、step time 非退化；有实际 token 时报告吞吐 |
| 失败 | 日志、证据及源码回滚可查 |

保留本轮 profile、baseline、operators、generation_requests、regression、model_report.md/json（实际生成的才记录）。建议第一天先验证一个 Triton 热点完整闭环，再拓展其他候选。已知限制：测试代码尚未执行实机验证，自动源码提取、Ascend C 多文件及融合自动集成仍待补齐；不是全自动验收完成版。


## 端到端收益硬门禁

单算子收益不是交付目标；只有同真实 workload 下整网主指标出现超过噪声且达到预设最低门槛的正收益，精度及保护指标同时通过，才计入有效优化。无整网收益、退化或未验证均不得作为优化成功交付。详见 [端到端收益验收](e2e-acceptance.md)。


## 单算子事务（当前执行原则）

每轮从当前已保留的模型版本出发，只选一个热点算子或一个不可拆分的融合算子组。AgentLoop 不干预 CANNBot 内部每一次代码迭代；由 CANNBot 自主执行分析、改写、精度与性能测量，按其退出条件交付最佳合格实现。整网只对该实现做独立对照；无需达到固定的单算子 2×/MFU 目标，只要满足单算子交付契约且实际整网收益超过预设噪声门槛、精度不劣化即可保留。失败或无收益回滚；下一候选必须以最新保留版本重新建立基线，禁止把多个算子的收益混在一起归因。第一版不自动循环所有候选。训练配置可选 target_operator（精确候选名称；融合用 operator_routes.json 的完整 name）；不指定则选择第一个可自动调优的 Triton 候选，其余标记 DEFERRED。生成/融合任务需要明确选定，当前仍需独立集成与回归，不能走现有 Triton finish 门禁。

调用 CANNBot 前必须按 [优化总原则 Skill](../skills/agentloop-optimization-principles/SKILL.md) 写本轮 optimization-decision.md（[模板](../skills/agentloop-optimization-principles/references/decision-record.md)），核定实际关键路径、收益预告和唯一算子。没有关键路径证据，先做诊断或记录不确定性；不要直接取累计 kernel 耗时最大的算子。
