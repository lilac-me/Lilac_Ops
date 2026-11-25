# AgentLoop Profile Opt Lite

本分支只维护大模型训练热点 Triton 算子优化链路：训练 Profiling → 热点识别 → 真实源码装配 → CANNBot 单算子迭代调优 → 补丁集成 → 整网训练性能/精度回归。不包含模型迁移、推理优化、算子重新生成、通用路由或文档 Agent。

## 目录

- `plugins/ops/engine/model_hotspot.py`：整网单批编排，prepare/tune/finish/run。
- `plugins/ops/engine/model_metrics.py`、`training_benchmark.py`：训练指标与验收。
- `plugins/ops/agents/profile-driven-opt-agent/`：Agent 入口定义。
- `skills/ops-skills/triton/profile-bottleneck-analyzer/`：热点识别。
- `skills/ops-skills/triton/triton-op-source-assembler/`：真实源码装配。
- `skills/ops-skills/triton/profile-driven-triton-opt/`：调优编排、产物契约、可选重构与整网接入文档。
- `skills/ops-skills/vendored-triton-op/`：CANNBot 调优依赖（本地安装，不提交生成的副本）。
- `scripts/vendor-triton-op-generator.sh`：上游安装与版本记录。

## 1. 安装

在具备 Python 3、Git、Node.js、实际 NPU 训练环境及 Agent CLI 的环境中：

```bash
cd AgentLoop
# 将 3 个本地 skill 和 6 个上游调优 skill 链接到实际训练仓的 Agent skill 目录
bash install.sh /path/to/model claude
# Codex 可改为 codex；也可只安装依赖，不投影：bash install.sh
```

安装器默认使用已有完整 vendor；不存在时调用原 vendor 脚本。可用 CANN_BOT_TAG、CANN_BOT_ROOT、CANN_BOT_SKILLS 固定上游版本。vendor 安装生成的版本信息见 VENDORED_TRITON.json；仓库已有清单仅表示历史快照，并不表示依赖已安装。若需要知识库，运行调优 Agent 前将 CANNBOT_KNOWLEDGE_ROOT 指向 cannbot-knowledge 的 knowledge 目录（见 vendor README）。安装不会覆盖目标仓已有同名 skill；冲突时请人工确认版本。

## 2. 配置训练

新增统一 Megatron / FSDP 训练接入，默认 20 step、前 10 step 训练预热、第 11 step profiler 预热、第 12–14 step 正式采集。FSDP 采用公共 TrainingHook，不区分 VeOmni / TorchTitan；Megatron 可使用原生 profiler 或同一个 Hook。详细接入见 [训练后端说明](skills/ops-skills/triton/profile-driven-triton-opt/docs/training-backend.md)，配置示例：[FSDP](skills/ops-skills/triton/profile-driven-triton-opt/docs/model-pass.fsdp.example.json) / [Megatron](skills/ops-skills/triton/profile-driven-triton-opt/docs/model-pass.megatron.example.json)。

以下为旧版分阶段命令配置（仍支持）：

复制 [model-pass.example.json](skills/ops-skills/triton/profile-driven-triton-opt/docs/model-pass.example.json) 至模型自己的配置目录，填写实际路径、profile_command、benchmark_command、source_map、workload。若使用自动 CLI 模式还必须配置 tune_command（支持 {request} 和 {prompt}）。

训练脚本负责真实开启/关闭框架 profiler，固定 checkpoint、数据、随机种子、并行策略、batch 和测量窗口。多 rank 产物使用 profile_subdir 指定统计 rank。source_map 需要提供真实源码，未匹配则跳过。具体环境变量及指标格式见 [整网单批接入说明](skills/ops-skills/triton/profile-driven-triton-opt/docs/model-hotspot-pass.md)。

## 3. 执行

```bash
python3 plugins/ops/engine/model_hotspot.py run --config /path/to/model-pass.json --out ./outputs
```

或分阶段：

```bash
python3 plugins/ops/engine/model_hotspot.py prepare --config /path/to/model-pass.json --out ./outputs
# 使用输出的 RUN_DIR，逐个让 Agent 执行 operators/op_*/prompt.txt
# 如果已配置 tune_command，也可执行：
python3 plugins/ops/engine/model_hotspot.py tune --run ./outputs/model_opt_xxx
python3 plugins/ops/engine/model_hotspot.py finish --run ./outputs/model_opt_xxx
```

已有 profile.zip 时可直接让 Agent 使用 profile-driven-triton-opt 的算子级入口；该入口不做整网回归。

## 4. 验收

热点按稳态 kernel 累计耗时占比排序（默认 3%，不是墙钟关键路径占比）。仅 strong 且有真实源码的算子进入调优。单算子默认目标 2x，真实用户源码为性能分母，官方 verifier/benchmark 为证据。合格补丁先试应用，再统一回归，训练异常、loss 超差或性能退化时回滚。整网只采集一次热点，无外层多轮寻优和 MFU 目标。

输出 model_report.md / model_report.json 及独立算子任务目录。未实测不填写整网收益。

## 5. 测试

```bash
python3 -m unittest discover -s plugins/ops/engine/tests -p 'test_*.py'
python3 -m pytest -q skills/ops-skills/triton/profile-bottleneck-analyzer/tests
# 安装 vendor 后可额外执行可选重构契约测试：
python3 skills/ops-skills/triton/profile-driven-triton-opt/scripts/test_restructure_contract.py
```

前两项为本地逻辑/合成数据测试，不代表真实 NPU 端到端性能验收。

新增训练采集与优化分流：Level1 / PipeUtilization / Text / record_shapes；MatMul MFU 与连续 Vector 融合候选见 [采集与分流契约](skills/ops-skills/triton/profile-driven-triton-opt/docs/training-backend.md#cannbot-profiling-采集契约与优化分流)。
