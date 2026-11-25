# 整网单批热点优化

一次采集确定本轮热点，逐个调用已有算子工作流，统一接回模型后只做一次整网回归。
整网没有目标 MFU、没有外层寻优循环，也不根据回归结果自动重新采集热点。

## 执行顺序

1. `profile_command` 启动原始训练并采集一个稳态窗口，结果放入本轮 `profile/`。
2. 解析一次 profiler，冻结候选与实际源码；多 rank 产物必须用 `profile_subdir` 选择统计 rank。
3. `benchmark_command` 在关闭 profiler 的配置下测量基线。
4. 对固定清单内的强候选串行调用 S3/S4–S6，算子内部沿用现有调优目标与退出规则。
5. 校验每个补丁，在临时目录先测试整批应用；冲突或不合格候选单独拒绝。
6. 将可采纳补丁应用到模型源码，用相同 `benchmark_command` 测量一次优化后训练。
7. 输出对比报告；训练失败、loss 超差或整网收益不足时还原本轮修改的源码，保留测量与诊断记录。

profiling 只采集一次。为避免采集开销干扰前后对比，另有两次关闭 profiler 的计时运行
（原始基线、优化后各一次）；它们不再次筛选热点。若没有可采纳补丁，报告 `NO_CHANGES`，
不再启动优化后训练，也不虚构「1×」实测结果。

## 配置与入口

统一 Megatron / FSDP 后端及固定 Step Hook 接入见 [training-backend.md](training-backend.md)。以下旧版命令配置仍可使用；新配置不同时填写旧的 phase commands。

复制同目录 `model-pass.example.json`，填写实际训练仓、启动命令及源码映射。
在已具备模型、数据、CANN/NPU 和 cannbot skills 的训练环境中运行；本机无需替训练框架重写启动器。
`model_root` 相对配置文件解析，命令 cwd 均为它。默认算子基线为真实源文件，调优期间只修改副本。

```bash
# 在 AgentLoop 仓根执行完整流程；tune_command 必须是能调用已安装 skills 的实际 Agent CLI。
python3 plugins/ops/engine/model_hotspot.py run \
  --config /path/to/model-pass.json --out /path/to/results
```

也可以由当前 Agent 负责逐个调用 skill，不额外启动 Agent CLI：

```bash
python3 plugins/ops/engine/model_hotspot.py prepare \
  --config /path/to/model-pass.json --out /path/to/results
# 使用打印的 RUN_DIR，按 operators/op_*/prompt.txt 逐个执行算子调优。
# 每个 request.json 已指定本轮候选、真实源码、shape/dtype、输出目录和指标配置。
python3 plugins/ops/engine/model_hotspot.py finish --run /path/to/results/model_opt_ID
```

`tune --run RUN_DIR` 可单独执行配置中的 `tune_command`。它串行处理尚未运行的任务，
每个任务失败会写入报告并继续其余候选。`finish` 对已经结束的运行拒绝重复应用。
未完成或失败的运行要显式检查产物；默认新建运行目录，不静默续用旧结果。

## 命令适配

命令使用 argv 数组，无隐式 shell 拼接。脚本启动可用 `["bash", "scripts/train.sh"]`。
支持参数占位符 `{agentloop_root}`、`{model_root}`、`{run_dir}`；`tune_command` 另支持
`{request}`（JSON 绝对路径）、`{prompt}`（完整任务文本，作为一个 argv 参数传入）。
不指定固定 CLI、模型或调优轮数；沿用运行环境已安装的 cannbot 和算子内层配置。

每个命令获得以下环境变量：

| 变量 | 用途 |
| --- | --- |
| `AGENTLOOP_RUN_ID` / `AGENTLOOP_RUN_DIR` | 本轮标识与独立产物目录 |
| `AGENTLOOP_MODEL_ROOT` | 当前训练仓 |
| `AGENTLOOP_PHASE` | `profile` / `baseline` / `tune` / `regression` |
| `AGENTLOOP_PROFILE_DIR` | 本轮 profiler 产物目录 |
| `AGENTLOOP_PROFILE_ENABLED` | 仅采集阶段为 `1`，其他阶段为 `0` |
| `AGENTLOOP_METRICS_PATH` | 当前整网测量 JSON 的输出路径 |
| `AGENTLOOP_WORKLOAD_ID` | 固定工作负载配置的 SHA-256 |

这些变量不会自动开启或关闭 MindSpeed 的 profiler。`profile_command` 必须把训练框架的
实际采集开关、输出目录和 schedule 配好，运行成功后在指定目录留下 `kernel_details.csv`。
`benchmark_command` 必须关闭该开关。运行环境不同的启动参数由模型自己的脚本适配，
不使用面向 vLLM serving 的采集脚本来启动训练。

原始模型和优化模型必须从相同 checkpoint、数据顺序、随机种子开始，固定精度、并行拓扑、
batch、序列长度、预热/测量窗口；输出 checkpoint 应写入各自运行目录，不能覆盖共同输入。
`workload` 是这些设置的记录，不会替启动脚本自动设置训练参数；执行前须核对两者一致。

## 整网指标

计时命令退出码须为 0，并写入 `AGENTLOOP_METRICS_PATH`：

```json
{
  "run_id": "从 AGENTLOOP_RUN_ID 读取",
  "phase": "baseline",
  "workload_id": "从 AGENTLOOP_WORKLOAD_ID 读取",
  "profiling_enabled": false,
  "samples": [
    {"step": 6, "step_time_ms": 100.0, "loss": 2.15, "tokens": 32768}
  ]
}
```

`phase` 同样读取环境变量。`samples` 仅包含预热后的实测 step，默认至少 5 个；
前后 step 编号必须完全一致。耗时为实际训练 step 的毫秒数，不能用 kernel 耗时之和代替。
loss 必填，逐 step 判断 `abs(new - old) <= loss_atol + loss_rtol * abs(old)`。
`tokens` 可全部省略；提供时必须为每步实际处理的全局 token 数，前后相同，吞吐按总 token / 总时间计算。
变长/packing 场景应由模型适配器填写实际 token 数，不能使用固定长度估算冒充实测工作量。

对输出 Megatron 风格 `iteration ... | elapsed time per iteration (ms): ... | lm loss: ...` 的训练，
可直接使用提供的 `training_benchmark.py` 包装原命令，自动留存日志、剔除预热样本并导出指标：

```bash
python3 /path/to/AgentLoop/plugins/ops/engine/training_benchmark.py \
  --warmup-steps 5 --tokens-per-step 32768 -- bash scripts/train_without_profile.sh
```

这条命令作为 `benchmark_command` 由外层设置环境并调用；只适用于该日志格式。
`--tokens-per-step` 仅适用于固定实际工作量，未知时省略。多 rank 重复日志会被拒绝，需选定统计 rank。
其它训练框架可直接生成相同 JSON。非有限数、旧 run 标识、不同工作量或窗口都不能通过验收。

## 算子交付与补丁集成

每个 task 的 `request.json` 与候选清单带哈希，`summary.json` 需增加本轮 `run_id` 和候选 `op_name`。
沿用内层原有字段和 `check_summary_contract.py`，包含：

- `perf_baseline` 指向模型仓真实源；`gate_ref` 仅用于正确性。
- `optimizer_rounds > 0`，完整的 optimizer/scan/强制点记录，`best_code` 及本轮 benchmark 证据。
- `patch.path`、`diff_from`、`diff_target` 均指向该任务目录内文件；原始副本必须与冻结源码相同。
- `patch.apply_target` 为真实源绝对路径，`apply_rel` 为它相对 `model_root` 的路径。
- `patch.resverified` 记录补丁目标版本的正确性验证；临时应用结果须与该版本逐字节一致。
- `REACHED` 或 `BELOW` 均可采纳，但必须有真实单算子 latency 收益；带 waiver/advisory 的补丁不自动集成。

每份补丁只允许修改当前候选对应的一个源文件。多个算子落在同文件时按顺序试应用，冲突的补丁
标为 `REJECTED`，其余可继续；最终组合的训练精度由整网回归检查。调用前后检测源哈希，
发现调优器直接改了模型仓或用户另有修改时停止，避免覆盖新的源码。

## 结果与失败处理

`model_report.md` / `model_report.json` 包含候选处理状态、基线/优化后平均与中位 step、吞吐、
实测加速比、loss 检查和补丁是否保留。整网指标是最终正收益验收门禁，不驱动新的外层寻优循环。
`min_step_improvement_pct` 默认 1.0，要求平均 Step Time 至少降低该百分比、中位 Step Time 有正改善且 loss 通过。无正收益或不足阈值时回滚。`max_step_regression_pct` 保留兼容，不能绕过正收益门禁。单次回归仅为初筛，稳定性须多次复测。

状态为 `COMPLETED`、`NO_CHANGES`、`ROLLED_BACK` 或 `FAILED`。命令异常/超时会终止其进程组；
整网验收失败会还原本轮触及的源文件并保留 `original/`、命令日志和实测结果。
强制断电或 SIGKILL 无法运行清理：`state.json` 的 `APPLYING/REGRESSION` 和 `original/` 用于人工恢复，
不要直接重复执行 finish。模型源码之外的训练产物由各启动器管理。
