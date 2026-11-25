# Megatron / FSDP 训练接入（v1）

AgentLoop 不感知 FSDP 的具体框架（VeOmni、TorchTitan 等），只要求原有启动命令能运行，以及在完整 optimizer iteration 边界推进 Hook 或原生 profiler。默认 20 个完整训练 step（相对于本次训练开始，不是 checkpoint 的全局 step），1–10 不采集、11 profiler warmup、12–14 active、15–20 不采集，正常退出。三次训练均独立从同一 checkpoint 和数据开始：profile、baseline、regression。后两次关闭 profiler，统计第 11–20 个 step。不会在训练进程内调用 CANNBot。

## FSDP（推荐 Hook）及 Megatron Hook

配置参考 model-pass.fsdp.example.json。启动器原样使用训练命令，自动将 engine 目录加入 PYTHONPATH，并传递下面的环境变量。远端所有节点和容器需要共享或同步 AgentLoop、模型和输出目录，启动器必须转发环境变量。

在已有训练循环外添加 context，完整梯度累积、优化器更新结束后，在所有 rank 调用一次 step。不要在 microbatch 内调用，不要另行开启原生 profiler。

```python
from training_backend import TrainingHook

with TrainingHook() as hook:
    for batch in train_loader:
        # 原有训练逻辑（示意，保持原训练代码）
        # 完整 optimizer iteration，包含所有 microbatch
        metrics = train_one_iteration(batch)
        # metrics 由训练框架实际计算：全局归约 loss、完整 step 墙钟毫秒；
        # 可选真实全局 token 数。不要把每卡 token 或异步 kernel 下发时间当整网指标。
        if hook.needs_metrics:
            stop = hook.step(loss=metrics.loss,
                             step_time_ms=metrics.step_time_ms,
                             tokens=metrics.global_tokens)
        else:
            stop = hook.step()
        if stop:
            break
# 走原框架正常退出/清理路径
```

训练框架若没有这些指标，需在其实际迭代边界建立正确的计时和 loss 归约；Hook 不额外插入 NPU synchronize 或 collective，也不虚构指标。tokens 可省略。若 loss 为 Tensor，应先按框架正常方式归约并转换为 Python float。计时需覆盖完整迭代的真实设备完成时间；仅 CPU enqueue 时间不合格。所有 rank 必须完成相同步数并退出；Hook 对提前退出或超过步数报错。默认从 torchrun 的 RANK 获取全局 rank，非 torchrun 启动器可显式传入 rank。离开 AgentLoop 环境时 Hook 无操作。

仅配置的一个 profiling rank 创建 torch_npu.profiler，使用 CPU+NPU、Level1、record_shapes、with_stack、Text 导出；其余 rank 正常训练。采集完成标记和 kernel_details.csv 由 engine 校验。第一版不做多 rank 合并。metric_rank 可以和 profiling rank 不同，但必须提供代表整网的实际指标。

## Megatron 原生接入

配置参考 model-pass.megatron.example.json。integration=native 时不调用 Hook。原有启动脚本需要根据以下环境变量映射真实训练参数（包括 profile start/end 的版本差异），保证所有 rank 按最大步数正常结束：

| 环境变量 | 语义 |
| --- | --- |
| AGENTLOOP_BACKEND | megatron / fsdp |
| AGENTLOOP_PHASE | profile / baseline / regression |
| AGENTLOOP_PROFILE_ENABLED | profile 为 1，其他为 0 |
| AGENTLOOP_MAX_STEPS | 本次完整迭代数，若框架采用全局 train-iters 需加 checkpoint 起始 step |
| AGENTLOOP_PROFILE_SKIP_FIRST | 训练预热步数 |
| AGENTLOOP_PROFILE_WARMUP | profiler 预热步数 |
| AGENTLOOP_PROFILE_ACTIVE | 正式采集步数 |
| AGENTLOOP_PROFILE_RANK | 全局采集 rank |
| AGENTLOOP_PROFILE_DIR | profiler 输出根 |
| AGENTLOOP_BENCHMARK_WARMUP_STEPS | benchmark 跳过步数 |
| AGENTLOOP_METRIC_RANK | 指标统计 rank |
| AGENTLOOP_METRICS_PATH | 自定义 JSON 指标输出路径 |

launch_command 的 argv 支持 {max_steps}、{profile_skip_first} 等由 AGENTLOOP_ 环境变量派生的小写占位符（不含原有 phase 变量；phase 开关请读环境）。也可直接在 bash 脚本读取上述变量。Megatron 原生模式默认使用原 training_benchmark.py 解析逐 iteration 日志；须 log-interval=1，单统计 rank，并在 benchmark 关闭 profiler。非 Megatron 日志可设置 training.metrics=json，自行输出原有指标格式。FSDP 也可使用 native+json，具体框架不进入 engine。

原生模式由启动脚本保证采集窗口；engine 校验 CSV 唯一性和 benchmark 样本数，不声称能自动识别任何框架的实际 profiler 状态。profile_subdir 选择唯一 CSV 的目录。Hook 模式自动固定为 rank_N。

## 执行及边界

保留原 model-pass.example.json 与 legacy profile_command / benchmark_command 入口；新 training 配置不可同时填写旧的两个命令。run 仍需配置实际 tune_command；prepare 可先完成采集/基线，再让 Agent 按 prompt.txt 调优后 finish。源码映射、单算子 CANNBot 门禁、补丁回滚不变。

当前固定窗口不保证训练已经稳态；如首次 JIT/编译或动态 shape 较慢，应扩大 skip_first、benchmark_warmup_steps 和 max_steps。不能直接用于不中断的生产训练，建议使用独立性能实验任务。需要真实 NPU 环境验收 profiler 版本、数据及日志。
