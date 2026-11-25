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

## CANNBot profiling 采集契约与优化分流

训练采集固定使用 Level1 + Text + PipeUtilization，record_shapes=True、with_stack=True、with_modules=False、profile_memory=False，CPU + NPU。原生模式必须将 AGENTLOOP_PROFILE_PROFILER_LEVEL、AGENTLOOP_PROFILE_EXPORT_TYPE、AGENTLOOP_PROFILE_AIC_METRICS、AGENTLOOP_PROFILE_RECORD_SHAPES、AGENTLOOP_PROFILE_WITH_STACK 等环境变量映射到真实 profiler；禁止静默降为 L0。Engine 校验选定 rank 的 kernel_details.csv、step_trace_time.csv、trace_view.json 及实际 shape/dtype、时间线和 AI Core 指标字段，失败停止。采集配置与证据写入 profile_contract.json（不能凭 CSV 证明全部请求开关的实际状态）。

分析器输出 analysis/opportunities.json（整网报告同时展示）：
- MatMul/GEMM/GMM 类（名称/Type 规则）：按耗时占比筛选，保存 shape/dtype 和可用流水指标；已有真实 Triton 源且 strong 的任务自动切换 metric_mode=mfu，调用 CANNBot triton-mfu-analyzer，真实 FLOPs、实测 dtype 峰值和 latency 才能计算 MFU。内建 CANN MatMul 只报告，不自动重写。
- Vector Fusion：只在同一 step、stream 中，按真实 start timestamp 排序，连续纯 Vector、前一输出 shape/dtype 与后一输入兼容、时间间隔在配置 fusion_max_gap_us 内（默认 5us）时生成候选；跨 stream、通信/非 Vector 算子、缺失时间线不推断融合。合并热点按累计 kernel 耗时占比过滤。相邻不是数据依赖，必须回查 trace、源码、真实张量依赖、外部消费者、stride、alias、前后向及数值精度。当前只输出待确认机会，不自动调用算子生成器、不改图；后续可接入 CANNBot fusion generation 和单独的图级补丁契约。
- 其他已有自定义 Triton：保持 latency 调优路径。

整网 MFU 不等于单算子 MFU。现有补丁门禁仍要求单算子真实 latency 收益和整网 loss/性能回归。
