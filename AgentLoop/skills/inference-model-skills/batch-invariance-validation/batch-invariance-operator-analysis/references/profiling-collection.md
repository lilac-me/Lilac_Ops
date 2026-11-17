# 阶段一 Profiling 调用

当 batch invariance 验证流程需要委托给其他 MindSpeed-Agent skill 时，使用本文件中的提示词模板。使用前必须替换尖括号占位符。本地和远程 Phase 1 都调用 `/inference-model-optimize:ascend-profiling-collection`；本地目标使用 `connection-mode=local` 直接采集，不建立 SSH 连接，跳过远程认证展示步骤。

## Phase 1 Profiling 采集

复用现有采集入口的完整链路：带 `--profiler-config` 的服务启动 → `/start_profile` → 合规请求 → `/stop_profile` → 逐 rank 离线分析。`ASCEND_PROFILER_OUTPUT` 在此流程中是分析产物目录，不是启用开关；不要以 `VLLM_PROFILING_MODE` 或未经目标版本验证的 `--profiling` 代替采集入口。必须保留 profiler 配置回显、窗口控制结果和逐 rank 的 `op_statistic.csv`。

先判断当前进程是否与目标推理进程处于同一服务器、容器或 conda 运行空间，并能直接访问模型、目标 NPU 和输出目录：满足时设置 `<TARGET_MODE>=LOCAL`，否则设置为 `REMOTE`。替换所有占位符后，将下面整段 prompt 提交给 `/inference-model-optimize:ascend-profiling-collection`。

本地模式只传 `connection-mode=local` 及采集业务参数，禁止传入 `host`、`container`、`conda-env`、`user`、`password`、`key-file` 等连接参数；该模式走下游 skill 的完整采集流程，不要求调用方直接执行其内部脚本。远程模式必须填写连接信息；该 skill 先展示完整远程参数表，用户确认前不得开始采集。不要传入下游 skill 的本地文件路径。

```text
/inference-model-optimize:ascend-profiling-collection
请完成 batch invariance Phase 1 Ascend profiling 采集：
1. target-mode=<LOCAL|REMOTE>。
   LOCAL：当前环境就是目标服务器、容器或 Conda 运行空间；只设置 connection-mode=local，不传 host、container、conda-env、user、password、key-file 等连接参数，CANN 路径=<CANN_DIR>；跳过远程参数表展示与确认步骤。local 仅支持单机采集。执行采集命令前必须在同一命令环境执行 `unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy`，确保 `/start_profile` 和 `/stop_profile` 的 `127.0.0.1` 请求不经过本地代理。
   REMOTE：连接参数为 host=<HOST>、connection-mode=<docker|direct|conda>、container/conda=<CONTAINER_OR_CONDA>、认证信息=<AUTH>、CANN 路径=<CANN_DIR>，开始前展示完整参数表并等待用户确认。
2. 本次运行记录目录=<RUN_RECORD_DIR>；所有采集产物写入 `<RUN_RECORD_DIR>/phase1/`。模型=<MODEL_PATH>，served-model-name=<SERVED_MODEL_NAME>，TP=<TP>，DP=1，设备=<ASCEND_RT_VISIBLE_DEVICES>，mode=enforce_eager，max-model-len=22528，enable_chunked_prefill=False，enable_prefix_caching=False，async_scheduling=False，block_size=128，generation-config=vllm，analyse-all=true（执行时传递 `--analyse-all`），tag=<TAG>，torch-profiler-dir=<RUN_RECORD_DIR>/phase1/profiling。服务启动时必须通过 serve_args 显式传入 `--generation-config vllm`，禁止模型目录中的 generation_config.json 补入未显式指定的 top_p、top_k 等采样参数。当前正式支持范围仅为 TP；其他并行配置只能形成对应的未验证证据。
3. 首次准入使用 profile-phase=middle、random-input-len=5、random-output-len=20、num-prompts=8、max-concurrency=8、max-num-seqs=8。定位已知失败时使用原失败现场的长度、请求并行数、seed 和调度参数，并逐项记录差异。
4. 服务和 workload 必须继承 VLLM_BATCH_INVARIANT=1，请求显式使用 temperature=0.6。正式请求不得包含 logprobs、top_p、top_k、min_p 或惩罚项；发送前保存并检查实际 payload，误传时返回 INVALID_PROFILING_CONFIG。token logprob 只能另行辅助采集，不得作为 Phase 1 算子验收证据。入口无法显式设置 temperature 时返回 BLOCKED_UNSUPPORTED_PROFILE_TEMPERATURE，不得假定默认值等价。
5. 直接完成服务启动、profiling 窗口控制、workload、服务停止和分析。窗口必须覆盖真实模型前向 logits 路径。复用服务前确认 profiler config、VLLM_BATCH_INVARIANT、模型、TP、设备和调度参数一致，并从日志确认第 2 条服务配置生效；无法确认时启动独立服务。
6. 必须启用 analyse-all；该参数控制对全部 `*_ascend_pt` rank 目录执行离线分析和产物校验，禁止沿用默认 False 只分析第一个目录。采集并分析全部 TP ranks，确认每个 rank 都生成 kernel_details.csv、trace_view.json 和 op_statistic.csv。在线解析失败但原始数据完整时逐 rank 离线分析并记录命令和退出码；缺少 op_statistic.csv 时返回 BLOCKED_MISSING_OP_STATISTIC，不得用其他文件替代。
7. 只清理本次启动的服务进程，确认端口和目标 NPU 无本次任务残留；失败或中断时也必须报告清理状态。
8. 成功时返回调用退出码、实际参数、rank 完整性、服务清理状态、manifest/采集报告（如生成），以及相对于运行记录目录的证据位置。失败或阻塞时返回明确 status、phase、error 和已有证据位置；不要展示 skill 安装目录或内部脚本路径，不得把下游 skill 的 status=ok 直接解释为 Phase 1 通过。
```
