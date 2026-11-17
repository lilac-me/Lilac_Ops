# 环境变量附录

- 交付基线脚本：`../assets/env_npu.sh`，只收「VLA/VLM 系通用 **且偏离官方默认值**」的项。
- 证据分三档：**【官方】**（昇腾官方文档/源码）/ **【语料】**（`model_examples/` 67 个模型的 test 脚本统计，括号里是覆盖数）/ **【实测】**（A3 真机）。
- ⚠️ **写默认值是 no-op**。不少模型脚本里抄了一堆等于默认值的项（见文末统计），本附录会标明，不必照抄。

---

# 一、交付基线（= `assets/env_npu.sh` 的内容）

| 变量 | 取值 | 为什么要它 | 证据 |
| --- | --- | --- | --- |
| `TASK_QUEUE_ENABLE` | `2` | 算子下发二级流水。官方建议值就是 2（虽然同页标注 Level2 仅在二进制场景生效）。**不是越大越好**：极重/超长序列任务可能 1 更好 | 【官方】默认 1、官方建议 2；【语料】VLA/VLM 15/15 都设，全仓 103 处用 2、7 处用 1（Cosmos-Predict2 用 1） |
| `CPU_AFFINITY_CONF` | `1` | CPU 绑核，避免线程抢占、减少跨 NUMA。不设 = 不绑核。扩散/推理类多用 2（细粒度） | 【官方】默认 0（不启用）；【语料】VLA/VLM 15/15，全仓 1×88 / 2×18 / 0×3 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 内存池扩展段，降低显存峰值。长序列模型建议开 | 【语料】VLA/VLM 11/15，全仓 34 处 |
| `TOKENIZERS_PARALLELISM` | `false` | 关 tokenizer 多进程告警 | 【语料】 |
| `WANDB_MODE` | `offline` | 离线环境 | 【语料】 |

**两条硬约束**（会互相抵消，容易踩）：

- `ASCEND_LAUNCH_BLOCKING=1` 时 task_queue 整个关闭，**`TASK_QUEUE_ENABLE` 失效**。【官方】
- `TASK_QUEUE_ENABLE=2` 因内存并发，**会导致 NPU 内存峰值上升**。【官方】。显存吃紧时先回退到 1 试试。

**基线脚本里还有一段不计入上表、也不是环境变量的固定降噪块**：调 `msnpureport` 把 Device 侧日志级别降到 ERROR（`-g error -d <i>` 逐卡 + `-e disable`）。三个实测要点：

- 容器里必须带 `--docker`，**不带会直接失败**（连 `msnpureport -r` 读当前级别都报 `should be used with --docker`）。语料里 32/33 份 `test/env_npu.sh` 都有这段，但在容器中一直是空操作；
- 它的错误信息打的是 **stdout**，`2>/dev/null` 挡不住 → 脚本用「先试 `-r`、失败再带 `--docker` 试」探测运行环境（不依赖 `/.dockerenv`），并把 stdout + stderr 一起丢；
- 没装 `msnpureport` 时用 `command -v` 探测、不报错。代价是每次 source 多约 1s、最多 9 次命令；**纯降噪，不设也能跑**，需要时整段注释掉即可。

## 基线里**未收录**的（值是官方默认，写了 = no-op）

| 变量 | 语料里写它的值 | 官方默认 | 说明 |
| --- | --- | --- | --- |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | `0` | `0` | 0 关 / 1 开（Host 日志打到**标准输出**） |
| `ASCEND_GLOBAL_LOG_LEVEL` | `3` | `3` | 0-DEBUG / 1-INFO / 2-WARNING / 3-ERROR / 4-NULL |
| `ASCEND_GLOBAL_EVENT_ENABLE` | `0` | `0` | Event 日志开关 |
| `HCCL_WHITELIST_DISABLE` | `1` | `1` | **默认就关闭白名单**。写 `=1` 等于不写；官方另提示关白名单会去掉一层安全边界，只在受信网络用 |

`ASCEND_SLOG_PRINT_TO_STDOUT` / `ASCEND_GLOBAL_LOG_LEVEL` / `ASCEND_GLOBAL_EVENT_ENABLE` 三项 VLA/VLM 系各 4/15 在用，`HCCL_WHITELIST_DISABLE` 13/15 在用，**覆盖率不低但全是默认值**，属历史抄写，不影响结果。

> 要调日志级别时用 `ASCEND_MODULE_LOG_LEVEL`，它的**优先级高于** `ASCEND_GLOBAL_LOG_LEVEL`（同时设置以模块级为准）。【官方】

---

# 二、按场景开（默认注释，开了记进档案卡口径）

| 变量 | 取值 | 什么时候开 | 证据 |
| --- | --- | --- | --- |
| `MULTI_STREAM_MEMORY_REUSE` | FSDP2 用 `2`，否则 `1` | 多流内存复用。**FSDP2 场景 `=2` 实测很有效**：大幅减少显存波动、避免内存重整 | 【语料】VLA/VLM 4/15；【官方】源码文档对值 2 标注「当前不推荐」（保守表述），但有 PR 把默认值由 1 改成 2；以实测为准，先验证精度无误再采纳 |
| `TORCH_HCCL_ZERO_COPY` | `1` | 集合通信片内零拷贝，减少 all_gather / reduce_scatter 的片内拷贝 | 【官方】**beta/实验特性，官方建议谨慎使用**；**仅 A3 训练系列支持**（A2、Ascend 950 需注释）；**依赖** `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；**不支持图模式**；先用 `npu-smi info` 确认卡型。【语料】VLA/VLM 2/15（StarVLA、GR00T-N1.7） |
| `COMBINED_ENABLE` | `1` | 优化非连续两算子组合场景 | 【语料】VLA/VLM 2/15 |
| `HCCL_SOCKET_IFNAME` | 如 `eth0` | 多网卡且自动选错通信网卡时显式指定 | 【官方】CANN `error_code.json` 的通信域建链失败处置里点名它。【语料】VLA/VLM 0/15（多机场景才有用） |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | 如 `16` | 控制 BLAS/OMP 线程数，与 `CPU_AFFINITY_CONF` 配合 | 【语料】各 5 处（GR00T-N1.7 / OpenDWM 等） |

---

# 三、仅精度跑开（性能跑与交付件必须保持注释）

| 变量 | 取值 | 说明 |
| --- | --- | --- |
| `HCCL_DETERMINISTIC` | `true` | 确定性计算，有性能代价 |
| `CLOSE_MATMUL_K_SHIFT` | `1` | 同上；属 ATB/MindIE 生态变量，非 CANN 核心文档收录项 |

两项与 HF32 是否互斥**尚无定论**，需实测确认。**HF32** 的开关有两种写法（等价，任选其一）：

- Python 属性：`torch_npu.npu.aclnn.allow_hf32 = True`（矩阵运算）/ `torch.npu.conv.allow_hf32 = False`（卷积密集的感知主干要关掉保精度）
- 环境变量：`ALLOW_MATMUL_HF32` / `ALLOW_CONV_HF32`（名从 `libtorch_npu.so` 中提取）

---

# 四、⚠️ 禁止留在交付件里的

| 变量 | 危害 |
| --- | --- |
| `ASCEND_LAUNCH_BLOCKING=1` | **同步执行**，便于定位算子级报错，但算子同步下发会让性能大幅下降；同时**让 `TASK_QUEUE_ENABLE` 整体失效**。只允许 debug 阶段临时用，**性能跑与交付件必须删除**。交付前自查：`grep -rn "ASCEND_LAUNCH_BLOCKING" test/` |

【语料】67 个模型的 test 脚本里**一次都没出现过**这个变量，交付件里出现它就是漏删。

---

# 五、其它（按需查询）

## HCCL 超时与建链

| 变量 | 值 | 说明 |
| --- | --- | --- |
| `HCCL_CONNECT_TIMEOUT` | **保持默认 120** | 官方范围 `[120, 7200]`、**默认 120**，实际建链超时 ≈ 配置值 + 20s。**默认值就是调试友好值，要调也是「不要调大」**：建域失败时长时间挂起而不是报错，正是把它调大造成的（语料里 20 处设成 1200，属多机慢建链场景）。低于下限报 `Config_Error_Invalid_Environment_Variable(EI0001)` |
| `HCCL_EXEC_TIMEOUT` | 默认 | 设备间执行同步等待时间（官方范围约 `[68, 17340]`、默认 1800 量级）。语料里有设 8000/9000 的，属特例，不设即可 |
| `HCCL_ASYNC_ERROR_HANDLING` | `0` | 关 watchdog，防长时间 eval 被判超时（UniAD 的做法）。语料 1 处 |
| `HCCL_NPU_SOCKET_PORT_RANGE` | 如 `60000-60015` | device 侧 socket 端口段。多进程共用同一 device IP 时必需；`os._exit(0)` 泄漏端口（EI0020 已绑定）时换段缓解 |
| `HCCL_IF_IP` / `HCCL_IF_BASE_PORT` | — | 多机/多网卡场景；同机多角色必须错开 |

## 显存

`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（已在基线）、`MULTI_STREAM_MEMORY_REUSE`（见 §二）、`PYTORCH_NO_NPU_MEMORY_CACHING`（so 中存在，需时查官方）。手段库见 `flow/2-optimize.md` §2。

## 图模式运行时调整（进脚本前先验证）

| 变量 | 值 | 说明 |
| --- | --- | --- |
| `TORCH_NPU_DEVICE_CAPABILITY` | 如 `8.0` | 配置 `torch_npu.npu.get_device_capability()` 的返回值（默认 `None`）。**官方只说它兼容 `torch.cuda.get_device_capability()` 接口、不代表 NPU 真实能力**；「图模式报错时先试它而不是直接关图模式」是**实测/社区实践**，不是官方结论。【语料】DiffusionDriveV2 用了 `=8.0` |

图模式迁移默认关、实机按报错回开的取舍见 `flow/1-migrate.md` §3.3。

## 设备可见性

| 变量 | 取值 | 说明 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | 如 `0,1,2,3,4,5,6,7` | 昇腾版的 `CUDA_VISIBLE_DEVICES`。迁移时源码里对 `CUDA_VISIBLE_DEVICES` 的引用要换成本变量（`set -u` 下引用未定义变量会直接 abort）。**交付脚本默认不写死**：语料 188 个 `test/*.sh` 里只有 **4 个**显式设置，惯例是交给命令行 / 调度器；训练模板（`../assets/train_Np.sh.template`）用的是「未设置时按 `--num_npus` 推导 `0..N-1`」，已设置则尊重环境值。 |

## 卡型相关

先用 `npu-smi info` 确认卡型：`TORCH_HCCL_ZERO_COPY` 仅 A3；`CPU_AFFINITY_CONF` 绑核支持 Atlas 训练系列 / A2 / A3。

## 诊断（挂起 / 卡死时）

torch_npu 自带 HCCL 超时转储，比手工 `npu-smi` + `ss` 更直接（名从 `libtorch_npu.so` 提取，取值需查官方）：

`TORCH_HCCL_ENABLE_MONITORING` · `TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC` · `TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC` · `TORCH_HCCL_DUMP_ON_TIMEOUT` · `TORCH_HCCL_TRACE_BUFFER_SIZE`

排查步骤仍见 `flow/1-migrate.md` §2。

## 离线

| 变量 | 作用 |
| --- | --- |
| `HF_HUB_OFFLINE` | `1` = 只用本地缓存，禁止联网 |
| `TRANSFORMERS_OFFLINE` | `1` = transformers 侧离线 |
| `HF_HOME` | 本地缓存根目录 |
| `HF_ENDPOINT` | 镜像站地址（离线环境一般与上两项二选一） |

代码侧改造（`from_pretrained` 改本地路径、`local_files_only=True`）见 `flow/1-migrate.md` §1.6。

## 特定场景（语料在用，但 VLA/VLM 系不用；不要当通用项照搬）

`ACLNN_CACHE_LIMIT`（12 个文件，10 万~100 万）· `ACL_OP_INIT_MODE`（11 个 `=0`）· `DYNAMIC_OP="ADD#MUL"`（11 个）· `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`（9 个）· `ENABLE_RUNTIME_V2`（6 个）· `GE_PROFILING_TO_STD_OUT` · `DISABLE_L2_CACHE` · `HCCL_OP_EXPANSION_MODE` · tcmalloc / `LD_PRELOAD`（8 个文件，均出现在**安装脚本**而非训练脚本）。**VLA/VLM 系 15 个模型里基本都不用**。
