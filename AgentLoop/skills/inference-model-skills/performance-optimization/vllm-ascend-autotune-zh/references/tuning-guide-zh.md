# vLLM/vLLM-Ascend 寻优指南

## 推荐搜索顺序

1. **发现当前环境特性**
   - 先运行 `scripts/discover_vllm_features.py` 生成 `features.json` 和 `features.md`。
   - 后续候选必须来自：CLI help、vLLM-Ascend envs.py、源码/docs、benchmark 脚本、或用户明确指定。
   - 不要把本文档的“常见候选”当成完整列表；它只是优先级经验。

2. **锁定正确性和工作负载**
   - 固定模型路径、tokenizer、设备、容器、脚本。
   - 比较不同配置前，固定 prompts、并发、输入长度、输出长度、`--ignore-eos` 和生成参数。
   - 固定随机输出长度 benchmark 中，如果模型 generation config 会导致提前 EOS 或短输出，使用 `--temperature 0`。
   - 每次正式 benchmark 先通过预检确认版本和设备，再运行 `scripts/run_with_sma.py --devices <设备集合>`，把 SMA/NPU 利用率打印到过程日志。

3. **拓扑（强制第一阶段，先于第 4 步及以后的任何参数寻优）**
   - 在卡数约束内枚举全部合法 TP/DP/EP 组合，逐个 benchmark，在固定 workload 下比较吞吐、TTFT、TPOT 和有效性。
   - 卡数 ≥ 2 时至少实际测试 2 种不同拓扑；只有一种合法拓扑时也要记录并写明理由。
   - 4 卡 MoE 模型通常先测 TP4/DP1 与 TP2/DP2。
   - MoE 模型要测试 expert parallel。若 FlashComm 等通信特性是启动或正确性的必要条件，则锁定它。
   - 拓扑候选穷尽并选出阶段最优拓扑后，才允许进入 scheduler/PCP/KVCacheOffload 等参数寻优。

4. **Batch scheduler**
   - 先调 `MAX_NUM_BATCHED_TOKENS`。长上下文候选常见值包括 `24576`、`28672`、`32768`、`36864`、`40960`。
   - 再调 `MAX_NUM_SEQS`、`BS`、`NUM_PROMPTS`。宣称吞吐提升时，要保证总生成 token 可比。
   - 提高并发不一定提高吞吐；TTFT 和尾部阻塞可能让整体 Output throughput 下降。
   - **⚠️ 并发档位：以服务端 `Maximum concurrency` 日志推算预算，预算允许就加测高档位**
     （详细方法见 `vllm-ascend-benchmark/references/max-concurrency-tuning.md`）：
     - 服务启动后就绪，读 stdout：`Maximum concurrency for <max-model-len> tokens per request: <N>x`，
       **每个 DP rank 一行** → 总并发预算 ≈ DP 数 × 单 rank 预测（实测：TP4/DP2+DCP2
       ≈ 289.67×2 ≈ 579；TP2/DP4 ≈ 118.57×4 ≈ 474；TP4/DP2 ≈ 156×2 ≈ 312；TP1/DP8 ≈ 33.38×8 ≈ 267）。
     - 固定 workload 仅作跨候选**对比基准**；真实吞吐上界必须按统一档位集（如 128/256/384/512/768/1024）
       加测预算的 50%/75%/100%，`num-prompts = max-concurrency`，画出吞吐-并发曲线找饱和点。
     - **提高 `MAX_NUM_SEQS` 才能让 bench 并发真正生效**：服务端 `--max-num-seqs` 是调度器同时处理
       序列数上限，低于 bench 并发时并发被截断（如 `--max-num-seqs 128` 时发 512 并发也只跑 128 个序列）。
       加测高档位前把服务端 `--max-num-seqs` 提到 ≥ 目标并发（如 1024），`HCCL_BUFFSIZE` 随之线性放大
       （maxBs 128→1024 时 NEEDED≈445MB×8≈3.6GB，按 NEEDED 设值），注意每 rank 显存占用增加、
       KV cache 会被压缩（读回 Maximum concurrency 实际值校准档位是否仍然成立）。
     - **⚠️ 并发上边界可能被 MoE dispatch 的 `HCCL_BUFFSIZE` 检查卡住，而非"并发上限已到"**：提高 `MAX_NUM_SEQS` 时若日志出现
       `HCCL_BUFFSIZE_EP is too SMALL ... NEEDED_HCCL_BUFFSIZE(...)=445MB, HCCL_BUFFSIZE=200MB, actual CCL_BUFFSIZE=400MB` 或
       `Check UB/window size or set workspace/hcomm cfg failed`（`aclnnMoeDistributeDispatchV4` error 561002），
       这是 vLLM-Ascend 的通信缓冲容量检查（需求随 `MAX_NUM_SEQS` 线性增长，默认 200MB、内部 ×2 用于 EP），
       应设置 `HCCL_BUFFSIZE` 大于日志中 NEEDED 值的一半（如 600~1024MB）后**重试同一并发候选**，而不是降并发或放弃。极值上调时注意每 rank 显存占用增加。
   - **⚠️ HCCL_BUFFSIZE 调低实验（显存优化）**：HCCL_BUFFSIZE 占用每 rank 显存，设得越大越浪费、可能压缩 KV
     cache 降低并发预算。高并发档位测试前，应**去掉 `HCCL_BUFFSIZE` env（默认 200MB）跑一档**，让报错日志
     打印 `NEEDED_HCCL_BUFFSIZE=xxxMB`，再配一个**更大的临近值**（如 NEEDED=445 → 设 600，勿一步设 2400）。
     调整后读回服务端 `Maximum concurrency` 验证 KV 预算是否改善；显存紧张时优先微调该值再考虑降 `GPU_MEMORY_UTILIZATION`。

5. **PCP/DCP**
   - 如果 `features.json` 中发现 PCP、DCP、prefill chunking/pipeline、dynamic chunked pipeline parallel 或 chunked pipeline parallel 参数，将其作为显式候选族。
   - PCP/DCP 候选必须同时比较 Output throughput、TTFT、TPOT、ITL、SMA 利用率和失败率。
   - 如果当前版本未发现 PCP/DCP 支持，要在报告中明确写“当前环境/文档未发现 PCP/DCP 可用参数”。

6. **KVCacheOffload / KV cache offload**
   - 如果发现 `KVCacheOffload`、`kv_cache_offload`、`kvcache_offload`、`cache_offload`、`cpu_offload`、`swap_space` 等参数，将其作为显式候选族。
   - offload 候选必须特别关注 host/device 传输导致的 TPOT/ITL 变差，以及 SMA 利用率是否下降。
   - 如果当前版本未发现 KVCacheOffload 支持，要在报告中明确写“当前环境/文档未发现 KVCacheOffload 可用参数”。

7. **显存利用率**
   - 在最佳有效点附近扫 `GPU_MEMORY_UTILIZATION`。
   - 接近最优时用 `0.001` 到 `0.005` 的小步长。
   - 过高可能导致稳定性或 HCCL 问题；过低可能降低 cache 能力和调度效率。

8. **最大模型长度**
   - 不要假设 `MAX_MODEL_LEN` 越小越快。
   - 必须给 chat template 和特殊 token 留余量。若 `input_len + output_len` 正好等于 `MAX_MODEL_LEN`，很容易 Bad Request。

9. **vLLM/vLLM-Ascend 特性**
   - 先 grep 当前安装源码里的 `VLLM_ASCEND_`，确认开关存在。
   - 一次只改一个特性，除非是在组合已验证有效项。
   - 启动失败就是当前模型/运行时不支持该候选；不要把它当作 benchmark 结果。

## 候选特性族

从 `features.json` 中按以下特性族筛选候选：

- **Serving/API**：OpenAI API server、端口、model alias、tokenizer、trust remote code、工具调用、chat template。通常只在影响请求路径或 tokenizer 开销时进入吞吐寻优。
- **并行与分布式**：TP、DP、EP、PP、expert parallel、distributed executor、Ray/multiprocessing、外部 DP。优先级最高，但要符合卡数和部署约束。
- **Scheduler/Batch**：`max-num-seqs`、`max-num-batched-tokens`、chunked prefill、DBO、request rate、并发、队列策略。
- **PCP/DCP**：PCP、DCP、prefill chunking/pipeline、dynamic chunked pipeline parallel、chunked pipeline parallel。
- **KV/Memory/Offload**：`gpu-memory-utilization`、`max-model-len`、KV cache dtype、prefix cache、KVCacheOffload、KV cache offload、cache offload、CPU offload、swap/cache 相关参数。
- **Compilation/Graph**：`compilation-config`、cudagraph capture sizes、eager/graph 模式、decode-only/full-graph 相关项。
- **Sampling/Output**：temperature、top-k、top-p、ignore-eos、max tokens。固定输出 benchmark 中，这些必须先锁定正确性，再谈吞吐。
- **Quantization**：量化格式、量化权重、W4A8/W8A8 等。只有权重和配置已存在时才纳入同轮比较。
- **LoRA/Adapter/Speculative/Guided/Multi-modal**：除非用户目标或 workload 涉及，否则只记录，不进入默认吞吐寻优。
- **Ascend 通信/融合/算子**：FlashComm、MLAPO、Fused MC2、NZ、TopK、Prefetch MLP、Matmul AllReduce、HCCL/ACL 相关环境变量。

## 常见 vLLM-Ascend 候选开关

- `VLLM_ASCEND_ENABLE_FLASHCOMM1`：MoE/expert-parallel 场景常见必要项或收益项。
- `VLLM_ASCEND_ENABLE_PREFETCH_MLP`：可能提升 decode，建议测开/关。
- `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE`：即使 deterministic sampling，也建议实测开/关。
- `VLLM_ASCEND_ENABLE_MLAPO`：可能有融合收益，但会消耗更多内存。
- `VLLM_ASCEND_ENABLE_NZ`：可能默认开启，也可结合 `additional_config.weight_nz_mode` 测试。
- `VLLM_ASCEND_ENABLE_FUSED_MC2`：MoE 融合算子，实验性；部分模型或 runtime 会启动失败。
- `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`：可能不支持；若出现属性缺失等启动错误，直接拒绝该候选。
- `HCCL_BUFFSIZE`：MoE 模型提高 `max-num-seqs` 的关键容量（默认 200MB，内部 ×2 作为 EP 用 `actual CCL_BUFFSIZE`）；并发候选报 `HCCL_BUFFSIZE_EP is too SMALL` 时按日志中 NEEDED 值上调（如 600），是"并发上边界"第一排查项。

## 有效性判断

有效 benchmark 必须满足：

- `Successful requests` 等于预期 prompts；
- `Failed requests` 为 0；
- 固定输出 benchmark 中，`Total generated tokens` 等于 `prompts * output_len`；
- 存在 `Output token throughput (tok/s)`。
- 存在对应 SMA 日志或明确说明本次为何无法采集 SMA。

以下结果按“先救后换”规则处理：先读日志判定可修复性，可修复则修复后重试同一候选（每个候选最多 2 次修复尝试），不可修复或修复耗尽才标记负样本并换下一个候选。

| 报错类型 | 常见根因 | 处置 |
| --------- | --------- | ------ |
| `Bad Request` | `MAX_MODEL_LEN` 太低、请求 schema 不匹配、model 参数不匹配 | 可修复：调大 `MAX_MODEL_LEN`（留 chat template 和特殊 token 余量）、修正请求参数后重试 |
| `Total generated tokens` 明显不足 | EOS 提前、generation config、服务错误 | 可修复：改用 `--temperature 0` + `--ignore-eos` 后重试 |
| 显存不足 / OOM | `gpu-memory-utilization`、`max-num-batched-tokens`、`max-num-seqs` 过高 | 可修复：降低显存占用参数后重试；若模型+拓扑本身放不下，换下一个拓扑候选 |
| `HCCL_BUFFSIZE_EP is too SMALL` / `Check UB/window size or set workspace/hcomm cfg failed` / `aclnnMoeDistributeDispatchV4` error 561002 | MoE dispatch/combine 的 HCCL 缓冲容量不足（`NEEDED_HCCL_BUFFSIZE` 随 `max-num-seqs` 线性增长，默认 200MB、内部 ×2 用于 EP） | **可修复**：调大 `HCCL_BUFFSIZE`（按报错日志中 NEEDED 值的一半以上，如 600~1024MB）后重试**同一并发候选**；引擎静默退出表现为 `SIGKILL rc=137` 且服务端无报错时也先 grep `HCCL_BUFFSIZE`/`NEEDED_HCCL` 再判定，勿直接归因"并发已达上限" |
| 端口冲突 | 端口被占用 | 可修复：更换 `PORT` 后重试 |
| 启动参数/环境变量拼写错误 | 命令或脚本笔误 | 可修复：对照 `vllm serve --help` 和源码修正后重试 |
| HCCL count mismatch / 巨大异常分配 | 通信或 sampling 路径不稳定 | 先重试一次并尝试 deterministic temperature 和已知良好的 HCCL/FlashComm；持续失败 → 不可修复，拒绝该候选 |
| Server startup failure（属性缺失/算子不支持） | 该特性/拓扑在当前模型或 runtime 不支持 | 不可修复：负样本，换下一个候选 |
| 需要重编译、量化权重或重搭分布式架构 | 高风险变更，无法快速回滚 | 不可修复（本寻优范围内）：负样本，换下一个候选 |
| 设备故障 / 环境损坏 | 硬件或依赖损坏 | 不可修复：停止寻优并上报，不得继续消耗寻优时间 |

## Qwen3-30B-A3B 四卡实战经验

这些是经验样例，不是所有模型的绝对规律：

- TP4/DP1 在长上下文 MoE 工作负载上优于 TP2/DP2。
- `MAX_NUM_BATCHED_TOKENS=32768` 优于 28672、36864、40960。
- `GPU_MEMORY_UTILIZATION=0.965` 是观察到的稳定高点附近。
- 并发从 48 提到 56/64 后吞吐下降，原因是尾部阻塞和 TTFT 拉长。
- 并发低于 48 时，总输出吞吐也下降。
- `MAX_MODEL_LEN=36864` 对 `32768+4096` 工作负载会 Bad Request；37120 有效但慢于 40960。
- 极小幅度单次提升可能是噪声，必须复跑确认。

## 最终报告格式

报告中必须包含：

- `features.json` 和 `features.md` 路径；
- 本次采用的参考文档网址，默认 `https://docs.vllm.ai/projects/ascend/en/latest/`，如有修改要写明；
- 预检确认的 vLLM 与 vLLM-Ascend 实际版本，以及用户确认记录；
- 最佳有效吞吐，并说明是单次最高还是复跑确认；
- PCP、DCP、KVCacheOffload/KV cache offload 是否发现、是否测试、测试结果或未测试原因；
- SMA/NPU 利用率日志路径和平均利用率摘要；
- 精确环境变量和脚本/命令（含 `ASCEND_RT_VISIBLE_DEVICES`）；
- 模型、工作负载、设备列表；
- TP、DP、EP；
- benchmark log 和 server log；
- 被拒绝候选及原因；
- 如果最长寻优时间耗尽，列出尚未测试的高价值候选。
