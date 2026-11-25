---
name: vllm-ascend-autotune-zh
description: 面向 Ascend/NPU 环境自动发现并寻优 vLLM 与 vLLM-Ascend 推理服务参数，以最大化 Output token throughput，并在推理/benchmark 过程中采集和打印 SMA/NPU 利用率日志。适用于用户要求列出当前版本 vLLM/vLLM-Ascend 全部可发现特性、优化吞吐、全局寻优、局部细扫、比较 TP/DP/EP、调优 PCP、DCP、KVCacheOffload、KV cache offload、max-num-batched-tokens、max-num-seqs、gpu-memory-utilization、vLLM-Ascend 环境变量、additional-config、compilation-config，或要求在指定最长寻优时间内达到目标吞吐的场景；支持 Qwen、DeepSeek、GLM、Kimi、Mixtral 以及其它模型。
targets-repo:
  - vllm-ascend
---

# vLLM Ascend 自动发现与寻优

## 目标

先发现当前运行环境里的 vLLM/vLLM-Ascend 特性，再基于发现结果执行吞吐寻优，直到满足以下任一条件：

- 达到并验证用户指定的目标 `Output token throughput (tok/s)`；
- 达到用户指定的最长寻优时间；
- 安全候选空间已基本穷尽，或遇到无法继续推进的真实阻塞。

优化时只比较有效 benchmark。失败请求、生成 token 不足、Bad Request、服务启动失败、EOS 提前结束等结果只能用于诊断，不能当作吞吐提升。

**TP/DP/EP 拓扑搜索是吞吐寻优的强制第一阶段**：进入 scheduler、PCP/DCP、KVCacheOffload 等其它参数寻优之前，必须先完成并行策略拓扑搜索（在卡数约束内比较全部合法 TP/DP/EP 组合），否则视为寻优未完成，不得对外宣称已找到最优配置。更换拓扑需要重启服务的场景（如流水线中被编排器调度），必须在重启前把已测拓扑结果写入 leaderboard，重启后继续累计，避免重复测试。

## 两层结构

1. **全特性发现层**
   - 从当前目标环境执行 `vllm serve --help`、`vllm bench serve --help`。
   - 定位 `vllm` 与 `vllm_ascend` 安装路径。
   - 扫描 `vllm_ascend/envs.py`、源码和 docs，提取 `VLLM_ASCEND_*`、additional-config 相关键、源码中可发现的 enable/use/weight/torchair/ascend 配置项。
   - 显式扫描 PCP、DCP、KVCacheOffload/KV cache offload/cache offload/CPU offload 相关关键词，确认当前版本是否支持。
   - 产出 JSON 和 Markdown 特性清单，作为后续候选空间依据。

2. **吞吐寻优层**
   - 从全特性清单中筛选“安全、可控、可能影响吞吐”的参数进入候选矩阵。
   - 按拓扑、scheduler、PCP/DCP、KVCacheOffload、memory、model length、compilation-config、vLLM-Ascend feature flags 的顺序逐步搜索。
   - 对新最优复跑确认，避免把噪声当作稳定提升。

当用户要求“包含所有特性”或“全局最优”时，必须先执行或更新全特性发现层；不要只依赖历史经验清单。

## 必要输入

开始前收集或从上下文推断：

- 访问方式：直连 SSH、跳板机、目标机器、容器名。
- benchmark 脚本路径或完整命令模板。
- 模型路径和 tokenizer 路径。
- 参考文档网址，默认 `https://docs.vllm.ai/projects/ascend/en/latest/`；用户可以改成固定的 vLLM/vLLM-Ascend 文档版本或内部文档地址。
- 设备集合，例如 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`；必须显式传给启动脚本，由脚本写入 benchmark/server/SMA 子进程的环境，不能只依赖当前 shell 临时 `export`。
- 最长寻优时间，例如 `2h`、`3600s`。
- 目标吞吐；如果用户没有给，可以按“尽可能提升”执行。
- 固定工作负载：`num_prompts`、`max_concurrency`、输入长度、输出长度、temperature、ignore-eos、有效性标准。
- 硬约束：卡数、允许的 TP/DP、是否允许杀进程释放资源。

如果用户输入与远端当前状态冲突，先检查当前状态，并说明采用的假设。

## 启动前强制门禁

启动任何基线、候选 benchmark、推理 server 或正式寻优前，必须先运行 `scripts/preflight_autotune.py`。预检必须打印并向用户确认：

- 当前 Python 环境中的 vLLM 版本；
- 当前 Python 环境中的 vLLM-Ascend 版本；
- 本次采用的参考文档 URL；
- 将写入启动子进程的 `ASCEND_RT_VISIBLE_DEVICES` 设备集合。

预检会提示“是否需要更改版本或设备集合？”。用户回答需要更改、回答不明确或未完成确认时，立即停止，不得启动寻优；只有用户明确确认沿用当前版本和设备集合后，才允许继续。若要更换版本，先更新目标环境并重新执行预检。

## 操作规则

- **大任务开始前提醒**：每次开始寻优/全量对比任务前，必须主动向用户说明以下默认行为，请求获准后执行：
  "我会在服务启动后读取 vLLM 的 `Maximum concurrency` 日志推算并发预算，若预测并发高于当前 benchmark 档位，会**自动加测更高并发档位**（预算的 50%/75%/100%，取整落入统一档位集 128/256/384/512/768/1024）来寻找真实吞吐上界；固定 workload 的对比基准仍会保留。"
  用户未明确同意前不得自动加测；同意一次后，整个任务期间无需逐次询问。
- **并发预算自适应（基于日志，不用用户提醒）**：每次服务启动/重启后就绪后，读取服务端 stdout 中
  `Maximum concurrency for <max-model-len> tokens per request: <N>x`（**每个 DP rank 一行，总并发预算 ≈ DP 数 × 单 rank 预测**），
  记录到 leaderboard 与报告。当预算 > 当前测试并发时，按统一档位集自动加测高档位
  （参考 `vllm-ascend-benchmark/references/max-concurrency-tuning.md` 的方法与实测表）。
  加测高档位时 num-prompts 必须 = max-concurrency（保证稳态覆盖）。**低并发只能测得下限，
  预测能撑满的配置才是真实能力上限**；只要预算允许就应测到档位集的最高档，不要停在用户
  固定的并发数上。
- 长时间寻优要持续给用户简短进展，避免用户以为断线。
- 只清理本次寻优范围内的 benchmark/server/worker 进程；除非用户允许，不要影响无关进程。
- benchmark 脚本是权威来源。先读脚本，确认哪些环境变量和 CLI 参数真实生效。
- 每次启动都必须通过 `scripts/run_with_sma.py --devices <设备集合>` 注入 `ASCEND_RT_VISIBLE_DEVICES`；在生成的命令、日志头和子进程环境中保留同一设备集合。
- 每次正式推理/benchmark 都要同步采集 SMA/NPU 利用率；除非用户明确跳过，使用 `scripts/run_with_sma.py` 或等价方式把 SMA 输出打印到过程日志。
- 维护 leaderboard：配置、日志路径、有效性、`Output token throughput`。
- 有效性优先于数值。无效结果只用于诊断，不参与“最优”比较。
- 全局寻优要分阶段：先全特性发现，再拓扑，再 batch/scheduler、PCP/DCP、KVCacheOffload、memory，再 compilation 和 vLLM-Ascend 特性开关。

## 错误分析与自愈（先救后换）

任何候选（开关或参数值）出现报错时，**禁止直接跳到下一个候选**，必须先执行错误分析：

1. **收集证据**：读取本次运行的 server log、benchmark log、SMA 日志与完整启动命令，定位报错原文与根因。
2. **判定可修复性**：
   - **可修复**：参数取值不当（`MAX_MODEL_LEN` 过小、`gpu-memory-utilization`/`max-num-batched-tokens` 过大导致 OOM、并发过高）、生成参数问题（EOS 提前、需 `--temperature 0` + `--ignore-eos`）、端口冲突、环境变量拼写错误、启动参数拼写错误。
   - **不可修复**：当前模型/runtime 不支持该特性（属性缺失、算子不支持）、需要重编译或量化权重等高风险变更、设备或环境本身故障。
3. **处理**：
   - 可修复 → 修复后重试**同一候选**，每个候选最多 2 次修复尝试；每次修复动作与结果写入 leaderboard。
   - 不可修复或修复尝试耗尽 → 将该候选标记为负样本，写明报错与判定原因，再进入下一个候选。
4. 修复尝试与最终判定必须写入最终报告的“被拒绝候选及原因”，避免后续重复踩同一个坑。

## 工作流

0. 执行启动前预检。
   - 运行 `scripts/preflight_autotune.py --devices 4,5,6,7 --reference-doc-url https://docs.vllm.ai/projects/ascend/en/latest/`。
   - 记录并向用户展示 vLLM 与 vLLM-Ascend 的实际版本；用户明确确认后才进入下一步。
   - 若设备集合或版本需要更改，先修改目标环境/启动参数，重新执行预检，不得绕过门禁。

1. 检查脚本和环境。
   - 读取 benchmark 脚本。
   - 找出可覆盖变量，例如 `BS`、`NUM_PROMPTS`、`MAX_NUM_SEQS`、`MAX_MODEL_LEN`、`MAX_NUM_BATCHED_TOKENS`、`TENSOR_PARALLEL_SIZE`、`DATA_PARALLEL_SIZE`、`GPU_MEMORY_UTILIZATION`、`LOG_DIR`、`PORT`。
   - 确认生成参数。固定输出长度压测通常需要 `--temperature 0` 和 `--ignore-eos`，避免模型配置导致提前结束。

2. 发现当前版本全特性。
   - 在目标容器/环境内运行 `scripts/discover_vllm_features.py`。
   - 保存 `features.json` 和 `features.md` 到本次寻优日志目录。
   - 将预检确认过的参考文档网址传给 `--reference-doc-url` 并写入发现报告；如果用户提供了其它 URL，以用户输入为准。
   - 若脚本不能直接复制到远端，可把脚本内容放入目标环境临时文件执行，或在目标环境用等价命令收集 `vllm --help`、`vllm_ascend/envs.py`、docs grep 结果。
   - 详细方法见 `references/feature-discovery-zh.md`。

3. 建立有效基线。
   - 优先跑用户当前默认配置，除非已有可信近期日志证明基线。
   - 基线运行必须同步采集 SMA；记录 `sma.log`、`npu_util.log`、`combined.log`。
   - 检查 successful requests、failed requests、generated tokens、throughput。
   - 记录 server log 和 benchmark log。

4. 从全特性清单筛候选。
   - 优先选择会影响吞吐且可通过环境变量或脚本参数隔离的项。
   - PCP、DCP、KVCacheOffload/KV cache offload 是显式候选族；即使当前自动发现结果为空，也要在报告里说明“未发现支持项/未纳入原因”。
   - 暂缓高风险项：会改变输出语义、要求重编译、要求量化权重、要求分布式架构重搭、或无法快速回滚的配置。
   - 对未知 vLLM-Ascend 环境变量，先查源码/文档作用，再决定是否进入候选矩阵。

5. 搜索高影响拓扑（强制第一阶段，先于第 6/7/8 步的所有其它参数寻优）。
   - 在卡数约束内比较 TP/DP/EP：枚举全部合法组合，逐个 benchmark，在固定 workload 下比较 Output token throughput、TTFT、TPOT 和有效性。
   - 卡数 ≥ 2 时至少实际测试 2 种不同拓扑；只有一种合法拓扑时也要记录并写明理由。
   - MoE 模型尽早测试 `--enable-expert-parallel`。
   - 更换拓扑需要重启服务时，把每个已测拓扑（含吞吐与有效性）写入 leaderboard；重启后从 leaderboard 续接，避免重复测试。
   - 如果某个通信特性是启动或正确性的必要条件，将其锁定，不再重复测试。
   - 拓扑候选穷尽并选出阶段最优拓扑后，才允许进入第 6 步的 scheduler 与 memory 寻优。

6. 调 scheduler 与 memory。
   - 先扫 `MAX_NUM_BATCHED_TOKENS`，围绕长上下文和卡数选择候选。
   - 若当前版本发现 PCP/DCP 相关参数，按文档约束分别测试 prefill chunking/pipeline 与 dynamic chunked pipeline parallel 对 TTFT、TPOT、Output throughput 的影响。
   - 若发现 KVCacheOffload/KV cache offload/cache offload 参数，单独建立 offload 候选；必须同时记录吞吐、TTFT/TPOT、SMA 利用率和是否出现 host/device 传输瓶颈。
   - 再扫 `GPU_MEMORY_UTILIZATION`，接近最佳点后用小步长细扫。
   - 谨慎调整 `MAX_NUM_SEQS`、`BS`、`NUM_PROMPTS`；比较吞吐时保持总工作负载可比。
   - `MAX_MODEL_LEN` 要留出 chat template 和特殊 token 余量，贴得过紧很容易 Bad Request。

7. 调 compilation 与 vLLM-Ascend 特性。
   - 根据发现结果选择 `compilation-config`、`cudagraph_capture_sizes`、additional-config、`VLLM_ASCEND_*`，以及 PCP、DCP、KVCacheOffload 相关 CLI/env/additional-config。
   - 一次只改一个开关，除非是在组合已验证赢家。
   - 启动失败或 benchmark 报错时，先按“错误分析与自愈（先救后换）”规则读日志判定可修复性：可修复则修复重试（最多 2 次），不可修复或修复耗尽才是该模型/运行时上的负样本，不是 benchmark 结果。

8. 验证最优。
   - 新最优只高一点点时必须复跑，排除噪声。
   - 用户要“单次最高”时可以报告单次峰值；否则优先报告稳定最优。
   - 结束前确认没有遗留本次寻优产生的 server/worker 进程，除非用户要求保留服务。

## SMA/NPU 利用率采集

本 skill 内置用户提供的两个 SMA 脚本：

- `scripts/sma.py`：每秒调用 `npu-smi info -t usages`，打印各 rank Aicore 利用率，并写 `npu_util.log`。
- `scripts/npu_util.py`：读取 `npu_util.log`，输出总利用率、样本数、平均利用率。

推荐用包装脚本运行推理/benchmark：

```bash
python scripts/run_with_sma.py \
  --devices 4,5,6,7 \
  --log-dir /path/to/logs/run-001 \
  --prefix qwen3-30b \
  -- python /path/to/your_benchmark_or_script.py
```

如果 benchmark 是 shell 脚本：

```bash
python scripts/run_with_sma.py \
  --devices 4,5,6,7 \
  --log-dir /path/to/logs/run-001 \
  --prefix qwen3-30b \
  -- bash /home/w00893744/optimization/run-qwen3-30B-A3B-vllm-bench.sh
```

包装脚本会同时生成：

- `<prefix>-bench-*.log`：推理/benchmark 输出；
- `<prefix>-sma-*.log`：SMA 实时输出；
- `<prefix>-combined-*.log`：benchmark 与 SMA 带时间戳的合并过程日志；
- `npu_util.log`：原始 SMA 采样文件；
- `sma_summary=...`：由 `npu_util.py` 计算出的平均利用率摘要。

若不能使用包装脚本，则在启动推理前手动后台运行：

```bash
cd /path/to/log-dir
python /path/to/scripts/sma.py 2>&1 | tee sma.log &
SMA_PID=$!
# run benchmark here
kill ${SMA_PID}
python /path/to/scripts/npu_util.py | tee -a sma.log
```

## 特性发现

在目标环境中运行：

```bash
python scripts/discover_vllm_features.py \
  --reference-doc-url https://docs.vllm.ai/projects/ascend/en/latest/ \
  --json-out /path/to/logs/features.json \
  --markdown-out /path/to/logs/features.md
```

若在容器中执行：

```bash
docker exec <container> python /path/to/discover_vllm_features.py \
  --reference-doc-url https://docs.vllm.ai/projects/ascend/en/latest/ \
  --json-out /tmp/features.json \
  --markdown-out /tmp/features.md
```

发现结果用于回答“当前环境包含哪些 vLLM/vLLM-Ascend 特性”，也用于扩大寻优候选空间。不要把 skill 内置示例当作完整特性列表。

## 候选生成

使用 `scripts/make_candidates.py` 生成分阶段候选矩阵：

```bash
python scripts/make_candidates.py \
  --devices 4,5,6,7 \
  --cards 4 \
  --max-seqs 48 \
  --baseline-throughput 745.83 \
  --target-throughput 780 \
  --max-time-min 120
```

该脚本输出 JSONL，每行包含环境变量覆盖和候选理由。它只是起点，不能替代读取实际脚本、全特性发现结果和日志。

详细调参策略见 `references/tuning-guide-zh.md`；全特性发现策略见 `references/feature-discovery-zh.md`。

## 结果解析

有效结果必须满足：

- `Successful requests` 等于预期请求数；
- `Failed requests` 为 `0`；
- 固定输出 benchmark 中 `Total generated tokens` 等于 `num_prompts * output_len`；
- 存在 `Output token throughput (tok/s)`。

使用 `scripts/parse_vllm_bench.py` 汇总日志：

```bash
python scripts/parse_vllm_bench.py /path/to/logs/**/*.log --expected-prompts 48 --expected-output-len 4096
```

最终报告必须包含：

- 特性发现文件路径：`features.json`、`features.md`；
- 本次采用的参考文档网址；
- 预检确认的 vLLM 与 vLLM-Ascend 实际版本，以及用户是否确认沿用；
- SMA 文件路径：`sma.log`、`npu_util.log`、`combined.log`，以及 `npu_util.py` 输出的平均利用率；
- 精确命令或环境变量；
- 模型、工作负载、设备集合（含 `ASCEND_RT_VISIBLE_DEVICES`）、TP/DP/EP；
- 吞吐和有效性字段；
- benchmark log 和 server log 路径；
- 被拒绝候选及原因（含每个候选报错的分析结论、修复尝试次数与动作、最终判定）；
- PCP、DCP、KVCacheOffload 是否发现、是否测试、测试结果或未测试原因；
- 若达到最长寻优时间，说明剩余未测试候选。
