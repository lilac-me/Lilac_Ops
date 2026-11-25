# AgentLoop 推理算子优化实机教程（AISBench）

分支：agentloop-infer-opt-lite。测试对象：已可运行的 vLLM-Ascend 服务。正式服务性能验收优先采用 AISBench；现有 vLLM benchmark 仍用于 Profiling 工作负载和快速排障。两者指标不可混用为前后对照。

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
git clone --single-branch -b agentloop-infer-opt-lite https://github.com/lilac-me/ops.git AgentLoop-Infer
cd AgentLoop-Infer/AgentLoop
bash scripts/test_init.sh
bash init.sh project claude /path/to/vllm-ascend
cd /path/to/vllm-ascend
claude
```

Codex / OpenCode 替换 claude 参数及启动命令。目标目录必须是实际代码仓。CANNBot 双插件先按上游安装并确保在同一 Agent 可见。

## 2. 准备 AISBench（服务压测客户端）

优先使用已有 AISBench 环境，并记录版本/commit，避免改动 vLLM 服务环境的依赖。若没有，可在独立 Python 环境安装；官方 vLLM-Ascend 文档示例：

```bash
git clone https://github.com/AISBench/benchmark.git AISBench
cd AISBench
python3 -m pip install -e ./ --use-pep517
python3 -m pip install -r requirements/api.txt
python3 -m pip install -r requirements/extra.txt
ais_bench -h
git rev-parse HEAD
```

依赖及 CLI 随版本变化，以实际 checkout 文档和 --help 为准。客户端需能访问模型 tokenizer 文件及测试服务地址。可独立于 NPU 服务进程运行。不要在公共网络开启 profiler API。

## 3. 固定推理实验

使用已有 vLLM-Ascend 启动命令，固定模型、权重、tokenizer、vLLM/vLLM-Ascend/CANN 版本、NPU、TP/DP/EP、量化、graph、KVCache、调度及采样。以下仅为测试模板（按实际配置替换，不强制改动已跑通的服务）：

```bash
vllm serve /path/to/weights --served-model-name test-model \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --max-model-len 4096
```

另一个终端（与服务同机时）：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

建议第一天用两种固定负载（示例，按资源能力调整，非性能目标）：
| 场景 | ISL | OSL | 并发 | 主要观测 |
| --- | ---: | ---: | ---: | --- |
| Prefill 偏重 | 2048 | 128 | 4 | TTFT、输入吞吐、MatMul |
| Decode 偏重 | 256 | 512 | 8 | TPOT、输出吞吐、KV 访存及 Launch |

先用固定 demo 数据集完成 8 请求冒烟，再使用 AISBench synthetic 或冻结的自定义请求集控制真实 ISL/OSL。记录实际 token 数；chat template 可能改变输入长度。ignore_eos=True 仅用于固定输出长度的性能实验，精度测试应使用约定的正常生成语义。请求、并发、到达率、采样、缓存冷暖和预热规则保持前后一致。

## 4. AISBench 性能基线及回归

按实际版本准备 GSM8K demo 数据集（下载说明见上游文档），查询模型/数据集配置路径：

```bash
ais_bench --models vllm_api_stream_chat \
  --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt \
  --mode perf --search
```

修改实际模型 task 配置（或独立自定义配置），至少核对以下字段：

| 字段 | 示例/含义 |
| --- | --- |
| path | 客户端可访问的实际权重/tokenizer 路径 |
| model | test-model，与服务 served-model-name 一致 |
| host_ip / host_port | 127.0.0.1 / 8000（远端按实际地址） |
| batch_size | 客户端最大并发，示例 4 |
| request_rate | 0 或明确固定的发送速率，以该版本语义为准 |
| max_out_len | 实际 OSL 目标 |
| generation_kwargs | 固定采样；性能固定长度场景可设 temperature=0、ignore_eos=True |

首次可加 --debug 排查服务请求错误，正式测试建议去掉以保留日志。命令使用流式 backend，避免非流式无法统计首 token 和 token 间隔。

```bash
ais_bench --models vllm_api_stream_chat \
  --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt \
  --summarizer default_perf --mode perf --num-prompts 8 --debug
```

冒烟通过后用正式冻结 workload、独立输出目录执行。示例：先预热，不计入正式结果；再重复三次。以下 dataset 名称和 num-prompts 替换为已冻结的正式任务配置；确保实际请求数满足并发。baseline 与 regression 使用完全相同命令及配置，只改变输出路径及模型优化代码。

```bash
for round in 1 2 3; do
  ais_bench --models vllm_api_stream_chat \
    --datasets YOUR_FIXED_DATASET \
    --summarizer default_perf --mode perf \
    --work-dir "./outputs/agentloop/baseline/run_${round}"
done
```

优化后重启服务、同规则预热，再以 regression/run_${round} 目录运行三次。保存原始 summary JSON、逐请求 CSV、配置快照、日志；文件实际路径按 AISBench 输出 Base path 核对。若当前版本没有 --work-dir 或 --num-prompts，按其 --help 或自定义配置设置，不静默替换 workload。

## 5. Agent 执行算子优化

在 Agent 中输入：

> 使用 agentloop-infer-opt。服务采用当前项目已验证的 vLLM-Ascend 启动脚本，固定模型、拓扑、图模式、量化和请求集。用 AISBench 流式压测建立无 Profiling 的 TTFT/TPOT/ITL/Throughput 基线；独立采集 Prefill、Decode 的 Level1 + PipeUtilization + record_shapes Profiling，核对实际阶段和 rank。先选择一个真实源码明确的热点 Triton 算子，按实现语言调用 CANNBot，验证真实 shape、精度和性能。Ascend C/CANN 原生热点走 tilelang2ascendc-ops-generator，连续 Vector 先核实真实数据流后再融合，MatMul 以真实形态判断 MFU 或 Latency。优化后在隔离环境集成、重启服务，使用同一套 AISBench 配置回归。保留原始产物；失败回滚，不伪造收益，不改生产服务。

采集使用现有 ascend-profiling-collection Skill：从已预热的服务使用实际版本的 profiler 控制，head/middle/tail 只是窗口，不等于实际 Prefill/Decode。建议 decode 采 10–20 个 worker iteration；需验证真实阶段和批次，缺证据标 unknown。检查实际 worker 是否真正开启 Level1、Text、PipeUtilization、record_shapes、with_stack；原生 CLI 并不保证这些参数。正式 AISBench 前后性能压测必须关闭 profiler。

### 手动排障入口

每个 rank 的真实 profile 同步到分析环境，使用 AgentLoop 根下命令：

```bash
python3 plugins/ops/engine/infer_profile_analyze.py \
  --profile /path/to/copied_rank_profile --rank 0 --phase unknown \
  --model-root /path/to/vllm-ascend --source-map /path/to/source-map.json \
  --out ./outputs/infer_run_001
```

若已取得实际 worker/scheduler 证据，改为 --phase prefill/decode/mixed 并加 --phase-evidence。输出 profile_contract、candidates、opportunities、operator_routes、requests/task_*。source-map 用实际 kernel→源码映射，不能凭名字猜。读取已核定任务的 request.json/prompt.txt 调用相应插件，或配置实际 Agent CLI 后运行：

```bash
python3 plugins/ops/engine/dispatch_operator_tasks.py \
  --run ./outputs/infer_run_001 \
  --config /path/to/cannbot-cli.json --task task_000
```

CLI 成功只是完成调度，必须独立检查真实算子精度与性能。推理自动补丁集成和 AISBench 结果自动解析/回归编排尚未实现；明天以隔离环境手工集成、同 workload AISBench 回归为实际验收路径。不可伪造自动执行。

## 6. AISBench 指标及验收

| 指标 | 单位 | 判定方向 |
| --- | --- | --- |
| Mean/P99 TTFT | ms | 降低，首 token 延迟 |
| Mean/P99 TPOT | ms/token | 降低，首 token 之后每 token 时间 |
| Mean/P99 ITL | ms | 降低，相邻 token 间隔 |
| Mean/P99 E2E | ms | 降低，请求完整延迟 |
| Request Throughput | req/s | 提升 |
| Input/Output/Total Token Throughput | token/s | 分别报告，不能混为一个口径 |
| Success/Failed Requests | count | 请求均成功、无新增失败 |
| Total Input/Generated Tokens | count | 实际工作量可比 |

区分逐请求 OutputTokenThroughput 的平均与整服务 Output Token Throughput；整网对比使用后者。报告三轮的中位数、各轮原值和波动；尾延迟指标以各轮实际 P99 对照，不把跨轮平均伪装成全样本 P99。固定请求集另做精度对照（正常采样语义、必要时 logprob/任务精度）；性能数据不能代替精度结果。

| 指标 | Baseline | Optimized | 变化 |
| --- | --- | --- | --- |
| Mean TTFT | 待测 | 待测 | - |
| P99 TTFT | 待测 | 待测 | - |
| Mean TPOT | 待测 | 待测 | - |
| P99 TPOT | 待测 | 待测 | - |
| Output Token Throughput | 待测 | 待测 | - |
| Request Throughput | 待测 | 待测 | - |
| 成功/失败数 | 待测 | 待测 | - |
| 实际输入/输出 tokens | 待测 | 待测 | - |
| 精度对照 | 待测 | 待测 | - |

同场景吞吐提升 = (optimized / baseline - 1) × 100%；延迟降低 = (1 - optimized / baseline) × 100%。首轮不设不现实的固定收益目标，先保证指标真实、可比和可复现。若 TTFT/TPOT 互有退化，按约定 SLO 判断，不直接采纳。

## 7. 已知边界及官方参考

- 当前已提交初始化、服务/采集脚本、热点与路由、双插件 CLI 调度；真实 NPU 联合测试未完成。AISBench 在本版为正式验收工具的使用约定，不新增第三套自动化框架。
- [vLLM-Ascend Using AISBench](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_ais_bench.html)
- [AISBench Service-Oriented Performance Evaluation](https://github.com/AISBench/benchmark/blob/master/docs/source_en/base_tutorials/scenes_intro/performance_benchmark.md)
- [vLLM-Ascend service profiling guide](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html)
