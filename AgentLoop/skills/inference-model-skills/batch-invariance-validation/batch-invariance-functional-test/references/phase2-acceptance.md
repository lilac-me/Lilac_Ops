# Phase 2 功能验收

命令示例以本 Skill 根目录（与 `SKILL.md` 同级）为工作目录。

## 目录

1. 参数接口
2. 验收矩阵
3. 输出与判定

## 1. 参数接口

`scripts/run_phase2_acceptance.sh` 每次执行一个外部定义的场景，对 TP、BS 和长度对取笛卡尔积，并在普通组合失败时立即退出非零。每个目标 TP 先验证模型能否正常加载并进入推理；若仅因权重无法装入设备（包括模型加载阶段 OOM），则标记 `SKIPPED_TP_RESOURCE_UNAVAILABLE`，保留原始错误并跳过该 TP 的后续 BS、长度和轮次。该资源性跳过不阻塞 Phase 2 整体通过，但不得记为该 TP 通过或使用其他 TP 结果外推；模型加载后的推理 OOM 仍按失败处理。Phase 2 判定通过时至少应有一个目标 TP 完成全部验收。

TP 计划由编排器根据用户限定的可见设备和模型结构自动生成，展示后可在同一回合写入 run_config.json 并继续，不等待额外确认。用户明确指定的 TP 必须作为 --requested-tps 约束传入规划器；无法满足时请求用户决定，禁止静默覆盖。自动规划入口为插件根目录下的 python -m engine.gates auto-plan，传入 --candidate、--model-config、--output 和 --schema。未提供加载探测结果时记录 load_probe_status=not_probed，仅表示结构候选计划；环境准入完成后由阶段执行验证实际加载能力，不能声称已通过容量验证。各主阶段采用最大合法 TP，矩阵覆盖合法的 1/2/4。范围内重规划自动记录变更历史，并从环境准入重新验证受影响阶段；旧结果不得外推。runner 的 VLLM_TEST_TP_LIST 必须由落盘计划显式传入，缺失仍返回非零。

runner 会在环境与算子预检前校验 `VLLM_BATCH_INVARIANT`。变量未设置时自动导出为 `1`；若调用方已设置，则值必须严格等于 `1`，显式传入 `0`、空字符串或其他值会输出 `BATCH_INVARIANT_MODE_REQUIRED` 并立即退出非零。pytest 用例也执行同一检查，不能在普通推理模式下形成阶段二通过结果。

A5 与 A2/A3 使用同一 Phase 2 runner，且必须先完成 A5 Phase 1 白名单分析。runner 会实检 Python 包、custom op SO 和 Torch 算子注册；缺失时由环境预检使用 `ascend950` 自动构建并完整复检。

prompt 生成器先使用被测模型 tokenizer 构造 token ID，再以 `TokensPrompt` 形式直接送入 vLLM。`VLLM_TEST_MAX_PROMPT_LEN` 是精确的基准 token 长度；例如设置为 20480 时，先以 20480 token 为基准，再为每条请求独立增加 `0..VLLM_TEST_PROMPT_LEN_VARIATION` 个 token。每条请求的实际输入长度必须严格等于对应的 `基准长度 + 本条随机增量`，而不是所有请求都固定等于基准长度。日志必须出现 `BATCH_INVARIANCE_EXACT_PROMPT_LENGTH` 且 `all_exact=true`，并记录逐条目标长度；否则标记 `BLOCKED_EXACT_PROMPT_LENGTH`，不得把近似长度结果作为正式证据。输出使用 `SamplingParams(ignore_eos=True)`，BS=1 和 BS=N 的每个请求都必须生成恰好 `max_tokens` 个 step；提前结束视为当前组合失败。比较时先验证完整 token ID 序列一致，再比较相同 token 路径上的 chosen-token logprob 是否按位一致。

当前自动化入口仅支持 TP 场景，通过 `VLLM_TEST_TP_LIST` 设置。PP、EP、DP 在方法上理论可复用但尚未验证，runner 没有对应配置入口；其他并行方式及混合并行仅属于可能支持。请求这些并行模式时，不得通过附加未识别环境变量伪造覆盖，也不得把 TP 结果外推为其通过证据。

采样参数仅泛化 `temperature`，通过 `VLLM_TEST_TEMPERATURE` 设置有限非负值，例如 `0` 或 `0.6`。`seed` 用于复现，`max_tokens`、`ignore_eos`、`logprobs` 用于长度或证据采集，均不作为采样泛化维度。当前不支持配置 `top_p`、`top_k`、`min_p`、presence/frequency/repetition penalty 等其他采样参数。runner 检测到对应的 `VLLM_TEST_*` 环境变量时输出 `UNSUPPORTED_SAMPLING_PARAMETER` 并返回非零，不会静默忽略。

当前验收不支持请求驱逐场景。完整读取 `scheduler-eviction.md`；保持 `disable_log_stats=False`，runner 会为每次 attempt 生成 `*_scheduler.json`。容量不足或静态 BS=N 的驱逐状态转移均返回非零并标记 `BLOCKED_EVICTION_UNSUPPORTED`。

长输入场景固定使用 `enable_chunked_prefill=False` 和 `block_size=128`，并与 Phase 1 profiling、Phase 3 Math500 的实际服务配置保持一致。为避免 `BS × prompt_len` 形成过大的单步 profiling 预算，未显式指定 `VLLM_TEST_MAX_NUM_BATCHED_TOKENS` 时，runner 将自动值封顶为 `VLLM_TEST_AUTO_MAX_NUM_BATCHED_TOKENS_CAP`，默认 `131072`，同时保证不低于单请求的 `prompt_len + max_tokens`。该调整只允许请求分波调度，不改变提交的 TP、BS、输入长度或 KV 容量要求。初始 `Running<BS, Waiting>0` 属于正常排队，不要求必须观察到 `Running=BS, Waiting=0`；只有满批运行后回退为运行数减少且等待数增加才判定驱逐。TP=4、BS=64 若在模型加载完成后因请求规模或运行时显存不足而失败，仍按失败处理，不能缩小 TP/BS 后外推通过。

调用方必须显式设置 `ASCEND_RT_VISIBLE_DEVICES`；runner 不默认选择设备，也不推算硬件拓扑。每个 attempt 在 pytest 前创建日志并记录设备、场景、模型、TP 和 BS。LLM 初始化明确出现 OOM、内存分配失败或 `No available memory for the cache blocks` 时标记 `SKIPPED_TP_RESOURCE_UNAVAILABLE`。只有同一日志行同时包含 `free memory on device`、`desired` 以及 `<` 或 `less than` 时，才视为瞬时显存不足并保持配置不变最多重试 2 次；`on startup. Desired GPU memory utilization` 等正常资源统计 INFO 不得触发重试。重试仍失败时保留全部日志并交由 Agent 分析，无法确定时反馈用户。其他启动错误不自动重试或跳过。

以下变量必须提供：

| 变量 | 格式 | 说明 |
| ------ | ------ | ------ |
| `VLLM_TEST_MODEL` | 路径 | 测试模型 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3` | 用户显式指定的目标 NPU ID 列表 |
| `VLLM_TEST_TP_LIST` | `1,2,4` 或 `"1 2 4"` | TP 列表 |
| `VLLM_TEST_MAX_PROMPT_NUM_LIST` | `16,32` 或 `"16 32"` | BS 列表 |
| `VLLM_TEST_PROMPT_MAX_TOKENS_PAIRS` | `100:512,1024:2048` | `prompt_len:max_tokens` 列表 |

可选变量：

| 变量 | 默认值 | 说明 |
| ------ | -------- | ------ |
| `VLLM_TEST_SCENARIO` | `phase2` | 场景标识 |
| `VLLM_TEST_TEMPERATURE` | `0.6` | 唯一支持泛化的采样参数；有限非负浮点数，如 `0`、`0.6` |
| `VLLM_TEST_REQUIRED_ROUNDS` | `1` | 每个组合要求连续通过的轮数 |
| `VLLM_TEST_MAX_NUM_BATCHED_TOKENS` | 未设置 | 设置后所有组合使用固定值 |
| `VLLM_TEST_MAX_MODEL_LEN` | `22578` | vLLM 最大模型长度；覆盖 Phase 2.1 基线的 20480 token 输入、2048 token 输出和最多 50 token 动态输入变化；关闭 chunked prefill 时，单步 token budget 不得低于该值 |
| `VLLM_TEST_BATCHED_TOKENS_OFFSET` | `101` | 未固定时计算原始预算 `BS * (prompt_len + offset)` |
| `VLLM_TEST_AUTO_MAX_NUM_BATCHED_TOKENS_CAP` | `131072` | 自动预算上限；最终值不低于 `VLLM_TEST_MAX_MODEL_LEN` |
| `VLLM_TEST_SEED` | `12345` | 首轮 seed，后续轮次加 1 |
| `VLLM_TEST_ENABLE_PREFIX_CACHING` | `False` | prefix caching 开关 |
| `VLLM_TEST_LOG_DIR` | `<RUN_RECORD_DIR>/phase2` | 本次运行的 Phase 2 记录目录 |
| `VLLM_TEST_RUN_CONFIG` | `<RUN_RECORD_DIR>/run_config.json` | 已确认 TP 计划的运行配置；runner 将其传给环境预检脚本 |
| `VLLM_TEST_POST_RUN_WAIT_TIME` | `5` | 连续轮次间等待秒数 |
| `VLLM_TEST_TRANSIENT_RESOURCE_RETRY_COUNT` | `2` | `free memory < desired` 启动错误的最大重试次数 |
| `VLLM_TEST_TRANSIENT_RESOURCE_RETRY_DELAY` | `10` | 瞬时启动显存错误的重试等待秒数 |
| `VLLM_TEST_OP_CHECK_TIMEOUT` | `120s` | 自定义算子预检超时 |
| `VLLM_TEST_LOG_STATS_INTERVAL` | `1` | vLLM 调度统计输出间隔（秒）；用于驱逐判定 |
| `VLLM_TEST_PROMPT_LEN_VARIATION` | `50` | `max_prompt_len` 是精确基准 token 长度；每条 prompt 在该基准上独立随机增加 `0..N` 个 token，并严格对齐到各自的目标长度；0 表示不变化 |

Phase 2.1 长输出和长输入保留默认 `VLLM_TEST_PROMPT_LEN_VARIATION=50`，每条请求在基准输入长度上独立增加 `0..50` token。`max_model_len=22578` 按最坏情况 `20480 + 50 + 2048` 设置，不能降低为 22528，否则最长请求会触发 `BLOCKED_PROMPT_LENGTH_EXCEEDS_MODEL_LIMIT`。

使 TP 与可见 NPU 数、模型和 vLLM Ascend 支持的 tensor parallel 配置匹配。除非验收规格要求固定值，否则不设置 `VLLM_TEST_MAX_NUM_BATCHED_TOKENS`。该参数只是单步调度 token budget，不代表 KV cache 总容量，不能通过增大它规避驱逐。

如果目标是 temperature 泛化，应分别执行完整场景，例如一轮 `VLLM_TEST_TEMPERATURE=0`、一轮 `VLLM_TEST_TEMPERATURE=0.6`，分别出具结论；不要在同一 BS=1/BS=N 对比内改变温度。

## 2. 验收矩阵

执行前固定 `<RUN_RECORD_DIR>`，所有场景共用其 `phase2/` 子目录。内部入口由 skill 自行解析，不在面向用户的记录中展示。

Phase 2.1 长输出：

```bash
VLLM_BATCH_INVARIANT=1 \
VLLM_TEST_SCENARIO=phase2_1_long_output \
VLLM_TEST_MODEL=/model_path/Qwen3-30B-A3B \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_TEST_TP_LIST=8 \
VLLM_TEST_MAX_PROMPT_NUM_LIST=64 \
VLLM_TEST_PROMPT_MAX_TOKENS_PAIRS=2048:20480 \
VLLM_TEST_REQUIRED_ROUNDS=1 \
VLLM_TEST_LOG_DIR=<RUN_RECORD_DIR>/phase2 \
VLLM_TEST_RUN_CONFIG=<RUN_RECORD_DIR>/run_config.json \
bash scripts/run_phase2_acceptance.sh
```

Phase 2.1 长输入：

```bash
VLLM_BATCH_INVARIANT=1 \
VLLM_TEST_SCENARIO=phase2_1_long_prompt \
VLLM_TEST_MODEL=/model_path/Qwen3-30B-A3B \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_TEST_TP_LIST=8 \
VLLM_TEST_MAX_PROMPT_NUM_LIST=64 \
VLLM_TEST_PROMPT_MAX_TOKENS_PAIRS=20480:2048 \
VLLM_TEST_MAX_NUM_BATCHED_TOKENS=131072 \
VLLM_TEST_REQUIRED_ROUNDS=1 \
VLLM_TEST_LOG_DIR=<RUN_RECORD_DIR>/phase2 \
VLLM_TEST_RUN_CONFIG=<RUN_RECORD_DIR>/run_config.json \
bash scripts/run_phase2_acceptance.sh
```

Phase 2.2 多维稳定性矩阵：

```bash
VLLM_BATCH_INVARIANT=1 \
VLLM_TEST_SCENARIO=phase2_2_generalization \
VLLM_TEST_MODEL=/model_path/Qwen3-30B-A3B \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
VLLM_TEST_TP_LIST=1,2,4 \
VLLM_TEST_MAX_PROMPT_NUM_LIST=16,32 \
VLLM_TEST_PROMPT_MAX_TOKENS_PAIRS=100:512,1024:2048 \
VLLM_TEST_REQUIRED_ROUNDS=3 \
VLLM_TEST_LOG_DIR=<RUN_RECORD_DIR>/phase2 \
VLLM_TEST_RUN_CONFIG=<RUN_RECORD_DIR>/run_config.json \
bash scripts/run_phase2_acceptance.sh
```

注意：TP=1 时部分模型权重可能无法单卡加载。发生模型加载 OOM 时，停止 TP=1 下后续测试，标记 `SKIPPED_TP_RESOURCE_UNAVAILABLE` 并在结果中呈现原始报错；该项不阻塞 Phase 2 整体通过，但不得将 TP=1 记为通过，也不得用 TP=2/4 的结果外推 TP=1。

## 3. 输出与判定

每次调用写入 `<RUN_RECORD_DIR>/phase2/batch_invariant_results_<timestamp>/`：

| 文件 | 含义 |
| ------ | ------ |
| `summary.csv` | 当前场景每个组合的状态、`ParallelMode=TP`、temperature、step/token/logprob 失败计数和日志路径 |
| `summary.md` | 当前场景汇总 |
| `*_attempt*.log` | 单轮 pytest 日志 |
| `*_attempt*_scheduler.json` | KV 容量、所需 KV tokens、最大使用率、preemption 和调度状态转移分析 |
| `*_attempt*_failure_manifest.json` | token/logprob 不一致请求的结构化证据；包含逻辑 request ID、精确 prompt token IDs、首个不一致位置和运行配置 |

三个场景全部通过才算 Phase 2 通过。每个请求必须在 `ignore_eos=True` 下完成目标输出步数；BS=1/BS=N 的完整 token ID 序列必须相同，随后才比较相同 token 路径上的 chosen-token logprob。以 `summary.csv` 的 `Status`、`SchedulerStatus`、scheduler JSON 和 attempt 日志中的 `BATCH INVARIANCE FAILURES:` 为准，不只看 pytest footer。容量不足、显式 preemption 或满批运行后的驱逐回退场景为 `BLOCKED`；初始排队不阻塞；缺少调度统计标记 `BLOCKED_SCHEDULER_EVIDENCE`。先调整 BS、长度或 KV 容量并重跑，不对这类错误选择 dump。token 或 logprob 不一致时固定原组合并生成 `failure_manifest.json`；随后读取 `model-debugging.md` 完成失败分类，并只在静态 slice 与动态 request ID 两种 dump 方法中选择。

报告必须明确 `parallel_mode=TP`、TP 数值和 temperature。对于 PP、EP、DP 或其他并行请求，只能报告理论/可能支持及未验证状态，除非后续新增对应入口和完整证据链。

最终回复按 `final-report.md` 分别列出三个场景的 `summary.csv`、`summary.md`、scheduler JSON、failure manifest 和 attempt 日志，并从 `summary.csv` 汇总状态、KV 容量/需求、最大 KV 使用率、preemptions、step mismatch、token mismatch 与 logprob mismatch。任一正式场景未执行或命中驱逐时，不得把 Phase 2 汇总写为 `PASS`。
