# Phase 3 Math500 验收

命令示例以本 Skill 根目录（与 `SKILL.md` 同级）为工作目录。

## 1. 职责边界

Phase 3 通过以下名称调用公共 skill：

```text
/mindspeed-agent:vllm-ascend-aisbench-accuracy
```

不要指定或要求读取该公共 skill 的 `SKILL.md` 文件路径，也不要直接调用其内部 Python 脚本。完整读取本 skill 的 [Math500 调用资料](math500-invocation.md) 中“Math500 验收证据”prompt，替换占位符后发起 skill 调用。

公共 skill 负责：

- 目标环境网络预检
- ais-bench 安装和 Math500 数据集下载
- AISBench vLLM API 配置
- vLLM Ascend 服务启动、健康检查和停止
- Math500 连续多轮评测
- 每轮 accuracy、汇总 JSON 和 Markdown 报告

本 skill 不再维护 ais-bench 安装脚本、服务脚本或 AISBench 兼容 patch。`scripts/validate_math500_acceptance.py` 只做公共 skill 完成后的 batch invariance 最终验收。

A5 与 A2/A3 一样，在自动安装及环境准入、Phase 1 和 Phase 2 通过后执行本阶段。

## 2. 固定验收配置

| 参数 | 要求 |
| ------ | ------ |
| 模型、TP、设备 | 与目标模型正式验收配置一致；当前已支持的并行验收范围仅为 TP |
| `VLLM_BATCH_INVARIANT` | `1`，通过公共 skill 的 `--extra-env` 传入 |
| 数据集 | `math500_gen_0_shot_cot_chat_prompt` 全量 500 样本 |
| 轮数 | `10` |
| `temperature` | Phase 3 标准验收默认 `0`，十轮保持一致；不要沿用 Phase 2 默认值 `0.6` |
| `seed` | `1234` |
| `max_out_len` | `20480` |
| `max_model_len` | `22528` |
| `max_num_seqs` | 调用方按服务资源确定，10 轮保持一致 |
| `max_num_batched_tokens` | `22528` |
| `block_size` | `128` |
| chunked prefill | 关闭，即 `enable_chunked_prefill=False`；与 Phase 1 profiling 和 Phase 2 保持一致 |
| 执行模式 | `distributed_executor_backend=mp`、`enforce_eager`；不得使用 dummy load |
| 显存利用率 | 默认 `0.9`；若目标场景另有明确值，以目标场景为准并记录 |
| prefix caching | 关闭，即 `enable_prefix_caching=False` |
| async scheduling | 关闭，即 `async_scheduling=False` |
| accuracy 阈值 | 默认 `93`，验收器同时接受 `93` 或 `0.93` |

`max_out_len=20480` 与 2048-token 输入余量之和可以等于 `max_model_len=22528`；公共 skill 仅在该总需求严格大于 `max_model_len` 时拒绝。公共 skill 必须在 `raw_result.config.serve_args` 中显式回显 `--no-enable-chunked-prefill`、`--no-enable-prefix-caching`、`--no-async-scheduling` 和 `--enforce-eager`，以及表中的长度、并行、block size、后端、显存利用率和 generation config 参数；`load_format` 未显式提供时按默认 `auto` 验收。vLLM 0.23.1 的 `disable_log_stats` 默认值为 False，且 CLI 不支持 `--no-disable-log-stats`，因此该项通过不传 `--disable-log-stats` 表达并按缺省 False 验收。

PP、EP、DP 的 Phase 3 编排在理论上可能复用公共 skill，但尚未完成本 batch invariance skill 的端到端验证；其他并行方式和混合并行仅属于可能支持。即使公共 skill 能启动这些配置，也只能形成实验性结果，不能计入当前已支持范围。

采样参数只泛化 temperature。`seed=1234` 是复现控制参数，不是额外采样泛化维度；不要透传 `top_p`、`top_k`、`min_p` 或惩罚项等未验证采样参数。

Phase 3 不设置 `ignore_eos=True`。该参数用于 Phase 2 强制产生固定数量的 token/logprob step；Math500 是精度任务，强制越过 EOS 会改变答案语义。Phase 3 的确定性证据是 10 轮按样本 ID 对齐后的完整原始 `prediction` 字符串一致。

Math500 全量 500 条请求由 `max_num_seqs=<MAX_NUM_SEQS>` 控制服务端最大并发并分批调度。运行期间出现任意数量的 `Waiting` 只表示请求排队，不能据此判断驱逐、抢占或验收失败，也不要求固定的 Running/Waiting 数量。Phase 3 不应用 Phase 2 静态 BS=N 的状态转移规则；调度异常只依据日志中明确的 preemption/eviction、请求失败、服务错误或容量不足。最终一致性以每轮 500 条成功预测按 `(data_abbr, id)` 对齐后，对完整原始 `prediction` 逐字符比较。

## 3. 调用公共 skill

完整读取 [Math500 调用资料](math500-invocation.md) 的“Math500 验收证据”，替换全部占位符后调用公共 skill；本文件不复制 prompt。调用前把 marker 文件写入 `<RUN_RECORD_DIR>/phase3/`，并用公共 skill 返回的实际输出目录排除历史 experiment。不要在面向用户的报告中展示公共 skill 的内部目录。

Phase 3 标准验收保持 `temperature=0`。Phase 2 即使使用 `temperature=0.6`，也不能把该值直接沿用到 Phase 3。若调用方明确要求额外探索非默认 Phase 3 温度，必须与标准验收分开执行和归档，且不能替代 `temperature=0` 的标准结论。

## 4. 收集原始预测证据

公共 skill 的 stdout JSON 和 `raw_result` 包含每轮 accuracy，但当前不包含预测 JSONL 路径。因此公共 skill 返回后，还必须从其实际输出目录收集同一次运行恰好 10 个 experiment 的非 failed 预测文件，并归档到 `<RUN_RECORD_DIR>/phase3/predictions/`。

- 当前环境就是 head 服务器/容器：直接读取本次开始时间之后生成的文件。
- 远程目标：使用公共 skill 的同一连接信息把这 10 个原始 JSONL 取回证据目录，再在 agent 所在环境运行验收器。
- 同一 experiment 找不到文件、找到多个候选、存在 `_failed.jsonl` 而无成功文件，或本次运行不是恰好 10 份，都不得挑选性忽略；判定 `FAIL/BLOCKED`。

不要用整个 JSONL 的文件 hash 直接判断文本是否一致。UUID 和行顺序可能变化，必须按 `(data_abbr, id)` 对齐后比较完整 `prediction` 字段。

## 5. 最终验收

同机且可以直接访问 outputs 目录时：

```bash
python3 scripts/validate_math500_acceptance.py \
  --accuracy-result <common-skill-raw_result.json> \
  --outputs-root <COMMON_SKILL_OUTPUTS_ROOT> \
  --started-after <RUN_RECORD_DIR>/phase3/start.marker \
  --expected-rounds 10 \
  --expected-count 500 \
  --expected-model <MODEL_PATH> \
  --expected-visible-devices <ASCEND_RT_VISIBLE_DEVICES> \
  --expected-cann-dir <CANN_DIR> \
  --expected-tp <TP> \
  --expected-dp 1 \
  --expected-pp 1 \
  --expected-enable-expert-parallel false \
  --expected-dataset math500_gen_0_shot_cot_chat_prompt \
  --expected-dataset-kind math \
  --expected-temperature 0 \
  --expected-seed 1234 \
  --expected-vllm-batch-invariant 1 \
  --expected-prefix-caching false \
  --expected-max-model-len 22528 \
  --expected-max-num-seqs <MAX_NUM_SEQS> \
  --expected-max-num-batched-tokens 22528 \
  --expected-block-size 128 \
  --expected-enable-chunked-prefill false \
  --expected-distributed-executor-backend mp \
  --expected-generation-config vllm \
  --expected-gpu-memory-utilization 0.9 \
  --expected-enforce-eager true \
  --expected-disable-log-stats false \
  --expected-load-format auto \
  --expected-async-scheduling false \
  --accuracy-threshold 93 \
  --report <RUN_RECORD_DIR>/phase3/phase3_acceptance.json
```

远程文件已取回时，使用同一命令和全部 `--expected-*` 参数，仅将 `--outputs-root` 与 `--started-after` 替换为十个显式输入：

```bash
python3 scripts/validate_math500_acceptance.py \
  --accuracy-result <common-skill-raw_result.json> \
  --prediction-file <round-1.jsonl> \
  --prediction-file <round-2.jsonl> \
  ... \
  --prediction-file <round-10.jsonl> \
  <与同机命令相同的全部 --expected-* 和 --accuracy-threshold 参数> \
  --report <RUN_RECORD_DIR>/phase3/phase3_acceptance.json
```

验收器按 `(data_abbr, id)` 对齐样本，忽略 UUID 和行顺序，并逐字符比较完整原始 `prediction`。以下条件必须同时满足：

1. 公共 skill 的 `raw_result.config` 与命令中显式提供的完整目标场景一致：模型、可见设备、CANN 路径、TP/DP/PP/EP、轮数、数据集及类型、temperature、seed、`VLLM_BATCH_INVARIANT=1`、prefix caching、最大长度、最大并发、批处理 token 上限、block size、chunked prefill、执行后端、显存利用率、eager 模式、load format、异步调度和日志统计开关逐项通过。
2. 公共 skill 的 10 轮状态均为 `ok` 且都有 accuracy。
3. 每轮 accuracy 不低于阈值。
4. 10 轮 accuracy 完全相同，不以平均值或近似容差替代。
5. 每轮恰好 500 个唯一且 `success=true` 的样本。
6. 10 轮样本集合完全相同，且每个样本的原始 `prediction` 字符串完全相同。
所有 `--expected-*` 参数必须来自调用前固定的正式目标配置，不得读取 `raw_result` 后将实际值回填为 expected。任一配置字段缺失、不合法或不一致，验收器都会在 `scenario_config` 中报告逐项结果，写出 `FAIL/ERROR` 并返回非零退出码。公共 skill 当前不会把 `max_out_len` 写入 `raw_result.config`，因此该值仍由调用命令、公共 Markdown 报告和 AISBench 配置共同取证，不在验收器内伪造校验。报告应与公共 skill 的 `raw_result`、Markdown 报告、10 份预测 JSONL 和 `phase3_acceptance.json` 共同归档。

最终回复按 `final-report.md` 列出公共 Markdown 报告、`raw_result`、每轮 accuracy、10 份预测 JSONL 和 `phase3_acceptance.json`。公共 skill 返回 `status=ok` 不能替代原始文本一致性报告。
