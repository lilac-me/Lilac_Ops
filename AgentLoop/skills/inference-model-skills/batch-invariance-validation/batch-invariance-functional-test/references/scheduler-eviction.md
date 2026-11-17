# 请求驱逐检查

命令示例以本 Skill 根目录（与 `SKILL.md` 同级）为工作目录。

## 1. 支持边界

当前 batch invariance 验收不支持请求驱逐/抢占（preemption）场景。驱逐会使请求释放 KV blocks、回到 waiting 并从头重算，改变被驱逐请求P阶段的长度，由原来的P阶段长度再额外加上。

这里的“请求驱逐”特指正在运行的请求被 preempt 并重新排队，不是 prefix cache 对无引用缓存块的普通淘汰。

命中驱逐或驱逐风险时，状态写 `BLOCKED_EVICTION_UNSUPPORTED`，不是模型数值 `FAIL`。调整参数消除风险后，重新执行原场景。

所有正式场景保持 `disable_log_stats=False`。该配置使日志周期性输出：

```text
Running: <N> reqs, Waiting: <M> reqs, GPU KV cache usage: <P>%
```

## 2. 容量预检

启动日志包含：

```text
Available KV cache memory: <GiB>
GPU KV cache size: <capacity> tokens
Maximum concurrency for <max_model_len> tokens per request: <value>x
```

使用 `GPU KV cache size` 作为每个 DP engine 的 KV token 容量。Phase 2 pytest 会用真实 tokenizer 获取每条 prompt 的 token 数，并计算：

```text
required_kv_tokens = sum(
  ceil((prompt_token_count_i + max_tokens) / block_size) * block_size
)
```

如果 `required_kv_tokens > capacity_kv_tokens`，在 BS=N 生成前直接输出 `BATCH_INVARIANCE_EVICTION_UNSUPPORTED` 并中止。不要等待运行时实际发生驱逐。

`max_num_batched_tokens` 是单次 scheduler step 的 token budget，不是 KV cache 总容量。增大它不能解决容量不足，反而可能提高瞬时压力。消除风险时按以下顺序选择：

1. 降低 BS。
2. 降低 prompt 或输出长度，但这属于新验收配置，不能冒充原场景通过。
3. 在资源允许时提高 `gpu_memory_utilization`，或增加/调整设备和 TP 配置以获得足够 KV 容量；修改后重新读取启动日志确认实际容量。
4. 如果正式规格固定且无法满足容量，场景保持 `BLOCKED_EVICTION_UNSUPPORTED`。

## 3. 运行时判定

执行：

```bash
python3 scripts/analyze_scheduler_log.py \
  --log <attempt-or-server.log> \
  --expected-batch-size <BS> \
  --static-batch \
  --max-sequence-length <prompt_len+max_tokens> \
  --block-size 128 \
  --report <evidence-dir>/scheduler_analysis.json
```

Phase 2 是一次性静态 BS=N 请求集合，使用 `--static-batch`。命中以下任一项即判不支持：

- 日志或测试 marker 已确认容量不足。
- 已观察到 `Running=BS, Waiting=0`，随后出现 `Running<BS, Waiting>0`。在没有新请求进入的静态批次中，这表示原运行请求被抢占并移回 waiting。

初始出现 `Running<BS, Waiting>0` 只证明请求受 token budget 约束而排队，不要求静态批次必须达到 `Running=BS, Waiting=0`，也不据此判定阻塞或驱逐。只有已经观察到 `Running=BS, Waiting=0` 后，又出现 `Running<BS, Waiting>0`，才能根据该状态转移判定请求驱逐。Phase 2 默认将 `VLLM_LOG_STATS_INTERVAL` 设为 1 秒；没有任何周期调度统计时标记 `BLOCKED_SCHEDULER_EVIDENCE`。Phase 3 存在持续请求流，不使用 Running/Waiting 状态转移推断驱逐，只检查容量和请求失败；server 日志没有周期调度统计时同样标记证据缺失。

## 4. 示例判定

以下状态序列属于请求驱逐：

```text
Running: 16 reqs, Waiting: 0 reqs, GPU KV cache usage: 99.5%
Running: 15 reqs, Waiting: 1 reqs, GPU KV cache usage: 94.1%
```

KV 使用率接近满载后，Running 从 16 降为 15、Waiting 从 0 升为 1，同时使用率明显下降，符合一个运行请求释放 KV blocks 并回到 waiting 的特征。

若启动容量为 `247040` tokens，目标为 BS=16、每条最多 `2048+20480=22528` tokens、`block_size=128`，则：

```text
required = 16 * 22528 = 360448 tokens
shortfall = 360448 - 247040 = 113408 tokens
```

该配置在运行前即应判 `BLOCKED_EVICTION_UNSUPPORTED`。
