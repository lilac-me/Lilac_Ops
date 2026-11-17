# msprobe Dump 验证 Demo Case

## 目录

1. 目标与结论边界
2. 公共前置条件
3. 静态批次 Demo
4. 动态批次 Demo
5. Dump 完整性与统计对比
6. 结果记录模板

## 1. 目标与结论边界

本参考用于验证 vLLM 昇腾场景中的 msprobe dump、静态 slice、动态 request ID 和统计对比链路。所有 Demo 默认使用：

```json
{
  "task": "statistics",
  "level": "mix",
  "async_dump": false,
  "extra_info": true,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

本轮不配置 `tensor_list`，不使用 `task: tensor`。统计一致只能说明当前统计级 dump 未发现差异，不能代替 Phase 2 的 token ID 和 chosen-token logprob 按位验收。

Demo 使用以下状态：

| 状态 | 含义 |
| ------ | ------ |
| `DUMP_POSITIVE_CONSISTENT` | BS=1 与目标 BS=N 数据可对齐，全部目标统计一致 |
| `DUMP_NEGATIVE_OPERATOR_BOUNDARY` | 找到首个输入统计一致、输出统计不一致的候选算子边界 |
| `BLOCKED_DUMP_ALIGNMENT` | 请求、step、rank、节点或 token 路径无法可靠对齐 |
| `BLOCKED_DUMP_ERROR` | dump 缺失、JSON 不完整或 `dump_error_info.log` 非空 |
| `NOT_RUN` | 未在真实目标环境运行 |

`DUMP_NEGATIVE_OPERATOR_BOUNDARY` 只表示统计证据中的候选边界，不直接等同于 kernel 根因。后续若需要根因确认，再单独规划 tensor 级验证。

## 2. 公共前置条件

执行前完整读取 `model-debugging.md` 要求的 msprobe 和 vLLM-Ascend 官方文档，并完成文档快照、版本与 SHA-256 记录。另需满足：

1. `VLLM_BATCH_INVARIANT=1`。
2. Eager 模式、prefix caching 关闭、异步调度关闭、`block_size=128`。
3. BS=1 与 BS=N 使用相同模型、TP、dtype、设备、prompt token IDs、seed、temperature 和 generation config。
4. 每次 BS=1、BS=N 使用新进程和独立 `dump_path`，使各自的第一次模型前向都从 `step0` 开始。
5. 保留实际 dump config、启动命令、服务日志、请求 payload、输出 token IDs、scheduler 日志和每个 rank 的 dump 路径。
6. 动态 request ID 场景确认 vLLM-Ascend 将 `scheduler_output.num_scheduled_tokens` 传入 `PrecisionDebugger.start(..., scheduled_tokens=...)`。

以下路径均为占位符，执行时替换为目标绝对路径：

```bash
export MODEL_PATH=/path/to/model
export DUMP_ROOT=/path/to/evidence/msprobe_demo
export TP_SIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0
```

### 静态 Demo 运行脚本

将以下内容保存为本次证据目录中的 `run_static_dump_demo.py`。这是运行工件，不合入 skill：

```python
import json
import os

from vllm import LLM, SamplingParams

model_path = os.environ["MODEL_PATH"]
dump_config = os.environ["DUMP_CONFIG"]
mode = os.environ["DUMP_MODE"]  # bs1 或 bsn
target_index = int(os.environ.get("TARGET_INDEX", "2"))
lengths = [4, 7, 3, 6]

llm = LLM(
    model=model_path,
    tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
    dtype="bfloat16",
    enforce_eager=True,
    distributed_executor_backend="mp",
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    async_scheduling=False,
    block_size=128,
    max_num_seqs=4,
    max_num_batched_tokens=4096,
    disable_log_stats=False,
    additional_config={"dump_config_path": dump_config},
)

tokenizer = llm.get_tokenizer()
texts = ["alpha beta", "gamma delta", "epsilon zeta", "theta lambda"]
prompt_ids = []
for text, length in zip(texts, lengths):
    source = tokenizer.encode(text, add_special_tokens=False)
    if not source:
        raise RuntimeError(f"tokenizer returned no tokens for {text!r}")
    ids = (source * ((length + len(source) - 1) // len(source)))[:length]
    if len(ids) != length:
        raise RuntimeError(f"failed to construct prompt length {length}")
    prompt_ids.append(ids)

prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
selected = [prompts[target_index]] if mode == "bs1" else prompts
params = SamplingParams(
    temperature=0,
    seed=1234,
    max_tokens=4,
    ignore_eos=True,
)
outputs = llm.generate(selected, params, use_tqdm=False)
evidence = {
    "mode": mode,
    "target_index": target_index,
    "prompt_lengths": [len(ids) for ids in prompt_ids],
    "prompt_token_ids": prompt_ids,
    "output_token_ids": [list(item.outputs[0].token_ids) for item in outputs],
}
print("DUMP_DEMO_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
```

静态 BS=1 与 BS=N 必须分别执行一次。例如：

```bash
DUMP_MODE=bs1 TARGET_INDEX=2 \
DUMP_CONFIG="$DUMP_ROOT/static_prefill_bs1.json" \
python3 "$DUMP_ROOT/run_static_dump_demo.py" \
  > "$DUMP_ROOT/static_prefill_bs1.log" 2>&1

DUMP_MODE=bsn TARGET_INDEX=2 \
DUMP_CONFIG="$DUMP_ROOT/static_prefill_bsn.json" \
python3 "$DUMP_ROOT/run_static_dump_demo.py" \
  > "$DUMP_ROOT/static_prefill_bsn.log" 2>&1
```

## 3. 静态批次 Demo

### 3.1 静态 Prefill

BS=4 的 prompt 长度固定为 `[4, 7, 3, 6]`，P 阶段 packed-token 总长度为：

```text
T_total = 4 + 7 + 3 + 6 = 20
```

完整边界 case：

| Case | 目标请求 | packed-token 区间 | slice |
| ------ | ---------- | ------------------- | ------- |
| `STATIC-PREFILL-HEAD` | 0 | `[0,4)` | `size=20, begin=0, end=4` |
| `STATIC-PREFILL-MIDDLE` | 2 | `[11,14)` | `size=20, begin=11, end=14` |
| `STATIC-PREFILL-TAIL` | 3 | `[14,20)` | `size=20, begin=14, end=20` |

中间请求的 BS=N 配置：

```json
{
  "task": "statistics",
  "dump_path": "/path/to/evidence/msprobe_demo/static_prefill_bsn",
  "rank": [],
  "step": [0],
  "level": "mix",
  "async_dump": false,
  "extra_info": true,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics",
    "slice": [{"dim": 0, "size": 20, "begin": 11, "end": 14}]
  }
}
```

对应 BS=1 配置不设置 slice：

```json
{
  "task": "statistics",
  "dump_path": "/path/to/evidence/msprobe_demo/static_prefill_bs1",
  "rank": [],
  "step": [0],
  "level": "mix",
  "async_dump": false,
  "extra_info": true,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

验收要点：

1. BS=N 的 step0 真实 packed-token 第 0 维必须为 20。
2. 中间请求 slice 后可比较张量的第 0 维必须为 3，与 BS=1 目标 prompt 长度一致。
3. head、middle、tail 分别独立采集，不能用一次 dump 外推三个边界都正确。

### 3.2 静态 Decode

运行脚本固定生成 4 个 token。`step0` 是 prefill 前向；后续模型前向为 decode。执行前从日志确认当前版本的实际 step 含义和目录序列，不允许仅按文件名猜测。

静态 decode 的第 0 维稳定为 BS=4，目标请求使用 row slice：

| Case | msprobe step | 目标 row | slice |
| ------ | -------------- | ---------- | ------- |
| `STATIC-DECODE-FIRST` | 1 | 0 | `size=4, begin=0, end=1` |
| `STATIC-DECODE-MIDDLE` | 2 | 2 | `size=4, begin=2, end=3` |
| `STATIC-DECODE-LATER` | 3 | 3 | `size=4, begin=3, end=4` |

目标 row 2、采集三个 decode step 的 BS=N 配置：

```json
{
  "task": "statistics",
  "dump_path": "/path/to/evidence/msprobe_demo/static_decode_bsn",
  "rank": [],
  "step": [1, 2, 3],
  "level": "mix",
  "async_dump": false,
  "extra_info": true,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics",
    "slice": [{"dim": 0, "size": 4, "begin": 2, "end": 3}]
  }
}
```

BS=1 使用相同 `step`，不配置 slice。比较 decode step `k` 前，必须确认 BS=1 和 BS=N 中该请求在此前生成的 token IDs 完全一致。若输出 token 首次在某一步分叉，只比较产生首个不同 token 的前向及更早步骤；后续 decode 输入已经不同，不能用于判定 batch invariance 算子边界。

## 4. 动态批次 Demo

动态场景禁止固定 row slice。统一使用 request ID：

```json
{
  "task": "statistics",
  "dump_path": "/path/to/evidence/msprobe_demo/dynamic_target",
  "rank": [],
  "step": [],
  "level": "mix",
  "async_dump": false,
  "extra_info": true,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics",
    "request_id": "chatcmpl-dyn-target"
  }
}
```

启动服务前设置：

```bash
export VLLM_BATCH_INVARIANT=1
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1

vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype bfloat16 \
  --enforce-eager \
  --block-size 128 \
  --max-num-seqs 4 \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --additional-config '{"dump_config_path":"/path/to/dynamic_config.json"}' \
  > "$DUMP_ROOT/dynamic_server.log" 2>&1
```

请求使用 `/v1/chat/completions`，请求头 `X-Request-ID: dyn-target` 对应配置中的 `chatcmpl-dyn-target`。必须从当前版本日志确认实际前缀；前缀不同则同步修改配置并重启服务。

每个 case 重启服务并在公共启动命令后追加以下参数：

| Case | 服务参数 |
| ------ | ---------- |
| `DYNAMIC-PREFILL-WAVE` | `--no-enable-chunked-prefill --max-num-batched-tokens 96` |
| `DYNAMIC-PREFILL-CHUNKED` | `--enable-chunked-prefill --max-num-batched-tokens 64` |
| `DYNAMIC-DECODE-STABLE` | `--no-enable-chunked-prefill --max-num-batched-tokens 4096` |
| `DYNAMIC-DECODE-JOIN-LEAVE` | `--no-enable-chunked-prefill --max-num-batched-tokens 4096` |
| `DYNAMIC-MIXED-PD` | `--enable-chunked-prefill --max-num-batched-tokens 64` |

将以下请求驱动保存为证据目录中的 `run_dynamic_dump_demo.py`。`REPEAT` 是构造不同 prompt 长度的重复次数；执行后必须以响应的 `usage.prompt_tokens` 和目标 tokenizer 复核实际 token 数，不能把重复次数当作 token 数：

```python
import concurrent.futures
import json
import os
import time

import requests

base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
model = os.environ["SERVED_MODEL_NAME"]
case = os.environ["DYNAMIC_CASE"]
join_delay = float(os.environ.get("JOIN_DELAY_S", "0.5"))
# 集群环境可能设置 HTTP(S)_PROXY；本地 vLLM 请求必须绕过代理，否则会得到无 body 的 502。
http = requests.Session()
http.trust_env = False


def send(request_id, repeat, max_tokens, delay=0.0):
    if delay:
        time.sleep(delay)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": (request_id + " token ") * repeat}],
        "temperature": 0,
        "seed": 1234,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }
    response = http.post(
        f"{base_url}/v1/chat/completions",
        headers={"Content-Type": "application/json", "X-Request-ID": request_id},
        json=payload,
        timeout=1800,
    )
    result = {
        "request_id": request_id,
        "status_code": response.status_code,
        "payload": payload,
        "response": response.json() if response.content else None,
    }
    print("DYNAMIC_DUMP_RESPONSE=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
    response.raise_for_status()
    return result


cases = {
    "wave": [
        ("dyn-bg-0", 8, 4, 0.0),
        ("dyn-bg-1", 16, 4, 0.0),
        ("dyn-target", 24, 4, 0.0),
        ("dyn-bg-2", 32, 4, 0.0),
    ],
    "chunked": [
        ("dyn-bg-0", 8, 4, 0.0),
        ("dyn-target", 256, 4, 0.0),
        ("dyn-bg-1", 8, 4, 0.0),
    ],
    "decode-stable": [
        ("dyn-bg-0", 16, 4, 0.0),
        ("dyn-bg-1", 16, 4, 0.0),
        ("dyn-target", 16, 4, 0.0),
        ("dyn-bg-2", 16, 4, 0.0),
    ],
    "decode-join-leave": [
        ("dyn-target", 16, 8, 0.0),
        ("dyn-short-0", 8, 1, 0.0),
        ("dyn-short-1", 8, 1, 0.0),
        ("dyn-late", 64, 4, join_delay),
    ],
    "mixed-pd": [
        ("dyn-target", 16, 8, 0.0),
        ("dyn-bg-0", 16, 8, 0.0),
        ("dyn-late-prefill", 256, 4, join_delay),
    ],
    "bs1": [("dyn-target", int(os.environ["TARGET_REPEAT"]), int(os.environ["TARGET_MAX_TOKENS"]), 0.0)],
}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(send, *args) for args in cases[case]]
    for future in futures:
        future.result()
```

执行示例：

```bash
DYNAMIC_CASE=decode-stable \
SERVED_MODEL_NAME=<served-model-name> \
python3 "$DUMP_ROOT/run_dynamic_dump_demo.py" \
  > "$DUMP_ROOT/dynamic_decode_stable_requests.log" 2>&1
```

每个动态 BS=N case 都要重启独立服务。随后使用同一目标 prompt 参数执行 `DYNAMIC_CASE=bs1`，并写入另一个 dump 目录。例如 `decode-stable` 的目标参数是 `TARGET_REPEAT=16 TARGET_MAX_TOKENS=4`；`mixed-pd` 是 `16/8`。

### 4.1 `DYNAMIC-PREFILL-WAVE`

关闭 chunked prefill，设置 `--max-num-batched-tokens 96`。同时提交 4 个请求，prompt 分别使用目标 tokenizer 预检为约 24、48、64、80 token，目标请求选择第三条。总 prompt token 数超过单 step budget，使请求分波进入 P 阶段。

兼容性检查：部分 vLLM 版本要求 `max_model_len <= max_num_batched_tokens`。若目标 prompt 必须大于 96，而该约束不能通过版本或配置解除，则本 case 记为 `BLOCKED_DUMP_ERROR`；不要提高 budget 后把普通批处理冒充 wave case。

验收：目标请求可以不在全局 step0；request ID dump 必须只在目标实际被调度的 P 阶段 step 产生目标统计。初始 `Running<4, Waiting>0` 只表示正常分波，不等价于驱逐。

### 4.2 `DYNAMIC-PREFILL-CHUNKED`

重启服务，增加 `--enable-chunked-prefill --max-num-batched-tokens 64`。目标请求使用 tokenizer 确认为至少 256 token 的 prompt，背景请求约 32 token，使目标 P 阶段跨多个全局 step。

验收：对目标命中的每个 P 阶段 step，request ID slice 后的 token 维长度必须等于该 step 中 `num_scheduled_tokens[目标request_id]`；所有 chunk 的累计长度必须等于目标实际 prompt token 数。

### 4.3 `DYNAMIC-DECODE-STABLE`

关闭 chunked prefill，4 个请求同时提交，均设置 `temperature=0`、`max_tokens=4`、`ignore_eos=true`。使用长度相近的 prompt，使 4 个请求在 decode 期间保持活跃。

验收：目标请求每个 decode step 的 `scheduled_tokens` 应为 1；按目标请求自己的第 1、2、3 次 decode 顺序与 BS=1 对齐，不要求两次运行的全局 step 编号相同。

注意：目标请求或背景请求在同一轮完成时，调度器可能先压缩 active batch row，再结束该请求。若日志仍打印了目标 request ID，但 slice 后的 position 已跳到背景请求的序列长度，说明 request-id slice 在终止边界错位；该 step 必须记为 `BLOCKED_DUMP_ALIGNMENT_INFRA`，不能把错位后的统计送入算子结论。应至少保留目标 request 仍 active 的前一 request-local decode step 作为有效证据。

### 4.4 `DYNAMIC-DECODE-JOIN-LEAVE`

先提交目标请求和两个 `max_tokens=1` 的短背景请求，目标请求设置 `max_tokens=8`。短请求结束后，再提交一个长背景请求，使目标请求所在活跃 batch 的大小和 row 发生变化。

验收：目标 request ID 的每个 decode dump 仍只包含 1 个目标 token；如果固定 row 会切到其他请求，而 request ID 仍能正确跟踪目标。若本次时序没有观察到背景请求离开和新请求加入，标记 `BLOCKED_DUMP_ALIGNMENT` 并调整请求延迟后重跑，不能外推通过。

### 4.5 `DYNAMIC-MIXED-PD`

先让目标请求进入 decode，再提交一个至少 256 token 的新请求。使用 `--enable-chunked-prefill --max-num-batched-tokens 64`，允许调度器在同一步处理 decode 和 prefill chunk。

验收：从 scheduler 证据确认某个全局 step 同时包含：

```text
目标请求: scheduled_tokens = 1（decode）
新请求: scheduled_tokens > 1（prefill 或 prefill chunk）
```

目标 request ID dump 在该混合 step 中只能保留目标的 1 个 decode token，不能包含新请求的 prefill tokens。未观察到混合 P/D step 时，本 case 为 `BLOCKED_DUMP_ALIGNMENT`，不得用普通纯 decode step 替代。

## 5. Dump 完整性与统计对比

### 5.1 完整性检查

对 BS=1 和 BS=N 分别执行：

```bash
find <dump-root> -name dump.json -type f -size +0c -print
find <dump-root> -name dump_error_info.log -type f -size +0c -print
find <dump-root> -name construct.json -type f -size +0c -print
```

逐个验证 JSON：

```bash
python3 -m json.tool <dump-root>/step0/rank0/dump.json >/dev/null
```

任一目标 rank 缺少 `dump.json`、JSON 不可解析或存在非空错误日志，输出 `BLOCKED_DUMP_ERROR`。

### 5.2 统计对比规则

按以下维度对齐：

```text
rank
  -> 阶段（prefill/decode）
  -> request-local step
  -> 节点名和调用序号
  -> input_args/input_kwargs/output
  -> tensor 位置
```

比较字段为 `shape`、`dtype`、`Max`、`Min`、`Mean`、`Norm`：

1. 节点集合或调用顺序不能对齐：`BLOCKED_DUMP_ALIGNMENT`。
2. 输入统计不同：差异来自上游，继续向前查找。
3. 输入统计一致且输出统计不同：记录当前节点为 `DUMP_NEGATIVE_OPERATOR_BOUNDARY`。
4. 所有可对齐节点统计一致：`DUMP_POSITIVE_CONSISTENT`。

对于动态场景，全局 step 编号不同不构成失败；必须按 request-local prefill chunk 或 decode 次序对齐。对于 decode，前序 token IDs 不一致后停止比较后续 step。

对已经完成 rank、阶段和 request-local step 对齐的两个 `dump.json`，可直接运行以下只读比较。脚本不保存文件，也不比较完整 tensor：

实际 vLLM-Ascend dump 可能包含调度器、packed-token 索引、KV-cache 写入、attention metadata 和 logits processor 等运行时节点。这些节点的输入形状或调用次数会随 BS、request row 和 cache block 排布改变，即使模型前向本身一致，也可能使“全量节点比较”返回 `BLOCKED_DUMP_ALIGNMENT`。因此应同时输出两层结论：

1. 先对齐并比较 `Module.model.*`（或模型实现中等价的 model-forward 节点）；该集合统计全部一致时，记为 `DUMP_POSITIVE_CONSISTENT_MODEL_FORWARD`。
2. 再单独记录 `NPU.npu_*kv*`、`NPU.npu_*attention*`、`Torch.index_select*`、`logits_processor`、scheduler/allocator 元数据等基础设施节点差异；这些差异只能记为 `BLOCKED_DUMP_ALIGNMENT_INFRA`，不能直接判定算子反例。

只有在模型前向节点已经对齐、且出现“输入统计一致而输出统计不同”时，才能升级为 `DUMP_NEGATIVE_OPERATOR_BOUNDARY`。若连模型前向节点都无法对齐，仍按 `BLOCKED_DUMP_ALIGNMENT` 处理。

```bash
python3 - <bs1-dump.json> <bsn-aligned-dump.json> <<'PY'
import json
import sys

STAT_FIELDS = ("shape", "dtype", "Max", "Min", "Mean", "Norm")


def tensor_stats(value, path=""):
    records = {}
    if isinstance(value, dict):
        if "shape" in value and "dtype" in value:
            records[path] = tuple((field, value.get(field)) for field in STAT_FIELDS)
        else:
            for key, item in value.items():
                records.update(tensor_stats(item, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            records.update(tensor_stats(item, f"{path}[{index}]"))
    return records


def load(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    data = payload.get("data")
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: missing non-empty data object")
    return data


try:
    bs1 = load(sys.argv[1])
    bsn = load(sys.argv[2])
except Exception as exc:
    print(f"BLOCKED_DUMP_ERROR: {exc}")
    raise SystemExit(2)

if list(bs1) != list(bsn):
    print("BLOCKED_DUMP_ALIGNMENT: node names or call order differ")
    raise SystemExit(3)

upstream_differences = []
for node_name in bs1:
    left = bs1[node_name]
    right = bsn[node_name]
    left_inputs = tensor_stats({
        "input_args": left.get("input_args"),
        "input_kwargs": left.get("input_kwargs"),
    })
    right_inputs = tensor_stats({
        "input_args": right.get("input_args"),
        "input_kwargs": right.get("input_kwargs"),
    })
    left_outputs = tensor_stats(left.get("output"), "output")
    right_outputs = tensor_stats(right.get("output"), "output")

    inputs_equal = left_inputs == right_inputs
    outputs_equal = left_outputs == right_outputs
    if inputs_equal and not outputs_equal:
        print("DUMP_NEGATIVE_OPERATOR_BOUNDARY")
        print(f"node={node_name}")
        print(f"bs1_output={left_outputs}")
        print(f"bsn_output={right_outputs}")
        raise SystemExit(1)
    if not inputs_equal or not outputs_equal:
        upstream_differences.append(node_name)

if upstream_differences:
    print("BLOCKED_DUMP_ALIGNMENT: differences exist but no aligned input/output boundary was found")
    print("nodes=" + json.dumps(upstream_differences))
    raise SystemExit(3)

print("DUMP_POSITIVE_CONSISTENT")
PY
```

多 rank 和多个 request-local step 必须逐对调用，不能只比较 rank0 或只比较一个 step 后给出全局结论。

## 6. 结果记录模板

每个 case 保存：

```text
Case ID:
状态: DUMP_POSITIVE_CONSISTENT / DUMP_POSITIVE_CONSISTENT_MODEL_FORWARD /
      DUMP_NEGATIVE_OPERATOR_BOUNDARY / BLOCKED_DUMP_ALIGNMENT /
      BLOCKED_DUMP_ALIGNMENT_INFRA / BLOCKED_DUMP_ERROR / NOT_RUN
模型与版本:
芯片、TP、rank:
服务与调度参数:
BS=1/BS=N 请求 payload:
目标 request ID / 静态 row:
prompt token IDs 与实际长度:
全局 step 与 request-local step 映射:
num_scheduled_tokens:
BS=1 dump 路径:
BS=N dump 路径:
首个候选差异节点:
输入统计:
输出统计:
日志与错误:
```

至少完成 `STATIC-PREFILL-MIDDLE`、`STATIC-DECODE-MIDDLE`、`DYNAMIC-PREFILL-CHUNKED` 和 `DYNAMIC-MIXED-PD`，才能说明静态/动态 P/D dump 链路均已被实际覆盖。其余 head、tail、稳定 decode 和 join/leave case 用于边界及稳定性补充。
