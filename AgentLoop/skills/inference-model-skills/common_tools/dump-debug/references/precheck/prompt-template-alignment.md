# 输入口径核对与推理 Template 对齐

> 适用：E3 / C 分支（数据集掉点），**aisbench 全量评测跑完之后、分析 bad case 之前**。
> 目的：先确认 bad / good 两侧喂给模型的输入本身是否一致。输入不一致时，精度数字不可比，
> 后续 bad case 分析、Dump、算子怀疑全部作废。

## 1. 为什么“固定偏移”指向模板

同一句输入文本，在 bad / good 两侧的输入 token 数不同，且**每条样本的差值都是同一个常数 d**
（例如 50 条样本全部 `+2`），说明差异不在逐样本内容，而在“每条输入都固定多/少了一段 token”。
能产生固定段的只有输入组装环节：

| 固定段的来源 | 典型形态 | 差值表现 |
|---|---|---|
| chat template（`tokenizer_config.json` / `chat_template.jinja` / `--chat-template`） | 多出 `<\|im_start\|>system …<\|im_end\|>`、`<\|assistant\|>`、`[INST]` 等角色/分隔标记 | 每条 `+k`（k = 模板固定 token 数） |
| system prompt / 默认指令 | 侧别默认 system 文本不同 | 每条 `+k`，k 通常较大 |
| BOS / EOS / `add_bos_token` / `add_generation_prompt` | 首尾多一个或两个 special token | 每条 `+1` / `+2` |
| tokenizer 版本（新增 special token） | `added_tokens` 不同 | 每条固定 `+k` |
| 截断策略（`--max-model-len`） | 超长样本被截到不同长度 | **不固定**（只有长样本差） |
| 数据集 / few-shot 构造 | 0-shot 与 5-shot、cot 前缀不同 | **不固定**，或整条内容不同 |

判据：**偏移固定 → 模板 / 配置未对齐；偏移不固定 → 数据集构造或 tokenizer 的逐样本差异。**

## 2. 执行

record 模式（已有两侧输入长度，服务已停时用）：

```bash
python3 scripts/compare_input_tokens.py \
  --bad  bad_input_len.jsonl --good good_input_len.jsonl \
  --min-pairs 3 --json-out input_len_summary.json > input_len_diff.jsonl
```

每行形如 `{"prompt": "...", "input_tokens": 123}`；长度可来自服务端 `usage.prompt_tokens`、
msProbe 日志或 aisbench 明细。字段名可用 `--length-field` 指定（默认自动识别
`input_tokens` / `input_token_len` / `prompt_tokens` / `usage.prompt_tokens` 等）。

replay 模式（两侧服务仍在，最可靠；同一条 prompt 各打一次，`max_tokens=1`）：

```bash
python3 scripts/compare_input_tokens.py \
  --bad-url  http://<bad-host>:8106 \
  --good-url http://<good-host>:8106 \
  --prompts  prompts.jsonl \
  --endpoint /v1/chat/completions --model <served-model-name> \
  --limit 50 --json-out input_len_summary.json > input_len_diff.jsonl
```

`prompts.jsonl` 可以直接用 aisbench 结果文件（自动识别 `origin_prompt` / `prompt` 等字段），
也可以抽成单列：

```bash
jq -c '.[] | {prompt: .origin_prompt}' <aisbench结果.json> > prompts.jsonl
```

replay 模式会带 `return_token_ids: true`，拿到 `prompt_token_ids` 后能定位**首个差异 token**；
服务端不返回 token ids 时自动退化为只比长度（结论不变，但看不到差异片段）。

## 3. 判定与处置

| 输出 | 含义 | 处置 |
|---|---|---|
| `verdict=template-offset` | 每条样本固定偏移 d（默认占比 ≥ 90%、样本 ≥ 3） | 输入口径未对齐，按 §4 核对、按 §5 回代；确认后记 `TRIAGED`，**不进 Dump**，对齐后重跑评测 |
| `verdict=input-not-comparable` | 偏移不固定 | 不是固定模板差：比对 tokenizer 版本、数据集配置、few-shot 构造，继续 C 分支 SB |
| `verdict=tokenizer-mismatch` | 长度一致但 token id 有差异 | 比对 tokenizer / vocab / special tokens 版本 |
| `verdict=aligned` | 长度与 token id 一致 | 输入口径 OK，按 SB 继续，差异在输出侧 |
| `verdict=insufficient` | 匹配样本不足 | 加 `--limit` 或补样本后重跑 |

阈值可调：`--min-pairs`（默认 3）、`--dominance`（默认 0.9）。偏移“基本固定、少量例外”
时先看例外样本是否超长（截断）或构造不同，再决定是否放宽。

## 4. 逐项核对清单

**模型侧（大概率）**：`tokenizer_config.json` 的 `chat_template`、`chat_template.jinja`、
`tokenizer.json` / `special_tokens_map.json` 的 `added_tokens`、`config.json` 是否同一份；
两侧模型目录是否同一版本（用 `compare_model_dirs.py`，小文件走全量哈希）。

**服务侧**：`--chat-template`（含 `--chat-template-content-format`）、`--tokenizer` 覆盖、
`--served-model-name`、`--enable-auto-tool-choice` / `--tool-call-parser`（会改变默认模板行为）、
`--max-model-len`（截断）、`--trust-remote-code`（影响自定义模板加载）。

**客户端 / AISBench 侧**：`vllm_api_stream_chat` 配置里的 `path` / `host_port`、
`generation_kwargs`、dataset 名（`*_0_shot_cot_chat_prompt` 与其他 shot / 非 chat 变体的差别）、
prompt 前缀构造；同一轮评测要保证两侧用的是同一个 dataset 配置。

## 5. 回代验证

单变量回代：权重、采样、并发、并发档位保持不变，只把 good 侧的模板/配置复制到 bad 侧
（或反向），重跑同一批 prompt：

- 偏移消失且精度恢复 → 确认是 template / 配置未对齐，结论 `TRIAGED`；
- 偏移仍在 → 继续往上追 tokenizer 版本、模型目录、`added_tokens`；
- 偏移消失但精度仍掉 → 输入已对齐，回到 SB 继续找输出侧根因（bad case / 算子 / Dump）。

回代必须保留两侧命令与前后 `input_len_summary.json`，避免“改完看着好了”却无法复盘。

## 6. 报告写法

在最终报告的「预检结论」写：命中层 `C2`、命中类型 `template 未对齐`、固定偏移 d tokens、
命中样本比例（如 48/50）、首个差异 token 位置与片段（含两侧 token id 或解码文本）、
回代结论。证据登记：

- `input_len_diff.jsonl`：逐样本 `good_len` / `bad_len` / `delta` / `first_diff`；
- `input_len_summary.json`：`verdict`、delta 分布、首个差异样本的公共前缀/后缀与差异片段。
