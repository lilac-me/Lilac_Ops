---
name: fsdp2-precision-alignment
description: 对 MindSpeed-LLM FSDP2 模型适配结果做精度对齐。适用于用户在完成 HuggingFace/native modeling 到 FSDP2 后端迁移后，要求用原生 modeling 文件、权重路径、Docker/服务器环境自行验证 logits 精度，生成减层对齐 harness、固定输入、native/FSDP2 logits、CPU 比较报告、中文精度对齐报告，或诊断 tokenizer、checkpoint、layer mapping、RoPE、mask、dtype、融合算子、FSDP2/TP/PP/CP/EP 精度问题的场景。
targets-repo:
  - mindspeed-llm
---

# FSDP2 精度对齐

## 核心规则

把精度对齐当成输入、权重、架构、执行路径、输出和指标的联合合同，不把一次余弦计算当成最终证明。

Reference 必须是用户提供的原生 Transformers `modeling_*.py` 推理。Candidate 必须进入 MindSpeed-LLM 仓库的真实 FSDP2 训练路径：`train_fsdp2.py` 参数体系、`ModelFactory.create`、`MindSpeedParallelEngine`、FSDP2/EP 包装和 `Trainer._compute_loss`。直接实例化适配后的 `modeling_*.py` 并调用 forward 只能算 standalone modeling smoke，禁止标记为 FSDP2 精度通过。

用户最少只需要提供：

- 原生 `modeling_*.py` 链接或路径。
- 权重目录路径；`config.json`、tokenizer、权重索引默认从该目录读取。
- 可运行 Docker/服务器环境。

若环境不能直接运行原生 modeling，例如 Transformers 版本不支持目标 `model_type`，必须在报告中标记为“环境前置失败/未认证”，不要把 fallback 结果说成原生 HF 对齐通过。

## 默认验收合同

默认门禁：

```text
相同 input_ids、attention_mask 和有效 token 位置
相同权重或明确声明的同源随机权重
完整 logits shape = [batch_size, sequence_length, vocab_size]
正式精度验收 sequence_length = 4096，且 4096 个位置全部是有效 token
logits 保存为有限 FP32
有效 token 的逐 token 全词表 logits 平均余弦 >= 0.98
```

严格执行：

- 只 tokenize 一次，native 和 FSDP2 复用同一个输入 payload。
- 正式验收默认 `batch_size=1, sequence_length=4096`；必须检查 `attention_mask.sum()==4096`，禁止用 padding 把短提示伪装成 4096。
- 短序列仅可标记为 smoke，不得替代 4096 序列的正式精度结论。
- 4096 reference 前向单卡放不下时不得缩短输入；先使用 `--device-map balanced`，自动映射仍产生热点时使用 `--device-map-json` 把 embedding、各 Transformer 层和输出头显式映射到多张卡。
- Reference 可使用 Accelerate 设备映射；Candidate 不得复用 Accelerate 映射冒充 FSDP2，必须由 `torchrun` 启动多进程并使用训练 YAML 中的 FSDP2/EP/TP/CP 配置。报告分别记录 reference 映射和 candidate distributed world size/parallel config。
- Candidate payload 必须记录并通过硬门禁：execution kind、`train_fsdp2.py` entrypoint、ModelFactory、ParallelEngine、Trainer forward、distributed backend/world size、FSDP size、EP size、writer rank 和 FSDP2 wrapped module count。
- Candidate 必须读取框架 WeightLoader 的加载报告并要求 `missing_parameter_count == 0`；存在随机初始化的缺失参数时，不得执行数值比较。
- 正式 FSDP2 认证默认要求 `world_size >= 2`、`fsdp_size == world_size`、FSDP2 wrapped module count `> 0`；任一不满足直接 BLOCKED。
- 每个批次必须通过 `--execution-label` 记录实际 candidate 路径；eager、单融合和组合融合不得共用一个含糊的 PASS 标签。
- payload 必须包含 `input_ids`、`attention_mask`、`input_sha256` 和元数据。
- 只比较 `attention_mask == 1` 的有效 token。
- 比较完整词表 logits，不用生成 token、概率或 hidden state 替代。
- 只裁剪明确记录的 vocab padding，并记录裁剪前后大小。
- shape/hash 不一致、NaN/Inf、近零向量、缺失权重或随机初始化未声明时直接失败。
- MoE 必须从权重索引统计实际 expert id 连续范围，并与 config 的 expert 数交叉校验；不一致时在推理前阻塞。只有核实权重语义后才允许通过 checkpoint 视图的显式 config override 修正，禁止静默裁专家或对 HashRouter id 取模。
- 平均 token 余弦是主门禁；Top-1、Top-k、绝对误差只做诊断。

## 执行流程

1. 固定代码、模型、权重路径、commit、Docker/服务器环境和验收阈值。
2. 从权重目录读取 config/tokenizer/权重索引，审计 `model_type`、层数、特殊字段和 tokenizer。
3. 审计原生 modeling 与 FSDP2 modeling 的层映射，列出 `global_layer -> native module -> fsdp2 module`。
4. 生成唯一固定输入，保存 `input_sha256`。
5. 先执行减层对齐：`1 -> 2 -> 4 -> full`；权重目录本身少于 4 层时按实际层数截断。
6. 第一轮关闭融合算子，但保留真实 distributed FSDP2；MoE 模型按训练配置启用 EP，建立框架级 eager 数值基准。
7. 导出 native/reference logits；失败时保存环境错误和堆栈。
8. 通过 `torchrun` 导出 FSDP2/candidate logits；必须调用 `ModelFactory.create` 构建包装模型，并通过 `Trainer._compute_loss(return_outputs=True)` 执行训练模式单步前向。
9. 在 CPU 上比较两份 payload，生成 JSON 报告。
10. 生成中文报告，按阶段写 PASS/FAIL/BLOCKED。
11. 基准通过后，每次只打开一个优化：fused RMSNorm、fused RoPE、FlashAttention、grouped GEMM、EP dispatcher、FSDP2/CP。
12. 若失败，从最早破坏的不变量开始诊断，不调阈值掩盖结构错误。

## 通用 Harness 产物

通用能力集中放在本 Skill 的 `scripts/`，模型仓库只保留薄适配器和 launcher，禁止为每个模型复制一整套比较脚本：

```text
skills/llm-skills/fsdp2-model-adaptation/fsdp2-precision-alignment/scripts/
  alignment_utils.py
  make_fixed_input.py
  dump_transformers_logits.py
  dump_mindspeed_fsdp2_logits.py
  compare_logits.py
  prepare_checkpoint_view.py
  run_alignment.py
  write_alignment_report.py
  audit_alignment_harness.py
  setup_transformers_runtime.sh

examples/fsdp2/<model>/precision_alignment/
  fsdp2_framework_alignment.yaml
  run_complete_alignment.sh
  run_synthetic_inference_smoke.py  # 可选，只证明结构/前向，不证明 checkpoint 精度
```

脚本要求：

- 不 source CANN、Conda 或系统环境；只继承用户给定容器当前环境。
- harness 运行时不隐式安装、升级或卸载任何包。若原生 modeling 需要更新 Transformers，必须由用户或执行流显式调用 `setup_transformers_runtime.sh` 创建隔离 venv；只在 venv 中安装固定 commit 的 Transformers/HF 依赖，不改基础环境的 `torch` 和 `torch_npu`。
- 所有路径和关键执行范围可由环境变量覆盖：`MODEL_DIR`、`REPO_ROOT`、`OUTPUT_DIR`、`MODEL_ID`、`LAYER_COUNTS`。
- launcher 不能把文档占位路径写死成默认值。
- 每个阶段输出独立目录，避免旧产物污染新结果。
- `run_alignment.py` 自动检测 tokenizer 是否含 `chat_template`；没有模板时使用 raw-text tokenization，并把选择写入固定输入元数据。
- `prepare_checkpoint_view.py` 只建立 symlink 视图并裁剪 config/index，不改原始权重。缺失当前层实际需要的任何 shard 时必须落盘 manifest 后阻塞，禁止随机补齐。
- `prepare_checkpoint_view.py` 必须记录 checkpoint 推导的 expert 数和所有 config override；expert 数不一致且没有显式修正时直接阻塞。
- `dump_transformers_logits.py --alignment-adapter` 只允许用于 standalone modeling smoke 或定位算子误差；正式 candidate 必须使用 `dump_mindspeed_fsdp2_logits.py`。
- `dump_mindspeed_fsdp2_logits.py` 必须复用训练 YAML 和仓库的参数解析器，不得在 harness 中手写另一套 FSDP 包装；固定输入可以绕过 dataloader，但 batch 字段和 loss 前向必须进入 `Trainer._compute_loss`。
- 每次执行保存 `run_record.json`、逐命令日志、checkpoint manifest、完整 logits payload、comparison JSON 和中文报告；命令失败也要保留记录。

## 减层对齐

默认层数序列：

```text
1, 2, 4, full
```

如果 `config.num_hidden_layers < 4`，使用可用层数去重后执行。例如 CI 裁剪权重为 4 层，则运行 `1,2,4`。

对 per-layer schedule 模型必须同步裁剪：

- `layer_types`
- `compress_ratios`
- `mlp_layer_types`
- `num_hidden_layers`
- `num_hash_layers` 不能超过当前裁剪层数

MoE 模型要额外记录 HashRouter/TopKRouter 层、expert 数、top-k、EP 是否参与。

若减层权重仍保留完整专家，而源 `config.json` 写的是更小专家数，先判定该目录为 config/checkpoint 不一致。HashRouter 的 `tid2eid` 可能包含完整专家范围，不能仅裁前 N 个专家权重；正式对齐应显式恢复 checkpoint 对应专家数，并据单卡显存选择足够的 EP 规模。

## 适配后验证常见兼容点

从 HuggingFace main 复制新模型到仓库环境时，先用减层 harness 暴露环境差异，再把兼容逻辑收敛到新模型目录内：

- 配置类如果依赖 `strict`、dataclass 或 `__post_init__`，确认当前 `PreTrainedConfig` 是否支持同名 hook；不支持时要手写 `__init__`，保留 extra kwargs，并手动展开 legacy 字段，例如 `compress_ratios -> layer_types`、`num_hash_layers -> mlp_layer_types`。
- `auto_docstring` 只服务文档生成；目标 Transformers 版本未注册新模型时，可以在本地适配文件中退化为 no-op，不影响数值路径。
- 自定义 router 激活要注册到当前版本的 `ACT2FN`。如果该版本 `ACT2FN.__getitem__` 会实例化 class，则注册 `nn.Module` 类；如果返回函数，则注册 callable。
- `create_sliding_window_causal_mask`、cache、RoPE、输出 dataclass 等 API 经常跨版本变更；优先用 `inspect.signature` 查容器中的真实签名，再做最小兼容包装。
- 不使用不存在的初始化 API，例如 `torch.nn.init.copy_`；复制 buffer/参数用 `with torch.no_grad(): target.copy_(source)`。
- 若官方 `AutoConfig/AutoModel` 不识别权重目录里的 `model_type`，报告中把 native Auto 路径标记为 `BLOCKED`，再用本地 eager reference 证明 FSDP2 适配路径可运行，但不能宣称原生 HF 对齐已通过。
- Transformers 新版本的 `output_loading_info` 可能用 `set` 返回 missing/unexpected keys；harness 应把 list、tuple、set、frozenset 统一规范化后再做空集合门禁。
- 自定义 modeling 复用官方扁平 checkpoint 时，需同时归一化 `config.model_type`，并按自定义模型类名注册官方 conversion mapping，防止 custom-code 保护逻辑跳过权重转换。
- 融合算子验证必须同时记录 requested 和 active 状态；运行时依赖不齐导致开关未生效时直接失败，不能仅凭配置值宣称融合路径已执行。

## 诊断顺序

低余弦或 NaN 时按以下顺序检查：

1. input hash、tokenizer、padding、special token 是否一致。
2. checkpoint 是否同源，权重 key 是否 missing/unexpected。
3. global/local layer mapping 是否错位。
4. embedding、LM head、norm、Q/K/V、gate/up/down、router/expert 权重 shape 是否一致。
5. attention mask 的 rank、dtype、`True=masked` 或 additive `0/-inf` 语义是否一致。
6. RoPE 的 position ids、interleaved 约定、partial rotary 维度、scaling 是否一致。
7. dtype 是否一致，是否混入 fp32/bf16/fp16。
8. fused kernel 是否改变 layout、contiguity 或 shape order。
9. FSDP2/TP/PP/CP/EP 的 writer rank 和 logits gather 是否正确。

## 指标语义与报告解释

所有指标只统计 `attention_mask == 1` 的有效 token。设有效 token 数为 `valid_token_count`，词表大小为 `vocabulary_size`，第 `token_index` 个 token 的 Reference/Candidate 完整词表 logits 分别为 `reference_logits[token_index, :]` 和 `candidate_logits[token_index, :]`。

每份中文精度报告必须在数值结果表之前包含下面的指标说明；禁止只输出 `P01`、`P05`、`Top-1`、`Top-5 overlap` 等未解释缩写。

| 字段 | 精确定义 | 解读方式 |
| ------ | ---------- | ---------- |
| `average token cosine` | 对每个有效 token 的两份完整词表 logits 计算 cosine，再对 `valid_token_count` 个结果求平均 | 主验收指标；越接近 1，整体 logits 方向越一致 |
| `minimum token cosine` | 所有有效 token cosine 的最小值 | 最差单个 token；容易受孤立异常点影响 |
| `P01 token cosine` | `torch.quantile(token_cosines, 0.01)`，即 token cosine 的第 1 百分位 | 约 99% 的有效 token cosine 不低于该值，用于观察最差约 1% token |
| `P05 token cosine` | `torch.quantile(token_cosines, 0.05)`，即 token cosine 的第 5 百分位 | 约 95% 的有效 token cosine 不低于该值，用于观察最差约 5% token |
| `median token cosine` | `torch.quantile(token_cosines, 0.50)` | 一半 token 不低于该值；比 minimum 更能反映典型 token |
| `maximum token cosine` | 所有有效 token cosine 的最大值 | 最接近的单个 token，不作为门禁 |
| `first/last valid token cosine` | 每条输入第一个/最后一个有效 token 的 cosine | 用于诊断首 token、长序列尾部误差和位置累积问题 |
| `Max Abs` | `max(abs(reference_logits - candidate_logits))`，在所有有效 token 和完整词表元素上取最大值 | 最严重的单个 logit 绝对偏差 |
| `Mean Abs` | 所有有效 token、所有词表元素绝对误差的平均值 | 平均逐元素偏差，受词表尺度影响 |
| `RMS Error` | `sqrt(mean((reference_logits - candidate_logits)^2))` | 对较大误差更敏感的逐元素误差指标 |
| `Top-1` | 每个有效 token 比较 `argmax` 词表 id；相同 token 数除以 `valid_token_count` | 最终第一候选词一致率；乘以 token 数可估算一致位置数量 |
| `Top-k overlap` | 对每个有效 token 计算 `len(reference_topk_set ∩ candidate_topk_set) / k`，再对 token 求平均 | 只比较候选集合，不要求集合内排名相同；`Top-5 overlap * 5` 表示每个 token 平均重合候选数 |

报告必须明确 `P01/P05` 是跨有效 token 的 cosine 分位数，不是元素误差分位数，也不是通过率；`Top-1` 是精确词表 id 一致率；`Top-k overlap` 是集合交集比例，不是 Top-k 排名相关系数。绝对误差与 RMS 在完整词表 logits 元素上统计，不能与 token cosine 的统计轴混用。

结果解读至少包含：

- 用 `average token cosine` 对照主阈值给出 PASS/FAIL。
- 用 minimum、P01、P05 判断误差是少量异常 token 还是整体退化。
- 将 `Top-1 * valid_token_count` 换算为近似一致 token 数，将 `Top-k overlap * k` 换算为每 token 平均重合候选数。
- 同时报告绝对误差和 RMS，但不得用 Top-1/Top-k 或绝对误差替代主 cosine 门禁。

## 报告结构

中文报告必须包含：

- 模型、modeling 来源、权重路径、repo commit、容器/设备环境。
- Reference 与 Candidate 的边界：原生 Transformers modeling vs MindSpeed-LLM FSDP2 training forward。
- Candidate 的 entrypoint、ModelFactory、ParallelEngine、Trainer forward、world size、FSDP/EP/TP/CP、FSDP wrapped module count 和 writer rank。
- 固定输入 hash、input shape、有效 token 数。
- 减层结果表：layer count、native 状态、FSDP2 状态、平均余弦、是否通过。
- 指标说明：在数值表之前完整解释 average/min/median/max/P01/P05 token cosine、first/last token cosine、Max Abs、Mean Abs、RMS Error、Top-1 和 Top-k overlap 的统计轴、公式与含义。
- 指标结果：使用上述统一字段输出逐层数据，并把 P01/P05、Top-1 和 Top-k overlap 换算为可读结论。
- 失败阶段和最早违反的不变量。
- 已验证项、未验证项、fallback 是否发生。
- 后续打开融合算子或并行路径的阶段建议。

## 输出要求

最终回复用中文说明：

- harness 路径和报告路径。
- 哪些阶段 PASS/FAIL/BLOCKED。
- 主指标和阈值。
- 是否使用了 fallback；若使用，明确哪些结论不能替代原生 HF 对齐。
- 远端环境中实际执行过的命令和产物位置。
