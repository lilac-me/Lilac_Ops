---
name: mindspeed-llm-mcore-open-model-migration
description: >-
  This skill should be used when the user asks to migrate an open-weight
  Hugging Face or custom-code causal language model into MindSpeed-LLM
  Megatron Core/MCore, add model-owned specs and launchers, map or convert
  checkpoints, adapt prompt/data templates, build single-node validation
  scripts, or diagnose TP/PP/CP/EP/ETP, MLA, DSA, MoE, MTP, and layer-identity
  migration failures.
license: MIT
targets-repo:
  - mindspeed-llm
---

# MindSpeed-LLM MCore 开源模型迁移

将开源权重或自定义代码的 decoder-only 因果语言模型迁移到
MindSpeed-LLM Megatron Core（MCore）路径。交付结果应包含模型独占的
`examples/mcore/<model>/` 示例、系列独占 Model Spec、权重映射与转换支持，
以及与实际架构一致的最小验证脚本。

## 读取仓库路由与迁移边界

开始迁移前，先阅读[MindSpeed-LLM 仓库路由](references/repository-routing.md)，
确认后端、MCore 调用链、checkpoint、数据和测试入口。

在决定修改示例、Model Spec、转换器或共享框架代码前，再阅读
[MCore 代码边界](references/mcore-code-boundaries.md)。优先复用已有组件，但不要
因为结构相似就让新模型直接使用另一个系列的最终 Spec。

## 收集输入

先从仓库和模型文件推导，无法推导时再向用户确认：

- MindSpeed-LLM 仓库路径、分支/提交及工作区改动；
- 模型名称、规范化 slug、目标任务和交付范围；
- HF 模型、tokenizer、配置、权重索引及自定义建模代码；
- 期望的 HF→MCore、MCore→HF、在线转换或离线转换路径；
- 设备数、TP/PP/CP/EP/ETP、`num-layer-list`、VPP、MTP 和 noop 层；
- 数据格式、Pack 模式、提示词模板和首次验证目标；
- 是否可执行 NPU 验证；若不可执行，只做静态验证并明确说明。

不得把相似模型目录当作目标模型目录。先将用户模型名规范化为小写 slug，
移除标点和分隔符；若仓库已有权威拼写，则以仓库为准。例如 `GLM5.2`
应落到 `examples/mcore/glm52/`，不得写入 `examples/mcore/glm5/`。

## 执行迁移工作流

维护以下检查表，前一项未通过时不要跳过：

```text
1. 审计 HF 配置、tokenizer、权重键和参考实现
2. 确认目标 slug、示例目录和最近的参考实现
3. 建立全局/局部层身份表
4. 决定 Spec、模型代码和转换分支
5. 实现示例、转换、数据及最小代码修改
6. 审计训练/转换/导出及并行切分一致性
7. 运行静态、转换、加载或最小端到端验证
```

### 1. 审计真实模型资产

不要只依赖 `config.json`。同时检查：

- 层数、隐藏维度、注意力头、KV 头、head dim、词表、norm、RoPE；
- MLA、DSA、MoE、MTP、SSM/混合注意力及辅助损失配置；
- tokenizer 类型、特殊 token、padding、chat template 和真实词表大小；
- safetensors 索引或 state dict 的键前缀、QKV 打包、专家、共享专家、
  辅助模块及 MTP 键；
- 参考 forward 的层构造条件、参数名、cache 和推理专用分支。

配置声明存在但权重索引缺少参数时，不得让运行时无条件注册该参数，除非明确
要求从头初始化并记录这一事实。

对稀疏或条件模块建立表格：

```text
全局层 | 配置类型 | HF 是否有键 | MCore 预期键
PP rank/局部层 | 转换动作
```

### 2. 明确 Model Spec 所有权

模型系列的架构合同应位于：

```text
mindspeed_llm/tasks/models/spec/<series>_spec.py
```

遵循以下规则：

- 同系列且结构兼容：扩展本系列 Spec，并保持已有模型默认行为；
- 同系列但结构重大变化：创建独立变体 Spec；
- 跨系列：创建新系列 Spec，可导入共享组件，但最终 `--spec` 必须归属目标系列；
- 修改共享 Spec 前，搜索所有引用它的示例并评估兼容性；
- 转换、训练、推理和导出必须传入 Spec 所消费的同一组架构参数。

### 3. 固化层身份规则

对所有影响参数注册、特殊模块或 checkpoint 键的模型，先定义：

```text
PP rank + 局部层号 -> 全局 decoder/MTP 层号
                  -> 模块类型 -> 局部 checkpoint 键
```

运行时构造、HF→MCore、MCore→HF、辅助 forward 和日志必须使用同一规则。

- `num-layer-list` 下，PP 偏移为 `sum(num_layer_list[:pp_rank])`；
- 明确 state dict 局部层通常从 0 开始，构造参数 `layer_number` 常从 1 开始；
- 优先使用权威 `layer_offset`，其次使用 `num-layer-list` 前缀和，最后才使用
  已验证的均匀 PP 算法；
- 对每个 PP rank 检查首层、特殊模块边界和末层；
- 同一 rank 同时出现相邻 `Missing`/`Unexpected decoder.layers.N.*` 时，
  先按层身份错误处理，不得关闭 strict load 掩盖问题。

至少覆盖一个均匀 PP 和一个非均匀 `num-layer-list` 回归。

### 4. 选择最小实现分支

优先顺序如下：

1. 仅新增 `examples/mcore/<slug>/` 启动与转换脚本；
2. 扩展目标系列已有 Spec；
3. 新建目标系列或变体 Spec；
4. 新建模型专用 transformer/checkpoint 模块；
5. 仅在行为确实通用且获得相应范围授权时修改共享框架。

任何共享修改都应有模型开关、默认兼容行为和回归验证。

### 5. 保证转换与运行时一致

同时检查：

- QKV、MLP、norm、embedding/LM head、专家和共享专家映射；
- tied/untied embedding、词表 padding 与最终 norm；
- MLA/DSA/MTP 特殊键和不存在参数；
- TP/PP/EP/ETP 及 `num-layer-list` 在两个转换方向中的一致性；
- 独立转换器、运行时参数解析器、`model_cfg.json` 和架构别名均注册
  `--model-type-hf`；
- 转换后严格加载，不得通过忽略缺失/多余键制造成功。

对于 MLA，逐个入口验证：

```text
kv_channels * rotary_percent == qk_rope_head_dim
```

对于 DSA 长序列训练，分别审计索引选择和 indexer loss；支持融合算子时成对
启用 fused lightning indexer 与 loss。BF16/FP16 的通用 DSA indexer 还应
保证 norm 输入与权重 dtype 一致，通常需要 `--init-norm-weight-in-fp32`。

### 6. 生成仓库风格脚本

- 从同任务、同系列或最接近模型的维护中示例复制结构；
- 保留相邻示例的 shell 序言、变量分组和最终短启动命令；
- 让 `DISTRIBUTED_ARGS`、`GPT_ARGS`、MLA/DSA/MoE/MTP、数据和输出参数
  各自归属清楚；
- 不在启动脚本中嵌入仅用于生成期检查的拓扑计算器、硬件发现、重试循环或
  大型验证框架；
- 不写入 `TORCH_HCCL_ZERO_COPY`；
- 不保留未使用变量、重复参数或互相冲突的 PP/VPP 配置；
- 正反转换脚本必须携带相同的结构参数。

正式预训练脚本是功能清单的权威来源。ST 脚本从它生成，并用明确 allowlist
记录差异。

### 7. 默认生成两类单机 ST

用户未明确指定时，生成且仅生成：

1. 降层/降专家的无模型切分 ST；
2. 保留完整层数和层身份的 PP/专家缩减 ST。

无切分 ST：

- 设置 TP/PP/CP/EP/ETP 为 1，允许普通数据并行；
- 删除 SP、CP 算法、`num-layer-list`、VPP、noop 分层和 all-to-all dispatcher；
- 优先缩减层数和专家数，不随意改变隐藏维度/头维度；
- `num_experts >= moe_router_topk`；
- 校验 `GBS % (MBS * WORLD_SIZE) == 0`；
- `--swap-optimizer` 必须和 `--use-distributed-optimizer` 成对出现；
- 文件名必须真实表达 `no-parallel` 或 `single-node`。

完整层 ST：

- 保留正式模型层数和合法的 PP 层列表；
- 保留全局特殊层身份；
- 缩减专家时尽量保持每 rank 本地专家数；
- `--moe-fb-overlap` 仅在 `ETP=1` 且 `EP>1` 时保留。

只有用户明确要求时才新增全并行压力覆盖脚本。

## 在生成期验证约束

在报告或静态检查中验证，而不是把计算器塞进启动脚本：

```text
WORLD_SIZE % (TP * PP * CP) == 0
WORLD_SIZE % (ETP * EP * PP) == 0
num_experts % EP == 0
GBS % (MBS * DP) == 0
num-layer-list 长度 == PP，且总和 == decoder 层数
MLA rotary width 一致
特殊层和 checkpoint 键归属一致
```

先追踪当前运行时实际注册的并行状态初始化器。启用 MoE Parallel Folding 时，
不要套用过时的 `(DP * CP) % EP == 0` 规则；应分别计算 dense DP 与 expert
DP，并验证实际 rank group 约束。

## 验证顺序

按可用环境逐级执行：

1. YAML/脚本语法、引用、命名和参数差异；
2. 模板目录零意外变更检查；
3. 均匀与非均匀 PP 层身份回归；
4. 两个转换方向的参数与模型类型注册检查；
5. 转换 dry-run 或抽样 state dict 检查；
6. 目标拓扑的模型初始化与 strict load；
7. 单机 ST；
8. 真实数据、序列长度和并行拓扑的完整端到端。

没有运行环境时，明确标注“仅静态验证”，给出用户下一条可直接运行的最小命令
和预期产物。

## 输出要求

最终报告必须包含：

- 迁移分支、目标 slug、目标示例目录和参考模板目录；
- Spec 所有权决策及最终 `--spec`；
- 按示例、模型/Spec、转换、数据和验证分类的文件清单；
- 权重键、条件模块、PP/TP/CP/EP/ETP、MTP 和现有模型影响决策；
- 模板目录未被意外修改的证据；
- 已执行验证、未验证范围和最小下一步命令；
- strict load 失败时首先检查的键或层身份模式。
