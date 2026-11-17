# 结构与参数分析

## 目录

- 必须回答
- 配置参数提取
- 层类型统计
- 参数量分析
- 核心创新分析

## 必须回答

1. 模型有哪些核心配置参数；
2. 模型由哪些层和模块构成；
3. 各类层分别有多少层；
4. Dense、MoE、Attention、Indexer、MTP 等模块如何组合；
5. 总参数量、激活参数量如何构成；
6. 相比此前主流模型，核心结构创新是什么；
7. 哪些创新属于模型结构创新，哪些只是工程实现优化；
8. 训练路径与推理路径是否一致。

## 配置参数提取

按以下维度整理参数（维度内参数为举例，以实际config为准）。

### 基础参数

- `vocab_size`
- `hidden_size`
- `num_hidden_layers`
- `max_position_embeddings`
- `hidden_act`
- `rms_norm_eps`
- `tie_word_embeddings`

### Attention 参数

- Linear Attention/SSM State Size；
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- Q/K/V 投影维度；
- LoRA Rank；
- RoPE 维度；
- Local Window；
- Sparse TopK；
- Compression Ratio；
- Cache Shape。

### MoE 参数

- Routed Expert 数；
- Shared Expert 数；
- Top-K；
- Expert Intermediate Size；
- Router 类型；
- Load Balancing 策略；
- Expert Bias；
- Capacity/Drop 策略。

### 其他结构参数

- Conv Kernel；
- MTP 层数；
- Dense Layer 数；
- 混合 Attention 排布；
- Indexer Head 和 Index TopK；

输出表：

| 参数类别 | 参数名 | 数值 | 作用 | 源码位置 |
| --- | --- | ---: | --- | --- |

## 层类型统计

必须解析每一层真实类型，不能只使用 `num_hidden_layers`。

输出：

| 层类型 | 层索引 | 层数 | Attention | FFN | 其他模块 |
| --- | --- | ---: | --- | --- | --- |

对交替层、周期层、条件层，必须给出生成规则和展开后的实际层索引。

## 参数量分析

参数量必须从模块到层到整网逐级拆解。

### 参数统计口径

- Linear：`out_features × in_features`，有 Bias 时加 `out_features`；
- Embedding：`vocab_size × hidden_size`；
- Norm：Weight 和 Bias 分别计数；
- MoE：
  - Total Parameter：全部 Expert；
  - Active Parameter：Top-K Expert + Shared Expert + Router；
- Buffer 不计入可训练参数；
- Tie Weight 只能计一次；

输出：

| 级别 | 模块名 | 权重 Shape | 数量 | 总参数量（精确值） | 总参数量（B） | Active 参数量（精确值） | Active 参数量（B） | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

参数量展示必须同时保留精确值和 B 换算值：

```text
parameters_billions = parameters_exact / 1_000_000_000
```

- 精确值是权威统计值，使用十进制整数；可使用千分位提升可读性，但不得使用科学计数法或缩写替代；
- B 列使用十进制 `10^9`，不使用 `2^30`，不得从 Checkpoint 字节数反推；
- 整网、附件和层型汇总的 B 值默认保留 3 位小数；模块级 B 值默认保留 6 位小数；
- 非零模块在 6 位小数下显示为 `0.000000 B` 时，写为 `<0.000001 B`，精确值列仍保留真实整数；
- Total 和 Active 必须各自拥有精确值列和 B 列；不适用的 Active 单元格使用 `N/A`，不得写 `0`；
- 摘要中的主模型 Total/Active、附件参数和 Checkpoint 逻辑参数合计也必须使用“精确值 + B”双口径展示；
- 论文或模型卡的 B 级近似值单列为“公开口径”，不得覆盖从源码推导的 B 换算值。

必须提供：

- 模块级参数；
- 典型层参数；
- 各层类型汇总；
- 整网参数；
- Total/Active 比例；
- 所有参数汇总和明细表的精确值/B 双列；
- 与框架 `numel()` 的校验差值。

验收要求：

```text
理论参数量与实际 state_dict/parameters 统计误差 = 0
```

若无法运行代码，应明确标注“静态推导结果，未运行校验”。

### 参数三账本与 Active 多口径

参数分析必须同时维护三套不可混用的账本：

1. **逻辑参数账本**：按解包后的数学权重 Shape 计数，用于结构与 FLOPs 推导；
2. **Active 参数账本**：按每 Token 实际选择的 Expert 和固定执行模块计数；
3. **Checkpoint 物理账本**：记录量化后字节数、Scale、分片和 Tensor 条目，用于存储与部署评估。

强制要求：

- 主模型与 MTP、Speculative Decoder、Vision Tower、Reward Head 等附件必须分开列账，再给合计；
- FP4/INT4 打包权重在逻辑账本中按解包 Shape 计数，在物理账本中按真实字节计数；
- 量化 Scale、Zero Point 和辅助元数据默认不计逻辑模型参数，但必须进入物理账本说明；
- 共享或别名参数按对象去重；`state_dict` 中的重复 Key 不得重复计入逻辑参数；
- 固定路由表、Cache、RoPE 表等非训练对象单列，不得混入可训练参数；
- 论文只给整数 B 级口径时，必须同时给精确推导值与论文近似值，说明两者是否一致；
- 不得用 Checkpoint 字节数反推逻辑参数量，也不得用逻辑参数量替代部署存储量。

对于未绑定 LM Head 的模型，Active 参数至少报告：

| Active 口径 | 组成 |
| --- | --- |
| Transformer Body Active | 层内固定参数 + Router + Top-K Routed Expert + Shared Expert |
| 论文可比 Active | 在 Body 上按论文惯例明确加入或排除 Embedding/Head |
| Logits 执行 Active | 实际生成全词表 Logits 时再加入未绑定 LM Head |

若无法确认论文惯例，必须同时报告可比口径和执行口径，不得只给一个无定义的 Active 数字。

对复杂模型，优先编写一次性独立计算脚本，从配置重新计算参数和 FLOPs；脚本仅作为中间校验材料，不额外作为最终交付物，除非用户要求。

## 核心创新分析

| 创新点 | 旧方案 | 当前方案 | 解决问题 | 训练收益 | 推理收益 | 代价与风险 |
| --- | --- | --- | --- | --- | --- | --- |

创新分析必须区分：

1. 架构创新；
2. 路由或稀疏化创新；
3. 状态压缩创新；
4. 并行或通信创新；
5. 算子融合创新；
6. 训练技巧；
7. 推理优化。

不得把普通实现差异包装成模型创新。

---
