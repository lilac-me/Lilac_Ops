# Shape 流程分析

## 目录

- 必须回答
- Shape 符号表
- Shape 分析层级
- 单步 Shape 描述格式
- 流程图输出规范
- 复杂流程数值示例
- Shape 校验
- Shape 可复现性与最低验证集

## 必须回答

1. 输入 Token 如何经过 Embedding、各类 Decoder Layer 和 LM Head；
2. 每个关键 Tensor 的名称、Shape 和语义；
3. 每一步经过什么运算得到下一个 Tensor；
4. 权重 Shape 与输入输出 Shape 的对应关系；
5. Attention、MoE、Indexer、Cache、Mask、并行切分等复杂部分的 Shape；
6. 优先梳理与训练相关的前向分支；若源码同时存在 Prefill、Decode、Cache 命中等推理分支，仅说明与训练 Shape 或实现存在实质差异的部分，不强制展开纯推理路径；
7. 对复杂 Mask、索引恢复、路由选择等流程提供小规模数值示例；
8. 输出可直接阅读的总分式流程图。

## Shape 符号表

| 符号 | 含义 |
|---|---|
| `B` | Batch Size |
| `S`或`S1` | 当前 Query 序列长度 |
| `S2` | KV Cache 总长度 |
| `H` | Hidden Size |
| `N` | Attention Head 数 |
| `Dh` | Head Dim |
| `E` | Expert 数 |
| `K` | Top-K |

每个报告必须先给出符号表，避免同一个字母在不同模块中含义不同。

## Shape 分析层级

### Level 0：整网

```text
input_ids
→ embedding
→ decoder layers
→ final norm
→ lm_head
→ logits/loss
```

### Level 1：层类型

分别绘制：

- Dense Attention + Dense MLP；
- Dense Attention + MoE；
- MLA/DSA + MoE；
- Linear Attention + MoE；
- MTP Layer；
- 其他条件分支。

### Level 2：复杂模块

单独展开：

- Attention Q/K/V 路径；
- MLA 低秩路径；
- Indexer；
- MoE Router 和 Expert Dispatch；
- Linear Attention State；
- Mask；
- MTP；
- Loss。

### Level 3：关键算子

对以下运算给出输入输出 Shape：

- Linear；
- MatMul/BMM/Einsum；
- Reshape/View；
- Transpose/Permute；
- Gather/Scatter；
- TopK；
- Softmax；
- Conv；
- AllToAll/AllGather/ReduceScatter；
- 自定义融合算子。

## 单步 Shape 描述格式

```text
Tensor A
名称：query
Shape：[B, S, N, Dh]
语义：每个 Query Token 的多头查询向量

运算
名称：QK MatMul
公式：query @ key.transpose(-1, -2)
缩并维度：Dh

Tensor B
名称：attention_scores
Shape：[B, N, S1, S2]
语义：每个 Query 对所有 Key 的相关性分数
```

不得只写 `[B,S,H] → [B,N,S,Dh]`，必须说明 Tensor 名、Shape、运算、权重、缩并维度、广播维度和语义。

## 流程图输出规范

### 视觉分类

| 类型 | 建议形状 | 建议颜色 |
|---|---|---|
| Tensor/中间变量 | 圆角矩形 | 浅蓝或浅灰蓝 |
| 运算过程 | 六边形或窄矩形 | 中性灰 |
| 权重/参数 | 带双边框矩形 | 浅金或浅米色 |
| Mask/索引 | 虚线边框矩形 | 浅灰 |
| Cache/状态 | 圆柱或带状态标记矩形 | 浅紫灰 |

要求：

- 使用非衬线字体；
- 每一个 Tensor 单独一个单元格；
- 每一个运算单独一个单元格；
- 权重不得与 Tensor 共用单元格；
- 每张流程图固定图例；
- 不使用五颜六色；
- 单元格文字与单元格大小成比例；
- 不出现大段无意义留白；
- 连线必须标注运算或维度变化；
- 总览页只展示主干；
- 细节页展开模块内部；
- 复杂 Mask 旁边提供小矩阵或 Token 条带可视化；
- 示例值与真实模型默认值必须明确区分。

### 推荐流程图分区

1. 整网总览；
2. 典型 Decoder Layer；
3. Attention/Linear Attention；
4. MoE；
5. Indexer；
6. Cache 和 Mask；
7. 训练/推理分支；
8. Shape 总表。

## 复杂流程数值示例

对以下流程必须提供小规模示例：

- TopK；
- 压缩和索引展开；
- Causal Mask；
- Sliding Window；
- MoE 路由；
- AllToAll；
- Cache 更新；
- Chunk/State 更新。

示例建议使用：

```text
S = 8、16 或 32
N = 2 或 4
TopK = 2
Compress Ratio = 2 或 4
```

示例必须包含真实索引或矩阵，不得只写抽象公式。

## Shape 校验

优先执行运行时校验：

```python
hooks = []

def hook(name):
    def fn(module, inputs, outputs):
        print(name, shape_of(inputs), shape_of(outputs))
    return fn
```

可选方法：

- `register_forward_hook`；
- FakeTensor；
- Meta Device；
- `torch.fx`；
- `torch.export`；
- 小配置实例化；
- 单模块单元测试；
- 在自定义 Kernel 前后增加 Shape Assert。

校验项：

- `reshape` 前后元素数一致；
- `transpose` 后维度顺序正确；
- `einsum` 缩并维正确；
- Cache 的 `S/T` 语义正确；
- Padding 与 Causal Mask 广播正确；
- TopK 输出宽度稳定；
- TP/EP/CP 后局部 Shape 正确。

## Shape 可复现性与最低验证集

Shape 报告必须区分以下边界：

- 主模型与附件模块；
- 训练 Forward 与推理 Forward；
- Prefill、Decode、Cache Hit 和 Speculative 分支；
- 全局逻辑 Shape 与 TP/EP/CP 本地 Shape；
- 逻辑有效宽度与 Kernel Padding/Block 对齐后的执行宽度。

即使无法实例化模型，也必须完成最低静态验证集：

1. 源码语法或 AST 解析通过；
2. 展开层索引的数量等于配置层数，附件层不混入主层；
3. 每个关键 `reshape/view/unflatten` 前后元素数守恒；
4. 每个 MatMul/Einsum 的缩并维相等；
5. 分组投影、Head 拆分和 TP 本地维度可整除；
6. Cache Offset、Ring Slot、Causal Mask 和 TopK 用至少一个真实小例子验证；
7. Markdown Shape 表与 `model_analysis.json.shape_flow` 使用相同符号和结果。

验证结果必须使用以下表格，不得把静态断言描述为运行时验证：

| 校验项 | 方法 | 结果 | 证据/限制 |
|---|---|---|---|

若完成运行时 Hook、FakeTensor、Profiler 或目标设备测试，再单独列出环境、输入 Shape、dtype、Commit 和结果。

---
