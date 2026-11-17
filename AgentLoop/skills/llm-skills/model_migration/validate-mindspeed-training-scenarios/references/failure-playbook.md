# 训练场景故障手册

## 目录

1. 启动与环境
2. 拓扑和 OOM
3. Checkpoint 转换
4. PP 层身份
5. Pack/DSA/TND
6. iteration 与 loss
7. 分布式级联错误

## 1. 启动与环境

先保存完整命令、场景配置 JSON、所有 rank 日志和第一个非零退出码。

| 现象 | 优先检查 |
| --- | --- |
| launcher 立即退出 | shell 语法、变量、路径、权限、解释器 |
| argparse 未识别模型类型 | 外层 CLI、运行时 parser、converter registry |
| import 失败 | 当前设备环境和仓库 `PYTHONPATH` |
| 端口占用 | 场景唯一端口、旧进程 |
| rank 数不符 | `nproc_per_node`、节点数、world size |

不要因某 rank 后续出现 HCCL 超时，就忽略更早的 Python 异常。

## 2. 拓扑和 OOM

### Preflight

重新计算：

```text
dense_DP = WORLD / (TP * PP * CP)
expert_DP = WORLD / (ETP * EP * PP)
grad_accum = GBS / (MBS * dense_DP)
local_experts = num_experts / EP
```

检查整除、SP/TP、layer-list、microbatch 和特殊层所有权。

### OOM 分类

1. 模型参数/optimizer state；
2. activation；
3. dense attention `[B,H,S,S]`；
4. DSA indexer 或 indexer loss；
5. pipeline bubble/microbatch；
6. checkpoint 转换峰值；
7. 内存碎片或残留进程。

配对实验中不要只给 candidate 增加额外内存优化。recompute 配对必须保持相同
TP/PP/CP/EP/ETP、MBS/GBS 和其他内存 flag。

DSA 长序列即使启用 sparse FlashAttention，也可能在 indexer loss 中构造 FP32
dense softmax。分别确认 index selection 和 loss 是否融合。

## 3. Checkpoint 转换

### 每个场景重复转换

检查：

- cache key 是否错误包含场景名/端口/日志等非结构字段；
- marker 与 shard 完整性检查是否在所有 rank 一致；
- 转换完成是否原子发布；
- launcher 是否在 cache 命中后仍传入强制重转 flag；
- 场景是否真的共享模型、拓扑、层切分和架构注册。

### 不完整 cache

目录存在但 marker、目标 iteration 或 shard 缺失时：

- 标记无效；
- 不从部分 shard 加载；
- 保留日志说明缺失项；
- 在受控临时目录重建并原子发布。

### Missing/Unexpected keys

按顺序检查：

1. 模型 slug 与 checkpoint 配置；
2. Spec/架构 flag；
3. TP/PP/EP/ETP；
4. `num-layer-list`；
5. 条件模块和 MTP；
6. converter 两方向的 key mapping。

不得关闭 strict load。相邻层号的 missing/unexpected 配对通常表明 PP offset 错误。

## 4. PP 层身份

对失败 rank 打印静态表，而不是在正式训练中增加噪声：

```text
pp_rank
stage offset
local layer
global layer
expected module type
expected local checkpoint prefix
actual registered keys
```

非均匀 PP 使用前缀和。检查 runtime、HF→MCore 和导出是否都使用同一规则。

shared/full 模块跨 stage 时，确认权重共享或 producer/consumer 通信是否由运行时
支持；不支持的 split 应在 preflight 拒绝。

## 5. Pack/DSA/TND

常见失败：

| 错误 | 根因候选 |
| --- | --- |
| shape/rank 不匹配 | 把 `[B,S,...]` 传给 TND，或未 flatten token |
| int64 被拒绝 | 算子要求 contiguous int32 |
| 长度多一个元素 | 未移除 `cu_seqlens` 开头累计零 |
| 文档间互相注意 | mask 未隔离 packed 文档 |
| recompute 才失败 | metadata 未通过 checkpoint wrapper |
| DSA 某 call site 为 None | 可选 tensor 未贯穿所有互斥路径 |

从 collator 到直接算子逐层记录 shape/dtype/device/value。修复后测试：

- 多文档不同长度；
- 单文档；
- 最短合法序列；
- 错误 dtype/device/rank；
- recompute on/off；
- DSA 稀疏阈值上下两侧。

不要通过关闭 DSA 或 Pack 让测试“通过”。

## 6. iteration 与 loss

### iteration 不对齐

检查：

- 两侧加载 checkpoint iteration 是否一致；
- 日志是否包含恢复前历史行；
- runner 是否追加到旧日志；
- `log_interval` 是否不同；
- 某侧是否提前退出；
- parser 是否把验证 loss 或多 rank 重复行当训练 loss。

比较前截取相同、连续的训练 iteration；缺口不能用插值填充。

### 平均差异超阈值

先验证 intentional delta，再检查：

- 数据顺序和 seed；
- MBS/GBS 与 grad accumulation；
- optimizer/lr schedule；
- checkpoint identity；
- dropout、精度和 fusion；
- TP/PP/EP 带来的合法数值差异；
- loss mask 和 token 数。

主合同按完整对齐区间均值判定。单步 spike 默认是诊断项，但应解释其位置和是否
伴随非有限值或数据边界。

### 不收敛

分别计算两侧首/末窗口均值和全序列线性斜率。若 reference 本身不收敛，不能把
candidate 的均值接近描述为成功实验；先检查学习率、数据和运行长度。

## 7. 分布式级联错误

定位顺序：

1. 最早时间戳；
2. 最小 rank 的首个 Python traceback 或算子错误；
3. 对应场景合同和输入；
4. 后续 collective/HCCL 退出仅标记为级联。

常见级联包括：

- 一个 rank checkpoint key 失败，其他 rank 等待 broadcast；
- 一个 PP stage shape 失败，其他 stage 等待 send/recv；
- 一个 expert rank OOM，其他 rank 等待 all-to-all；
- 一个 rank 数据耗尽，其他 rank 等待 gradient collective。

修复原始错误后重新运行，不能根据级联错误单独增加超时或重试。

## 结果记录

每次故障记录：

```text
scenario/config identity
last completed stage
first primary exception
violated contract
root cause
minimal fix
regression added
artifacts invalidated
next command
```

只有证据支持时才写“根因”；否则标记“当前最强假设”和验证它的最小实验。
