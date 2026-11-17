# 精度对齐故障诊断手册

## 目录

1. 分类失败阶段
2. Checkpoint 与输入
3. NaN/Inf
4. 低余弦模式
5. Mask、RoPE 与 dtype
6. 分布式与融合算子

## 1. 分类失败阶段

先记录最后完成阶段：

```text
environment -> gate -> runtime regression -> model construction
-> strict load -> forward -> save -> compare -> finalize
```

- 环境验证失败：不表示模型或 checkpoint 错误；
- 门禁预期拒绝：不表示目标阶段实现错误；
- preflight/fused regression 失败：先修 tensor/operator 边界；
- 模型构造前失败：不要声称 checkpoint load 或 forward 失败；
- 多 rank NPU/HCCL 报错：先找第一个 rank 的最早 Python/算子异常。

## 2. Checkpoint 与输入

低余弦前先排除：

- HF 与 MCore 权重来源不同；
- 转换参数和训练/推理参数不一致；
- strict load 有 missing/unexpected/mismatched/newly initialized；
- TP/PP/EP/ETP 或 `num-layer-list` 与 checkpoint 不同；
- 两侧独立 tokenize；
- hash、mask、padding、特殊 token 或词表裁剪不同；
- logits layout 不是 `[B,S,V]`；
- 比较了错误 writer rank 或陈旧产物。

`Missing decoder.layers.N.x` 与相邻 `Unexpected decoder.layers.M.x` 成对出现时，
优先审计全局/局部层偏移。

## 3. NaN/Inf

不得替换或裁剪非有限值。定位第一个产生非有限输出的模块：

1. 开启同步执行以确定真实失败点；
2. 在模块边界记录 shape、dtype、有限性和极值；
3. 二分缩小到 embedding、attention、norm、MLP/MoE 或 LM head；
4. 检查混合 dtype、错误 mask、空/全遮蔽行、溢出和未加载权重；
5. 将复现转成小型行为回归。

HF 多设备异步调度若已知需要 module-output synchronization，必须默认保留并写入
元数据。同步诊断模式只能定位问题，不能证明无同步的生产路径有效。

## 4. 低余弦模式

| 现象 | 优先检查 |
| --- | --- |
| 全部 token 接近 0 | checkpoint、架构、QKV/MLP 映射、layout、词表 |
| 首 token 很低 | embedding、LM head、norm、dense/MoE、基础架构 |
| 早期高、后续衰减 | mask、position ids、RoPE、cache、累计 kernel 差异 |
| 仅长序列下降 | RoPE scaling、稀疏选择、数值累计、长序列 kernel |
| 仅优化阶段下降 | 吸收/融合/重排、matmul 结合顺序、dtype |
| 平均通过但个别 token 很低 | padding、文档边界、局部 mask 或 near-tie |
| 余弦高但 Top-1 不满 | 候选 logits 接近时可能正常；查看 margin |

不要调阈值掩盖结构错误。先对照首 token、逐 token 曲线、低分位和最差位置。

## 5. Mask、RoPE 与 dtype

### Mask

- 记录外层和直接消费者的 shape/dtype/value；
- bool `True=masked` 转成精确 shape 的 additive `0/-inf`；
- 原地加法前完成 squeeze、expand、dtype 与 contiguity；
- 覆盖合法 rank-4 输入和错误 rank/head/batch/query/key；
- 即使 `S <= topk`，构造 DSA 也可能执行 dense mask。

### RoPE

- 验证 MCore 实际生成器使用的维度；
- MLA 要求 `kv_channels * rotary_percent` 与源 rotary width 一致；
- 同名 interleave flag 不能直接映射，必须追踪 checkpoint 行布局和 runtime 配对；
- 用真实函数比较非零、跨位置的完整 QK score；
- 位置 0 或同位置 dot product 可能因正交不变性产生假通过。

### Dtype

- 分别审计 backbone norm、attention 内 norm、router/indexer norm 和融合替代；
- 复制 launcher flag 时把参数存储 dtype、norm/activation、输出 dtype 和首个
  下游 linear/GMM kernel 作为组合合同；
- 局部 subtype 需要 FP32 时，在最窄模型边界恢复参数，不做全局 dtype 切换。

## 6. 分布式与融合算子

- 所有 PP rank 参与 schedule，只有真实 writer 保存；
- TP vocab gather 后再计算完整词表指标；
- 非均匀 PP 使用前缀和；
- 可选 tensor 在 feature flag 下变为必需时，追踪所有 wrapper、checkpoint/
  recompute helper 和互斥 call site；
- 使用 AST 或轻量 runtime 回归证明每个 call site 转发参数；
- 融合 preflight 使用生产 head count、latent/projected width、optional input 和
  真实输出恢复函数；
- 比较恢复后的值与独立 permutation 得到的 canonical tensor，不能只看 shape。

修复每个重复故障后，将其固化为最窄稳定边界上的可执行回归；故障说明文档不能
替代回归。
