# 精度对齐参考：比较方法与常见差异源

## 比较方法

### hook 采集

- 在目标层注册 forward hook，保存输出张量（fp32 拷贝，避免低精度舍入掩盖差异）
- 比较维度按粒度递进：embedding → 每层 attention/MLP/MoE → 每层输出 → 最终 norm → logits → loss → 梯度 → 参数更新
- 大张量比较可先比统计量（max/mean/std），异常再下钻到逐元素

### 指标

| 指标 | 适用 |
| ------ | ------ |
| max abs diff | 任意层，定位用 |
| 相对误差 `‖a-b‖/‖b‖` | 激活/梯度 |
| 余弦相似度 | logits（全词表）、梯度方向 |
| loss 差 | 整体健康度（单独不作为放行依据） |

### 二分定位

1. 取网络中点层，比较该层输出
2. 差异超阈值 → 差异在 [0, mid]，否则在 [mid, N]
3. 递归收缩至相邻两层，锁定第一个差异节点
4. 输出：节点名、层号、输入快照、diff 数据

## 常见差异源清单

| 类别 | 检查点 |
| ------ | -------- |
| 数据 | tokenizer 版本、padding 方向、position id、mask（causal/padding/多模态） |
| 随机性 | dropout 是否关闭、seed 是否一致、dataloader 顺序 |
| dtype | 中间 fp16/fp32 转换点、norm 是否强制 fp32、累计量 dtype |
| 算子 | attention 后端（NPU FA vs SDPA）、rope 实现、softmax 实现差异 |
| 并行 | SP 切分后 mask/position 对齐、EP 路由一致性、梯度 allreduce 顺序 |
| 权重 | 转换映射遗漏、分片聚合顺序、初始化未加载 |

## 并行精度检查

- FSDP2：满分片下梯度 reduce 顺序与单卡累加顺序不同，允许阈值内差异；差异过大检查 reduce dtype
- Ulysses SP：确认每个 SP rank 的序列段 mask 与 position 正确；attention 输出 allgather 后与单卡全序列对齐比较
- EP：token 路由结果应与单卡完全一致（top-k 决定），不一致先查 router 权重加载
