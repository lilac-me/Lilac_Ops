# 精度对齐必需输入

每次任务开始时使用此清单。优先从文件推导，只询问仍未知且会阻塞对应阶段的值。

## 必需输入

| 类别 | 必需信息 | 用途 |
| --- | --- | --- |
| 任务与门禁 | 模型/变体、对齐范围、主指标与阈值，是否包含 cache、生成、MTP、长上下文或优化 kernel | 定义“对齐”的含义；默认门禁为有效 token 全词表 logits 平均余弦 `>=0.98` |
| 代码身份 | MindSpeed-LLM 与 Transformers 路径、分支/提交/版本及相关未提交改动 | 同名模型的不同实现可能不等价 |
| HF 资产 | 配置、tokenizer、全部权重 shard/index、自定义建模代码 | 构建参考并审计真实权重键 |
| MCore 实现 | Model Spec、示例 launcher，或实现它们的授权范围 | 确定目标架构和代码边界 |
| 权重相等性 | 用户明确确认所选 HF 与 MCore checkpoint 包含相同权重 | 这是输入事实，不是工具验证；未确认时必须记录风险 |
| 输入定义 | 固定输入，或 prompt/token ids、最大长度、padding/batch 要求 | 保证两侧使用完全相同的 token 和 mask |
| 并行布局 | TP/PP/CP/EP/ETP/DP、`num-layer-list`、VPP、MTP、SP、noop 层 | 匹配 checkpoint 切分和层身份 |
| 运行时 | NPU/GPU/CPU、设备/节点、PyTorch、torch_npu/CANN、MindSpeed/Megatron 版本和执行方式 | 决定可用 kernel 及验证是静态还是可执行 |
| MCore 环境所有权 | 已知可用设备 shell、`sys.executable`、必需 import、launcher 是否只能继承环境 | 防止 launcher 选择 toolkit 或掩盖缺失安装 |
| 依赖隔离 | Transformers 是否与 MindSpeed 依赖冲突、精确约束、临时 venv、在线索引或 wheelhouse | 防止污染设备环境 |
| 路径与权限 | 模型、tokenizer、checkpoint、输入、输出、日志及本地/远端映射 | 防止晚期失败或读取陈旧产物 |
| 当前证据 | 继续旧任务时提供完整命令、未截断日志和报告 | 保留真实失败上下文 |

## 缺失信息模板

```text
模型/变体与请求范围：
主指标/阈值：

MindSpeed-LLM 仓库与提交：
Transformers 路径与提交/版本：
HF 模型/tokenizer：
MCore checkpoint：
HF→MCore 转换命令/源 checkpoint：
Model Spec/示例 launcher：

固定输入，或 prompt/token ids/最大长度：
TP/PP/CP/EP/ETP/DP：
num-layer-list/VPP/MTP/SP/noop-layers：

硬件/节点：
PyTorch/torch_npu/CANN/MindSpeed/Megatron：
已知可用设备环境与 sys.executable：
必须通过的 import：
输出/日志：
当前命令与完整错误：
允许修改范围和不可用操作：
```

## 可选但有价值的信息

- 已知可用 logits、历史通过报告、层级 activation 或转换验证；
- QKV 打包、GQA/MQA、norm/residual、logits scaling、RoPE、mask、
  MoE router、MLA/DSA 和 MTP 所有权说明；
- state-dict 键清单、权重索引、抽样 hash 或 tensor 摘要；
- 多语言、代码、数学、padding、长上下文等代表性输入；
- 显存、卡数、时间、磁盘和离线限制；
- 必须验证的 FlashAttention、稀疏、吸收、融合或量化路径；
- 容器/环境锁定、commit hash 和 checkpoint checksum；
- 是否允许 hook、层级 dump、同步 NPU 模式和小型诊断输入。

## 决策规则

- 文件能直接证明的值不再询问用户。
- 从活动环境解析基础 Python；拒绝文档占位路径、HF venv Python 和无法导入
  MindSpeed/Megatron 的解释器。
- venv 所有权检查同时记录词法 launcher 路径、resolved binary、
  `sys.prefix` 和 `sys.base_prefix`。
- fresh device shell 必需 import 失败时，报告环境前置条件；launcher 不自动
  source CANN。
- 区分本地分析路径和设备执行路径。
- 改变范围、权重相等性结论、验收阈值或远端执行权限前必须获得用户确认。
- 无 NPU 时完成实现和静态验证，并给出精确远端命令与预期产物。
- checkpoint 来源不明时，不能把低余弦直接归因于模型实现。
