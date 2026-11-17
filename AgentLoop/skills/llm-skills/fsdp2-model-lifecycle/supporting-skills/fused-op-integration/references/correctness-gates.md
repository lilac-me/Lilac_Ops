# 融合候选正确性门禁

本文件用于阶段二接入边界审计和阶段五融合候选验收。它把容易被自然语言跳过的规则固化为必需产物；主控、模型接入和融合接入 Skill 在对应阶段必须读取。

## 1. 阶段二纯 eager 边界

阶段二只允许模型结构、FSDP2/并行兼容、EP eager 合同、注册、FLOPS 和启动配置。对 `input_commit..working_tree` 审计新增行，并输出 `adaptation_boundary_audit`：

- 搜索 NPU fused kernel、Grouped GEMM/GMM、融合开关、模型专属 fused helper、patch 注册，以及仅为 fused 参数布局服务的 conversion mapping。
- 每个命中记录文件、行、分类和处置。新加入的可执行性能路径为 `failed`；未修改的上游声明或不可避免的通用接口须给出源码证据才能放行。
- `ep_forward` 可以在阶段二存在，但必须走 eager 专家计算。不能把接口合同解释为提前实现 fused dispatcher/GMM。
- wrapper + `MindSpeedPatchesManager` 不是新模型接入主路径。

## 2. 结构签名

阶段二生成 eager 结构签名；每个融合候选在开关关闭和开启时分别生成并逐字段比较。签名至少包含：

- config 中影响计算语义的字段和实际层数。
- 每个 module path 对应的 class。
- 参数名、shape、dtype、requires_grad、tied-weight group。
- state_dict key 与 checkpoint conversion 目标。
- MoE 的 expert 轴、gate/up 拼接顺序、down 权重布局、expert 数和 top-k。

融合候选不得通过替换 module class、改变参数 shape/state_dict key、展平或重排专家参数来获得性能。确有架构性必要时，先返回模型接入阶段建立并重新验收新 eager 基线；该变化不能作为融合单变量与旧基线比较。

## 3. Forward 语义不变量矩阵

为每项记录“原生源码步骤 → eager 实现位置 → fused 实现位置 → probe 证据”。至少覆盖：

- Attention：`q_proj/k_proj/v_proj`、reshape、`q_norm/k_norm`、RoPE、KV repeat/GQA 责任、mask、attention、transpose/reshape/contiguous 和输出 projection。Qwen3-MoE 的 Q/K norm 不得省略或移动到 RoPE 之后。
- RMSNorm：epsilon、归一化维、目标模型原生的输入/weight 计算 dtype 和输出 dtype；Qwen3-MoE 需检查 FP32 计算后回输入 dtype。
- RoPE：rotate-half/interleaved 约定、cos/sin 广播维、position id、缓存、输入输出 dtype/layout。
- MoE/GMM：router logits/top-k、token permute、`tokens_per_expert`、三维专家权重顺序、gate/up 拼接、激活、down projection、路由权重、unpermute、空 expert。
- Flash Attention：默认门禁为 NPU、纯 causal、无 padding、无 sliding-window和支持的 dtype/layout/sequence 条件；任一条件不满足必须真实执行 eager fallback。只有目标 kernel 对额外语义已有实现和专项 probe 证据时才能扩大支持范围。核对 mask 极性和 shape、`sparse_mode`、GQA/MQA 的 KV repeat 责任以及输出布局。

## 4. 必需 probe

在远端长训练前执行确定性小输入 probe：

1. 同一权重、输入和随机种子比较 eager/fused 输出。
2. 比较 input gradient 和所有受影响参数 gradient，确认 finite、非空且在约定容差内。
3. 覆盖边界 shape：至少一个空 expert、多个 expert token 数、GQA（如适用）、纯 causal，以及每个应触发 eager fallback 的 Flash 条件。
4. 记录每个融合开关的 requested 与 active；active 只能来自运行时分支证据或计数器，不能从 YAML 推断。
5. 再执行真实 FSDP2 训练的 forward、backward、optimizer、逐步 loss、性能和显存门禁。
6. 使用真实权重执行 1/2 层完整词表 logits 对齐。随机初始化训练 loss 不能替代该门禁。

## 5. 证据范围

- 完整架构训练：使用原始层数，证明训练链路和该形态下的 loss/性能/显存；权重不完整时可以明确声明随机初始化。
- 减层真实权重：默认 1 层 smoke、2 层验收，证明被覆盖层的 checkpoint 数值语义，不声称完整层数权重已验收。
- reduced-layer 训练：只证明相同减层配置的局部 A/B；不得外推完整模型收益。

## 6. 原子组合

用户明确授权不可拆分组合时，可以不做逐算子训练和性能比较，但不能跳过每个成员的结构与语义 probe。组合训练、loss、性能、显存和真实权重 logits 只对融合全关基线做整体 A/B，不归因单算子。任一成员未 active 或任一门禁失败，整个组合 rejected；不自动二分或选择性接受。
