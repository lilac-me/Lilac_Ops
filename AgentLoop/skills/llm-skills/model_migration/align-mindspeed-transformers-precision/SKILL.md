---
name: align-mindspeed-transformers-precision
description: >-
  This skill should be used when the user asks to build, review, debug, or
  validate full-vocabulary logits precision alignment between a Hugging Face
  Transformers causal language model and its MindSpeed-LLM Megatron Core
  implementation, verify shared inputs and checkpoint parity, enforce valid-token
  cosine thresholds, interpret reports, or diagnose NaN/Inf, low cosine,
  tokenizer, RoPE, mask, dtype, kernel, and TP/PP/CP/EP/ETP layer-mapping issues.
license: MIT
targets-repo:
  - mindspeed-llm
---

# MindSpeed-LLM 与 Transformers 精度对齐

建立可复现的 Transformers 基准、MindSpeed-LLM 结果和证据充分的比较报告。
把精度对齐视为输入、权重、架构、执行、输出和指标的联合合同，而不是一次孤立
的余弦计算。

## 读取按需参考

- 开始任务前阅读[必需输入](references/required-inputs.md)；
- 设计 payload、生产器或比较器时阅读[工作流与数据合同](references/workflow-and-contracts.md)；
- 创建环境、launcher、阶段门禁或恢复策略前阅读
  [运行时与恢复合同](references/runtime-harness-and-resume.md)；
- 运行失败、出现 NaN/Inf 或余弦未达标时阅读
  [故障诊断手册](references/diagnosis-playbook.md)；
- 对外解释 JSON 结果时阅读[报告解读](references/report-interpretation.md)。
- 生成固定输入、导出两侧 logits、运行 CPU 比较或验证内置工具时，阅读
  [内置精度对齐工具](references/precision-alignment-tools.md)。

修改模型独占对齐目录后，运行：

```bash
python skills/llm-skills/model_migration/align-mindspeed-transformers-precision/scripts/audit_alignment_harness.py \
  <alignment-dir>
```

修复全部环境所有权和 shell 安全错误后再执行设备端 forward。该审计不能替代
模型专用的 tensor、RoPE、mask 或融合算子回归。

## 使用默认验收合同

用户未另行指定时采用：

```text
相同 input_ids、attention_mask 和有效 token 位置
相同且未篡改的权重，语义等价的架构
完整 logits 形状 [batch, sequence, HF vocab]
保存为有限 FP32 数值
有效 token 的逐 token 全词表 logits 平均余弦 >= 0.98
```

严格执行：

- 只 tokenize 一次，两个框架复用同一输入；
- 同时保存 `input_ids`、`attention_mask`、SHA-256 和元数据；
- 两份 logits payload 都嵌入共享输入；
- 仅比较 `attention_mask == 1` 的位置；
- 比较完整词表 logits，不用生成 token、概率或 hidden state 替代；
- 只裁剪已知的 MCore 词表 padding，并记录原始/保存词表大小；
- shape/hash 不一致、NaN/Inf、近零向量、缺失权重或随机初始化时直接拒绝；
- 平均 token 余弦是主门禁，Top-1、Top-k 与绝对误差默认仅用于诊断。

用户确认 HF 与 MCore checkpoint 权重相同时，将其记录为“用户提供的输入事实”，
不得描述为工具验证。用户未确认时，把 checkpoint identity 标记为未解决风险。

## 执行有序流程

```text
1. 固定代码、模型、checkpoint、指标和并行布局
2. 固定环境所有权、恢复与失效规则
3. 审计 HF 配置、代码、tokenizer、权重和加载结果
4. 审计 MCore args、Spec、转换器和全局/局部层身份
5. 建立架构、执行和直接 callee 的 tensor 合同
6. 生成唯一固定输入并计算 hash
7. 保存有限的 Transformers 全词表 logits
8. 保存有限的 MindSpeed-LLM 全词表 logits
9. 在 CPU 上比较并写出 JSON 报告
10. 从最早违反的不变量开始诊断
11. 只重跑被修改失效的产物
12. 将模型专用代码隔离到独占目录
13. 创建或更新中文操作指南
```

### 1. 先审计再编码

检查 HF 配置嵌套字段、tokenizer、权重索引、真实 forward、加载缺失/多余键，
以及 MCore Spec、运行组件、launcher、转换映射和 checkpoint 生成命令。

对于 MoE/dense、MLA/DSA、混合注意力/SSM、noop、非均匀 PP 和 MTP，
建立以下表格：

```text
全局层 | HF 模块/键 | MCore 模块/键 | PP rank/局部层 | 是否匹配
```

### 2. 从数值基准开始

第一阶段追求语义可比性，不追求吞吐：

- 两侧使用相同计算 dtype，通常为 BF16，保存 FP32 logits；
- HF 使用 eager attention（若支持）；
- MCore 使用可加载 checkpoint 的最直接非吸收、非稀疏路径；
- 关闭 dropout、cache、生成、采样、MTP logits、SP、CP 和 VPP，
  除非架构本身要求；
- DP=1，并保留加载 checkpoint 必需的 TP/PP/EP/ETP；
- 诊断异步失败时使用同步执行。

直接基准通过后，每次只启用一个优化，并把执行配置写入元数据。稀疏语义覆盖
与具体融合 kernel 覆盖应拆开验证。

### 3. 审计直接消费者边界

不要停留在公开 forward 签名。对每个转换 tensor 追踪到首个真实消费者，记录：

- rank、各轴语义、singleton/broadcast 轴；
- dtype 和布尔/加性 mask 的值语义；
- device、layout、contiguity 和 batch 扩展；
- 是否执行原地运算、scatter/gather 或融合算子。

在最窄的模型适配边界规范化数据。若一个外层 tensor 分叉给不同表示的消费者，
分别在各消费者入口转换，不做全局改写。

DSA 常见边界必须覆盖：

```text
外层 [B|1,1,Sq,Sk] bool（True=masked）
 -> 内层 [B,Sq,Sk] additive（0/-inf）
```

生产合法的 rank-4 mask 必须被显式 squeeze、batch-expand 并转换后接受；应拒绝
错误 rank、非 singleton head 轴和不兼容形状。回归必须实际执行真实原地加法，
不能只做源码字符串检查。

融合算子回归应从生产路径真实 tensor 出发，覆盖投影/吸收后的最终维度、原始
输出布局、生产恢复函数及下一消费者。不要把 launcher 的 `shape_order` 错当成
融合算子的固定输出布局。

### 4. 使用两种序列覆盖

- 短序列：低成本验证主 forward 与权重映射；
- 特性序列：DSA 使用 `S > index_topk`，长上下文跨越 RoPE 缩放边界，
  padding 使用多样本右 padding，cache/生成和 MTP 独立验收。

`S <= index_topk` 只能证明 DSA 模块构造和 dense mask 路径，不能证明稀疏选择。

### 5. 拆分生产器与比较器

默认使用本 skill 内置的通用工具，使 skill 不依赖外部 MindSpeed-LLM
`tests/` 目录：

```text
scripts/precision_alignment/make_fixed_input.py
scripts/precision_alignment/dump_transformers_logits.py
scripts/precision_alignment/dump_logits.py
scripts/precision_alignment/compare_logits.py
scripts/precision_alignment/alignment_utils.py
```

保持固定输入、HF dump、MCore dump 和 CPU 比较为独立入口。模型差异优先通过
provider、forward、output selector 和 layout hook 适配，不要复制通用 dumper。

执行 MindSpeed 侧导出器时，仍须使用目标 MindSpeed-LLM 仓库已安装
`mindspeed_llm`、`megatron`、`torch` 和 `torch_npu` 的设备环境，并从仓库根
运行。skill 内置脚本消除的是“脚本文件必须存在于外部 `tests/` 目录”的依赖，
不是框架运行时依赖。若目标仓库也存在同名工具，不得在同一次工作流中混用两套
文件；先比较版本，明确选择一套并把实际脚本路径记录到产物元数据。

复用旧产物前必须验证输入/hash、有限 FP32 shape、checkpoint identity、
producer/tool 合同版本、执行标志和模型要求的同步屏障；文件存在不代表可复用。

### 6. 隔离 Transformers 依赖

当本地 Transformers 依赖与 MindSpeed 设备环境冲突时：

- 用模型独占临时 venv 执行 tokenize 和 HF dump；
- 保持原 MindSpeed 环境执行 preflight、MCore dump 和 CPU 比较；
- 不升级、卸载或修复用户的 MindSpeed、torch 或 torch_npu 环境；
- 环境安装必须是显式 setup 动作，不能成为 dump 的隐式副作用；
- 使用绝对 HF venv Python 和命令级 `PYTHONPATH`，不得全局激活；
- MCore launcher 继承当前设备环境，不选择或 source CANN；
- launcher 必须拒绝被 source，并保留继承的 `PYTHONPATH`；
- 校验 `transformers.__file__` 确实来自指定 checkout，并保存运行时 JSON 报告。

大 logits 不要在设备上整体执行 `logits.float().cpu()`；应按 token 分块以源 dtype
复制到 CPU，再在 CPU 转 FP32 并写入预分配 `[B,S,V]` tensor。

### 7. 验证分布式输出所有权

所有 PP rank 都参与调度。仅真实 pipeline-last writer（通常最后 PP、TP0、DP0、
CP0）保存 logits。保存前 gather TP 词表切片，并显式规范化到 `[B,S,V]`。

非均匀 PP 使用 `num-layer-list` 前缀和计算全局层偏移。不得关闭严格加载掩盖
局部/全局层映射错误。

## 支持受控诊断绕过

阶段门禁默认严格。需要提前诊断后续阶段时：

- 仅允许一个模型独占环境变量指定目标阶段的直接前驱；
- 只在值精确匹配直接前驱时绕过前驱报告检查；
- 输出 `gate_bypassed=true`、`diagnostic_only=true` 和
  `sequential_precision_certified=false`；
- 不绕过目标阶段自身 flag、算子、输入、有限性和比较；
- 不创建伪造的 passing JSON，也不把诊断结果表述为链路认证；
- 完成诊断后取消变量，从最早未认证阶段严格重跑；
- 增加严格拒绝、精确绕过、未知值拒绝和正常认证回归。

若用户明确跳过整个阶段，还应原子记录：

```text
state=skipped
skipped_by_user=true
numeric_comparison_executed=false
passed=false
diagnostic_only=true
sequential_precision_certified=false
```

跳过不等于禁用特性。下游累计阶段可执行该特性，但只能得到诊断结果。

## 从最早破坏的不变量诊断

按以下顺序：

1. 进程/环境；
2. checkpoint 严格加载；
3. 输入、hash、shape、layout、词表；
4. NaN/Inf 和近零向量；
5. 首 token 与逐 token 相似度模式；
6. 架构和执行路径；
7. 优化 kernel 与长序列特性。

平均余弦接近零通常是结构问题；首 token 很低优先检查 checkpoint、embedding、
LM head、norm 和基础架构；早期高而后续衰减优先检查 mask、position ids、RoPE、
cache 或累积 kernel 差异。不得调低阈值、修改 logits 或替换非有限值制造通过。

## 隔离模型专用实现

skill 内置通用工具目录保持模型无关：

```text
scripts/precision_alignment/
```

模型专用适配器、合同、preflight、launcher、阶段元数据、回归和中文操作指南放在：

```text
examples/mcore/<model-or-variant>/precision_alignment/
```

只有通用工具确实缺少跨模型能力，且用户授权共享范围修改时，才扩展通用目录。
生产代码缺陷应在真实缺陷位置修复并添加回归，不要藏进对齐目录。

## 输出要求

交付或标明：

- 固定输入与 digest；
- HF 和 MCore logits payload 及日志；
- CPU 比较命令和 JSON 报告；
- 模型、checkpoint、运行时和并行元数据；
- 模型独占中文操作指南；
- 修改文件、已执行验证和未解决风险；
- 主门禁与诊断指标分开的结果摘要；
- 无 NPU/权重时用户可执行的最小下一条命令。

只有请求范围达到阈值时才能报告对齐完成；否则明确区分“实现完成”和“数值验证
待执行”。
