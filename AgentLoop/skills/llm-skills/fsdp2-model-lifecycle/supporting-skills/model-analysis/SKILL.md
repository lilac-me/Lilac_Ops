---
name: llm-model-analysis
description: >-
  This skill should be used when the user asks to analyze one or more
  large-language-model source implementations for architecture, tensor-shape
  flows, parameter counts, NPU operator support, FLOPs, or MFU, including
  Dense, MoE, MLA, sparse or linear attention, SSM, hybrid-attention, MTP,
  checkpoint-storage, training, and inference paths.
---

# LLM 模型分析

## 功能概述

从用户指定的模型源码和不可变版本出发，生成可复核、可重现的结构分析、
Shape 流程、NPU 算子支持度以及 FLOPs/MFU 报告。支持 Dense、MoE、MLA、
稀疏或线性 Attention、SSM、混合 Attention、MTP 和其他新型结构。

不要只复述配置或论文。以实际执行源码为主证据，区分训练与推理、逻辑参数与
物理 Checkpoint、算法有效计算与源码实际执行量，并明确所有未运行和未确认项。

## 使用方式

- 在 MindSpeed-Agent 中显式调用 `/mindspeed-agent:llm-model-analysis`；
- 或在用户提供模型仓库、配置、建模源码、自定义 Kernel、权重索引、参数量、
  Shape、NPU 支持度、FLOPs 或 MFU 分析需求时自动匹配。

完整分析默认交付：

```text
model-analysis/<model-name>/
├── model_analysis.md
└── model_analysis.json
```

除非用户明确要求其他格式，每个模型只交付上述两个文件。

## 读取参考规范

开始分析前，根据任务范围读取以下文件。完整模型分析必须全部读取；局部问题只读取
`evidence-and-workflow.md` 和对应专项文件，生成正式报告时再读取
`output-contract.md` 与 `acceptance-checklist.md`。

| 参考文件 | 读取时机 | 内容 |
|---|---|---|
| [证据与工作流](references/evidence-and-workflow.md) | 每次任务 | 输入、证据边界、状态标签、失败处理和执行顺序 |
| [结构与参数](references/structure-and-parameters.md) | 分析结构、创新或参数量 | 层类型、参数三账本、Active 多口径 |
| [Shape 分析](references/shape-analysis.md) | 梳理前向和并行 Shape | Tensor/运算格式、流程图、数值示例和校验 |
| [NPU 支持度](references/npu-support.md) | 评估昇腾适配 | 官方证据、非 Eager 扫描、支持矩阵和风险分级 |
| [FLOPs 与 MFU](references/flops-and-mfu.md) | 推导计算量或 MFU | 模块公式、执行/有效口径和交叉验证 |
| [输出契约](references/output-contract.md) | 生成正式交付物 | 固定五章目录和 `model_analysis.json` Schema |
| [验收清单](references/acceptance-checklist.md) | 交付前 | 人工检查、机器断言和质量门禁 |

## 操作步骤

1. 记录 Repository、完整 Commit Hash 或不可变 Tag、分析日期、主文件、配置变体、
   Checkpoint Manifest 和可用验证环境。
2. 读取配置、模型入口、真实 `forward()`、Kernel、训练/推理入口和权重索引；配置、
   论文、Checkpoint 与源码不一致时输出差异，不静默选取。
3. 展开真实主层索引并分离 MTP、Speculative Decoder、Vision Tower、Reward Head 等
   附件；建立模块调用关系和训练/推理边界。
4. 建立逻辑参数、Active 参数和物理 Checkpoint 三套账本，逐模块、逐层型、整网复算；Markdown 同时展示精确参数量和十进制 B 换算值。
5. 按整网、层类型、复杂模块和关键算子四级推导 Shape；验证元素守恒、缩并维、广播、
   可整除性、Cache、Mask、TopK 和并行本地 Shape。
6. 扫描所有非 Eager、自定义 Kernel、dtype/layout 假设和分布式调用；分别评价原始源码、
   Eager Fallback 与外部等价实现。
7. 从实际 Tensor Shape 推导模块和整网 FLOPs，区分源码执行、算法有效、训练等价和
   参考推理口径；结合真实端到端输入计算 MFU。
8. 按固定五章 Markdown 与固定 JSON Schema 输出结果，执行人工验收和机器断言，失败后
   修正再交付。

## 不可违反的质量门禁

- 不得只根据模型名称或 `config.json` 推测实际结构；
- 不得把推理实现、外部框架验证或静态文档支持表述为当前训练源码已验证；
- 不得把 Total Parameter、Active Parameter 和 Checkpoint 字节数混为一谈；
- 不得遗漏主路径自定义 Kernel、Backward、dtype、layout、动态 Shape 和分布式约束；
- 不得用固定参数量近似代替 MoE、稀疏 Attention、Indexer、SSM 等逐算子 FLOPs 推导；
- 没有真实设备、权重、Profiler 或训练环境时必须写明 `not_run`，不得伪造运行结论；
- Markdown 与 JSON 的层数、参数、NPU 整网状态、FLOPs 和 MFU 示例必须精确一致；
- 公开源码引用优先使用 Commit 固定链接，外部证据必须标明版本、设备与独立实现范围。

## 注意事项

- 需要联网查询 NPU 支持度时，优先使用昇腾官方 CANN 与 PyTorch 适配文档；技术问题只
  采用官方文档、源码仓库或论文等一手来源。
- 需要运行模型时，先检查依赖、权重、显存/内存、设备和分布式条件；无法安全运行时，
  完成静态验证并列出继续验证所需输入。
- 用户要求“从零重做”时，不读取旧报告的数值和结论，只复用固定输出 Schema。
- 单模型与批量分析使用同一口径；不同模型、不同 Commit 的配置和源码不得混用。
