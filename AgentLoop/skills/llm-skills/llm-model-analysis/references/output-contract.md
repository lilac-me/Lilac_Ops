# 输出契约

## 目录

- 批量分析工作流
- model_analysis.json 固定输出契约
- 统一最终报告结构
- 五章最低内容密度

## 批量分析工作流

单模型与多模型使用完全一致的输出目录规范，仅通过模型二级目录区分不同模型。每个模型只输出一个完整 Markdown 分析报告和一个结构化数据文件，不再拆分多个 Markdown、CSV、JSON 或 PDF 文件。

```text
model-analysis/
├── model_a/
│   ├── model_analysis.md
│   └── model_analysis.json
├── model_b/
│   ├── model_analysis.md
│   └── model_analysis.json
└── model_c/
    ├── model_analysis.md
    └── model_analysis.json
```

单模型分析时仅生成一个模型目录，例如 `model_a/`；多模型分析时并列生成多个模型目录。各模型目录中的文件名称、格式和报告结构保持一致。参数量、NPU 算子支持矩阵、FLOPs/MFU 公式等结构化结果统一写入 `model_analysis.json`。

### `model_analysis.json` 固定输出契约

`model_analysis.json` 必须使用以下顶层字段、顺序和数据类型；模型不存在的复杂模块可在对应对象内省略，不得删除顶层字段：

```json
{
  "schema_version": "1.0",
  "model": {},
  "configuration": {},
  "layer_types": [],
  "parameters": {},
  "shape_flow": {},
  "custom_kernels": [],
  "npu_support": {},
  "flops": {},
  "mfu": {},
  "training_vs_inference": {},
  "unknowns": [],
  "sources": []
}
```

必需子结构：

| 顶层字段 | 必需内容 |
| --- | --- |
| `model` | `name`、`repository`、`revision`、`analysis_date`、`status`、`primary_files` |
| `configuration` | 影响结构、Shape、参数、FLOPs 和量化的扁平化实际值 |
| `layer_types` | 每类的 `type`、`indices`、`count`、`attention`、`ffn`、`routing/other` |
| `parameters` | `counting_basis`、`module_counts`、`block_counts`、`model_counts`、`validation` |
| `shape_flow` | `symbols`、`network` 及源码实际存在的 Attention/MoE/Indexer/MTP/Cache 等对象 |
| `custom_kernels` | `name`、`implementation`、`purpose`、可选 `source` 与关键 Shape/Layout 约束 |
| `npu_support` | `overall_as_is`、`overall_conclusion`、`matrix`、`required_changes` |
| `flops` | `basis`、`formulas`、`per_token_per_layer`、`examples_batch_1`、`cross_checks` |
| `mfu` | `formula`、`model_tflops_per_device_formula`、`required_inputs`、`hypothetical_example` |
| `training_vs_inference` | 训练能力、Loss、Decorator、训练路径与推理专属特性 |
| `unknowns` | 信息缺口字符串数组，不得省略 |
| `sources` | URL/本地来源数组；公开来源优先使用 Commit 固定链接 |

JSON 一致性规则：

- 参数量、FLOPs、字节数、层数和索引使用 JSON Number，不使用带 `B/GFLOPs` 后缀的字符串；单位写入字段名或独立口径字段；
- JSON 中的参数量继续保留精确 Number 作为权威值；Markdown 参数表额外展示的 B 列必须由该精确值除以 `1_000_000_000` 生成，不改变 JSON Schema；
- 公式使用字符串，数值示例使用 Number；
- `layer_types[].count` 必须等于 `indices` 长度，主层 Count 之和必须等于配置层数；
- Markdown 摘要、参数表、NPU 整网状态、FLOPs 示例和 MFU 示例必须与 JSON 精确一致；
- 未运行的校验写明确状态，例如 `not_run_weights_not_downloaded`，不得留空或假设通过；
- 外部 NPU 路径在 Matrix 中单列，不得覆盖 `overall_as_is`；
- 对同一模型重复分析时保持 Schema 和字段含义稳定，便于自动 Diff。

---

## 统一最终报告结构

最终 Markdown 报告必须固定使用以下五个二级标题。原有分析内容只能作为三级或更低级标题归入对应章节，不得再将配置、参数量、Cache、NPU 矩阵、MFU 等并列为十几个二级标题。

```markdown
# 模型分析报告

## 1. 摘要

### 1.1 执行摘要
### 1.2 源码与版本信息
### 1.3 分析范围、关键结论与未确认项

## 2. 结构分析

### 2.1 模型配置参数
### 2.2 整体模型结构与层类型统计
### 2.3 核心创新
### 2.4 参数量与 Active 参数量
### 2.5 训练路径与推理路径差异

## 3. Shape 流程梳理

### 3.1 Shape 符号表
### 3.2 整网前向 Shape 总览
### 3.3 各类 Decoder Layer Shape
### 3.4 Attention、Indexer、MoE 等复杂模块 Shape
### 3.5 Cache、Mask 和关键前向分支
### 3.6 复杂流程数值示例
### 3.7 Shape 静态或运行时校验

## 4. NPU 算子支持度分析

### 4.1 标准 PyTorch Eager 路径
### 4.2 非 Eager 和自定义算子清单
### 4.3 NPU 支持矩阵
### 4.4 NPU 风险分级
### 4.5 NPU 适配建议与验证用例

## 5. FLOPs 和 MFU 公式推导

### 5.1 FLOPs 统计口径
### 5.2 模块级 FLOPs
### 5.3 单层与整网前向 FLOPs
### 5.4 整网训练 FLOPs
### 5.5 MFU 公式与代入示例
### 5.6 FLOPs 交叉校验、假设与未确认项
```

目录约束：

1. 二级标题必须且只能是上述五章，名称和顺序保持一致；
2. 原 18 节要求全部保留，只调整到对应的三级或更低级标题下，不得因目录归并而删减内容；
3. “假设、未确认项与后续验证”按内容归属放入 `1.3`、`3.7`、`4.5` 或 `5.6`，避免在报告末尾形成与五大章并列的独立章节；
4. 表格、公式、流程图、状态标签和验收口径保持原规范；
5. `model_analysis.json` 的字段与结构化内容要求保持不变，不受 Markdown 目录调整影响。

### 五章最低内容密度

| 章节 | 摘要必须可见的结论 | 正文必须包含的复核材料 |
| --- | --- | --- |
| 1. 摘要 | Revision、主层排布、主模型 Total/Active 精确值与 B 值、附件参数精确值与 B 值、NPU As-Is、代表性 FLOPs、验证状态 | 源码清单、证据等级、未确认项表 |
| 2. 结构分析 | 主模型与附件边界、Active 多口径 | 完整配置、展开层索引、模块/层型/整网三层参数表，每个参数值同时展示精确值和 B 换算值、训练/推理差异 |
| 3. Shape | 整网主干和最复杂模块 | Tensor 名、权重 Shape、缩并/广播、Cache/Mask、真实小例子、静态/运行时校验表 |
| 4. NPU | 原始整网状态与推荐替代路径 | Eager 表、全部非 Eager Kernel、Forward/Backward/dtype/Shape Matrix、P0-P4、适配测试 |
| 5. FLOPs/MFU | 代表长度的 Forward/Training 数值与 MFU 输入缺口 | 模块公式、整网组合、执行/有效双口径、纯代入或真实 MFU |

摘要应先给结论再给限制，不能只写模型介绍。正文中的每个精确数字必须能追溯到源码 Shape、配置、计算公式或 Checkpoint Manifest。

---
