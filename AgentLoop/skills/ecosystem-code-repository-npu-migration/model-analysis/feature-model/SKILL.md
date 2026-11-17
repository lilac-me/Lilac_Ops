---
name: feature-model
description: This skill should be used when the user asks to build the required repository-wide functional and executable-entry map before an NPU migration, enumerate in-scope tests/examples/scripts/documented commands, select all feasible migration-related unit checks, and deduplicate only genuinely equivalent examples. Use for whole-repository migration planning or any direct request to understand and reduce a repository validation surface.
version: 0.16.0
domain: ecosystem-code-repository-npu-migration
function: model-analysis
tags:
  - ascend
  - npu
  - migration
  - pytorch
author: addsubmuldiv
---

# NPU 功能模型

## 目的

在进行昂贵的 NPU 工作前，从整体上阅读仓库并回答两个问题：

1. 这个仓库实际提供哪些功能和组件能力？
2. 哪些测试或示例执行相同的 NPU 路径，可以用一个代表项覆盖同一分组？

这个 skill 创建语义地图和验证计划。它不运行目标、不修复代码，也不判断 NPU 支持是否通过。

## 使用场景

每次仓库级迁移进行源码适配前都要使用它。当需要在不运行目标的情况下理解仓库功能、测试、示例或代表性验证范围时，也可以直接使用。

## 工作流程

1. 阅读请求范围、`README*`、安装和使用文档、软件包元数据、源码布局、配置、范围内所有 `tests/`、`examples/` 和 `scripts/` 目录、文档中的命令以及 CI 启动命令。将发现的文件和命令与最终 entry 列表比对，避免静默遗漏。跟踪 wrapper 到真实 Python entrypoint，不要仅根据文件名分类。
2. 先用用户可见功能描述仓库，例如 SFT、DPO、GRPO、推理、评测、导出或 checkpoint save/load。不要先套用通用分类法再强行归类仓库。
3. 记录可能改变 NPU 执行路径的维度：更新策略、runtime/launcher、模型架构或模态、设备拓扑、分布式/并行模式、精度、packing/varlen、checkpoint 行为、自定义 kernel 以及 vLLM 等外部 runtime。
4. 将范围内每个 unit/component test、端到端示例、可执行脚本或文档启动命令各列一次到 `.npu-agent/feature_model.json`。每个可执行 entry 都要包含仓库原生 `command`。使用仓库相对路径作为 `entry_id`；同一文件暴露不同命令或配置时，使用 `<path>#<command-or-config-name>`。不要使用 `example-0001` 等序列 ID。
5. 将每个 entry 映射到用户可见功能和组件能力。只有功能目标及所有相关 NPU 敏感维度都等价时，才创建 equivalence group。
6. 编写 `.npu-agent/coverage_plan.json`。将所有可行的迁移相关 unit/component test 放入 `capability_checks`；不要为了减少执行而去重不同测试。`representative_groups` 只放示例、脚本或文档启动命令，选择仍能执行真实路径的最小仓库原生目标，并将所有未选成员保留为 deferred。对无法执行的每个 entry 记录明确的 `OUT_OF_SCOPE`、`MANUAL` 或 `BLOCKED` 决策。
7. 运行 `scripts/validate_feature_model_artifacts.py --repo <repo> --json`。修复结构或引用错误，然后自行审查语义分组；校验器成功不是运行时证据。

## 简短示例

| 入口 | 分类 | 规划决策 |
| --- | --- | --- |
| `tests/test_npu_device.py` | 设备选择组件能力 | 作为低成本能力检查 |
| `examples/sft_lora_tiny.py` | LoRA SFT、仓库 runtime、单设备 | 作为 LoRA SFT 分组的代表项 |
| `examples/sft_lora_large.py` | 相同功能和 NPU 敏感路径、更大资产 | 延后到 `sft_lora_tiny.py` |
| `examples/sft_full.py` | 全参数 SFT | 因更新策略不同而单独分组 |
| `examples/grpo_vllm.py` | 带 rollout 和 vLLM 的 GRPO | 因 runtime 和数据流不同而单独分组 |

不要仅因为两个名称都包含 `SFT` 就合并 entry。除非源码检查证明 NPU 敏感路径等价，否则 full 与 LoRA、dense 与 MoE、单设备与分布式、Hugging Face 与 Megatron、普通序列与 packing/varlen 都必须分开。

## 输出

- 第一次编写任一文件前，阅读从仓库检查到代表项选择的 [操作示例](references/worked_example.md)。
- `feature_model.json` 说明仓库包含什么：summary、capabilities、features、entries、equivalence groups 以及 feature-to-capability prerequisites。遵循 [字段 schema](schemas/feature_model.schema.json) 和 [完整示例](examples/feature_model.json)。
- `coverage_plan.json` 说明后续验证应分两阶段运行：先运行所有可行的迁移相关 unit/component check，再运行带 deferred 等价成员和明确范围决策的示例代表项。遵循 [字段 schema](schemas/coverage_plan.schema.json) 和 [完整示例](examples/coverage_plan.json)。

两个示例 JSON 文件描述同一个虚构仓库，必须作为配对文件阅读。复制它们的结构和跨文件 ID 关系，不要复制虚构仓库事实。

## 边界

- 不要执行仓库目标或修改迁移源码。
- 不要仅根据名称、grep 输出或目录位置推断功能。
- 不要创建 queue、state、cursor、attempt 或 controller 文件。
- 不要将执行状态、`PASS` 或最终支持结论写入任一产物。
- 将代表项选择视为规划判断；后续支持结论由真实命令和日志决定。

## 完成条件

返回两个经过校验的文件、简洁的仓库功能摘要、每个分组的等价性理由、选定的代表项，以及可见的 deferred 或 scope-out entry。

## 资源

- `references/modeling_guide.md`：分类能力、NPU 敏感维度、equivalence group 或代表项前阅读。
- `scripts/validate_feature_model_artifacts.py`：只读结构和交叉引用校验器。
