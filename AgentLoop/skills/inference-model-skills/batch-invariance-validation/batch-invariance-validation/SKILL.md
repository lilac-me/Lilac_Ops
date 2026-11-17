---
name: batch-invariance-validation
description: This skill should be used when the user asks to orchestrate an end-to-end vLLM Ascend batch invariance validation, resume a staged run, or aggregate evidence into a final acceptance result.
version: 1.1.0
domain: inference-model
function: batch-invariance-validation
tags: [vllm-ascend, batch-invariance, orchestration]
author: mindspeed-agent
targets-repo:
  - vllm-ascend
---

# 批次不变性验证编排

通过插件状态机依次执行 TP 计划门禁、环境准入、阶段一算子分析、阶段二功能测试、阶段三数据集验收和最终汇总。状态恢复必须依据已校验的运行配置、转移日志和阶段产物，不得依据对话跳过前置阶段。

## TP 计划门禁

参数收集时可在目标运行环境执行 `python ../batch-invariance-environment/scripts/check_batch_invariant_env.py --detect-chip`，将返回的 A2/A3/A5 写入候选配置的 `chip`；保留原始型号作为后续环境报告证据。该模式只读取型号，不执行环境准入、不检查或安装包。随后由自动规划器根据可见设备和模型结构生成并写入 TP 计划。型号无法识别时先进入自主恢复，不猜测为 A3。

根据用户给定的可见设备和模型结构自动生成计划。候选 TP 必须是可见卡数量的因子，并满足模型 attention/KV head 整除约束；再用模型加载探测排除资源不可用的 TP。各阶段优先使用最大合法 TP，矩阵覆盖合法的 1/2/4，诊断使用失败 TP 以下最近的合法 TP。

自动计划将候选、兼容、排除依据、阶段计划和诊断 TP 原子写入 `run_config.json`。正常情况下不要求逐阶段确认；只有需要额外设备或改变用户明确限定的资源范围时才暂停。TP 变化必须隔离新旧证据。

## 调度与证据

依次调用 `batch-invariance-environment`、`batch-invariance-operator-analysis`、`batch-invariance-functional-test` 和 `batch-invariance-dataset-acceptance`。每个阶段只接受统一 `stage_result.json`；`BLOCKED` 保存 `reason_code` 与 `resume_state` 后暂停，证据充分的业务失败进入 `failed`，全部门禁通过后真实执行 `finalize` 并进入 `done`。

可恢复问题进入证据驱动的受限恢复状态。目标是恢复原失败状态；允许分析证据、实施不降低验收标准的最小修复、完整复验并记录反思。相同问题最多五轮，阶段最多十轮，全流程最多二十轮。五轮耗尽后停止修改并向用户说明全部证据。

运行目录必须包含 `run_config.json`、`state_transitions.jsonl`、`PROGRESS.md`、`verification.json`、各阶段目录和 `batch_invariance_final_report.md`。转移日志只追加并记录配置与证据哈希。最终报告按 [报告规范](references/final-report.md) 聚合；环境检查复用 [预检脚本](../batch-invariance-environment/scripts/check_batch_invariant_env.py)，阶段调用提示按需读取 [Profiling 调用资料](../batch-invariance-operator-analysis/references/profiling-collection.md) 和 [Math500 调用资料](../batch-invariance-dataset-acceptance/references/math500-invocation.md)。
