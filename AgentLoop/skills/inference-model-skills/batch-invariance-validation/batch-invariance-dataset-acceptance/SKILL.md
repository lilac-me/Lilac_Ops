---
name: batch-invariance-dataset-acceptance
description: This skill should be used when the user asks to complete Math500 batch invariance acceptance across ten deterministic evaluation rounds and verify accuracy and sample-level text consistency.
version: 1.1.0
domain: inference-model
function: batch-invariance-validation
tags: [math500, acceptance, consistency]
author: mindspeed-agent
targets-repo:
  - vllm-ascend
---

# 阶段三：数据集验收

阶段二通过后，按 [Math500 验收规范](references/math500-acceptance.md) 和 [Agent 调用资料](references/math500-invocation.md) 执行 Math500 全量评测 10 轮。固定 `temperature=0`，关闭 prefix caching、chunked prefill 和 async scheduling；不得为获得通过结果改变配置。

公共精度流程负责环境准备、服务生命周期、评测和原始预测采集，但其成功不等于本阶段通过。收集同一运行的原始预测后，使用 [Math500 验收脚本](scripts/validate_math500_acceptance.py) 生成 `phase3/phase3_acceptance.json`；所有期望参数必须在执行前固定，禁止从实际结果反向填写。

环境、服务、下载或证据不完整等执行问题进入受限恢复状态，最多五轮且每轮必须采用新证据或不同修复动作。真实精度不达标、评分漂移或原始文本漂移属于证据充分的业务失败，不通过重复运行制造 PASS。

通过必须同时满足：精度达到预先固定的标准、每轮均有 500 条成功样本、10 轮评分完全一致、逐样本原始生成文本完全一致。确定的不一致输出 `FAIL`；环境异常或证据不足进入 `blocked_phase3`，保存原因码和恢复状态。

每轮命令、日志、公共评测报告、原始预测、配置和一致性结果均写入 `phase3/`。输出统一 `stage_result.json`，产物必须带相对路径、类型和 SHA-256，并交接到 `finalize`。
