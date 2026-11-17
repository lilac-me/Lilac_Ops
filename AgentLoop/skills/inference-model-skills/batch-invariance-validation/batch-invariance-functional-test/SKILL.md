---
name: batch-invariance-functional-test
description: This skill should be used when the user asks to run model-level batch invariance functional tests, validate long-input or long-output scenarios, execute a TP matrix, or diagnose token and logprob mismatches.
version: 1.1.0
domain: inference-model
function: batch-invariance-validation
tags: [functional-test, tensor-parallel, logprob]
author: mindspeed-agent
targets-repo:
  - vllm-ascend
---

# 阶段二：功能测试验证批次不变性

读取 [阶段二验收规范](references/phase2-acceptance.md) 和 [调度与驱逐规则](references/scheduler-eviction.md)，通过 [阶段二执行入口](scripts/run_phase2_acceptance.sh) 严格依次运行：长输出 TP=8、BS=64、约 `2k→20k`；长输入 TP=8、BS=64、约 `20k→2k`；矩阵 TP=1/2/4、BS=16/32、`100→512` 与 `1k→2k`，每组 3 轮。

实际 TP 必须来自已确认的 `run_config.json`，并通过 `VLLM_TEST_RUN_CONFIG` 传给执行入口。修改 TP 时返回确认门禁，重新确认后隔离证据。

比较完整 token ID 序列和相同 token 路径的 chosen-token logprob。单个 TP 在模型加载阶段因显存或 KV cache 容量不足时记录 `SKIPPED_TP_RESOURCE_UNAVAILABLE`，不计为通过、不阻塞其他 TP，也不得外推；阶段整体至少一个目标 TP 完整通过。

请求驱逐不受支持、证据不完整、提示长度或数量不合规时进入 `blocked_phase2`。数值不一致时保存当前 attempt 的 `failure_manifest.json`，再按 [异常定位资料](references/model-debugging.md) 依次检查 scheduler、profiling、单算子，使用 TP=1/2 缩小通信范围，最后依据证据选择 msprobe slice 或 request-id dump。证据不足时返回资料定义的 `BLOCKED_*` 原因码。

可恢复问题进入受限恢复状态。每轮读取新日志和 manifest，实施不降低场景标准的最小修复并重跑原场景，再记录前后证据和反思；同一问题五轮仍未闭环时才通知用户。

每个场景保存配置、命令、脚本版本或提交、日志、调度分析、失败清单和汇总。输出统一 `stage_result.json`，产物必须带相对路径、类型和 SHA-256。
