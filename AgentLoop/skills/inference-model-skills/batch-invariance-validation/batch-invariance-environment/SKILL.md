---
name: batch-invariance-environment
description: This skill should be used when the user asks to verify environment admission for vLLM Ascend batch invariance, including package, custom operator, device, and confirmed TP prerequisites.
version: 1.1.0
domain: inference-model
function: batch-invariance-validation
tags: [vllm-ascend, environment, npu]
author: mindspeed-agent
targets-repo:
  - vllm-ascend
---

# 批次不变性环境准入

只接受已通过 Schema 校验且 TP 计划已明确确认的 `run_config.json`。缺少确认字段时返回 `BLOCKED_TP_PLAN_UNCONFIRMED`，不得补写用户回复或启动后续阶段。

使用 [环境预检脚本](scripts/check_batch_invariant_env.py) 检查并保存：vLLM 与 vLLM Ascend 的版本和提交快照、`batch_invariant_ops` 包及来源、自定义算子动态库、Torch 算子注册，以及 A2/A3/A5 芯片、可见设备、拓扑和 TP 匹配关系。完整命令、标准输出、标准错误和退出码写入 `environment/`；缺少 Git 元数据时记录 `unavailable`。

- 五项全部通过：输出 `PASS`，交接到 `phase1_profiling`。
- A2、A3 或 A5 缺少算子包、动态库或 Torch 注册：从匹配的 vLLM Ascend 源码自动构建，随后重新执行完整准入。
- 其他包、设备或 TP 问题：使用已注册原因码进入 `blocked_environment`，修复后重新执行完整准入。

芯片分类只采用预检脚本的结果：先判断 A5，再将 910B/A2g3 归为 A2，其余 910 归为 A3；其他型号或检测失败返回 `BLOCKED_CHIP_DETECTION`。不要将原始型号字符串与 A2/A3/A5 直接比较。A2 仅在白名单选择时复用 A3 分区。

在目标运行环境执行 `python scripts/check_batch_invariant_env.py --run-config <run_config.json> --report-json <运行目录>/environment/preflight.json`（脚本路径取上面的链接）。报告分开记录芯片原名、规范分类、构建平台、安装能力和安装状态。`SUPPORTED` 仅代表本工具支持该平台的手动构建指引，不保证构建成功或本机具备构建依赖；`READY` 只代表包、SO 加载和必需算子注册通过，不替代 NPU 数值测试或设备/TP 检查。

OPP 路径按 `--opp-path`、非空 `ASCEND_OPP_PATH`、兼容默认路径的优先级选择；报告包含最终路径及来源。已有 SO 先复检，不重复构建。必须检查 FIA、MatMul、ReduceSum 三个批次不变性算子；缺失或损坏统一使用 `BLOCKED_OPERATOR_INSTALLATION`。自动构建参数为 A2=`ascend910b`、A3=`ascend910_93`、A5=`ascend950`，且只使用与已安装版本匹配的源码树。

输出统一 `stage_result.json`，包含运行标识、阶段、状态、结果、原因码、恢复状态、尝试次数、配置摘要、带 SHA-256 的产物和精确交接值。命令与修复提示可按需读取 [Agent 调用资料](references/environment-admission.md)。
