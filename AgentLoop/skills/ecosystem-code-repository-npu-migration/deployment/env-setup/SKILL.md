---
name: env-setup
description: This skill should be used when the user asks to research current Ascend CANN, torch, torch_npu and Python compatibility; inspect a target repository's installation documentation; create or reuse a dedicated non-base conda environment; install or repair the NPU stack and target checkout; and validate the live interpreter. Use for environment planning, creation, package installation, target import provenance, runtime repair, or environment-only validation.
version: 0.16.0
domain: ecosystem-code-repository-npu-migration
function: deployment
tags:
  - ascend
  - npu
  - migration
  - pytorch
author: addsubmuldiv
---

# NPU 环境设置

## 角色

在连续的 prompt-first 任务中，为目标仓库准备一个隔离且可用的昇腾 NPU 环境。根据当前证据阅读文档并选择命令；脚本只负责探测或 smoke 验证结果。

## 使用场景

用于 CANN/torch/torch_npu/Python 兼容性、conda 环境创建或复用、基础软件栈修复、目标仓库安装或环境校验。基础软件栈通过后，将 vLLM 专项工作路由到 `$ecosystem-code-repository-npu-migration:vllm-ascend-env`。

## 工作流程

### 1. 确认约束和机器环境

- 确认目标 checkout、请求的 CANN 路径/版本、torch、torch_npu、Python、可选能力和分布式范围。
- 检查架构、NPU 可见性、conda/mamba 是否可用、当前进程、请求的 CANN 布局以及用户明确提供的解释器。
- 仅将 `scripts/detect_npu_env.py` 用作事实探测。直接核验绝对 `sys.executable`、软件包版本和 `module.__file__` 路径。

### 2. 理解并选择 runtime 组合

- 在分析软件包兼容性或修复漂移前，阅读 [torch_npu_runtime_model.md](references/torch_npu_runtime_model.md)。
- 阅读 [compatibility_research.md](references/compatibility_research.md)，并针对请求的 CANN/torch/torch_npu/Python 组合刷新当前官方来源。
- 将用户明确指定的版本视为硬约束。如果没有受支持的组合满足这些约束，应在尝试安装前返回 `BLOCKED_ENVIRONMENT`。

### 3. 阅读仓库安装契约

- 阅读 [repository_installation.md](references/repository_installation.md)。
- 检查适用的 `README*`、`INSTALL*`、docs、`pyproject.toml`、`setup.py`、`setup.cfg`、requirements、conda YAML、lockfile 和 bootstrap 脚本。
- 选择仓库支持的安装器、extras 和 editable/wheel 模式。对于为避免 CUDA-only 依赖或保护选定 NPU 软件栈而采取的偏离，要说明理由。

### 4. 选择或创建 conda 环境

- 阅读 [environment_policy.md](references/environment_policy.md)。
- 只复用用户提供且经现场校验的有效非 base 绝对 Python，或复用本次任务中之前创建、并重新通过现场检查的同一环境。
- 否则创建 `<repo-slug>-npu-py<major><minor>-<repo-path-hash>`。名称冲突且无法确认归属时，追加 UTC 后缀，不要接管或覆盖已有环境。
- 优先使用 mamba，否则使用 conda。两者都不存在时返回 `BLOCKED_ENVIRONMENT`；绝不退回 base、系统 Python 或 venv。
- 所有 pip、探测、导入和仓库命令都使用解析出的绝对 Python。

### 5. 安装并校验基础软件栈

- 只加载用户选择的 CANN 安装，并按文档规定的顺序安装官方兼容的 torch/torch_npu 组合。
- 阅读 [base_runtime_policy.md](references/base_runtime_policy.md)。
- 唯一的 mandatory base gate 是：

  ```bash
  "<selected-python>" -c 'import torch; torch.npu.set_device(0)'
  ```

- 退出码为 0 表示基础环境对本 skill 可用，仅证明 backend 注册和逻辑设备选择。
- 将 `npu_smoke.py` 或 `pip check` 用作诊断或更深入的可选检查，不要作为普遍的 PASS 要求。仅在分布式范围内运行 `hccl_smoke.py`，仅在明确请求 Ray 范围时运行 `ray_npu_smoke.py`。

### 6. 安装目标 checkout 并保护软件栈

- 安装前读取软件包版本和导入位置。
- 在选定环境中执行仓库原生安装方式。不要把 `pip install --no-deps -e .` 作为通用默认方案。
- 验证目标软件包从预期 checkout 导入。
- 重新读取 torch 和 torch_npu 版本及导入路径，然后重跑精确的 mandatory base gate。
- 如果依赖改变或破坏软件栈，重新安装用户约束的官方兼容组合并重试。如果仓库无法与其共存，返回包含精确冲突的 `BLOCKED_TARGET_INSTALL`。
- 后续每次安装能力组件后，都再次运行同一个 gate。

## 输出

在当前任务中返回证据：选定的环境名和绝对 Python、查询过的官方来源、请求和实际安装的版本、阅读过的仓库文件、精确安装命令和退出码、导入来源、base gate 结果以及未测试范围。不要创建环境产物或安装命令台账。

## 边界

不要修改系统 Python 或 CANN，不要使用 conda base，不要静默放宽用户版本约束，不要安装无关 extras，不要修改目标源码、执行训练 entry 或仅凭环境设置声称仓库支持。

## 完成条件

目标安装后，选定的非 base Python 通过 mandatory base gate 且目标导入解析到预期 checkout 时返回 `PASS`。否则返回包含失败命令和要求的 `BLOCKED_ENVIRONMENT` 或 `BLOCKED_TARGET_INSTALL`。说明未运行的可选检查。

## 资源

- [torch_npu_runtime_model.md](references/torch_npu_runtime_model.md)：用于了解 runtime 架构、基础版本匹配和依赖漂移修复。
- [compatibility_research.md](references/compatibility_research.md)：选择版本前阅读。
- [environment_policy.md](references/environment_policy.md)：创建或复用环境前阅读。
- [repository_installation.md](references/repository_installation.md)：安装目标前阅读。
- [base_runtime_policy.md](references/base_runtime_policy.md)：运行 mandatory gate 或可选 smoke 前阅读。
- [capability_install_policy.md](references/capability_install_policy.md)：安装可选软件包前阅读。
- [ray_npu_policy.md](references/ray_npu_policy.md)：仅在 Ray 范围内阅读。
- `scripts/detect_npu_env.py`：可选的只读事实探测。
- `scripts/npu_smoke.py`、`scripts/hccl_smoke.py`、`scripts/ray_npu_smoke.py`：可选的专项校验器。
