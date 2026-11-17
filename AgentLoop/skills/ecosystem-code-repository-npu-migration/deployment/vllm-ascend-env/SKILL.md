---
name: vllm-ascend-env
description: This skill should be used when the user asks to install, repair and validate vLLM with vLLM-Ascend inside an existing healthy Ascend NPU conda environment. Use when inference or RL scope requires compatible vLLM packages, wheel inspection, local model binding and a real NPU generation smoke; do not use for base torch_npu setup or repository-wide migration planning.
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

# vLLM-Ascend 环境

## 角色

在 `env-setup` 选定的绝对 Python 中准备请求的 vLLM 能力，保护基础 torch/torch_npu 软件栈，并在资产允许时证明真实生成。

## 使用场景

仅当 mandatory base gate 通过且请求的 workload 确实需要 vLLM/vLLM-Ascend 时使用。

## 工作流程

1. 阅读 [vllm_ascend_install_policy.md](references/vllm_ascend_install_policy.md)，并刷新当前官方兼容性/安装来源。
2. 检查选定的 Python、torch/torch_npu 版本和导入结果。需要时使用 `scripts/probe_vllm_ascend_wheels.py` 探测候选 wheel 元数据。
3. 在该环境中安装精确兼容的 vLLM 和 vLLM-Ascend 组合；不要创建另一个环境。
4. 重新检查 torch/torch_npu 版本，并重跑 `"<selected-python>" -c 'import torch; torch.npu.set_device(0)'`。在 vLLM 测试前修复任何软件栈漂移。
5. 未明确模型时，请求 `$ecosystem-code-repository-npu-migration:local-assets` 查找有界模型。运行 `scripts/vllm_ascend_smoke.py` 执行真实生成。
6. 直接返回版本、命令、导入路径、base gate 结果、生成输出和限制。

## 边界

不要接受 CPU/CUDA fallback，不要从软件包导入推断仓库推理支持，不要安装目标仓库，也不要写入 mandatory capability 产物。

## 完成条件

只对实际证明的范围返回 PASS。仅有软件包/导入就绪而没有生成结果时，只能算部分能力证据；受保护的 base gate 失败会阻塞 vLLM 安装结果。

## 资源

- [vllm_ascend_install_policy.md](references/vllm_ascend_install_policy.md)：版本、安装和修复策略。
- `scripts/probe_vllm_ascend_wheels.py`：wheel 元数据探测。
- `scripts/vllm_ascend_smoke.py`：真实生成 smoke。
