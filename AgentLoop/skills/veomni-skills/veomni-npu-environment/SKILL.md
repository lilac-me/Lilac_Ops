---
name: veomni-npu-environment
description: >-
  This skill should be used when the user asks to check whether the Ascend NPU
  environment satisfies VeOmni installation requirements, inspect chip model,
  Python/VeOmni/PyTorch/TorchNPU/CANN/Transformers version compatibility, verify
  NPU device visibility, run a minimal TorchNPU smoke test, distinguish officially
  supported from experimental software combinations, or produce an environment
  report with remediation suggestions.
version: 0.1.0
domain: common
function: deployment
tags:
  - veomni
  - environment
  - torch-npu
  - cann
  - smoke-test
author: mindspeed-agent
license: MIT
compatibility: "VeOmni >= 0.1, CANN 8.x"
---

# VeOmni 昇腾环境检查

## 目的

检查 VeOmni 在昇腾 NPU 环境上的安装和版本配套情况，输出问题清单与修改建议。

## 前置条件

- 可登录目标环境（物理机/容器），或明确声明仅做静态分析（无 NPU 时）
- 用户已授权执行检查类命令（只读命令；安装/卸载类命令必须先征得同意）

## 检查流程

1. **硬件与系统**
   - 芯片型号：`npu-smi info`
   - CPU 架构：`uname -m`（x86_64 / aarch64）
   - 操作系统：`cat /etc/os-release`
   - NPU 设备可见性：`npu-smi info -l`、Python 内 `torch.npu.device_count()`
2. **软件版本**
   - Python：`python --version`
   - VeOmni：`pip show veomni` 或仓库 `git rev-parse HEAD`（记录 source revision）
   - PyTorch：`python -c "import torch; print(torch.__version__)"`
   - TorchNPU：`python -c "import torch_npu; print(torch_npu.__version__)"`
   - CANN：`cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` 或 `set_env.sh` 路径
   - Transformers：`python -c "import transformers; print(transformers.__version__)"`
3. **最小冒烟测试**

   ```bash
   python -c "
   import torch, torch_npu
   x = torch.randn(4, 4).npu()
   y = (x @ x).sum()
   torch.npu.synchronize()
   print('TorchNPU smoke OK:', y.item())
   "
   ```

4. **配套判定**：对照 VeOmni 官方 README/安装文档，区分**正式支持组合**与**实验性组合**，逐项标记 OK / WARN / FAIL。
5. **输出证据**：按产物契约写 `environment_report.json`；面向人的环境结论由最终过程记录统一汇总。

## 输出契约

`environment_report.json`（Schema：`benchmarks/veomni/schemas/environment_report.json`）：

```json
{
  "mode": "npu | static",
  "hardware": {"chip": "string", "cpu_arch": "string", "os": "string", "device_count": 0},
  "software": {"python": "string", "veomni": "string", "pytorch": "string",
                "torchnpu": "string", "cann": "string", "transformers": "string"},
  "smoke_test": {"executed": true, "passed": true, "error": null},
  "compatibility": [{"component": "string", "status": "OK | WARN | FAIL", "note": "string"}],
  "issues": ["string"],
  "suggestions": ["string"]
}
```

## 门禁与约束

- 放行条件：版本和硬件信息完整，NPU 最小测试通过；静态分析模式必须 `"mode": "static"` 明确标记，且不得宣称 NPU 可用。
- 只执行只读检查命令；安装、升级、环境变量持久化等修改操作必须先获得用户授权。
- 环境不兼容时任务标记 `BLOCKED`，给出明确修改建议（预期版本、安装顺序、驱动/CANN 升级路径）。
