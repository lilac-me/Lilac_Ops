---
name: ascendc-adapter-layer
description: >-
  This skill should be used when the user asks to design the host-side adapter
  layer for an Ascend C operator: choosing between single-op direct-invoke
  (<<<>>> + torch extension torch.ops.npu.xxx) vs aclnn registry invocation
  (op_host OpDef/Tiling/InferShape + Python aclnn call), producing the adapter
  code (tiling computation, workspace, Python wrapper), and writing the
  deliverable README that tells users how to verify and call the operator.
  Typical triggers include "算子怎么接入 torch/aclnn", "写适配层", "aclnn 调用",
  "TORCH_LIBRARY 注册", "交付件 README 怎么写", "用户拿到算子后怎么用".
version: 1.0.0
domain: ops
function: ascendc
tags:
  - ascendc
  - adapter
  - aclnn
  - torch-extension
  - delivery
author: mindspeed-agent
---

# Ascend C 算子适配层设计（交付件如何被用户消费）

## 定位

算子 kernel（NPU 计算）只是交付件的一半；**适配层**决定用户怎么调用算子，也决定交付件的形态。
本 Skill 覆盖两种调用方式的选择、适配层代码设计，以及交付件 README（用户使用说明）。

> 配套 `ascendc-op-dev`（开发编排）、`ascendc-direct-invoke-template` / `ascendc-registry-invoke-template`
> （工程模板）、`torch-ascendc-op-extension`（torch 接入）。设计阶段就要选定调用方式。

## 两种调用方式（先选型）

| 维度 | 单算子调用（direct-invoke） | aclnn 注册调用 |
| ------ | ------------------------------ | ---------------- |
| 内核调用 | `<<<>>>`（或 extern C 函数） | aclnn 框架调度 |
| Host 侧 | 自写 main / torch 扩展 | op_host（OpDef + Tiling + InferShape） |
| Python 调用 | `torch.ops.npu.xxx(...)`（TORCH_LIBRARY 扩展） | `aclnn_xxx(...)`（C++ API）或 torch_npu 内建 |
| 适用 | POC / 单算子验证 / 快速原型 | 生产入库 / 入图（GE/CANN） / 上库 |
| 交付形态 | 工程 + `<<<>>>` main + torch 扩展 .so | 算子包（op_host+op_kernel 编译进 CANN） |
| 快速验证 | `bash run.sh`（standalone）+ pytest | aclnn UT/ST |

**决策建议**：

- 快速验证 / POC / 与 PyTorch 对比 → **单算子调用**（`<<<>>>` + torch 扩展，可复用 pytest）
- 生产 / 入图 / 多算子编排 → **aclnn 注册**（`ascendc-registry-invoke-template`）

## 单算子调用适配层（推荐路径）

### 1. 内核与 tiling

- `op_host/<op>_tiling.h`：Tiling 结构体（**kernel/host 共用单一数据源**）
- `op_kernel/<op>_kernel.cpp`：device kernel（ASC 语言）
- `op_host/<op>.cpp`：host + `<<<>>>` main（**独立快速验证入口**，直调 POC 形态）

### 2. torch 扩展（`torch.ops.npu.<op>`）

- `op_extension/<op>_torch.cpp`：`torch::Tensor` 接口 → `ComputeTiling()` → 分配 workspace →
  kernel 串联 launch。**必须 `c10_npu::getCurrentNPUStream().stream(true)`（清 queue 防乱序）**
- `op_extension/register.cpp`：`TORCH_LIBRARY_FRAGMENT(npu, m){ m.def(...) }` +
  `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` + **Meta backend**（torch.compile/fx 需要）
- 多输出用 `std::tuple<at::Tensor, at::Tensor>`

### 3. Python 侧封装（可选，面向用户的最外层）

- 把预处理（排序/重排/映射）与 kernel 调用封装为一个函数，如：

  ```python
  def nms3d(boxes, scores, thresh):
      order = torch.argsort(scores, descending=True)   # NPU 预处理
      boxes_sorted = torch.index_select(boxes, 0, order)
      keep, num = torch.ops.npu.nms3d_kernel(boxes_sorted, thresh)
      return order[keep[:num.item()].long()]
  ```

- 单一函数即用户的入口，隐藏内部 kernel 细节

### 4. 交付 README 必须包含

- 每个文件的作用（kernel / host / extension / scripts / CMake）
- 快速验证步骤（编译 → standalone 验证 → pytest）
- Python 调用示例（含环境初始化）

## aclnn 注册调用适配层（生产）

### 1. op_host 三件套

- `op_def` 注册：`Input/Output/Attr` + `SetInferShape` + `SetInferDataType` + `AICore().SetTiling` + `AddConfig("ascend910b"...)` + `OP_ADD`
- `InferShape` / `InferDataType`：推导输出 shape/dtype
- `TilingFunc(TilingContext*)`：`GetInputShape`/`GetAttrs` → 计算 tiling → `SaveToBuffer` → `SetBlockDim` → `GetWorkspaceSizes`

### 2. 入图/算子包

- 编译进 CANN 算子包；图引擎按 TilingKey 分发 kernel 模板（fp16/fp32 用 TilingKey 分支）

### 3. Python 侧（用户入口）

- C++：`EXEC_NPU_CMD(aclnnXxx, ...)` 调 aclnn API
- Python：通过 aclnn 的 python 绑定，或 torch_npu 已有 op，或自定义 `torch.autograd.Function` 包装

  ```python
  class XxxFunction(Function):
      @staticmethod
      def forward(ctx, boxes, scores, thresh):
          ...
          out, num = mx_driving._C.xxx(boxes, thresh)   # 内部 EXEC_NPU_CMD(aclnnXxx)
          return ...
  ```

### 4. 交付 README 必须包含

- 算子规格（I/O/dtype/约束）、构建打包方式、Python/C++ 调用示例、UT/ST 运行方法

## 适配层设计注意事项

- **Tiling 单一数据源**：`xxx_tiling.h` 被 kernel / host / torch / op_host 共享，禁止重复定义
- **stream 同步**：torch 扩展必须 `stream(true)`；`zeros_like` 会乱序，用 `empty_like`/`zeros`
- **Meta backend**：`TORCH_LIBRARY_IMPL(npu, Meta)` 必写（torch.compile / aclgraph 依赖）
- **workspace**：多 kernel 中间 buffer 用 device tensor（`at::empty`/`at::zeros`）或 `GetWorkspaceSizes`
- **交付物自解释**：用户拿到交付件后，README 应让其在 5 分钟内完成「编译 + 验证 + 调用」

## 参考资料

- `torch-ascendc-op-extension`（单算子 → torch.ops.npu.xxx 全流程）
- `ascendc-direct-invoke-template` / `ascendc-registry-invoke-template`（两种工程模板）
- `ascendc-docs-gen`（算子文档/README 模板）
- 成熟参考：aclnn 注册 + csrc/ops.py 适配层模式（参照各算子仓库的 aclnn 工程结构，如
  `op_host`(OpDef/Tiling/InferShape) + `csrc`(aclnn 封装) + `ops`(Python 入口)）
