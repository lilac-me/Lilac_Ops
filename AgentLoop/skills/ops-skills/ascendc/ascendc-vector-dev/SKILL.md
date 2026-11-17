---
name: ascendc-vector-dev
description: >-
  This skill should be used when the user asks to write, design, or review an
  Ascend C kernel that must move data with DMA, use hardware vector math
  (Sin/Cos/Atan/Sqrt), vectorize the inner loop of reduction/NMS-style
  algorithms, double-buffer with TQue, or correctly synchronize vector/scalar
  pipes. It codifies the mature Ascend C vectorization patterns observed in
  production kernels: DMA-based data movement instead of
  scalar GM reads, hardware vector math via the scalar-in-tensor trick +
  SetFlag/WaitFlag, reformulating sequential loops as vector ops (mask AND),
  double buffering, and the common correctness traps (cache-line scalar write
  races, missing EnQue/DeQue sync, vector-write to scalar-read barriers, NaN
  resurrection). Typical triggers include "为什么这么慢/全标量", "算子怎么向量化",
  "DataCopy 怎么用", "SetFlag/WaitFlag", "写个不踩坑的 Ascend C kernel".
version: 1.0.0
domain: ops
function: ascendc
tags:
  - ascendc
  - vectorization
  - datacopy
  - kernel-patterns
  - best-practices
author: mindspeed-agent
---

# Ascend C 算子向量化开发（基础）

## 定位

向量化是 Ascend C 开发的基础能力，应在**设计与编码阶段**即贯彻，否则后期优化成本极高
（如 NMS 全标量 select 70ms → 向量 `&` 后 <1ms）。本 Skill 把生产级 kernel 中沉淀的
向量化模式固化为开发规范（参考公开 CANN 示例与各算子仓库的 op_host/op_kernel 工程）。

> 与 `ascendc-op-dev`（开发编排）、`ascendc-api-best-practices`（API 正确用法）、
> `ascendc-tiling-design`（Tiling）配套使用。本 Skill 聚焦**向量化编码模式与同步正确性**。

## 核心原则

### 原则 1：数据移动一律走 DMA，禁止标量逐元素 GM 读

- 批量数据用 `DataCopy` / `DataCopyPad`（`GlobalTensor → LocalTensor`搬运到 UB或`LocalTensor → GlobalTensor`搬出到GM）。`DataCopy`一次搬运量需32B对齐（8个float类型数据/16个half类型数据）。
- 禁止用 `GetValue` / `__gm__` 指针解引用逐个读 GM 元素（每元素一次 L2 命中延迟，N² 场景放大）。
- 示例（成熟版）：

  ```cpp
  DataCopyParams cp{1, (uint16_t)(7 * sizeof(float)), 0, 0};  // 一次搬一个 box 记录（7 float）
  DataCopy(curLocal, boxGm[cur_box * 7], 16);                  // 16 对齐批量搬
  // 多 box 批量：DataCopyParams{(uint16_t)n, (uint16_t)(7*sizeof(T)), 0, 0} + DataCopyPad
  ```

- box 记录只有 7 个 float，用 `dataAlign=16` 对齐批量搬（多取填充不影响），换向量化 DMA。

### 原则 2：数学函数用硬件向量指令（标量入 tensor 技巧）

- **向量 API 有硬件指令**：
  `AscendC::Sin / Cos / Atan / Sqrt / Rsqrt / Exp / Ln / Reciprocal / Abs / Relu` 等。
- 用法：向量化计算，数据由GM批量搬入UB后，利用Ascendc API进行计算，并批量搬出到GM。**禁止手写多项式近似**（又慢又有 ulp 误差）。以下为示例：

  ```cpp
  DataCopy(localTensorA, aGM, 32); // a变量从GM上搬入UB，长度为32，满足32B对齐要求
  DataCopy(localTensorB, bGM, 32);
  Sin(localTensorC, localTensorA, localTensorB, 32); // 向量化计算，避免单独读取
  DataCopy(cGM, localTensorC, 32); // c变量从UB上搬出到GM，长度为32，符合32B对齐要求
  ```

### 原则 3：顺序算法改写成「并行矩阵 + 向量归约」

- 只要可行，把「串行贪心 / 依赖循环」拆成两步：
  1. **并行矩阵**：每对独立计算，输出掩码/中间值（按核×块切分，全并行）。
  2. **顺序收尾**：用**向量 op** 做内层归约（如 NMS 的 `maskTemp = maskTemp & maskRow`），
     避免 O(N²) 标量内层循环。
- NMS 示例（成熟版 gather_nms3d_mask）：

  ```cpp
  // 逐行 i：maskTemp[j] 累积「是否未被任何 keep 框抑制」
  for (i = 0; i < boxNum; ++i) {
      if (maskTemp.GetValue(i) == 1) {   // 标量读存活状态（低频，可接受）
          keepTemp.SetValue(keepNum++, i);
          DataCopyPad(maskLocal, maskGm[i * maskNum], cp, pad);
          maskTemp = maskLocal & maskTemp;   // 向量 AND：整行一次向量 op
          PipeBarrier<PIPE_ALL>();
      }
  }
  ```

- 内层 `for j` 如果能变成一次向量 op（AND/乘/选），性能量级提升。

### 原则 4：双缓冲 + 三段流水

- 用 `TQue<QuePosition::VECIN|VECOUT, BUFFER_NUM=2>` 做 CopyIn/Compute/CopyOut 流水，
  `EnQue/DeQue` 天然提供 MTE→V 同步，隐藏搬运延迟。

  ```cpp
  pipe.InitBuffer(inQueue, BUFFER_NUM, tile * sizeof(T));   // BUFFER_NUM=2 双缓冲
  LocalTensor<T> x = inQueue.AllocTensor<T>();
  DataCopyPad(x, xGm[off], cp, pad);
  inQueue.EnQue(x);
  LocalTensor<T> xr = inQueue.DeQue<T>();   // 等 MTE 完成后才返回
  // ... compute ...
  inQueue.FreeTensor(xr);
  ```

### 原则 5：同步原语（正确性的根）

| 场景 | 用法 |
| ------ | ------ |
| 向量 op 写 UB → 标量 `GetValue` 读 | `SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);` |
| DataCopy 后消费（同核内不用队列） | `SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);` 或用 TQue EnQue/DeQue |
| 标量写 → DataCopy 出（UB→GM） | `SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);` |
| 全流水排空 | `PipeBarrier<PIPE_ALL>();`（V/MTE2/MTE3/S 单流水用 `PipeBarrier<PIPE_V>()` 等） |
| 队列化异步搬运 | `TQue` `AllocTensor → DataCopy → EnQue → DeQue → ... → FreeTensor` |

## 常见易错点（编码自查清单）

### R1：多核标量写共享缓存行 → 写入丢失/错乱

- 多核并发 `SetValue`/`__gm__` 写，若不同核写同一缓存行（如每 box 记录 48B，跨 64B 行），
  会发生丢失/错乱（实测 stride=48B 时 boxExt 一半写成 0/错位）。
- **规避**：记录按 64B 对齐（stride=16 个 float）；或改用 DMA（`DataCopy`）批量写；或单核写。
- 校验：同输入**连续跑 ≥3 次**结果一致（确定性测试）。

### R2：DataCopy 异步未同步 → 读到旧/垃圾数据

- `DataCopy`/`DataCopyPad` 是异步的，直接在其后读 UB 会竞态。
- **规避**：用 `TQue` EnQue/DeQue，或 `SetFlag/WaitFlag<HardEvent::V_MTE2>`。

### R3：向量写 UB → 标量读，仅 `PipeBarrier<PIPE_V>` 可能不够

- 实测 `PipeBarrier<PIPE_V>` 后标量读向量写的数据仍可能不确定；用
  `SetFlag/WaitFlag<HardEvent::V_S>`（标量读向量结果的专用事件）或 `PipeBarrier<PIPE_ALL>`。

### R4：避免 `Select`（DAV_2201 使用保留 UBUF）

- `Select` 在 arch22 标记 `__ASC_USE_RESERVED_UBUF__`，实测多行全 1 mask 时结果不确定。
- **规避**：用算术组合或整型 `&` 做掩码（见原则 3）。

### R5：0×NaN 复活

- 用 `p/(p+n)` 类算术掩码时，`s==0` 会 `0/0→NaN`，`alive=0` 的框再乘 NaN 变 NaN 复活。
- **规避**：分母加极小量（如 `1e-20f`），或确保掩码由整型 `&` 产生（无除法）。

### R6：与 CPU golden 逐操作对齐

- 转写 golden 时符号/象限极易错（如 `check_in_box2d` 的 `cos(-θ)/sin(-θ)` 展开）。
- **规避**：编码后用**中间量交叉验证**（dump boxExt / iou 矩阵与 Python golden 逐元素比，
  而非只看最终 keep 是否相等），再跑全量精度。

### R7：多核写缓冲必须预清零或全覆盖

- 上三角/稀疏输出：host 侧 `at::zeros` 预清零，或对跳过区做**向量 ones 写**
  （成熟版 `if (i >= start + dataAlign) DataCopy(maskGm[...], oneLocal, dataAlign);`）。

## 参考资料

- 成熟算子（production patterns）示例（公开 CANN 示例/算子仓库中常见）：
  - NMS 类：mask 矩阵并行 + 向量 Sin/Cos/Atan；gather 用向量 `&` 归约
  - IoU/几何类：批量 box DMA 搬运 + 向量 trig
  - 逐元素类：多元素并行向量化
- 配套 Skill：`ascendc-op-dev`（开发编排）、`ascendc-api-best-practices`（API 用法）、
  `ascendc-tiling-design`（Tiling）、`ascendc-op-debug`（运行时调试）
