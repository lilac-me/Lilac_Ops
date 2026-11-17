---
name: ops-bwd-golden
description: >-
  This skill should be used when the user asks to generate a backward torch
  golden reference (gradient benchmark) for operator precision validation on
  Ascend NPU, starting from a forward implementation in Triton or NumPy. It
  normalizes the forward code into a readable standard torch implementation,
  derives gradients via PyTorch automatic differentiation as the backward
  benchmark, and produces a functionally equivalent torch operator as the
  readable precision standard, regardless of the target implementation language
  (Triton / Ascend C / TileLang). Typical triggers include "生成反向 torch
  标杆", "把 triton 算子转成 torch 参考", "numpy 转 torch 作为精度标准", "用
  torch 自动求导生成梯度标杆", or "gradcheck 验证梯度".
version: 1.1.0
domain: ops
function: common
tags:
  - golden
  - backward
  - torch
  - autograd
  - reference
author: mindspeed-agent
compatibility: MindSpeed >= 0.1.0, CANN >= 9.1.0, Triton >= 3.2.2
targets-repo:
  - triton-ascend
---

# 反向 torch 标杆（golden）生成

## 功能概述

本 Skill 提供**跨语言通用**的算子精度标准生成能力，流程固定为三步：

1. **归一化**：把输入的前向实现（Triton 代码 / NumPy 代码 / 自然语言）转成**可读的标准 torch 实现**；
2. **反向标杆**：对标准 torch 实现用 **PyTorch 自动求导**推演出反向梯度（标杆），并**可视化其链式推导过程**（前向分解 → 逐层梯度公式 → 中间梯度），不只给 `.grad` 终值；
3. **精度标准**：产出功能等价、过程可读的 torch 算子（前向 + 反向），作为后续算子实现（Triton / Ascend C）的精度 golden。

选择 torch 作为归一化目标，是因为 torch 表达式向量化、**过程可读**，便于逐行核对语义；而其自动求导是反向梯度最可信的 golden，**不手写梯度公式**。

## 定位

```text
输入：Triton 代码 / NumPy 代码 / 自然语言（前向实现）
        │
        ▼  归一化
标准 torch 实现（可读、功能等价）
        │
        ├─ 前向 → 前向 golden（torch 前向输出）
        └─ 反向 → 反向 golden（torch 自动求导 .grad）
        │
        ▼
等功能 torch 算子 = 精度标准（供目标算子实现比对）
```

---

## 交付件（产出物）

本 Skill 的产物为三份文件，落盘到 `outputs/<算子名>/golden/`（无该目录时与算子实现同目录）：

| 交付件 | 内容 | 说明 |
| ------ | ------ | ------ |
| `golden_ref.py` | 标准 torch 前向实现 + 反向标杆函数 | 功能等价、过程可读，精度标准的唯一载体 |
| `golden_bwd_trace.md` | 反向过程可视化：前向分解 → 逐层梯度公式 → 中间梯度 | 反向算子的**数学/正确性规格**（算什么 + 数值目标），非可直接运行的 kernel |
| `golden_test.py` | 前向 `allclose` + 反向 `gradcheck` 比对脚本 | 把目标算子实现与 golden 逐项比对 |
| `golden_report.md` | 精度比对报告 | shape×dtype 矩阵 + 前向/反向误差 + 达标结论 |

### `golden_ref.py` 结构

```python
import torch

def reference_forward(x, w):
    """标准 torch 前向实现：可读、功能等价（归一化产物）。"""
    return torch.nn.functional.linear(x, w)  # 替换为归一化得到的逻辑

def reference_backward(x, w):
    """反向标杆：torch 自动求导。"""
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    y = reference_forward(x, w)
    y.sum().backward()
    return x.grad, w.grad

class GoldenOp(torch.nn.Module):
    """等功能 torch 算子（精度标准），可直接 import 复用。"""
    def forward(self, x, w):
        return reference_forward(x, w)
```

### `golden_test.py` 结构

```python
# 前向比对：assert_precision(实际, reference_forward(x, w), dtype, shape)   # 复用 ops-test-gen
# 反向比对：gradcheck(custom_op, (x, w)) 或 allclose(实际梯度, reference_backward(x, w))
```

### `golden_report.md` 结构

| shape × dtype | 前向误差（max / 超限比） | 反向误差（gradcheck / 超限比） | 结论 |
| ------ | ------ | ------ | ------ |
| (4,8) × fp32 | … | … | ✅/❌ |

---

## 第一部分：输入归一化（→ 标准 torch 实现）

### 1.1 输入形式与处理

| 输入形式 | 处理方式 |
| --------- | --------- |
| **Triton kernel 代码** | 提取**计算语义**（每元素/每 tile 算什么），忽略 tiling/block/grid/掩码细节，映射为 torch 向量化表达 |
| **NumPy 代码** | 逐算子映射到 torch，broadcasting 语义两者一致 |
| **自然语言 / 数学公式** | 规格化为 torch 表达式 |
| **已有 torch 代码** | 直接作为标准实现，规范 dtype/device/注释 |

### 1.2 Triton → torch 映射要点

按语义提取，不逐行翻译（Triton 的 SIMT tile 逻辑与 torch 向量化是两套范式）：

| Triton 原语 | torch 等价 |
| --------- | --------- |
| `tl.load(ptr + offs, mask=...)` | 切片 / `x[mask]`（掩码即索引） |
| `tl.store(ptr + offs, val)` | `out[...] = val`（或返回新 tensor） |
| `tl.arange` + 逐元素运算（`+` `*` `tl.exp` `tl.sigmoid` …） | torch 向量化算术（`+` `*` `torch.exp` `torch.sigmoid` …） |
| `tl.sum(x, axis=k)` | `torch.sum(x, dim=k)` |
| `tl.max(x, axis=k)` | `torch.max(x, dim=k)` |
| `tl.dot(a, b)` | `torch.matmul(a, b)` |
| `tl.where(cond, a, b)` | `torch.where(cond, a, b)` |

**提取步骤**：通读 kernel → 确定每个输出元素的表达式 → 去掉 grid/block/掩码/累加循环的分块外壳 → 写成 batch 维 + 归约维的向量化 torch 表达式。

### 1.3 NumPy → torch 映射要点

| NumPy | torch |
| ----- | ----- |
| `np.add / np.multiply / np.divide / np.subtract` | `torch.add / mul / div / sub`（或 `+ - * /`） |
| `np.exp / np.log / np.sqrt / np.sin …` | `torch.exp / log / sqrt / sin …` |
| `np.sum(x, axis=k)` | `torch.sum(x, dim=k)` |
| `np.matmul(a, b)` / `a @ b` | `torch.matmul(a, b)` / `a @ b` |
| `np.where(cond, a, b)` | `torch.where(cond, a, b)` |
| `np.float32 / np.float16` | `torch.float32 / torch.float16` |

NumPy 与 torch 的 broadcasting 规则一致，dtype 直接对应；注意 `np.sum` 默认对全部轴求和，`torch.sum` 默认也是全部，显式写 `axis`/`dim` 避免歧义。

---

## 第二部分：前向标准 torch 实现（可读）

归一化后的标准实现是精度标准的**前向部分**：

```python
import torch

def reference_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """标准 torch 前向实现：与目标算子前向语义一一对应，过程可读。"""
    return torch.nn.functional.linear(x, w)  # 替换为归一化得到的逻辑
```

| 要求 | 说明 |
| ------ | ------ |
| 语义对齐 | 与目标算子前向公式一致（含广播/规约/激活/归一化） |
| dtype 一致 | 与算子同 dtype；累加用 fp32 再回写 |
| 设备一致 | 与算子同设备（`npu`），避免设备迁移误差 |
| 固定 seed | 有随机性（dropout 等）必须固定 seed |
| 可读 | 优先 `torch.nn.functional` / 原生算子，过程清晰，便于核对语义 |

---

## 第三部分：反向 torch 标杆（autograd）

### 3.1 模板

```python
def reference_backward(x: torch.Tensor, w: torch.Tensor):
    """反向标杆：torch 自动求导。返回与目标 backward 输出对齐的梯度。"""
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    y = reference_forward(x, w)
    loss = y.sum()                     # 标量化（sum 最常用；也可 (y * weight).sum()）
    loss.backward()
    return x.grad, w.grad              # 与目标 backward 返回值逐项对齐
```

### 3.2 梯度对齐规则

- **返回值逐项对齐**：反向标杆的梯度列表必须与目标 `backward` 返回值个数/顺序一致；不需要梯度的输入对应 `None`。
- **广播/规约语义**：广播维反向应**规约**、规约维反向应**广播**——torch 自动求导自动满足，自研反向需对照核对。
- **`requires_grad` 组合**：逐输入判断是否需要梯度，避免对 `requires_grad=False` 的输入做无谓计算。

### 3.3 gradcheck / gradgradcheck

对自定义 `torch.autograd.Function`，用数值差分逼近解析梯度，与 torch 自动求导双向对照：

```python
from torch.autograd import gradcheck, gradgradcheck

# 在 CPU 上用 float64 跑（见下方说明），不要用 npu
x = torch.randn(4, 8, device='cpu', dtype=torch.float64, requires_grad=True)
w = torch.randn(8, 8, device='cpu', dtype=torch.float64, requires_grad=True)

gradcheck(custom_op, (x, w), eps=1e-6, atol=1e-5, rtol=1e-3)       # 反向 vs 自动求导
gradgradcheck(custom_op, (x, w), eps=1e-6, atol=1e-3, rtol=1e-2)   # 二阶（可选）
```

> **gradcheck 必须在 CPU 上用 float64 跑**：昇腾 NPU 不支持 double dtype（会被降级成 fp32），
> fp32 有限差分噪声会淹没真实梯度差异，造成数值/解析 Jacobian 错位误报（实战验证：同一反向在
> NPU fp32 下 gradcheck 误报、在 CPU float64 下才 PASS）。**前提是 `reference_forward` 不能
> 内部下转 dtype**——若参考实现里写了 `.to(torch.float32)`，float64 输入会被舍入、有限差分失效。
> 计算精度应取 `max(输入 dtype, fp32)`（bf16/fp16 上转 fp32，fp32/fp64 保持原样）。
>
> 反向精度的**直接比对**（显式反向 vs `torch.autograd.grad`，无有限差分）不受此限，可在 NPU
> fp32 上跑，比 gradcheck 更可靠，作为设备侧主验收手段。

### 3.4 反向过程可视化（梯度推导轨迹）

`.grad` 只给反向的**终值**，掩盖了「梯度怎么一步步算出来」的链式过程。开发反向算子需要**过程可读**的推导轨迹，方法分三步：

1. **前向分解**：把前向拆成命名中间量，每一步对应一个基本算子；
2. **逐层梯度公式**：按链式法则从 `dy` 出发，逐层写出每个中间量的梯度；
3. **中间梯度数值**：用 `retain_grad` / `torch.autograd.grad` 拿到每个中间量的梯度，验证公式并可视化数值流。

```python
# 前向分解 + 中间梯度保留
z1 = x @ w                          # matmul, (M,N)
z1.retain_grad()
z2 = z1 + b                         # bias 广播, (M,N)
z2.retain_grad()
y = torch.nn.functional.silu(z2)    # 激活, (M,N)
y.sum().backward()                  # 反向

# 逐层梯度（链式，自 dy 向下）—— 即反向算子开发蓝图
# dy   : 上游梯度 (M,N)
# dz2  = dy ⊙ silu'(z2)  → z2.grad 即为 dz2（可直接比对）
# db   = Σ dz2 (dim=0)     → bias 反向
# dz1  = dz2               → add 反向（广播维规约）
# dx   = dz1 @ w^T         → matmul 反向
# dw   = x^T @ dz1         → matmul 反向
```

产出 `golden_bwd_trace.md`：前向分解表 + 逐层梯度公式表 + 各中间量的 shape/dtype/数值样本，作为反向算子的**数学/正确性规格**——明确「每个梯度输出算什么、规约/广播轴、需保存的中间量、分步数值目标」。

> **边界说明**：trace 只到「数学规格」层，**不含** tiling、UB 预算、访存对齐、跨块规约策略、grid 约束、`autograd.Function` 封装等工程实现细节。把这些数学规格翻译成可运行的 Triton/Ascend C 反向 kernel，需由生成 skill（`triton-op-dev` / `ascendc-op-dev`）承接——trace 是它的输入规格，不是替代品。

---

## 第四部分：精度阈值

| 数据类型 | rtol | atol | 说明 |
| --------- | :---: | :---: | ------ |
| float32 | 1e-5 | 1e-5 | 严格精度（训练） |
| float16 | 1e-3 | 1e-3 | 通用基线 |
| bfloat16 | 2e-3 | 1e-2 | 精度略低 |
| int8 / int32 | 0 | 0 | 整数精确匹配 |
| fp8（e4m3/e5m2） | 1e-2 | 1e-1 | 低精度放宽 |

> 反向链式误差会累积，规约 / softmax / norm 类反向 fp16 下 rtol 可放宽至 5e-3，测试中注明原因。
>
> 非线性函数近似误差：golden 的 `torch.rsqrt` / `torch.exp` 是准确实现，而 Triton 的
> `tl.rsqrt` 等可能是硬件近似（相对误差 ~1e-2），导致反向比对出现 ~1e-2 量级的相对误差。
> 此类误差属预期，报告中注明来源即可，不必当 bug 追。

---

## 第五部分：校验流程

```text
输入（Triton / NumPy / 自然语言）
        │  归一化
        ▼
标准 torch 实现（前向可读）
        ├─ 前向输出 = 前向 golden
        └─ autograd → .grad = 反向 golden
        │
        ▼
比对目标算子实现（前向 allclose / 反向 gradcheck 或逐项 allclose）
```

- 前向：`torch.allclose(实际, 前向 golden, rtol, atol)`，失败时输出最大差异位置与超限比例（断言工具复用 `ops-test-gen` 的 `assert_precision`）。
- 反向：`gradcheck` 或逐项 `allclose(实际梯度, 反向标杆, rtol, atol)`。
- 报告：记录 shape×dtype 矩阵、每个组合的正向/反向误差、最大差异位置。

---

## 注意事项

- **反向标杆只信 torch 自动求导**：不要手写梯度公式当作 golden（手写公式是被验证对象）。
- **规约类反向的累加必须 fp32**：权重/偏置梯度（`d_weight = Σ dw`）沿 tokens/heads 规约，累加 buffer 用 bf16 会随规约长度累积误差（实战：16384 token 下误差从 0 涨到 0.5）；用 fp32 累加、最后一次性转回目标 dtype。
- **归一化先于一切**：Triton/NumPy 输入必须先转成可读 torch 实现，语义不清时先澄清。
- **标杆不是算子实现**：本 Skill 只产出「正确数值基准」，不负责目标算子（Triton/Ascend C）的性能与实现。
- **与 `ops-test-gen` 分工**：本 Skill 生成前向/反向标杆；`ops-test-gen` 用它构造测试套件（shape×dtype 矩阵、边界、断言、回归）。

---

## 参考信息

- [PyTorch autograd.Function 文档](https://pytorch.org/docs/stable/notes/extending.html)
- [PyTorch gradcheck 文档](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html)
- [PyTorch 与 NumPy 对照](https://pytorch.org/docs/stable/numpy.html)
- [PyTorch testing 文档](https://pytorch.org/docs/stable/testing.html)

### 相关 Skill

| Skill | 关系 |
| ------ | ------ |
| `ops-test-gen` | 下游 — 用本 Skill 的前向/反向标杆构造精度测试套件与断言 |
| `triton-op-dev` | 下游 — Triton 算子前向/反向的梯度比对基准 |
| `ascendc-op-dev` | 下游 — Ascend C 算子前向/反向的梯度比对基准 |
