---
name: tilelang-op-dev
description: >-
  This skill should be used when the user asks to generate, compile, debug, or
  optimize a TileLang operator for Ascend NPU from a design document or an
  existing GPU/PyTorch reference. It covers @tilelang.jit kernel writing with
  T.Kernel / T.copy / T.gemm / T.Pipelined / T.Parallel, Ascend hardware
  constraint compliance, and performance tuning. Two backends are covered:
  tilelang-mlir-ascend on Atlas A2/A3 (arch22) and tilelang built with USE_ASCEND
  on Ascend 950 (arch35). Typical triggers include "用 TileLang 写个算子",
  "生成 TileLang GEMM kernel", "这个 TileLang kernel 太慢", "把这段 CUDA/Triton
  改成 TileLang", "TileLang 编译报错怎么排查".
version: 1.0.0
domain: ops
function: tilelang
tags:
  - tilelang
  - code-generation
  - optimization
  - gemm
  - npu
author: mindspeed-agent
compatibility: "tilelang >= 0.1.15, numpy == 1.26.4, ml-dtypes < 0.5, Python 3.12, CANN >= 9.0.0 (Ascend 950) / CANN 8.3+ (A2/A3 tilelang-mlir-ascend)"
targets-repo:
  - tilelang
  - tilelang-mlir-ascend
---

# TileLang 算子开发与性能优化

## 功能概述

本 Skill 覆盖算子开发三阶段流水线中的**阶段二**：按上游 `tilelang-design` 的详细设计文档生成 NPU 可运行的 TileLang 算子代码，并通过系统性优化使算子达到生产级性能。

```text
┌────────────────────────────┐   ┌────────────────────────────┐   ┌──────────────────────┐
│ 阶段一：需求分析与详设        │ → │ 阶段二：代码生成与优化       │ → │ 阶段三：精度性能验收    │
│ tilelang-design           │   │ tilelang-op-dev（本 Skill）│   │ ops-test-gen          │
└────────────────────────────┘   └────────────────────────────┘   └──────────────────────┘
```

本 Skill 是 `ops-implement-agent` 与 `ops-optimize-agent` 在 `dev_lang = tilelang` 路径下的核心能力。

## 适用场景

- 从详细设计文档生成 TileLang 算子代码
- GPU Triton / CUDA / PyTorch → TileLang 迁移
- 自然语言 / PyTorch / NumPy → TileLang 算子实现
- 性能不达标的 TileLang 算子优化（延迟、吞吐、UB 溢出、pipeline 利用率低）
- Autotune 自动搜索最优 tiling 与 num_stages

## 支持的输入形式

| 输入形式 | 支持程度 |
| --------- | :---: |
| 详细设计文档（来自阶段一） | ✅ 完整 |
| GPU Triton / CUDA 实现 | ✅ 完整（按语义重写为 tile-based） |
| PyTorch 实现 | ✅ 完整 |
| 自然语言描述 / NumPy 公式 | ✅ 完整 |

---

## 阶段 0：环境确认（必做）

生成代码前先确认目标环境可用，并据此选择后端路线。完整安装/构建步骤与环境变量见 [references/ascend-env-setup.md](references/ascend-env-setup.md)。

```python
import torch, torch_npu
print("NPU available:", torch.npu.is_available())

try:
    import tilelang
    print("tilelang version:", getattr(tilelang, "__version__", "unknown"))
except ImportError:
    print("tilelang 未安装")
```

**后端分流**（来自设计文档的 `target_arch`）：

| 目标芯片 | 后端 | 探测方式 |
| ------ | ------ | ------ |
| A2/A3（arch22） | tilelang-mlir-ascend | `import tilelang` 走 MLIR 适配器，需 CANN + `ACL_OP_INIT_MODE=1` |
| 950（arch35） | tilelang（`USE_ASCEND=ON`） | 主仓 Ascend 后端，需 CANN 9.2+ + `torch_npu` |

---

## 第一部分：代码生成

### 步骤一：收集需求

确认算子功能、I/O（shape/dtype/layout）、精度、性能约束；缺失则主动询问。信息严重缺失、或算子含 TileLang 无法表达的核心操作（递归、复杂原子、动态线程通信）时，回退 Ascend C / Triton。

### 步骤二：生成 TileLang 算子代码

#### 2.1 GEMM / Matmul 模板

```python
import torch
import tilelang
from tilelang.ascend import language as T   # 昇腾专用 language（含 SimdVF/SimtVF）

@tilelang.jit(out_idx=[-1], target="ascend")
def gemm(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        grid_m = T.ceildiv(M, block_M)
        grid_n = T.ceildiv(N, block_N)
        # 昇腾 T.Kernel 是一维 AI-core grid，二维 tile 展平到一维
        with T.Kernel(grid_m * grid_n) as bx:
            by = bx // grid_n
            bxx = bx % grid_n
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bxx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            # fp32 累加器 → fp16 输出必须在 VF 块内完成 cast（DMA 不允许隐式类型转换）
            C_cast = T.alloc_fragment((block_M, block_N), dtype)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(block_M, block_N):
                    C_cast[i, j] = C_local[i, j].astype(dtype)
            T.copy(C_cast, C[by * block_M, bxx * block_N])
    return main

def launch(A: torch.Tensor, B: torch.Tensor):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    kernel = gemm(M, N, K, 128, 128, 32)
    return kernel(A, B)   # out_idx=[-1] → 内核自分配并返回输出 C
```

> **生产 GEMM 建议**：手写 `T.gemm` 的输出 fragment 在昇腾是 fractal 布局，epilogue 落盘需处理
> 布局变换，易踩「Cannot assign task / scalar-register inputs unavailable」错误。生产环境优先用内置
> `tilelang.ascend.op.gemm`（`tilelang.ascend.op.gemm.gemm_mad` 已处理 fractal layout 与 epilogue）。

#### 2.2 Element-wise 模板

```python
import torch
import tilelang
from tilelang.ascend import language as T

@tilelang.jit(out_idx=[-1], target="ascend")
def add_kernel(N, block_N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N)) as bx:
            A_shared = T.alloc_shared((block_N,), dtype)
            B_shared = T.alloc_shared((block_N,), dtype)
            C_shared = T.alloc_shared((block_N,), dtype)
            T.copy(A[bx * block_N], A_shared)
            T.copy(B[bx * block_N], B_shared)
            # T.Parallel 必须包在 VF 块内（昇腾语义检查强制）
            with T.SimtVF(threads=128):
                for i in T.Parallel(block_N):
                    C_shared[i] = A_shared[i] + B_shared[i]
            T.copy(C_shared, C[bx * block_N])
    return main

# 调用：kernel = add_kernel(N, 1024); C = kernel(A, B)  # out_idx=[-1] → 返回 C
```

#### 2.3 编码原则（昇腾专用，务必遵守）

- **import 用 `from tilelang.ascend import language as T`**（含 `SimdVF`/`SimtVF`），不要用顶层 `tilelang.language`
- **`@tilelang.jit(out_idx=[-1], target="ascend")` 直接装饰返回 PrimFunc 的外层函数**，不要套一层零参 `kernel()` 包装
- **`T.Kernel(N)` 是一维 AI-core grid，无 `threads` 参数**；二维 tile grid 需展平到一维（`by = bx // grid_n; bxx = bx % grid_n`）
- **线程并行用 `T.SimtVF(threads=...)`、寄存器级 SIMD 用 `T.SimdVF()`**；`T.Parallel` 必须包在 VF 块内
- **DMA（`T.copy`）不允许隐式 dtype cast**；fp32 累加器 → fp16 输出须在 VF 块内 `astype` 后再 `T.copy`
- 全局↔片上数据经 `T.copy`；片上计算经 `T.gemm`（输入 `shared`、累加器 `fragment`）
- 累加器 `T.alloc_fragment` + `T.clear` 初始化，dtype 用 fp32
- 循环分块用 `T.Pipelined`，`num_stages` 默认 2–3
- 所有 shape/tiling 参数以 `@tilelang.jit` 实参传入（compile-time 特化），禁止 runtime 可变

### 步骤三：昇腾 NPU 适配

- 输入输出 tensor 设备为 `npu`（`torch.rand(..., device="npu")`）
- `T.Kernel` 是**一维 AI-core grid**：单维算子直接 `T.Kernel(ceildiv(N, block_N))`；二维算子（GEMM 等）把 tile grid 展平到一维（`by = bx // grid_n; bxx = bx % grid_n`）
- 线程并行用 `T.SimtVF(threads=...)`、SIMD 用 `T.SimdVF()`，`T.Parallel` 必须在其内部
- `T.gemm` 输入 `shared`、累加器 `fragment`，M/N/K tiling 对齐后端 Cube 粒度；epilogue 的 fp32→fp16 cast 在 VF 块内完成
- dtype：forward fp16/bf16，累加 fp32

### 步骤四：精度验收

| 数据类型 | rtol | atol |
| --------- | :---: | :---: |
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 2e-3 | 1e-2 |

复用 `ops-test-gen` 的 `assert_precision()` + `get_tolerance()`；正式验收走完整 L1–L5 套件。

---

## 第二部分：性能优化

### 步骤五：瓶颈定位

首选 `torch_npu.profiler.profile(Level1)` → `kernel_details.csv`（见 `ops-profiling`）。按症状分类：

| 症状 | 可能原因 | 手法 |
| ------ | ------ | ------ |
| 编译失败 / UB 溢出 | 片上 tile 过大 | 减小 BM/BK/BN，或关 double buffer |
| 延迟远高于 PyTorch | pipeline 未重叠 | 调 `num_stages`，检查 `T.Pipelined` |
| GPU 直迁性能差 | grid 策略不当 | 展平一维 grid / 调整 tiling |
| 搬运占比高（MTE2） | `T.copy` 离散 / 未对齐 | 连续访存 + 尾轴对齐 |
| 计算占比高（Vector/Cube） | tiling 非最优 | autotune 搜索 BM/BN/BK、num_stages |

### 步骤六：优化手法

1. **Pipeline 调优**：`num_stages` 在 2–4 间搜索，平衡 UB 占用与搬运/计算重叠
2. **Tiling autotune**：`@tilelang.jit` 支持传不同 `block_M/block_N/block_K/num_stages`，穷举候选后取最优
3. **片上复用**：`T.alloc_shared` buffer 生命周期不重叠时复用；`T.alloc_fragment` 累加器常驻
4. **dtype 降精度**：forward fp16/bf16，仅累加 fp32
5. **尾轴对齐**：K/N 尾轴 pad 到后端对齐粒度，避免自动 padding 膨胀
6. **epilogue 融合**：激活/归一化/量化在 `T.Parallel` 内完成，省一次 GM 往返

### 步骤七：调试

- 编译报错 → 先确认后端路线正确（A2/A3 走 MLIR 适配器、950 走主仓），再看 `ACL_OP_INIT_MODE` / `TILELANG_ASCEND_MODE`（见 references）
- UB 溢出 → 减小片上 tile 或 `num_stages`
- 精度异常 → 检查累加器 dtype（须 fp32）、`T.clear` 是否缺失
- 数值 NaN/Inf → 检查 mask / pad / 分母零

---

## 注意事项

- 每次只改一个维度（tiling / num_stages / dtype），改后重测精度
- TileLang 是 compile-time 特化，tiling 参数变化会触发重编译，缓存按参数 key 区分
- 跨芯片 UB 差异：950（256KB）→ A2/A3（192KB）迁移需重新核算片上预算

## 参考信息

- [TileLang 主仓库](https://github.com/tile-ai/tilelang)
- [tilelang-mlir-ascend（A2/A3）](https://github.com/tile-ai/tilelang-mlir-ascend)
- [TileKernels（DeepSeek TileLang 算子库）](https://github.com/deepseek-ai/TileKernels)
- [构建与环境配置](references/ascend-env-setup.md)

## 参考知识库（references/）

| 参考文件 | 内容 |
| ------ | ------ |
| [ascend-env-setup.md](references/ascend-env-setup.md) | tilelang-mlir-ascend（A2/A3）与 tilelang（950）的安装/源码构建、环境变量、算子支持矩阵 |
