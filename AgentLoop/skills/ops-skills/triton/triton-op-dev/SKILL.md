---
name: triton-op-dev
description: >-
  This skill should be used when the user asks to generate, optimize, or develop
  Triton operator implementations for Ascend NPU — from writing new kernels and
  migrating GPU Triton code, to resolving UB overflow or coreDim limit issues,
  improving kernel latency and throughput, and applying tiling, memory access
  optimization, and autotune strategies. Typical triggers include "generate a
  Triton kernel for...", "write a Triton operator to...", "optimize this Triton
  kernel", "fix UB overflow", "this kernel is too slow", or "improve Ascend NPU
  utilization for this operator".
version: 1.0.0
domain: ops
function: triton
tags:
  - triton
  - ascend
  - operator
  - code-generation
  - optimization
  - cann
author: mindspeed-agent
compatibility: MindSpeed >= 0.1.0, CANN >= 9.1.0, Triton >= 3.2.2
targets-repo:
  - triton-ascend
---

# Triton 算子开发与性能优化

## 功能概述

本 Skill 覆盖算子开发三阶段流水线中的**阶段二**全部工作：按上游详细设计文档生成 NPU 适配的 Triton 算子代码，并通过系统性优化使算子达到生产级性能。

## 定位：三阶段流水线

```text
┌──────────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────┐
│  阶段一：需求分析与详设         │ →  │  阶段二：算子开发与优化     │ →  │  阶段三：精度性能验收    │
│  ops-design           │    │  triton-op-dev（本 Skill） │    │  ops-test-gen           │
└──────────────────────────────┘    └──────────────────────────┘    └──────────────────────┘
```

本 Skill 是 `ops-implement-agent` 的核心能力，接收详设文档，产出可运行的算子代码 + 性能报告。

## 适用场景

- 从详细设计文档生成 Triton 算子代码
- GPU Triton kernel → 昇腾 NPU 迁移适配
- 自然语言/PyTorch/NumPy → Triton 算子实现
- 性能不达标的算子优化（延迟、吞吐、UB 溢出、coreDim 超限）
- Autotune 自动搜索最优 BLOCK_SIZE 和编译选项
- Kernel Hang / Timeout 排查

## 支持的输入形式

| 输入形式 | 支持程度 | 说明 |
| --------- | :---: | ------ |
| 详细设计文档（来自阶段一） | ✅ 完整 | 按设计方案编码，效率最高 |
| GPU Triton 实现 | ✅ 完整 | 自动适配 NPU 硬件约束 |
| 自然语言描述 | ✅ 完整 | 从零生成 |
| PyTorch 自定义算子 | ✅ 完整 | 提取计算语义 → Triton |
| NumPy / 数学公式 | ✅ 完整 | 直接映射为 Triton 原语 |
| CUDA 源码（`.cu`） | ⚠️ 算法参考 | 不做行对行翻译 |
| Ascend C 算子 | ❌ 反向迁移 | 使用对应 Skill |

---

## 第一部分：昇腾 NPU 硬件约束速查

以下约束是 Triton 算子开发和优化的硬限制，必须在编码和调优全过程中遵守。

### 一、Grid 总量约束

- **规则**：`grid_x × grid_y × grid_z ≤ 65535`（UINT16_MAX）
- **超限错误**：`coreDim=xxxx can't be greater than UINT16_MAX`
- **解决方案（按优先级）**：
  1. 增大 BLOCK_SIZE，使 `BLOCK_SIZE ≥ ceil(N / 65535)`，取 `triton.next_power_of_2()`
  2. 设置 `export TRITON_ALL_BLOCKS_PARALLEL=1`（无顺序依赖时），该环境变量会**自动调整 logical core 数匹配物理 core 数**，使 grid 可超过 65535
  3. 配合 `TRITON_ALL_BLOCKS_PARALLEL` 使用时，可在 kernel 的 `triton.Config` 或 launch 参数中指定 `auto_blockify_size=N` 控制 block 大小
  4. 引入 `BLOCK_SIZE_SUB` 二次切分（见下方复合约束）

### 二、UB（Unified Buffer）约束

| 芯片系列 | UB 总大小 | doublebuffer 后 | 典型产品 |
| --------- | :---: | :---: | ------ |
| A2 / A3 | 192KB | ~96KB | Atlas 800T/I A2/A3 |
| A5 | 256KB | ~128KB | Ascend 950 系列 |

- **超限错误**：`ub overflow, requires xxxx bits while <N> bits available!`
- **跨芯片注意**：A5 → A2/A3 迁移时 UB 缩小，原通过的算子可能溢出
- **解决方案**：减小 BLOCK_SIZE / 引入 BLOCK_SIZE_SUB / 减少中间变量 / `compile_hint("mayDiscretememaccess")`

### 三、Grid + UB 复合约束

增大 BLOCK_SIZE 满足 grid → 反向触发 UB 溢出。使用 **BLOCK_SIZE_SUB 二次切分**：

```python
MAIN_BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(N, 65535))
SUB_BLOCK_SIZE = 1024
grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)

@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    for sub in range(tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)):
        offs = base + sub * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
        ...
```

### 四、内存对齐约束

| 算子类型 | 对齐要求 | 说明 |
| --------- | --------- | ------ |
| VV（纯向量算子） | 尾轴 32 字节对齐 | 仅使用 Vector Core |
| CV（Cube-Vector 融合） | 尾轴 512 字节对齐 | 包含 `tl.dot` |

未对齐时编译器自动 padding 会显著降低性能。**借轴转置**：当 `numel() % 256B == 0` 时，将多维视为 1D 以满足对齐。

### 五、Grid 核心分配策略

优先将 grid 固定为 NPU 物理核心数，kernel 内部用循环分发：

```python
import torch_npu, triton.runtime.driver as driver
props = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
NUM_CORE = props["num_vectorcore"]   # 纯向量算子
# NUM_CORE = props["num_aicore"]    # 含 tl.dot
grid = (NUM_CORE,)

@triton.jit
def kernel(...):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(N, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        ...
```

**核心数限制**（调试时限制算子使用的核心数）：

```bash
# 限制 AI Core 和 Vector Core 数量（逗号分隔，如 14 个 AI Core + 28 个 Vector Core）
export NPU_DEVICE_LIMIT=14,28
```

**Ascend 950 compile_mode**（A5 系列新增）：

```python
# 通过 triton.Config 或 launch meta-parameter 控制 SIMD/SIMT 编译模式
kernel[grid](..., compile_mode="simd")                # 纯 SIMD 向量化
kernel[grid](..., compile_mode="unstructured_in_simt") # 默认，SIMT 中混合非结构化 SIMD
kernel[grid](..., compile_mode="simt_only")            # 纯 SIMT，不尝试 SIMD
```

### 六、其他硬约束

| 约束项 | 说明 |
| ------- | ------ |
| **指针别名** | 禁止同一 tensor 作为多个不同指针参数传入 kernel |
| **控制流 + 内存访问** | 不要在不同分支中构造不同基地址的指针后在分支外做 `tl.load`/`tl.store` |
| **Autotune BLOCK_SIZE** | 候选值必须为 2 的幂 |
| **max_autotune kernel_type** | `"vector"` / `"cube"` / `"mix"` |
| **D-cache 异常** | 所有输入 tensor 必须在 NPU 设备上 |
| **循环迭代变量** | 不要直接作为访存偏移量，先计算偏移再传入 |

---

## 第二部分：算子代码生成

### 步骤一：收集需求

确认：算子功能、输入输出（shape/dtype/layout）、精度要求、性能约束。如有缺失主动询问。

#### 无法生成的场景

| 场景 | 原因 | 替代方案 |
| ------ | ------ | --------- |
| 信息严重缺失（shape/dtype 未知） | 无法设计 tiling | 请用户补充 |
| Grid+UB 夹逼无解 | 物理上限冲突 | 拆分为多 kernel |
| Triton 不支持的核心操作 | 编程模型限制 | 改用 Ascend C |
| 外部依赖/状态 | kernel 是无状态纯计算 | Host 侧管理 |
| CANN 版本过旧 | API 不可用 | 升级 CANN |

### 步骤二：生成 Triton 算子代码

#### 2.1 代码模板

```python
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    input_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """算子功能简述"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask)
    output = x  # 替换为实际计算
    tl.store(output_ptr + offsets, output, mask=mask)


def launch(input_tensor):
    N = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    kernel_name[grid](input_tensor, input_tensor, N, BLOCK_SIZE=BLOCK_SIZE)
    return input_tensor
```

#### 2.2 编码原则

- BLOCK_SIZE 首选 1024，在 [MIN_BLOCK_SIZE, MAX_BLOCK_SIZE_BY_UB] 内取 2 的幂
- 内存合并访问：连续线程访问连续内存
- 掩码处理：边界使用 mask 防越界
- 常量折叠：`tl.constexpr` 标注编译期常量

### 步骤三：昇腾 NPU 适配

#### 3.1 Python 侧设备适配

```python
import torch_npu
x = torch.rand(size, device='npu')  # cuda → npu
```

#### 3.2 Grid 与核心分配

- GPU 大 grid → NPU 固定物理核心数 + 内部循环（见第一部分 §五）
- 优先 1D grid（2D 在 NPU 内部合并为 1D）

#### 3.3 单核数据与计算

- BLOCK_SIZE 在 UB 上限内尽可能大
- 优先 float16/bf16，累加用 float32
- `tl.dot` 检查 M/N/K tiling、累加器 dtype

#### 3.4 Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}) for bs in [128, 256, 512, 1024, 2048]
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

昇腾进阶 `max_autotune`（参见第三部分 §六）。

### 步骤四：精度验收与单元测试

| 数据类型 | rtol | atol | 说明 |
| --------- | :---: | :---: | ------ |
| float32 | 1e-5 | 1e-5 | 严格精度 |
| float16 | 1e-3 | 1e-3 | 通用基线 |
| bfloat16 | 2e-3 | 1e-2 | 精度略低 |
| int8/int32 | 0 | 0 | 精确匹配 |
| fp8 | 1e-2 | 1e-1 | 低精度放宽 |

> 累加密集型算子（softmax / large reduction）fp16 下 rtol 可放宽至 5e-3。

验收检查清单：

- [ ] 编译通过，无 MLIR/SCCL 错误
- [ ] 所有 shape × dtype 精度通过
- [ ] 大 shape 通过（grid ≤ 65535，无 UB 溢出）
- [ ] 边界场景通过（单元素、全零、极值）
- [ ] 空张量不崩溃
- [ ] 整数算子额外验证 int32/int64（atol=rtol=0）
- [ ] CV 算子额外验证 K 维度奇数、非 512B 对齐

---

## 第三部分：性能优化

> **前置依赖**：算子已通过正确性验证。优化是迭代过程——每次只改一个维度，测量对比后再进入下一轮。

### 优化工作流

```text
性能采集 → 瓶颈定位 → 优化实施 → 精度验收 → 迭代
```

### 步骤五：瓶颈定位

#### 5.1 获取基线

```python
import torch, torch_npu

def quick_bench(fn, *args, warmup=10, repeat=100):
    for _ in range(warmup): fn(*args)
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat): fn(*args)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / repeat  # ms

triton_ms = quick_bench(lambda: launch_kernel(x, y))
torch_ms = quick_bench(lambda: reference_impl(x, y))
print(f"Triton: {triton_ms:.4f}ms | Torch: {torch_ms:.4f}ms | Speedup: {torch_ms/triton_ms:.2f}x")
```

#### 5.2 瓶颈分类

| 症状 | 可能原因 | 跳转到 |
| ------ | --------- | -------- |
| 编译：`ub overflow` | UB 超限 | 步骤六 |
| 编译：`coreDim > UINT16_MAX` | grid 超 65535 | 步骤五 |
| 延迟远高于 PyTorch | tiling 或内存访问差 | 步骤六、七 |
| GPU 直迁性能差 ≥ 3× | Grid 策略不当 | 步骤五 |
| 小 shape 性能差 | 启动开销大 | 步骤五 |
| 特定 shape 骤降 | 尾轴未对齐 | 步骤七 |
| 延迟中等但未达预期 | BLOCK_SIZE 非最优 | 步骤九 |

#### 5.3 Profiling 采集

> 首选 `torch_npu.profiler.profile(Level1)` → `kernel_details.csv`（见 ops-profiling skill）。
> 以下 msprof 方式作为补充参考。

```bash
msprof --output=./prof_output --application="python test_kernel.py"
# op_summary_*.csv: aiv_mte2_ratio 高→搬运瓶颈 / 低→计算瓶颈

# 指定算子 + 硬件指标
msprof op --output=./prof_output --application="python test_kernel.py" \
    --kernel-name=my_kernel --aic-metrics=MemoryDetail
```

#### 5.4 mssanitizer 内存检测

```bash
export TRITON_ENABLE_SANITIZER=true
mssanitizer python test_kernel.py
```

### 步骤六：Grid 与核心分配优化

**P1 优先级**：GPU 风格 `grid = (ceil(N/BLOCK_SIZE),)` → 固定物理核心数 + 步长分发（见第一部分 §五），收益 2×–10×。

**coreDim 超 65535**：

```python
min_block = triton.next_power_of_2(triton.cdiv(N, 65535))
BLOCK_SIZE = max(1024, min_block)
# 增大后 UB 溢出 → BLOCK_SIZE_SUB
```

### 步骤七：Tiling 与 UB 优化

#### 7.1 BLOCK_SIZE 调优

- 在 UB 上限内尽可能大，首选 1024/2048
- 候选 2 的幂：128, 256, 512, 1024, 2048, 4096...
- 通过 autotune 搜索（步骤九）

#### 7.2 BLOCK_SIZE_SUB（UB 溢出修复，P0）

参见第一部分 §三。二次切分控制单次循环的 UB 占用。

#### 7.3 减少 UB 占用的其他手段

| 手段 | 效果 | 场景 |
| ------ | ------ | ------ |
| 减少中间变量 | 直接释放 UB | 可合并的临时张量 |
| 复用已分配 buffer | 避免重复分配 | 生命周期不重叠 |
| 拆分复杂 kernel | 每个 kernel UB 降低 | 粗粒度算子 |
| 更小 dtype（fp16 替 fp32） | 数据量减半 | 精度允许 |

### 步骤八：内存访问模式优化

#### 8.1 尾轴对齐

未对齐触发自动 padding。**借轴转置**：当 `numel() % 256B == 0` 时，视为 1D 满足对齐。

#### 8.2 离散访存 → 聚集访存

```python
# 优化前：离散访存
val = tl.load(x_ptr + idx * stride_x, mask=mask)

# 优化后：先搬入 UB 再 gather
x_shared = tl.load(x_ptr + tl.arange(0, M) * stride_x)  # 大批量
val = tl.gather(x_shared, idx, 0)                        # UB 内挑选
```

#### 8.3 block_ptr shape/stride 优化

体现实际内存布局，使每个 block 访问连续内存：

```python
# 好：2D 视图体现连续维度
block_ptr = tl.make_block_ptr(
    base=ptr, shape=(1024, 32), strides=(32, 1),
    block_shape=(BT, 32), order=(1, 0)
)
```

#### 8.4 隐式转置加载

利用 `strides` 和 `order` 在加载时完成转置，省去显式 `tl.trans`：

```python
x_ptr_t = tl.make_block_ptr(
    base=x_ptr, shape=(N, M), strides=(stride_xn, stride_xm),
    offsets=(bn, bm), block_shape=(BLOCK_N, BLOCK_M), order=(1, 0)
)
x_tile = tl.load(x_ptr_t)  # 直接以转置视图加载
```

#### 8.5 compile_hint("mayDiscretememaccess")

离散访存触发 32B 对齐 padding 导致 UB 膨胀时：

```python
# triton-ascend ≥ 3.4.0
import triton.language.extra.cann.extension as extension
value = tl.load(pointer)
extension.compile_hint(value, "mayDiscretememaccess")

# triton-ascend ≤ 3.2.0
value = tl.load(pointer)
tl.compile_hint(value, "mayDiscretememaccess")
```

### 步骤九：计算与数据类型优化

#### 9.1 数据类型选择

| 场景 | 推荐 dtype | 理由 |
| ------ | ----------- | ------ |
| 前向计算 | float16/bf16 | 带宽减半，NPU 原生高效 |
| 累加中间结果 | float32 | 避免精度损失 |
| tl.dot 输入 | float16 | Cube Core 原生支持 |
| tl.dot 累加器 | float32 | 精度保障 |

#### 9.2 向量化比较——i64/i32 → fp32

NPU 上 i64/i32 比较降级为标量，转换为 fp32 启用 SIMD：

```python
# 优化前（标量）
mask = cols < N  # int64 cols，标量比较

# 优化后（向量化）
cols_cmp = cols.to(tl.float32)   # int64 → fp32
mask = cols_cmp < N               # SIMD 比较
```

#### 9.3 tl.dot 优化（CV 算子）

- M/N/K tiling 对齐 Cube Core 分块粒度
- K 维度较大时循环分片控制数据量
- 考虑 `tile_mix_vector_loop` 平衡 Cube/Vector 流水线

### 步骤十：Autotune 自动调优

#### 10.1 基础 autotune（BLOCK_SIZE）

```python
@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': s}) for s in [128, 256, 512, 1024, 2048]],
    key=['n_elements'],
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

#### 10.2 max_autotune（昇腾全维度搜索）

```python
from triton.backends.ascend.runtime import max_autotune

@max_autotune(
    configs=[triton.Config({'BLOCK_SIZE': s}) for s in [128, 256, 512, 1024, 2048]],
    key=['n_elements'],
    kernel_type="vector",  # "vector" / "cube" / "mix"
    num_stages=[1, 2],
    enable_ubuf_saving=[True, False],
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr, **META):
    ...
```

`max_autotune` 是对 `@triton.autotune` 的增强包装，自动将 base config 与各 tuning 参数做**笛卡尔积展开**后送给标准 `autotune` 执行。它**不替代** autotune，而是叠加在上面。

##### kernel_type 与支持的 tuning 参数

| `kernel_type` | 场景 | 支持的参数（默认值） |
| -------------- | ------ | --------------------- |
| **`"vector"`** | 纯向量算子 | `num_stages=[1, 2]`、`enable_ubuf_saving=[True]` |
| **`"cube"`** | 纯 Cube 算子 | `num_stages=[1, 2]`、`unit_flag=[False]`、`limit_auto_multi_buffer_of_local_buffer=["no-l0c"]` |
| **`"mix"`** （默认） | Cube+Vector 融合 | 上述全部 + `limit_auto_multi_buffer_only_for_local_buffer=[False]`、`set_workspace_multibuffer=[2, 4]`、`enable_hivm_auto_cv_balance=[True]`、`tile_mix_vector_loop=[2, 4]`、`tile_mix_cube_loop=[2, 4]` |

> **参数值必须是 list/tuple**。标量会触发校验错误。不支持的参数（如 vector kernel 传 `tile_mix_vector_loop`）会打印 warning 并忽略。

##### 参数优先级（从高到低）

1. `max_autotune` 的显式 kwarg（如 `num_stages=[1, 2]`）
2. base `Config` 的 `kwargs` 中已固定的值（如 `triton.Config({'enable_ubuf_saving': False})` 将阻止该参数展开）
3. 默认值表（上表）

> 想固定某个参数不参与搜索 → 在 base config 的 `kwargs` 中设为单值。不想固定 → 在 `max_autotune` kwarg 中传 list，或不传让它用默认 list 展开。

```bash
export TRITON_PRINT_AUTOTUNING=1  # 打印最优参数
```

### 步骤十一：编译优化选项

> **重要**：算子代码中**不应直接使用 AscendNPU-IR 的内部 `compile_options` 参数**。
> `compile_options` 是 BiSheng 编译器（AscendNPU-IR）的内部接口，Triton Ascend 的 `JITFunction.__call__` **不接受此 kwarg**（标准 Triton 社区版会直接抛 `KeyError`，我们在 xuxu 容器上实测验证了这一点）。
> AscendNPU-IR 官方文档也**未将 `compile_options` 列为对外接口**。所有编译器选项均通过 `triton.Config` 或 launch meta-parameter 传入。

#### 11.1 编译选项的正确传入方式

Triton Ascend 支持两类传入方式：

**方式 A：`triton.Config`（推荐，配合 autotune）**

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 2048}, multibuffer=False),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(...):
    ...
```

**方式 B：Launch meta-parameter（直接调用时）**

```python
kernel[grid](x, y, out, N, BLOCK_SIZE=1024,
    multibuffer=True,
    inject_barrier_all=True,
)
```

> 两种方式等效。autotune 场景用方式 A，直接调用场景用方式 B。

#### 11.2 可用编译选项速查

##### 通用选项

| 选项 | 类型 | 默认值 | 说明 |
| ------ | ------ | :---: | ------ |
| `multibuffer` | bool | `True` | **启用 doublebuffer / ping-pong 流水线**（等价于 AscendNPU-IR 的 `enable-auto-multi-buffer`） |
| `num_stages` | int | `2` | 流水线级数（`1` 或 `2`），通过 autotune 搜索最优值 |
| `enable_ubuf_saving` | bool | `True` | UB buffer 复用节省 |
| `compile_mode` | str | `"unstructured_in_simt"` | 编译模式：`"simd"` / `"unstructured_in_simt"` / `"simt_only"`（Ascend 950+） |
| `auto_blockify_size` | int | `1` | 自动 blockify 的 block 大小，配合 `TRITON_ALL_BLOCKS_PARALLEL` 使用 |

##### 同步与屏障

| 选项 | 类型 | 默认值 | 说明 |
| ------ | ------ | :---: | ------ |
| `inject_barrier_all` | bool | `None` | 核内同步 barrier（适用于 mix/aic/aiv kernel） |
| `inject_block_all` | bool | `None` | 核间同步 block barrier（仅 mix kernel） |
| `disable_auto_inject_block_sync` | bool | `None` | 禁用自动 block sync |
| `sync_solver` | bool | `None` | HIVM 同步求解器 |

##### CV 融合优化（mix / cube kernel）

| 选项 | 类型 | 默认值 | 说明 |
| ------ | ------ | :---: | ------ |
| `enable_auto_bind_sub_block` | bool | `None` | 自动子 block 绑定 |
| `enable_hivm_auto_cv_balance` | bool | `True` | 自动 Cube/Vector 负载均衡 |
| `unit_flag` | bool | `False` | Cube 输出同步优化 |
| `tile_mix_vector_loop` | int | `2` | Vector 循环分片数（`2`/`4`/`8`） |
| `tile_mix_cube_loop` | int | `2` | Cube 循环分片数（`2`/`4`/`8`） |

##### Multibuffer 范围控制

| 选项 | 类型 | 默认值 | 说明 |
| ------ | ------ | :---: | ------ |
| `limit_auto_multi_buffer_only_for_local_buffer` | bool | `False` | 仅对 local buffer 开启 multibuffer |
| `limit_auto_multi_buffer_of_local_buffer` | str | `"no-l0c"` | Local buffer multibuffer 范围：`"no-limit"` / `"no-l0c"` |
| `set_workspace_multibuffer` | int | `2` | Workspace multibuffer 级数（`2` / `4`） |

> 以上选项均通过 `triton.Config` 或 launch meta-parameter 传入，**无需也不应使用 `compile_options`**。

#### 11.3 优化目标→正确做法映射

| 优化目标 | ❌ 内部 API（禁止使用） | ✅ Triton Ascend 公开接口 |
| --------- | ---------------------- | -------------------------- |
| 自动 doublebuffer | `compile_options={"enable-auto-multi-buffer": True}` | `multibuffer=True`（默认已开） |
| UB buffer 复用 | `compile_options={"enable-ubuf-saving": True}` | `enable_ubuf_saving=True`（默认已开） |
| 离散访存提示 | `compile_options={"mayDiscretememaccess": True}` | `extension.compile_hint(value, "mayDiscretememaccess")`（见步骤八 §8.5） |
| Cube/Vector 均衡 | — | `enable_hivm_auto_cv_balance=True`（默认已开） |
| 算子融合 | `compile_options={"enable-hfusion-compile": True}` | 通过 `torch.compile` 图编译 |
| 核内同步 | — | `inject_barrier_all=True` |
| 核间同步 | — | `inject_block_all=True` |
| 大 grid 超 65535 | — | `TRITON_ALL_BLOCKS_PARALLEL=1` + `auto_blockify_size=N` |

#### 11.4 兼容性写法（生成算子代码时推荐）

当目标环境的 Triton Ascend 版本不确定时，生成代码应使用兼容性写法。**底线是永远不写 `compile_options`**，而是通过 `triton.Config` 或 launch 参数传入编译选项：

```python
# autotune 场景：通过 triton.Config 传入编译选项
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, multibuffer=True, num_stages=ns)
        for bs in [128, 256, 512, 1024, 2048]
        for ns in [1, 2]
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr, **META):
    ...

# 直接调用场景：通过 launch meta-parameter 传入
kernel[grid](x, y, out, N, BLOCK_SIZE=1024,
    multibuffer=True,
    inject_barrier_all=True,  # 仅在 hang 时启用
)
```

若目标环境不支持 `triton.Config` 中传入编译选项（极老的版本），编译选项会被静默忽略（不抛异常）。核心计算逻辑不受影响。

> **底线**：永远不要将 `compile_options={...}` 写入算子发布代码。该写法在 Triton Ascend 官方文档中**不存在**，是 AscendNPU-IR 编译器的内部接口，不保证跨版本兼容。所有编译优化需求都应通过 `triton.Config` 或 launch meta-parameter 实现。

### 步骤十二：Kernel Hang / Timeout 排查

按顺序尝试，每次只改一个参数：

1. **核内同步**：`inject_barrier_all=True`（适用于 mix/aic/aiv kernel）
2. **核间同步**：`inject_block_all=True`（仅 mix kernel）

```python
kernel[grid](..., inject_barrier_all=True, inject_block_all=True)
```

其他原因：varlen kernel 索引越界、UB 溢出未报错而静默 hang、多核缺少 block sync。

### 步骤十三：精度验收

每次优化后重跑精度。快速诊断用 `torch.allclose`，正式验收用 `ops-test-gen` skill 的完整套件。

---

## 优化模式速查表

| 症状 | 优化手段 | 预期收益 | 参考 |
| ------ | --------- | :---: | :---: |
| 编译：UB overflow | BLOCK_SIZE_SUB 二次切分 | — | 步骤七 |
| 编译：coreDim > 65535 | 增大 BLOCK_SIZE 或 `TRITON_ALL_BLOCKS_PARALLEL=1` | — | 步骤六 |
| 编译：UB overflow（对齐膨胀） | `compile_hint("mayDiscretememaccess")` | — | 步骤八 |
| GPU 直迁性能差 | Grid 固定物理核心数 | 2×–10× | 步骤六 |
| 小 shape 性能差 | 固定核心数，减少启动开销 | 2×–5× | 步骤六 |
| 中大 shape 延迟高 | autotune 搜索 BLOCK_SIZE | 1.5×–3× | 步骤十 |
| 特定 shape 骤降 | 尾轴对齐 + 借轴转置 | 2×–5× | 步骤八 |
| 离散访存慢 | load→UB→gather | 1.5×–3× | 步骤八 |
| 搬运占比高 | 调整 num_stages + 优化 tiling | 1.2×–2× | 步骤七、八 |
| CV 算子性能不达预期 | max_autotune(kernel_type="mix") | 1.5×–3× | 步骤十 |
| 标量降级 | 重构 block_ptr shape/stride | 5×–10× | 步骤八、九 |
| i64/i32 比较慢 | int→fp32 向量化比较 | 2×–5× | 步骤九 |
| 整体吞吐低 | `multibuffer=True` + `num_stages` 调优 | 1.2×–2× | 步骤十一 |
| Kernel Hang | `inject_barrier_all` + `inject_block_all` | — | 步骤十二 |

---

## 注意事项

- **一次只改一个维度**：同时改 BLOCK_SIZE 和 grid 无法判断哪个有效
- **每次改后重测精度**：优化可能改变浮点运算顺序
- **doublebuffer 减半**：估算用有效 UB（A2/A3: ~96KB, A5: ~128KB）
- **跨芯片 UB 差异**：A5→A2/A3 迁移重新核算
- **大 shape 先压 grid**：N > 10⁷ 时 grid 是首要问题
- **小 shape 先压核心数**：N < 10⁴ 时固定核心数
- **2D grid 等效 1D**：`(4,5)` 与 `(20,)` 等效，优先 1D
- **CANN ≥ 9.0**：Triton-Ascend 3.2.1 对应 CANN 9.0.0
- **指针别名禁止**：同一 tensor 不能作为多个不同指针参数
- **已知 UB 浪费模式**：`Not` 操作（~5× 膨胀）、`max_dim0`+int64、`<Nx1xf32>` padding → `<Nx8xf32>`

## 调试环境变量

```bash
TRITON_DEBUG=1                  # dump 文件到 ~/.triton/dump/（支持 2、3 更细粒度）
TRITON_DEVICE_PRINT=1           # 启用 tl.device_print（每线程 GM buffer 最大 16KB）
TRITON_PRINT_AUTOTUNING=1       # 打印 autotune 最优参数及耗时
TRITON_ENABLE_SANITIZER=true    # 内存越界/竞争检测
MLIR_ENABLE_DUMP=1              # dump 每次 MLIR 优化前后的 IR（可设为 kernel 名过滤）
LLVM_IR_ENABLE_DUMP=1           # dump 每次 LLVM IR 优化前后的 IR
MLIR_ENABLE_REMARK=1            # 输出 MLIR 编译 remark（含性能警告）
USE_IR_LOC=1                    # 在生成 IR 中包含源码位置信息
TRITON_REPRODUCER_PATH=./repro  # 生成 MLIR 复现文件
TRITON_ASCEND_COMPILE_SPEED_OPT=1  # JIT 编译器检测到编译失败后跳过后续 pass（加速问题定位）
ENABLE_PRINT_UB_BITS=1          # inductor 场景下打印当前 UB 用量
TRITON_ALWAYS_COMPILE=1         # 忽略编译缓存，强制重编译
TRITON_INTERPRET=1              # 使用 Triton 解释器（kernel 内支持 Python 断点）
```

## 参考信息

### Triton-Ascend

- [Triton-Ascend 官方仓库](https://github.com/triton-lang/triton-ascend)
- [Triton-Ascend 在线文档](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)
- [算子开发指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/programming_guide/index.md)
- [GPU→NPU 迁移指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/migration_guide/migrate_from_gpu.md)
- [Autotune 指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/autotune_guide.md)
- [UB Overflow 排查](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/debug_guide/ub_overflow.md)
- [FAQ](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/FAQ.md)

### AscendNPU-IR（BiSheng 编译器）

> **注意**：AscendNPU-IR 是 Triton Ascend 的底层编译器，其内部 API（如 `compile_options`）**不应直接在算子代码中使用**。
> 算子开发请使用 Triton Ascend 的公开接口（`max_autotune`、`compile_hint`、`inject_barrier_all` 等，详见步骤十一）。
> 以下链接仅作为底层编译原理参考，非算子开发 API 手册。

- [AscendNPU-IR 仓库](https://gitcode.com/Ascend/AscendNPU-IR)
- [编译选项参考](https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/zh_cn/user_guide/compile_option.md)（底层编译选项，非算子对外接口）
- [最佳实践](https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/zh_cn/user_guide/best_practice.md)

### 其他

- [Triton 官方文档](https://triton-lang.org/)
- [CANN 算子开发指南](https://www.hiascend.com/document)

### 相关 Skill

| Skill | 关系 |
| ------ | ------ |
| `ops-design` | 上游 — 提供详细设计文档作为编码依据 |
| `ops-test-gen` | 下游 — 精度和性能测试用例构造 |
