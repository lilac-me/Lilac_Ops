---
name: ops-test-gen
description: >-
  This skill should be used when the user asks to construct precision or
  performance test cases for operators, generate correctness verification or
  latency/throughput benchmarks for Triton / Ascend C kernels on Ascend NPU,
  validate numerical accuracy against PyTorch reference implementations, compare
  operator performance, create parameterized test suites covering shape × dtype
  combinations, or set up automated regression test suites. Typical triggers
  include "生成精度测试用例", "验证算子精度", "生成性能测试用例",
  "这个算子的延迟是多少", "对比性能和参考实现的差距", "构造精度验收测试",
  or "构造性能 benchmark".
version: 1.0.0
domain: ops
function: common
tags:
  - precision
  - performance
  - test-generation
  - correctness
  - benchmark
  - ascend
author: mindspeed-agent
compatibility: MindSpeed >= 0.1.0, CANN >= 9.1.0
targets-repo:
  - triton-ascend
---

# 算子精度与性能测试

## 功能概述

本 Skill 覆盖算子开发三阶段流水线中的**阶段三**全部工作：为昇腾 NPU 算子生成结构化的精度验收测试和性能基准测试用例，覆盖 shape × dtype 参数化矩阵、边界场景、数值稳定性检查、延迟/吞吐量基准、性能回归检测。

## 定位：三阶段流水线

```text
┌──────────────────────────────┐    ┌──────────────────────┐    ┌──────────────────────────┐
│  阶段一：需求分析与详设         │ →  │  阶段二：算子开发与优化  │ →  │  阶段三：精度性能验收        │
│  ops-design           │    │  triton-op-dev         │    │  ops-test-gen（本 Skill）   │
└──────────────────────────────┘    └──────────────────────┘    └──────────────────────────┘
```

本 Skill 是 `ops-implement-agent` 的核心能力，接收算子代码并产出精度和性能测试报告。

## 适用场景

- 新算子开发完成后需要精度验收和性能基线
- 算子迁移（GPU→NPU）后需要验证数值一致性
- 算子优化后需要量化性能提升、确认精度未退化
- 需要构建自动化精度/性能回归测试套件
- 排查特定 shape 下的精度异常或性能骤降

---

## 第一部分：精度测试

### 精度标准

| 数据类型 | rtol | atol | 说明 |
| --------- | :---: | :---: | ------ |
| float32 | 1e-5 | 1e-5 | 严格精度（训练） |
| float16 | 1e-3 | 1e-3 | 通用推理 |
| bfloat16 | 2e-3 | 1e-2 | 精度略低于 fp16 |
| int8 / int32 | 0 | 0 | 整数必须精确匹配 |
| fp8（e4m3/e5m2） | 1e-2 | 1e-1 | 低精度量化 |

> 累加密集型（softmax / large reduction / attention）fp16 下 rtol 可放宽至 5e-3，在测试注释中说明原因。

### 1. 参考实现

```python
import torch

def reference_impl(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """PyTorch 原生 API 等价计算，作为精度 ground truth。
    注意：dtype 一致、计算顺序与 kernel 对齐、固定 seed（如有随机性）。
    """
    return torch.add(x, args[0])  # 替换为实际逻辑
```

### 2. 精度断言工具

```python
def get_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    thresholds = {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (2e-3, 1e-2),
        torch.int32: (0, 0),
        torch.int8: (0, 0),
    }
    return thresholds.get(dtype, (1e-3, 1e-3))


def assert_precision(actual, expected, dtype, shape,
                     rtol_override=None, atol_override=None):
    """精度断言，失败时输出最大差异位置和超限比例。"""
    rtol, atol = get_tolerance(dtype)
    if rtol_override is not None: rtol = rtol_override
    if atol_override is not None: atol = atol_override

    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")

    if torch.allclose(actual, expected, rtol=rtol, atol=atol):
        return

    diff = (actual.float() - expected.float()).abs()
    max_idx = diff.argmax().item()
    exceed_mask = diff > (atol + rtol * expected.float().abs())
    raise AssertionError(
        f"Precision FAILED | shape={shape} dtype={dtype}\n"
        f"  rtol={rtol} atol={atol}\n"
        f"  max_diff={diff.max().item():.6e} at idx={max_idx}\n"
        f"  mean_diff={diff.mean().item():.6e}\n"
        f"  exceed_ratio={exceed_mask.float().mean().item():.2%}"
    )
```

### 3. Shape × Dtype 参数化矩阵

```python
SHAPE_MATRIX = {
    "vector_1d": [
        ("tiny", (1,)), ("small", (128,)), ("medium", (4096,)),
        ("large", (65536,)), ("odd", (999,)), ("prime", (2053,)),
    ],
    "matrix_2d": [
        ("square_s", (128, 128)), ("square_l", (1024, 1024)),
        ("rect_w", (128, 4096)), ("rect_h", (4096, 128)),
        ("odd", (127, 255)), ("unaligned", (513, 1023)),
    ],
    "reduction": [
        ("reduce_last", (256, 512)), ("reduce_first", (512, 256)),
        ("reduce_all", (1024, 1024)), ("odd_dim", (127, 513)),
    ],
}
DTYPE_LIST = [torch.float32, torch.float16, torch.bfloat16]
```

### 4. 边界与特殊值测试

```python
def generate_edge_cases(dtype: torch.dtype) -> list:
    fp16_max = 65504.0
    cases = [
        ("empty", torch.empty(0, dtype=dtype)),
        ("zeros", torch.zeros(1024, dtype=dtype)),
        ("ones", torch.ones(1024, dtype=dtype)),
        ("max_val", torch.full((1024,), fp16_max, dtype=dtype)),
        ("min_val", torch.full((1024,), -fp16_max, dtype=dtype)),
        ("mixed_sign", torch.linspace(-10.0, 10.0, 1024, dtype=dtype)),
        ("tiny_val", torch.full((1024,), 1e-8, dtype=dtype)),
        ("neg_tiny_val", torch.full((1024,), -1e-8, dtype=dtype)),
    ]
    if dtype in (torch.float16, torch.bfloat16):
        cases.append(("subnormal", torch.full((1024,), 1e-9, dtype=dtype)))
    return cases
```

### 5. 数值稳定性检查

```python
def test_numerical_stability(launch_fn, dtype):
    shape = (1024,)
    # 全零 → 不应有 NaN/Inf
    x = torch.zeros(shape, device="npu", dtype=dtype)
    out = launch_fn(x)
    assert not torch.isnan(out).any(), "NaN with zero input"
    assert not torch.isinf(out).any(), "Inf with zero input"
    # 极大值 → 不应溢出
    x = torch.full(shape, 65504.0, device="npu", dtype=dtype)
    out = launch_fn(x)
    assert not torch.isnan(out).any(), "NaN with max value"
    # 负值（如有 sqrt/log）
    x = torch.full(shape, -1.0, device="npu", dtype=dtype)
    out = launch_fn(x)
    # 根据算子语义判断 NaN 是否合理
```

### 6. 精度测试 checklist

- [ ] L1：编译通过，无 MLIR/SCCL 错误
- [ ] L2：所有 shape × dtype 与参考实现一致
- [ ] L3：边界场景（极值、零值、负值、空张量）不崩溃且结果合理
- [ ] L4：数值稳定性，无不合理的 NaN/Inf
- [ ] L5：空张量正确处理
- [ ] L6：确定性验证（相同输入多次运行结果一致）
- [ ] 整数算子额外验证 int32/int64（atol=rtol=0）
- [ ] CV 算子额外验证 K 维度奇数、非 512B 对齐场景
- [ ] 累加密集型可适当放宽 fp16 阈值并注释说明

---

## 第二部分：性能测试

### 7. 基础延迟测试

```python
import torch, torch_npu

def benchmark_latency(fn, warmup=10, repeat=100) -> dict:
    """测量算子延迟（从 kernel_details.csv Duration(us) 提取，纯 NPU device 时间）。

    使用 torch_npu.profiler Level1 采集，从 ASCEND_PROFILER_OUTPUT/kernel_details.csv
    的 Duration(us) 列提取每个算子的 NPU 设备侧耗时，不含 CPU/launch 开销。
    """
    for _ in range(warmup): fn()
    torch.npu.synchronize()

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=repeat),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof/bench"),
    ) as prof:
        for _ in range(repeat):
            fn()
            torch.npu.synchronize()
            prof.step()

    # 从 kernel_details.csv 读取 Duration(us) — 纯 NPU device 耗时
    import csv, glob
    csv_files = glob.glob("./prof/bench/**/kernel_details.csv", recursive=True)
    durations = []
    for f in csv_files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                name = row.get("Name", "")
                dur = float(row.get("Duration(us)", 0))
                # 只取计算 kernel, 剔除 Free/Cast
                if dur > 0 and not any(x in name for x in ("Free", "Cast", "Computing")):
                    durations.append(dur)

    t = torch.tensor(durations) if durations else torch.tensor([0.0])
    return {
        "mean_us": t.mean().item(), "min_us": t.min().item(),
        "max_us": t.max().item(), "std_us": t.std().item(),
        "source": "kernel_details.csv Duration(us) — NPU device time",
    }
```

### 8. 与 PyTorch 参考的性能对比

```python
def compare_with_reference(triton_fn, torch_fn, shapes, dtypes=None,
                           warmup=10, repeat=100) -> list[dict]:
    if dtypes is None: dtypes = [torch.float16, torch.float32]
    results = []
    for shape in shapes:
        for dtype in dtypes:
            x = torch.randn(shape, device="npu", dtype=dtype)
            y = torch.randn(shape, device="npu", dtype=dtype)
            triton_r = benchmark_latency(lambda: triton_fn(x, y), warmup, repeat)
            torch_r = benchmark_latency(lambda: torch_fn(x, y), warmup, repeat)
            results.append({
                "shape": shape, "dtype": str(dtype),
                "triton_ms": triton_r["mean_ms"], "torch_ms": torch_r["mean_ms"],
                "speedup": torch_r["mean_ms"] / triton_r["mean_ms"],
                "triton_p99_ms": triton_r["p99_ms"], "torch_p99_ms": torch_r["p99_ms"],
            })
    return results
```

### 10. Shape 扫描测试

```python
SHAPE_PRESETS = {
    "1d_small":  [(1024,), (2048,), (4096,)],
    "1d_large":  [(65536,), (131072,), (1048576,)],
    "1d_odd":    [(999,), (1333,), (2047,)],
    "2d_square": [(256, 256), (512, 512), (1024, 1024)],
    "2d_rect":   [(128, 4096), (4096, 128), (256, 65536)],
    "2d_odd":    [(127, 255), (513, 1023)],
}

def shape_sweep_benchmark(launch_fn, reference_fn, dtype=torch.float16) -> list[dict]:
    results = []
    for category, shapes in SHAPE_PRESETS.items():
        for shape in shapes:
            try:
                r = compare_with_reference(launch_fn, reference_fn, [shape], [dtype])[0]
                r["category"] = category; results.append(r)
            except Exception as e:
                results.append({"category": category, "shape": shape, "error": str(e)})
    return results
```

### 11. 性能回归检测

```python
import json

def save_baseline(results: list[dict], path="perf_baseline.json"):
    with open(path, "w") as f: json.dump(results, f, indent=2)

def check_regression(results, threshold=0.10, baseline_path="perf_baseline.json") -> list[dict]:
    """与基线对比，检测退步 > threshold 的用例。"""
    with open(baseline_path) as f: baseline = json.load(f)
    baseline_map = {(tuple(r["shape"]), r["dtype"]): r for r in baseline}
    regressions = []
    for r in results:
        key = (tuple(r["shape"]), r["dtype"])
        if key in baseline_map:
            prev = baseline_map[key]["speedup"]
            curr = r["speedup"]
            drop = (prev - curr) / prev
            if drop > threshold:
                regressions.append({
                    "shape": r["shape"], "dtype": r["dtype"],
                    "prev_speedup": prev, "curr_speedup": curr,
                    "drop_pct": drop * 100,
                })
    return regressions
```

### 12. 性能测试 checklist

- [ ] 预热排除首次编译开销（≥ 5-10 次 warmup）
- [ ] 使用 `torch.npu.synchronize()` 确保异步操作完成后计时
- [ ] 至少 50 次测量取统计值（mean / min / max / p99 / std）
- [ ] 覆盖典型 shape、大 shape（grid 压力）、非对齐 shape
- [ ] 与 PyTorch 参考实现对比，记录 speedup
- [ ] 长期稳定性测试（≥ 1000 次调用，无衰减）
- [ ] 基线保存到版本控制，支持回归检测

---

## 第三部分：完整测试文件模板

```python
# test_<op_name>.py
import pytest, time, torch, torch_npu

# from your_module import launch_kernel
# def reference_impl(x): ...

SHAPES = [(128,), (4096,), (65536,), (999,), (256, 256), (128, 4096)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


class Test<OpName>:
    """<算子名称> 精度与性能测试套件。"""

    # ===== L1: 编译通过 =====
    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    def test_compile(self):
        x = torch.randn(1024, device="npu", dtype=torch.float16)
        out = launch_kernel(x)
        assert out.shape == x.shape and out.device.type == "npu"

    # ===== L2: 正确性 =====
    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_correctness(self, shape, dtype):
        x = torch.randn(shape, device="npu", dtype=dtype)
        ref = reference_impl(x.cpu()).to("npu", dtype=dtype)
        out = launch_kernel(x)
        assert_precision(out, ref, dtype, shape)

    # ===== L3: 边界场景 =====
    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    def test_edge_cases(self, dtype=torch.float16):
        for name, x in generate_edge_cases(dtype):
            x = x.to("npu")
            out = launch_kernel(x)
            assert out.shape == x.shape, f"[{name}] shape mismatch"

    # ===== L4: 数值稳定性 =====
    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    def test_numerical_stability(self, dtype=torch.float16):
        test_numerical_stability(launch_kernel, dtype)

    # ===== L5: 空张量 =====
    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_empty_tensor(self, dtype):
        x = torch.empty(0, device="npu", dtype=dtype)
        out = launch_kernel(x)
        assert out.numel() == 0

    # ===== L6: 确定性 =====
    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    def test_deterministic(self, dtype=torch.float16):
        torch.manual_seed(42)
        x = torch.randn(1024, device="npu", dtype=dtype)
        assert torch.equal(launch_kernel(x), launch_kernel(x))

    # ===== P1: 延迟基准 =====
    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    @pytest.mark.parametrize("shape", SHAPES)
    def test_latency(self, shape, dtype=torch.float16):
        x = torch.randn(shape, device="npu", dtype=dtype)
        y = torch.randn(shape, device="npu", dtype=dtype)
        r = benchmark_latency(lambda: launch_kernel(x, y), warmup=10, repeat=50)
        assert r["mean_ms"] > 0
        print(f"[{shape}] latency: {r['mean_ms']:.4f}ms (p99={r['p99_ms']:.4f})")

    # ===== P2: vs PyTorch 性能 =====
    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    @pytest.mark.parametrize("shape", SHAPES)
    def test_vs_reference(self, shape, dtype=torch.float16):
        x = torch.randn(shape, device="npu", dtype=dtype)
        y = torch.randn(shape, device="npu", dtype=dtype)
        triton_r = benchmark_latency(lambda: launch_kernel(x, y), warmup=10, repeat=50)
        torch_r = benchmark_latency(lambda: reference_impl(x, y), warmup=10, repeat=50)
        speedup = torch_r["mean_ms"] / triton_r["mean_ms"]
        assert speedup > 0.5, f"Too slow: speedup={speedup:.2f}x"
        print(f"[{shape}] speedup: {speedup:.2f}x")

    # ===== P3: 稳定性 =====
    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    def test_throughput_stability(self):
        """稳定性测试: 1000次调用无衰减。精确延迟用 benchmark_latency (kernel_details.csv Duration)。"""
        x = torch.randn(4096, device="npu", dtype=torch.float16)
        y = torch.randn(4096, device="npu", dtype=torch.float16)
        for _ in range(1000):
            launch_kernel(x, y)
            torch.npu.synchronize()
        # 不崩溃即通过 — 精确延迟对比见 benchmark_latency


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

---

## 注意事项

- 性能测试建议在独占 NPU 设备上运行，避免负载干扰
- `torch.npu.synchronize()` 是关键，忘记同步会导致延迟测量严重偏低
- 首次调用含 kernel 编译开销，warmup 必须排除
- 参考实现放在 CPU 执行或确保与 Triton kernel 计算语义完全等价
- fp16 下 `torch.randn` 用固定 seed 确保可复现
- 不同 NPU 芯片（A2/A3/A5）浮点结果可能有微小差异，阈值需考虑芯片间差异
- 涉及随机性的算子（dropout 等）需固定 seed
- 性能基线保存在版本控制中，便于追踪性能变化趋势

## 参考信息

- [PyTorch testing 文档](https://pytorch.org/docs/stable/testing.html)
- [PyTorch Benchmarking 最佳实践](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [Triton-Ascend 精度对齐指南](https://github.com/triton-lang/triton-ascend)
- [CANN 性能分析工具文档](https://www.hiascend.com/document)

### 相关 Skill

| Skill | 关系 |
| ------ | ------ |
| `ops-design` | 上游 — 提供测试策略和验收门禁 |
| `triton-op-dev` | 上游 — 提供待测试的算子代码 |
| `ops-bwd-golden` | 上游 — 提供反向 torch 标杆（梯度 golden），反向精度以此为基准 |
