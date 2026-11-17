# TileLang 昇腾环境构建与配置

TileLang 在昇腾上有两条后端路线，安装方式完全不同，落地阶段按 `target_arch` 分流。

## 依赖版本约束（必读）

TileLang 各依赖存在**版本耦合冲突**，安装时须按本节约束处理，否则会破坏同环境内的 triton-ascend / scipy。

### Ascend 950（tilelang 主仓）—— 已在 mindspeed-llm 容器实测

| 依赖 | 约束 | 说明 |
| ------ | ------ | ------ |
| Python | **3.12** | tilelang 0.1.15 官方 wheel 为 cp312；TileKernels 要求 ≥3.12 |
| tilelang | **>= 0.1.15** | 昇腾 950 后端自该版本起可用 |
| torch / torch_npu | **版本必须配对** | 实测 `torch 2.10.0+cpu` + `torch_npu 2.10.0.post6`，二者须来自同一 CANN 配套 |
| CANN | **>= 9.0.0（建议 9.2.0+）** | 950 驱动配套；TileKernels 要求 CANN 9.2.0+ |
| **numpy** | **== 1.26.4（钉死）** | ⚠️ 冲突点：tilelang 依赖 `ml-dtypes 0.6.x` 会强升 numpy 到 2.5.x，破坏 `triton-ascend(==1.26.4)` 与 `scipy(<2.3)`；安装后必须立即钉回 1.26.4 |
| **ml-dtypes** | **< 0.5（用 0.4.1）** | 0.6.x 要求 numpy≥2，与 numpy 1.26 冲突 |

传递依赖（pip 自动装，无需手动指定）：`apache-tvm-ffi`、`z3-solver<4.15.5,>=4.13.0`、`torch-c-dlpack-ext`、`cloudpickle`、`tqdm`、`typing-extensions`、`psutil`。

### Atlas A2/A3（tilelang-mlir-ascend）

| 依赖 | 约束 |
| ------ | ------ |
| Python | **3.7 – 3.11.4**（含） |
| CANN | 8.3.RC1 / 8.5 |
| torch_npu | 与 CANN 版本匹配 |

### 冲突规避安装顺序（950）

```bash
pip install "tilelang>=0.1.15"
# 关键：立即钉回 numpy 1.x + numpy-1 兼容的 ml-dtypes，防止破坏容器内 triton-ascend/scipy
pip install "numpy==1.26.4" "ml-dtypes<0.5"
```

## 一、Atlas A2/A3（arch22）— tilelang-mlir-ascend

`tilelang-mlir-ascend` 是 A2/A3 的 MLIR 适配器，提供两条编译路线：**AscendNPU IR（MLIR）** 与 **Ascend C & PTO**。需源码构建。

### 前置条件

| 项 | 要求 |
| ------ | ------ |
| CANN Toolkit | 8.3.RC1.alpha002（较新版本支持到 8.5） |
| Python | 3.7 – 3.11.4（含） |
| torch_npu | 与 CANN 版本匹配 |

### 构建步骤

```bash
# 1. 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 克隆（含子模块）
git clone --recursive https://github.com/tile-ai/tilelang-mlir-ascend.git
cd tilelang-mlir-ascend

# 3. 安装 Python 依赖
pip install attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil \
            protobuf==3.20.0 scipy requests absl-py

# 4. 构建（可选 --bishengir-path 指定本地 AscendNPU-IR 构建）
bash install_npuir.sh
#   有本地 Bisheng/AscendNPU-IR 时：
#   bash install_npuir.sh --bishengir-path /path/to/ascendnpu-ir

# 5. 安装运行时绑定
pip install pybind11 torch_npu
```

### 运行环境变量

| 变量 | 默认 | 说明 |
| ------ | ------ | ------ |
| `ACL_OP_INIT_MODE` | 未设置 | 设为 `1` 启用 ACL 初始化 |
| `TILELANG_ASCEND_MODE` | `Expert` | `Expert`（优化算子）/ `Developer`（开发原语 `T.Pipelined`/`T.Parallel`/double buffer 等） |

### 算子支持面（A2/A3）

- 核心：GEMM、batch_gemm、vector add、Flash Attention（forward/backward）、卷积
- Developer 模式原语：`T.Pipelined`、`T.Parallel`（自动向量化）、`T.alloc_shared`、`T.alloc_fragment`、double buffer、自动 buffer 复用
- 向量算子：VFlip、VLn、VNot、VAbs、VAnd、VOr、VCmp、VPad、VPow、VRec、VRelu、VRSqrt、VSel、VShl、VShr、VXor、VTranspose、VInterleave、VGather

> 性能参考：GEMM ~0.98x、向量（DeepSeek-V4 hc_sinkhorn）~0.96x、Flash Attention（cube-vector）~0.95x，对比手写 AscendC。

## 二、Ascend 950（arch35）— tilelang 主仓

Ascend 950 在 tilelang 主仓原生支持（`USE_ASCEND=ON`），原生代码生成 + 自动调度同步 + SIMD/SIMT 向量编程。

### pip 安装（推荐，已在 mindspeed-llm 容器实测）

```bash
source <CANN>/ascend-toolkit/set_env.sh     # 如 /home/pkg/b080/ascend-toolkit/set_env.sh
pip install "tilelang>=0.1.15"
# 关键：tilelang 会带入 ml-dtypes 0.6.x 并强升 numpy 到 2.5.x，会破坏容器内
# triton-ascend(==1.26.4) / scipy(<2.3)。安装后立即钉回 numpy 1.x + 兼容 ml-dtypes：
pip install "numpy==1.26.4" "ml-dtypes<0.5"
```

实测结果（`tilelang 0.1.15` + `torch 2.10.0+cpu` + `torch_npu` + CANN 9.x / 950 DT）：

- `import tilelang` 正常；target 用 `"ascend"`（自动检测 `torch.npu.is_available()`），execution_backend 用 `"tvm_ffi"`/`"cython"`
- **昇腾 API 差异（务必注意）**：
  - `from tilelang.ascend import language as T`（非顶层 `tilelang.language`）
  - `T.Kernel(N)` 是一维 AI-core grid，**无 `threads` 参数**；二维 tile 展平到一维
  - 线程并行 `T.SimtVF(threads=...)`、SIMD `T.SimdVF()`；`T.Parallel` 必须包在 VF 块内
  - `T.copy`（DMA）不允许隐式 dtype cast；fp32→fp16 须在 VF 块内 `astype` 后再 copy
- 验证：element-wise add 在 950 上 `allclose=True`（max_abs_err=0.0）；GEMM 的 `T.gemm`/`T.Pipelined` 可编译，但 epilogue 需处理 fractal 布局，生产建议用内置 `tilelang.ascend.op.gemm`

### 源码构建（备选）

```bash
# 需 CANN + torch_npu
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
mkdir build && cd build
cmake .. -DUSE_ASCEND=ON -DCMAKE_BUILD_TYPE=Release
make -j
```

## 三、TileKernels（DeepSeek 算子库，可选参考）

DeepSeek 的 TileLang 算子库，提供 MoE routing、Engram gating、FP8/FP4 量化（fused SwiGLU）、Manifold HyperConnection、RoPE、随机数等 kernel。

- Ascend 支持：**Ascend 950 NPU + CANN 9.2.0+**
- 运行时自动选择后端（NVIDIA / Huawei），同一套 Python API 双后端通用

## 四、环境探测脚本

```python
import os
import torch, torch_npu

print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
print("ACL_OP_INIT_MODE:", os.environ.get("ACL_OP_INIT_MODE"))
print("TILELANG_ASCEND_MODE:", os.environ.get("TILELANG_ASCEND_MODE"))

try:
    import tilelang
    print("tilelang:", getattr(tilelang, "__version__", "unknown"))
except ImportError as e:
    print("tilelang 未安装:", e)
```

## 五、常见问题

| 症状 | 排查 |
| ------ | ------ |
| `import tilelang` 失败 | A2/A3 未跑 `install_npuir.sh`；950 未用 `USE_ASCEND=ON` 构建 |
| ACL 初始化失败 | 设置 `ACL_OP_INIT_MODE=1` |
| 原语不可用（`T.Pipelined` 等） | 切换到 `TILELANG_ASCEND_MODE=Developer` |
| Python 版本不兼容 | A2/A3 后端上限 Python 3.11.4 |
