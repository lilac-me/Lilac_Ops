# 迁移阶段：环境搭建 + 代码 NPU 迁移

本阶段产出：模型在 NPU 上正常训练。通过标准：**训练能正常迭代，loss 有限且无异常值**（无 NaN / Inf / 恒 0 / 异常跳变），目标卡数能起分布式并完整跑通前向 + 反向 + optimizer step（≥3 步）。有 GPU 基线时用相同 global batch / max steps / seed 对齐 Final loss 与任务指标（官方 `docs/zh/tools/precision_compare.md`），无 GPU 基线时按本段句 1 口径。

环境变量明细见 `../appendix-env.md`，运行基线脚本 `../../assets/env_npu.sh`（交付时由 `test/env_npu.sh` source 之，见 3-deliver.md）。

## 1. 环境搭建

### 1.1 版本配对（严格）

torch 必须装 CPU 版，设备后端由 torch_npu 提供：

```text
torch          2.7.1+cpu      ← CPU 版
torch-npu      2.7.1          ← 提供 NPU 后端（以 wheel 源实际可用的 postX 为准）
torchvision    0.22.1+cpu     ← CPU 版
```

- 主版本必须严格配对（torch 2.7.1 ↔ torch_npu 2.7.1.postX）。模型把 torch 锁定在 torch_npu 无对应配对的版本时，需改写锁定降到配对版本。锁定一般在 requirements / pyproject / 安装命令的版本声明里；这通常是迁移要解决的第一个冲突。
- **torch_npu 有两套版号，不要被语料带偏**：装包用 `torch_npu-<torch版本>.postN`（如 `torch_npu-2.7.1.post4`，PyPI / wheel 源上只有这一套），而 `model_examples/*/README.md` 版本表里写的是 **TorchNPU 产品版本**（`26.0.0` / `26.1.0`，旧模型还有 `7.1.0`）。两者是同一版本的不同编号（分支名 `v2.7.1-26.0.0` 能把对应关系看出来），不是冲突。`postN` 跟着 CANN 配套走，以配套表为准。
- 安装用昇腾官方 wheel 仓库/镜像站下载的配对 whl 且 `--force-reinstall --no-deps`（`--no-deps` 关键，否则 CPU 版会被换回 CUDA 版）。装任何依赖后复查 `pip list | grep "^torch"` 确认仍是 +cpu 版，依赖声明 `torch>=x` 会把版本换掉。

### 1.2 先 source CANN

装了 torch_npu 的 env，任何 `import torch` 之前必须先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。否则 torch 自动加载 torch_npu 后端时找不到 `libhccl.so` / `libascendcl.so`，报错却指向 torch，容易误判为 torch 安装问题。最小验证：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch; print(torch.__version__)"
```

### 1.3 NPU 上需要注意的依赖

| 包 | 说明 |
| --- | --- |
| `flash-attn` | CUDA 扩展装不上；用 transformers 自带 NPU 后端（见 §3.4） |
| 社区 CUDA 版 `triton` | 装昇腾适配版 `triton-ascend`（CANN<9.0 用 3.2.0，≥9.0 用 ≥3.2.1）。两者**并非不能共存**，一般直接装 `triton-ascend` 即可，无需先卸社区版 |
| `nvidia-*`（cublas/cudnn/nccl/cusparse） | 全删 |
| `+cu118` / `+cu121` 后缀的 torch/torchvision/torchaudio | **换成 CPU 版**：`pip install torch --index-url https://download.pytorch.org/whl/cpu`（torchvision / torchaudio 同）。设备后端由 torch_npu 提供 |
| `xformers` | `enable_xformers=False` |
| `apex` | 能装上，但内部 **CUDA** 加速路径不可用 → 装了也无收益，不装。⚠️ CANN 安装目录自带的 `precision_performance_advice.json` 会建议 `apex.optimizers.NpuFusedAdamW`：那指的是**昇腾适配版**里的 `NpuFused*` 优化器支线，等价物 `torch_npu.optim.NpuFusedAdamW` 已列在 ref-ops 优化器表，**不必为了它装 apex** |
| `tensorrt-cu12` | 装不上 / 不用 |
| `liger_kernel` | **禁止剥离**：模型自带就保留原样，它是**正确性路径**（GPU 基线同款，`apply_liger_kernel_to_*` 会改 forward），不是加速手段（Ascend 上 FLCE 退化为普通 CE、无融合收益，见 ref-ops「已排除手段」）。需新装 → 用支持 Ascend 的源 |

删依赖后 `pip install -e .` 可能被 `pyproject.toml` 的构建依赖声明再拉回来，注意复查。`torch/_dynamo/_triton.py` 的 CUDA 能力校验报错（`None >= (9,0)`）只在调用 torch.compile 时出现；torch.compile 在部分模型可正常用，遇该校验报错时关闭它即可。

### 1.4 视频解码后端（视频/VAE 类先查再装）

| 后端 | 需编译 | 说明 |
| --- | --- | --- |
| `decord` / `torchcodec` | 是 | PyPI 无 aarch64 wheel，需源码编译。 |
| `torchvision_av` | 否 | 模型自带的开关（如 GR00T-N1.5 的 `--video_backend`），模型不支持就换不了 |
| `av` + `imageio-ffmpeg` | 否 | |

参考模型 README 的「安装 decord / 安装 torchcodec」一节可直接照抄（从 `references/ref-models.md` 挑结构最像的模型，去 `DrivingSDK/model_examples/<模型>/README.md` 取）。

**decord**：`FFMPEG_DIR` 填 ffmpeg 实际装在哪（填错 cmake 找不到 ffmpeg）：

| ffmpeg 从哪来 | `cmake` 参数 |
| --- | --- |
| 源码装到 `/usr/local/ffmpeg`（另需在 `/etc/profile.d/ffmpeg.sh` 配 `PATH` 与 `LD_LIBRARY_PATH`） | `-DFFMPEG_DIR:PATH="/usr/local/ffmpeg/"` |
| conda | `-DFFMPEG_DIR:PATH=$CONDA_PREFIX` |
| 系统包（`yum install -y ffmpeg ffmpeg-devel`） | 不传 |

编译尾巴固定：`cd ../python && python setup.py sdist bdist_wheel`，再装 `decord/python/dist/decord-<版本>-cp<py>-cp<py>-linux_aarch64.whl` —— **文件名里的 Python 版本与架构按实际改**（语料是 cp310 / aarch64）。

**torchcodec**：先 `conda install -c conda-forge pybind11`，再 `git checkout` 目标版本号后 `pip install -e . --no-build-isolation`（缺 `--no-build-isolation` 会去拉 CUDA 版 torch）。

装完验证：`python -c "import decord; print(decord.__version__)"`（torchcodec 同理）。

- `import decord` 必须放 train.py 顶部、`import torch_npu` 之前（两者争抢 libgomp，顺序错了段错误或死锁）。
- 缺 `ffprobe`/`ffmpeg`：DataLoader worker 报 `FileNotFoundError: 'ffprobe'`，容易误判为模型/分布式问题；装了还要确认在训练进程的 PATH 里（launch 脚本显式指定 python 路径时 PATH 仍是系统默认）。首次跑训练前先 `ffprobe -version`。
- 缺 `libgl1`/`libglib`：`ImportError: libGL.so.1` → `apt-get install -y libgl1 libglib2.0-0`。

### 1.5 其它前置依赖

- `bitsandbytes`：仅量化场景（fork `multi-backend-refactor-npu` 分支编译）。
- `mx_driving`：transformers 侧的融合补丁（RMSNorm / RoPE / 图像预处理），必须装上。

  ```bash
  cd <DrivingSDK>
  bash ci/build.sh --python=<当前解释器版本，如 3.10>   # 只 build wheel，不安装
  cd dist && pip install --force-reinstall mx_driving-*.whl
  # 当无源码或源码编译失败时降级，降级包可能落后 master 而少补丁类：
  #   pip install --force-reinstall mx_driving -i <镜像>   # 别写多余的版本号：不写即取最新正式版

  # 自查（不过 = 装的还是旧版）
  python -c "import torch, torch_npu; from mx_driving.patcher import transformers_patch; print('ok')"
  ```

  芯片名含 `950`（A5）时 `ci/build.sh` 加 `--a5`，910/310 系不加。用法 / 版本闸门 / 收益见 2-optimize.md §4.2。

### 1.6 离线 / 数据集与权重

- 离线：四个开关的取值与含义见 `../appendix-env.md` §离线；这里只讲代码侧：`from_pretrained("org/model")` 要改本地路径（调用点可能很分散，逐个改，或用 `local_files_only=True` / 直接路径替换）。
- `config.json` 里的嵌套绝对路径（如 `backbone` 字段指向作者机器）是最常见的隐性前置：不改就报文件不存在。
- 数据集与预训练权重都要备齐：权重确认 from_pretrained 所需文件齐全（`config.json` / `model.safetensors` / tokenizer / preprocessor）且离线预下载；数据集确认格式、目录结构与数量；norm stats 这类训练前全量扫描离线先跑一遍。
- pip 镜像：实测取可用者，不要硬编码（同一镜像在不同机器/代理下表现完全不同；curl 探测返回 200 才可用，且要装一个小包确认 wheel 下载路径通）。

## 2. 启动方式

**启动入口不统一，不要假定 `test/*.sh` 就是启动器，先向下追溯一层再确定 launcher。** 全仓 188 个 `test/*.sh` 实测（类别可重叠）：`torchrun` 37 个、`torch.distributed.run|launch` 11 个、`accelerate`（含 `python -m accelerate.commands.launch`）6 个；另有 63 个**只是再包一层自家脚本**（`bash scripts/xxx.sh` / `bash torch_run.sh`，真正的启动器在被包的那层里），49 个直接 `python xxx.py`。VLM/VLA 系以 torchrun 与 accelerate 两类为主，感知类多是 `nohup python3 <入口>.py` + mmcv 自带 launcher。全仓**没有 ranktable 场景**（实测 grep 无命中）。

torchrun 写法：

```bash
torchrun --nnodes=$N --nproc-per-node=$M --node-rank=$R --master-addr=$ADDR --master-port=$PORT train.py ...
```

单卡也建议 `torchrun --nproc-per-node=1`（保证 `LOCAL_RANK` 有定义，避免代码读该变量时 KeyError）。

已知问题：

- **HCCL teardown 挂起**（部分 CANN 版本）：主流程跑完后 `dist.destroy_process_group()` 卡死（日志已打出「训练完成」但进程不退出）。修法：收尾显式 `sys.stdout.flush(); sys.stderr.flush(); os._exit(0)`（不跑 atexit/析构，直接退）。副作用：通信域从不析构，device 侧 socket 端口残留到下次运行（报 `EI0020: ... already bound` 或直接挂起）；缓解：`HCCL_NPU_SOCKET_PORT_RANGE` 换端口段 + 两次多卡任务之间留间隔。不要在收尾处调自定义 `cleanup()` 后再正常 return。
- **多卡启动卡死 / HCCL 报错定位**：第 0 步查资源占用（`npu-smi info`、`npu-smi info -t proc-mem -i <id>`、`ss -anp | grep -E ':(16666|6000[0-9])'`）。一张卡同一时刻只能运行一个多卡程序，共享机器上别人的任务或自己上次的残留进程都会表现为代码问题；清理残留只按精确 PID 杀，不要 `pkill -f <pattern>`。
- **现象「模型加载完之后无输出」**：`init_process_group` 是惰性的，第一个集合通信（barrier / all_reduce / FSDP 的 sync_module_states 广播）才建域。
- **三步缩小范围**：① 分段插桩计时（from_pretrained / .to / apply_fsdp 各打耗时日志）→ ② 最小隔离脚本（init_pg + 建模型 + .to + barrier）→ ③ 纯 HCCL 冒烟（init + all_reduce 跑卡数矩阵，区分跨卡不通还是超过某阈值才不通）。
- **禁止调大 `HCCL_CONNECT_TIMEOUT`**：默认 120s 就是调试友好值（官方范围 [120, 7200]，实际建链超时 ≈ 配置值 +20s；低于下限报 EI0001 非法配置）。建域失败时长时间挂起而不报错，**正是把它调大造成的**：语料里有设成 1200 的，属多机慢建链特例，不要当默认值照搬。取值与出处见 `../appendix-env.md`。

## 3. 代码迁移

### 3.1 静态分析（动手前，只读不跑）

1. 找入口：从启动脚本（.sh / torchrun / mpirun / deepspeed）逐级追到训练主函数（train.py / main.py），确认数据加载 → 模型构造 → 优化器/调度器 → checkpoint 的调用顺序；标出第一个碰 `torch.cuda` / `init_process_group` 的位置（决定入口块注入点）。
2. 产出数据流一页纸（入口 → dataloader → 前向 → 损失 → 反向 → 优化器 → 存档），写进**模型档案卡**的「数据流一页纸」字段；**按 `../../assets/model-card.template.md` 建模型档案卡**（每模型一份，作为 A→B→C 跨阶段的压缩载体与审计凭据载体）。
3. **组件体检**：`bash ../../scripts/scan_gpu_deps.sh <repo>`，扫 6 类「改字符串解决不了、会在 import 或算子调用处硬失败」的组件（flash-attn / 图模式 / TransformerEngine / 自定义 CUDA 扩展 / vLLM·triton / CUDA 诊断代码），每项指向处理章节。**cuda/nccl 字面量与 API 实参不在扫描范围**（由 `transfer_to_npu` 运行期兜底）。结论决定下面 §3.2 的入口块与处理路线。
4. 查 `../ref-models.md` 找 2~3 个结构最像的已适配模型，读其 patch——**只当参考线索，不当改动清单**：结构/版本差异可能很大，改哪里、改成什么值都要在本模型上重新判定。
5. 迁移期每处人工改动当时在档案卡「审计凭据」段记一条（改了什么 + 为什么 + 工具是否覆盖不到）；见 3-deliver.md 冗余审计凭据对账制，不要留到交付时再回溯。

### 3.2 一键迁移（优先，不改源码）

训练主函数入口块，放在 train.py 顶部、所有受影响 import 之前：

```python
import torch, torch_npu
from torch_npu.contrib import transfer_to_npu
```

**入口块加完先直接跑，不要逐处修改 CUDA 调用。** 下面这段是它到底能兜多少、兜不住什么；读完就不必再去逐个修改 CUDA 调用。真兜不住的部分，也是**运行后按报错/静默错定点修复**，而不是提前预判修改。

`transfer_to_npu` 分两层，**两层都是白名单机制，不是普适规则**（依据：`torch_npu/contrib/transfer_to_npu.py` 源码）：

  1. **方法 / 函数别名**：`.cuda()`、`.to("cuda")`、`torch.cuda.*`（`amp` / `set_device` / `is_available` / `get_device_capability` / `memory_stats` 等同名一族）整体别名到 NPU 实现；
  2. **实参值级替换**：只在**被包装的白名单函数**上生效：torch 顶层约 47 个（含 `load` / `zeros` / `randn` / `set_default_device` …）、`torch.Tensor.*`、`nn.Module.to`，以及显式包装的 `init_device_mesh` / `DeviceMesh.__init__` / `DDP.__init__` / `FSDP.__init__` / `autocast.__init__` / `DeviceContext.__init__` / `torch.cuda.*`；**关键字实参只认 4 个名字**：`device` / `device_type` / `map_location` / `device_id`（位置实参则整串替换）。`init_process_group` 的 `nccl → hccl` 走另一套 `_wrapper_hccl`。

  → 常见写法全部兜住（实测）：`torch.load(map_location="cuda")`、`.to("cuda")`、`init_device_mesh("cuda")`、`autocast("cuda")`。

  → **边界（易错点）**：任意第三方函数 / vendored 库里的同名形参**不在白名单内**（如 `Foo(x, device="cuda")`），这类调用点必须自己改，交付审计时也不能当「工具已覆盖」删掉。

- 它兜不住的只有三类，**都不是现在改**，而是跑通后按报错/静默错定点修复（改写要点见 §3.3）：① 控制流/定义级字面量（`== "nccl"` / `startswith("cuda")` / 函数默认参数 / 模块常量）；② `torch.device(...)` **构造出来的对象**：transfer 只在它作为实参时兜底，对象本身仍是 cuda；③ f-string 拼出的设备串（`f"cuda:{rank}"`）：它不是完整字面量，值级替换拆不开。
- **用视频解码后端时上面这个顺序要改**：`import decord` 必须排在 `import torch_npu` 之前（§1.4，两者争抢 libgomp）。语料的写法是 `import torch` → `import decord` → `if <NPU>: import torch_npu`。

组件体检的结论在 §3.1 第 3 步产出，按结论对号入座（体检本身不再在跑完之后重跑一遍）：

| 组件 | 处理 |
| --- | --- |
| flash-attn | §3.4 |
| 图模式（torch.compile / torch.jit / CUDA Graph） | §3.3 |
| TransformerEngine | ../ref-models.md |
| 自定义 CUDA 扩展 / 算子 | §3.3 |
| vLLM / triton | §3.3 |
| CUDA 诊断代码（pynvml / nvidia-smi / libcudart 等） | §3.3 |

### 3.3 必须人工处理的耦合点

**设备字符串与 device 对象**（大部分已被 transfer_to_npu 在函数边界兜底，只处理它覆盖不到的）：

- 白名单函数上的 `device` / `device_type` / `map_location` / `device_id` 实参**已由 transfer_to_npu 值级兜底，不用改**（白名单范围与边界见 §3.2）；`torch.jit.load` 不在白名单 → 显式 `map_location="npu"`。
- 控制流/定义级字面量（`== "cuda"` / `startswith("cuda")` / `in ("cuda","cpu")`、函数默认参数、模块常量）：**先运行，报错/静默错出现后再定点修复（运行时驱动），不要提前逐处改**。改的时候**判断与分支体要一起改**：`if get_backend() == "nccl": return torch.device(f"cuda:{rank}")` 这类，只把 `"nccl"` 换成 `"hccl"` 会让代码进 hccl 分支却仍构造 cuda device，**必须连同分支体一起改成 npu**
- `torch.device("cuda", rank)` 构造级 → `"npu"`
- `if device.startswith("cuda")` → 放行 npu
- 只作函数传参的 device 对象不用改（transfer_to_npu 在函数边界兜底）；内部做 `.type == "cuda"` 比较 / 筛选 / 存配置复用的必须显式改（FSDP2 `MixedPrecisionPolicy`、`optimize_fully_shard` 等）
- `CUDA_VISIBLE_DEVICES` → `ASCEND_RT_VISIBLE_DEVICES`（`set -u` 下引用未定义变量会直接 abort）

**图模式（torch.compile / jit）**：

- ⚠️ **`transfer_to_npu` 会把 `torch.jit.script` / `torch.jit.script_method` 直接变成 no-op**（源码 `_patch_jit_script()`，只发一条 RuntimeWarning）：用 `@torch.jit.script` 的模块在开了入口块之后会静默退化成普通 Python 函数；性能变化但结果不变，不要把它当作报错线索。
- 迁移阶段先关图模式只为了「跑通 + 易于定位」：`torch_npu.npu.set_compile_mode(jit_compile=False)`（等价 `torch.npu.set_compile_mode`）。**不是一律关死**：实机运行时可按报错情况调整：图模式很多报错源于算子/版本能力检查，可先试虚拟设备能力号规避，例如 `export TORCH_NPU_DEVICE_CAPABILITY=8.0`，而不是遇到报错就关图模式（**这是实测/社区实践，不是官方结论**：官方只说该变量配置 `torch_npu.npu.get_device_capability()` 的返回值、不代表 NPU 真实能力，详见 `../appendix-env.md`）。图模式 fusion 收益在部分模型上明显，调通后值得回开试跑（性能阶段结合 2-optimize.md §1/§4；变量见 ../appendix-env.md）。

**私有格式（NZ format）**：

- `allow_internal_format` 属**优化阶段**的可尝试项（和其他环境变量一样实测取舍），不是迁移必做：`torch.npu.config.allow_internal_format = False` 关闭 NZ 私有格式。迁移阶段一般不动，需要显式 NZ 转换/布局相关报错时按 2-optimize.md §1 处理。

**分布式后端**：

- `init_process_group` 的 `"nccl"` 参数位由 transfer_to_npu 兜底，一般不用改；上游 legacy `backend="gloo"`（CPU checkpoint 转换遗留，gloo 无法广播 npu tensor）要在 patch 里固定为 `"hccl"`
- `init_device_mesh("cuda")` → `"npu"`；混合后端 `"cuda:nccl,cpu:gloo"` → `"npu:hccl,cpu:gloo"`
- 能力检测 `torch.cuda.nccl` / `dist.is_nccl_available()` → 已重定向为 hccl 版；dist 辅助函数（`broadcast_object_list` / `barrier` / FSDP 的 device 参数）按后端返回 npu device

**AMP / dtype**：

- `tf32=True → False`（NPU 无意义）；昇腾 HF32 开关属性能项（2-optimize.md §1）
- **fp64 → fp32**：NPU 无 fp64，无替代方案，必须逐处核对（重点：扩散/flow-matching 采样器、位置编码、all-reduce 统计量、DeepSpeed ZeRO 内部）

**FSDP / DeepSpeed**：

- FSDP 原生支持 NPU，wrap 粒度是性能项，迁移阶段不动
- DeepSpeed ZeRO-1/2 内部用 fp64，用 ZeRO 就必打补丁：

```bash
# 路径中的 3.10 按 env 的 Python 版本替换（3.11/3.12 相应改）
sed -i 's/g\.data\.double()\.detach()/g.data.float().detach()/g' \
  "${CONDA_PREFIX}/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py"
```

- DeepSpeed 版本没统一答案，实测为准（FastWAM 用 0.14.4，GR00T-N1.7 用 0.19.3）；zero3 下 batch size 常需下调

**CUDA 诊断代码（会硬失败）**：`pynvml` / `nvmlDeviceGetMemoryInfo` / `torch.cuda.temperature()` / `power_draw()` / `clock_rate()` / `ctypes libcudart` / `nvidia-smi` 调用 / NVML 手工绑核（`os.sched_setaffinity`）→ 删除（绑核改用 `CPU_AFFINITY_CONF`）；`torch.cuda.utilization()` / `max_memory_allocated()` 可保留（NPU 下可用）。

**依赖与自定义扩展**：`torch_scatter` → mx_driving 的 `scatter_max`/`scatter_mean`（注意 dtype：`feat.to(torch.float)`、`indices.to(torch.int32)`）；自定义 CUDA 扩展 → mx_driving 对应实现或 `patcher.skip_import`；大规模坏 import（LAVIS/timm 的 `__init__` 引一堆 CUDA-only 模块）需裁剪；TransformerEngine → 优先换昇腾适配版 `TransformerEngineNPU`（同名包、与 CUDA 版不同装，按配套表核对版本），不适配再剥离（见 `../ref-models.md` 第二部分）。模型自带的 Liger-Kernel 补丁（`apply_liger_kernel_to_*`）**保留原样**、属正确性路径（GPU 基线同款）；不要与 mx_driving.patcher 对同一模块重复 patch。

**训练协议核对**：训练路径还要核对 processor 输出 → batch key → forward 签名（labels / mask / position_ids）的契约，不凭推理 demo 判定训练可跑。

**随机性与可复现（正确性，必做）**：若训练入口未固定全局随机种子，未从 checkpoint
加载的模块（专家塔、新增 head）每次进程初值不同 → 同配置两跑 loss 不一致、A/B 精度
对齐被初值噪声污染。迁移早期在训练入口（任何随机初始化之前）加：

```python
from seed_all import seed_all  # ../../assets/seed_all.py
seed_all(cfg.seed, is_npu=True)   # torch.manual_seed + torch.npu.manual_seed_all
```

验证：同卡同 seed 跑两遍，逐 step loss 应完全一致（实测 max|Δloss|=0，40 步）。

### 3.4 Attention 决策

**范围**：上游原本就走 FA（主干、视觉塔、文本塔任一处），NPU 上 `import flash_attn` 失败退成 sdpa/eager → 本阶段改回 FA，按原代码的写法选下面三条里的一条。上游原本就不走 FA、想主动换上提速 → 属优化阶段（2-optimize.md §4.4）。

根因：GPU 侧 `import flash_attn` 的写法在 NPU 上不存在，import 直接失败，训练从 FA 退化为 fallback（sdpa/eager）或崩溃。transformers ≥4.51 内置 `transformers.integrations.npu_flash_attention`（`npu_flash_attn_func` / `npu_flash_attn_varlen_func`，与官方 flash-attn 同名函数签名对齐，内部路由 `npu_fusion_attention`）。**优先走 transformers 的 NPU FA 后端，不要主动手写。**

按实现方式分三类，改动代价从小到大（前两条要求 transformers ≥4.51；版本不够就走第 3 条）：

1. **`attn_implementation="flash_attention_2"`（transformers 原生后端选择）** —— 只改后端选择：NPU 上指定 `flash_attention_2`，或把「import flash_attn 失败退 sdpa」的 fallback 改成先试 torch_npu。**改法是静默的**：运行时 dispatch 读的是嵌套 config，`text_config._attn_implementation` / `vision_config._attn_implementation` 没跟着改就是没生效，见 §3.5（打印「已生效」但实际没变），靠回读 config 确认。

2. **直接 import flash_attn 函数**（`flash_attn_func` / `flash_attn_varlen_func` / `_flash_attention_forward` 封装）—— 别名导入替换，本体不动：

   ```python
   if is_torch_npu_available():
       from transformers.integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
       from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
   ```

   只有上面这两个 FA 入口函数是真「同名同签名」，可以直接别名替换。下面两条照搬会踩：

   - **`flash_attn.bert_padding` 的 `pad_input` / `unpad_input` / `index_first_axis` 不能从 `npu_flash_attention` 取**。4.51~4.55 该模块确有同名函数，但 `unpad_input` **返回 5 元组**（多一个 `used_seqlens_in_batch`），flash_attn 版是 4 元组，语料里 `x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)` 这种照搬会 `ValueError: too many values to unpack`；而 **4.56 起这三个函数已从该模块整体移除**（改私有落在 `transformers.modeling_flash_attention_utils`：`_pad_input` / `_unpad_input` / `_index_first_axis`）。要走 varlen 就在模型侧自己实现 padding/unpadding，或钉住 ≤4.55 并同步改解包处。
   - `_flash_supports_window_size` 要重算（`"window_size" in inspect.signature(flash_attn_func).parameters`，NPU 版签名不同）。

3. **非 transformers**（TE DiT、独立 CLIP/T5 塔、`nn.MultiheadAttention`、自研 ViT）—— 才手写 `npu_fusion_attention`，调用约定见 `../ref-ops.md`。这一类要连 attention 带 RoPE 一起重写，三个陷阱里**只有广播维那个会报错，前两个静默**：

   - **mask 极性相反（静默）**：`npu_fusion_attention` 的 `atten_mask` True = 屏蔽，与 SDPA 相反。写法 `atten_mask=(~attention_mask).unsqueeze(1)`。症状：输出全同值、loss 不下降。
   - **RoPE cos/sin 内容：半宽 vs 全宽（静默）**：上游 flash-attn 路径会先把 cos/sin 砍成半宽（语料 `apply_rotary_pos_emb_flashatt` 里成对出现的 `cos = cos.chunk(2, dim=-1)[0]`），`npu_rotary_mul` 要**全宽**（`rotate_half` 语义）。两者**形状相同**，用错**静默算错**（A3 实测 max|Δ|=7.9，不报错、loss 不收敛），最难查的一种，只能靠模块级对拍。修法：删掉那两行 chunk。
   - **RoPE cos/sin 广播维（报错）**：规则是「**cos/sin 的秩与 q 相同，且 q 的 head 维（倒数第 2 维）处为 1**」，其余维填真值或 1 都行。A3 实测的三种正确形态：q `[B,S,H,D]` + `[1,S,1,D]`、q `[B,S,H,D]` + `[B,S,1,D]`、q `[N,H,D]`（视觉塔，3 维）+ `[N,1,D]`；秩不匹配或 head 维没插 1 都直接 RuntimeError，定位成本低。**不要按「两种 layout」记忆，按 head 维对齐**。

   把 `npu_rotary_mul` 换成融合内核是优化阶段的事（2-optimize.md §4.3）。

### 3.5 常见静默错速查

跑得起来但数值是错的，按症状反查：

| 症状 | 优先怀疑 | 修法 |
| --- | --- | --- |
| 输出全同值、loss 不下降 | attention mask 极性相反 | `atten_mask=(~mask)` |
| loss 下降极慢或不降 | RoPE cos/sin **内容**错（半宽 vs 全宽） | `npu_rotary_mul` 要全宽（`rotate_half` 语义）；广播维见 §3.4 |
| NaN / grad 异常 | fp64→fp32 未覆盖 | 全链路逐处核对 |
| varlen mask 索引静默错位 | `torch.bucketize` 默认 `right=False` | 统一 `right=True` 再 −1 |
| 打印「已生效」但实际没变 | config 参数在模型构造之后修改（构造时已固化） | 构造前传或重建模块 |

## 4. 报错速查

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `No module named 'torch_npu'` | 未装或装错环境 | 确认 conda env |
| `libhccl.so: cannot open shared object file` | CANN 未激活 | `source set_env.sh` |
| torch 被换成 CUDA 版 / 与 torch_npu 版本不匹配 | 依赖声明 `torch>=x` 被依赖解析改掉 | 重装配对版本（`--force-reinstall --no-deps`），详见 §1.1 |
| `ImportError: flash_attn` | NPU 装不了 | 见 3.4 |
| `undefined symbol` / 段错误（视频解码） | ffmpeg 版本不匹配，或 decord/torch_npu import 顺序 | 对齐版本；`import decord` 提到最前 |
| 编译找不到 CANN 头文件 | `ASCEND_TOOLKIT_HOME` 未设 | `source set_env.sh` |
| `bc: command not found` | 脚本用 bc 算 FPS | 装 bc 或改用 awk |
| `AttributeError: ... 'generate_id'` | 部分 wandb 新版本将 `wandb.util.generate_id` 标记废弃或移除 | shim：`wandb.sdk.lib.runid.generate_id`，回退 `wandb.sdk.wandb_run.Run.generate_id` |
| `ModuleNotFoundError: triton`（昇腾） | 社区 triton 不跑 NPU | `pip install triton-ascend` |
| 算子级报错定位不出（想看同步现场） | 异步下发会覆盖真实报错点 | 临时 `export ASCEND_LAUNCH_BLOCKING=1` 定位，**定位完立刻去掉**：它会让 `TASK_QUEUE_ENABLE` 整体失效、性能大幅下降，**性能跑与交付件里绝不出现**（详见 `../appendix-env.md` §四） |

自检（分项）：`pip list | grep "^torch"`（确认 +cpu 且配对）、`torch.npu.device_count()`、`npu-smi info`（卡型决定 `TORCH_HCCL_ZERO_COPY`）、import transformers 的 `npu_flash_attn_func`、`torchrun --nproc-per-node=2` 跑 all-reduce 冒烟（配 `../../scripts/` 里的最小脚本或自写）。

→ 迁移通过后进入优化阶段：`2-optimize.md`
