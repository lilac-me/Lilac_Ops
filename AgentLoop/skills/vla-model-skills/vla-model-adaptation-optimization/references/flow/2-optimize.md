# 性能优化阶段

前置：迁移已通过（loss 正常迭代、无异常值），基线已归档。**档案卡「基线」字段是跨阶段唯一读入的口径来源**（`../../assets/model-card.template.md`），工作区的踩坑记录只留同一行摘要供后续模型参考。

本阶段产出：**有量化收益且没伤精度的优化**。完成标准：至少一项优化假设走完闭合实验并有书面结论；保留/回退唯一裁判 = 量化收益（中性/负收益一律回退并记录）；只做了配置调参或实验结论是回退默认值的，写「开箱基线，优化进行中」。`num_workers` 之类不改变模型计算路径的配置调整不算优化。

纪律：先采一次 profiling 量出瓶颈类别（先量后猜）→ 一次只加一项 → 立刻做精度对齐 → 不过就回退，不要累积到最后一起排查。精度跑与性能跑分开（性能跑关掉 `HCCL_DETERMINISTIC` / `CLOSE_MATMUL_K_SHIFT`，见 `../appendix-env.md`）。

**改了源码的优化项**（融合算子、FSDP wrap 粒度等）在改动行上方留一行标记 `# [可选优化] <收益>`：交付审计据此区分「迁移必需」与「可选优化」，读 patch 的人也靠它判断哪行能回退。验证过的优化项**固化为模型默认**，开关只是过程工具不须交付。

手段按性价比顺序推进：**零代码 → 显存/FSDP → 融合算子 → host-bound → batch/数据 → 优化器**。每项与 §0 采到的瓶颈类别对上才做，对不上跳过。

## 0. profiling：采集与分析

### 0.1 采集

`torch_npu.profiler`（不是 `torch.profiler`；`activities` 用 `ProfilerActivity.NPU`；`torch.profiler.ProfilerActivity` 没有 NPU 成员）。`with_modules` 是**静默无效**，不是会报错：传了照跑，但产出与不传时逐文件一致、没有任何 module 维度（实测），不要指望它给出模块归属。要点：

- `skip_first=10` 跳过启动阶段（dataloader 未打满、HCCL 建连、首步 JIT），从 step 0 采到的空闲占比是假的
- `active=1` 通常够（定位瓶颈类别即可）；`wait` / `warmup` / `repeat` 留 1
- 训练照常跑满 max_train_steps，不要采集后立即 break（污染同一次跑的 FPS 数据）
- 用目标训练规模卡数（常态 8 卡），不要缩卡下结论（通信占比/FSDP/每卡吞吐随卡数变）；资源不足时允许缩卡，但口径必须注册并标「仅本口径有效」
- **用环境开关控制，不做成命令行参数**（`PROFILE=1 bash train_8p.sh` 这种；理由见 3-deliver.md「交付前检查」：避免交付后仍需用户拼接三个参数）。但开关要确认**真的接通了**：`grep -n "PROFILE\|profile" test/*.sh` 确认脚本有这一段，不要假设「沿用现有脚本加两个变量」一定成立

```python
# 脚本里用开关控制（PROFILE=1 启用），不做成命令行参数
profile_steps      = int(os.getenv("PROFILE_STEPS", "0"))      # 0 = 不开
profile_dir        = os.getenv("PROFILE_DIR", "./profile_out")
profile_skip_first = int(os.getenv("PROFILE_SKIP_FIRST", "10"))

# 训练循环前
prof = None
if profile_steps > 0:
    from torch_npu.profiler import (profile as npu_profile, ProfilerActivity,
        schedule as npu_schedule, tensorboard_trace_handler, _ExperimentalConfig, ProfilerLevel)
    prof = npu_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=npu_schedule(skip_first=profile_skip_first, wait=1, warmup=1,
                              active=profile_steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(os.path.join(profile_dir, f"rank{global_rank}")),
        experimental_config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1))
    prof.start()
# 每 step 末尾（optimizer.step() 之后）：prof.step()；循环正常结束后：prof.stop()
```

要看 cube/vector 利用率加 `aic_metrics=AiCMetrics.PipeUtilization`（torch_npu 2.7 里枚举名是 `AiCMetrics` 不是 `AiCoreMetrics`），`kernel_details.csv` 会多 `aic_mac_ratio` / `aiv_vec_ratio` 列。

产出 `<profile_dir>/rank<N>/<...>_ascend_pt/ASCEND_PROFILER_OUTPUT/{step_trace_time.csv, kernel_details.csv, op_statistic.csv}`。**采集完接着走 §0.2**——这几个 csv 是它的输入，不是终点。raw 目录（`*_ascend_pt` 下的 `.db` / `trace_view.json`，8 卡一轮 GB 量级）确认 csv 非空后可删，但那属于收尾清理，不是本步的完成标志。

单步采集只用于定位瓶颈类别，不要用它报收益（单步有波动）；**收益数字一律取日志多步均值**（跳过 warmup，与 3-deliver.md 的 FPS 提取脚本同口径；口径不一致会让自测数字和 ModelZoo 上报数字对不上）。

### 0.2 分析：msprof-analyze advisor

```bash
pip install msprof-analyze   # 官方工具（Python≥3.7.5，建议 3.9+）；训练环境通常是 3.10 → 单独一个 env 装
msprof-analyze advisor all -d <profile_dir>/rank0/<...>_ascend_pt -o <analysis_out> -l cn --force
```

输出 HTML + xlsx + 终端摘要，覆盖：整网耗时拆解（**计算 / 未掩盖通信 / 空闲，先看这三行**）、字节对齐、通信包大小、AICPU 算子、动态 shape、Slow Dataloader、亲和 API。

注意事项：

- CANN 版本识别会错位（工具已知版本列表滞后，静默退到最接近版本），「亲和 API 接口」这一条不可信，需人工复核；其余分析项不受影响
- 看算子 Top-N 要剔除通信算子（`allgatherAicpuKernel` / `reduce_scatterAicpuKernel`，它们已计入「未掩盖通信」，再算进算子耗时是重复统计，会得出「算子耗时远超 step time」的荒谬结论）；只看计算/搬运类内核，通信在「未掩盖通信」「通信包大小」里看
- advisor 只按固定 API 名匹配，识别不了手写逐元素组合（手写 dense norm 分发成多个 aclnn 算子、没走 `npu_rms_norm`，报告不报）→ 用 `../../scripts/discover_fused_patchers.py` 语义扫描兜住（见 §4）

### 0.3 瓶颈类别 → 手段

| profiling 表现 | 常见根因 | 手段 |
| --- | --- | --- |
| 空闲占比高 / Slow Dataloader / kernel 间隙大 | host 侧下发跟不上 device | §1 二级流水/绑核 + §5 去同步/向量化、日志降频、`gc.disable()`、`num_workers`/`prefetch_factor`/`pin_memory` |
| 计算占比不低但 kernel 又多又小 / `aic_mac_ratio` 低 | micro batch 太小吃不满 cube | §6 加大 `micro_batch_size` 配梯度累积、循环向量化、§4 融合算子 |
| 未掩盖通信占比高 | 固定通信开销没被摊薄、FSDP/ZeRO wrap 粒度不合适、通信包过小 | 先 §6 加大 batch 摊薄，仍高再动 §3 FSDP 粒度（两种粒度目标卡数实测，结论会随卡数翻转） |
| AICPU 算子命中 | 算子无 NPU 实现静默落 CPU（`torch.linalg.*` 高发） | §5 用 NPU 算子拼等价实现 |
| 算子 top-N 集中 norm/rope/attention | 纯计算热点 | §4 patcher 或融合算子 |
| 显存异常 / OOM / 长序列 | 显存池 / 增长 / shape | §2 显存 |

## 1. 零代码（环境变量先把一轮跑通）

不改代码、不动计算路径，所有模型先试一遍；非对应瓶颈无收益（A/B 只差 env 时口径必须一致）。**基线直接用 `../../assets/env_npu.sh`**，变量含义与出处见 `../appendix-env.md`。

- **禁止往 `env_npu.sh` 里累加变量**：交付件里的大部分环境变量用不上，VLA 场景尤其如此。收录判据就两条：VLA/VLM 系通用**且偏离官方默认值**（写默认值 = no-op）。语料里那些 `HCCL_WHITELIST_DISABLE=1`、`ASCEND_GLOBAL_LOG_LEVEL=3` 之类属历史抄写，不必跟。
- **按瓶颈开的项**（开了记进档案卡口径）：`MULTI_STREAM_MEMORY_REUSE=2`（FSDP2）、`TORCH_HCCL_ZERO_COPY=1`（仅 A3、beta）、`COMBINED_ENABLE=1`。
- **HF32**（昇腾版 tf32）：`torch_npu.npu.aclnn.allow_hf32 = True`，DiT/扩散类可试；conv 密集感知主干相反 `torch.npu.conv.allow_hf32 = False` 保精度（LMDrive）；与确定性开关是否互斥建议实测确认。
- **私有格式**：`torch.npu.config.allow_internal_format = False` 关闭 NZ 私有格式，属可尝试项（非迁移必做，见 1-migrate.md §3.3）。
- **确定性**：精度跑开 `HCCL_DETERMINISTIC=true` + `CLOSE_MATMUL_K_SHIFT=1`；性能跑与交付件关（取值、代价、与 HF32 的关系见 `../appendix-env.md` §三）。
- host bound 时 `gc.disable()` / `gc.set_threshold()` 减少 GC 停顿。
- ⚠️ **`ASCEND_LAUNCH_BLOCKING=1` 只能 debug 阶段临时用**：算子同步下发会使性能大幅下降，并让 `TASK_QUEUE_ENABLE` 整体失效。**性能跑与交付件里必须删掉**（语料 67 个模型一次都没出现过，出现即漏删；详见 `../appendix-env.md` §四，交付前自查见 3-deliver.md）。
- 注意：env 改数值不算完成标准里的优化；A/B 只差 env 时两次跑口径必须一致。

## 2. 显存

显存/OOM 处置序（成本递增，生效即停）：① 跑满卡 → ② 减 batch/seq（不改代码）→ ③ 代码级（重计算/reduce dtype，逐个精度对齐）。
env 开关作辅助；`ASCEND_LAUNCH_BLOCKING` 只作诊断，禁作交付（见 §1）。
任何一步后 re-profiling + 口径注册。

| 手段 | 说明 |
| --- | --- |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 内存池扩展段，长序列模型或对显存要求较高时建议开，用来降低显存 |
| `MULTI_STREAM_MEMORY_REUSE=2`（FSDP2） | **FSDP2 场景实测 `=2` 很有效**：大幅减少显存波动、避免内存重整、提升性能，优先试 2（先验证精度无误再采纳）；FSDP1/DDP 用 1。取值含义与出处见 `../appendix-env.md` §二 |
| `torch.npu.empty_cache()` | 每 step 末尾（RL 场景） |
| `torch.npu.synchronize()` 插在 attention 后 | 修 FSDP2 + DiT 显存增长 |
| 辅助模型 CPU offload | 绕过 pipeline 的 device/dtype 内省（`to()` 吞掉、`device`/`dtype` 属性谎报），模型实际留 CPU |

## 3. FSDP wrap 粒度

「未掩盖通信占比高」的第一处置点。按算子粒度包 FSDP（`transformer_layer_cls=[nn.Linear, nn.Embedding, ...]`）会使通信次数激增，一般按 **decoder block / DiT block 粒度**包。不是硬性要求，两种粒度在目标卡数实测，结论会随卡数翻转。参考 Lingbot-vla 让模型自己暴露要包的类。改变规约顺序，必须做精度对齐。

## 4. 融合算子（norm / RoPE / attention）

何时做：profiling 确认该算子在热环、Vector/访存占比高之后。注意力见 §4.4。

### 4.1 先找对对象（RMSNorm / RoPE）

改任何算子前先语义扫描：

```bash
python ../../scripts/discover_fused_patchers.py --source-root <模型源码根>
```

输出是**定位清单**（文件:行号 + 签名 + eps 属性名 + 与替换形态是否匹配），不是成品。

判定（语义特征，类名前缀只是提示）：RMSNorm 类 = forward 含手写 `rsqrt`（`pow/mean/rsqrt` 形态）且未含 `npu_rms_norm`；RoPE 函数 = 含 `rotate_half`（或内联 cos/sin + 负切片形态）且未含 `npu_rotary_mul`。扫描结果是候选列表，是否替换结合调用栈确认热环。

决策：目标是**把热环里的手写 RMSNorm / RoPE 换成融合算子**（这类替换收益通常在 10% 量级，值得优先找），按类定义在哪选路径——① 在库里 → §4.2 mx_driving patcher；② 在模型仓内（副本 / 自定义命名）→ §4.3 直接改源码；③ 扫描全空或不在热环 → 不动。两条路径收益同源，差别只在改动落点。

**判据是「类定义在哪个模块」。** 框架常 vendor 自己的模块副本（类名与 transformers 同名但 `__module__` 不同）；mx_driving 的 Patch 打的是带 `transformers.` 前缀的**全限定库路径**，只命中 site-packages，碰不到仓内副本——按类名去 `add(TransformersNPU)` 会打到框架实际不用的类 = 零收益。cosmos3 即此例：`Qwen3VLTextRMSNorm` 定义在 `cosmos_framework/.../qwen3_vl/qwen3_vl.py`，该模型最终 0 处用 patcher、全走 §4.3。**源码里常同时存在多个副本**（不同变体各一份，还可能互相 import），只有当前加载的那个在热环，先按配置 / import 链筛掉死代码。

### 4.2 mx_driving.patcher（Qwen-VL 系现成补丁）

属性：几乎全是性能优化（融合算子、预处理到 device）。**按命中对象分三类**：① transformers 库级补丁（`TransformersNPU`/`QWen3VLNPU`：RMSNorm / RoPE / 占位掩码 / 图像预处理）：使用了 transformers 内 Qwen 系列组件时必启用，收益较大；② diffusers 补丁：diffusers DiT 类用；③ 感知类（MSDA/DeformConv/bev/spconv 等）：感知/3D 任务用，纯 VLM/VLA 用不到。

安装：**见 1-migrate.md §1.5**（源码编译首选，无源码或编译失败时降级到预编译包；自查见该节）。

用法（放所有会被 patch 的 import 之前）。整套跑通用预置 patch：

```python
from mx_driving.patcher import Patcher, TransformersNPU
Patcher().add(TransformersNPU).apply()
```

逐项优化 / 定位问题时改成单点 `add`，按序逐项加、每加一项对齐一次，不要一次 add 全套（出问题无法定位）：

| 序 | Patch | 闸门（不匹配静默 no-op，回读 forward 确认是否生效）/ 备注 |
| --- | --- | --- |
| ① | `BaseImageProcessorFast_preprocess` + `Qwen2VL_preprocess` | transformers 4.57.x |
| ② | `Qwen3VLTextRMSNorm` | 4.x(x≥57) 或 5.x |
| ③ | `Qwen3VLRoPE`（风险最高，单独加单独验） | 4.x(x≥57) 或 5.x |
| ④ | `Qwen3VL_get_placeholder_mask` | 4.x(x≥57) 或 5.x |
| — | `Qwen3RMSNorm` / `Qwen3RoPE`（Qwen3 文本侧） | 4.x(x≥51) 或 5.x |
| — | `FlashAttention` / `FlashAttentionVarlen` | 仅 4.51/4.52（4.53+ 上游已内建） |

① 需配套 torchrun。

其它可用 API：`skip_import("flash_attn")`（import 变 no-op，避免污染整包）、`replace_import(mod, Name=Impl)`、`disable(name)`、`brake_at(N)`（跑 N 步自动停，性能测试专用）、`apply(allow_internal_format=True|False)`（NZ 私有格式开关，与 §1 的 `torch.npu.config.allow_internal_format` 是同一件事的另一个入口）。

预置 patch：`TransformersNPU` = 表中 Qwen 系 + FlashAttention 那几项；`QWen3VLNPU` = 只含 Qwen3-VL 三件套（②③④）。

### 4.3 落地形态与模板

「诊断 grep → 原始形态 → 优化形态 → 边界」全文见 `../ref-ops.md`：

| 算子 | 原始形态（手写，未走融合内核） | 优化形态 |
| --- | --- | --- |
| RMSNorm | `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w`（含 `.to(fp32)` 分段） | `torch_npu.npu_rms_norm(x, w, epsilon=eps)[0]` |
| RoPE | `(q * cos) + (rotate_half(q) * sin)`（或内联负切片形态） | `torch_npu.npu_rotary_mul(q, cos, sin)` |
| activation | `nn.GELU(approximate="tanh")` | `torch_npu.npu_fast_gelu` |

注意：**测收益必须用精确算子名**（`torch_npu.npu_rms_norm`），不要用 `F.rms_norm` 当等价替代：两者数值一致（实测 max|Δ|=0），但走的不是同一条内核路径，用它测会得不出真实收益（耗时差异未实测，这里只保证"不要用它替代"）；cos/sin layout 用错静默算错（1-migrate.md §3.4 陷阱）；属性名不统一（transformers `variance_epsilon` vs diffusers `eps`），替换前先看类里实际叫什么；带 bias 的变体 bias 在 kernel 外加法。融合改变规约/累加顺序，bf16 下 ~1e-2 量级数值差属正常，做精度对齐即可。

落地：**直接改源码**，就是上表那两行的形式；交付物本来就是 patch，不必绕道外挂 patcher 文件。A/B 开关用 git（`git stash` / `git checkout -- <file>`）。改完按 §9 做精度对齐。行上留标记（见开篇纪律），示例：

```python
# [可选优化] 融合算子，+X% 吞吐；去掉不影响跑通
out = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
```

**不稳定的项**（只在特定 shape / 模型变体下出错）再加个模块级开关，它比整个 patcher 文件轻，出问题的人改一个字就行：`_USE_FUSED_NORM = True`。

### 4.4 注意力后端（eager / sdpa → FA）

何时做：默认配置下未使用 FA 时。

- **transformers 系**：`attn_implementation="flash_attention_2"`（后端本体与陷阱见 1-migrate.md §3.4）。4.53+ 由 transformers 自带；4.51/4.52 上 attention mask 缓存那部分由 mx_driving 的 `FlashAttention` / `FlashAttentionVarlen` 补（闸门见 §4.2 表）。
- **非 transformers**（DiT / 自研塔 / `nn.MultiheadAttention`）：手写 `npu_fusion_attention`，三个陷阱（mask 极性、RoPE 半宽 vs 全宽、cos/sin 广播维）见 1-migrate.md §3.4，调用约定见 `../ref-ops.md`。

两类都改变规约顺序，按 §9 做精度对齐。

## 5. Host-bound：device 在等 host

profiling 表现：空闲占比高、kernel 间隙大、AICore 利用率低但 step time 长。

① **循环单点计算 → 向量化**（`grep -n "for .* in range"` 模型 forward 与数据处理路径）。
② **CPU 计算转 NPU 小算子拼接**：`torch.linalg.*` 等无 NPU 算子静默走 CPU（advisor AICPU 命中），用等价组合拼（如 SVD 拼 logdet）。
③ **减少同步**：`.item()` / `.cpu()` / `.numpy()` / `.tolist()` 会插 `aclrtSynchronizeStream` 打断流水；日志降频（`accelerator.gather(...).item()` 是全设备同步，把 gather/logging 块挪进 `if need_log`）。
④ **数据加载**（若 Slow Dataloader 命中，见 §7）。NVML 手工绑核（`os.sched_setaffinity`）在迁移时应删掉，改用 `CPU_AFFINITY_CONF`（见 §1）。留着是死代码，且和 `CPU_AFFINITY_CONF` 抢同一件绑核的事。

## 6. batch 与 cube 利用率

两个机制：① **摊薄每步固定开销**（micro_batch 太小时 FSDP 通信、host 算子下发这些与 batch 无关的开销占大头；加大 batch 会同时改变瓶颈类别：同一个模型小 batch 是通信受限，大 batch 可能已变计算受限）；② **使 cube 利用率饱和**（M 维太小吃不满 AI Core 固定分块，VLA 的动作头尤其容易触发：动作 chunk 维度短）。

注意：访存型模型（elementwise 占大头、`aic_mac_ratio` 恒低）加大 batch 后 cube 利用率仍低 → 是访存受限不是未充分利用，该转向 elementwise 融合 / 减 Cast / HF32（§4、§1），不要继续加大 batch。

手段：

- **吞吐诊断**：micro_batch ∈ {8, 16, 32, 64}（显存放得下继续翻倍）各跑 N 步（N≥20，取 StepTime **均值**并跳过 warmup，口径同 §9），`samples/s = global_batch_size / step_time`，画曲线找平台期；配合 profiling 看通信占比（高 → 继续加 batch；计算占比高 → 上 §4/§1 计算侧手段）。注意 plateau 后继续加大 batch 收益很小。
- **保精度收敛**：加大 micro_batch + 梯度累积维持 global batch；显存不够先 §2 的 `expandable_segments` / gradient checkpointing 腾空间；动作头/小 MLP 的逐样本循环批量化。

注意：改 batch 会改 FPS 口径，前后对比必须同一 batch 配置；直接加大 global batch 只是诊断手段、不是精度收敛配置（要重标学习率/热身）。

## 7. 数据加载

- `pin_memory=True` 必须开（NPU 上）；H2D 拷贝放预处理前、包在 `autocast(enabled=False)` 内；预处理不要放在 `model.forward`（逐样本 PIL/numpy 转换挪到 Dataset / collate_fn）
- 视频解码类 `num_workers` / `prefetch_factor` 可能需很大（32 / 8）；每样本 open/seek/decode/close 是最差模式，考虑帧缓存或换后端（1-migrate.md §1.4）
- 耗时构造（数据集、norm stats 全量扫描）做磁盘缓存
- `num_workers` **不破坏单臂可复现性**（worker 由 `base_seed + worker_id` 播种，同 seed 同值两跑实测逐元素一致 max|Δ|=0），但它**换了随机数的来源**：`0` 走主进程全局流（与模型初始化、训练循环共用一条），`>0` 走各 worker 自己的流；实测同 seed 下 `0` 与 `2` 拿到的数据 max|Δ|=3.96。所以 **`num_workers` 是 A/B 口径的一部分，两臂必须同值**（§9 复采时连同卡数、batch 一起核对）；精度跑与性能跑各自固定一个值即可
- 启动开销：`from_pretrained` 后紧接 `load_state_dict` 覆盖 = 重复执行一遍 CPU 全参数随机初始化（torchrun 下每个 rank 重复），加 `low_cpu_mem_usage=True`（meta device 建模型直接 materialize）

## 8. 优化器

- `torch.optim.AdamW(..., fused=True)` 首选（一行即开，与 MindSpeed 底层同一套算子，不用引依赖）；梯度裁剪 `optimizer.clip_grad_norm_fused_()` 替代 `clip_grad_norm`。写法/等价项/边界见 `../ref-ops.md` 优化器节。
- 注意：非万能（个别模型实测回退），按模型 A/B 后保留；`master_weights=True` 去掉（`capturable=True` 可留，均以实测为准）；仓里 `fused=False if is_torch_npu_available()` 的规避写法不要照搬（个别中间版本问题已修）；自建 fused（HF Trainer 场景）step 张量要在 device 上。换实现改变累加顺序，必须精度对齐。

## 9. 优化项记录与收益口径

- 性能跑用 `patcher.brake_at(N)` 或 `--max_steps` 提前刹车，与精度跑分开。`FPS = global_batch_size / step_time`，step_time 取后 N 步**均值**（跳过 warmup）；脚本与日志提取见 `3-deliver.md`。
- **精度对齐前置（步骤 0）：先固定全局随机种子**。只固定 dataloader 局部 generator 不够：
  模型随机初始化部分每进程初值不同，A/B 轨迹差无法归因于优化项（实测噪声可把 max|Δloss|
  0.245 抬到 1.470）。训练入口调 `seed_all(seed, is_npu=True)`（`../../assets/seed_all.py`），
  须在模型初始化之前。先自证「同卡同 seed 跑两遍逐 step loss 一致」再开始任何 A/B。
- 每项优化完成后：① **精度对齐**（同上，固定 seed 前提下前后 loss 曲线一致、无异常值；有特殊替换做模块级对拍：同一输入对比替换前后 logits / mask 应用，**容差按 dtype 分档：fp32 用 1e-4~1e-3，bf16 放宽到 ~1e-2**（融合改变规约/累加顺序所致，同 ../ref-ops.md））；② **再采一次 profiling** 与开箱基线对比计算 / 通信 / 空闲三项，收益要能对上手段（只有 FPS 变好但三项说不清变在哪 = 噪声或口径不一致；卡数、batch、num_workers 两次必须一致）；③ 数值填进**模型档案卡的「优化记录表」**（列定义见 `../../assets/model-card.template.md`；**失败项也记一行**，避免重复尝试）。该表是交付时 MIGRATION_LOG「优化项」段的来源（`3-deliver.md`）。

- 阶段结束时（进 C 前）把优化清单压成 ≤10 行写进档案卡「检查点」B 结论字段（`../../assets/model-card.template.md`），此后决策只引用档案卡，不回溯 A/B 过程细节（见 SKILL.md 上下文预算）。

计算侧优化的收益会被通信占比稀释（省下的计算时间有一部分本来就掩盖在通信里），结论必须在目标卡数、目标 batch 口径下复测；未掩盖通信占比高时先做 batch 扫描摊薄（§6），摊不掉再转向通信侧（FSDP 粒度、通信包对齐）。

## 10. profiling 失败降级

- **设备侧解析器系统性失败**：host 侧 `trace_view.json` 能导出但 `CANNTimelineParser` 等解析器全 `run failed`，缩小数据量复测仍失败 → 版本不兼容（torch_npu 与 CANN 不匹配），advisor 缺「整网耗时拆解」只剩 env 建议 + 亲和 API。升级配对或按下面三种判定手段拿结论。
- **无 kernel 时间线判定 device-bound 的三种手段**：① 框架自带 phase 计时（forward/backward/optimizer/dataloader 墙钟）：host 总墙钟明显小于 step 时间 → host 侧跟不上 device；② 规模缩放：token/seq 翻 N 倍看 step 是否近似线性（通信与 batch 无关是常数项，线性扩展 = 计算主导）；③ env A/B：`TASK_QUEUE_ENABLE` 0/1/2 与 `CPU_AFFINITY_CONF` 0/1/2 各跑一遍全部 ≈ 基线 → host 侧无瓶颈。

→ 优化完成进入交付：`3-deliver.md`
