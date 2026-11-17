# 参考模型索引 + 扩散/DiT 专项（查询用）

按需查询，不必通读。第一部分在迁移阶段流程 3（找参考模型）读；第二部分仅当目标是 DiT / 扩散 / 视频 / TE 结构时读。

找参考模型的方法：按**主干 + 头部 + 分片 / 并行策略**三个维度匹配，挑 2~3 个最像的，读其 patch。

表里的「分片 / 并行策略」只记**分布式方案本身**——它决定去查哪一节：FSDP / DDP → `flow/2-optimize.md` §3（wrap 粒度）；DeepSpeed ZeRO → `flow/1-migrate.md` §3.3（必打 fp64 补丁）；Megatron TP / CP → 本文第二部分。训练框架（Accelerate / Lightning / mmcv runner）与启动器（torchrun / `accelerate launch`）不占这一列。

语料根目录：`DrivingSDK/model_examples/` → <https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples>。表中任一模型名可拼链接核对。

# 第一部分：按结构找参考模型

## VLA / 机器人策略

| 模型 | VLM 主干 | 头部 | 分片 / 并行策略 | 最适合参考什么 |
| --- | --- | --- | --- | --- |
| Lingbot-vla | Qwen2.5-VL | π0 flow-matching | FSDP2 | π0 系 VLA 首选参考：RMSNorm/RoPE/eager-attn 替换、FSDP2 wrap 粒度、模型自己暴露 wrap 类 |
| GR00T-N1.7 | Qwen3 | flow-matching | DeepSpeed ZeRO | hccl/nccl 双路径最干净；attention 后端选择写法；one-hot 替代花式索引 |
| GR00T-N1.6 | Eagle/Qwen | flow-matching | DeepSpeed ZeRO | attention import 别名写法 |
| GR00T-N1.5 | Eagle | flow-matching | DeepSpeed ZeRO | 内容最全（fused AdamW、TrainingArguments、launcher）；attention 部分是早期手写，不要参考 |
| StarVLA | Qwen3-VL | — | DeepSpeed ZeRO | attention 后端选择写法；带保护的 torch_npu 引入；`TORCH_HCCL_ZERO_COPY` 卡型注释 |
| Pi-0 | PaliGemma | flow-matching | DDP | Accelerate 场景绕过 GradScaler；`env_npu.sh` 模板；ModelZoo 收尾块 |
| Pi-0.5 | PaliGemma | flow-matching | DDP | 同 Pi-0，另含 openpi 上游 patch |
| OpenVLA | Prismatic/Llama | 离散动作 | FSDP1 | 最小外挂 monkey-patch 模块；`is_bf16_supported` 替换 |
| Dexvla | Qwen2-VL | 扩散动作头 | DeepSpeed ZeRO | 多 stage 训练脚本；bitsandbytes |
| FastWAM | — | 视频世界动作模型 | DeepSpeed ZeRO | DeepSpeed ZeRO fp64 修复；日志降频 |

## 扩散策略 / 规控

| 模型 | 结构 | 分片 / 并行策略 | 最适合参考什么 |
| --- | --- | --- | --- |
| Diffusion-Planner | 扩散策略 | DDP | 工作量下限标定：纯扩散结构几乎开箱可用（`gc.disable()`、DDP `find_unused_parameters`） |
| DiffusionDrive | 扩散 + 感知 | DDP | Gen-3 patcher 最佳实践（`migrate_to_ascend/`）；A3 关 TensorIndex |
| DiffusionDriveV2 | 扩散 + RL(GRPO) | DDP | 同 DiffusionDrive 的新式迁移（`migrate_to_ascend/`，无 .patch）；截断扩散 + 锚点内优势估计 |
| GameFormer-Planner | Transformer 规控 | DDP | `npu_fusion_attention` 调用范例 |
| QCNet / HiVT / MultiPath++ | 图网络轨迹预测 | — | `torch_scatter` 替换；CPU→NPU 算子拼接（slogdet） |

## 世界模型 / 视频生成 / 扩散大模型

| 模型 | 主干 | 头部 | 分片 / 并行策略 | 最适合参考什么 |
| --- | --- | --- | --- | --- |
| Cosmos-Transfer1 | T5 + CLIP | DiT (TE) | Megatron TP + CP | fp64 采样器降精度；TE→Megatron TP 转换；朴素 all-gather CP；VAE `map_location` |
| Cosmos-Predict2 | T5 | DiT (TE) | FSDP2 | FSDP2 显存无界增长修复；EMA RNG 错位修复；pynvml 剥离；辅助模型 CPU offload |
| Cosmos-Drive-Dreams | — | DiT | — | 同 Transfer1（复用其 patch） |
| Cosmos-Reason1 | Qwen2.5-VL | RL (GRPO) | FSDP2 + TP（rollout 侧） | transformers NPU shim 用法；RoPE layout；手写 HCCL ctypes；混合后端 |
| OpenDWM | CLIP | diffusers DiT | — | patcher 打到 diffusers 库（`normalization.RMSNorm`）；CLIP `"BSH"` layout；数据集 pickle 缓存 |
| NWM | — | DiT | — | `npu_fast_gelu`；`allow_hf32=True`；`NpuFusedMultiheadAttention` 整体替换 |
| MagicDrive-V2 | — | STDiT | ZeRO + 序列并行 | xformers 关闭；确定性对比；norm CSE |
| VGGT | ViT | 几何回归 | — | H2D 拷贝时机；fp64 位置编码 |

## VLM / 驾驶大模型

| 模型 | 主干 | 分片 / 并行策略 | 最适合参考什么 |
| --- | --- | --- | --- |
| Cosmos3 | Qwen3-VL / Qwen3-VL-MoE / Nemotron-3-Dense-VL | FSDP2 | **Qwen3-VL 系 vendored 副本的改源码范例**：RMSNorm/RoPE/VAE 都直接改仓内文件（**0 处用 mx_driving patcher**，印证 2-optimize §4.1 的「扫描命中的是仓内定义」）；自带 flash2 注意力层；pynvml 用 try/except 降级而非删除 |
| Senna | LLaVA + Vicuna | DeepSpeed ZeRO | bitsandbytes NPU 编译（唯一）；bf16 关闭；zero3 batch 下调 |
| LMDrive | BLIP2 + Vicuna (LAVIS) | DDP | 大规模 import 裁剪（LAVIS/timm 坏 import 毒化整包）；`torch_scatter`→mx_driving；`conv.allow_hf32=False`；fp64 all-reduce |
| DriverAgent | Attention + LSTM | — | fp32 孤岛系统化范例；LSTM 在 autocast 下 NPU fp16 反向 bug |

## 感知 / BEV / 点云（本 skill 覆盖较浅，需要时直接读）

| 类别 | 模型 |
| --- | --- |
| BEV | BEVFormer, BEVDet, BEVDet4D, BEVDepth, BEVNeXt, MatrixVT, BEVFusion |
| OCC | PanoOcc, FlashOCC, FBOCC, TPVFormer, SurroundOcc |
| 端到端 | UniAD, VAD, SparseDrive |
| 视觉检测 | Sparse4D, StreamPETR, DETR, DETR3D, Deformable-DETR, FCOS, FCOS3D, Mask2Former, CenterNet, Yolov8, DinoV3 |
| 轨迹预测 | DenseTNT, HPTR |
| 点云 | PointPillar, CenterPoint, PointTransformerV3, SalsaNext, Panoptic-PolarNet |
| 车道线/地图 | MapTR, MapTRv2, PivotNet, LaneSegNet |

共性：依赖 mmcv/mmdet/mmdet3d（需配套 patch）、大量 mx_driving 融合算子（MSDA / bev_pool / spconv / voxelization）、Gen-1 或 Gen-3 patcher。

## 按需求反查

| 我要解决的问题 | 去看 |
| --- | --- |
| Qwen-VL 系主干怎么改 | Lingbot-vla, GR00T-N1.7, StarVLA, Cosmos-Reason1 |
| flow-matching / 扩散动作头 | Diffusion-Planner（下限）, Lingbot-vla, Cosmos-Transfer1（采样器 fp64） |
| diffusers DiT | OpenDWM（库级 patch）, NWM |
| FSDP2 | Cosmos-Predict2（显存）, Lingbot-vla（wrap 粒度） |
| DeepSpeed ZeRO | FastWAM（fp64 修复）, Senna, Dexvla |
| Megatron TP / CP | Cosmos-Transfer1（唯一） |
| RL / vLLM | Cosmos-Reason1（唯一） |
| 量化 / bitsandbytes | Senna（唯一）, Dexvla |
| 视频解码 / VAE | Cosmos-Transfer1（decord）, FastWAM（torchcodec）, GR00T-N1.5（torchvision_av） |
| 无 .patch 的新式迁移 | DiffusionDrive, DiffusionDriveV2, PanoOcc（`migrate_to_ascend/`） |
| 完全原生 NPU 实现 | DriverAgent（无上游，直接写） |
| 交付物模板 | SparseDrive（最完整 README）, Pi-0（VLA 最干净） |
| 性能脚本模板 | Lingbot-vla（`train_16p_perf.sh`）, Pi-0（ModelZoo 收尾块） |

## 已完成案例

与上面各表不同，本表为按本 skill 流程跑通并交付的案例（非语料索引）。其中 loss 均无异常值，短跑区间内呈**下降趋势**。

| 模型 | 上游仓库 @ commit | 主干 / 头部 / 并行策略 | 性能（开箱 → 优化后，口径随行） | 状态 |
| --- | --- | --- | --- | --- |
| Spirit-v1.5 | `Spirit-AI-Team/spirit-v1.5` @ 6d67377 | Qwen3-VL-4B-Instruct / BaseDiT 16 层（hidden 1536 / 32 head） / FSDP1 | 8p · global 256 · 稳态窗口均值：**49.7 → 72.2** samples/s | A✅ B✅ C✅ |
| FasterWAM | `hustvl/FasterWAM` @ 8366781 | Wan2.2-TI2V-5B / SparseActionDiT（0.56B，30 层/hidden 1024） / DeepSpeed ZeRO-1 | 8p · global 128 · 稳态窗口均值：**37.29 → 45.89** samples/s | A✅ B✅ C✅ |

**新模型迁移完成后在本表加一行**（行格式见列头）。

---

# 第二部分：扩散 / DiT / 视频 / TransformerEngine 专项

适用：模型含 DiT、flow-matching / 扩散采样器、视频 VAE、时序注意力、T5/CLIP 文本编码器。

## DiT 里的 attention 选型

部分模型保留 SDPA：Cosmos-Transfer1 的 DiT 保留 `F.scaled_dot_product_attention` 只调 layout；MagicDrive-V2 只是关掉 xformers。`npu_fusion_attention` 只用在原本就用 flash-attn 的地方（ViT、CLIP 文本塔、MHA）。

不是硬性要求，只反映「多数模型没必要换」。评估时区分两类改动：**换算法**（SDPA→融合算子，改变数值路径，谨慎）vs **兼容性修复**（如 diffusers `AttnProcessor2_0` 修 mask 广播，报错而非静默，按需加）。

唯一例外：NWM 的 `NpuFusedMultiheadAttention` 整体替换 `nn.MultiheadAttention`（`input_layout="BSND"` 免转置）。**它是 NWM 仓内自建模块（`from npu_fused_modules import ...`），不是 torch_npu 的现成 API**，不要到 `torch_npu.*` 里找。它 `return attn_output, None` 写死，所以永远返回 `attn_output_weights=None`：`need_weights=True` 会**静默**拿到空值。

## 采样器 fp64 → fp32

最重要的一条（见 flow/1-migrate.md §3.3）。指数积分器 / DPM-Solver++ / RK2 的 `phi1`/`phi2`/`expm1` 及所有中间变量都要降精度。

## TransformerEngine：优先换 TransformerEngineNPU（不适配再剥离）

昇腾有同名适配库 `TransformerEngineNPU`（gitcode.com/Ascend/TransformerEngineNPU，包名仍是
`transformer_engine`，提供 `te.pytorch.*` 同名接口；Megatron-LM 场景配 `megatron_adaptor` 一行启用；
当前配套参考：Mcore 0.18.0 / CANN 9.0.0 / PyTorch 2.7.1 / torch_npu 26.0.0 / Python 3.12）。
处理顺序：

1. 环境达标 → **替换为 TransformerEngineNPU**（与 CUDA 版 TE **不能同装**，先卸载再装；按仓库 README
   配套表核对 CANN/torch/torch_npu/triton-ascend/Python 版本；不支持 NVFP4 量化）。
2. 版本不满足 / 只想最小跑通 → 按下面剥离降级（Cosmos 系 DiT 全构建在 TE 上，Tier 1 不处理跑不起来）。

(a) TE → 普通 torch（无 TP，最简单）：

```python
-import transformer_engine as te
+import torch_npu
-backend: str = "transformer_engine",
+backend: str = "torch",
-self.q_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)
+self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
-self.t_embedding_norm = te.pytorch.RMSNorm(model_channels, eps=1e-6)   # 最容易漏
+self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)
```

`t_embedding_norm`（timestep embedding 的 norm）也是 TE RMSNorm，最容易漏。TE RMSNorm 的替代实现见 `ref-ops.md`（带 bias 那版）。相关清理：CUDA-graph 辅助函数（`get_all_rng_states` / `graph_safe_rng_available` / `TransformerEngineBaseModule`）改成 no-op。

(b) TE → Megatron TP（仅当模型原本就用 TP）：`te.pytorch.Linear` → `ColumnParallelLinear` / `RowParallelLinear`。三个陷阱：① 返回 tuple，每个 call site 要解包 `layer(x)[0]`；② 输入必须 `.contiguous()`；③ wgrad helper 需加 5-D 分支，且必须同时 patch `megatron.core.utils` 和 `megatron.core.tensor_parallel.layers`（只打一个是静默 no-op）。另需 `config.gradient_accumulation_fusion = False` 和显式 RNG tracker 初始化。

(c) CP / SP（仅当模型原本就用）：全仓没有模型在 NPU 上跑通 ring attention / ulysses；唯一 CP 实现是朴素 KV all-gather + 负载均衡 chunk 重排 + 完整 SDPA（Cosmos-Transfer1），需三次显式 synchronize；SP 用 ColossalAI zero2-seq 的 ascend 分支（MagicDrive-V2）。迁移阶段建议先关掉（`cp_size=1` / `tp_size=1`），用 batch 与梯度检查点解决序列长度问题。

## diffusers 库级 patch（对 diffusers-DiT 最有用）

`OpenDWM/tools/patch.py`：一处覆盖所有 qk-norm（`AdaLayerNormZero` / `FluxTransformer2DModel` / `SD3Transformer2DModel` 都走它）：

```python
def RMSNorm_forward(normalization: ModuleType, options: Dict):
    def _RMSNorm_forward(self, hidden_states):
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
    if hasattr(normalization, "RMSNorm"):
        normalization.RMSNorm.forward = _RMSNorm_forward
```

属性名是 `self.eps`（diffusers）而非 `self.variance_epsilon`（transformers）。属优化阶段（flow/2-optimize.md §4）。

peft：`infer_device()` 在 NPU 上返回 "cpu"，monkey-patch 成 `lambda: "npu"`（Cosmos-Predict2）。

## 视频

- VAE 的 conv3d 本身可用，无需分解。TorchScript 加载的 VAE 必须显式 `map_location="npu"`（`torch.jit.load` 不在 transfer 值级实参覆盖范围；`torch.load` 的 `map_location="cuda"` 值级会被兜底）。
- 解码后端：`torchvision_av`（首选，免编译）> `av`+`imageio-ffmpeg` > `decord` / `torchcodec`（需编译）。`import decord` 必须在 `import torch_npu` 之前（flow/1-migrate.md §1.4）。
- 长视频 / 多视角显存：减小 `max_img_per_gpu` / `img_nums` / `batch_size`；FSDP2 + DiT 显存增长；`MULTI_STREAM_MEMORY_REUSE=2`（flow/2-optimize.md §2；FSDP2 场景实测有效：减少显存波动、避免内存重整、提升性能，先验证精度无误再采纳）。

## 文本编码器

- T5-XXL：离线预抽取 embedding，不在训练时跑；抽取脚本加 `transfer_to_npu` 头。注意多视角路径的 shape bug（`t5_embedding.reshape(-1, hidden_dim)` 后再用）。
- CLIP 文本塔：`"BSH"` layout 范例见 `ref-ops.md`；`attention_mask + causal_attention_mask` 先合并再传。

## 其它

| 项 | 说明 |
| --- | --- |
| 确定性 | 扩散 / flow-matching 采样要可复现就开（`HCCL_DETERMINISTIC` + `CLOSE_MATMUL_K_SHIFT`）；取值、代价、与 HF32 的关系见 `appendix-env.md` §三 |
| EMA | EMA 副本的随机初始化会消耗 RNG、打乱随机流 |
| DDP + 扩散 | `find_unused_parameters=True`（Diffusion-Planner）；CFG guidance fn 可先禁用 |
| 多阶段训练 | 扩散类常按 stage 拆脚本：`train_8p_stage{1,2,3}.sh` |
| 重复 norm 计算 | CSE 掉，既省算力又避免不确定性 |
| `xformers` | 一律关闭 |
| 数据顺序可复现 | `os.listdir()` 包 `sorted()`；时间戳种子换常量 |

## 工作量标定

**Diffusion-Planner 是最好的参照**：扩散策略，与 π0 系 flow-matching 动作头最同源，整个 patch 文件 123 行 / 6 个文件（净增删 35 行）：`import torch_npu` + `transfer_to_npu`、`gc.disable()`、去掉硬编码 `MASTER_ADDR/MASTER_PORT`（交给 torchrun）、DDP 加 `find_unused_parameters=True`、requirements 调整 + step loss 打印。

结论：纯 DiT/扩散结构在 NPU 上基本开箱可用。真正的工作量在 VLM 主干（attention / norm / rope / 图像预处理）和分布式包装上。
