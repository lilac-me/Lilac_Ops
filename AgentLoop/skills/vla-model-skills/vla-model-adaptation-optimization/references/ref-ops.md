# 算子对照（查询用）

按需查询，不必通读。融合算子替换属优化阶段（flow/2-optimize.md §4），除非不替换就无法运行。Attention 先看 flow/1-migrate.md §3.4。走 transformers 的模型用自带 NPU 后端即可，不需要本文的 `npu_fusion_attention`。

## 用到最多的算子（按「有几个模型用到」排，2026-09 实测）

```text
npu_fusion_attention 14    npu_rms_norm 12        multi_scale_deformable_attn 9
npu_rotary_mul 6           npu_max_pool2d 5       npu_deformable_conv2d 5
npu_add_relu 4             npu_assign_target_of_single_head 2   npu_voxel_pooling_train 1
```

数字会随语料增长，只用来判断「哪些算子值得先看」；**算子名以实际包的导出为准**（`torch_npu.*` / `mx_driving.*`），不要照本文档的字面拼写：MSDA 就是反例，`mx_driving` 导出的是 `multi_scale_deformable_attn` 和 `npu_multi_scale_deformable_attn_function`，**没有** `npu_multi_scale_deformable_attn` 这个名字。

完整清单：`DrivingSDK/docs/zh/api/context/`（63 个算子文档）；`mx_driving` 全量导出见其 `__init__.py` 的 `__all__`（实测 86 项）。

## `npu_fusion_attention`

适用边界：仅非 transformers 的自定义 attention（TE DiT、独立 CLIP/T5 塔、`nn.MultiheadAttention`、自研 ViT）。

### 调用约定

| 参数 | 约定 |
| --- | --- |
| `input_layout` | `"BSND"`（`(B,S,N,D)` 免转置）/ `"TND"`（varlen，替代 `flash_attn_varlen_func`）/ `"BNSD"` / `"BSH"`（已合并 head 维，免 reshape） |
| `keep_prob` | `1 - dropout_p`（与 dropout_p 相反） |
| `atten_mask` | **True = 屏蔽**（与 SDPA 相反，见 flow/1-migrate.md 3.4 陷阱 1）。causal 用上三角 bool + `sparse_mode=3`（3=右下对齐 causal，官方示例用 `torch.triu(ones(n,n,bool), diagonal=1)`；标准左上对齐 causal 用 `sparse_mode=2`，transformers 集成用 `NPU_FA2_SPARSE_MODE=2` 切换）；全 mask 传完整 mask + `sparse_mode=1`；非 causal 传 `None` |
| `scale` | `1.0 / math.sqrt(head_dim)` |
| `actual_seq_qlen/kvlen` | varlen 时 `tuple(cu_seqlens[1:].cpu().numpy().tolist())`（去前导 0；`.cpu()` 是已知 host 同步开销） |
| 返回值 | **tuple，永远取 `[0]`** |
| 其它 | `pse=None`；DiffusionDrive 还传 `pre_tockens=65536, next_tockens=65536, sync=False, inner_precise=0` |

### 范例

批量（Lingbot-vla，替换整段 einsum+softmax）：

```python
att_output = torch_npu.npu_fusion_attention(
    query=query_states, key=key_states, value=value_states,
    head_num=num_att_heads, input_layout="BSND",
    atten_mask=(~attention_mask).unsqueeze(1),      # 取反
    scale=head_dim**-0.5)[0]
att_output = att_output.reshape(bsize, seq_len, num_att_heads * head_dim)
```

varlen + causal（GR00T-N1.5）：

```python
atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(device)
output = torch_npu.npu_fusion_attention(
    query_states, key_states, value_states, head_num, "BSND",
    keep_prob=1.0 - self.dropout_p if self.training else 1.0,
    atten_mask=atten_mask_npu, scale=self.softmax_scale,
    actual_seq_qlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
    actual_seq_kvlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
    sparse_mode=3)[0]
```

配套的 `pad_input` / `unpad_input` **不能**从 `transformers.integrations.npu_flash_attention` 取：该模块的同名函数返回 5 元组（flash_attn 版是 4 元组），且 4.56 起已被移除（详见 flow/1-migrate.md §3.4 第 2 条）。

`"BSH"` layout（OpenDWM CLIP 文本塔，免 4-D reshape）：

```python
attn_output = torch_npu.npu_fusion_attention(
    query_states, key_states, value_states, head_num=self.num_heads, pse=None,
    atten_mask=attn_mask.bool(), input_layout="BSH",
    scale=1.0 / math.sqrt(query_states.shape[-1]),
    sparse_mode=1, keep_prob=1 - dropout_p)[0]
```

## RMSNorm → `npu_rms_norm`

诊断特征（手写 `pow/mean/rsqrt` 形态，未走融合内核）：

```bash
grep -rn "rsqrt\|variance_epsilon\|\.eps" <模型目录> --include=*.py
```

原始形态（NPU 上拆成 5+ 个逐元素 kernel：Cast/Pow/Mean/Add/Rsqrt/Mul）：

```python
# transformers LlamaRMSNorm 官方形态
hidden_states = hidden_states * torch.rsqrt(
    hidden_states.pow(2).mean(-1, keepdim=True) + variance_epsilon) * self.weight
# 自研变体常见：先升 fp32 再算（分段更多）
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
return (self.weight * hidden_states).to(orig_dtype)
```

优化形态（一个融合内核，返回 `(y, rstd)` 取 `[0]`）：

```python
return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
```

边界：

- **测收益必须用精确算子名**（`torch_npu.npu_rms_norm`）：`F.rms_norm` 能跑、数值也一致（实测 max|Δ|=0），但走的不是同一条内核路径，拿它测会把收益测没（耗时差异未实测）
- 属性名不统一：transformers 用 `self.variance_epsilon`，diffusers / 自研常用 `self.eps`，替换前先看类里实际叫什么
- 带 bias 的变体：bias 在 kernel 外面加：`y = npu_rms_norm(...)[0]; if bias is not None: y = y + bias`
- dtype 对齐：若 `weight.dtype != hidden_states.dtype` 且 weight 是 fp16/bf16，先 `hidden_states.to(weight.dtype)` 再调
- 权重初始化语义按模型原实现（`reset_parameters` 是 `zeros_` 还是 `ones_` 照搬）
- 替换改变规约/累加顺序，**容差按 dtype 分档：fp32 1e-4~1e-3，bf16 放宽到 ~1e-2**，做精度对齐（flow/2-optimize.md §9）

落地形态：`../scripts/discover_fused_patchers.py` 定位到具体文件:行号，然后**直接改源码**（flow/2-optimize.md §4.3）。注意改到模型实际 import 的模块：

```python
# 模型源码里的原实现（如 LlamaRMSNorm.forward）：
#   hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
# 改成：
out = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
```

## RoPE → `npu_rotary_mul`

诊断特征（手写旋转应用，Vector 开销高）：

```bash
grep -rn "rotate_half\|apply_rotary_pos_emb" <模型目录> --include=*.py
```

原始形态（每次前向重复构造旋转、逐元素组合）：

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim); sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin)
```

优化形态（一个融合内核）：

```python
q_rot = torch_npu.npu_rotary_mul(q, cos, sin)
```

边界：

- **cos/sin 两处陷阱，后果不同**（A3 实测）：① **内容（半宽 vs 全宽）**：上游 flash-attn 路径会先把 cos/sin 砍成半宽 `chunk(2, dim=-1)[0]`，`npu_rotary_mul` 要全宽（`rotate_half` 语义），两者**形状相同、用错静默算错**（实测 max|Δ|=7.9，不报错、loss 不收敛）；② **广播维（按 q 的 head 维对齐）**：规则是「**秩与 q 相同 + q 的倒数第 2 维（head 维）处为 1**」，其余维真值或 1 皆可。实测三种正确形态：q `[B,S,H,D]`+`[1,S,1,D]`、q `[B,S,H,D]`+`[B,S,1,D]`、q `[N,H,D]`（视觉塔）+`[N,1,D]`。**秩不匹配 / head 维没插 1 直接 RuntimeError**，定位成本低。详见 flow/1-migrate.md §3.4
- 只替换旋转**应用**这一步；`freqs` / `cos` / `sin` 的构造与缓存保留上游逻辑（Qwen 系 M-RoPE 3 维 position_ids、视觉 2D-RoPE 只负责喂 cos/sin）
- 上游 `apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=1)` 这类调用，cos/sin 的补维由**该函数自己**做；替换时把补好的 cos/sin 直接喂 `npu_rotary_mul` 即可。**不要指望算子替你对齐**，形状不匹配它会直接 RuntimeError（实测）
- 替换改变浮点顺序 → 优化阶段，做精度对齐

## 激活

```python
-approx_gelu = lambda: nn.GELU(approximate="tanh")
+approx_gelu = lambda: torch_npu.npu_fast_gelu
```

（NWM。lambda 返回的是函数而非 Module，timm 的 `Mlp` 会调 `act_layer()`，所以能工作。）

## 优化器

**首选：`fused=True`**：走的就是 torch_npu 接管的融合 AdamW 内核，与 MindSpeed 的 `AdamW` 底层是同一套算子，一个参数就能开，不用引入 MindSpeed 依赖：

```python
optimizer = torch.optim.AdamW(params, lr=..., betas=..., weight_decay=..., fused=True)
```

仓里能看到 `fused=False if is_torch_npu_available() else fused`（Lingbot-vla）这类禁用写法，不要照搬，那是个别 torch_npu 中间版本的规避手段，新版本已修复，先实测 `fused=True`，能跑就把规避去掉。

其它写法（读上游 patch 时会遇到）：

| 写法 | 出处 | 说明 |
| --- | --- | --- |
| `torch.optim.AdamW(..., fused=True)` | StarVLA | 推荐，成本最低 |
| MindSpeed：`torch.optim.AdamW = mindspeed...AdamW` | Pi-0、OpenDWM | 底层算子与 fused=True 一致，多一层依赖 |
| `torch_npu.optim.NpuFusedAdamW` | 官方文档、LMDrive | 等价可用 |
| 自建 `torch._fused_adamw_` 包装 | GR00T-N1.5、cosmos-rl | HF Trainer 场景下接入用 |
| 仅 `foreach=True` | NWM | 非融合，只是批量化，收益小于 fused |

要点：`master_weights=True` 是 CUDA 侧优化参数（NPU 官方无等价说明，NpuFusedAdamW 无此参数），建议去掉（实测确认）；`capturable=True` 保留但同样以实测为准；自建 fused AdamW 时 step 张量要在 device 上；`apex` 的 `amp_C.multi_tensor_l2norm` / `multi_tensor_scale` 需纯 torch 重写。换优化器实现改变累加顺序，属优化阶段，做精度对齐。

## 感知 / 3D（mx_driving 的感知部分，VLM/VLA 用不到；其 Qwen 系 patcher 见 flow/2-optimize.md §4.2）

| 原实现 | NPU 替换 |
| --- | --- |
| `MultiScaleDeformableAttnFunction.apply` | `mx_driving.multi_scale_deformable_attn(...)` |
| bev_pool CUDA ext | `from mx_driving import bev_pool, bev_pool_v2, bev_pool_v3` |
| `torch_scatter.scatter_max/mean` | `from mx_driving import scatter_max, scatter_mean`（注意 dtype：`feat.to(torch.float)`、`indices.to(torch.int32)`） |
| spconv | `from mx_driving import SparseConv3d, SubMConv3d` |
| `nn.MaxPool2d` | `mx_driving.npu_max_pool2d`（torch_npu.npu 命名空间无此接口） |
| ResNet 残差 `out+=identity; relu(out)` | `npu_add_relu` |
| `deform_conv2d` | `from mx_driving import deform_conv2d, DeformConv2dFunction` |

部分算子无 NPU 实现时回落 CPU：`boxes_bev_iou_cpu(box_preds.cpu(), cur_gt.cpu()).npu()`。

## 其它

| 场景 | 做法 |
| --- | --- |
| NCHW → NZ format | `torch_npu.npu_format_cast(x, 29)`（FRACTAL_NZ=29；30 是 NCDHW）。配置侧有两个等价入口：`torch.npu.config.allow_internal_format` 与 mx_driving 的 `Patcher().apply(allow_internal_format=...)` |
| 花式索引 `self.W[cat_ids]` | 改 one-hot matmul（NPU 上 index-select 慢，GR00T-N1.7） |
| `torch.logdet` / `torch.linalg.slogdet`（**静默走 CPU**） | `torch.svd` 拼：`log(prod(S))`（MultiPath++ 的 `custom_logdet` 就是这个） |
| 确定性种子 | `torch_npu.npu.manual_seed_all(seed)` + `manual_seed(seed)` |

**怎么发现「静默走 CPU」**：NPU 上没实现的算子会回落 CPU，只打一条 **C++ 侧 stderr 告警**（走 C++ 流，Python 的 `warnings` 捕获不到）：

```text
[W… VariableFallbackKernel.cpp] Warning: CAUTION: The operator 'aten::_linalg_slogdet.sign'
is not currently supported on the NPU backend and will fall back to run on the CPU.
```

profiling 里表现为 AICPU 算子命中（flow/2-optimize.md §0.3），但 advisor 不一定报得全。**跑测试时把 stderr 留档、`grep npu_cpu_fallback`** 是成本最低的自查手段。

## 已排除手段（实测无效，不要重复尝试）

- Liger FLCE → Ascend 退化为 materialized CE，无融合收益（三 agent 验证）
- ASCEND_LAUNCH_BLOCKING=1 → 慢 3.9×，只作诊断
- FSDP2 reduce_dtype 与默认一致 → 无额外收益

模型特有项（HF32/prefetch 等）只进该模型的踩坑记录（工作区）。
