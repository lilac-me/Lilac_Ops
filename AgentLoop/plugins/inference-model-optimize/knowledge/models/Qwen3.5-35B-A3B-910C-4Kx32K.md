# Qwen3.5-35B-A3B（MoE 256 experts / top-8，GQA kv_heads=2，线性注意力 GDN 变体）· 910C/A3（16 die）· 4K×32K 场景

## ① 模型与场景档案

| 项 | 值 |
| ---- | ---- |
| 模型名 | Qwen3.5-35B-A3B |
| 机型 | 910C/A3（16 die） |
| NPU 卡数 | 16 |
| 精度 | BF16 |
| 测试场景 | 4K×32K（4096 / 32768） |
| 并发 | 384（饱和点） |
| 优化目标 | 高吞吐 |
| 推理框架 | vLLM |

## ② 阶段最优配置结论

- **最优拓扑**：**TP4DP4EP16**（`--enable-expert-parallel`，EP=16 固定用满 16 die）
- **DCP**：**`-dcp 2`（--decode-context-parallel-size 2）**：kv_heads=2 → tp4/H=2，KV cache 复制 2x→1x，同显存 KV 容量翻倍
- **最优并发 / 饱和点**：**384**（256 / 384 峰值 / 512 回落）⚠️ 该 384 为 DCP 前后同并发对比；DCP 释放显存后并发上探（448/512）尚未实测，按 common-guide「DCP + 并发上探」方法论下一步应补测
- **一句话结论**：EP 必须开启 + DCP2 削减 KV 复制 + 高并发，是本场景最大提升来源（总提升 +97%）

## ③ 拓扑 / 参数对比

**拓扑对比（256 并发，EP16）**：

| 配置（TP×DP×EP） | 备注 |
| --- | --- |
| TP4DP4EP16 | 最优 |
| TP2DP8EP16 | 波动大（复测不稳定） |
| TP1DP16EP16 | |
| TP8DP2EP16 | 最初配置 |

**EP 开关**：EP16 vs EP-off（**+21.0%，EP 必须开启**）

**DCP 对照**：无 DCP 基线（DCP2 = +9.7%）；最初 TP8DP2EP16@256（总提升 **+97%**）

## ④ 性能数据与曲线

- **最优性能**：两轮复测一致；TPOT 41.9ms；TTFT 45.7s（DCP 通信开销，吞吐优先可接受）
- **DCP 场景**：DCP2 前后同并发对比增益 +9.7%；并发上探（448/512）尚未实测

## ⑤ 踩坑记录

| 日期 | 现象/报错 | 根因 | 处置 | 状态 |
| ------ | ----------- | ------ | ------ | ------ |
| 2026-08 | `HCCL_BUFFSIZE_EP is too SMALL` + `error 561002` | MoE dispatch/combine 前 CCL 缓冲不足；OPT-D 下 maxBs 256→512，`NEEDED_HCCL_BUFFSIZE=1185MB` | `--extra-env HCCL_BUFFSIZE=1300` 稳过 | 已解决 |
| 2026-08 | `TASK_QUEUE_ENABLE=2` 与图模式不兼容 | 特性冲突 | 不启用 | 已规避 |
| 2026-08 | `enable_kv_nz` raise | 仅 MLA 支持，本模型 GQA 不支持 | 关闭 | 已规避 |
| 2026-08 | `multistream_overlap_shared_expert` 与 `enable_fused_mc2=1` 互斥 | 特性冲突 | 不同时开启 | 已规避 |

> 经验备注：allgather 频次翻倍是 `shared_expert_dp` 副作用；DSARandomUniform/Sort 采样开销是次要瓶颈。OPT-D（`enable_shared_expert_dp`+`multistream_overlap_shared_expert`+`enable_prefill_mc2`+HCCL_BUFFSIZE=1300）吞吐持平（+0.45% 噪声内）但 stddev 369.5→37.1（稳定性 10 倍提升），可选稳定性变体。

## ⑥ 环境配置

- **启动命令**：服务端关键参数含 `--tp 4 --dp 4 --enable-expert-parallel --decode-context-parallel-size 2 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`；CANN `/mnt/share/b30079549/CANN/B150`；端口 8000
- **环境变量**：`VLLM_VERSION=0.27.1` `OMP_PROC_BIND=false` `OMP_NUM_THREADS=10` `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
- **其它**：`enable_cpu_binding` 默认 True 已生效；`fast_moe_cold_start=True` 权重加载 ~2.57s（缓存命中）
