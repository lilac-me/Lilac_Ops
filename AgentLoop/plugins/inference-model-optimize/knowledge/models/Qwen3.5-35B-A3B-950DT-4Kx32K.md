# Qwen3.5-35B-A3B（MoE 256 experts / top-8，GQA kv_heads=2，线性注意力 GDN 变体）· 950DT/A5（8 卡）· 4K×32K 场景

## ① 模型与场景档案

| 项 | 值 |
| ---- | ---- |
| 模型名 | Qwen3.5-35B-A3B |
| 机型 | 950DT/A5（8 卡） |
| NPU 卡数 | 8 |
| 精度 | BF16 |
| 测试场景 | 4K×32K（4096 / 32768） |
| 并发 | 640（峰值） |
| 优化目标 | 高吞吐 |
| 推理框架 | vLLM |

## ② 阶段最优配置结论

- **最优拓扑**：**TP4DP2EP8 + DCP2 + KVOFF** @640（TTFT 41.2s 回落档）；基础 DCP2 同档
- **最优并发 / 饱和点**：640（512→640 并发上探，+5.4%）
- **关键参数**：DCP2、KVCacheOffload（KVOFF）、并发上探
- **一句话结论**：DCP2 削减 KV 复制 + 并发上探是本场景最大杠杆；FLASHCOMM1/FUSED_MC2/SHARED_EXPERT_DP 均无显著增益

## ③ 拓扑 / 参数对比

| 配置 | 备注 |
| --- | --- |
| TP4DP2EP8 + DCP2 + KVOFF @640 | 最优 |
| TP4DP2EP8 + 基础 DCP2 | 与最优同档 |
| TP4DP2EP8 + DCP2 @512 | 并发上探前 |

## ④ 性能数据与曲线

- **最优性能**：@640（TTFT 41.2s 回落档）；基础 DCP2 同档
- **DCP 场景**：DCP2 前后 + 并发 512→640 上探 = +5.4%
- **负面经验**：TP1/DP8 曾被误判（HCCL_BUFFSIZE 未调 → 假并发边界），调 600 后合法 → **并发上探每档必须核实是否被 HCCL_BUFFSIZE 截断**（~90000 dispatch tokens 经验值不可作上界）
- FLASHCOMM1 / FUSED_MC2 / SHARED_EXPERT_DP 均无显著增益

## ⑤ 踩坑记录

| 日期 | 现象/报错 | 根因 | 处置 | 状态 |
| ------ | ----------- | ------ | ------ | ------ |
| 2026-08 | TP1/DP8 并发边界假象 | HCCL_BUFFSIZE 未调整，假并发边界 | `HCCL_BUFFSIZE=600` 后合法 | 已解决 |

## ⑥ 环境配置

- **启动命令**：服务端关键参数含 `--tp 4 --dp 2 --enable-expert-parallel --decode-context-parallel-size 2`（+KVCacheOffload）；CANN `/home/y00806874/ascend`（750DT）
- **环境变量**：`VLLM_VERSION=0.27.1` `OMP_PROC_BIND=false` `OMP_NUM_THREADS=10` `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
