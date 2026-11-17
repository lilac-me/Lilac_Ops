# DeepSeek-V4-Flash-BF16（MoE 256 experts top-6，MQA 1 头，DSA 压缩注意力，582GB）· 950DT/A5（8 卡 × 96GB）· 4K×32K 场景

## ① 模型与场景档案

| 项 | 值 |
| ---- | ---- |
| 模型名 | DeepSeek-V4-Flash-BF16 |
| 机型 | 950DT/A5（8 卡） |
| NPU 卡数 | 8 |
| 精度 | BF16 |
| 测试场景 | 4K×32K（4096 / 32768） |
| 并发 | c160 ~ c224（饱和区） |
| 优化目标 | 高吞吐 |
| 推理框架 | vLLM |

## ② 阶段最优配置结论

- **最优拓扑**：**TP2DP4EP8**（DP4 提供最大 KV 并行度；TP4DP2 @c96 输 27%，见拓扑排序）
- **最优并发 / 饱和点**：**c160 ~ c224**（c224 峰值、c160 稳健档 TTFT 更低；c192 gmu0.9 回落）
- **一句话结论**：并发是 A5 的第一杠杆（并发上探收益约 10 倍）

## ③ 拓扑 / 参数对比

**拓扑对比（@c96 可比）**：

| 配置（TP×DP×EP） | 备注 |
| --- | --- |
| TP2DP4EP8 | 最优（+27%） |
| TP4DP2EP8 | 起点 |

> @c8 两者持平（并发拉高才显现差异）。DP 数=KV 并行度，A3 经验在 A5 验证。

**A5 并发曲线（gmu0.9）**：c8 / c24 / c48 / c96 / c128 / c160 / c192 → 饱和点 c128-160

## ④ 性能数据与曲线

- **最优性能**：c224（峰值）；稳健档 c160（TTFT 更低）
- **gmu 杠杆**：0.9→0.95 @c160 吞吐 +49%（KV 5.94→10.52 GiB/rank）
- **A5 特有发现**：
  - A5 单卡算力强（TPOT 23.5ms @c8 vs A3 40ms+），但 8 卡 KV 空间远小于 16 die → **并发是 A5 的第一杠杆**（并发上探收益约 10 倍）
  - DCP 在 A5 也**不可用**（DSV4 MQA + 压缩注意力 + FULL_DECODE_ONLY 图模式：服务 1800s 超时未就绪）；与 A3 同结论，DCP 永久跳过

## ⑤ 踩坑记录

| 日期 | 现象/报错 | 根因 | 处置 | 状态 |
| ------ | ----------- | ------ | ------ | ------ |
| 2026-08 | DCP2 失败后残留 VLLM::EngineCore/Worker 进程占显存 77GB | DCP 失败未清理 | `kill -9 <EngineCore/Worker pid>` 清理后再启动 | 已解决 |
| 2026-08 | 下个服务启动报 `Free memory on device (14.34/91.75 GiB) ... less than desired GPU memory utilization` | 残留进程占显存 | `kill -9` 清理后再启动 | 已解决 |
| 2026-08 | **容器内存（非 HBM）是 bench 客户端极限**：gmu0.97 + c224 齐发导致 `vllm bench serve` 端 OOM（rc=137 Killed） | 客户端内存耗尽 | gmu 上限受客户端内存而非 HBM 约束，0.95 为稳健最优 | 已规避 |

## ⑥ 环境配置

- **启动命令**：服务端关键参数含 `--tp 2 --dp 4 --enable-expert-parallel --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --kv-cache-dtype bfloat16 --max-num-seqs <并发>`；gmu 0.95；CANN `/home/y00806874/ascend`
- **环境变量**：`VLLM_VERSION=0.27.1` / `VLLM_ASCEND_DSV4_KV_BF16=1` / `OMP_PROC_BIND=false` / `OMP_NUM_THREADS=10` / `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` / `HF_HUB_OFFLINE=1` / `HCCL_BUFFSIZE=1024`
- **其它**：A3/910C 机型经验见 `DeepSeek-V4-Flash-910C-4Kx32K.md`
