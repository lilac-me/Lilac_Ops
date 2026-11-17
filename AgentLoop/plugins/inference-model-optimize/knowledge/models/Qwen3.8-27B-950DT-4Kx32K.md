# Qwen3.8-27B（dense + 48 层线性注意力 mamba，GQA 4KV 头）· 950DT/A5（8 卡）· 4K×32K 场景

## ① 模型与场景档案

| 项 | 值 |
| ---- | ---- |
| 模型名 | Qwen3.8-27B |
| 架构 | dense + 48 层线性注意力 mamba，GQA 4KV 头 |
| 机型 | 950DT/A5（8 卡） |
| NPU 卡数 | 8 |
| 精度 | BF16 |
| 测试场景 | 4K×32K（4096 / 32768） |
| 并发 | 128（最优） |
| 优化目标 | 高吞吐 |
| 推理框架 | vLLM |

## ② 阶段最优配置结论

- **最优拓扑**：**TP8/DP1/EP1 + DCP2**（DCP 硬约束：`-dcp N` 要求 TP > KV 头数 4 → 仅 TP8+DCP2 合法；TP4+DCP2 被 pydantic 拒）
- **最优并发 / 饱和点**：**128**（256 并发饱和负样本：+0.9% 且 TTFT mean 568s → 最优 128）
- **关键参数**：图模式 FULL_DECODE_ONLY；batched 65536（131072+FLASHCOMM1 在本环境 prefill 病态 → 负面样本）
- **一句话结论**：compute-bound（48 层 mamba 顺序解码聚合带宽饱和，逼近理论极限）；剩余提升需算子级工程（框架外）

## ③ 拓扑 / 参数对比

| 配置 | 备注 |
| --- | ------------------------- |
| TP8/DP1/EP1 + DCP2 @128 | 最优（+29.6% vs 基线） |
| 基线 TP8/DP1/EP1 | 起点 |
| @256 并发 | +0.9% 且 TTFT mean 568s，饱和负样本 |

## ④ 性能数据与曲线

- **基线 → 最优**：DCP2 增益 **+29.6%**；TPOT 33.59ms；TTFT 10.4s；128/128 0 失败
- **瓶颈结论**：compute-bound（48 层 mamba 顺序解码聚合带宽饱和）

## ⑤ 踩坑记录

| 日期 | 现象/报错 | 根因 | 处置 | 状态 |
| ------ | ----------- | ------ | ------ | ------ |
| 2026-08 | docker restart 后 triton "0 active driver" | bisheng 不在 PATH | 补 bisheng PATH | 已解决 |
| 2026-08 | HCCL EI0020 Bind_IP_Port 端口耗尽 | HCCl socket 端口耗尽 | `HCCL_CONNECT_TIMEOUT=1800` `HCCL_NPU_SOCKET_PORT_RANGE=58000-59999` | 已解决 |
| 2026-08 | NZ=2 失败：`EI0015 RootInfoDetect failed` | 布局开关冲突 | 本模型不用 NZ | 已规避 |
| 2026-08 | 启动前残留 VLLMWorker_TP 幽灵进程（17.4GB/卡） | 上一次运行未清理 | 启动前清理残留 VLLMWorker_TP | 已解决 |

## ⑥ 环境配置

- **启动命令**：服务端关键参数含 `--tp 8 --dp 1 --decode-context-parallel-size 2 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --max-num-batched-tokens 65536`；CANN `/home/y00806874/ascend`
- **环境变量**：`VLLM_VERSION=0.27.1` `OMP_PROC_BIND=false` `OMP_NUM_THREADS=10` `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` + `HCCL_CONNECT_TIMEOUT=1800` `HCCL_NPU_SOCKET_PORT_RANGE=58000-59999` + bisheng PATH
