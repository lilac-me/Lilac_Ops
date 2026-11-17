# DeepSeek-V4-Flash-BF16（MoE 256 experts top-6，MQA 1 头，DSA 压缩注意力，582GB）· 910C/A3（16 die）· 4K×32K 场景

## ① 模型与场景档案

| 项 | 值 |
| ---- | ---- |
| 模型名 | DeepSeek-V4-Flash-BF16 |
| 机型 | 910C/A3（16 die） |
| NPU 卡数 | 16 |
| 精度 | BF16 |
| 测试场景 | 4K×32K（4096 / 32768） |
| 并发 | 288（饱和点；平衡档 256） |
| 优化目标 | 高吞吐 |
| 推理框架 | vLLM |

## ② 阶段最优配置结论

- **最优拓扑**：**TP2DP8EP16**（--enable-expert-parallel；EP-off 不可行：554GB/rank > 64GB/die）
- **最优并发 / 饱和点**：**288**（over-budget，突破 gmu 预算；s=36/rank 后 TPOT 批量项回升 → 饱和点）；平衡档 256
- **关键参数**：gmu 0.9（0.95 在 MoE a2a+prefill 峰值 OOM：194MiB 缺口即崩）；`HCCL_BUFFSIZE=1024` 必带（DP16 报 561002）；FULL_DECODE_ONLY；`--max-num-batched-tokens 8192`（c160 起，缓解并发 prefill）
- **一句话结论**：per-rank 解码串行延迟主导时，`max-num-seqs/并发` 突破 gmu-KV 预算是最大收益杠杆（并发上探收益约 ×4）

## ③ 拓扑 / 参数对比

**拓扑对比（96 并发可比）**：

| 配置（TP×DP×EP） | 备注 |
| --- | --- |
| TP2DP8EP16 | 最优 |
| TP1DP16EP16 | per-rank 串行 ×2 |
| TP4DP4 / TP8DP2 | 并发效率更低 |

**并发梯度（A3，co 上限）**：基线 c48 → 峰值 c288（约 ×4）；平衡档 c256

## ④ 性能数据与曲线

- **最优性能**：co=288（约 ×4 于基线 c48）；TPOT 68.2ms（s=16 前亚线性）；TTFT 37.1s（p99 78s，须 mean+P90 同报）
- **核心方法论（本模型新增）**：MoE 长输出模型在 **per-rank 解码串行延迟主导** 时（TPOT 增 vs 并发亚线性），**`max-num-seqs/并发` 突破 gmu-KV 预算是最大收益杠杆**；下"硬件墙"结论前必须先跑并发梯度（本模型曾被误判为 HBM 带宽墙，差 2 个数量级）
  - **并发预算主辅结合**：主=以 vllm `Maximum concurrency`/gmu 预算公式做**并发初设**（常规模型布局准确、优先采用）；辅=每档测试时看引擎日志 **`GPU KV cache usage`** 实测利用率排查——`<60% 且吞吐停滞` → 非 KV 瓶颈（继续上探）；`>85%` → 真 KV 受限（降档）。**DSA/混合 KV 布局（kv_cache_groups）模型该换算会失真**（DSV4 gmu0.9 报 1.13x/rank，实测 36 个/rank 仅用 54% 池），此类模型以 KV 利用率实测为准
  - **多 DP 下看 KV 利用率的前提**：`--api-server-count` 默认 = DP 数（serve.py internal LB 分支）、请求入口分散、APIServer 日志看不到统一 `GPU KV cache usage` → **必须显式 `--api-server-count 1`**（入口统一，APIServer 打印全部 Engine 的 KV usage；官方 DSV4 教程 DP 场景均带此参数）
  - **max-num-seqs 口径**：per-DP-group 上限（DSV4 教程），超限请求 waiting 时间计入 TTFT/TPOT → 性能测试保证 `--max-num-seqs × DP >= 总并发`（本 run -MaxSeqs=并发满足）

## ⑤ 踩坑记录

| 日期 | 现象/报错 | 根因 | 处置 | 状态 |
| ------ | ----------- | ------ | ------ | ------ |
| 2026-08 | spec decode（MTP/DSPARK）不可用 | checkpoint 无 DSpark 专属权重（仅 V3.1 风格 MTP：e_proj/enorm + 1 层自有 256 专家 MoE）；dspark 缺 dspark_block_size/target_layer_ids/markov_rank 属性；mtp 可启动但接受率 0.019% → **-47%** | 永久关闭 spec decode | 已规避 |
| 2026-08 | DYNAMIC_EPLB env 无效 | v1 runner 下需 VLLM_USE_V2_MODEL_RUNNER=1 + --enable-eplb 全套 | 使用完整套件 | 已规避 |
| 2026-08 | MQA（kv_heads=1）模型 DCP 收益场景不成立 | KV 复制仅 2 倍 | DCP 未采用；并发上探直接可行 | 已规避 |

## ⑥ 环境配置

- **启动命令**：服务端关键参数含 `--tp 2 --dp 8 --enable-expert-parallel --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --max-num-batched-tokens 8192 --api-server-count 1 --max-num-seqs <并发>`；gmu 0.9
- **环境变量**：`VLLM_VERSION=0.27.1` / `OMP_PROC_BIND=false` / `OMP_NUM_THREADS=10` / `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` / `HF_HUB_OFFLINE=1` / `HCCL_BUFFSIZE=1024`；CANN B150
- **其它**：A5/950DT 机型经验见 `DeepSeek-V4-Flash-950DT-4Kx32K.md`
