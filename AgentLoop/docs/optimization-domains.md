# 推理优化场景与验收

推理系统优化由场景驱动，不以算子优化为总入口。每项优化必须定义固定 workload、baseline、候选方案、精度/性能对照与回滚。

| 场景 | 优化对象 | 关键指标 | 状态 |
| --- | --- | --- | --- |
| 算子 | Vector fusion、GEMM/GMM、Attention | latency、实际形态 MFU、带宽 | 候选分析 |
| 量化 | 权重/激活/KV 量化、尺度与精度 | HBM、TTFT、TPOT、任务精度 | 规划 |
| 池化 | KVCache block、跨请求复用、显存/资源池 | cache hit、容量、碎片、读写带宽、隔离 | 规划 |
| PD | Prefill/Decode 分离、KV 传输、资源比例 | TTFT、TPOT、goodput、传输带宽、排队 | 规划 |
| 并行通信 | TP/DP/EP、跨卡、Overlap | throughput、尾延迟、通信占比 | 规划 |
| 调度图模式 | continuous batching、chunked prefill、graph、多流 | 请求成功率、吞吐、尾延迟 | 参数寻优脚本已迁入 |

统一服务验收：固定版本、模型、请求、采样、真实 token 数、并行与缓存策略；若专项改变其中一项则明确唯一变量。无 profiler 前后多次压测，比较 TTFT/TPOT/ITL/E2E、输入输出吞吐和 SLO goodput。精度使用固定真实请求与参考输出/Logits 或任务准确率；KV 状态和跨请求隔离必须检查。独立环境集成，失败回滚。第一版未实现这些场景的统一执行器。


## 端到端收益硬门禁

单算子收益不是交付目标；只有同真实 workload 下整网主指标出现超过噪声且达到预设最低门槛的正收益，精度及保护指标同时通过，才计入有效优化。无整网收益、退化或未验证均不得作为优化成功交付。正式验收使用 AISBench。详见 [端到端收益验收](e2e-acceptance.md)。
