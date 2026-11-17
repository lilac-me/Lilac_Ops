---
name: veomni-performance-tuning
description: >-
  This skill should be used when the user asks to analyze and optimize VeOmni
  training performance on Ascend NPU: fix hardware/model/data/parallel baseline,
  configure VeOmni ProfileConfig to collect profiling, analyze step time,
  throughput, MFU, memory and per-rank variance, classify compute/communication/
  host/memory/dataloader bottlenecks, apply single-variable optimizations with
  identical measurement methodology, or rerun precision validation after tuning.
version: 0.1.0
domain: common
function: performance-optimization
tags:
  - veomni
  - profiling
  - bottleneck
  - performance
author: mindspeed-agent
license: MIT
compatibility: "VeOmni >= 0.1"
---

# VeOmni 性能调优

## 目的

分析和优化 VeOmni 在昇腾 NPU 上的训练性能，产出可对比、可复现的优化报告。

## 调优流程

1. **基线固化**
   - 固化硬件、模型、数据、并行配置；记录 source revision
   - 运行基线测试，记录 step time、吞吐、显存到 `performance_baseline.json`
2. **Profiling 采集**
   - 配置 VeOmni `ProfileConfig`（开启 step 间隔采集，避免全量采集引入过大开销）
   - 采集多个 step，丢弃前几个 warmup step
3. **指标分析**
   - step time 分布（均值/方差/离群 step）、吞吐（tokens/s）、MFU（口径见 veomni-mfu-calculation）
   - 显存峰值与各 rank 差异（rank 间不均衡是常见瓶颈信号）
4. **瓶颈分类**

   | 瓶颈类型 | 特征信号 |
   | --------- | --------- |
   | 计算 | 计算算子占比高且 NPU 利用率高 → 已接近硬件上限 |
   | 通信 | 通信算子（allreduce/allgather）占比高 → 调并行度/通信量化 |
   | Host | kernel 间空隙大、CPU 侧调度长 → 减少 host 开销/算子下沉 |
   | 显存 | 频繁重计算/换入换出 → 调重计算策略、micro batch |
   | 数据加载 | dataloader 等待 → 预取、worker 数、预处理下沉 |

   热点下钻：Attention、MoE（dispatch/combine 通信）、通信、数据处理。
5. **定向优化**
   - **每次只修改一个主要变量**，其余条件不变
   - 使用与基线完全相同的测试口径复测
   - 收益必须超过测试噪声（多次运行的波动范围），否则视为无效
6. **精度复测**
   - 优化后**必须**重新执行精度验证（veomni-precision-alignment）
   - 精度回退 → 回滚该优化并记录 `PRECISION_REGRESSED`
7. **迭代**：重复 2-6，直至无有效优化项或达到目标。

## 输出契约

- `performance_baseline.json`（Schema：`benchmarks/veomni/schemas/performance_report.json` 的 baseline 部分）
- `performance_report.json`（Schema：`benchmarks/veomni/schemas/performance_report.json`）：

```json
{
  "model": "string",
  "baseline": {"step_time_ms": 0.0, "throughput_tokens_per_s": 0.0, "memory_gb": 0.0, "noise_band": 0.0},
  "profiling": {"tool": "ProfileConfig", "steps_collected": 0, "warmup_excluded": true},
  "bottlenecks": [{"type": "compute|communication|host|memory|dataloader", "evidence": "string", "share": 0.0}],
  "optimizations": [
    {"variable": "string", "before": 0.0, "after": 0.0, "gain_percent": 0.0,
     "exceeds_noise": true, "precision_rerun": "PASS | FAIL | PENDING"}
  ],
  "conclusion": "string"
}
```

## 门禁与约束

- 放行条件：相同基线和测试口径，收益超过测试噪声，精度未回退。
- 单变量原则不可打破；同时改多项后无法归因，报告无效。
- 精度先于性能：任何性能结论都以精度复测通过为前提。
