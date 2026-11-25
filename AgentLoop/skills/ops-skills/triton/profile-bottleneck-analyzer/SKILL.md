---
name: profile-bottleneck-analyzer
description: Analyze a selected inference rank profiling CSV for hot kernels, vector fusion opportunities and MatMul candidates.
---
# 推理热点识别

优先使用 plugins/ops/engine/infer_profile_analyze.py 对单 rank 本地 profile 执行产物校验、热点识别和语言路由。直接 parse_profile.py 只分析 CSV，不证明阶段与实现语言。耗时占比按 worker iteration 内累计 kernel duration 计算，不等于服务关键路径。Vector 邻接不是融合合法性；MatMul 目标按真实 Prefill/Decode shape 与算术强度决定。输出只代表待验证候选。
