---
name: vllm-ascend-benchmark
description: >-
  This skill should be used when the user asks to "跑个 benchmark", "对比性能",
  "压测一下", "测下吞吐", "看下有没有性能回退", "run benchmark",
  "benchmark throughput", or wants to run vLLM online-serving performance
  benchmarks on a managed remote container. Supports single-run and multi-run
  (warm-service) modes.
version: 1.0.0
domain: inference-model
function: performance-optimization
tags:
  - vllm-ascend
  - benchmark
  - performance
author: Liccol
hooks:
  PreToolUse:
    - matcher: "Bash|PowerShell"
      prompt: |
        STOP. Before executing any command, you MUST:
        1. Present the full parameter table to the user (including: target host+container/conda-env,
           CANN dir, model, TP/DP, serve-args, bench-args, runs, warmup-runs, extra-env).
        2. Wait for the user to explicitly confirm all parameters.
        If you have not done both of these, DO NOT execute — present the parameter table first.

    # --- PowerShell vs Git Bash ---
    - tools: ["Bash"]
      matcher: "ssh.*\\s/(home|workspace|tmp|usr|vllm)"
      prompt: |
        ⚠️ Git Bash (MSYS) 会自动转换 Linux 路径 → 远程命令失败。
        请用 **PowerShell** 执行 SSH 命令。如确需 Bash，路径前加 //。

    # --- JSON 双引号转义 ---
    - tools: ["Bash", "PowerShell"]
      matcher: "(compilation_config|additional_config|profiler.config)"
      prompt: |
        ⚠️ JSON 参数中的双引号必须转义：PowerShell 用 `\"`，Git Bash 用 `\\"`。
targets-repo:
  - vllm-ascend
---

# vllm-ascend-benchmark

## 功能概述

在远程容器或conda环境上运行 `vllm bench serve` 性能基准测试。支持单次和多次（warm-service）模式。
脚本自动管理服务启动 → benchmark 执行 → 服务停止的完整流程。

- 结果输出在 stdout（JSON），吞吐量等核心指标可直接读取
- 本地结果保存在 `.workspace-local/benchmark/`

## 使用方式

- 直接调用：`/mindspeed-agent:vllm-ascend-benchmark`
- 描述 benchmark 需求，由 Claude 自动匹配调用

## 操作步骤

### 步骤一：确认参数

需确认的参数包括：

| 参数 | 说明 |
| ------ | ------ |
| 个人工号 | `--employee-id <工号>`（必填） |
| 目标 | `--host <ip> [--mode docker] --container <name>`（Docker exec）、`--host <容器IP> --mode direct --password <pwd>`（直连容器SSH）、`--host <ip> --mode conda --conda-env <name>`（Conda） |
| 连接模式 | `--mode {docker,conda,direct}`（默认 `docker`）：统一选择连接拓扑，透传给底层 serving |
| CANN 路径 | `--cann-dir <path>`（默认 `/usr/local/Ascend`），执行前必须确认 |
| 模型 | `--model <remote-path>` |
| TP / DP | `--tp <N> --dp <N>` |
| 服务参数 | `--serve-args` 后的参数（如 `--enable-expert-parallel --max-model-len 4096`） |
| 测试参数 | `--bench-args` 后的参数（如 `--num-prompts 96 --input-len 1024 --output-len 512`） |
| 并发数 | `--max-concurrency <N>`（在 `--bench-args` 中） |
| 轮数 | `--runs <N>`（默认 1，单轮快速验证） |
| 预热轮数 | `--warmup-runs <N>`（默认 0） |
| 环境变量 | `--extra-env KEY=VALUE`（如 `ASCEND_RT_VISIBLE_DEVICES=8,9,10,11`） |
| 多机 Worker | `--worker-hosts <csv>`（多机部署时必填，逗号分隔 worker IP 列表） |
| 多机参数 | `--worker-password`、`--worker-container`、`--worker-conda-env` 等（可选，默认同 head） |

### 步骤二：执行 benchmark

通过 `--employee-id <工号>` 将工号传给 serving，用于其 `/home/<工号>/serving/<实例>/` 服务运行目录；详见[工作目录与归属](../../workspace-ownership.md)。benchmark 客户端仍在原 `/tmp` 目录执行。

先读服务端并发预算日志（服务启动后就绪时）再定档：`grep "Maximum concurrency" <serve-stdout>`，
每个 DP rank 一行 → 总并发预算 ≈ DP 数 × 单 rank 预测。若预测预算 > 当前档位集上限，
**自动加测高档位**（统一档位集 128/256/384/512/768/1024，取预算 50%/75%/100% 就近落档，
`--num-prompts = --max-concurrency`）；加测高档位前确保服务端 `--max-num-seqs` ≥ 目标并发、
`HCCL_BUFFSIZE` 按 maxBs 线性放大（详见
[references/max-concurrency-tuning.md](references/max-concurrency-tuning.md)）。
每次 benchmark 的参数表（含并发档位选择依据：固定档 / 预算推算档 / 用户指定）必须展示给用户确认。

```bash
# docker 模式 — SSH 到宿主机, docker exec 进容器（默认）
python3 skills/inference-model-skills/performance-optimization/vllm-ascend-benchmark/scripts/bench_run.py --employee-id <工号> \
  --host <ip> --container <name> --key-file <path> \
  --model <remote-path> --tp 2 --dp 2 --runs 1 \
  --extra-env ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 \
  --serve-args --enable-expert-parallel --max-model-len 4096 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --bench-args --num-prompts 96 --max-concurrency 96 \
    --input-len 1024 --output-len 512

# direct 模式 — SSH 直连容器/虚拟机 IP
python3 skills/inference-model-skills/performance-optimization/vllm-ascend-benchmark/scripts/bench_run.py --employee-id <工号> \
  --mode direct --host <容器IP> --password <pwd> \
  --model <remote-path> --tp 2 --dp 2 --runs 1 \
  --serve-args --max-model-len 4096 \
  --bench-args --num-prompts 96 --max-concurrency 96

# conda 模式 — SSH 到宿主机 + conda 环境
python3 skills/inference-model-skills/performance-optimization/vllm-ascend-benchmark/scripts/bench_run.py --employee-id <工号> \
  --mode conda --host <ip> --conda-env <env-name> --key-file <path> \
  --model <remote-path> --tp 2 --dp 2 --runs 1 \
  --serve-args --enable-expert-parallel --max-model-len 4096 \
  --bench-args --num-prompts 96 --max-concurrency 96 \
    --input-len 1024 --output-len 512

# 单次运行benchmark
python3 .../bench_run.py --host <ip> --container <name> --model <path> --employee-id <工号> \
  --tp 4 --dp 2 --runs 1

# 多次运行benchmark + 预热
python3 .../bench_run.py --host <ip> --container <name> --model <path> --employee-id <工号> \
  --tp 4 --dp 2 --runs 3 --warmup-runs 1

# ===== 多机部署 =====
# docker 模式 — 两节点
python3 .../bench_run.py --host 192.168.1.100 --container my-container --employee-id <工号> \
  --worker-hosts 192.168.1.101 \
  --model /data/models/Qwen2-72B --tp 4 --dp 2 --runs 3 \
  --bench-args --num-prompts 128 --max-concurrency 64

# direct 模式 — 两容器直连
python3 .../bench_run.py --employee-id <工号> \
  --mode direct --host 10.0.0.5 --password hdpwd \
  --worker-hosts 10.0.0.6 --worker-password wkpwd \
  --model /data/models/Qwen2-72B --tp 4 --dp 2 --runs 3 \
  --bench-args --num-prompts 128 --max-concurrency 64

# conda 模式 — 两节点
python3 .../bench_run.py --employee-id <工号> \
  --mode conda --host 192.168.1.100 --conda-env vllm_env \
  --worker-hosts 192.168.1.101 --worker-conda-env vllm_env \
  --model /data/models/Qwen2-72B --tp 4 --dp 2 --runs 3 \
  --bench-args --num-prompts 128 --max-concurrency 64
```

> **多机 benchmark 行为**：脚本先通过 serve_start 启动多机集群（head + workers），然后对所有节点执行 `vllm bench serve`（workload 只打 head 的 API 端点），完成后停止整个集群。所有 worker 参数（`--worker-*`）透传给底层 serving。

### 步骤三：解读结果

单轮输出示例：

```json
{
  "status": "ok",
  "metrics": {
    "output_throughput": 2540.3,
    "mean_tpot_ms": 33.0,
    "mean_ttft_ms": 1998.9
  }
}
```

多轮输出包含 `aggregated` 字段（mean/stddev）。

### 步骤四：生成本地报告

benchmark 执行完成后，必须将结果整理为报告并保存在本地 `.workspace-local/benchmark/` 目录下。

**报告内容**：

1. **执行概要**：执行时间、目标机器、模型、TP/DP 配置、并发数等
2. **性能指标**：以表格形式列出关键指标（吞吐量、TTFT、TPOT、e2e 延迟等）
3. **多轮统计**（多轮模式下）：均值、标准差、逐轮数据
4. **结论**：性能是否达标，有无回归

**报告文件命名**：`<YYYY-MM-DDTHH-mm-ssZ>_<host>_report.md`

**报告示例结构**：

```markdown
# Benchmark 性能测试报告

> **执行时间**: 2026-07-14 08:30:00 UTC
> **目标机器**: 192.168.1.1
> **模型**: Qwen3-30B-A3B
> **配置**: TP=2, DP=2, max-model-len=4096

## 测试结果

| 指标 | 数值 |
|------|------|
| output_throughput | 2540.3 tok/s |
| mean_ttft_ms | 1998.9 ms |
| mean_tpot_ms | 33.0 ms |

## 结论

- 吞吐量达到预期水平（> 2000 tok/s）
- TTFT 在可接受范围内（< 2s）
```

## 注意事项

- **不要手动构造 `vllm bench serve` 命令**，所有操作通过 `bench_run.py` 执行
- **⚠️ `--port` 必须作为脚本级参数**：`--port 8006` 放在 `bench_run.py` 主参数中（`--serve-args` 之前），**不可**放入 `--serve-args`。脚本使用该端口传给 `serve_start.py` 进行健康检查和探测；若放入 `--serve-args`，vllm 绑定到指定端口但脚本自动分配了另一个端口，导致健康检查探测错误端口而超时
- benchmark 完成后服务会自动停止；如果启动失败也会调用 `serve_stop.py --force` 清理
- 跨平台：macOS/Linux/WSL 用 `python3`，Windows 用 **PowerShell** + `py -3`
- **⚠️ 并发档位选择：优先按服务端 Maximum concurrency 预测推算预算**：每次服务启动后，vLLM 会在 stdout 按 `[kv_cache_utils.py] Maximum concurrency for <max-model-len> tokens per request: <N>x` 逐 rank 打印"打满 KV cache 时可支撑的并发数"。**每个 DP rank 打印一行，总并发预算 ≈ DP 数 × 单 rank 预测**（如 TP4/DP2+DCP2 每 rank 289.67x × 2 ≈ 579）。选并发档位时：① 默认按用户/固定 workload；② 若目标是对比各拓扑真实吞吐上限，应对"总并发预测相近（±20% 内）"的组合**加测高并发档位**（预测的 ~50%/75%/100%），不得因固定 workload 结论跳过（低并发只能测得下限，预测近满的配置吞吐才是真实上界）。读取命令：`grep "Maximum concurrency" <serve-stdout>`。详见 [references/max-concurrency-tuning.md](references/max-concurrency-tuning.md)
- **⚠️ JSON 参数引号约束**：`--compilation-config` 等包含 JSON 字符串的参数，在 Windows PowerShell 下双引号会被 `CommandLineToArgvW` 剥离，导致 vllm JSON 解析失败（如 `{cudagraph_mode:FULL_DECODE_ONLY}` 缺少引号）。**必须在 PowerShell 单引号内对 JSON 的双引号做 `\"` 转义**，即 `'{\"key\":\"value\"}'`。原理：PowerShell 单引号内 `\"` 为字面量 `\` + `"`，经 Windows `CommandLineToArgvW` 解析后 `\"` 转义为字面双引号，从而正确传递 JSON 字符串。Linux/macOS/WSL 直接用单引号即可，无需转义

- **⚠️ 跨节点（多机）`--headless` 铁律**：`--headless` 永远由脚本自动加在 Worker 侧，**Head 节点禁止 `--headless`**（Head 必须承担 API Server，headless 会导致 `/health`、`/v1/models` 就绪失败、DP Engine 握手 5 分钟超时崩溃）。即使 `--serve-args` 里写了 `--headless`，脚本也会对 Head 自动剥离，仅 Worker 保持 `--headless`，无需手动区分
- **⚠️ 跨节点 DP 就绪等待**：大模型（如 397B MoE）跨节点加载 + 图模式 `FULL_DECODE_ONLY` 编译可能远超 120s。脚本默认 DP RPC 等待 `--dp-rpc-timeout 3600`、health 超时 1800s，避免 worker 未启动就超时
- **⚠️ A5 通信网卡**：A5（Ascend950DT）跨节点数据并行通信必须走专用网卡 `data0.3001`，`HCCL_IF_IP` 直接用本机管理 IP（`--host`/`--worker-hosts`），`GLOO/IP/HCCL_SOCKET_IFNAME=data0.3001`。**禁止**用 `ifconfig data0.3001 | grep inet | awk` 提取 IP。A3（Ascend910）不设此项（默认网卡）

## 参考信息

- [behavior.md](references/behavior.md) — 多轮运行、预热和清理的详细行为
- [command-recipes.md](references/command-recipes.md) — 常见调用模式
- [acceptance.md](references/acceptance.md) — benchmark 结果验收标准
- [max-concurrency-tuning.md](references/max-concurrency-tuning.md) — **依据服务端 Maximum concurrency 日志推算并发预算并选高并发档位（新模型必须遵守）**
- 依赖 skill：`vllm-ascend-serving`（服务启动/停止）
