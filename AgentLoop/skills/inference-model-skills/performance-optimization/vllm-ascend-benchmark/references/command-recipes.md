# Benchmark Command Recipes

## ⚠️ Windows PowerShell JSON 参数转义

`--compilation-config` 等 JSON 参数在 Windows PowerShell 下双引号会被 `CommandLineToArgvW` 剥离，**必须在单引号内对 JSON 双引号做 `\"` 转义**。

```powershell
# ❌ 错误：双引号被剥离，vllm JSON 解析失败
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'

# ✅ 正确：\" 在 Windows 命令行解析后还原为字面双引号
--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'
```

Linux/macOS/WSL 直接使用单引号即可，无需转义。

## ⚠️ `--port` 参数位置约束

`--port` 必须作为 `bench_run.py` 的主参数，**不可**放进 `--serve-args` 中：

```bash
# ✅ 正确：--port 作为 bench_run.py 主参数
python3 bench_run.py --employee-id <工号> --host <ip> --model <path> --port 8006 --tp 2 \
  --serve-args --max-model-len 4096 \
  --bench-args --num-prompts 96

# ❌ 错误：--port 放入 --serve-args 会导致端口不匹配
python3 bench_run.py --employee-id <工号> --host <ip> --model <path> --tp 2 \
  --serve-args --port 8006 --max-model-len 4096 \
  --bench-args --num-prompts 96
```

原因：`bench_run.py` 将主参数中的 `--port` 传给 `serve_start.py`，后者用该端口分配并做健康检查。若 `--port` 放在 `--serve-args` 中，vllm 绑定到指定端口，但 `serve_start.py` 自动分配了另一个端口，健康检查探测错误端口导致超时。

## Single-run: minimal

```bash
python3 skills/inference-model-skills/performance-optimization/vllm-ascend-benchmark/scripts/bench_run.py --employee-id <工号> \
  --host 173.131.1.2 --container my-container \
  --model /home/weights/Qwen3.5-0.8B \
  --port 12345 \
  --tp 1 \
  --dp 1
```

## Single-run: minimal (direct Docker connection)

```bash
python3 skills/inference-model-skills/performance-optimization/vllm-ascend-benchmark/scripts/bench_run.py --employee-id <工号> \
  --host 173.131.1.2 \
  --container my-vllm-container \
  --password mypassword \
  --model /home/weights/Qwen3.5-0.8B \
  --tp 1 \
  --dp 1 \
  --port 12345
```

## Single-run: full-featured (graph mode)

```bash
python3 skills/inference-model-skills/performance-optimization/vllm-ascend-benchmark/scripts/bench_run.py --employee-id <工号> \
  --host 173.131.1.2 --container my-container \
  --model /home/weights/Qwen3-Next-80B-A3B-Instruct \
  --tp 2 \
  --dp 2 \
  --port 12345 \
  --extra-env OMP_NUM_THREADS=10 \
  --extra-env HCCL_BUFFSIZE=1024 \
  --extra-env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --serve-args \
    --max-model-len 40960 \
    --trust-remote-code \
    --async-scheduling \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 64 \
    --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --bench-args \
    --num-prompts 256 \
    --max-concurrency 64 \
    --output-len 1024
```

## Multi-run with warmup: statistical benchmarking

Start the service once, run 3 iterations, discard the first as warmup, aggregate the remaining 2:

```bash
python3 skills/inference-model-skills/performance-optimization/vllm-ascend-benchmark/scripts/bench_run.py --employee-id <工号> \
  --host 173.131.1.2 --container my-container \
  --model /home/weights/Qwen3.5-35B-A3B \
  --tp 2 \
  --dp 2 \
  --port 12345 \
  --runs 3 --warmup-runs 1 \
  --serve-args \
    --max-model-len 4096 \
    --trust-remote-code \
    --async-scheduling \
    --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --bench-args \
    --num-prompts 64 \
    --max-concurrency 64 \
    --output-len 1024
```
