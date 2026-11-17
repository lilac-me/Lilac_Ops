#!/usr/bin/env python3
"""启动一次 MindSpeed/Megatron 训练，从逐 iteration 日志导出整网计时和 loss。"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from model_metrics import validate_metrics

FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
STEP = re.compile(r"\biteration\s+(\d+)\s*/", re.I)
TIME = re.compile(r"elapsed time per iteration\s*\(ms\)\s*:\s*(" + FLOAT + ")", re.I)
LOSS = re.compile(r"\blm loss\s*:\s*(" + FLOAT + ")", re.I)


def parse_log(text, warmup_steps, tokens_per_step=None):
    samples = []
    for line in text.splitlines():
        step, timing, loss = STEP.search(line), TIME.search(line), LOSS.search(line)
        if step and "elapsed time per iteration" in line.lower() and not (timing and loss):
            raise ValueError("训练指标行缺少合法耗时或 lm loss（含 NaN/Inf 时拒绝测量）")
        if not (step and timing and loss):
            continue
        sample = {"step": int(step[1]), "step_time_ms": float(timing[1]), "loss": float(loss[1])}
        if tokens_per_step is not None:
            sample["tokens"] = tokens_per_step
        samples.append(sample)
    if len({s["step"] for s in samples}) != len(samples):
        raise ValueError("日志包含重复 iteration；请使用单个统计 rank 的日志")
    return samples[warmup_steps:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--tokens-per-step", type=int)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.warmup_steps < 0:
        ap.error("需要训练命令和非负 warmup-steps")
    output = Path(os.environ["AGENTLOOP_METRICS_PATH"])
    log = output.with_suffix(".training.log")
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    data = {
        "run_id": os.environ["AGENTLOOP_RUN_ID"],
        "phase": os.environ["AGENTLOOP_PHASE"],
        "workload_id": os.environ["AGENTLOOP_WORKLOAD_ID"],
        # 命令配置必须关闭训练框架自己的 profiler，不能仅依赖该环境变量。
        "profiling_enabled": False,
        "samples": parse_log(log.read_text(errors="replace"), args.warmup_steps, args.tokens_per_step),
    }
    validate_metrics(data, data["run_id"], data["phase"], data["workload_id"], 1)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
