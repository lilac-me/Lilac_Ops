#!/usr/bin/env python3
"""Generate staged vLLM/vLLM-Ascend tuning candidates as JSONL.

The output is intentionally conservative: inspect the target benchmark script
before running any candidate because script defaults and CLI flags vary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict


@dataclass
class Candidate:
    name: str
    stage: str
    env: dict[str, str]
    rationale: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--devices",
        required=True,
        help="Comma-separated device ids; this value is emitted as ASCEND_RT_VISIBLE_DEVICES, e.g. 4,5,6,7",
    )
    p.add_argument("--cards", type=int, required=True)
    p.add_argument("--max-seqs", type=int, default=48)
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--bs", type=int, default=None)
    p.add_argument("--max-batched-tokens", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.965)
    p.add_argument("--baseline-throughput", type=float, default=None)
    p.add_argument("--target-throughput", type=float, default=None)
    p.add_argument("--max-time-min", type=float, default=None)
    return p.parse_args()


def normalize_devices(raw: str) -> str:
    devices = [part.strip() for part in raw.split(",")]
    if not devices or any(not part.isdigit() for part in devices):
        raise SystemExit("--devices must be a comma-separated list of numeric device ids, e.g. 4,5,6,7")
    return ",".join(devices)


def add_common(env: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    common = {
        "ASCEND_RT_VISIBLE_DEVICES": args.devices,
        "MAX_NUM_SEQS": str(args.max_seqs),
        "NUM_PROMPTS": str(args.num_prompts or args.max_seqs),
        "BS": str(args.bs or args.max_seqs),
    }
    common.update(env)
    return common


def main() -> None:
    args = parse_args()
    args.devices = normalize_devices(args.devices)
    cands: list[Candidate] = []

    # Topology first. For 4 cards, TP4/DP1 is often a strong MoE baseline; TP2/DP2
    # is a useful health baseline. For other counts, include divisors.
    if args.cards == 4:
        topologies = [(4, 1), (2, 2)]
    else:
        topologies = [(args.cards, 1)]
        if args.cards % 2 == 0:
            topologies.append((args.cards // 2, 2))
        if args.cards % 4 == 0:
            topologies.append((args.cards // 4, 4))

    for tp, dp in topologies:
        cands.append(
            Candidate(
                name=f"topology-tp{tp}-dp{dp}",
                stage="topology",
                env=add_common(
                    {
                        "TENSOR_PARALLEL_SIZE": str(tp),
                        "DATA_PARALLEL_SIZE": str(dp),
                        "MAX_NUM_BATCHED_TOKENS": str(args.max_batched_tokens),
                        "GPU_MEMORY_UTILIZATION": f"{args.gpu_memory_utilization:.4g}",
                    },
                    args,
                ),
                rationale="Compare TP/DP layout before fine tuning memory and scheduler knobs.",
            )
        )

    best_tp = str(args.cards)
    best_dp = "1"

    for mbt in sorted({24576, 28672, args.max_batched_tokens, 36864, 40960}):
        cands.append(
            Candidate(
                name=f"mbt-{mbt}",
                stage="scheduler",
                env=add_common(
                    {
                        "TENSOR_PARALLEL_SIZE": best_tp,
                        "DATA_PARALLEL_SIZE": best_dp,
                        "MAX_NUM_BATCHED_TOKENS": str(mbt),
                        "GPU_MEMORY_UTILIZATION": f"{args.gpu_memory_utilization:.4g}",
                    },
                    args,
                ),
                rationale="Sweep max-num-batched-tokens around the baseline.",
            )
        )

    mem_center = args.gpu_memory_utilization
    for mem in [mem_center - 0.01, mem_center - 0.005, mem_center, mem_center + 0.001, mem_center + 0.005, mem_center + 0.01]:
        if 0.80 <= mem <= 0.99:
            cands.append(
                Candidate(
                    name=f"mem-{mem:.4f}".replace(".", ""),
                    stage="memory",
                    env=add_common(
                        {
                            "TENSOR_PARALLEL_SIZE": best_tp,
                            "DATA_PARALLEL_SIZE": best_dp,
                            "MAX_NUM_BATCHED_TOKENS": str(args.max_batched_tokens),
                            "GPU_MEMORY_UTILIZATION": f"{mem:.4f}",
                        },
                        args,
                    ),
                    rationale="Fine sweep gpu-memory-utilization near the best valid point.",
                )
            )

    for bs in sorted({max(1, args.max_seqs - 4), args.max_seqs, args.max_seqs + 8, args.max_seqs + 16}):
        cands.append(
            Candidate(
                name=f"concurrency-{bs}",
                stage="concurrency",
                env=add_common(
                    {
                        "TENSOR_PARALLEL_SIZE": best_tp,
                        "DATA_PARALLEL_SIZE": best_dp,
                        "MAX_NUM_BATCHED_TOKENS": str(args.max_batched_tokens),
                        "MAX_NUM_SEQS": str(max(args.max_seqs, bs)),
                        "NUM_PROMPTS": str(max(args.num_prompts or args.max_seqs, bs)),
                        "BS": str(bs),
                        "GPU_MEMORY_UTILIZATION": f"{args.gpu_memory_utilization:.4g}",
                    },
                    args,
                ),
                rationale="Probe whether concurrency is below, at, or above the scheduling sweet spot.",
            )
        )

    for feature_name, env in [
        ("mlapo0", {"VLLM_ASCEND_ENABLE_MLAPO": "0"}),
        ("noprefetch", {"VLLM_ASCEND_ENABLE_PREFETCH_MLP": "0"}),
        ("notopk", {"VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "0"}),
        ("fusedmc2-1", {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}),
    ]:
        cands.append(
            Candidate(
                name=f"feature-{feature_name}",
                stage="feature-flags",
                env=add_common(
                    {
                        "TENSOR_PARALLEL_SIZE": best_tp,
                        "DATA_PARALLEL_SIZE": best_dp,
                        "MAX_NUM_BATCHED_TOKENS": str(args.max_batched_tokens),
                        "GPU_MEMORY_UTILIZATION": f"{args.gpu_memory_utilization:.4g}",
                        **env,
                    },
                    args,
                ),
                rationale="Test a vLLM-Ascend feature flag; startup failure is an acceptable negative result.",
            )
        )

    try:
        for cand in cands:
            print(json.dumps(asdict(cand), ensure_ascii=False, sort_keys=True))
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            return


if __name__ == "__main__":
    main()
