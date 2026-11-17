#!/usr/bin/env python3
"""Classify a migration failure log with deterministic pattern hints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple


class PatternHint(NamedTuple):
    category: str
    pattern: str
    knowledge_topics: tuple[str, ...]


PATTERNS: tuple[PatternHint, ...] = (
    PatternHint(
        "CANN path or environment error",
        r"(?:ASCEND|CANN|set_env|aclrt|driver)[^\n]{0,180}(?:not found|missing|failed|error)",
        ("backend-runtime",),
    ),
    PatternHint(
        "torch/torch_npu dependency drift or version mismatch",
        r"(?:torch_npu|torch)[^\n]{0,180}(?:version|undefined symbol|ABI)[^\n]{0,180}(?:mismatch|incompatible|not support|undefined|failed)",
        ("backend-runtime",),
    ),
    PatternHint(
        "Ray NPU worker visibility or placement issue",
        r"(?:Ray|ray actor|ray worker|ASCEND_RT_VISIBLE_DEVICES|resources=)[^\n]{0,220}(?:NPU|device|resource|visible|placement)[^\n]{0,160}(?:missing|empty|zero|unavailable|not found|failed|error)",
        ("distributed-serving", "backend-runtime"),
    ),
    PatternHint(
        "vLLM generation or weight synchronization issue",
        r"(?:vllm|EngineCore|weight sync|update_weights|collective_rpc|rollout)[^\n]{0,220}(?:timeout|failed|error|mismatch|not support|unavailable)",
        ("distributed-serving",),
    ),
    PatternHint(
        "launcher, Hydra or configuration issue",
        r"(?:hydra|omegaconf|config path|config name|override|argparse|unrecognized arguments|shell launcher)[^\n]{0,220}(?:missing|not found|invalid|failed|error|required|unrecognized)",
        ("checkpoint-launchers",),
    ),
    PatternHint(
        "torch.cuda.amp or AMP incompatibility",
        r"(?:torch\.cuda\.amp|GradScaler|autocast)[^\n]{0,220}(?:cuda|device_type|unsupported|not support|failed|error|TypeError)",
        ("backend-runtime",),
    ),
    PatternHint(
        "CUDA API incompatibility",
        r"(?:torch\.cuda|\.cuda\(|device\(['\"]cuda|CUDA error|cuda runtime)[^\n]{0,220}",
        ("backend-runtime",),
    ),
    PatternHint(
        "NCCL to HCCL or distributed rendezvous issue",
        r"(?:nccl|hccl|process group|all_reduce|all-reduce|broadcast|barrier|collective|rank)[^\n]{0,220}(?:timeout|timed out|failed|error|mismatch|not support|hang)",
        ("distributed-serving",),
    ),
    PatternHint(
        "torch.nn.DataParallel unsupported path",
        r"(?:torch\.nn\.)?DataParallel[^\n]{0,180}",
        ("backend-runtime", "distributed-serving"),
    ),
    PatternHint(
        "FlashAttention or fused-attention issue",
        r"(?:flash_attn|flash-attn|FlashAttention|npu_fusion_attention|fusion attention)[^\n]{0,220}(?:unsupported|not support|failed|error|shape|layout|dtype|mask|missing|not found)",
        ("operator-semantics",),
    ),
    PatternHint(
        "Triton kernel unsupported or compiler failure",
        r"(?:triton|Triton)[^\n]{0,220}(?:unsupported|not support|compile|compiler|failed|error|backend|kernel)",
        ("operator-semantics",),
    ),
    PatternHint(
        "custom CUDA extension",
        r"(?:CUDAExtension|\.cu\b|nvcc|custom cuda|load_inline)[^\n]{0,220}",
        ("operator-semantics", "backend-runtime"),
    ),
    PatternHint(
        "unsupported op or missing kernel",
        r"(?:unsupported|not support|No kernel|kernel[^\n]{0,80}not found|operator[^\n]{0,80}not found)[^\n]{0,220}",
        ("operator-semantics",),
    ),
    PatternHint(
        "dtype, precision or numerical mismatch",
        r"(?:dtype|precision|bf16|float16|fp16|numerical|allclose|NaN|Inf)[^\n]{0,220}(?:mismatch|expected|unsupported|not support|failed|error|false)",
        ("backend-runtime", "operator-semantics"),
    ),
    PatternHint(
        "checkpoint, state dict or optimizer state issue",
        r"(?:state_dict|checkpoint|optimizer state|resume|safetensors|missing shard)[^\n]{0,220}(?:missing|unexpected|mismatch|failed|error|not found|incomplete|shape)",
        ("checkpoint-launchers", "backend-runtime"),
    ),
    PatternHint(
        "packing or varlen sample-boundary issue",
        r"(?:cu_seqlens|varlen|packed|padding-free|position_ids|attention mask)[^\n]{0,220}(?:mismatch|invalid|failed|error|shape|boundary|unsupported)",
        ("operator-semantics",),
    ),
    PatternHint(
        "quantization CUDA-only backend",
        r"(?:bitsandbytes|GPTQ|AWQ|exllama|int4|int8|quant)[^\n]{0,220}(?:cuda|unsupported|not support|failed|error|missing)",
        ("operator-semantics", "backend-runtime"),
    ),
    PatternHint(
        "third-party library hardcoding CUDA",
        r"(?:xformers|apex)[^\n]{0,220}(?:cuda|unsupported|not support|failed|error|missing)",
        ("backend-runtime", "operator-semantics"),
    ),
    PatternHint(
        "Python import or configuration error",
        r"(?:ModuleNotFoundError|ImportError|AttributeError|KeyError|ValueError|TypeError|Traceback)[^\n]{0,220}",
        (),
    ),
)


def classify_text(text: str, logs: list[str]) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    topics: list[str] = []
    for hint in PATTERNS:
        found = re.search(hint.pattern, text, re.IGNORECASE)
        if not found:
            continue
        excerpt = re.sub(r"\s+", " ", found.group(0)).strip()[:300]
        matches.append({
            "category": hint.category,
            "excerpt": excerpt,
            "knowledge_topics": list(hint.knowledge_topics),
        })
        for topic in hint.knowledge_topics:
            if topic not in topics:
                topics.append(topic)
    return {
        "status": "CLASSIFIED" if matches else "UNCLASSIFIED",
        "primary_category": matches[0]["category"] if matches else None,
        "knowledge_topics": topics,
        "matches": matches,
        "logs": logs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    text_parts = []
    for path in args.logs:
        if path.exists():
            text_parts.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(text_parts)

    result = classify_text(text, [str(path) for path in args.logs])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "CLASSIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
