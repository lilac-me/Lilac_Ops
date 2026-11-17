#!/usr/bin/env python3
"""Qwen3 MoE batch-invariance case generator and runner.

Use `--dry-run` to validate the generated plan without importing or allocating
NPU tensors. Run without `--dry-run` on an Ascend/NPU host to execute the ops and
write a JSON result file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


TP_VALUES = (1, 4, 8)
DEFAULT_TOTAL_CASES = 10000


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 768
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 128
    num_experts_per_tok: int = 8
    vocab_size: int = 151936
    block_size: int = 128
    max_prompt_num: int = 64
    max_prompt_len: int = 2048


MODEL = ModelConfig()
CHUNKED_PREFILL_BLOCK_SIZE = 128
NZ_FMT_LAST_DIM = 32


@dataclass(frozen=True)
class TPProfile:
    tp_size: int
    q_heads: int
    kv_heads: int
    local_experts: int
    qkv_out: int
    o_proj_in: int
    shared_intermediate: int


def tp_profile(tp_size: int) -> TPProfile:
    if tp_size not in TP_VALUES:
        raise ValueError(f"unsupported tp_size={tp_size}; expected one of {TP_VALUES}")
    q_heads = MODEL.num_attention_heads // tp_size
    if tp_size <= MODEL.num_key_value_heads:
        kv_heads = MODEL.num_key_value_heads // tp_size
    else:
        kv_heads = 1
    local_experts = MODEL.num_experts // tp_size
    return TPProfile(
        tp_size=tp_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        local_experts=local_experts,
        qkv_out=(q_heads + 2 * kv_heads) * MODEL.head_dim,
        o_proj_in=q_heads * MODEL.head_dim,
        shared_intermediate=MODEL.intermediate_size // tp_size,
    )


@dataclass(frozen=True)
class PlanItem:
    operator: str
    tier: str
    tp_size: int


@dataclass
class CaseResult:
    case_id: int
    operator: str
    api: str
    tp_size: int
    tier: str
    scenario: str
    batch_axis: str
    conclusion: str
    satisfied: bool | None
    bitwise_equal: bool | None
    dtype: str
    shape: str
    detail: str


@dataclass
class CaseContext:
    case_id: int
    operator: str
    tier: str
    tp: TPProfile
    rng: random.Random
    dry_run: bool
    max_token_rows: int
    max_expert_rows: int
    max_fia_batch: int
    max_fia_seq: int
    max_sampler_rows: int
    max_sample_size: int
    fia_chunked_prefill_tokens: int | None
    fia_chunked_prefill_chunk_size: int | None
    fia_chunked_prefill_data_pattern: str


TORCH: Any = None
TORCH_NPU: Any = None
BATCH_INVARIANT_OPS: Any = None
DTYPE: Any = None
SPLIT_MAX_PARTS: int | None = 16
SPLIT_FORCE_SINGLETONS = False
FIA_SCENARIOS: tuple[str, ...] = (
    "bsh_prefill",
    "bnsd_prefill",
    "bsh_paged_decode",
    "tnd_chunked_prefill",
    "tnd_prefill_token_batch",
    "tnd_paged_decode",
)
FIA_CHUNKED_PREFILL_DATA_PATTERNS: tuple[str, ...] = ("large_small", "range")


TARGET_OPERATORS = (
    "torch_npu.npu_swiglu",
    "torch_npu.npu_scatter_pa_kv_cache",
    "torch.scatter_",
    "torch_npu.npu_rms_norm",
    "torch_npu.npu_moe_init_routing_v2",
    "torch_npu.npu_moe_token_unpermute",
    "torch_npu.npu_grouped_matmul",
    "torch_npu.npu_moe_gating_top_k",
    "torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant",
    "torch.ops.batch_invariant_ops.npu_matmul_batch_invariant",
    "torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant",
)


TIER_WEIGHTS = (
    ("small", 0.40),
    ("medium", 0.38),
    ("large", 0.17),
    ("boundary", 0.05),
)


def stable_operator_seed(operator: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(operator))


def allocate_counts(total_cases: int, operators: tuple[str, ...]) -> dict[str, int]:
    if total_cases < len(operators):
        raise ValueError(
            f"total_cases={total_cases} is smaller than operator count={len(operators)}"
        )
    base = total_cases // len(operators)
    remainder = total_cases % len(operators)
    return {
        op: base + (1 if idx < remainder else 0)
        for idx, op in enumerate(operators)
    }


def tier_plan(count: int) -> list[str]:
    raw = []
    assigned = 0
    for tier, weight in TIER_WEIGHTS:
        tier_count = int(round(count * weight))
        raw.extend([tier] * tier_count)
        assigned += tier_count
    while assigned < count:
        raw.append("medium")
        assigned += 1
    if len(raw) > count:
        raw = raw[:count]
    return raw


def make_case_plan(
    total_cases: int,
    operators: tuple[str, ...],
    seed: int,
    shuffle: bool,
) -> list[PlanItem]:
    counts = allocate_counts(total_cases, operators)
    plan: list[PlanItem] = []
    for op_index, operator in enumerate(operators):
        tiers = tier_plan(counts[operator])
        for local_index, tier in enumerate(tiers):
            tp_size = TP_VALUES[(local_index + op_index) % len(TP_VALUES)]
            plan.append(PlanItem(operator=operator, tier=tier, tp_size=tp_size))
    if shuffle:
        random.Random(seed).shuffle(plan)
    return plan


def force_plan_tp(plan: list[PlanItem], tp_size: int | None) -> list[PlanItem]:
    if tp_size is None:
        return plan
    return [
        PlanItem(operator=item.operator, tier=item.tier, tp_size=tp_size)
        for item in plan
    ]


def result(
    ctx: CaseContext,
    api: str,
    scenario: str,
    batch_axis: str,
    conclusion: str,
    satisfied: bool | None,
    bitwise_equal: bool | None,
    dtype: str,
    shape: Any,
    detail: str,
) -> CaseResult:
    return CaseResult(
        case_id=ctx.case_id,
        operator=ctx.operator,
        api=api,
        tp_size=ctx.tp.tp_size,
        tier=ctx.tier,
        scenario=scenario,
        batch_axis=batch_axis,
        conclusion=conclusion,
        satisfied=satisfied,
        bitwise_equal=bitwise_equal,
        dtype=dtype,
        shape=json.dumps(shape, sort_keys=True) if not isinstance(shape, str) else shape,
        detail=detail,
    )


def planned(
    ctx: CaseContext,
    api: str,
    scenario: str,
    batch_axis: str,
    dtype: str,
    shape: Any,
    detail: str,
) -> CaseResult:
    return result(
        ctx,
        api,
        scenario,
        batch_axis,
        "PLANNED",
        None,
        None,
        dtype,
        shape,
        detail,
    )


def load_runtime(device_id: int, dtype_name: str, seed: int) -> None:
    global TORCH, TORCH_NPU, BATCH_INVARIANT_OPS, DTYPE

    import torch
    import torch_npu
    import batch_invariant_ops

    TORCH = torch
    TORCH_NPU = torch_npu
    BATCH_INVARIANT_OPS = batch_invariant_ops

    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is unavailable in this Python environment")
    if not torch.npu.is_available():
        count = None
        try:
            count = torch.npu.device_count()
        except Exception:
            count = "unknown"
        raise RuntimeError(
            "NPU execution requested, but torch.npu.is_available() is False "
            f"(device_count={count}). Use --dry-run here or run on an Ascend host."
        )
    if device_id >= torch.npu.device_count():
        raise RuntimeError(
            f"device_id={device_id} is out of range; device_count={torch.npu.device_count()}"
        )

    torch.npu.set_device(device_id)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype={dtype_name}")
    DTYPE = dtype_map[dtype_name]


def dtype_text(dtype: Any) -> str:
    if isinstance(dtype, str):
        return dtype
    return str(dtype).replace("torch.", "")


def sync() -> None:
    TORCH.npu.synchronize()


def npu_randn(shape: tuple[int, ...], dtype: Any | None = None) -> Any:
    dtype = DTYPE if dtype is None else dtype
    return TORCH.randn(shape, dtype=TORCH.float32, device="npu").to(dtype)


def npu_rand(shape: tuple[int, ...], dtype: Any | None = None) -> Any:
    dtype = DTYPE if dtype is None else dtype
    return TORCH.rand(shape, dtype=TORCH.float32, device="npu").to(dtype)


# === Input value ranges captured from real vLLM inference on 2x NPU3 ===
# Model: Qwen3-30B-A3B, TP=2, bfloat16, batch-invariant mode.
# Captured min/max widened by 1.5x (activation) / 2x (cache/weight) for generalization.
# Format: {scenario: {tensor_name: (low, high)}}
#
# "inferred" = operator not directly called in this model config;
# range estimated from data-flow analysis of captured sister operators.

_OP_VALUE_RANGES = {
    # --- torch_npu.npu_swiglu (captured) ---
    "swiglu": {
        "x": (-125.0, 103.0),       # captured [-83, 68.5] ×1.5
    },
    # --- torch_npu.npu_rms_norm (captured) ---
    "rms_norm_x": {
        "x": (-22.0, 22.0),         # captured [-14.8, 14.6] ×1.5
    },
    "rms_norm_weight": {
        "weight": (-2.5, 52.0),     # captured [-1.6, 34.5] ×1.5; mean≈1.4 → preserve asymmetry
    },
    # --- torch_npu.npu_scatter_pa_kv_cache (captured) ---
    "scatter_key": {
        "key": (-216.0, 198.0),     # captured [-144, 132] ×1.5
    },
    "scatter_value": {
        "value": (-15.0, 22.0),     # captured [-9.6, 14.8] ×1.5
    },
    "scatter_cache": {
        "key_cache": (-15.0, 41.0),   # captured [-7.2, 20.2] ×2
        "value_cache": (-1.8, 4.2),   # captured [-0.9, 2.1] ×2
    },
    # --- torch_npu.npu_moe_init_routing_v2 (inferred) ---
    "moe_init_hidden": {
        "hidden_states": (-30.0, 30.0),  # inferred from rms_norm.x upstream
    },
    # --- torch_npu.npu_moe_token_unpermute (captured) ---
    "moe_unpermute_tokens": {
        "tokens": (-348.0, 1032.0), # captured [-232, 688] ×1.5
    },
    # --- torch_npu.npu_grouped_matmul (captured) ---
    "gmm_x": {
        "x": (-368.0, 696.0),       # captured [-245, 464] ×1.5
    },
    "gmm_weight": {
        "weight": (-1.6, 3.2),      # captured [-0.8, 1.6] ×2
    },
    # --- torch_npu.npu_moe_gating_top_k (inferred) ---
    "router_logits": {
        "logits": (-20.0, 20.0),    # inferred from gate weight + hidden states
    },
    # --- FIA attention (captured) ---
    "fia_query": {
        "query": (-114.0, 126.0),   # captured [-76, 84] ×1.5
    },
    "fia_key": {
        "key": (-216.0, 198.0),     # captured [-144, 132] ×1.5
    },
    "fia_value": {
        "value": (-20.0, 30.0),     # captured [-9.6, 14.8] ×2
    },
    "fia_cache": {
        "cache": (-15.0, 41.0),     # captured [-7.2, 20.2] ×2 (same as scatter cache)
    },
    # --- batch_invariant matmul (captured) ---
    "matmul_self": {
        "self": (-85.0, 231.0),     # captured [-56.2, 154] ×1.5
    },
    "matmul_other": {
        "other": (-3.0, 3.2),       # captured [-1.5, 1.6] ×2
    },
    # --- batch_invariant reduce_sum (captured) ---
    "reduce_softmax": {
        "x": (0.0, 1.0),            # captured [0, 1] — softmax output, not widened
    },
    # --- torch.scatter_ (inferred) ---
    "scatter_mask": {
        "self": (0.0, 0.0),         # zero-filled mask
    },
    "scatter_logits": {
        "src": (-22.0, 22.0),       # inferred from rms_norm.x upstream
    },
}


def _rng_uniform(low: float, high: float, shape: tuple[int, ...],
                 dtype: Any | None = None) -> Any:
    """Generate tensor with values uniformly distributed in [low, high]."""
    dtype = DTYPE if dtype is None else dtype
    return (TORCH.rand(shape, dtype=TORCH.float32, device="npu") * (high - low) + low).to(dtype)


def _rng_tensor(scenario: str, tensor_name: str, shape: tuple[int, ...],
                dtype: Any | None = None, shift: float = 0.0) -> Any:
    """Generate tensor with values in the generalized range for the given scenario."""
    low, high = _OP_VALUE_RANGES[scenario][tensor_name]
    return _rng_uniform(low + shift, high + shift, shape, dtype=dtype)


def to_cpu(value: Any) -> Any:
    return value.detach().cpu().contiguous()


def flatten(value: Any) -> list[Any]:
    if isinstance(value, TORCH.Tensor):
        return [to_cpu(value)]
    if isinstance(value, (tuple, list)):
        out: list[Any] = []
        for item in value:
            out.extend(flatten(item))
        return out
    raise TypeError(f"unsupported output type: {type(value)!r}")


def bitwise_equal(a: Any, b: Any) -> bool:
    if a.dtype in (TORCH.float16, TORCH.bfloat16):
        return bool(TORCH.equal(a.view(TORCH.int16), b.view(TORCH.int16)))
    if a.dtype == TORCH.float32:
        return bool(TORCH.equal(a.view(TORCH.int32), b.view(TORCH.int32)))
    return bool(TORCH.equal(a, b))


def compare(left: Any, right: Any) -> tuple[bool, str]:
    lhs = flatten(left)
    rhs = flatten(right)
    if len(lhs) != len(rhs):
        return False, f"output_count {len(lhs)} != {len(rhs)}"
    for index, (a, b) in enumerate(zip(lhs, rhs)):
        if a.shape != b.shape or a.dtype != b.dtype:
            return (
                False,
                f"output {index} metadata {tuple(a.shape)}/{a.dtype} != "
                f"{tuple(b.shape)}/{b.dtype}",
            )
        if not bitwise_equal(a, b):
            if a.is_floating_point():
                diff = (a.float() - b.float()).abs()
                return (
                    False,
                    f"output {index} not bitwise equal; "
                    f"max_diff={float(diff.max()):.6e}; mean_diff={float(diff.mean()):.6e}",
                )
            return False, f"output {index} not bitwise equal"
    return True, ""


def compare_result(
    ctx: CaseContext,
    api: str,
    scenario: str,
    batch_axis: str,
    full: Any,
    split: Any,
    dtype: Any,
    shape: Any,
    detail: str,
) -> CaseResult:
    sync()
    exact, compare_detail = compare(full, split)
    joined = "; ".join(item for item in (detail, compare_detail) if item)
    return result(
        ctx,
        api,
        scenario,
        batch_axis,
        "PASS" if exact else "FAIL",
        exact,
        exact,
        dtype_text(dtype),
        shape,
        joined,
    )


def failure_result(ctx: CaseContext, exc: BaseException) -> CaseResult:
    message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    message = message.replace("\n", " | ")
    return result(
        ctx,
        ctx.operator,
        "case_exception",
        "unknown",
        "CALL_FAILED",
        None,
        None,
        dtype_text(DTYPE) if DTYPE is not None else "unallocated",
        "unknown",
        message,
    )


def sample_from_tier(ctx: CaseContext, small: list[int], medium: list[int], large: list[int], boundary: list[int]) -> int:
    choices = {
        "small": small,
        "medium": medium,
        "large": large,
        "boundary": boundary,
    }[ctx.tier]
    return ctx.rng.choice(choices)


def cap(value: int, limit: int) -> int:
    return max(1, min(value, limit))


def sample_token_rows(ctx: CaseContext) -> int:
    value = sample_from_tier(
        ctx,
        small=[1, 2, 4, 8, 16, 32, 64],
        medium=[96, 128, 256, 512],
        large=[768, 1024, 1536, 2048],
        boundary=[1, 64, 512, 2048, 8192, MODEL.max_prompt_num * MODEL.max_prompt_len],
    )
    return cap(value, ctx.max_token_rows)


def sample_expert_rows(ctx: CaseContext) -> int:
    value = sample_from_tier(
        ctx,
        small=[8, 16, 32, 64, 128],
        medium=[256, 512, 1024, 2048],
        large=[4096, 8192, 16384],
        boundary=[8, 512, 4096, 65536, MODEL.max_prompt_num * MODEL.max_prompt_len * MODEL.num_experts_per_tok],
    )
    return cap(value, ctx.max_expert_rows)


def sample_sampler_rows(ctx: CaseContext) -> int:
    value = sample_from_tier(
        ctx,
        small=[1, 2, 4, 8],
        medium=[16, 32],
        large=[64, 128],
        boundary=[1, 64, 256, 512],
    )
    return cap(value, ctx.max_sampler_rows)


def sample_sample_size(ctx: CaseContext) -> int:
    value = sample_from_tier(
        ctx,
        small=[8, 16, 32, 64],
        medium=[128, 256, 512],
        large=[1024, 2048],
        boundary=[1, 64, 512, 2048, 4096],
    )
    return cap(value, ctx.max_sample_size)


def sample_fia_batch(ctx: CaseContext) -> int:
    value = sample_from_tier(
        ctx,
        small=[1, 2],
        medium=[2, 3, 4],
        large=[4, 6, 8],
        boundary=[1, 4, 8, MODEL.max_prompt_num],
    )
    return cap(value, ctx.max_fia_batch)


def sample_fia_seq(ctx: CaseContext, decode: bool) -> int:
    if decode:
        choices = {
            "small": [16, 32, 64],
            "medium": [128, 256],
            "large": [512, 1024],
            "boundary": [16, 512, 2048, 22528],
        }[ctx.tier]
    else:
        choices = {
            "small": [1, 2, 4, 8, 16],
            "medium": [32, 64, 128],
            "large": [256, 512, 1024],
            "boundary": [1, 64, 2048],
        }[ctx.tier]
    return cap(ctx.rng.choice(choices), ctx.max_fia_seq)


def sample_fia_kv_lengths(ctx: CaseContext, batch: int) -> list[int]:
    """Per-sequence KV lengths for a mixed-length decode batch.

    Mirrors the online DecodeOnly shape (attention_v1.py:1245/:1255:
    actual_seq_lengths_kv = attn_metadata.seq_lens_list) where sequences at
    different absolute positions share one FIA call with different kv lengths.
    """
    base = sample_fia_seq(ctx, decode=True)
    deltas = [0, 0, 1, 3, 16, 32, 64, 128, 256, -1, -16, -32]
    lengths = [max(1, cap(base + ctx.rng.choice(deltas), ctx.max_fia_seq)) for _ in range(batch)]
    if batch >= 2 and len(set(lengths)) == 1:
        unique = sorted({max(1, base - 1), base, min(ctx.max_fia_seq, base + 1)})
        lengths = [unique[idx % len(unique)] for idx in range(batch)]
    return lengths


def sample_fia_tnd_tokens(ctx: CaseContext) -> int:
    choices = {
        "small": [1, 2, 4, 8, 16],
        "medium": [32, 64, 128],
        "large": [256, 512, 1024],
        "boundary": [1, 64, 512, 2048],
    }[ctx.tier]
    return cap(ctx.rng.choice(choices), ctx.max_fia_seq)


def sample_fia_chunked_prefill_tokens(ctx: CaseContext) -> int:
    if ctx.fia_chunked_prefill_tokens is not None:
        return ctx.fia_chunked_prefill_tokens
    choices = {
        "small": [32, 64, 128],
        "medium": [256, 512, 1024],
        "large": [1024, 2048, 4096],
        "boundary": [128, 2048, 8192, ctx.max_fia_seq],
    }[ctx.tier]
    tokens = cap(ctx.rng.choice(choices), ctx.max_fia_seq)
    return max(2, tokens) if ctx.max_fia_seq >= 2 else tokens


def make_fia_chunk_sizes(
    tokens: int,
    rng: random.Random,
    max_chunk_size: int | None = None,
) -> list[int]:
    if tokens <= 1:
        return [tokens]
    if max_chunk_size is not None:
        budget = min(max_chunk_size, tokens - 1)
    else:
        budgets = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        valid_budgets = [budget for budget in budgets if budget < tokens]
        min_budget = max(1, tokens // 12)
        candidates = [budget for budget in valid_budgets if budget >= min_budget]
        budget = rng.choice(candidates or valid_budgets) if valid_budgets else 1
    chunks = []
    offset = 0
    while offset < tokens:
        window = min(budget, tokens - offset)
        must_split = window > 1 and (window == budget or not chunks)
        chunks.extend(random_subchunks(window, rng, must_split))
        offset += window
    if len(chunks) == 1:
        first = max(1, tokens // 2)
        chunks = [first, tokens - first]
    return chunks


def random_subchunks(length: int, rng: random.Random, must_split: bool) -> list[int]:
    if length <= 1:
        return [length]
    if length >= 128:
        max_parts = 4
    elif length >= 16:
        max_parts = 3
    else:
        max_parts = 2
    max_parts = min(max_parts, length)
    min_parts = 2 if must_split and max_parts >= 2 else 1
    part_count = rng.randint(min_parts, max_parts)
    if part_count == 1:
        return [length]
    cuts = sorted(rng.sample(range(1, length), part_count - 1))
    bounds = [0] + cuts + [length]
    return [bounds[idx + 1] - bounds[idx] for idx in range(len(bounds) - 1)]


def split_slices(length: int, rng: random.Random, parts: int | None = None) -> list[slice]:
    if length <= 1:
        return [slice(0, length)]
    if SPLIT_FORCE_SINGLETONS:
        return [slice(idx, idx + 1) for idx in range(length)]
    if parts is None:
        max_parts = length if SPLIT_MAX_PARTS is None else min(SPLIT_MAX_PARTS, length)
        parts = rng.randint(2, max_parts)
    else:
        parts = min(parts, length)
    cuts = sorted(rng.sample(range(1, length), parts - 1))
    bounds = [0] + cuts + [length]
    return [slice(bounds[idx], bounds[idx + 1]) for idx in range(len(bounds) - 1)]


def shuffled_slices(length: int, rng: random.Random, parts: int | None = None) -> list[slice]:
    slices = split_slices(length, rng, parts)
    rng.shuffle(slices)
    return slices


def restore_cat(parts: list[tuple[slice, Any]], dim: int) -> Any:
    return TORCH.cat([tensor for _, tensor in sorted(parts, key=lambda item: item[0].start)], dim=dim)


def restore_outputs_along_dim(full: Any, parts: list[tuple[slice, Any]], dim: int = 0) -> Any:
    if isinstance(full, TORCH.Tensor):
        return restore_cat(parts, dim=dim)
    restored = []
    for out_index in range(len(full)):
        restored.append(restore_cat([(slc, value[out_index]) for slc, value in parts], dim=dim))
    return tuple(restored)


def case_swiglu(ctx: CaseContext) -> CaseResult:
    api = "torch_npu.npu_swiglu"
    if ctx.rng.random() < 0.45:
        scenario = "dense_shared_expert"
        rows = sample_token_rows(ctx)
        last_dim = ctx.tp.shared_intermediate * 2
    else:
        scenario = "moe_routed_expert"
        rows = sample_expert_rows(ctx)
        last_dim = MODEL.moe_intermediate_size * 2
    shape = {"x": [rows, last_dim], "out": [rows, last_dim // 2]}
    detail = (
        f"qwen3 tp={ctx.tp.tp_size}; hidden={MODEL.hidden_size}; "
        "split+shuffle+restore on dim0"
    )
    if ctx.dry_run:
        return planned(ctx, api, scenario, "x dim0", "bfloat16", shape, detail)
    x = _rng_tensor("swiglu", "x", (rows, last_dim))
    full = TORCH_NPU.npu_swiglu(x, dim=-1)
    parts = [
        (slc, TORCH_NPU.npu_swiglu(x[slc].contiguous(), dim=-1))
        for slc in shuffled_slices(rows, ctx.rng)
    ]
    return compare_result(ctx, api, scenario, "x dim0", full, restore_cat(parts, dim=0), DTYPE, shape, detail)


def case_scatter_pa_kv_cache(ctx: CaseContext) -> CaseResult:
    api = "torch_npu.npu_scatter_pa_kv_cache"
    rows = sample_token_rows(ctx)
    heads = ctx.tp.kv_heads
    block_size = MODEL.block_size
    min_slots = max(256 * block_size, rows + block_size)
    num_blocks = math.ceil(min_slots / block_size)
    total_slots = num_blocks * block_size
    cache_shape = [num_blocks, block_size, heads, MODEL.head_dim]
    shape = {
        "key": [rows, heads, MODEL.head_dim],
        "value": [rows, heads, MODEL.head_dim],
        "key_cache": cache_shape,
        "value_cache": cache_shape,
        "slot_mapping": [rows],
    }
    detail = (
        f"qwen3 tp={ctx.tp.tp_size}; cache_mode=Norm; unique slot_mapping; "
        "split+shuffle writes"
    )
    if ctx.dry_run:
        return planned(ctx, api, "kv_cache_token_scatter", "key/value dim0 + slot_mapping dim0", "bfloat16", shape, detail)

    key = _rng_tensor("scatter_key", "key", (rows, heads, MODEL.head_dim))
    value = _rng_tensor("scatter_value", "value", (rows, heads, MODEL.head_dim))
    slot_dtype = ctx.rng.choice([TORCH.int32, TORCH.int64])
    slots = TORCH.randperm(total_slots, dtype=TORCH.int64, device="npu")[:rows].to(slot_dtype).contiguous()
    key_base = _rng_tensor("scatter_cache", "key_cache", tuple(cache_shape))
    value_base = _rng_tensor("scatter_cache", "value_cache", tuple(cache_shape))

    key_full = key_base.clone()
    value_full = value_base.clone()
    TORCH_NPU.npu_scatter_pa_kv_cache(key, value, key_full, value_full, slots, cache_mode="Norm")

    key_split = key_base.clone()
    value_split = value_base.clone()
    for slc in shuffled_slices(rows, ctx.rng):
        TORCH_NPU.npu_scatter_pa_kv_cache(
            key[slc].contiguous(),
            value[slc].contiguous(),
            key_split,
            value_split,
            slots[slc].contiguous(),
            cache_mode="Norm",
        )
    return compare_result(
        ctx,
        api,
        "kv_cache_token_scatter",
        "key/value dim0 + slot_mapping dim0",
        (key_full, value_full),
        (key_split, value_split),
        DTYPE,
        shape,
        f"{detail}; slot_dtype={dtype_text(slot_dtype)}",
    )


def make_sampler_indices(rows: int, sample_size: int, vocab: int) -> Any:
    cols = TORCH.arange(sample_size, dtype=TORCH.int64, device="npu").reshape(1, sample_size)
    offsets = TORCH.arange(rows, dtype=TORCH.int64, device="npu").reshape(rows, 1) * sample_size
    return ((cols + offsets) % vocab).contiguous()


def case_scatter(ctx: CaseContext) -> CaseResult:
    api = "torch.scatter_"
    if ctx.rng.random() < 0.75:
        rows = sample_token_rows(ctx)
        n_group = ctx.rng.choice([8, 16, 32])
        topk_group = min(ctx.rng.choice([1, 2, 4, 8]), n_group)
        shape = {"group_mask": [rows, n_group], "group_idx": [rows, topk_group]}
        detail = "grouped_topk mask path; scatter dim=1; split+shuffle+restore"
        if ctx.dry_run:
            return planned(ctx, api, "grouped_topk_router_group_mask", "dim0", "bfloat16", shape, detail)
        group_idx = TORCH.randint(0, n_group, (rows, topk_group), dtype=TORCH.int64, device="npu")
        base = TORCH.zeros((rows, n_group), dtype=DTYPE, device="npu")
        full = base.clone()
        full.scatter_(1, group_idx, 1.0)
        parts = []
        for slc in shuffled_slices(rows, ctx.rng):
            out = base[slc].clone()
            out.scatter_(1, group_idx[slc].contiguous(), 1.0)
            parts.append((slc, out))
        return compare_result(ctx, api, "grouped_topk_router_group_mask", "dim0", full, restore_cat(parts, 0), DTYPE, shape, detail)

    rows = sample_sampler_rows(ctx)
    sample_size = sample_sample_size(ctx)
    shape = {"logits": [rows, MODEL.vocab_size], "indices": [rows, sample_size], "src": [rows, sample_size]}
    detail = "sampler logits path from existing tests; scatter dim=-1; row-local indices"
    if ctx.dry_run:
        return planned(ctx, api, "sampler_logits_scatter", "dim0", "bfloat16", shape, detail)
    logits = _rng_tensor("scatter_logits", "src", (rows, MODEL.vocab_size))
    logits_idx = make_sampler_indices(rows, sample_size, MODEL.vocab_size)
    logits_src = _rng_tensor("scatter_logits", "src", (rows, sample_size))
    full = logits.clone()
    full.scatter_(-1, logits_idx, logits_src)
    parts = []
    for slc in shuffled_slices(rows, ctx.rng):
        out = logits[slc].clone()
        out.scatter_(-1, logits_idx[slc].contiguous(), logits_src[slc].contiguous())
        parts.append((slc, out))
    return compare_result(ctx, api, "sampler_logits_scatter", "dim0", full, restore_cat(parts, 0), DTYPE, shape, detail)


def rms_call(x: Any, weight: Any) -> Any:
    return TORCH_NPU.npu_rms_norm(x, weight, epsilon=1e-6)


def case_rms_norm(ctx: CaseContext) -> CaseResult:
    api = "torch_npu.npu_rms_norm"
    rows = sample_token_rows(ctx)
    point = ctx.rng.random()
    if point < 0.60:
        scenario = "layer_norm"
        x_shape = [rows, MODEL.hidden_size]
        weight_shape = [MODEL.hidden_size]
    elif point < 0.85:
        scenario = "attention_q_norm"
        x_shape = [rows, ctx.tp.q_heads, MODEL.head_dim]
        weight_shape = [MODEL.head_dim]
    else:
        scenario = "attention_kv_norm"
        x_shape = [rows, ctx.tp.kv_heads, MODEL.head_dim]
        weight_shape = [MODEL.head_dim]
    shape = {"x": x_shape, "weight": weight_shape}
    detail = "epsilon=1e-6; split+shuffle+restore on token dim0"
    if ctx.dry_run:
        return planned(ctx, api, scenario, "x dim0", "bfloat16", shape, detail)
    x = _rng_tensor("rms_norm_x", "x", tuple(x_shape))
    weight = _rng_tensor("rms_norm_weight", "weight", tuple(weight_shape), shift=1.4 if "layer" in scenario else 0.0)
    full = rms_call(x, weight)
    split_outputs = []
    slices = shuffled_slices(rows, ctx.rng)
    for out_index in range(len(full)):
        parts = [(slc, rms_call(x[slc].contiguous(), weight)[out_index]) for slc in slices]
        split_outputs.append(restore_cat(parts, dim=0))
    return compare_result(ctx, api, scenario, "x dim0", full, tuple(split_outputs), DTYPE, shape, detail)


def make_topk_ids(rows: int, topk: int, experts: int, offset: int) -> Any:
    row = TORCH.arange(rows, dtype=TORCH.int32, device="npu").reshape(rows, 1)
    col = TORCH.arange(topk, dtype=TORCH.int32, device="npu").reshape(1, topk)
    return ((row + col + offset) % experts).contiguous()


def normalized_probs(rows: int, topk: int) -> Any:
    probs = TORCH.rand((rows, topk), dtype=TORCH.float32, device="npu")
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs.to(DTYPE)


def moe_init_finalize_call(x: Any, expert_idx: Any, probs: Any, scale: Any, experts: int) -> Any:
    expanded_x, expanded_row_idx, _, _ = TORCH_NPU.npu_moe_init_routing_v2(
        x,
        expert_idx,
        scale=scale,
        active_num=x.shape[0] * expert_idx.shape[1],
        expert_capacity=-1,
        expert_num=experts,
        drop_pad_mode=0,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        quant_mode=-1,
        active_expert_range=[0, experts],
        row_idx_type=1,
    )
    total = x.shape[0] * expert_idx.shape[1]
    gather_indices = TORCH.empty((total,), dtype=expanded_row_idx.dtype, device=expanded_row_idx.device)
    gather_indices[expanded_row_idx.to(TORCH.int64)] = TORCH.arange(
        total, dtype=expanded_row_idx.dtype, device=expanded_row_idx.device
    )
    return TORCH_NPU.npu_moe_token_unpermute(
        expanded_x.contiguous(),
        gather_indices.contiguous(),
        probs=probs,
    )


def selected_and_shuffled_rows(rows: int, rng: random.Random) -> tuple[list[int], list[int]]:
    if rows <= 1:
        return [0], [0]
    if rows <= 4:
        selected_count = rows
    else:
        low = max(1, rows // 4)
        high = max(low, min(rows, (rows * 3) // 4))
        selected_count = rng.randint(low, high)
    selected = sorted(rng.sample(range(rows), selected_count))
    shuffled = selected[:]
    for _ in range(10):
        rng.shuffle(shuffled)
        if shuffled != selected:
            break
    return selected, shuffled


def case_moe_init_routing(ctx: CaseContext) -> CaseResult:
    api = "torch_npu.npu_moe_init_routing_v2"
    rows = sample_token_rows(ctx)
    topk = MODEL.num_experts_per_tok
    experts = ctx.tp.local_experts
    selected, shuffled = selected_and_shuffled_rows(rows, ctx.rng)
    shape = {
        "hidden_states": [rows, MODEL.hidden_size],
        "topk_ids": [rows, topk],
        "expanded_hidden": [rows * topk, MODEL.hidden_size],
        "expanded_row_idx": [rows * topk],
        "expert_tokens": [experts],
    }
    detail = (
        "init output order is expert-grouped, so the executable check uses "
        "init + token_unpermute as the batch-invariant oracle; subset+shuffle+restore"
    )
    if ctx.dry_run:
        return planned(ctx, api, "paired_init_unpermute_oracle", "hidden_states/topk_ids dim0", "bfloat16", shape, detail)

    x = _rng_tensor("moe_init_hidden", "hidden_states", (rows, MODEL.hidden_size))
    topk_ids = make_topk_ids(rows, topk, experts, ctx.rng.randint(0, experts - 1))
    probs = normalized_probs(rows, topk)
    scale = TORCH.rand((rows,), dtype=TORCH.float32, device="npu") + 0.25
    full = moe_init_finalize_call(x, topk_ids, probs, scale, experts)

    selected_index = TORCH.tensor(selected, dtype=TORCH.int64, device="npu")
    shuffled_index = TORCH.tensor(shuffled, dtype=TORCH.int64, device="npu")
    local = moe_init_finalize_call(
        x.index_select(0, shuffled_index).contiguous(),
        topk_ids.index_select(0, shuffled_index).contiguous(),
        probs.index_select(0, shuffled_index).contiguous(),
        scale.index_select(0, shuffled_index).contiguous(),
        experts,
    )
    local_cpu = to_cpu(local)
    restored_cpu = TORCH.empty_like(local_cpu)
    selected_pos = {row: pos for pos, row in enumerate(selected)}
    for local_pos, original_row in enumerate(shuffled):
        restored_cpu[selected_pos[original_row]] = local_cpu[local_pos]

    return compare_result(
        ctx,
        api,
        "paired_init_unpermute_oracle",
        "hidden_states/topk_ids dim0",
        full.index_select(0, selected_index),
        restored_cpu,
        DTYPE,
        shape,
        f"{detail}; selected_rows={len(selected)}; local_experts={experts}",
    )


def case_moe_token_unpermute(ctx: CaseContext) -> CaseResult:
    api = "torch_npu.npu_moe_token_unpermute"
    rows = sample_token_rows(ctx)
    topk = MODEL.num_experts_per_tok
    total = rows * topk
    shape = {
        "permuted_tokens": [total, MODEL.hidden_size],
        "sorted_indices": [total],
        "probs": [rows, topk],
        "out": [rows, MODEL.hidden_size],
    }
    detail = "canonical topk order; int32 sorted_indices; split+shuffle+restore on token rows"
    if ctx.dry_run:
        return planned(ctx, api, "weighted_token_restore", "probs dim0 / token rows", "bfloat16", shape, detail)

    permuted = _rng_tensor("moe_unpermute_tokens", "tokens", (total, MODEL.hidden_size))
    sorted_indices = TORCH.arange(total, dtype=TORCH.int32, device="npu")
    probs = normalized_probs(rows, topk)
    full = TORCH_NPU.npu_moe_token_unpermute(permuted, sorted_indices, probs=probs)

    parts = []
    for slc in shuffled_slices(rows, ctx.rng):
        start = slc.start * topk
        stop = slc.stop * topk
        local_total = stop - start
        local_indices = TORCH.arange(local_total, dtype=TORCH.int32, device="npu")
        local = TORCH_NPU.npu_moe_token_unpermute(
            permuted[start:stop].contiguous(),
            local_indices,
            probs=probs[slc].contiguous(),
        )
        parts.append((slc, local))
    return compare_result(ctx, api, "weighted_token_restore", "probs dim0 / token rows", full, restore_cat(parts, 0), DTYPE, shape, detail)


def random_counts(total: int, groups: int, rng: random.Random) -> list[int]:
    counts = [0] * groups
    if total <= groups:
        for idx in rng.sample(range(groups), total):
            counts[idx] = 1
        return counts
    cuts = sorted(rng.sample(range(total + groups - 1), groups - 1))
    prev = -1
    for idx, cut in enumerate(cuts + [total + groups - 1]):
        counts[idx] = cut - prev - 1
        prev = cut
    if sum(counts) != total:
        counts[-1] += total - sum(counts)
    return counts


def expert_ranges(counts: list[int]) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for count in counts:
        stop = start + count
        ranges.append((start, stop))
        start = stop
    return ranges


def overlap_counts(counts: list[int], slc: slice) -> list[int]:
    output = []
    for start, stop in expert_ranges(counts):
        output.append(max(0, min(stop, slc.stop) - max(start, slc.start)))
    return output


def gmm_call(x: Any, weight: Any, counts: list[int]) -> Any:
    group_list = TORCH.tensor(counts, dtype=TORCH.int64, device="npu")
    return TORCH_NPU.npu_grouped_matmul(
        [x],
        [weight],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]


def case_grouped_matmul(ctx: CaseContext) -> CaseResult:
    api = "torch_npu.npu_grouped_matmul"
    rows = sample_expert_rows(ctx)
    experts = ctx.tp.local_experts
    if ctx.rng.random() < 0.52:
        scenario = "gmm1_gate_up"
        k_dim = MODEL.hidden_size
        n_dim = MODEL.moe_intermediate_size * 2
    else:
        scenario = "gmm2_down"
        k_dim = MODEL.moe_intermediate_size
        n_dim = MODEL.hidden_size
    shape = {"x": [rows, k_dim], "weight": [experts, k_dim, n_dim], "group_list": [experts]}
    detail = "count-mode group_list; contiguous M-axis slices with recomputed local counts"
    if ctx.dry_run:
        return planned(ctx, api, scenario, "x dim0 / sorted expert-token M", "bfloat16", shape, detail)

    counts = random_counts(rows, experts, ctx.rng)
    x = _rng_tensor("gmm_x", "x", (rows, k_dim))
    weight = _rng_tensor("gmm_weight", "weight", (experts, k_dim, n_dim))
    full = gmm_call(x, weight, counts)
    parts = []
    for slc in shuffled_slices(rows, ctx.rng):
        parts.append((slc, gmm_call(x[slc].contiguous(), weight, overlap_counts(counts, slc))))
    return compare_result(ctx, api, scenario, "x dim0 / sorted expert-token M", full, restore_cat(parts, 0), DTYPE, shape, detail)


def case_moe_gating_top_k(ctx: CaseContext) -> CaseResult:
    api = "torch_npu.npu_moe_gating_top_k"
    rows = sample_token_rows(ctx)
    topk = MODEL.num_experts_per_tok
    shape = {
        "router_logits": [rows, MODEL.num_experts],
        "topk": topk,
        "checked_outputs": ["y_out", "expert_idx_out"],
    }
    detail = (
        "Qwen3 standard topk: k_group=1, group_count=1, renorm=0, norm_type=0; "
        "compare y_out/expert_idx_out only because norm_out is an optional auxiliary output"
    )
    if ctx.dry_run:
        return planned(ctx, api, "qwen3_standard_topk", "router_logits dim0", "float32", shape, detail)

    logits = _rng_tensor("router_logits", "logits", (rows, MODEL.num_experts), dtype=TORCH.float32)
    full = TORCH_NPU.npu_moe_gating_top_k(
        logits,
        topk,
        k_group=1,
        group_count=1,
        group_select_mode=0,
        renorm=0,
        norm_type=0,
        out_flag=False,
    )
    full_checked = full[:2]
    parts = []
    for slc in shuffled_slices(rows, ctx.rng):
        local = TORCH_NPU.npu_moe_gating_top_k(
            logits[slc].contiguous(),
            topk,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=0,
            out_flag=False,
        )
        parts.append((slc, local[:2]))
    split = restore_outputs_along_dim(full_checked, parts, dim=0)
    return compare_result(ctx, api, "qwen3_standard_topk", "router_logits dim0", full_checked, split, "float32", shape, detail)


def causal_mask_2048() -> Any:
    return TORCH.triu(
        TORCH.ones((2048, 2048), dtype=TORCH.int8, device="npu"),
        diagonal=1,
    ).contiguous()


def unit_tnd_seq_lengths(tokens: int) -> list[int]:
    return list(range(1, tokens + 1))


def fia_call(
    query: Any,
    key: Any,
    value: Any,
    tp: TPProfile,
    layout: str,
    actual_q: list[int],
    actual_kv: list[int],
    sparse_mode: int,
    atten_mask: Any | None = None,
    block_table: Any | None = None,
    block_size: int = 0,
) -> Any:
    scale = TORCH.tensor(1.0 / math.sqrt(float(MODEL.head_dim)), dtype=TORCH.float32, device="npu")
    return TORCH.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant(
        query,
        key,
        value,
        num_heads=tp.q_heads,
        num_key_value_heads=tp.kv_heads,
        input_layout=layout,
        scale=scale,
        atten_mask=atten_mask,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        sparse_mode=sparse_mode,
        inner_precise=0,
        block_table=block_table,
        block_size=block_size,
        actual_seq_lengths=actual_q,
        actual_seq_lengths_kv=actual_kv,
    )


def fia_attention_output(value: Any) -> Any:
    return value[0] if isinstance(value, (tuple, list)) else value


def make_large_small_fia_tensor(shape: tuple[int, ...], dtype: Any, role: str) -> Any:
    width = shape[-1]
    pattern = TORCH.empty((width,), dtype=TORCH.float32, device="npu")
    if role == "query":
        large, small = 64.0, 1.0 / 65536.0
        pattern[0::4] = large
        pattern[1::4] = small
        pattern[2::4] = large
        pattern[3::4] = -small
    elif role == "key":
        large, small = 64.0, 1.0 / 65536.0
        pattern[0::4] = large
        pattern[1::4] = small
        pattern[2::4] = -large
        pattern[3::4] = small
    elif role == "value":
        large, small = 32768.0, 1.0 / 65536.0
        pattern[0::4] = large
        pattern[1::4] = small
        pattern[2::4] = -large
        pattern[3::4] = -small
    else:
        raise ValueError(f"unsupported FIA large/small tensor role={role}")

    view_shape = (1,) * (len(shape) - 1) + (width,)
    tensor = pattern.to(dtype).reshape(view_shape).expand(shape).clone()
    if shape[0] > 1:
        token_ids = TORCH.arange(shape[0], dtype=TORCH.int32, device="npu")
        token_sign = (1 - 2 * token_ids.remainder(2)).to(dtype)
        tensor.mul_(token_sign.reshape((shape[0],) + (1,) * (len(shape) - 1)))
    return tensor.contiguous()


def make_fia_chunked_prefill_inputs(ctx: CaseContext, tokens: int) -> tuple[Any, Any, Any]:
    query_shape = (tokens, ctx.tp.q_heads, MODEL.head_dim)
    key_shape = (tokens, ctx.tp.kv_heads, MODEL.head_dim)
    value_shape = (tokens, ctx.tp.kv_heads, MODEL.head_dim)
    if ctx.fia_chunked_prefill_data_pattern == "large_small":
        return (
            make_large_small_fia_tensor(query_shape, TORCH.float16, "query"),
            make_large_small_fia_tensor(key_shape, TORCH.float16, "key"),
            make_large_small_fia_tensor(value_shape, TORCH.float16, "value"),
        )
    if ctx.fia_chunked_prefill_data_pattern == "range":
        return (
            _rng_tensor("fia_query", "query", query_shape, dtype=TORCH.float16),
            _rng_tensor("fia_key", "key", key_shape, dtype=TORCH.float16),
            _rng_tensor("fia_value", "value", value_shape, dtype=TORCH.float16),
        )
    raise ValueError(f"unsupported FIA chunked prefill data pattern={ctx.fia_chunked_prefill_data_pattern}")


def make_tnd_paged_kv_cache(key: Any, value: Any, block_size: int) -> tuple[Any, Any, Any]:
    tokens = int(key.shape[0])
    hkv = int(key.shape[1] * key.shape[2])
    blocks = math.ceil(tokens / block_size)
    key_cache = TORCH.zeros((blocks, block_size, hkv), dtype=key.dtype, device=key.device)
    value_cache = TORCH.zeros((blocks, block_size, hkv), dtype=value.dtype, device=value.device)
    key_cache.view(-1, hkv)[:tokens].copy_(key.reshape(tokens, hkv))
    value_cache.view(-1, hkv)[:tokens].copy_(value.reshape(tokens, hkv))
    block_table = TORCH.arange(blocks, dtype=TORCH.int32, device=key.device).reshape(1, blocks)
    return key_cache, value_cache, block_table


def case_fia(ctx: CaseContext) -> CaseResult:
    api = "torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant"
    scenario = ctx.rng.choice(FIA_SCENARIOS)
    if scenario == "tnd_chunked_prefill":
        tokens = sample_fia_chunked_prefill_tokens(ctx)
        chunk_sizes = make_fia_chunk_sizes(tokens, ctx.rng, ctx.fia_chunked_prefill_chunk_size)
        block_size = CHUNKED_PREFILL_BLOCK_SIZE
        blocks_per_seq = math.ceil(tokens / block_size)
        hkv = ctx.tp.kv_heads * MODEL.head_dim
        shape = {
            "query": [tokens, ctx.tp.q_heads, MODEL.head_dim],
            "key_cache": [blocks_per_seq, block_size, hkv],
            "value_cache": [blocks_per_seq, block_size, hkv],
            "block_table": [1, blocks_per_seq],
            "full_actual_seq_lengths": [tokens],
            "full_actual_seq_lengths_kv": [tokens],
            "chunk_sizes": chunk_sizes,
            "data_pattern": ctx.fia_chunked_prefill_data_pattern,
            "large_small_values": {
                "qk_large": 64.0,
                "qk_small": 1.0 / 65536.0,
                "value_large": 32768.0,
                "value_small": 1.0 / 65536.0,
            },
            "atten_mask": [2048, 2048],
        }
        detail = (
            f"TND paged chunked prefill; P={tokens}; block_size={block_size}; "
            f"max_chunk_size={ctx.fia_chunked_prefill_chunk_size or 'random'}; "
            f"data_pattern={ctx.fia_chunked_prefill_data_pattern}; "
            f"q_heads={ctx.tp.q_heads}; kv_heads={ctx.tp.kv_heads}; "
            "full path runs one FIA call with actual_q/kv=[P]; staged path "
            "runs ordered chunks with actual_q=[chunk] and actual_kv=[cumulative_P]"
        )
        if ctx.dry_run:
            return planned(ctx, api, scenario, "prompt P chunk axis", "float16", shape, detail)

        query, key, value = make_fia_chunked_prefill_inputs(ctx, tokens)
        key_cache, value_cache, block_table = make_tnd_paged_kv_cache(key, value, block_size)
        atten_mask = causal_mask_2048()

        full = fia_attention_output(
            fia_call(
                query,
                key_cache,
                value_cache,
                ctx.tp,
                "TND",
                [tokens],
                [tokens],
                3,
                atten_mask,
                block_table,
                block_size,
            )
        )
        parts = []
        offset = 0
        for chunk in chunk_sizes:
            end = offset + chunk
            blocks = math.ceil(end / block_size)
            local = fia_attention_output(
                fia_call(
                    query[offset:end].contiguous(),
                    key_cache,
                    value_cache,
                    ctx.tp,
                    "TND",
                    [chunk],
                    [end],
                    3,
                    atten_mask,
                    block_table[:, :blocks].contiguous(),
                    block_size,
                )
            )
            parts.append(local)
            offset = end
        split = TORCH.cat(parts, dim=0)
        return compare_result(ctx, api, scenario, "prompt P chunk axis", full, split, TORCH.float16, shape, detail)

    if scenario == "tnd_prefill_token_batch":
        tokens = sample_fia_tnd_tokens(ctx)
        shape = {
            "query": [tokens, ctx.tp.q_heads, MODEL.head_dim],
            "key": [tokens, ctx.tp.kv_heads, MODEL.head_dim],
            "value": [tokens, ctx.tp.kv_heads, MODEL.head_dim],
            "actual_seq_lengths": [tokens],
            "actual_seq_lengths_kv": [tokens],
            "atten_mask": [2048, 2048],
        }
        detail = (
            f"TND causal prefill; q_heads={ctx.tp.q_heads}; kv_heads={ctx.tp.kv_heads}; "
            "T is treated as the batch axis by assigning one token per logical request; "
            "actual_seq_lengths are cumulative [1..T] and rebased for every local slice"
        )
        if ctx.dry_run:
            return planned(ctx, api, scenario, "T dim0", "float16", shape, detail)

        query = _rng_tensor("fia_query", "query", (tokens, ctx.tp.q_heads, MODEL.head_dim), dtype=TORCH.float16)
        key = _rng_tensor("fia_key", "key", (tokens, ctx.tp.kv_heads, MODEL.head_dim), dtype=TORCH.float16)
        value = _rng_tensor("fia_value", "value", (tokens, ctx.tp.kv_heads, MODEL.head_dim), dtype=TORCH.float16)
        atten_mask = causal_mask_2048()
        actual = unit_tnd_seq_lengths(tokens)
        full = fia_call(query, key, value, ctx.tp, "TND", actual, actual, 3, atten_mask)
        parts = []
        for slc in shuffled_slices(tokens, ctx.rng):
            local_tokens = slc.stop - slc.start
            local_actual = unit_tnd_seq_lengths(local_tokens)
            local = fia_call(
                query[slc].contiguous(),
                key[slc].contiguous(),
                value[slc].contiguous(),
                ctx.tp,
                "TND",
                local_actual,
                local_actual,
                3,
                atten_mask,
            )
            parts.append((slc, local))
        split = restore_outputs_along_dim(full, parts, dim=0)
        return compare_result(ctx, api, scenario, "T dim0", full, split, TORCH.float16, shape, detail)

    if scenario == "tnd_paged_decode":
        tokens = sample_fia_batch(ctx)
        kv_lengths = sample_fia_kv_lengths(ctx, tokens)
        per_seq_blocks = [math.ceil(kv / MODEL.block_size) for kv in kv_lengths]
        block_offsets = []
        total_blocks = 0
        for blocks in per_seq_blocks:
            block_offsets.append(total_blocks)
            total_blocks += blocks
        max_blocks = max(per_seq_blocks)
        hkv = ctx.tp.kv_heads * MODEL.head_dim
        shape = {
            "query": [tokens, ctx.tp.q_heads, MODEL.head_dim],
            "key_cache": [total_blocks, MODEL.block_size, hkv],
            "value_cache": [total_blocks, MODEL.block_size, hkv],
            "block_table": [tokens, max_blocks],
            "actual_seq_lengths": [tokens],
            "actual_seq_lengths_kv": [tokens],
        }
        kv_preview = kv_lengths[:8]
        kv_preview = f"{kv_preview}{'...' if len(kv_lengths) > 8 else ''}"
        detail = (
            f"TND paged decode with MIXED kv lengths; q_heads={ctx.tp.q_heads}; "
            f"kv_heads={ctx.tp.kv_heads}; block_size={MODEL.block_size}; "
            f"kv_lengths={kv_preview}; actual_seq_lengths_kv is plain per-seq "
            "kv length (vLLM seq_lens_list), sparse_mode=0"
        )
        if ctx.dry_run:
            return planned(ctx, api, scenario, "T dim0", "float16", shape, detail)

        query = _rng_tensor("fia_query", "query", (tokens, ctx.tp.q_heads, MODEL.head_dim), dtype=TORCH.float16)
        key = _rng_tensor("fia_cache", "cache", (total_blocks, MODEL.block_size, hkv), dtype=TORCH.float16)
        value = _rng_tensor("fia_cache", "cache", (total_blocks, MODEL.block_size, hkv), dtype=TORCH.float16)
        block_table = TORCH.zeros((tokens, max_blocks), dtype=TORCH.int32, device="npu")
        for idx, (offset, blocks) in enumerate(zip(block_offsets, per_seq_blocks)):
            block_table[idx, :blocks] = TORCH.arange(offset, offset + blocks, dtype=TORCH.int32, device="npu")
        block_table = block_table.contiguous()
        actual_q = unit_tnd_seq_lengths(tokens)
        actual_kv = kv_lengths
        full = fia_call(
            query,
            key,
            value,
            ctx.tp,
            "TND",
            actual_q,
            actual_kv,
            0,
            None,
            block_table,
            MODEL.block_size,
        )
        parts = []
        for slc in shuffled_slices(tokens, ctx.rng):
            local_tokens = slc.stop - slc.start
            local_q = unit_tnd_seq_lengths(local_tokens)
            local = fia_call(
                query[slc].contiguous(),
                key,
                value,
                ctx.tp,
                "TND",
                local_q,
                actual_kv[slc],
                0,
                None,
                block_table[slc].contiguous(),
                MODEL.block_size,
            )
            parts.append((slc, local))
        split = restore_outputs_along_dim(full, parts, dim=0)
        return compare_result(ctx, api, scenario, "T dim0", full, split, TORCH.float16, shape, detail)

    batch = sample_fia_batch(ctx)
    if scenario == "bsh_paged_decode":
        q_len = 1
        kv_lengths = sample_fia_kv_lengths(ctx, batch)
        per_seq_blocks = [math.ceil(kv / MODEL.block_size) for kv in kv_lengths]
        block_offsets = []
        total_blocks = 0
        for blocks in per_seq_blocks:
            block_offsets.append(total_blocks)
            total_blocks += blocks
        max_blocks = max(per_seq_blocks)
        hq = ctx.tp.q_heads * MODEL.head_dim
        hkv = ctx.tp.kv_heads * MODEL.head_dim
        shape = {
            "query": [batch, q_len, hq],
            "key_cache": [total_blocks, MODEL.block_size, hkv],
            "value_cache": [total_blocks, MODEL.block_size, hkv],
            "block_table": [batch, max_blocks],
            "actual_seq_lengths": [batch],
            "actual_seq_lengths_kv": [batch],
        }
        detail = (
            f"paged decode with MIXED kv lengths; q_heads={ctx.tp.q_heads}; "
            f"kv_heads={ctx.tp.kv_heads}; block_size={MODEL.block_size}; "
            "split+shuffle batch dim"
        )
        if ctx.dry_run:
            return planned(ctx, api, scenario, "batch dim0", "float16", shape, detail)

        query = _rng_tensor("fia_query", "query", (batch, q_len, hq), dtype=TORCH.float16)
        key = _rng_tensor("fia_cache", "cache", (total_blocks, MODEL.block_size, hkv), dtype=TORCH.float16)
        value = _rng_tensor("fia_cache", "cache", (total_blocks, MODEL.block_size, hkv), dtype=TORCH.float16)
        block_table = TORCH.zeros((batch, max_blocks), dtype=TORCH.int32, device="npu")
        for idx, (offset, blocks) in enumerate(zip(block_offsets, per_seq_blocks)):
            block_table[idx, :blocks] = TORCH.arange(offset, offset + blocks, dtype=TORCH.int32, device="npu")
        block_table = block_table.contiguous()
        actual_q = [q_len] * batch
        actual_kv = kv_lengths
        full = fia_call(query, key, value, ctx.tp, "BSH", actual_q, actual_kv, 0, None, block_table, MODEL.block_size)
        parts = []
        for slc in shuffled_slices(batch, ctx.rng):
            local = fia_call(
                query[slc].contiguous(),
                key,
                value,
                ctx.tp,
                "BSH",
                actual_q[slc],
                actual_kv[slc],
                0,
                None,
                block_table[slc].contiguous(),
                MODEL.block_size,
            )
            parts.append((slc, local))
        split = restore_outputs_along_dim(full, parts, dim=0)
        return compare_result(ctx, api, scenario, "batch dim0", full, split, TORCH.float16, shape, detail)

    seq = sample_fia_seq(ctx, decode=False)
    sparse_mode = 3
    if scenario == "bsh_prefill":
        hq = ctx.tp.q_heads * MODEL.head_dim
        hkv = ctx.tp.kv_heads * MODEL.head_dim
        shape = {
            "query": [batch, seq, hq],
            "key": [batch, seq, hkv],
            "value": [batch, seq, hkv],
            "atten_mask": [2048, 2048],
        }
        detail = (
            f"non-paged causal prefill; q_heads={ctx.tp.q_heads}; "
            f"kv_heads={ctx.tp.kv_heads}; sparse_mode=3; int8 upper-triangular mask"
        )
        if ctx.dry_run:
            return planned(ctx, api, scenario, "batch dim0", "float16", shape, detail)
        query = _rng_tensor("fia_query", "query", (batch, seq, hq), dtype=TORCH.float16)
        key = _rng_tensor("fia_key", "key", (batch, seq, hkv), dtype=TORCH.float16)
        value = _rng_tensor("fia_value", "value", (batch, seq, hkv), dtype=TORCH.float16)
        actual = [seq] * batch
        layout = "BSH"
    else:
        shape = {
            "query": [batch, ctx.tp.q_heads, seq, MODEL.head_dim],
            "key": [batch, ctx.tp.kv_heads, seq, MODEL.head_dim],
            "value": [batch, ctx.tp.kv_heads, seq, MODEL.head_dim],
            "atten_mask": [2048, 2048],
        }
        detail = (
            f"non-paged causal prefill BNSD; q_heads={ctx.tp.q_heads}; "
            f"kv_heads={ctx.tp.kv_heads}; sparse_mode=3; int8 upper-triangular mask"
        )
        if ctx.dry_run:
            return planned(ctx, api, scenario, "batch dim0", "float16", shape, detail)
        query = _rng_tensor("fia_query", "query", (batch, ctx.tp.q_heads, seq, MODEL.head_dim), dtype=TORCH.float16)
        key = _rng_tensor("fia_key", "key", (batch, ctx.tp.kv_heads, seq, MODEL.head_dim), dtype=TORCH.float16)
        value = _rng_tensor("fia_value", "value", (batch, ctx.tp.kv_heads, seq, MODEL.head_dim), dtype=TORCH.float16)
        actual = [seq] * batch
        layout = "BNSD"

    atten_mask = causal_mask_2048()
    full = fia_call(query, key, value, ctx.tp, layout, actual, actual, sparse_mode, atten_mask)
    parts = []
    for slc in shuffled_slices(batch, ctx.rng):
        local = fia_call(
            query[slc].contiguous(),
            key[slc].contiguous(),
            value[slc].contiguous(),
            ctx.tp,
            layout,
            actual[slc],
            actual[slc],
            sparse_mode,
            atten_mask,
        )
        parts.append((slc, local))
    split = restore_outputs_along_dim(full, parts, dim=0)
    return compare_result(ctx, api, scenario, "batch dim0", full, split, TORCH.float16, shape, detail)


def case_matmul_batch_invariant(ctx: CaseContext) -> CaseResult:
    api = "torch.ops.batch_invariant_ops.npu_matmul_batch_invariant"
    rows = sample_token_rows(ctx)
    scenario = ctx.rng.choice(
        [
            "qkv_projection",
            "attention_o_projection",
            "moe_gate_projection",
            "shared_gate_up_projection",
            "shared_down_projection",
            "lm_head_shard_projection",
        ]
    )
    if scenario == "qkv_projection":
        k_dim = MODEL.hidden_size
        n_dim = ctx.tp.qkv_out
        family = f"[N,2048]@[2048,{ctx.tp.qkv_out}]"
    elif scenario == "attention_o_projection":
        k_dim = ctx.tp.o_proj_in
        n_dim = MODEL.hidden_size
        family = f"[N,{ctx.tp.o_proj_in}]@[{ctx.tp.o_proj_in},2048]"
    elif scenario == "moe_gate_projection":
        k_dim = MODEL.hidden_size
        n_dim = MODEL.num_experts
        family = "[N,2048]@[2048,128]"
    elif scenario == "shared_gate_up_projection":
        k_dim = MODEL.hidden_size
        n_dim = ctx.tp.shared_intermediate * 2
        family = f"[N,2048]@[2048,{ctx.tp.shared_intermediate * 2}]"
    elif scenario == "shared_down_projection":
        k_dim = ctx.tp.shared_intermediate
        n_dim = MODEL.hidden_size
        family = f"[N,{ctx.tp.shared_intermediate}]@[{ctx.tp.shared_intermediate},2048]"
    else:
        k_dim = MODEL.hidden_size
        n_dim = min(MODEL.vocab_size, 8192)
        family = f"[N,2048]@[2048,151936] represented by vocab shard [2048,{n_dim}]"
        rows = min(rows, ctx.max_sampler_rows)

    shape = {"self": [rows, k_dim], "other": [k_dim, n_dim], "model_family": family}
    detail = "row-wise matmul; split+shuffle+restore on M dim"
    if ctx.dry_run:
        return planned(ctx, api, scenario, "self dim0 / M", "bfloat16", shape, detail)

    left = _rng_tensor("matmul_self", "self", (rows, k_dim))
    right = _rng_tensor("matmul_other", "other", (k_dim, n_dim))
    full = TORCH.ops.batch_invariant_ops.npu_matmul_batch_invariant(left, right)
    parts = [
        (slc, TORCH.ops.batch_invariant_ops.npu_matmul_batch_invariant(left[slc].contiguous(), right))
        for slc in shuffled_slices(rows, ctx.rng)
    ]
    return compare_result(ctx, api, scenario, "self dim0 / M", full, restore_cat(parts, 0), DTYPE, shape, detail)


def case_reduce_sum_batch_invariant(ctx: CaseContext) -> CaseResult:
    api = "torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant"
    rows = sample_token_rows(ctx)
    keepdim = True if ctx.rng.random() < 0.85 else False
    shape = {
        "x": [rows, MODEL.num_experts],
        "dim": -1,
        "keepdim": keepdim,
        "out": [rows, 1] if keepdim else [rows],
    }
    detail = "MoE softmax denominator path: exp(router_logits), reduce dim=-1"
    if ctx.dry_run:
        return planned(ctx, api, "router_softmax_denominator", "x dim0", "float32", shape, detail)

    logits = _rng_tensor("router_logits", "logits", (rows, MODEL.num_experts), dtype=TORCH.float32)
    x = TORCH.exp(logits)
    full = TORCH.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x, -1, keepdim)
    parts = [
        (slc, TORCH.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x[slc].contiguous(), -1, keepdim))
        for slc in shuffled_slices(rows, ctx.rng)
    ]
    return compare_result(ctx, api, "router_softmax_denominator", "x dim0", full, restore_cat(parts, 0), "float32", shape, detail)


OPERATOR_CASES: dict[str, Callable[[CaseContext], CaseResult]] = {
    "torch_npu.npu_swiglu": case_swiglu,
    "torch_npu.npu_scatter_pa_kv_cache": case_scatter_pa_kv_cache,
    "torch.scatter_": case_scatter,
    "torch_npu.npu_rms_norm": case_rms_norm,
    "torch_npu.npu_moe_init_routing_v2": case_moe_init_routing,
    "torch_npu.npu_moe_token_unpermute": case_moe_token_unpermute,
    "torch_npu.npu_grouped_matmul": case_grouped_matmul,
    "torch_npu.npu_moe_gating_top_k": case_moe_gating_top_k,
    "torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant": case_fia,
    "torch.ops.batch_invariant_ops.npu_matmul_batch_invariant": case_matmul_batch_invariant,
    "torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant": case_reduce_sum_batch_invariant,
}


def run_one_case(item: PlanItem, case_id: int, args: argparse.Namespace) -> CaseResult:
    seed = args.seed ^ (case_id * 0x9E3779B1) ^ stable_operator_seed(item.operator)
    ctx = CaseContext(
        case_id=case_id,
        operator=item.operator,
        tier=item.tier,
        tp=tp_profile(item.tp_size),
        rng=random.Random(seed),
        dry_run=args.dry_run,
        max_token_rows=args.max_token_rows,
        max_expert_rows=args.max_expert_rows,
        max_fia_batch=args.max_fia_batch,
        max_fia_seq=args.max_fia_seq,
        max_sampler_rows=args.max_sampler_rows,
        max_sample_size=args.max_sample_size,
        fia_chunked_prefill_tokens=args.fia_chunked_prefill_tokens,
        fia_chunked_prefill_chunk_size=args.fia_chunked_prefill_chunk_size,
        fia_chunked_prefill_data_pattern=args.fia_chunked_prefill_data_pattern,
    )
    try:
        return OPERATOR_CASES[item.operator](ctx)
    except Exception as exc:
        return failure_result(ctx, exc)


def summarize(results: list[CaseResult], plan: list[PlanItem]) -> dict[str, Any]:
    def count_by(items: list[Any], key: Callable[[Any], str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            value = key(item)
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    operator_tp_counts: dict[str, dict[str, int]] = {}
    operator_tier_counts: dict[str, dict[str, int]] = {}
    scenario_counts: dict[str, int] = {}
    for res in results:
        operator_tp_counts.setdefault(res.operator, {})
        operator_tp_counts[res.operator][str(res.tp_size)] = operator_tp_counts[res.operator].get(str(res.tp_size), 0) + 1
        operator_tier_counts.setdefault(res.operator, {})
        operator_tier_counts[res.operator][res.tier] = operator_tier_counts[res.operator].get(res.tier, 0) + 1
        key = f"{res.operator} | {res.scenario}"
        scenario_counts[key] = scenario_counts.get(key, 0) + 1

    plan_counts = count_by(plan, lambda item: item.operator)
    return {
        "case_count": len(results),
        "plan_count": len(plan),
        "summary": count_by(results, lambda res: res.conclusion),
        "operator_counts": count_by(results, lambda res: res.operator),
        "operator_tp_counts": operator_tp_counts,
        "operator_tier_counts": operator_tier_counts,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "plan_counts": plan_counts,
    }


def validate_dry_run(summary: dict[str, Any], selected_operators: tuple[str, ...]) -> list[str]:
    expected_tp_values = summary.get("expected_tp_values", TP_VALUES)
    errors = []
    if summary["case_count"] != summary["plan_count"]:
        errors.append(f"case_count {summary['case_count']} != plan_count {summary['plan_count']}")
    for operator in selected_operators:
        if operator not in summary["operator_counts"]:
            errors.append(f"missing operator {operator}")
            continue
        tp_counts = summary["operator_tp_counts"].get(operator, {})
        for tp_size in expected_tp_values:
            if str(tp_size) not in tp_counts:
                errors.append(f"operator {operator} missing tp={tp_size}")
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-cases", type=int, default=DEFAULT_TOTAL_CASES)
    parser.add_argument(
        "--cases-per-operator",
        type=int,
        default=None,
        help="Generate this many cases for each selected operator. Overrides --total-cases.",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--tp-size",
        type=int,
        choices=TP_VALUES,
        default=None,
        help="Force all generated cases to one TP size, e.g. --tp-size 1.",
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--output", default="/workspace/batch_invariance_10000_results.json")
    parser.add_argument("--dry-run", action="store_true", help="generate and validate case metadata only")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the generated plan")
    parser.add_argument("--operators", nargs="+", choices=TARGET_OPERATORS, default=None)
    parser.add_argument(
        "--fia-scenarios",
        nargs="+",
        choices=FIA_SCENARIOS,
        default=None,
        help="Restrict FIA scenarios. Useful for forcing tnd_prefill_token_batch or tnd_chunked_prefill smoke tests.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--empty-cache-interval", type=int, default=25)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--max-split-parts",
        type=int,
        default=16,
        help="Max random batch-axis split count. Use 0 to allow up to length.",
    )
    parser.add_argument(
        "--force-singleton-split",
        action="store_true",
        help="Force every batch-axis element to be tested as its own slice.",
    )

    parser.add_argument("--max-token-rows", type=int, default=2048)
    parser.add_argument("--max-expert-rows", type=int, default=8192)
    parser.add_argument("--max-fia-batch", type=int, default=4)
    parser.add_argument("--max-fia-seq", type=int, default=512)
    parser.add_argument(
        "--fia-chunked-prefill-tokens",
        type=int,
        default=None,
        help="Force P length for tnd_chunked_prefill. Overrides --max-fia-seq for that scenario.",
    )
    parser.add_argument(
        "--fia-chunked-prefill-chunk-size",
        type=int,
        default=None,
        help="Max staged chunk size for tnd_chunked_prefill, e.g. 8192. Actual chunks are randomized.",
    )
    parser.add_argument(
        "--fia-chunked-prefill-data-pattern",
        choices=FIA_CHUNKED_PREFILL_DATA_PATTERNS,
        default="large_small",
        help="Input data pattern for tnd_chunked_prefill. large_small stresses cancellation and tiny terms.",
    )
    parser.add_argument("--max-sampler-rows", type=int, default=8)
    parser.add_argument("--max-sample-size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    global SPLIT_MAX_PARTS, SPLIT_FORCE_SINGLETONS, FIA_SCENARIOS

    args = parse_args(argv)
    selected_operators = tuple(args.operators) if args.operators else TARGET_OPERATORS
    if args.cases_per_operator is not None and args.cases_per_operator <= 0:
        print("ERROR cases-per-operator must be > 0", file=sys.stderr)
        return 2
    total_cases = (
        args.cases_per_operator * len(selected_operators)
        if args.cases_per_operator is not None
        else args.total_cases
    )
    if total_cases < len(selected_operators):
        print(
            f"ERROR total-cases must be >= selected operator count ({len(selected_operators)})",
            file=sys.stderr,
        )
        return 2
    if args.max_split_parts < 0:
        print("ERROR max-split-parts must be >= 0", file=sys.stderr)
        return 2
    if args.fia_chunked_prefill_tokens is not None and args.fia_chunked_prefill_tokens <= 0:
        print("ERROR fia-chunked-prefill-tokens must be > 0", file=sys.stderr)
        return 2
    if args.fia_chunked_prefill_chunk_size is not None and args.fia_chunked_prefill_chunk_size <= 0:
        print("ERROR fia-chunked-prefill-chunk-size must be > 0", file=sys.stderr)
        return 2
    SPLIT_MAX_PARTS = None if args.max_split_parts == 0 else args.max_split_parts
    SPLIT_FORCE_SINGLETONS = bool(args.force_singleton_split)
    if args.fia_scenarios is not None:
        FIA_SCENARIOS = tuple(args.fia_scenarios)

    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("HCCL_DETERMINISTIC", "strict")

    if not args.dry_run:
        try:
            load_runtime(args.device, args.dtype, args.seed)
        except Exception as exc:
            print(f"ERROR runtime preflight failed: {exc}", file=sys.stderr)
            return 2

    plan = force_plan_tp(make_case_plan(total_cases, selected_operators, args.seed, args.shuffle), args.tp_size)
    results: list[CaseResult] = []
    started = time.time()
    for case_id, item in enumerate(plan):
        res = run_one_case(item, case_id, args)
        results.append(res)
        if args.verbose:
            print(json.dumps(asdict(res), ensure_ascii=False), flush=True)
        if args.stop_on_failure and res.conclusion in {"FAIL", "CALL_FAILED"}:
            print(f"STOP case={case_id} conclusion={res.conclusion}", flush=True)
            break
        if not args.dry_run and (case_id + 1) % args.empty_cache_interval == 0:
            sync()
            TORCH.npu.empty_cache()
        if args.progress_interval > 0 and (case_id + 1) % args.progress_interval == 0:
            summary = summarize(results, plan[: case_id + 1])["summary"]
            print(f"PROGRESS {case_id + 1}/{len(plan)} {json.dumps(summary, sort_keys=True)}", flush=True)

    summary = summarize(results, plan[: len(results)])
    summary["expected_tp_values"] = [args.tp_size] if args.tp_size is not None else list(TP_VALUES)
    dry_run_errors = validate_dry_run(summary, selected_operators) if args.dry_run else []
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "dry_run" if args.dry_run else "execute",
        "seed": args.seed,
        "device": None if args.dry_run else args.device,
        "dtype": args.dtype,
        "total_cases_requested": total_cases,
        "cases_per_operator_requested": args.cases_per_operator,
        "env": {
            "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
            "HCCL_DETERMINISTIC": os.environ.get("HCCL_DETERMINISTIC"),
        },
        "model_config": asdict(MODEL),
        "tp_profiles": {str(tp): asdict(tp_profile(tp)) for tp in TP_VALUES},
        "limits": {
            "max_token_rows": args.max_token_rows,
            "max_expert_rows": args.max_expert_rows,
            "max_fia_batch": args.max_fia_batch,
            "max_fia_seq": args.max_fia_seq,
            "max_sampler_rows": args.max_sampler_rows,
            "max_sample_size": args.max_sample_size,
            "max_split_parts": "length" if SPLIT_MAX_PARTS is None else SPLIT_MAX_PARTS,
            "force_singleton_split": SPLIT_FORCE_SINGLETONS,
            "fia_scenarios": list(FIA_SCENARIOS),
            "tp_size": args.tp_size,
        },
        "dry_run_validation_errors": dry_run_errors,
        **summary,
        "results": [asdict(res) for res in results],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("SUMMARY " + json.dumps(summary["summary"], ensure_ascii=False, sort_keys=True), flush=True)
    print("OPERATOR_COUNTS " + json.dumps(summary["operator_counts"], ensure_ascii=False, sort_keys=True), flush=True)
    print("RESULT_FILE " + str(output), flush=True)
    print(f"ELAPSED_SECONDS {time.time() - started:.2f}", flush=True)

    if dry_run_errors:
        print("DRY_RUN_VALIDATION_ERRORS " + json.dumps(dry_run_errors, ensure_ascii=False), flush=True)
        return 1
    if not args.dry_run:
        failed = summary["summary"].get("FAIL", 0) + summary["summary"].get("CALL_FAILED", 0)
        return 0 if failed == 0 and summary["summary"].get("PASS", 0) == len(results) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
