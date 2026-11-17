#!/usr/bin/env python3
"""
vLLM Memory Calculator & Parallel Strategy Planner for Ascend NPU.

Reads a HuggingFace config.json and computes:
  - Weight classification (Expert / Replicated / TP-Sharded / Vision)
  - Per-card weight memory under quantization×parallelism matrix
  - KV cache memory (MLA-aware: not TP-sharded)
  - NPU overhead breakdown (driver + HCCS + activation + fragmentation)
  - EP divisibility constraint check
  - Parallel strategy search (TP × DP × EP × PP) with ranking
  - Physical machine mapping (16-card Ascend 910B nodes)

Supports: Dense, MoE (standard + LatentMoE), MLA, GQA, MHA,
         MSA/DSA sparse attention, MTP, Multimodal vision tower,
         PP pipeline parallelism, FP8/W8A8/W4A8 quantization.

Usage:
  python vllm_memory_calculator.py <config.json> [options]

Options:
  --batch N        Batch size (default: 8)
  --seq N          Sequence length (default: 4096)
  --dtype TYPE     Data type: bf16, fp16, fp8, w8a8, w4a8 (default: bf16)
  --npu-memory N   Single NPU card memory in GB (default: 64)
  --max-devices N  Maximum number of cards (default: 16)
  --tp-range N,N,N TP candidate values (default: 1,2,4,8,16)
  --pp-range N,N   PP candidate values (default: 1,2)
  --strategy       Search for optimal parallel strategy
  --json           Output as JSON
  --kvcache-json F Load pre-computed kvcache_calculator.py JSON output
  --verbose        Show detailed breakdown
"""

import json
import sys
import os
import argparse
import math

# ─── Constants ───
DTYPE_BYTES = {
    "bf16": 2, "fp16": 2, "fp8": 1, "fp8_e4m3": 1,
    "w8a8": 1, "w4a8": 0.5, "int8": 1, "int4": 0.5,
}

QUANT_BYTES = {
    "bf16": 2, "fp16": 2,
    "w8a8": 1.0,
    "w4a8": 0.5,
}

# Standard EP and quant sweep for matrix (from ascend-model-analysis 3.4.3)
DEFAULT_EP_VALUES = [8, 16, 32, 64]
DEFAULT_QUANT_VALUES = ["w8a8", "w4a8"]

# NPU overhead defaults (Ascend 910B)
NPU_DRIVER_GIB = 3.0       # Driver + runtime
NPU_HCCS_GIB = 4.5         # HCCS communication buffers
NPU_FRAG_GIB = 2.0         # PyTorch allocator fragmentation
NPU_MULTISTREAM_GIB = 1.0  # Multi-stream overlap overhead
NPU_SAFE_MARGIN_GIB = 3.0  # Safety margin


# ═══════════════════════════════════════════════════════════════
# Config Loading & Parameter Extraction
# ═══════════════════════════════════════════════════════════════

def load_config(path):
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.exists(path):
        print(f"ERROR: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(cfg, *keys):
    for k in keys:
        if k in cfg:
            return cfg[k]
    return None


def extract_params(cfg):
    """Extract normalized parameters from config.json."""
    p = {}

    p["hidden_size"] = norm(cfg, "hidden_size", "dim")
    p["num_layers"] = norm(cfg, "num_hidden_layers", "n_layers")
    p["num_heads"] = norm(cfg, "num_attention_heads", "n_heads")
    p["num_kv_heads"] = norm(cfg, "num_key_value_heads", "n_kv_heads", "num_attention_heads")
    p["head_dim"] = norm(cfg, "head_dim")
    if p["head_dim"] is None and p["hidden_size"] and p["num_heads"]:
        p["head_dim"] = p["hidden_size"] // p["num_heads"]

    # MLA
    p["kv_lora_rank"] = norm(cfg, "kv_lora_rank")
    p["q_lora_rank"] = norm(cfg, "q_lora_rank")
    p["qk_head_dim"] = norm(cfg, "qk_head_dim", "qk_nope_head_dim")
    p["qk_rope_head_dim"] = norm(cfg, "qk_rope_head_dim", 0)
    p["v_head_dim"] = norm(cfg, "v_head_dim") or p.get("qk_head_dim") or 0

    # MoE
    p["num_experts"] = norm(cfg, "num_local_experts", "n_routed_experts", "num_experts", 1)
    p["top_k"] = norm(cfg, "num_experts_per_tok", "n_activated_experts", "num_experts_per_token", 1)
    p["moe_intermediate_size"] = norm(cfg, "moe_intermediate_size") or norm(cfg, "intermediate_size")
    p["num_shared_experts"] = norm(cfg, "n_shared_experts", "num_shared_experts", 0)
    p["shared_intermediate_size"] = norm(cfg, "shared_expert_intermediate_size", "shared_intermediate_size") or norm(cfg, "intermediate_size")
    p["intermediate_size"] = norm(cfg, "intermediate_size")
    p["dense_intermediate_size"] = norm(cfg, "dense_intermediate_size", "intermediate_size")
    p["routed_expert_hidden_size"] = norm(cfg, "routed_expert_hidden_size")

    # Vocab
    p["vocab_size"] = norm(cfg, "vocab_size")
    p["tie_word_embeddings"] = cfg.get("tie_word_embeddings", False)

    # Context
    p["max_position_embeddings"] = norm(cfg, "max_position_embeddings", "max_seq_len", 4096)

    # MTP
    p["num_mtp"] = norm(cfg, "num_nextn_predict_layers", "num_mtp_modules", 0)

    # Layer frequencies
    p["moe_layer_freq"] = cfg.get("moe_layer_freq", [])
    p["dense_layers"] = cfg.get("dense_layers", cfg.get("dense_layer_indices", []))

    # Compute MoE/dense layer counts
    if p["moe_layer_freq"] and isinstance(p["moe_layer_freq"], (list, tuple)) and len(p["moe_layer_freq"]) == p["num_layers"]:
        p["num_moe_layers"] = sum(1 for x in p["moe_layer_freq"] if x > 0)
        p["num_dense_layers"] = p["num_layers"] - p["num_moe_layers"]
    elif p["num_experts"] and p["num_experts"] > 1:
        if p["dense_layers"]:
            p["num_dense_layers"] = len(p["dense_layers"])
        else:
            p["num_dense_layers"] = p.get("first_k_dense_replace", 1)
        p["num_moe_layers"] = p["num_layers"] - p["num_dense_layers"]
    else:
        p["num_moe_layers"] = 0
        p["num_dense_layers"] = p["num_layers"]

    # Has MLA?
    p["has_mla"] = bool(p.get("kv_lora_rank"))

    # Vision
    vision = cfg.get("vision_config", {})
    p["has_vision"] = bool(vision)
    p["vision_hidden_size"] = vision.get("hidden_size")
    p["vision_num_layers"] = vision.get("num_hidden_layers", 0)

    # Sparse attention
    sparse_config = cfg.get("sparse_attention_config", {})
    p["sparse_index_dim"] = sparse_config.get("sparse_index_dim") or sparse_config.get("index_dim")

    # Quantization
    qt = cfg.get("quantization_config", {})
    p["quant_method"] = qt.get("quant_method", "")

    return p


# ═══════════════════════════════════════════════════════════════
# Weight Computation
# ═══════════════════════════════════════════════════════════════

def weight_embedding(p):
    return p["vocab_size"] * p["hidden_size"]


def weight_lm_head(p):
    if p["tie_word_embeddings"]:
        return 0
    return p["vocab_size"] * p["hidden_size"]


def weight_attention_per_layer(p):
    """Attention projections per layer (non-MoE part)."""
    h = p["hidden_size"]
    nq = p["num_heads"]
    nkv = p["num_kv_heads"] or nq
    d = p["head_dim"]

    if p["has_mla"]:
        total = 0
        q_lora = p.get("q_lora_rank") or p.get("kv_lora_rank") or 0
        kv_lora = p.get("kv_lora_rank") or 0
        qk_dim = p.get("qk_head_dim") or 0
        rope_dim = p.get("qk_rope_head_dim") or 0
        v_dim = p.get("v_head_dim") or qk_dim

        if q_lora:
            total += h * q_lora  # q_a_proj
            total += q_lora * nq * (qk_dim + rope_dim)  # q_b_proj
        if kv_lora:
            total += h * (kv_lora + rope_dim)  # kv_a_proj
            total += kv_lora * nq * (qk_dim + v_dim)  # kv_b_proj
        if v_dim:
            total += nq * v_dim * h  # o_proj
        return total
    else:
        # Full GQA/MHA
        q = h * nq * d
        k = h * nkv * d
        v = h * nkv * d
        o = nq * d * h
        return q + k + v + o


def weight_dense_mlp_per_layer(p):
    h = p["hidden_size"]
    inter = p["dense_intermediate_size"] or p["intermediate_size"] or 0
    return 3 * h * inter


def weight_moe_expert(p):
    """Single MoE expert weight."""
    h = p.get("routed_expert_hidden_size") or p["hidden_size"]
    inter = p["moe_intermediate_size"] or p["intermediate_size"] or 0
    return 3 * h * inter  # w13 (gate+up=2×h×inter) + w2 (inter×h)


def weight_moe_gate_per_layer(p):
    return p["hidden_size"] * p["num_experts"]


def weight_shared_expert(p):
    h = p["hidden_size"]
    inter = p["shared_intermediate_size"] or p["intermediate_size"] or 0
    return 3 * h * inter


def weight_norms(p):
    """All RMSNorm/LayerNorm weights."""
    h = p["hidden_size"]
    return h * (p["num_layers"] * 2 + 1)  # 2 per layer + final


def weight_vision_tower(p):
    """Estimate vision tower params."""
    if not p.get("has_vision") or not p.get("vision_num_layers"):
        return 0
    v_hidden = p["vision_hidden_size"] or p["hidden_size"]
    v_layers = p["vision_num_layers"]
    v_inter = v_hidden * 4
    # Rough: attention + MLP + norms per layer
    return v_layers * (4 * v_hidden * v_hidden + 3 * v_hidden * v_inter + 2 * v_hidden)


def weight_mtp(p):
    """MTP speculative layers."""
    n_mtp = p.get("num_mtp", 0)
    if n_mtp == 0:
        return 0
    attn = weight_attention_per_layer(p)
    mlp = weight_dense_mlp_per_layer(p)
    return (attn + mlp + p["hidden_size"] * 2) * n_mtp


# ═══════════════════════════════════════════════════════════════
# Weight Classification (v2.0 core)
# ═══════════════════════════════════════════════════════════════

def classify_weights(p):
    """
    Classify all weights into:
      - expert:        MoE routed experts (EP-sharded)
      - replicated:    LatentMoE projections, moe_gate, fused_qkv_a (NOT TP-sharded)
      - tp_sharded:    Attention QKV/Out, shared MLP, embedding, lm_head (TP-sharded)
      - vision:        Vision tower (replicated in data mode)
      - bf16_fixed:    Norms (tiny, always BF16)
    """
    classifications = {
        "expert": {"params": 0, "components": [], "split": "EP"},
        "replicated": {"params": 0, "components": [], "split": "replicated"},
        "tp_sharded": {"params": 0, "components": [], "split": "TP"},
        "vision": {"params": 0, "components": [], "split": "replicated"},
        "bf16_fixed": {"params": 0, "components": [], "split": "replicated"},
    }

    has_mla = p["has_mla"]
    n_layers = p["num_layers"]
    n_dense = p["num_dense_layers"]
    n_moe = p["num_moe_layers"]
    n_experts = p["num_experts"]
    n_shared = p["num_shared_experts"]

    # ── BF16 Fixed (norms — tiny, never quantized) ──
    norm_params = weight_norms(p)
    classifications["bf16_fixed"]["params"] += norm_params
    classifications["bf16_fixed"]["components"].append(f"RMSNorm/LayerNorm ({n_layers*2+1}个)")

    # ── Embedding (TP-sharded along vocab dim) ──
    emb = weight_embedding(p)
    classifications["tp_sharded"]["params"] += emb
    classifications["tp_sharded"]["components"].append("Embedding (vocab TP)")

    # ── LM Head ──
    lmh = weight_lm_head(p)
    if lmh > 0:
        classifications["tp_sharded"]["params"] += lmh
        classifications["tp_sharded"]["components"].append("LM Head (vocab TP)")
    else:
        classifications["tp_sharded"]["components"].append("LM Head (tied with embedding)")

    # ── Attention ──
    attn_per_layer = weight_attention_per_layer(p)
    if has_mla:
        # MLA: fused_qkv_a_proj is ReplicatedLinear! (LatentMoE-style)
        # Only O projection is TP-sharded
        h = p["hidden_size"]
        q_lora = p.get("q_lora_rank") or p.get("kv_lora_rank") or 0
        kv_lora = p.get("kv_lora_rank") or 0
        rope_dim = p.get("qk_rope_head_dim") or 0
        qk_dim = p.get("qk_head_dim") or 0
        v_dim = p.get("v_head_dim") or qk_dim
        n_heads = p["num_heads"]

        # Replicated: q_a_proj + q_b_proj + kv_a_proj + kv_b_proj
        rep_per_layer = 0
        if q_lora:
            rep_per_layer += h * q_lora  # q_a_proj
            rep_per_layer += q_lora * n_heads * (qk_dim + rope_dim)  # q_b_proj
        if kv_lora:
            rep_per_layer += h * (kv_lora + rope_dim)  # kv_a_proj
            rep_per_layer += kv_lora * n_heads * (qk_dim + v_dim)  # kv_b_proj

        classifications["replicated"]["params"] += rep_per_layer * n_layers
        classifications["replicated"]["components"].append(
            f"MLA fused_qkv_a/b_proj ({n_layers}层, ReplicatedLinear)"
        )

        # TP-sharded: O projection only
        if v_dim:
            tp_per_layer = n_heads * v_dim * h
            classifications["tp_sharded"]["params"] += tp_per_layer * n_layers
            classifications["tp_sharded"]["components"].append(
                f"MLA O_proj ({n_layers}层, TP-sharded)"
            )
    else:
        # Full Attention: all TP-sharded
        classifications["tp_sharded"]["params"] += attn_per_layer * n_layers
        classifications["tp_sharded"]["components"].append(
            f"{'GQA' if p['num_kv_heads'] < p['num_heads'] else 'MHA'} Q/K/V/O ({n_layers}层, TP-sharded)"
        )

    # ── Dense MLP (TP-sharded) ──
    dense_mlp = weight_dense_mlp_per_layer(p)
    if n_dense > 0:
        classifications["tp_sharded"]["params"] += dense_mlp * n_dense
        classifications["tp_sharded"]["components"].append(
            f"Dense SwiGLU MLP ({n_dense}层, TP-sharded)"
        )

    # ── MoE (per-layer experts) ──
    if n_experts > 1 and n_moe > 0:
        expert_size = weight_moe_expert(p)
        total_expert = expert_size * n_experts * n_moe
        classifications["expert"]["params"] += total_expert
        classifications["expert"]["components"].append(
            f"MoE routed experts ({n_experts}x{expert_size/1e6:.1f}M x {n_moe}层, EP-sharded)"
        )

        # MoE gate (Replicated, per-layer)
        gate = weight_moe_gate_per_layer(p)
        classifications["replicated"]["params"] += gate * n_moe
        classifications["replicated"]["components"].append(
            f"MoE router gate ({n_moe}层, Replicated)"
        )

        # Shared experts (TP-sharded)
        if n_shared > 0:
            shared_exp = weight_shared_expert(p)
            classifications["tp_sharded"]["params"] += shared_exp * n_shared * n_moe
            classifications["tp_sharded"]["components"].append(
                f"Shared experts ({n_shared}×{n_moe}层, TP-sharded)"
            )

        # LatentMoE: routed_expert_hidden_size projections (Replicated)
        if p.get("routed_expert_hidden_size") and p["routed_expert_hidden_size"] != p["hidden_size"]:
            # The down_proj and up_proj between hidden<->expert_hidden are ReplicatedLinear
            rh = p["routed_expert_hidden_size"]
            h = p["hidden_size"]
            rep_latent = (h * rh * 2 + rh * h) * n_moe  # up_proj + gate_proj + down_proj
            classifications["replicated"]["params"] += rep_latent
            classifications["replicated"]["components"].append(
                f"LatentMoE routed_expert projections ({n_moe}层, ReplicatedLinear)"
            )

    # ── Vision Tower (replicated in data mode) ──
    vis = weight_vision_tower(p)
    if vis > 0:
        classifications["vision"]["params"] += vis
        classifications["vision"]["components"].append("Vision Tower (data-parallel replicated)")

    # ── MTP (TP-sharded + norms) ──
    mtp = weight_mtp(p)
    if mtp > 0:
        classifications["tp_sharded"]["params"] += mtp
        classifications["tp_sharded"]["components"].append(
            f"MTP speculative layers ({p['num_mtp']}层, TP-sharded)"
        )

    return classifications


# ═══════════════════════════════════════════════════════════════
# KV Cache (for vLLM memory)
# ═══════════════════════════════════════════════════════════════

def compute_kv_per_token(p, dtype_bytes):
    """Compute per-token KV cache bytes for the whole model."""
    n_layers = p["num_layers"]
    n_kv = p["num_kv_heads"] or p["num_heads"]
    d = p["head_dim"]

    if p["has_mla"]:
        # MLA: NOT TP-sharded! Each rank holds full KV cache
        kv_lora = p.get("kv_lora_rank", 0)
        rope_dim = p.get("qk_rope_head_dim", 0)
        per_layer = (kv_lora + rope_dim) * dtype_bytes
    else:
        # GQA/MHA
        per_layer = n_kv * d * 2 * dtype_bytes

    return per_layer * n_layers


def total_kv_cache(p, batch_size, seq_len, dtype_bytes, tp=1):
    """Total KV cache. For MLA, TP doesn't reduce KV."""
    per_token = compute_kv_per_token(p, dtype_bytes)
    if p["has_mla"]:
        # MLA KV cache is NOT reduced by TP
        return per_token * batch_size * seq_len
    else:
        # GQA KV cache IS reduced by TP (heads split)
        return per_token * batch_size * seq_len / tp


# ═══════════════════════════════════════════════════════════════
# NPU Overhead Model
# ═══════════════════════════════════════════════════════════════

def compute_npu_overhead(ep, tp, is_moe, has_mla):
    """Compute NPU overhead breakdown in GiB."""
    # Driver + runtime (fixed)
    driver = NPU_DRIVER_GIB

    # HCCS buffers scale with communication volume
    # EP all-to-all dominates for MoE, TP all-reduce for dense
    if is_moe:
        hccs = NPU_HCCS_GIB + (ep / 32) * 1.5  # Larger EP → more all-to-all buffers
    else:
        hccs = NPU_HCCS_GIB * 0.5

    # Activation peak (MoE dispatch: top_k × expansion for prefill)
    activation = 2.0
    if is_moe:
        activation += 2.0  # MoE dispatch buffer

    # Fragmentation
    frag = NPU_FRAG_GIB

    total = driver + hccs + activation + frag
    if ep > 128 or tp > 16:
        total += NPU_MULTISTREAM_GIB

    return {
        "driver": driver,
        "hccs": hccs,
        "activation": activation,
        "fragmentation": frag,
        "total": total,
    }


# ═══════════════════════════════════════════════════════════════
# Quant×Parallel Memory Matrix (from ascend-model-analysis §3.4)
# ═══════════════════════════════════════════════════════════════

def compute_memory_matrix(p, classifications, dtype_bytes, batch_size, seq_len, tp):
    """
    Compute per-card weight memory for all EP×quant combinations.
    Implements the ascend-model-analysis 3.4.3 matrix.
    """
    n_experts = p["num_experts"]
    is_moe = n_experts > 1

    # ── K (BF16, always replicated) ──
    k_bf16_params = (
        classifications["bf16_fixed"]["params"] +
        classifications["vision"]["params"]
    )
    k_bf16_gb = k_bf16_params * 2 / 1e9  # BF16 = 2 bytes

    # ── Q_rep (quantized, replicated, NOT TP-sharded) ──
    q_rep_params = (
        classifications["replicated"]["params"]
    )

    # ── Q_experts (quantized, EP-sharded) ──
    q_experts_params = classifications["expert"]["params"]

    # ── Q_tp (quantized, TP-sharded) ──
    q_tp_params = classifications["tp_sharded"]["params"]

    results = []
    ep_values = DEFAULT_EP_VALUES
    quant_values = DEFAULT_QUANT_VALUES

    for ep_val in ep_values:
        # EP divisibility check
        ep_ok = n_experts % ep_val == 0
        for quant in quant_values:
            qb = QUANT_BYTES[quant]

            # Q_rep: replicated → per card = full
            q_rep_gb = q_rep_params * qb / 1e9

            # Q_tp: TP-sharded → per card = full / TP
            q_tp_gb = q_tp_params * qb / 1e9 / tp

            # Q_experts: EP-sharded → per card = full / EP
            if ep_ok:
                q_exp_gb = q_experts_params * qb / 1e9 / ep_val
            else:
                q_exp_gb = None  # EP not divisible

            # Total weight per card
            if ep_ok:
                total_gb = k_bf16_gb + q_rep_gb + q_tp_gb + q_exp_gb
            else:
                total_gb = None

            results.append({
                "ep": ep_val,
                "tp": tp,
                "quant": quant,
                "ep_ok": ep_ok,
                "K_bf16_gb": round(k_bf16_gb, 2),
                "Q_rep_gb": round(q_rep_gb, 2),
                "Q_tp_gb": round(q_tp_gb, 2),
                "Q_experts_gb": round(q_exp_gb, 2) if q_exp_gb is not None else None,
                "weight_total_gb": round(total_gb, 2) if total_gb is not None else None,
                "pct_64gb": round(total_gb / 64 * 100, 1) if total_gb is not None else None,
                "pct_96gb": round(total_gb / 96 * 100, 1) if total_gb is not None else None,
            })

    return results, {
        "K_bf16_params": k_bf16_params,
        "K_bf16_gb": k_bf16_gb,
        "Q_rep_params": q_rep_params,
        "Q_tp_params": q_tp_params,
        "Q_experts_params": q_experts_params,
    }


# ═══════════════════════════════════════════════════════════════
# Parallel Strategy Search
# ═══════════════════════════════════════════════════════════════

def factorize(n):
    """Return all factors of n."""
    factors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)


def search_strategies(p, classifications, dtype_bytes, args):
    """
    Search for feasible (TP, EP, DP, PP) configurations.
    Returns ranked list of strategies.
    """
    n_experts = p["num_experts"]
    npu_memory_gib = args.npu_memory - NPU_SAFE_MARGIN_GIB  # usable memory
    max_devices = args.max_devices
    batch_size = args.batch
    seq_len = args.seq
    tp_candidates = args.tp_range
    pp_candidates = args.pp_range
    is_moe = n_experts > 1

    strategies = []

    for pp in pp_candidates:
        for tp in tp_candidates:
            # TP constraint: ≤ 16 (single node HCCS)
            if tp > 16:
                continue

            # Compute TP-sharded per-card weight for this TP
            q_tp_params = classifications["tp_sharded"]["params"]
            if is_moe:
                q_tp_gb_bf16 = q_tp_params * 2 / 1e9 / tp
            else:
                q_tp_gb_bf16 = q_tp_params * 2 / 1e9 / tp

            # EP candidates must divide num_experts
            if is_moe:
                ep_candidates = [f for f in factorize(n_experts) if f <= max_devices]
            else:
                ep_candidates = [1]

            for ep in ep_candidates:
                dp = ep // tp
                if dp == 0 or ep % tp != 0:
                    continue

                total_cards = pp * tp * dp
                if total_cards > max_devices:
                    continue

                # Per-card weight
                _, matrix_info = compute_memory_matrix(
                    p, classifications, dtype_bytes, batch_size, seq_len, tp
                )

                # Use BF16 as baseline (worst case)
                total_weight_gb = matrix_info["K_bf16_gb"]

                # Q_rep (replicated)
                q_rep = classifications["replicated"]["params"]
                total_weight_gb += q_rep * 2 / 1e9

                # Q_tp
                total_weight_gb += q_tp_gb_bf16

                # Q_experts
                if is_moe:
                    q_exp = classifications["expert"]["params"]
                    total_weight_gb += q_exp * 2 / 1e9 / ep

                # Apply PP scaling
                if pp > 1:
                    # PP splits layers across stages
                    total_weight_gb = total_weight_gb / pp
                    # But vision tower only on PP stage 0
                    if p["has_vision"]:
                        vis_gb = classifications["vision"]["params"] * 2 / 1e9
                        total_weight_gb += vis_gb * (1 - 1.0 / pp)

                # KV Cache
                kv_gb = total_kv_cache(p, batch_size, seq_len, dtype_bytes, tp) / 1e9
                if pp > 1:
                    kv_gb = kv_gb / pp

                # NPU overhead
                overhead = compute_npu_overhead(ep, tp, is_moe, p["has_mla"])
                total_overhead = overhead["total"]

                total_gb = total_weight_gb + kv_gb + total_overhead
                fits = total_gb < npu_memory_gib

                machines = total_cards // 16 + (1 if total_cards % 16 else 0)

                strategies.append({
                    "ep": ep, "tp": tp, "dp": dp, "pp": pp,
                    "total_cards": total_cards,
                    "machines": machines,
                    "weight_gb": round(total_weight_gb, 2),
                    "kv_gb": round(kv_gb, 2),
                    "overhead_gb": round(total_overhead, 2),
                    "total_gb": round(total_gb, 2),
                    "npu_limit_gib": round(npu_memory_gib, 2),
                    "fits": fits,
                })

    # Sort: fits first, then by total_cards (fewer is better)
    strategies.sort(key=lambda s: (not s["fits"], s["total_cards"], s["total_gb"]))
    return strategies


# ═══════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════

def format_params(n):
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    elif n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    else:
        return f"{n}"


def format_gb(n):
    if n is None:
        return "N/A"
    return f"{n:.2f} GB"


def print_results(p, classifications, matrix, matrix_info, strategies, args):
    """Print human-readable report."""
    dtype_bytes = DTYPE_BYTES[args.dtype]
    is_moe = p["num_experts"] > 1

    print("=" * 80)
    print("  vLLM Memory Calculator & Parallel Strategy Planner")
    print("  for Ascend NPU (CANN Ecosystem)")
    print("=" * 80)

    # ── Model Summary ──
    print(f"\n{'─'*60}")
    print(f"  Model Summary")
    print(f"{'─'*60}")
    print(f"  hidden_size:          {p['hidden_size']}")
    print(f"  num_layers:           {p['num_layers']} (dense={p['num_dense_layers']}, moe={p['num_moe_layers']})")
    print(f"  attention:            {'MLA' if p['has_mla'] else 'GQA' if p['num_kv_heads'] < p['num_heads'] else 'MHA'}")
    if p["has_mla"]:
        print(f"    kv_lora_rank:       {p['kv_lora_rank']}")
        print(f"    qk_rope_head_dim:   {p['qk_rope_head_dim']}")
    if is_moe:
        print(f"  num_experts:          {p['num_experts']} (top_k={p['top_k']})")
        print(f"  moe_intermediate:     {p['moe_intermediate_size']}")
        print(f"  num_shared_experts:   {p['num_shared_experts']}")
    print(f"  vocab_size:           {p['vocab_size']}")
    print(f"  max_position_embeds:  {p['max_position_embeddings']}")
    print(f"  tie_word_embeddings:  {p['tie_word_embeddings']}")

    # ── Weight Classification ──
    print(f"\n{'─'*60}")
    print(f"  Weight Classification (BF16 baseline)")
    print(f"{'─'*60}")
    total_params = 0
    for cat, info in classifications.items():
        if info["params"] > 0:
            gb = info["params"] * 2 / 1e9
            total_params += info["params"]
            print(f"  {cat:<20} {format_params(info['params']):>10}  ({info['split']:<12})  {gb:>8.2f} GB")
            for comp in info["components"]:
                print(f"    └ {comp}")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':<20} {format_params(total_params):>10}          {total_params*2/1e9:>8.2f} GB")

    # ── KV Cache ──
    kv_per_token = compute_kv_per_token(p, dtype_bytes)
    kv_kib = kv_per_token / 1024
    kv_at_max = total_kv_cache(p, args.batch, args.seq, dtype_bytes, tp=1) / 1e9
    print(f"\n{'─'*60}")
    print(f"  KV Cache (dtype={args.dtype}, batch={args.batch}, seq={args.seq})")
    print(f"{'─'*60}")
    print(f"  Per-token KV:         {kv_kib:.4f} KiB/token")
    print(f"  Per-token KV total:   {kv_per_token:.2f} B/token")
    print(f"  KV Cache@{args.seq}:  {kv_at_max:.2f} GB")
    if p["has_mla"]:
        print(f"  ⚠ MLA KV Cache — NOT TP-sharded (each rank holds full KV)")
    else:
        print(f"  GQA KV Cache — TP-sharded (÷TP per rank)")

    # ── Quant×Parallel Memory Matrix ──
    if is_moe and matrix:
        print(f"\n{'─'*80}")
        print(f"  §3.4 单卡权重显存矩阵 (Quantization × Parallelism)")
        print(f"  Convention: K=BF16(replicated), Q_rep=quantized(replicated), Q_experts=quantized(EP-sharded)")
        print(f"{'─'*80}")
        print(f"  {'EP':>4}  {'TP':>4}  {'Quant':>6}  {'K BF16':>8}  {'Q_rep':>8}  {'Q_tp':>8}  {'Q_exp/卡':>10}  {'权重合计':>10}  {'64GB%':>6}  {'96GB%':>6}  {'判定':>6}")
        print(f"  {'-'*90}")

        for row in matrix:
            ep = row["ep"]
            tp = row["tp"]
            q = row["quant"]
            if not row["ep_ok"]:
                print(f"  {ep:>4}  {tp:>4}  {q:>6}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>10}  {'—':>10}  {'—':>6}  {'—':>6}  {'EP×':>6}")
                continue

            total = row["weight_total_gb"]
            pct64 = row["pct_64gb"]
            pct96 = row["pct_96gb"]

            if pct64 and pct64 > 100:
                verdict = "⚠ OOM"
            elif pct64 and pct64 > 90:
                verdict = "⚠ 边缘"
            elif total:
                verdict = "✓"
            else:
                verdict = "—"

            print(f"  {ep:>4}  {tp:>4}  {q:>6}  {row['K_bf16_gb']:>7.1f}G  {row['Q_rep_gb']:>7.1f}G  {row['Q_tp_gb']:>7.1f}G  {row['Q_experts_gb']:>9.1f}G  {total:>9.1f}G  {pct64:>5.1f}%  {pct96:>5.1f}%  {verdict:>6}")

        # ── Trend analysis ──
        print(f"\n  ── 趋势分析 ──")
        print(f"  K_BF16 (固定复制的BF16部分):  {matrix_info['K_bf16_gb']:.2f} GB")
        print(f"  Q_rep (量化复制部分):          {matrix_info['Q_rep_params']*DTYPE_BYTES['w8a8']/1e9:.2f} GB (W8A8)")
        print(f"  Q_experts (专家权重总量):      {matrix_info['Q_experts_params']*2/1e9:.2f} GB (BF16)")
        print(f"")
        print(f"  EP 翻倍 ≠ 显存减半: K+Q_rep 基线固定，专家权重才随 EP 缩放")
        print(f"  W4A8 vs W8A8 节省的绝对量随 EP 增大而缩小（高 EP 下量化收益递减）")
        if matrix[0].get("pct_64gb") and matrix[0]["pct_64gb"] > 100:
            print(f"  ⚠ HBM 断崖: EP={matrix[0]['ep']}/W8A8 装不进 64GB 卡")

    # ── Strategy Search ──
    if args.strategy and strategies:
        print(f"\n{'─'*80}")
        print(f"  Parallel Strategy Search (NPU={args.npu_memory}GB, max_seq={args.seq}, max_devices={args.max_devices})")
        print(f"{'─'*80}")
        print(f"  {'EP':>6}  {'TP':>4}  {'DP':>4}  {'PP':>3}  {'总卡':>6}  {'机器(16卡)':>10}  {'权重':>8}  {'KV':>8}  {'开销':>8}  {'总计':>8}  {'判定':>8}")
        print(f"  {'-'*90}")

        recommended = None
        for i, s in enumerate(strategies):
            if s["fits"] and recommended is None:
                recommended = s

            verdict = "✅ 推荐" if s == recommended else ("✓ 可行" if s["fits"] else "✗ OOM")
            print(f"  {s['ep']:>6}  {s['tp']:>4}  {s['dp']:>4}  {s['pp']:>3}  {s['total_cards']:>6}  {s['machines']:>10}台  {s['weight_gb']:>7.1f}G  {s['kv_gb']:>7.1f}G  {s['overhead_gb']:>7.1f}G  {s['total_gb']:>7.1f}G  {verdict:>8}")

        if recommended:
            print(f"\n  ★ 推荐配置: EP={recommended['ep']}, TP={recommended['tp']}, DP={recommended['dp']}, PP={recommended['pp']}")
            print(f"    总卡数={recommended['total_cards']}, 机器数={recommended['machines']}台(16卡/台)")
            print(f"    单卡显存: 权重{recommended['weight_gb']}G + KV{recommended['kv_gb']}G + 开销{recommended['overhead_gb']}G = {recommended['total_gb']}G")

    # ── NPU Overhead ──
    overhead = compute_npu_overhead(16, 8, is_moe, p["has_mla"])
    print(f"\n{'─'*60}")
    print(f"  NPU 隐性开销拆解 (Ascend 910B)")
    print(f"{'─'*60}")
    print(f"  驱动预留:         {overhead['driver']:.1f} GiB")
    print(f"  HCCS 通信缓冲:    {overhead['hccs']:.1f} GiB")
    print(f"  激活峰值:         {overhead['activation']:.1f} GiB")
    print(f"  碎片/分配器:       {overhead['fragmentation']:.1f} GiB")
    print(f"  合计:             {overhead['total']:.1f} GiB")
    print(f"  建议可用显存:     {args.npu_memory - overhead['total'] - NPU_SAFE_MARGIN_GIB:.1f} GiB (扣除开销+安全余量)")

    print(f"\n{'='*80}")


def print_json(p, classifications, matrix, matrix_info, strategies, args):
    """Print JSON output."""
    dtype_bytes = DTYPE_BYTES[args.dtype]
    is_moe = p["num_experts"] > 1

    output = {
        "model": {
            "hidden_size": p["hidden_size"],
            "num_layers": p["num_layers"],
            "num_dense_layers": p["num_dense_layers"],
            "num_moe_layers": p["num_moe_layers"],
            "attention_type": "MLA" if p["has_mla"] else "GQA" if p["num_kv_heads"] < p["num_heads"] else "MHA",
            "has_mla": p["has_mla"],
            "num_experts": p["num_experts"],
            "top_k": p["top_k"],
            "has_vision": p["has_vision"],
            "num_mtp": p["num_mtp"],
        },
        "dtype": args.dtype,
        "batch_size": args.batch,
        "seq_len": args.seq,
        "npu_memory_gb": args.npu_memory,
        "weight_classification": {
            cat: {
                "params": info["params"],
                "params_formatted": format_params(info["params"]),
                "gb_bf16": info["params"] * 2 / 1e9,
                "split": info["split"],
                "components": info["components"],
            }
            for cat, info in classifications.items()
        },
        "kv_cache": {
            "per_token_bytes": compute_kv_per_token(p, dtype_bytes),
            "per_token_kib": compute_kv_per_token(p, dtype_bytes) / 1024,
            "total_gb_seq": total_kv_cache(p, args.batch, args.seq, dtype_bytes, tp=1) / 1e9,
            "mla_not_tp_sharded": p["has_mla"],
        },
        "memory_matrix_3_4": matrix,
        "matrix_info": matrix_info,
    }

    if strategies:
        output["strategies"] = strategies
        output["recommended"] = strategies[0] if strategies and strategies[0]["fits"] else None

    print(json.dumps(output, indent=2, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="vLLM Memory Calculator & Parallel Strategy Planner for Ascend NPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to config.json or directory containing it")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--seq", type=int, default=4096, help="Sequence length (default: 4096)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp8", "w8a8", "w4a8"], help="Data type (default: bf16)")
    parser.add_argument("--npu-memory", type=int, default=64, help="Single NPU card memory in GB (default: 64)")
    parser.add_argument("--max-devices", type=int, default=16, help="Maximum number of cards (default: 16)")
    parser.add_argument("--tp-range", type=str, default="1,2,4,8,16", help="TP candidate values (default: 1,2,4,8,16)")
    parser.add_argument("--pp-range", type=str, default="1,2", help="PP candidate values (default: 1,2)")
    parser.add_argument("--strategy", action="store_true", help="Search for optimal parallel strategy")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--kvcache-json", type=str, help="Path to pre-computed kvcache_calculator.py JSON output")
    parser.add_argument("--verbose", action="store_true", help="Show detailed breakdown")
    args = parser.parse_args()

    # Parse ranges
    args.tp_range = [int(x) for x in args.tp_range.split(",")]
    args.pp_range = [int(x) for x in args.pp_range.split(",")]

    # Load config
    cfg = load_config(args.config)

    # Extract parameters
    p = extract_params(cfg)

    # If kvcache-json provided, merge in pre-computed values
    if args.kvcache_json:
        with open(args.kvcache_json, "r") as f:
            kvcache_data = json.load(f)
        p.update(kvcache_data.get("key_parameters", {}))

    # Classify weights
    dtype_bytes = DTYPE_BYTES[args.dtype]
    classifications = classify_weights(p)

    # Compute memory matrix
    matrix, matrix_info = compute_memory_matrix(
        p, classifications, dtype_bytes, args.batch, args.seq, tp=8
    )

    # Strategy search
    strategies = None
    if args.strategy:
        strategies = search_strategies(p, classifications, dtype_bytes, args)
        if not strategies:
            print("WARNING: No feasible strategy found!", file=sys.stderr)

    # Output
    if args.json:
        print_json(p, classifications, matrix, matrix_info, strategies, args)
    else:
        print_results(p, classifications, matrix, matrix_info, strategies, args)


if __name__ == "__main__":
    main()
