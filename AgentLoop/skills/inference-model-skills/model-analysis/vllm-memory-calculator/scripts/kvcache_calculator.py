#!/usr/bin/env python3
"""
KV Cache & Parameter Calculator for Ascend NPU Model Analysis.

Reads a HuggingFace config.json and computes:
  - Architecture type detection (MHA / GQA / MLA / GDN / CSA+HCA)
  - Per-component parameter breakdown with exact formulas
  - Per-token KV cache size for each attention type
  - Active parameters per token
  - Total KV cache at target context lengths
  - Compression ratio vs MHA baseline

Supports: Dense, MoE, MLA, GQA, MHA, GDN (Linear+Attention hybrid),
         CSA+HCA (DeepSeek V4), MSA (MiniMax sparse), DSA (GLM/DeepSeek),
         MTP speculative layers, Multimodal vision tower.

Usage:
  python kvcache_calculator.py <config.json> [options]

Options:
  --seq N         Single sequence length (default: 4096)
  --all            Show multiple standard sequence lengths
  --batch N       Batch size (default: 1)
  --dtype TYPE    Data type: bf16, fp16, fp8, w8a8, w4a8 (default: bf16)
  --json          Output as JSON
  --context N     Target max context length for total KV (default: from config)
  --verbose       Show per-layer and per-component breakdown
"""

import json
import sys
import os
import argparse

# ─── dtype byte mapping ───
DTYPE_BYTES = {
    "bf16": 2, "fp16": 2, "fp8": 1, "fp8_e4m3": 1,
    "w8a8": 1, "w4a8": 0.5, "int8": 1, "int4": 0.5,
}

# ─── standard sequence lengths for --all ───
STD_SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]


def load_config(path):
    """Load config.json, handling both file paths and directory paths."""
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.exists(path):
        print(f"ERROR: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(cfg, *keys):
    """Return the first key that exists in config, or None."""
    for k in keys:
        if k in cfg:
            return cfg[k]
    return None


def get_dtype_bytes(dtype_str):
    """Convert dtype string to bytes per element."""
    if dtype_str in DTYPE_BYTES:
        return DTYPE_BYTES[dtype_str]
    print(f"WARNING: unknown dtype '{dtype_str}', falling back to bf16 (2 bytes)", file=sys.stderr)
    return 2


# ═══════════════════════════════════════════════════════════════
# Architecture Detection
# ═══════════════════════════════════════════════════════════════

def detect_architecture(cfg):
    """
    Detect model architecture type from config.json keys.
    Returns a dict with architecture info and key parameters.
    """
    arch = {
        "type": "unknown",
        "subtypes": [],
        "features": [],
    }

    # ── MLA detection ──
    has_mla = any(k in cfg for k in ["kv_lora_rank", "q_lora_rank"])
    if has_mla:
        arch["features"].append("MLA")
        arch["subtypes"].append("mla")

    # ── Attention type ──
    n_heads = norm(cfg, "num_attention_heads", "n_heads")
    n_kv_heads = norm(cfg, "num_key_value_heads", "n_kv_heads")
    if n_heads and n_kv_heads:
        if n_kv_heads == n_heads:
            if not has_mla:
                arch["type"] = "MHA"
        else:
            if not has_mla:
                arch["type"] = "GQA"
            arch["features"].append(f"GQA({n_kv_heads}kv/{n_heads}q)")

    if has_mla:
        arch["type"] = "MLA"

    # ── MoE detection ──
    n_experts = norm(cfg, "num_local_experts", "n_routed_experts", "num_experts")
    top_k = norm(cfg, "num_experts_per_tok", "top_k", "n_activated_experts")
    if n_experts and n_experts > 1:
        arch["features"].append(f"MoE({n_experts} experts, top-{top_k or '?'})")
        arch["subtypes"].append("moe")

    # ── Linear Attention (GDN / Delta / GLA) ──
    linear_config = cfg.get("linear_attn_config", cfg.get("linear_attention_config", {}))
    if isinstance(linear_config, dict):
        kda_layers = linear_config.get("kda_layers", [])
        if kda_layers:
            arch["features"].append(f"Linear/KDA({len(kda_layers)} layers)")
            arch["subtypes"].append("gdn")
            if arch["type"] in ("unknown", "MHA", "GQA"):
                arch["type"] = "GDN"

    use_delta = cfg.get("use_gated_delta_rule", False)
    if use_delta:
        arch["features"].append("GatedDeltaRule")
        arch["subtypes"].append("delta")

    use_gla = cfg.get("gate_lr") is not None or cfg.get("linear_attention_dim") is not None
    if use_gla and "gdn" not in arch["subtypes"]:
        arch["features"].append("SimpleGLA")
        arch["subtypes"].append("gla")
        if arch["type"] in ("unknown", "MHA", "GQA"):
            arch["type"] = "GDN"

    # ── CSA+HCA (DeepSeek V4) ──
    csa_ratio = cfg.get("csa_compression_ratio")
    if csa_ratio:
        arch["features"].append(f"CSA(ratio={csa_ratio})")
        arch["subtypes"].append("csa")
        arch["type"] = "CSA+HCA"

    # Check architectures list
    architectures = cfg.get("architectures", [])
    for a in architectures:
        if "DeepSeekV4" in a:
            arch["type"] = "CSA+HCA"
            arch["subtypes"].append("csa")

    # ── Sparse Attention (MSA / DSA) ──
    sparse_config = cfg.get("sparse_attention_config", {})
    if isinstance(sparse_config, dict) and sparse_config.get("sparse_topk_blocks"):
        arch["features"].append(f"MSA(topk={sparse_config['sparse_topk_blocks']}, block={sparse_config.get('sparse_block_size','?')})")
        arch["subtypes"].append("msa")

    index_topk = norm(cfg, "index_topk")
    index_n_heads = norm(cfg, "index_n_heads")
    if index_topk and index_n_heads:
        arch["features"].append(f"DSA(topk={index_topk}, heads={index_n_heads})")
        arch["subtypes"].append("dsa")

    # ── Multimodal ──
    if "vision_config" in cfg or cfg.get("has_vision", False):
        arch["features"].append("Multimodal")
        arch["subtypes"].append("vision")

    # ── MTP / Speculative ──
    mtp = norm(cfg, "num_nextn_predict_layers", "num_mtp_modules")
    if mtp:
        arch["features"].append(f"MTP({mtp} layers)")

    # ── RoPE variants ──
    if cfg.get("rope_mode") == "3d" or cfg.get("mrope_section"):
        arch["features"].append("MRoPE/3D-RoPE")

    if cfg.get("partial_rotary_factor", 1.0) < 1.0:
        arch["features"].append(f"PartialRoPE(factor={cfg['partial_rotary_factor']})")

    # ── QK Norm ──
    if cfg.get("use_qk_norm") and cfg.get("qk_norm_type") == "per_head":
        arch["features"].append("QK-Norm(per_head)")

    # ── Hyper-Connection ──
    if cfg.get("use_hyper_connection") or cfg.get("hyper_connection"):
        arch["features"].append("HyperConnection")

    # ── Fallback ──
    if arch["type"] == "unknown":
        if n_heads:
            arch["type"] = "MHA"  # default assumption

    return arch


# ═══════════════════════════════════════════════════════════════
# Parameter Extraction (normalized)
# ═══════════════════════════════════════════════════════════════

def extract_params(cfg):
    """Extract and normalize all key parameters from config.json."""
    p = {}

    p["hidden_size"] = norm(cfg, "hidden_size", "dim")
    p["num_layers"] = norm(cfg, "num_hidden_layers", "n_layers")
    p["num_heads"] = norm(cfg, "num_attention_heads", "n_heads")
    p["num_kv_heads"] = norm(cfg, "num_key_value_heads", "n_kv_heads", "num_attention_heads")
    p["head_dim"] = norm(cfg, "head_dim")
    # For MLA, prefer explicit dims over computed head_dim
    p["qk_nope_head_dim"] = norm(cfg, "qk_nope_head_dim")
    p["qk_rope_head_dim"] = norm(cfg, "qk_rope_head_dim")
    if p["head_dim"] is None and p["hidden_size"] and p["num_heads"]:
        p["head_dim"] = p["hidden_size"] // p["num_heads"]
    # Explicit head_dim from MLA config overrides computed
    if p.get("qk_nope_head_dim") and p.get("qk_rope_head_dim"):
        p["head_dim"] = p["qk_nope_head_dim"] + p["qk_rope_head_dim"]

    # MLA
    p["kv_lora_rank"] = norm(cfg, "kv_lora_rank")
    p["q_lora_rank"] = norm(cfg, "q_lora_rank")
    p["qk_head_dim"] = norm(cfg, "qk_head_dim", "qk_nope_head_dim")
    p["qk_rope_head_dim"] = norm(cfg, "qk_rope_head_dim")
    p["v_head_dim"] = norm(cfg, "v_head_dim")

    # MoE
    p["num_experts"] = norm(cfg, "num_local_experts", "n_routed_experts", "num_experts", 1)
    p["top_k"] = norm(cfg, "num_experts_per_tok", "n_activated_experts", "num_experts_per_token", 1)
    p["moe_intermediate_size"] = norm(cfg, "moe_intermediate_size")
    if p["moe_intermediate_size"] is None:
        p["moe_intermediate_size"] = norm(cfg, "intermediate_size")
    p["num_shared_experts"] = norm(cfg, "n_shared_experts", "num_shared_experts", 0)
    p["shared_intermediate_size"] = norm(cfg, "shared_expert_intermediate_size", "shared_intermediate_size")
    if p["shared_intermediate_size"] is None:
        p["shared_intermediate_size"] = norm(cfg, "intermediate_size")

    # Dense MLP
    p["intermediate_size"] = norm(cfg, "intermediate_size")
    p["dense_intermediate_size"] = norm(cfg, "dense_intermediate_size", "intermediate_size")

    # LatentMoE
    p["routed_expert_hidden_size"] = norm(cfg, "routed_expert_hidden_size")
    p["hidden_act"] = norm(cfg, "hidden_act", "silu")

    # Vocabulary
    p["vocab_size"] = norm(cfg, "vocab_size")
    p["tie_word_embeddings"] = cfg.get("tie_word_embeddings", False)

    # Sparse attention
    sparse_config = cfg.get("sparse_attention_config", {})
    if isinstance(sparse_config, dict):
        p["sparse_topk_blocks"] = sparse_config.get("sparse_topk_blocks")
        p["sparse_block_size"] = sparse_config.get("sparse_block_size")
        p["sparse_index_dim"] = sparse_config.get("sparse_index_dim", sparse_config.get("index_dim"))
        p["sparse_num_index_heads"] = sparse_config.get("sparse_num_index_heads", sparse_config.get("index_n_heads"))

    # DSA Indexer
    p["index_topk"] = norm(cfg, "index_topk")
    p["index_n_heads"] = norm(cfg, "index_n_heads")
    p["index_head_dim"] = norm(cfg, "index_head_dim")

    # Layer frequencies
    p["moe_layer_freq"] = cfg.get("moe_layer_freq", [])
    p["dense_layers"] = cfg.get("dense_layers", cfg.get("dense_layer_indices", []))
    p["sparse_attention_freq"] = cfg.get("sparse_attention_freq", [])

    # Context
    p["max_position_embeddings"] = norm(cfg, "max_position_embeddings", "max_seq_len", 4096)

    # MTP
    p["num_mtp"] = norm(cfg, "num_nextn_predict_layers", "num_mtp_modules", 0)

    # Vision
    vision = cfg.get("vision_config", {})
    p["has_vision"] = bool(vision)
    if isinstance(vision, dict):
        p["vision_hidden_size"] = vision.get("hidden_size")
        p["vision_num_layers"] = vision.get("num_hidden_layers")
        p["vision_num_heads"] = vision.get("num_attention_heads")
        p["vision_intermediate_size"] = vision.get("intermediate_size")
        p["vision_patch_size"] = vision.get("patch_size")
        p["vision_image_size"] = vision.get("image_size")

    # Normalization
    p["rms_norm_eps"] = norm(cfg, "rms_norm_eps", 1e-6)
    p["use_qk_norm"] = cfg.get("use_qk_norm", False)
    p["qk_norm_type"] = cfg.get("qk_norm_type", "per_head" if p["use_qk_norm"] else None)

    # RoPE
    p["rope_theta"] = norm(cfg, "rope_theta", 10000.0)
    p["partial_rotary_factor"] = norm(cfg, "partial_rotary_factor", 1.0)

    # Quantization
    p["quantization"] = cfg.get("quantization_config", {})

    # Derived: count layer types
    p["num_moe_layers"] = 0
    p["num_dense_layers"] = p["num_layers"]  # default all dense
    p["num_sparse_attn_layers"] = 0
    p["num_linear_attn_layers"] = 0

    if p["moe_layer_freq"] and isinstance(p["moe_layer_freq"], (list, tuple)) and len(p["moe_layer_freq"]) == p["num_layers"]:
        p["num_moe_layers"] = sum(1 for x in p["moe_layer_freq"] if x > 0)
        p["num_dense_layers"] = p["num_layers"] - p["num_moe_layers"]
    elif p["num_experts"] and p["num_experts"] > 1:
        if p["dense_layers"]:
            p["num_dense_layers"] = len(p["dense_layers"])
        else:
            p["num_dense_layers"] = p.get("first_k_dense_replace", 1)
        p["num_moe_layers"] = p["num_layers"] - p["num_dense_layers"]

    if p["sparse_attention_freq"] and isinstance(p["sparse_attention_freq"], (list, tuple)) and len(p["sparse_attention_freq"]) == p["num_layers"]:
        p["num_sparse_attn_layers"] = sum(1 for x in p["sparse_attention_freq"] if x > 0)

    # Linear attention layers
    linear_config = cfg.get("linear_attn_config", cfg.get("linear_attention_config", {}))
    if isinstance(linear_config, dict):
        kda_layers = linear_config.get("kda_layers", [])
        if kda_layers:
            p["num_linear_attn_layers"] = len(kda_layers)
            p["num_full_attn_layers"] = p["num_layers"] - p["num_linear_attn_layers"]

    return p


# ═══════════════════════════════════════════════════════════════
# Parameter Counting
# ═══════════════════════════════════════════════════════════════

def count_embedding(p):
    """Embedding: vocab_size × hidden_size"""
    return p["vocab_size"] * p["hidden_size"]


def count_lm_head(p):
    """LM Head: 0 if tied with embedding, else vocab_size × hidden_size"""
    if p["tie_word_embeddings"]:
        return 0
    return p["vocab_size"] * p["hidden_size"]


def count_attention_full(p):
    """Full Attention (GQA/MHA) per layer: Q + K + V + O projections"""
    h = p["hidden_size"]
    nq = p["num_heads"]
    nkv = p["num_kv_heads"]
    d = p["head_dim"]
    # Q: h × nq*d, K: h × nkv*d, V: h × nkv*d, O: nq*d × h
    q = h * nq * d
    k = h * nkv * d
    v = h * nkv * d
    o = nq * d * h
    # Optional bias
    return q + k + v + o


def count_attention_mla(p):
    """MLA attention per layer: Q path + KV path + O projection"""
    h = p["hidden_size"]
    q_lora = p.get("q_lora_rank") or p.get("kv_lora_rank")  # fallback
    kv_lora = p.get("kv_lora_rank") or 0
    qk_dim = p.get("qk_head_dim") or 0
    qk_rope = p.get("qk_rope_head_dim") or 0
    v_dim = p.get("v_head_dim") or 0
    n_heads = p["num_heads"]

    total = 0

    # Q path: q_a_proj (h→q_lora) + q_b_proj (q_lora→n_heads×qk_dim)
    if q_lora and qk_dim:
        total += h * q_lora  # q_a_proj
        total += q_lora * n_heads * qk_dim  # q_b_proj (nope part)
        if qk_rope:
            total += q_lora * n_heads * qk_rope  # q_b_proj (rope part)

    # KV path: kv_a_proj (h→kv_lora+qk_rope) + kv_b_proj (kv_lora→n_heads×(qk_dim+v_dim))
    if kv_lora:
        # kv_a_proj: hidden → kv_lora_rank + qk_rope_head_dim
        kv_a_out = kv_lora + qk_rope
        total += h * kv_a_out

        # kv_b_proj (nope): kv_lora_rank → n_heads × qk_head_dim
        if qk_dim:
            total += kv_lora * n_heads * qk_dim

        # kv_b_proj (v): kv_lora_rank → n_heads × v_head_dim
        if v_dim and v_dim != qk_dim:
            total += kv_lora * n_heads * v_dim
        elif qk_dim:
            # Often v_head_dim = qk_head_dim, shared in b_proj
            total += kv_lora * n_heads * qk_dim

    # O projection: n_heads × v_dim × hidden_size
    out_dim = v_dim or qk_dim
    if out_dim and n_heads:
        total += n_heads * out_dim * h

    return total


def count_dense_mlp(p):
    """Dense SwiGLU MLP per layer: gate + up + down = 3 × hidden × intermediate"""
    h = p["hidden_size"]
    inter = p["dense_intermediate_size"] or p["intermediate_size"] or 0
    return 3 * h * inter


def count_moe_expert(p):
    """Single MoE expert: w1(w13 gate+up) + w2(down)"""
    h = p.get("routed_expert_hidden_size") or p["hidden_size"]
    inter = p["moe_intermediate_size"] or p["intermediate_size"] or 0
    # w13 (gate + up): 2 × h × inter, w2 (down): inter × h
    return 2 * h * inter + inter * h  # = 3 × h × inter


def count_moe_gate(p):
    """MoE router gate per layer: hidden_size × num_experts"""
    return p["hidden_size"] * p["num_experts"]


def count_shared_expert(p):
    """Shared expert (SwiGLU): 3 × hidden × shared_intermediate"""
    h = p["hidden_size"]
    inter = p["shared_intermediate_size"] or p["intermediate_size"] or 0
    return 3 * h * inter


def count_indexer(p):
    """Sparse attention Indexer (MSA/DSA) per layer."""
    h = p["hidden_size"]
    n_heads = p.get("sparse_num_index_heads") or p.get("index_n_heads") or 0
    idx_dim = p.get("sparse_index_dim") or p.get("index_head_dim") or 0

    if not n_heads or not idx_dim:
        return 0

    total = 0
    # Q projection: h × n_heads × idx_dim
    total += h * n_heads * idx_dim
    # K projection: h × idx_dim (usually single-head)
    total += h * idx_dim
    # Per-head Q norm, per-head K norm
    total += n_heads * idx_dim  # q_norm
    if p.get("qk_norm_type") == "per_head" or p.get("sparse_index_dim"):
        total += idx_dim  # k_norm (single head)
    return total


def count_per_head_qk_norm(p):
    """Per-head QK norm weights."""
    if not (p.get("use_qk_norm") and p.get("qk_norm_type") == "per_head"):
        return 0
    n_heads = p["num_heads"]
    n_kv = p["num_kv_heads"] or n_heads
    d = p["head_dim"]
    return (n_heads + n_kv) * d


def count_rms_norm(hidden_size):
    """RMSNorm: hidden_size parameters (weight)."""
    return hidden_size


def count_vision_tower(p):
    """Vision tower parameters (ViT-style)."""
    if not p.get("has_vision"):
        return 0
    vp = p
    v_hidden = vp.get("vision_hidden_size") or vp["hidden_size"]
    v_layers = vp.get("vision_num_layers") or 0
    v_heads = vp.get("vision_num_heads") or 0
    v_inter = vp.get("vision_intermediate_size") or (4 * v_hidden)
    v_head_dim = v_hidden // v_heads if v_heads else 0

    if not v_layers:
        return 0

    total = 0
    # Per vision layer: attention + MLP + 2×LayerNorm
    attn = (v_hidden * v_heads * v_head_dim) * 4  # Q,K,V,O
    mlp = 3 * v_hidden * v_inter
    norms = 2 * v_hidden * 2  # 2 RMSNorm per layer, weight+bias
    total += (attn + mlp + norms) * v_layers

    # Patch embedding
    patch = vp.get("vision_patch_size", 16)
    img = vp.get("vision_image_size", 224)
    if patch and img:
        num_patches = (img // patch) ** 2
        total += num_patches * v_hidden * 3 * patch * patch  # rough

    # Projector: hidden_size → vision_hidden (or vice versa)
    total += v_hidden * vp["hidden_size"] * 2

    return total


def count_mtp_layers(p):
    """MTP (Multi-Token Prediction) layer parameters."""
    n_mtp = p.get("num_mtp", 0)
    if n_mtp == 0:
        return 0
    # Each MTP layer ≈ 1 transformer layer (attention + MLP) + embedding
    h = p["hidden_size"]
    n_heads = p["num_heads"]
    n_kv = p["num_kv_heads"] or n_heads
    d = p["head_dim"]
    has_mla = bool(p.get("kv_lora_rank"))

    per_layer = 0
    if has_mla:
        per_layer += count_attention_mla(p)
    else:
        per_layer += count_attention_full(p)
    per_layer += count_dense_mlp(p)
    per_layer += count_rms_norm(h) * 2
    # Extra embedding for MTP
    per_layer += h  # layer norm weight
    return per_layer * n_mtp


# ═══════════════════════════════════════════════════════════════
# Full Parameter Computation
# ═══════════════════════════════════════════════════════════════

def compute_all_params(p, arch):
    """Compute complete parameter breakdown."""
    result = {
        "components": {},
        "total": 0,
        "active_per_token": 0,
    }

    h = p["hidden_size"]
    n_layers = p["num_layers"]
    n_dense = p["num_dense_layers"]
    n_moe = p["num_moe_layers"]
    n_experts = p["num_experts"]
    top_k = p["top_k"]
    n_shared = p["num_shared_experts"]

    # ── Embedding ──
    emb = count_embedding(p)
    result["components"]["embedding"] = {"params": emb, "formula": f"vocab({p['vocab_size']}) × hidden({h})"}
    result["total"] += emb

    # ── LM Head ──
    lm_head = count_lm_head(p)
    result["components"]["lm_head"] = {"params": lm_head, "formula": f"vocab({p['vocab_size']}) × hidden({h})"}
    if lm_head == 0:
        result["components"]["lm_head"]["note"] = "tied with embedding"
    result["total"] += lm_head

    # ── Per-layer attention ──
    has_mla = bool(p.get("kv_lora_rank"))
    if has_mla:
        attn_per_layer = count_attention_mla(p)
        attn_formula = f"MLA: q_a_proj+qv_b_proj+kv_a_proj+kv_b_proj+o_proj"
    else:
        attn_per_layer = count_attention_full(p)
        attn_formula = f"{'GQA' if p['num_kv_heads'] < p['num_heads'] else 'MHA'}: q+k+v+o_proj"

    result["components"]["attention"] = {
        "params": attn_per_layer * n_layers,
        "per_layer": attn_per_layer,
        "num_layers": n_layers,
        "formula": attn_formula,
    }
    result["total"] += attn_per_layer * n_layers

    # ── Per-head QK Norm ──
    qk_norm_per_layer = count_per_head_qk_norm(p)
    if qk_norm_per_layer > 0:
        result["components"]["qk_norm"] = {
            "params": qk_norm_per_layer * n_layers,
            "per_layer": qk_norm_per_layer,
            "num_layers": n_layers,
            "formula": f"(n_heads({p['num_heads']}+n_kv({p['num_kv_heads']})) × head_dim({p['head_dim']})",
        }
        result["total"] += qk_norm_per_layer * n_layers

    # ── Dense MLP ──
    dense_mlp_per_layer = count_dense_mlp(p)
    if n_dense > 0:
        result["components"]["dense_mlp"] = {
            "params": dense_mlp_per_layer * n_dense,
            "per_layer": dense_mlp_per_layer,
            "num_layers": n_dense,
            "formula": f"3 × hidden({h}) × intermediate({p['dense_intermediate_size'] or p['intermediate_size']})",
        }
        result["total"] += dense_mlp_per_layer * n_dense

    # ── MoE experts (per-layer: each MoE layer has its own 896 experts) ──
    if n_experts > 1 and n_moe > 0:
        expert_size = count_moe_expert(p)
        expert_h = p.get("routed_expert_hidden_size") or h
        expert_inter = p["moe_intermediate_size"] or p["intermediate_size"] or 0
        all_experts = expert_size * n_experts * n_moe
        result["components"]["moe_routed_experts"] = {
            "params": all_experts,
            "per_expert": expert_size,
            "num_experts": n_experts,
            "num_moe_layers": n_moe,
            "formula": f"per_expert: 3x{expert_h}x{expert_inter}; total: per_expert x {n_experts} x {n_moe}",
        }
        result["total"] += all_experts

        # MoE gate (per-layer)
        gate_per_layer = count_moe_gate(p)
        result["components"]["moe_gate"] = {
            "params": gate_per_layer * n_moe,
            "per_layer": gate_per_layer,
            "num_layers": n_moe,
            "formula": f"hidden({h}) x num_experts({n_experts})",
        }
        result["total"] += gate_per_layer * n_moe

    # ── Shared experts (per-layer) ──
    if n_shared > 0 and n_moe > 0:
        shared_size = count_shared_expert(p)
        result["components"]["shared_experts"] = {
            "params": shared_size * n_shared * n_moe,
            "per_expert": shared_size,
            "num_shared": n_shared,
            "num_layers": n_moe,
            "formula": f"3 x hidden({h}) x shared_intermediate({p['shared_intermediate_size'] or p['intermediate_size']})",
        }
        result["total"] += shared_size * n_shared * n_moe

    # ── Indexer ──
    indexer_per_layer = count_indexer(p)
    n_sparse = p.get("num_sparse_attn_layers", 0)
    if n_sparse > 0 and indexer_per_layer > 0:
        result["components"]["indexer"] = {
            "params": indexer_per_layer * n_sparse,
            "per_layer": indexer_per_layer,
            "num_layers": n_sparse,
            "formula": "q_proj + k_proj + q_norm + k_norm",
        }
        result["total"] += indexer_per_layer * n_sparse

    # ── RMSNorm / LayerNorm ──
    # 2 per layer (attn_norm + ffn_norm) + final norm
    norm_params = count_rms_norm(h) * (n_layers * 2 + 1)
    result["components"]["norms"] = {
        "params": norm_params,
        "per_norm": h * 2,  # weight + optional bias
        "num_norms": n_layers * 2 + 1,
        "formula": f"hidden({h}) × (2×{n_layers}+1)",
    }
    result["total"] += norm_params

    # ── Vision Tower ──
    if p.get("has_vision"):
        vision_params = count_vision_tower(p)
        result["components"]["vision_tower"] = {
            "params": vision_params,
            "formula": "ViT encoder + patch_embed + projector",
        }
        result["total"] += vision_params

    # ── MTP layers ──
    mtp_params = count_mtp_layers(p)
    if mtp_params > 0:
        result["components"]["mtp_layers"] = {
            "params": mtp_params,
            "num_mtp": p["num_mtp"],
            "formula": f"MTP × {p['num_mtp']}: (attn+mlp+norm) per layer",
        }
        result["total"] += mtp_params

    # ── Active parameters per token ──
    # All non-MoE-routed weights + (top_k / num_experts) of routed expert weights
    non_expert_total = result["total"] - result["components"].get("moe_routed_experts", {}).get("params", 0)
    routed_params = result["components"].get("moe_routed_experts", {}).get("params", 0)
    # Include lm_head in active (inference convention)
    active_expert_fraction = (top_k / n_experts) if n_experts > 0 else 1.0
    active_params = non_expert_total + routed_params * active_expert_fraction
    result["active_per_token"] = active_params
    result["active_formula"] = (
        f"non_expert({non_expert_total / 1e9:.2f}B) + "
        f"expert_routed({routed_params / 1e9:.2f}B) × top_k({top_k})/n_experts({n_experts})"
    )

    return result


# ═══════════════════════════════════════════════════════════════
# KV Cache Computation
# ═══════════════════════════════════════════════════════════════

def compute_kv_cache(p, arch, dtype_bytes, batch_size=1):
    """Compute per-token KV cache for each attention type."""
    result = {
        "per_token_per_layer_bytes": {},
        "per_token_total_bytes": 0,
        "mha_baseline_bytes": 0,
        "compression_ratio": 1.0,
        "breakdown": [],
    }

    h = p["hidden_size"]
    n_layers = p["num_layers"]
    n_heads = p["num_heads"]
    n_kv = p["num_kv_heads"] or n_heads
    d = p["head_dim"]
    has_mla = bool(p.get("kv_lora_rank"))

    # ── MHA baseline (for compression ratio) ──
    mha_per_layer = n_heads * d * 2 * dtype_bytes  # K + V, all heads
    result["mha_baseline_bytes"] = mha_per_layer

    # ── Full Attention (GQA) ──
    if not has_mla:
        gqa_per_layer = n_kv * d * 2 * dtype_bytes
        result["per_token_per_layer_bytes"]["full_attention"] = gqa_per_layer
        result["breakdown"].append({
            "type": "Full Attention (GQA/MHA)",
            "formula": f"num_kv_heads({n_kv}) × head_dim({d}) × 2(K+V) × dtype_bytes({dtype_bytes})",
            "bytes_per_layer": gqa_per_layer,
        })

    # ── MLA ──
    if has_mla:
        kv_lora = p.get("kv_lora_rank", 0)
        qk_rope = p.get("qk_rope_head_dim", 0)
        mla_per_layer = (kv_lora + qk_rope) * dtype_bytes
        result["per_token_per_layer_bytes"]["mla"] = mla_per_layer
        result["breakdown"].append({
            "type": "MLA (absorption mode)",
            "formula": f"(kv_lora_rank({kv_lora}) + qk_rope_head_dim({qk_rope})) × dtype_bytes({dtype_bytes})",
            "bytes_per_layer": mla_per_layer,
        })

    # ── Linear Attention (GDN) ──
    n_linear = p.get("num_linear_attn_layers", 0)
    if n_linear > 0:
        # Fixed recurrent state, no per-token cost
        state_size = n_heads * d * d * dtype_bytes
        result["breakdown"].append({
            "type": "Linear Attention (GDN/GLA/Delta)",
            "formula": f"Fixed state: n_heads({n_heads}) x head_dim({d})^2 x dtype_bytes",
            "bytes_per_layer": 0,
            "state_bytes_per_session": state_size,
        })

    # ── Sparse-Attention Indexer Key Cache ──
    idx_dim = p.get("sparse_index_dim") or p.get("index_head_dim")
    n_sparse = p.get("num_sparse_attn_layers", 0)
    if idx_dim and n_sparse > 0:
        indexer_per_layer = idx_dim * dtype_bytes
        result["breakdown"].append({
            "type": "MSA/DSA Indexer Key Cache",
            "formula": f"index_dim({idx_dim}) × dtype_bytes({dtype_bytes})",
            "bytes_per_layer": indexer_per_layer,
        })

    # ── CSA+HCA (DeepSeek V4) ──
    csa_ratio = norm(p, "csa_compression_ratio") if isinstance(p, dict) else None
    if arch["type"] == "CSA+HCA" and csa_ratio:
        swa_per_layer = n_kv * d * 2 * dtype_bytes / csa_ratio  # approximate
        result["breakdown"].append({
            "type": "CSA (Compressed Sliding Window)",
            "formula": f"GQA_per_layer / ratio({csa_ratio})",
            "bytes_per_layer": swa_per_layer,
        })

    # ── Aggregate ──
    total_per_layer = sum(item["bytes_per_layer"] for item in result["breakdown"])
    result["per_token_total_bytes"] = total_per_layer * n_layers * batch_size
    result["compression_ratio"] = (
        mha_per_layer / max(total_per_layer, 1) if total_per_layer > 0 else 1.0
    )

    return result


def compute_kv_at_context(kv_result, seq_len, batch_size=1):
    """Compute total KV cache at a given context length."""
    per_token = kv_result["per_token_total_bytes"]
    return per_token * seq_len  # already includes batch_size in per_token_total_bytes if batch>1


# ═══════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════

def format_params(n):
    """Format parameter count in human-readable form."""
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    elif n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    else:
        return f"{n}"


def format_bytes(b):
    """Format bytes in human-readable form."""
    if b >= 1024**4:
        return f"{b/1024**4:.2f} TiB"
    elif b >= 1024**3:
        return f"{b/1024**3:.2f} GiB"
    elif b >= 1024**2:
        return f"{b/1024**2:.2f} MiB"
    elif b >= 1024:
        return f"{b/1024:.2f} KiB"
    else:
        return f"{b:.2f} B"


def format_gb_bytes(b):
    """Format in GB (10^9) for consistency with report convention."""
    return f"{b / 1e9:.4f} GB"


def print_report(cfg_path, p, arch, params, kv, args):
    """Print human-readable report."""
    dtype_bytes = get_dtype_bytes(args.dtype)
    print("=" * 72)
    print(f"  KV Cache & Parameter Calculator")
    print(f"  Config: {cfg_path}")
    print("=" * 72)

    # ── Architecture ──
    print(f"\n{'─'*50}")
    print(f"  Architecture Detection")
    print(f"{'─'*50}")
    print(f"  Type:       {arch['type']}")
    print(f"  Features:   {', '.join(arch.get('features', []))}")

    # ── Key parameters ──
    print(f"\n{'─'*50}")
    print(f"  Key Parameters")
    print(f"{'─'*50}")
    print(f"  hidden_size:          {p['hidden_size']}")
    print(f"  num_hidden_layers:    {p['num_layers']}")
    print(f"  num_attention_heads:  {p['num_heads']}")
    print(f"  num_key_value_heads:  {p['num_kv_heads']}")
    print(f"  head_dim:             {p['head_dim']}")
    if p.get("kv_lora_rank"):
        print(f"  kv_lora_rank:         {p['kv_lora_rank']}")
    if p.get("q_lora_rank"):
        print(f"  q_lora_rank:          {p['q_lora_rank']}")
    if p["num_experts"] > 1:
        print(f"  num_experts:          {p['num_experts']}")
        print(f"  top_k:                {p['top_k']}")
        print(f"  moe_intermediate:     {p['moe_intermediate_size']}")
        print(f"  num_shared_experts:   {p['num_shared_experts']}")
        print(f"  moe_layers / dense:   {p['num_moe_layers']} / {p['num_dense_layers']}")
    print(f"  vocab_size:           {p['vocab_size']}")
    print(f"  max_position_embeds:  {p['max_position_embeddings']}")
    print(f"  tie_word_embeddings:  {p['tie_word_embeddings']}")

    # ── Parameter breakdown ──
    print(f"\n{'─'*50}")
    print(f"  Parameter Breakdown")
    print(f"{'─'*50}")
    print(f"  {'Component':<30} {'Params':>12} {'%':>8}")
    print(f"  {'-'*50}")
    for name, comp in params["components"].items():
        pct = comp["params"] / params["total"] * 100 if params["total"] > 0 else 0
        print(f"  {name:<30} {format_params(comp['params']):>12} {pct:>7.1f}%")
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':<30} {format_params(params['total']):>12} {'100.0%':>8}")
    print(f"\n  Active params/token:  {format_params(params['active_per_token'])}")
    if params.get("active_formula"):
        print(f"  Formula:              {params['active_formula']}")

    # ── KV Cache ──
    print(f"\n{'─'*50}")
    print(f"  KV Cache Analysis (dtype={args.dtype}, batch={args.batch})")
    print(f"{'─'*50}")
    print(f"  Unit convention: KiB=1024B, GB=10^9B")
    print()
    for item in kv["breakdown"]:
        print(f"  {item['type']}:")
        print(f"    Formula:   {item['formula']}")
        if item["bytes_per_layer"] > 0:
            print(f"    Per layer: {item['bytes_per_layer']:.2f} B = {item['bytes_per_layer']/1024:.4f} KiB/token")
        else:
            print(f"    Per token: {item.get('state_bytes_per_session', 0)/1024/1024:.2f} MiB (fixed state)")

    print(f"\n  Per-token KV total:  {kv['per_token_total_bytes']:.2f} B = {kv['per_token_total_bytes']/1024:.4f} KiB/token")
    print(f"  MHA baseline:         {kv['mha_baseline_bytes'] * p['num_layers']:.2f} B = {kv['mha_baseline_bytes'] * p['num_layers'] / 1024:.4f} KiB/token")
    print(f"  Compression ratio:    {kv['compression_ratio']:.2f}× vs MHA")

    # ── Context length table ──
    if args.all:
        print(f"\n{'─'*80}")
        print(f"  KV Cache at Standard Context Lengths (batch={args.batch}, dtype={args.dtype})")
        print(f"{'─'*80}")
        print(f"  {'Seq Len':>12}  {'Tokens':>12}  {'KV Cache (GiB)':>18}  {'KV Cache (GB)':>18}")
        print(f"  {'-'*70}")
        for sl in STD_SEQ_LENS:
            if sl > p.get("max_position_embeddings", 0) * 2:
                break
            kv_bytes = compute_kv_at_context(kv, sl, args.batch)
            kv_gib = kv_bytes / (1024**3)
            kv_gb = kv_bytes / 1e9
            print(f"  {sl:>12,}  {sl:>12,}  {kv_gib:>17.2f} GiB  {kv_gb:>17.2f} GB")
    else:
        sl = args.seq
        kv_bytes = compute_kv_at_context(kv, sl, args.batch)
        kv_gib = kv_bytes / (1024**3)
        kv_gb = kv_bytes / 1e9
        print(f"\n  KV Cache at seq_len={sl}: {kv_gib:.2f} GiB = {kv_gb:.2f} GB")

    # ── Context limit ──
    ctx = args.context or p["max_position_embeddings"]
    kv_ctx = compute_kv_at_context(kv, ctx, args.batch)
    print(f"  KV Cache at {ctx:,} tokens:  {kv_ctx/1024**3:.2f} GiB = {kv_ctx/1e9:.2f} GB")

    # ── FP8 comparison ──
    if args.dtype in ("bf16", "fp16"):
        fp8_bytes = 1
        fp8_kv_total = kv["per_token_total_bytes"] / dtype_bytes * fp8_bytes
        fp8_ctx = fp8_kv_total * ctx
        print(f"\n  FP8 KV Cache at {ctx:,} tokens: {fp8_ctx/1024**3:.2f} GiB (saves {(kv_ctx-fp8_ctx)/1024**3:.2f} GiB vs {args.dtype})")

    print(f"\n{'='*72}")


def print_json(cfg_path, p, arch, params, kv, args):
    """Print JSON output."""
    dtype_bytes = get_dtype_bytes(args.dtype)
    output = {
        "config_path": cfg_path,
        "architecture": arch,
        "key_parameters": {
            "hidden_size": p["hidden_size"],
            "num_hidden_layers": p["num_layers"],
            "num_attention_heads": p["num_heads"],
            "num_key_value_heads": p["num_kv_heads"],
            "head_dim": p["head_dim"],
            "kv_lora_rank": p.get("kv_lora_rank"),
            "q_lora_rank": p.get("q_lora_rank"),
            "qk_head_dim": p.get("qk_head_dim"),
            "qk_rope_head_dim": p.get("qk_rope_head_dim"),
            "v_head_dim": p.get("v_head_dim"),
            "num_experts": p["num_experts"],
            "top_k": p["top_k"],
            "moe_intermediate_size": p["moe_intermediate_size"],
            "num_shared_experts": p["num_shared_experts"],
            "intermediate_size": p["intermediate_size"],
            "vocab_size": p["vocab_size"],
            "tie_word_embeddings": p["tie_word_embeddings"],
            "max_position_embeddings": p["max_position_embeddings"],
            "num_moe_layers": p["num_moe_layers"],
            "num_dense_layers": p["num_dense_layers"],
            "num_sparse_attn_layers": p["num_sparse_attn_layers"],
            "num_linear_attn_layers": p["num_linear_attn_layers"],
            "num_mtp": p["num_mtp"],
            "has_vision": p["has_vision"],
            "dtype": args.dtype,
            "dtype_bytes": dtype_bytes,
        },
        "parameters": {
            "total": params["total"],
            "total_formatted": format_params(params["total"]),
            "active_per_token": params["active_per_token"],
            "active_per_token_formatted": format_params(params["active_per_token"]),
            "components": {k: {"params": v["params"], "params_formatted": format_params(v["params"]), "formula": v.get("formula", "")} for k, v in params["components"].items()},
        },
        "kv_cache": {
            "per_token_total_bytes": kv["per_token_total_bytes"],
            "per_token_kib": kv["per_token_total_bytes"] / 1024,
            "mha_baseline_bytes": kv["mha_baseline_bytes"] * p["num_layers"],
            "compression_ratio": kv["compression_ratio"],
            "breakdown": kv["breakdown"],
        },
    }

    # Context length table
    ctx_table = []
    ctx = args.context or p["max_position_embeddings"]
    for sl in STD_SEQ_LENS:
        if sl > p.get("max_position_embeddings", 0) * 2:
            break
        kv_bytes = compute_kv_at_context(kv, sl, args.batch)
        ctx_table.append({"seq_len": sl, "kv_gib": round(kv_bytes / (1024**3), 2), "kv_gb": round(kv_bytes / 1e9, 2)})
    output["kv_cache"]["context_table"] = ctx_table
    output["kv_cache"]["at_max_context"] = {
        "seq_len": ctx,
        "kv_gib": round(compute_kv_at_context(kv, ctx, args.batch) / (1024**3), 2),
        "kv_gb": round(compute_kv_at_context(kv, ctx, args.batch) / 1e9, 2),
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="KV Cache & Parameter Calculator for Ascend NPU Model Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to config.json or directory containing it")
    parser.add_argument("--seq", type=int, default=4096, help="Sequence length (default: 4096)")
    parser.add_argument("--all", action="store_true", help="Show table for multiple standard sequence lengths")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp8", "w8a8", "w4a8"], help="Data type (default: bf16)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--context", type=int, default=None, help="Override max context length (default: from config)")
    parser.add_argument("--verbose", action="store_true", help="Show per-layer and per-component breakdown")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Detect architecture
    arch = detect_architecture(cfg)

    # Extract normalized parameters
    p = extract_params(cfg)

    # Compute parameters
    params = compute_all_params(p, arch)

    # Compute KV cache
    dtype_bytes = get_dtype_bytes(args.dtype)
    kv = compute_kv_cache(p, arch, dtype_bytes, args.batch)

    # Output
    if args.json:
        print_json(args.config, p, arch, params, kv, args)
    else:
        print_report(args.config, p, arch, params, kv, args)


if __name__ == "__main__":
    main()
