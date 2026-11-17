#!/usr/bin/env python3
"""Dump full-vocabulary logits from a MindSpeed-LLM/Megatron checkpoint.

The default path uses MindSpeed-LLM's ``GPTModelInfer`` and pipeline
``ForwardStep``.  Models that need a different provider or forward contract can
inject callables without copying this file.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

# Precision alignment starts from an eager numerical baseline. Torch-compiled
# helpers exist in several independent model paths (BDA, grouped-GEMM
# activations, and others), so disabling only one fusion is not sufficient.
# Set ALIGNMENT_ENABLE_TORCH_COMPILE=1 only for a separate optimized-path run.
ALIGNMENT_TORCH_COMPILE_ENABLED = (
    os.environ.get("ALIGNMENT_ENABLE_TORCH_COMPILE", "0") == "1"
)
ALIGNMENT_TOOL_VERSION = "2026.07.21.2"
if not ALIGNMENT_TORCH_COMPILE_ENABLED:
    # These variables must be set before importing torch and model modules so
    # @torch.compile decorators resolve to their eager behavior.
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

if not ALIGNMENT_TORCH_COMPILE_ENABLED:
    # Also cover environments whose site initialization imported torch before
    # this script had a chance to set the environment variables.
    import torch._dynamo

    torch._dynamo.config.disable = True

    def _alignment_eager_compile(model=None, *args, **kwargs):
        """Return the original callable for both torch.compile invocation forms."""
        del args, kwargs
        if model is None:
            return lambda function: function
        return model

    # Environment/config switches can be observed too late when torch or
    # torch_npu is imported by site initialization. Replacing the decorator
    # before importing Megatron/MindSpeed makes the eager baseline deterministic.
    torch.compile = _alignment_eager_compile
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "compile"):
        torch.compiler.compile = _alignment_eager_compile

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module as import_layer_spec
from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.inference.text_generation.generation import _build_attention_mask_and_position_ids
from megatron.training import get_args, initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from mindspeed_llm.tasks.inference.module import GPTModelInfer, MegatronModuleForCausalLM

try:
    from .alignment_utils import (
        convert_logits_dtype,
        load_fixed_input,
        normalize_logits_layout,
        parse_json_object,
        select_output_tensor,
        to_json_compatible,
    )
except ImportError:  # Direct script execution.
    from alignment_utils import (  # type: ignore
        convert_logits_dtype,
        load_fixed_input,
        normalize_logits_layout,
        parse_json_object,
        select_output_tensor,
        to_json_compatible,
    )


CallableType = Callable[..., Any]


def add_alignment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("precision alignment")
    group.add_argument("--input-ids-path", required=True)
    group.add_argument("--logits-output-path", required=True)
    group.add_argument(
        "--target-vocab-size",
        "--hf-vocab-size",
        dest="target_vocab_size",
        type=int,
        default=None,
        help=(
            "Crop padded vocabulary logits to this size; defaults to a positive "
            "--vocab-size, otherwise keeps the returned vocabulary."
        ),
    )
    group.add_argument(
        "--alignment-save-dtype",
        "--dump-logits-dtype",
        dest="alignment_save_dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    group.add_argument(
        "--alignment-logits-layout",
        "--logits-layout",
        dest="alignment_logits_layout",
        choices=("auto", "bsv", "sbv"),
        default="auto",
        help="Layout returned by the forward function before normalization.",
    )
    group.add_argument(
        "--alignment-output-selector",
        default="auto",
        help="Tensor selector: auto, attr:NAME, key:NAME, or index:N.",
    )
    group.add_argument(
        "--alignment-model-provider",
        default=None,
        help="Optional import path MODULE:CALLABLE replacing the default GPT provider.",
    )
    group.add_argument(
        "--alignment-forward-function",
        default=None,
        help=(
            "Optional MODULE:CALLABLE called as "
            "fn(model, input_ids, attention_mask, megatron_args) on every rank."
        ),
    )
    group.add_argument(
        "--alignment-framework",
        default="mindspeed-llm",
        help="Framework identifier written into the output payload.",
    )
    group.add_argument(
        "--alignment-metadata-json",
        default=None,
        help="Extra JSON metadata inline or as @/path/to/file.json.",
    )
    group.add_argument("--alignment-allow-sequence-parallel", action="store_true")
    group.add_argument("--alignment-allow-context-parallel", action="store_true")
    group.add_argument("--alignment-allow-mtp", action="store_true")
    group.add_argument("--alignment-allow-kv-cache", action="store_true")
    group.add_argument("--alignment-allow-virtual-pipeline", action="store_true")
    return parser


def import_callable(spec: Optional[str]) -> Optional[CallableType]:
    if not spec:
        return None
    if ":" in spec:
        module_name, attribute = spec.split(":", 1)
    else:
        module_name, separator, attribute = spec.rpartition(".")
        if not separator:
            raise ValueError(
                f"Invalid callable {spec!r}; use MODULE:CALLABLE or MODULE.CALLABLE."
            )
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"Imported object {spec!r} is not callable.")
    return value


def default_model_provider(pre_process: bool = True, post_process: bool = True):
    """Build the default MCore inference model using the repository contract."""
    args = get_args()
    if not args.use_mcore_models:
        raise ValueError(
            "The reusable dumper supports MCore models only; pass --use-mcore-models."
        )

    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_layer_spec(args.spec)
    elif args.transformer_impl == "transformer_engine":
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
        )
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
        )

    model_kwargs = {
        "config": config,
        "transformer_layer_spec": transformer_layer_spec,
        "vocab_size": args.padded_vocab_size,
        "max_sequence_length": args.max_position_embeddings,
        "pre_process": pre_process,
        "post_process": post_process,
        "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
        # Gather tensor-parallel vocabulary shards before the writer saves logits.
        "parallel_output": False,
        "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
        "position_embedding_type": args.position_embedding_type,
        "rotary_percent": args.rotary_percent,
        "rotary_base": args.rotary_base,
        "seq_len_interpolation_factor": args.rotary_seq_len_interpolation_factor,
        "rope_scaling": args.use_rope_scaling,
    }
    constructor_parameters = inspect.signature(GPTModelInfer).parameters
    accepts_extra_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in constructor_parameters.values()
    )
    if not accepts_extra_kwargs:
        model_kwargs = {
            name: value
            for name, value in model_kwargs.items()
            if name in constructor_parameters
        }
    return GPTModelInfer(**model_kwargs)


def _world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _int_arg(args: argparse.Namespace, name: str, default: int = 1) -> int:
    value = getattr(args, name, default)
    return default if value is None else int(value)


def _parallel_rank(function_name: str) -> int:
    getter = getattr(mpu, function_name, None)
    return int(getter()) if callable(getter) else 0


def apply_inference_baseline(args: argparse.Namespace) -> None:
    """Normalize training-oriented Megatron defaults before model construction."""
    required_values = {
        # Backward-only; otherwise ColumnParallelLinear requires a CUDA APEX
        # weight-gradient extension even during an Ascend inference run.
        "gradient_accumulation_fusion": False,
        # The fused BDA helper enters torch.compile/Inductor. The direct
        # alignment baseline uses the eager equivalent, avoiding CUDA-specific
        # Inductor device discovery and giving a more comparable operation path.
        "bias_dropout_fusion": False,
    }
    overrides = {}
    for name, effective_value in required_values.items():
        requested_value = getattr(args, name, None)
        if requested_value != effective_value:
            overrides[name] = {
                "requested": requested_value,
                "effective": effective_value,
            }
        setattr(args, name, effective_value)
    args.alignment_runtime_overrides = overrides
    if overrides and _rank() == 0:
        print(
            "[ALIGNMENT] Applied inference baseline overrides: "
            + json.dumps(overrides, ensure_ascii=False, sort_keys=True)
        )


def validate_baseline(args: argparse.Namespace) -> None:
    violations = []
    if getattr(args, "gradient_accumulation_fusion", False):
        violations.append("training-only gradient accumulation fusion is enabled")
    if getattr(args, "sequence_parallel", False) and not args.alignment_allow_sequence_parallel:
        violations.append("sequence parallel is enabled")
    if _int_arg(args, "context_parallel_size") != 1 and not args.alignment_allow_context_parallel:
        violations.append("context parallel size is not 1")
    if getattr(args, "mtp_num_layers", None) not in (None, 0) and not args.alignment_allow_mtp:
        violations.append("MTP is enabled")
    if getattr(args, "use_kv_cache", False) and not args.alignment_allow_kv_cache:
        violations.append("KV cache is enabled")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        if not args.alignment_allow_virtual_pipeline:
            violations.append("virtual pipeline parallelism is enabled")
        elif not args.alignment_forward_function:
            violations.append(
                "virtual pipeline requires a custom --alignment-forward-function"
            )
    if violations:
        rendered = "; ".join(violations)
        raise ValueError(
            "Unsupported precision-alignment baseline configuration: "
            f"{rendered}. Disable it, or explicitly allow it after supplying a "
            "forward implementation that preserves the output contract."
        )


def run_default_forward(model: Any, input_ids: torch.Tensor) -> Any:
    """Run the repository ForwardStep while tolerating supported API variants."""
    signature = inspect.signature(ForwardStep)
    parameters = signature.parameters
    inference_context = None

    if "max_sequence_length" in parameters:
        forward_step = ForwardStep(
            model,
            max_batch_size=int(input_ids.shape[0]),
            max_sequence_length=int(input_ids.shape[1]),
        )
    elif "inference_context" in parameters:
        inference_context = StaticInferenceContext(
            int(input_ids.shape[0]),
            int(input_ids.shape[1]),
        )
        forward_step = ForwardStep(model, inference_context)
    else:
        raise RuntimeError(
            "Unsupported ForwardStep API. Provide --alignment-forward-function "
            "for this MindSpeed-LLM/Megatron version."
        )

    causal_mask, position_ids = _build_attention_mask_and_position_ids(input_ids)
    call_parameters = inspect.signature(forward_step.__call__).parameters
    call_kwargs = {}
    if "recv_buffer_seq_length" in call_parameters:
        call_kwargs["recv_buffer_seq_length"] = int(input_ids.shape[1])
    if "inference_context" in call_parameters:
        if inference_context is None:
            inference_context = StaticInferenceContext(
                int(input_ids.shape[0]),
                int(input_ids.shape[1]),
            )
        call_kwargs["inference_context"] = inference_context
    return forward_step(input_ids, position_ids, causal_mask, **call_kwargs)


def is_writer_rank() -> bool:
    return (
        mpu.is_pipeline_last_stage()
        and _parallel_rank("get_tensor_model_parallel_rank") == 0
        and _parallel_rank("get_data_parallel_rank") == 0
        and _parallel_rank("get_context_parallel_rank") == 0
    )


def collect_parallel_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "world_size": _world_size(),
        "tp": _int_arg(args, "tensor_model_parallel_size"),
        "pp": _int_arg(args, "pipeline_model_parallel_size"),
        "ep": _int_arg(args, "expert_model_parallel_size"),
        "etp": _int_arg(args, "expert_tensor_parallel_size"),
        "cp": _int_arg(args, "context_parallel_size"),
        "dp": _int_arg(args, "data_parallel_size"),
        "sequence_parallel": bool(getattr(args, "sequence_parallel", False)),
        "vpp": getattr(args, "virtual_pipeline_model_parallel_size", None),
        "mtp_num_layers": getattr(args, "mtp_num_layers", None),
    }


def collect_execution_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Record common precision-relevant flags without changing model behavior."""
    field_names = (
        "bf16",
        "fp16",
        "params_dtype",
        "transformer_impl",
        "normalization",
        "norm_epsilon",
        "fp32_residual_connection",
        "gradient_accumulation_fusion",
        "bias_dropout_fusion",
        "attention_softmax_in_fp32",
        "position_embedding_type",
        "rotary_base",
        "rotary_percent",
        "apply_rope_fusion",
        "use_flash_attn",
        "use_sparse_flash_attn",
        "multi_latent_attention",
        "enable_mla_absorb",
        "enable_dsa_indexer",
        "rmsnorm_weight_in_fp32",
        "init_norm_weight_in_fp32",
        "use_kv_cache",
    )
    return {
        name: to_json_compatible(getattr(args, name))
        for name in field_names
        if hasattr(args, name)
    }


def dump_logits() -> None:
    args = get_args()
    validate_baseline(args)

    input_ids_cpu, attention_mask_cpu, digest, _ = load_fixed_input(
        args.input_ids_path,
        require_right_padding=True,
    )
    if not getattr(args, "load", None):
        raise ValueError("--load is required; precision alignment must not use random weights.")
    if int(input_ids_cpu.shape[1]) > int(args.max_position_embeddings):
        raise ValueError(
            f"Input sequence length {input_ids_cpu.shape[1]} exceeds "
            f"max_position_embeddings={args.max_position_embeddings}."
        )
    if int(input_ids_cpu.max().item()) >= int(args.padded_vocab_size):
        raise ValueError(
            f"input_ids contains token ID {int(input_ids_cpu.max().item())}, but "
            f"padded_vocab_size={args.padded_vocab_size}."
        )

    model_provider = import_callable(args.alignment_model_provider) or default_model_provider
    model = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load,
    )
    model.eval()

    device = torch.device("npu", torch.npu.current_device())
    input_ids = input_ids_cpu.to(device=device, dtype=torch.long)
    attention_mask = attention_mask_cpu.to(device=device, dtype=torch.bool)

    custom_forward = import_callable(args.alignment_forward_function)
    with torch.inference_mode():
        if custom_forward is None:
            raw_output = run_default_forward(model, input_ids)
        else:
            raw_output = custom_forward(model, input_ids, attention_mask, args)

    if is_writer_rank():
        raw_logits = select_output_tensor(raw_output, args.alignment_output_selector)
        selected_output_shape = list(raw_logits.shape)
        logits = normalize_logits_layout(
            raw_logits,
            input_ids_cpu,
            requested_layout=args.alignment_logits_layout,
        )
        normalized_shape = list(logits.shape)
        configured_vocab_size = getattr(args, "vocab_size", None)
        if configured_vocab_size is not None and int(configured_vocab_size) <= 0:
            configured_vocab_size = None
        target_vocab_size = (
            args.target_vocab_size
            if args.target_vocab_size is not None
            else configured_vocab_size or int(logits.shape[-1])
        )
        target_vocab_size = int(target_vocab_size)
        if target_vocab_size <= 0 or target_vocab_size > int(logits.shape[-1]):
            raise ValueError(
                f"target_vocab_size={target_vocab_size} is outside the returned "
                f"vocabulary dimension {logits.shape[-1]}."
            )
        logits = logits[..., :target_vocab_size].float()
        if not torch.isfinite(logits).all():
            nan_count = int(torch.isnan(logits).sum().item())
            inf_count = int(torch.isinf(logits).sum().item())
            raise FloatingPointError(
                f"MindSpeed-LLM logits contains non-finite values: "
                f"nan_count={nan_count}, inf_count={inf_count}."
            )

        saved_logits = convert_logits_dtype(logits.cpu(), args.alignment_save_dtype)
        metadata = {
            "checkpoint": str(Path(args.load).resolve()),
            "raw_logits_shape": selected_output_shape,
            "normalized_logits_shape": normalized_shape,
            "logits_shape": list(saved_logits.shape),
            "raw_padded_vocab_size": int(normalized_shape[-1]),
            "saved_vocab_size": int(saved_logits.shape[-1]),
            "saved_dtype": str(saved_logits.dtype),
            "model_provider": args.alignment_model_provider or "default:GPTModelInfer",
            "forward_function": args.alignment_forward_function or "default:ForwardStep",
            "output_selector": args.alignment_output_selector,
            "forward_attention_mask": (
                "custom forward received the saved boolean attention mask"
                if custom_forward is not None
                else "default ForwardStep uses its internal causal mask; only "
                "right-padded valid prefixes are compared"
            ),
            "parallel": collect_parallel_metadata(args),
            "execution": collect_execution_metadata(args),
            "torch_compile": {
                "tool_version": ALIGNMENT_TOOL_VERSION,
                "alignment_opt_in": ALIGNMENT_TORCH_COMPILE_ENABLED,
                "compile_guard": (
                    "disabled by environment, Dynamo config, and eager decorator"
                    if not ALIGNMENT_TORCH_COMPILE_ENABLED
                    else "explicitly enabled"
                ),
                "TORCH_COMPILE_DISABLE": os.environ.get("TORCH_COMPILE_DISABLE"),
                "TORCHDYNAMO_DISABLE": os.environ.get("TORCHDYNAMO_DISABLE"),
            },
            "runtime_overrides": getattr(args, "alignment_runtime_overrides", {}),
            "writer_rank": _rank(),
        }
        extra_metadata = parse_json_object(args.alignment_metadata_json)
        if extra_metadata:
            metadata["user_metadata"] = extra_metadata

        output_payload = {
            "framework": args.alignment_framework,
            "format_version": 1,
            "input_ids": input_ids_cpu,
            "attention_mask": attention_mask_cpu,
            "valid_lengths": attention_mask_cpu.sum(dim=-1).to(torch.long),
            "input_sha256": digest,
            "logits": saved_logits,
            "metadata": metadata,
        }
        output_path = Path(args.logits_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_payload, output_path)
        print(
            json.dumps(
                {
                    "output": str(output_path.resolve()),
                    "framework": args.alignment_framework,
                    "input_sha256": digest,
                    "logits_shape": list(saved_logits.shape),
                    "valid_token_count": int(attention_mask_cpu.sum().item()),
                    "writer_rank": _rank(),
                    "metadata": to_json_compatible(metadata),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    _barrier()


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    initialize_megatron(
        extra_args_provider=add_alignment_args,
        args_defaults={
            "tokenizer_type": "NullTokenizer",
            "no_load_rng": True,
            "gradient_accumulation_fusion": False,
            "bias_dropout_fusion": False,
        },
    )
    args = get_args()
    apply_inference_baseline(args)
    if _rank() == 0:
        print(
            f"[ALIGNMENT] tool_version={ALIGNMENT_TOOL_VERSION}, "
            f"torch_compile_enabled={ALIGNMENT_TORCH_COMPILE_ENABLED}"
        )
    if getattr(args, "model_type", None) is None:
        args.model_type = ModelType.encoder_or_decoder
    dump_logits()


if __name__ == "__main__":
    main()
