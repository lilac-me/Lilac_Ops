#!/usr/bin/env python3
"""Dump full-vocabulary Transformers logits under the shared alignment contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from .alignment_utils import (
        FORMAT_VERSION,
        load_fixed_input,
        normalize_logits_layout,
        select_output_tensor,
        to_json_compatible,
    )
except ImportError:  # Direct script execution.
    from alignment_utils import (  # type: ignore
        FORMAT_VERSION,
        load_fixed_input,
        normalize_logits_layout,
        select_output_tensor,
        to_json_compatible,
    )


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
TRANSFORMERS_ALIGNMENT_TOOL_VERSION = "2026.08.05.1"


@dataclass(frozen=True)
class WeightManifest:
    keys: frozenset[str] | None
    missing_files: tuple[str, ...]
    checkpoint_format: str


class TransformersAlignmentAdapter:
    """Override only model-specific validation, loading, forward, or metadata."""

    name = "generic-causal-lm"
    framework = "transformers"
    force_module_output_finite_checks = False

    def load_config(self, args: argparse.Namespace) -> Any:
        return AutoConfig.from_pretrained(
            args.model_dir,
            trust_remote_code=args.trust_remote_code,
            local_files_only=not args.allow_remote_files,
        )

    def validate_config(self, config: Any) -> None:
        del config

    def validate_weight_manifest(
        self,
        manifest: WeightManifest,
        config: Any,
        args: argparse.Namespace,
    ) -> None:
        del config, args
        if manifest.missing_files:
            raise RuntimeError(
                "Checkpoint shard files are missing "
                f"({len(manifest.missing_files)}): {list(manifest.missing_files[:10])}"
            )

    def load_model(
        self,
        config: Any,
        args: argparse.Namespace,
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        load_kwargs: dict[str, Any] = {
            "config": config,
            "dtype": DTYPES[args.dtype],
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": not args.allow_remote_files,
            "low_cpu_mem_usage": True,
            "output_loading_info": True,
        }
        if args.attn_implementation:
            load_kwargs["attn_implementation"] = args.attn_implementation
        device_map = resolve_device_map(args)
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        loaded = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, loading_info = loaded
        else:
            model, loading_info = loaded, {}
        if not isinstance(loading_info, dict):
            raise TypeError(f"Unexpected loading_info type: {type(loading_info)!r}")
        return model, loading_info

    def validate_runtime(self, model: torch.nn.Module, config: Any) -> None:
        del model, config

    def input_device(self, model: torch.nn.Module, fallback: str) -> torch.device:
        device = model.get_input_embeddings().weight.device
        if device.type == "meta":
            return torch.device(fallback)
        return device

    def forward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        args: argparse.Namespace,
    ) -> Any:
        del args
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

    def should_track_module(self, name: str, module: torch.nn.Module) -> bool:
        del name
        return not any(module.children())

    def collect_metadata(
        self,
        model: torch.nn.Module,
        config: Any,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        del model, config, args
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Accelerate device map (default: "auto"); use "none" for model.to(device).',
    )
    parser.add_argument(
        "--device-map-json",
        help="Explicit JSON object mapping module names to device indexes; overrides --device-map.",
    )
    parser.add_argument("--device", default="npu:0", help="Used when --device-map=none.")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument(
        "--dequantize-fp8",
        action="store_true",
        help="Request checkpoint FP8 dequantization before the forward pass.",
    )
    parser.add_argument(
        "--allow-missing-weights",
        action="store_true",
        help="Smoke-test only; the saved artifact is marked alignment_eligible=false.",
    )
    parser.add_argument("--diagnose-nonfinite", action="store_true")
    parser.add_argument("--synchronize-module-outputs", action="store_true")
    parser.add_argument(
        "--logits-copy-token-chunk-size",
        type=int,
        default=64,
        help=(
            "Token chunk size for copying logits to CPU in their source dtype before "
            "CPU FP32 conversion. This avoids one full-vocabulary FP32 conversion on "
            "the accelerator."
        ),
    )
    parser.add_argument(
        "--logits-output-selector",
        default="auto",
        help="auto, attr:NAME, key:NAME, or index:N.",
    )
    parser.add_argument(
        "--logits-layout",
        choices=["auto", "bsv", "sbv"],
        default="bsv",
    )
    parser.add_argument(
        "--alignment-adapter",
        default=None,
        help="Optional MODULE:OBJECT returning a TransformersAlignmentAdapter.",
    )
    return parser.parse_args()


def resolve_device_map(args: argparse.Namespace) -> str | dict[str, Any] | None:
    if args.device_map_json:
        device_map = json.loads(args.device_map_json)
        if not isinstance(device_map, dict) or not device_map:
            raise ValueError("--device-map-json must be a non-empty JSON object.")
        return device_map
    return None if args.device_map.lower() == "none" else args.device_map


def import_adapter(spec: str) -> TransformersAlignmentAdapter:
    module_name, separator, object_name = spec.partition(":")
    if not separator or not module_name or not object_name:
        raise ValueError("--alignment-adapter must use MODULE:OBJECT syntax.")
    value = getattr(importlib.import_module(module_name), object_name)
    if isinstance(value, type):
        value = value()
    elif callable(value) and not hasattr(value, "load_config"):
        value = value()
    required_methods = (
        "load_config",
        "validate_config",
        "validate_weight_manifest",
        "load_model",
        "validate_runtime",
        "input_device",
        "forward",
        "should_track_module",
        "collect_metadata",
    )
    missing_methods = [name for name in required_methods if not callable(getattr(value, name, None))]
    if missing_methods:
        raise TypeError(
            "The alignment adapter does not implement the required contract: "
            f"missing={missing_methods}, type={type(value)!r}."
        )
    return value


def load_weight_manifest(model_dir: str) -> WeightManifest:
    model_path = Path(model_dir)
    index_candidates = (
        ("model.safetensors.index.json", "safetensors-index"),
        ("pytorch_model.bin.index.json", "pytorch-bin-index"),
    )
    for filename, checkpoint_format in index_candidates:
        index_path = model_path / filename
        if not index_path.is_file():
            continue
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")
        missing_files = tuple(
            sorted(name for name in set(weight_map.values()) if not (model_path / name).is_file())
        )
        return WeightManifest(
            keys=frozenset(str(key) for key in weight_map),
            missing_files=missing_files,
            checkpoint_format=checkpoint_format,
        )

    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if safetensor_files:
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError(
                "safetensors is required to inspect a non-sharded safetensors checkpoint."
            ) from error
        keys: set[str] = set()
        for filename in safetensor_files:
            with safe_open(str(filename), framework="pt", device="cpu") as handle:
                keys.update(handle.keys())
        return WeightManifest(frozenset(keys), (), "safetensors")

    bin_files = sorted(model_path.glob("pytorch_model*.bin"))
    if bin_files:
        return WeightManifest(None, (), "pytorch-bin")
    return WeightManifest(None, ("<no supported checkpoint weights found>",), "unknown")


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def floating_tensors(value: Any, prefix: str = "output") -> list[tuple[str, torch.Tensor]]:
    tensors = []
    if torch.is_tensor(value):
        if value.is_floating_point():
            tensors.append((prefix, value))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            tensors.extend(floating_tensors(item, f"{prefix}[{index}]"))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            tensors.extend(floating_tensors(item, f"{prefix}.{key}"))
    return tensors


def synchronize_tensor_device(tensor: torch.Tensor) -> None:
    if tensor.device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(tensor.device)
    elif tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def copy_logits_to_cpu_fp32(
    logits: torch.Tensor,
    token_chunk_size: int,
) -> torch.Tensor:
    """Copy [B,S,V] logits to CPU by token chunks, then convert each chunk to FP32."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with rank 3, got shape={tuple(logits.shape)}.")
    if token_chunk_size <= 0:
        raise ValueError("--logits-copy-token-chunk-size must be positive.")

    cpu_logits = torch.empty(tuple(logits.shape), dtype=torch.float32, device="cpu")
    sequence_length = logits.shape[1]
    for start in range(0, sequence_length, token_chunk_size):
        end = min(start + token_chunk_size, sequence_length)
        source_chunk = logits[:, start:end].detach().contiguous()
        synchronize_tensor_device(source_chunk)
        # Keep the device-to-host transfer in the source dtype. BF16 values are
        # exactly representable in FP32, so the following conversion is lossless
        # and, critically, runs on CPU rather than allocating a full FP32 tensor
        # on the accelerator.
        cpu_chunk = source_chunk.cpu().float()
        cpu_logits[:, start:end].copy_(cpu_chunk)
    return cpu_logits.contiguous()


def register_output_synchronization(
    model: torch.nn.Module,
    adapter: TransformersAlignmentAdapter,
) -> list[Any]:
    handles = []

    def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        tensors = floating_tensors(output)
        if tensors:
            synchronize_tensor_device(tensors[0][1])

    for name, module in model.named_modules():
        if adapter.should_track_module(name, module):
            handles.append(module.register_forward_hook(hook))
    return handles


def register_nonfinite_diagnostics(
    model: torch.nn.Module,
    adapter: TransformersAlignmentAdapter,
) -> tuple[list[Any], dict[str, Any]]:
    state: dict[str, Any] = {"first": None}
    handles = []

    def make_hook(module_name: str):
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if state["first"] is not None:
                return
            for tensor_name, tensor in floating_tensors(output):
                finite = torch.isfinite(tensor)
                if bool(finite.all()):
                    continue
                finite_values = tensor[finite]
                state["first"] = {
                    "module": module_name,
                    "tensor": tensor_name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "nan_count": int(torch.isnan(tensor).sum().item()),
                    "inf_count": int(torch.isinf(tensor).sum().item()),
                    "finite_min": (
                        finite_values.float().min().item() if finite_values.numel() else None
                    ),
                    "finite_max": (
                        finite_values.float().max().item() if finite_values.numel() else None
                    ),
                }
                print(
                    "[NONFINITE] First invalid module output:\n"
                    + json.dumps(state["first"], ensure_ascii=False, indent=2),
                    flush=True,
                )
                return

        return hook

    for name, module in model.named_modules():
        if adapter.should_track_module(name, module):
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles, state


def module_output_finite_checks_enabled(
    adapter: TransformersAlignmentAdapter,
    args: argparse.Namespace,
) -> tuple[bool, bool]:
    """Return (enabled, forced_by_adapter) for scheduling-sensitive finite checks."""
    forced_by_adapter = bool(
        getattr(adapter, "force_module_output_finite_checks", False)
    )
    return bool(args.diagnose_nonfinite or forced_by_adapter), forced_by_adapter


def print_device_map_summary(model: torch.nn.Module) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        print("[INFO] hf_device_map: not present (model.to device path)", flush=True)
        return
    counts: dict[str, int] = {}
    for device in device_map.values():
        key = str(device)
        counts[key] = counts.get(key, 0) + 1
    print(
        "[INFO] hf_device_map module counts:\n"
        + json.dumps(counts, ensure_ascii=False, indent=2),
        flush=True,
    )


def nonfinite_summary(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    invalid_token_mask = (~finite).any(dim=-1)
    token_positions = torch.nonzero(invalid_token_mask, as_tuple=False)[:5]
    first_invalid_positions = []
    for batch_index, sequence_index in token_positions.tolist():
        vocab_indices = torch.nonzero(
            ~finite[batch_index, sequence_index],
            as_tuple=False,
        )[:5].flatten()
        first_invalid_positions.extend(
            [batch_index, sequence_index, int(vocab_index)]
            for vocab_index in vocab_indices.tolist()
        )
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "inf_count": int(torch.isinf(tensor).sum().item()),
        "invalid_token_count": int(invalid_token_mask.sum().item()),
        "first_invalid_positions": first_invalid_positions[:10],
    }


def _loading_list(loading_info: dict[str, Any], key: str) -> list[Any]:
    value = loading_info.get(key, [])
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError(f"loading_info[{key!r}] must be a collection, got {type(value)!r}")
    return list(value)


def main(adapter: TransformersAlignmentAdapter | None = None) -> None:
    args = parse_args()
    if args.alignment_adapter:
        adapter = import_adapter(args.alignment_adapter)
    if adapter is None:
        adapter = TransformersAlignmentAdapter()
    adapter_name = str(getattr(adapter, "name", type(adapter).__name__))
    print(
        f"[ALIGNMENT] Transformers tool version: {TRANSFORMERS_ALIGNMENT_TOOL_VERSION}",
        flush=True,
    )

    input_ids, attention_mask, digest, _ = load_fixed_input(
        args.input_file,
        require_right_padding=True,
    )
    config = adapter.load_config(args)
    if args.dequantize_fp8:
        quantization_config = getattr(config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            quantization_config["dequantize"] = True
        elif quantization_config is not None and hasattr(quantization_config, "dequantize"):
            quantization_config.dequantize = True
    adapter.validate_config(config)
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is not None and int(input_ids.max()) >= int(vocab_size):
        raise ValueError("input_ids contains a token outside config.vocab_size.")

    manifest = load_weight_manifest(args.model_dir)
    adapter.validate_weight_manifest(manifest, config, args)
    model, loading_info = adapter.load_model(config, args)
    missing_keys = _loading_list(loading_info, "missing_keys")
    unexpected_keys = _loading_list(loading_info, "unexpected_keys")
    mismatched_keys = _loading_list(loading_info, "mismatched_keys")
    error_messages = _loading_list(loading_info, "error_msgs")
    if error_messages:
        raise RuntimeError(f"Model loading reported errors: {error_messages[:10]}")
    if mismatched_keys:
        raise RuntimeError(f"Model loading reported mismatched keys: {mismatched_keys[:30]}")
    if missing_keys and not args.allow_missing_weights:
        raise RuntimeError(f"Model loading reported missing keys: {missing_keys[:30]}")
    if missing_keys:
        print(
            "[WARN] Model has missing weights; output is smoke-test-only and not alignment eligible.",
            flush=True,
        )

    resolved_device_map = resolve_device_map(args)
    if resolved_device_map is None:
        model.to(torch.device(args.device))
    model.eval()
    adapter.validate_runtime(model, config)
    print_device_map_summary(model)

    device = adapter.input_device(model, args.device)
    module_output_finite_checks, adapter_forced_finite_checks = (
        module_output_finite_checks_enabled(adapter, args)
    )
    synchronization_handles = []
    if args.synchronize_module_outputs:
        print(
            "[INFO] Synchronizing selected module outputs on their accelerator devices.",
            flush=True,
        )
        synchronization_handles = register_output_synchronization(model, adapter)
    diagnostic_handles = []
    diagnostic_state: dict[str, Any] = {"first": None}
    if module_output_finite_checks:
        if args.diagnose_nonfinite:
            message = (
                "[WARN] --diagnose-nonfinite performs per-module synchronization and may "
                "change accelerator scheduling."
            )
        else:
            message = (
                f"[INFO] Adapter {adapter_name} requires per-module finite-output checks "
                "as a precision-mode scheduling barrier."
            )
        print(message, flush=True)
        diagnostic_handles, diagnostic_state = register_nonfinite_diagnostics(model, adapter)
    try:
        with torch.inference_mode():
            output = adapter.forward(
                model,
                input_ids.to(device),
                attention_mask.to(device),
                args,
            )
    finally:
        for handle in synchronization_handles:
            handle.remove()
        for handle in diagnostic_handles:
            handle.remove()

    raw_logits = select_output_tensor(output, args.logits_output_selector)
    logits = normalize_logits_layout(raw_logits, input_ids, args.logits_layout)
    logits = copy_logits_to_cpu_fp32(logits, args.logits_copy_token_chunk_size)
    if vocab_size is not None and logits.shape[-1] != int(vocab_size):
        raise RuntimeError(f"Expected vocab={vocab_size}, got {logits.shape[-1]}")
    if not torch.isfinite(logits).all():
        summary = nonfinite_summary(logits)
        print(
            "[NONFINITE] Final logits summary:\n"
            + json.dumps(summary, ensure_ascii=False, indent=2),
            flush=True,
        )
        first_module = diagnostic_state.get("first")
        if first_module is None and not module_output_finite_checks:
            hint = " Re-run with --diagnose-nonfinite to locate the first invalid module."
        elif first_module is None:
            hint = " No tracked module was non-finite; inspect causal-LM postprocessing."
        else:
            hint = f" First invalid module: {first_module['module']}."
        raise FloatingPointError("Transformers logits contains NaN or Inf." + hint)

    valid_lengths = attention_mask.sum(dim=-1)
    last_valid = torch.stack(
        [logits[row, int(length) - 1] for row, length in enumerate(valid_lengths.tolist())]
    )
    logits_digest = tensor_sha256(logits)
    alignment_eligible = (
        not args.allow_missing_weights
        and not missing_keys
        and not mismatched_keys
        and not error_messages
    )
    metadata = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "model_type": getattr(config, "model_type", None),
        "adapter": adapter_name,
        "alignment_tool_version": TRANSFORMERS_ALIGNMENT_TOOL_VERSION,
        "compute_dtype": args.dtype,
        "saved_dtype": str(logits.dtype),
        "attn_implementation": args.attn_implementation,
        "device_map": resolved_device_map,
        "module_output_finite_checks": module_output_finite_checks,
        "module_output_finite_checks_forced_by_adapter": adapter_forced_finite_checks,
        "logits_copy": {
            "path": "token-chunked-source-dtype-d2h-then-cpu-fp32",
            "token_chunk_size": args.logits_copy_token_chunk_size,
        },
        "logits_shape": list(logits.shape),
        "checkpoint_format": manifest.checkpoint_format,
        "alignment_eligible": alignment_eligible,
        "missing_keys": missing_keys,
        "unexpected_keys_count": len(unexpected_keys),
        "unexpected_keys": unexpected_keys[:100],
        "mismatched_keys": mismatched_keys,
    }
    metadata.update(to_json_compatible(adapter.collect_metadata(model, config, args)))
    payload = {
        "format_version": FORMAT_VERSION,
        "framework": str(getattr(adapter, "framework", "transformers")),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_sha256": digest,
        "logits_sha256": logits_digest,
        "logits": logits,
        "metadata": metadata,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "input_sha256": digest,
                "logits_sha256": logits_digest,
                "logits_shape": list(logits.shape),
                "valid_token_count": int(attention_mask.sum()),
                "alignment_eligible": alignment_eligible,
                "module_output_finite_checks": module_output_finite_checks,
                "logits_copy_token_chunk_size": args.logits_copy_token_chunk_size,
                "last_valid_token_top10": last_valid.topk(
                    min(10, last_valid.shape[-1]), dim=-1
                ).indices.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
