#!/usr/bin/env python3
"""Dump logits through the MindSpeed-LLM FSDP2 training forward path."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

try:
    from .alignment_utils import input_sha256, safe_torch_load, to_json_compatible
except ImportError:  # Direct script execution.
    from alignment_utils import input_sha256, safe_torch_load, to_json_compatible  # type: ignore


FRAMEWORK = "mindspeed-llm-fsdp2-training"
EXECUTION_KIND = "mindspeed-llm-fsdp2-training-forward"
FORMAT_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--expected-sequence-length", type=int, default=4096)
    parser.add_argument("--logits-copy-token-chunk-size", type=int, default=32)
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="Override one parsed FSDP2 YAML field; VALUE uses YAML scalar syntax.",
    )
    return parser.parse_args()


def _flatten_dataclasses(root_args: Any) -> types.SimpleNamespace:
    values: dict[str, Any] = {}
    for field in dataclasses.fields(root_args):
        section = getattr(root_args, field.name)
        values.update(vars(section))
    return types.SimpleNamespace(**values)


def _apply_override(root_args: Any, expression: str) -> None:
    if "=" not in expression or "." not in expression.split("=", 1)[0]:
        raise ValueError(f"Invalid --config-override {expression!r}; expected SECTION.FIELD=VALUE")
    dotted_name, raw_value = expression.split("=", 1)
    section_name, field_name = dotted_name.split(".", 1)
    if not hasattr(root_args, section_name):
        raise ValueError(f"Unknown config section: {section_name}")
    section = getattr(root_args, section_name)
    if not hasattr(section, field_name):
        raise ValueError(f"Unknown config field: {dotted_name}")
    setattr(section, field_name, yaml.safe_load(raw_value))


def load_framework_args(args: argparse.Namespace) -> tuple[Any, types.SimpleNamespace, Any]:
    repo_root = Path(os.environ.get("MINDSPEED_LLM_REPO_ROOT", Path.cwd())).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from train_fsdp2 import Arguments, MindSpeedAutoTrainer
    from mindspeed_llm.fsdp2.utils.arguments import fsdp2_parse_args
    from mindspeed_llm.fsdp2.utils.global_vars import set_args

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], str(args.config_file.resolve())]
        root_args = fsdp2_parse_args(Arguments)
    finally:
        sys.argv = old_argv

    for expression in args.config_override:
        _apply_override(root_args, expression)

    model_dir = str(args.model_dir.resolve())
    root_args.model.model_name_or_path = model_dir
    root_args.model.tokenizer_name_or_path = model_dir
    root_args.model.train_from_scratch = False
    root_args.model.init_model_with_meta_device = True
    root_args.training.stage = "pt"
    flat_args = _flatten_dataclasses(root_args)
    set_args(flat_args)
    return root_args, flat_args, MindSpeedAutoTrainer


def _copy_logits_to_cpu_fp32(logits: torch.Tensor, token_chunk_size: int) -> torch.Tensor:
    if token_chunk_size <= 0:
        raise ValueError("--logits-copy-token-chunk-size must be positive")
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B,S,V], got {tuple(logits.shape)}")
    flat = logits.detach().reshape(-1, logits.shape[-1])
    output = torch.empty(flat.shape, dtype=torch.float32, device="cpu")
    for start in range(0, flat.shape[0], token_chunk_size):
        end = min(start + token_chunk_size, flat.shape[0])
        chunk = flat[start:end].to(device="cpu", non_blocking=False).float()
        if not torch.isfinite(chunk).all():
            raise FloatingPointError(f"FSDP2 logits contains NaN/Inf in token chunk {start}:{end}")
        output[start:end].copy_(chunk)
    return output.reshape(logits.shape).contiguous()


def _tensor_sha256(tensor: torch.Tensor, token_chunk_size: int = 32) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    flat = value.reshape(-1, value.shape[-1])
    for start in range(0, flat.shape[0], token_chunk_size):
        digest.update(flat[start : start + token_chunk_size].numpy().tobytes(order="C"))
    return digest.hexdigest()


def _count_fsdp_modules(model: torch.nn.Module) -> tuple[int, list[str]]:
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        FSDPModule = ()  # type: ignore[assignment]

    names = []
    for name, module in model.named_modules():
        is_native_fsdp = bool(FSDPModule) and isinstance(module, FSDPModule)
        mro_names = {base.__name__ for base in type(module).__mro__}
        if is_native_fsdp or "FSDPModule" in mro_names:
            names.append(name or "<root>")
    return len(names), names


def _unwrap_logits(outputs: Any) -> torch.Tensor:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    if hasattr(logits, "full_tensor"):
        logits = logits.full_tensor()
    if not torch.is_tensor(logits):
        raise TypeError(f"Training forward returned non-tensor logits: {type(logits)!r}")
    return logits


def main() -> None:
    args = parse_args()
    root_args, flat_args, auto_trainer_cls = load_framework_args(args)
    auto_trainer_cls._initialize(
        seed=root_args.training.seed,
        deterministic=root_args.training.deterministic,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("FSDP2 certification requires a distributed world size of at least 2")
    if root_args.parallel.fsdp_size != world_size:
        raise RuntimeError(
            f"parallel.fsdp_size={root_args.parallel.fsdp_size} must equal world_size={world_size}"
        )

    from mindspeed_llm.fsdp2.models.model_factory import ModelFactory
    from mindspeed_llm.fsdp2.optim.optimizer import OptimizerFactory
    from mindspeed_llm.fsdp2.train.trainer import Trainer

    model = ModelFactory.create(root_args.model, root_args.parallel)
    weight_load_report = getattr(model.model, "_mindspeed_weight_load_report", {})
    if not weight_load_report:
        raise RuntimeError("FSDP2 WeightLoader did not publish a checkpoint load report")
    missing_parameters = weight_load_report.get("missing_parameters", [])
    if missing_parameters:
        raise RuntimeError(
            "FSDP2 checkpoint load left missing parameters: "
            + ", ".join(missing_parameters[:30])
        )
    fsdp_module_count, fsdp_module_names = _count_fsdp_modules(model)
    if fsdp_module_count <= 0:
        raise RuntimeError("ModelFactory returned a model with no detectable FSDP2-wrapped modules")

    optimizer = OptimizerFactory.create(
        model=model,
        ep_size=root_args.parallel.ep_size,
        lr=root_args.training.lr,
        optimizer_type=root_args.training.optimizer,
        weight_decay=root_args.training.weight_decay,
        betas=(root_args.training.adam_beta1, root_args.training.adam_beta2),
        adam_epsilon=root_args.training.adam_epsilon,
    )
    model.apply_model_hooks(optimizer)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        train_dataloader=(),
        args=root_args.training,
        parallel_args=root_args.parallel,
        optimization_args=root_args.optimization,
        data_args=root_args.data,
        ckpt_manager=None,
        monitor=None,
        tokenizer=None,
    )

    payload = safe_torch_load(args.input_file)
    input_ids = payload["input_ids"].detach().cpu().to(torch.long).contiguous()
    attention_mask = payload["attention_mask"].detach().cpu().to(torch.long).contiguous()
    digest = input_sha256(input_ids, attention_mask)
    if payload.get("input_sha256") != digest:
        raise ValueError("Fixed input SHA-256 mismatch")
    if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
        raise ValueError("input_ids and attention_mask must have the same [B,S] shape")
    if input_ids.shape[-1] != args.expected_sequence_length:
        raise ValueError(
            f"Expected sequence length {args.expected_sequence_length}, got {input_ids.shape[-1]}"
        )
    if not bool(attention_mask.eq(1).all()):
        raise ValueError("Formal FSDP2 alignment input must contain no padding")

    device = torch.accelerator.current_device()
    training_inputs = {
        "input_ids": input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "labels": input_ids.to(device, non_blocking=True),
        "use_cache": False,
        "return_dict": True,
    }
    trainer.model.train()
    with torch.no_grad():
        loss_result = trainer._compute_loss(
            training_inputs,
            return_outputs=True,
            num_items_in_batch=None,
        )
    logits = _unwrap_logits(loss_result.outputs)
    logits_shape = list(logits.shape)
    if logits_shape[:2] != list(input_ids.shape):
        raise ValueError(f"Training logits shape {logits_shape} does not match input {list(input_ids.shape)}")
    loss_value = float(loss_result.loss.detach().float().cpu())
    if not torch.isfinite(torch.tensor(loss_value)):
        raise FloatingPointError(f"Training forward loss is not finite: {loss_value}")

    execution = {
        "kind": EXECUTION_KIND,
        "entrypoint": "train_fsdp2.py",
        "model_factory": "mindspeed_llm.fsdp2.models.model_factory.ModelFactory.create",
        "parallel_engine": f"{type(model).__module__}.{type(model).__name__}",
        "trainer_forward": "mindspeed_llm.fsdp2.train.trainer.Trainer._compute_loss",
        "distributed_initialized": dist.is_initialized(),
        "distributed_backend": dist.get_backend(),
        "world_size": world_size,
        "writer_rank": 0,
        "fsdp_size": root_args.parallel.fsdp_size,
        "ep_size": root_args.parallel.ep_size,
        "ep_fsdp_size": root_args.parallel.ep_fsdp_size,
        "tp_size": root_args.parallel.tp_size,
        "cp_size": root_args.parallel.cp_size,
        "fsdp_implementation": root_args.parallel.fsdp_implementation,
        "fsdp_wrapped_module_count": fsdp_module_count,
        "fsdp_wrapped_modules": fsdp_module_names,
        "model_training": trainer.model.training,
        "no_grad_forward": True,
        "loss_computed_by_trainer": True,
        "loss": loss_value,
        "weight_load_missing_parameter_count": weight_load_report.get(
            "missing_parameter_count", "unavailable"
        ),
    }
    dist.barrier()
    if rank == 0:
        logits_cpu = _copy_logits_to_cpu_fp32(logits, args.logits_copy_token_chunk_size)
        output_payload = {
            "format_version": FORMAT_VERSION,
            "framework": FRAMEWORK,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_sha256": digest,
            "logits_sha256": _tensor_sha256(logits_cpu, args.logits_copy_token_chunk_size),
            "logits": logits_cpu,
            "metadata": {
                "model_dir": str(args.model_dir.resolve()),
                "config_file": str(args.config_file.resolve()),
                "model_id": root_args.model.model_id,
                "execution": execution,
                "parallel": {
                    name: getattr(root_args.parallel, name)
                    for name in ("fsdp_size", "ep_size", "ep_fsdp_size", "tp_size", "cp_size")
                },
                "optimization": vars(root_args.optimization),
                "config_overrides": args.config_override,
                "logits_shape": logits_shape,
                "logits_saved_dtype": str(logits_cpu.dtype),
            },
        }
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_payload, args.output_file)
        print(json.dumps(to_json_compatible({
            "output": str(args.output_file.resolve()),
            "framework": FRAMEWORK,
            "input_sha256": digest,
            "logits_shape": logits_shape,
            "execution": execution,
        }), ensure_ascii=False, indent=2))
    dist.barrier()


if __name__ == "__main__":
    main()
