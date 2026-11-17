#!/usr/bin/env python3
"""Run native-versus-FSDP2 layerwise full-vocabulary logits alignment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


STATUS_FILENAME = "alignment_status.json"


class AlignmentStageError(RuntimeError):
    """A classified orchestration failure that can be diagnosed without reading source."""

    def __init__(self, stage: str, message: str, *, log: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.log = log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--candidate-world-size", type=int, default=2)
    parser.add_argument("--candidate-master-port", type=int, default=29500)
    parser.add_argument("--candidate-config-override", action="append", default=[])
    parser.add_argument("--checkpoint-config-override", action="append", default=[])
    parser.add_argument("--candidate-pythonpath", action="append", default=[])
    parser.add_argument("--reference-python", default=sys.executable)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument("--layer-counts", default="1,2,4,full")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--device-map",
        default="none",
        help='Accelerate device map for the Transformers reference; use "none" for model.to(device).',
    )
    parser.add_argument(
        "--device-map-json",
        help="Explicit module-to-device JSON for the Transformers reference; overrides --device-map.",
    )
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument(
        "--accelerator-cleanup-delay",
        type=float,
        default=0,
        help="Seconds to wait after each logits process so accelerator runtime memory is released.",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--exact-sequence-length", type=int, default=4096)
    parser.add_argument(
        "--input-format",
        choices=("auto", "chat", "raw"),
        default="auto",
        help="Tokenize prompts as chat messages or raw text. Auto uses tokenizer_config.json.",
    )
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--model-label", default="Causal LM")
    parser.add_argument(
        "--execution-label",
        default="eager",
        help="Human-readable candidate execution path, for example eager or rmsnorm,rotary.",
    )
    parser.add_argument("--modeling-source", required=True)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--transformers-source", type=Path)
    return parser.parse_args()


def command_environment(args: argparse.Namespace, candidate: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    paths = []
    if args.transformers_source and not candidate:
        paths.append(str(args.transformers_source.resolve()))
    if candidate:
        paths.append(str(args.repo_root.resolve()))
        paths.extend(args.candidate_pythonpath)
        env["MINDSPEED_LLM_REPO_ROOT"] = str(args.repo_root.resolve())
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_command(name: str, command: list[str], log_path: Path, env: dict[str, str]) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "name": name,
        "command": shlex.join(command),
        "returncode": completed.returncode,
        "log": str(log_path),
        "output_tail": completed.stdout[-4000:],
    }


def parse_layer_counts(value: str, full_layer_count: int) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip().lower()
        layer_count = full_layer_count if item == "full" else int(item)
        layer_count = min(layer_count, full_layer_count)
        if layer_count > 0 and layer_count not in result:
            result.append(layer_count)
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_status(
    output_dir: Path,
    status: dict[str, object],
    *,
    state: str | None = None,
    current_stage: str | None = None,
    layer_count: int | None = None,
    error: BaseException | None = None,
) -> None:
    if state is not None:
        status["state"] = state
    if current_stage is not None:
        status["current_stage"] = current_stage
    status["layer_count"] = layer_count
    if error is not None:
        status["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "log": getattr(error, "log", None),
        }
    write_json(output_dir / STATUS_FILENAME, status)


def run_alignment(args: argparse.Namespace) -> int:
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0,1]")
    if args.candidate_world_size < 2:
        raise ValueError("--candidate-world-size must be at least 2 for FSDP2 certification")
    tools_dir = Path(__file__).resolve().parent
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "format_version": 1,
        "state": "RUNNING",
        "current_stage": "initialize",
        "layer_count": None,
        "run_record": str(output_dir / "run_record.json"),
        "report": str(output_dir / "alignment_report.md"),
    }
    update_status(output_dir, status)

    update_status(output_dir, status, current_stage="checkpoint_index")
    config = read_json(model_dir / "config.json")
    full_layer_count = int(config["num_hidden_layers"])
    layer_counts = parse_layer_counts(args.layer_counts, full_layer_count)
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        index_path = model_dir / "pytorch_model.bin.index.json"
    index = read_json(index_path)
    index_files = sorted(set(index["weight_map"].values()))
    missing_files = [name for name in index_files if not (model_dir / name).is_file()]

    reference_env = command_environment(args)
    candidate_env = command_environment(args, candidate=True)
    input_file = output_dir / "alignment_input.pt"
    update_status(output_dir, status, current_stage="make_fixed_input")
    input_command = [
        args.reference_python,
        str(tools_dir / "make_fixed_input.py"),
        "--tokenizer-dir",
        str(model_dir),
        "--output-file",
        str(input_file),
        "--max-length",
        str(args.max_length),
        "--exact-sequence-length",
        str(args.exact_sequence_length),
    ]
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    tokenizer_config = read_json(tokenizer_config_path) if tokenizer_config_path.is_file() else {}
    use_raw_prompts = args.input_format == "raw" or (
        args.input_format == "auto" and not tokenizer_config.get("chat_template")
    )
    if use_raw_prompts:
        input_command.append("--no-chat-template")
    prompts = args.prompt or ["介绍一下 DeepSeek V4 的模型结构。", "Write a Python cosine similarity function."]
    for prompt in prompts:
        input_command.extend(["--prompt", prompt])
    input_result = run_command("make_fixed_input", input_command, output_dir / "make_fixed_input.log", reference_env)
    if input_result["returncode"] != 0:
        raise AlignmentStageError(
            "make_fixed_input",
            "Fixed input generation failed",
            log=str(input_result["log"]),
        )

    update_status(output_dir, status, current_stage="validate_fixed_input")
    input_probe = subprocess.run(
        [args.reference_python, "-c", "import json,sys,torch;p=torch.load(sys.argv[1],map_location='cpu',weights_only=True);m=p['attention_mask'];print(json.dumps({'sha':p['input_sha256'],'shape':list(p['input_ids'].shape),'valid_token_count':int(m.sum()),'all_tokens_valid':bool(m.all())}))", str(input_file)],
        env=reference_env,
        text=True,
        capture_output=True,
        check=True,
    )
    input_metadata = json.loads(input_probe.stdout)
    expected_valid_tokens = input_metadata["shape"][0] * args.exact_sequence_length
    if input_metadata["shape"][-1] != args.exact_sequence_length:
        raise RuntimeError(
            f"Input sequence length is {input_metadata['shape'][-1]}, expected {args.exact_sequence_length}."
        )
    if input_metadata["valid_token_count"] != expected_valid_tokens or not input_metadata["all_tokens_valid"]:
        raise RuntimeError("Exact-length precision input must contain no padding tokens.")
    update_status(output_dir, status, current_stage="probe_reference_environment")
    version_probe = subprocess.run(
        [args.reference_python, "-c", "import transformers;print(transformers.__version__)"],
        env=reference_env,
        text=True,
        capture_output=True,
        check=True,
    )

    stages = []
    for layer_count in layer_counts:
        stage_dir = output_dir / f"layers_{layer_count}"
        stage_dir.mkdir()
        checkpoint_view = stage_dir / "checkpoint_view"
        commands = []
        update_status(
            output_dir,
            status,
            current_stage="prepare_checkpoint_view",
            layer_count=layer_count,
        )
        prepare_command = [
            args.reference_python,
            str(tools_dir / "prepare_checkpoint_view.py"),
            "--source-dir",
            str(model_dir),
            "--output-dir",
            str(checkpoint_view),
            "--layer-count",
            str(layer_count),
        ]
        for expression in args.checkpoint_config_override:
            prepare_command.extend(["--config-override", expression])
        prepare = run_command(
            "prepare_checkpoint_view",
            prepare_command,
            stage_dir / "prepare_checkpoint_view.log",
            reference_env,
        )
        commands.append(prepare)
        stage = {
            "layer_count": layer_count,
            "checkpoint_view_status": "PASS" if prepare["returncode"] == 0 else "BLOCKED",
            "reference_status": "NOT_RUN",
            "candidate_status": "NOT_RUN",
            "comparison_status": "NOT_RUN",
            "inference_status": "NOT_RUN",
            "commands": commands,
        }
        if prepare["returncode"] != 0:
            stage["failure"] = "prepare_checkpoint_view"
            stages.append(stage)
            continue
        stage["checkpoint_manifest"] = read_json(
            checkpoint_view / "checkpoint_view_manifest.json"
        )

        reference_file = stage_dir / "reference_logits.pt"
        update_status(
            output_dir,
            status,
            current_stage="dump_native_logits",
            layer_count=layer_count,
        )
        reference_command = [
            args.reference_python,
            str(tools_dir / "dump_transformers_logits.py"),
            "--model-dir",
            str(checkpoint_view),
            "--input-file",
            str(input_file),
            "--output-file",
            str(reference_file),
            "--dtype",
            args.dtype,
            "--device-map",
            args.device_map,
            "--device",
            args.device,
            "--attn-implementation",
            "eager",
            "--dequantize-fp8",
        ]
        if args.device_map_json:
            reference_command.extend(["--device-map-json", args.device_map_json])
        reference = run_command(
            "dump_native_logits",
            reference_command,
            stage_dir / "dump_native_logits.log",
            reference_env,
        )
        commands.append(reference)
        stage["reference_status"] = "PASS" if reference["returncode"] == 0 else "BLOCKED"
        if args.accelerator_cleanup_delay > 0:
            time.sleep(args.accelerator_cleanup_delay)

        candidate_file = stage_dir / "candidate_logits.pt"
        update_status(
            output_dir,
            status,
            current_stage="dump_mindspeed_fsdp2_training_logits",
            layer_count=layer_count,
        )
        candidate_command = [
            args.candidate_python,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node",
            str(args.candidate_world_size),
            "--master-port",
            str(args.candidate_master_port + layer_count),
            str(tools_dir / "dump_mindspeed_fsdp2_logits.py"),
            "--config-file",
            str(args.candidate_config.resolve()),
            "--model-dir",
            str(checkpoint_view),
            "--input-file",
            str(input_file),
            "--output-file",
            str(candidate_file),
            "--expected-sequence-length",
            str(args.exact_sequence_length),
        ]
        for expression in args.candidate_config_override:
            candidate_command.extend(["--config-override", expression])
        candidate = run_command(
            "dump_mindspeed_fsdp2_training_logits",
            candidate_command,
            stage_dir / "dump_mindspeed_fsdp2_training_logits.log",
            candidate_env,
        )
        commands.append(candidate)
        stage["candidate_status"] = "PASS" if candidate["returncode"] == 0 else "BLOCKED"
        stage["inference_status"] = stage["candidate_status"]
        if args.accelerator_cleanup_delay > 0:
            time.sleep(args.accelerator_cleanup_delay)

        if reference["returncode"] == 0 and candidate["returncode"] == 0:
            comparison_file = stage_dir / "comparison_report.json"
            update_status(
                output_dir,
                status,
                current_stage="compare_logits",
                layer_count=layer_count,
            )
            comparison = run_command(
                "compare_logits",
                [
                    args.reference_python,
                    str(tools_dir / "compare_logits.py"),
                    "--reference-file",
                    str(reference_file),
                    "--candidate-file",
                    str(candidate_file),
                    "--candidate-framework",
                    "mindspeed-llm-fsdp2-training",
                    "--candidate-execution-kind",
                    "mindspeed-llm-fsdp2-training-forward",
                    "--minimum-candidate-world-size",
                    str(args.candidate_world_size),
                    "--require-candidate-fsdp-wrapped",
                    "--require-candidate-complete-weight-load",
                    "--report-file",
                    str(comparison_file),
                    "--threshold",
                    str(args.threshold),
                ],
                stage_dir / "compare_logits.log",
                reference_env,
            )
            commands.append(comparison)
            if comparison_file.is_file():
                comparison_report = read_json(comparison_file)
                stage["metrics"] = comparison_report["metrics"]
                stage["average_token_cosine"] = stage["metrics"]["average_token_cosine"]
                stage["candidate_execution"] = comparison_report.get("candidate_execution", {})
                stage["comparison_status"] = "PASS" if comparison_report["passed"] else "FAIL"
            else:
                stage["comparison_status"] = "BLOCKED"
            if comparison["returncode"] != 0 and stage["comparison_status"] != "FAIL":
                stage["failure"] = "compare_logits"
        else:
            stage["failure"] = (
                "dump_native_logits"
                if reference["returncode"]
                else "dump_mindspeed_fsdp2_training_logits"
            )
        stages.append(stage)

    required_missing_shards = set()
    for stage in stages:
        manifest_path = output_dir / f"layers_{stage['layer_count']}" / "checkpoint_view" / "checkpoint_view_manifest.json"
        if manifest_path.is_file():
            required_missing_shards.update(read_json(manifest_path).get("missing_required_shards", []))
    passed = all(stage["comparison_status"] == "PASS" for stage in stages)
    failed_stage = next(
        (
            {"layer_count": stage["layer_count"], "stage": stage.get("failure", "compare_logits")}
            for stage in stages
            if stage["comparison_status"] != "PASS"
        ),
        None,
    )
    record = {
        "format_version": 3,
        "status": "PASS" if passed else "FAIL/BLOCKED",
        "failed_stage": failed_stage,
        "model_label": args.model_label,
        "execution_label": args.execution_label,
        "modeling_source": args.modeling_source,
        "model_dir": str(model_dir),
        "repo_root": str(args.repo_root.resolve()),
        "repo_commit": args.repo_commit,
        "transformers_version": version_probe.stdout.strip(),
        "device": args.device,
        "device_map": args.device_map,
        "device_map_json": json.loads(args.device_map_json) if args.device_map_json else None,
        "candidate_config": str(args.candidate_config.resolve()),
        "candidate_world_size": args.candidate_world_size,
        "candidate_config_overrides": args.candidate_config_override,
        "checkpoint_config_overrides": args.checkpoint_config_override,
        "dtype": args.dtype,
        "threshold": args.threshold,
        "accelerator_cleanup_delay": args.accelerator_cleanup_delay,
        "input_sha256": input_metadata["sha"],
        "input_shape": input_metadata["shape"],
        "input_valid_token_count": input_metadata["valid_token_count"],
        "input_all_tokens_valid": input_metadata["all_tokens_valid"],
        "exact_sequence_length": args.exact_sequence_length,
        "input_command": input_result,
        "checkpoint_audit": {
            "index_shard_count": len(index_files),
            "missing_shard_count": len(missing_files),
            "missing_shards": missing_files,
            "missing_required_shard_count": len(required_missing_shards),
            "missing_required_shards": sorted(required_missing_shards),
        },
        "stages": stages,
    }
    record_path = output_dir / "run_record.json"
    write_json(record_path, record)
    update_status(output_dir, status, current_stage="write_alignment_report", layer_count=None)
    subprocess.run(
        [
            args.reference_python,
            str(tools_dir / "write_alignment_report.py"),
            "--run-record",
            str(record_path),
            "--report-file",
            str(output_dir / "alignment_report.md"),
        ],
        env=reference_env,
        check=True,
    )
    status["failed_stage"] = failed_stage
    update_status(
        output_dir,
        status,
        state="PASS" if passed else "FAIL/BLOCKED",
        current_stage="completed",
        layer_count=None,
    )
    return 0 if passed else 2


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    try:
        return run_alignment(args)
    except BaseException as exc:
        if output_dir.is_dir():
            status_path = output_dir / STATUS_FILENAME
            status = read_json(status_path) if status_path.is_file() else {
                "format_version": 1,
                "current_stage": getattr(exc, "stage", "initialize"),
            }
            if isinstance(exc, AlignmentStageError):
                status["current_stage"] = exc.stage
            update_status(output_dir, status, state="FAILED", error=exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
