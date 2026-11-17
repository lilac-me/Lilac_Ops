#!/usr/bin/env python3
"""Validate Math500 scores and exact raw predictions from an AISBench run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def _normalise_score(value: Any) -> Decimal:
    try:
        score = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid accuracy value: {value!r}") from exc
    if score > 1:
        score /= Decimal("100")
    if score < 0 or score > 1:
        raise ValueError(f"accuracy must be in [0, 1] or [0, 100]: {value!r}")
    return score


def _normalise_temperature(value: Any) -> Decimal:
    try:
        temperature = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid temperature value: {value!r}") from exc
    if not temperature.is_finite() or temperature < 0:
        raise ValueError(f"temperature must be finite and non-negative: {value!r}")
    return temperature


def _normalise_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _normalise_option_name(value: str) -> str:
    return value.lstrip("-").replace("_", "-")


def _parse_serve_args(values: Any) -> dict[str, list[str | bool]]:
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError("config.serve_args must be an array of strings")

    options: dict[str, list[str | bool]] = {}
    index = 0
    while index < len(values):
        token = values[index]
        if not token.startswith("--"):
            raise ValueError(f"config.serve_args contains an orphan value: {token!r}")
        if "=" in token:
            name, option_value = token.split("=", 1)
            options.setdefault(_normalise_option_name(name), []).append(option_value)
            index += 1
            continue

        name = _normalise_option_name(token)
        if index + 1 < len(values) and not values[index + 1].startswith("--"):
            options.setdefault(name, []).append(values[index + 1])
            index += 2
        else:
            options.setdefault(name, []).append(True)
            index += 1
    return options


def _last_serve_option(
    options: dict[str, list[str | bool]], name: str
) -> str | bool | None:
    values = options.get(_normalise_option_name(name), [])
    return values[-1] if values else None


def _serve_bool(options: dict[str, list[str | bool]], name: str) -> bool:
    normalized_name = _normalise_option_name(name)
    positive = _last_serve_option(options, normalized_name)
    negative = _last_serve_option(options, f"no-{normalized_name}")
    if positive is not None and negative is not None:
        raise ValueError(
            f"config.serve_args contains both --{normalized_name} and --no-{normalized_name}"
        )
    if negative is not None:
        return False
    if positive is None:
        return False
    if positive is True:
        return True
    return _normalise_bool(positive, name=f"serve option --{normalized_name}")


def _scenario_check(expected: Any, actual: Any) -> dict[str, Any]:
    def json_value(value: Any) -> Any:
        return str(value) if isinstance(value, Decimal) else value

    return {
        "expected": json_value(expected),
        "actual": json_value(actual),
        "pass": actual == expected,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_sha256(prediction: str) -> str:
    return hashlib.sha256(prediction.encode("utf-8")).hexdigest()


def _sample_label(key: tuple[str, str]) -> str:
    return f"{key[0]}:{key[1]}"


def _load_predictions(path: Path) -> dict[tuple[str, str], str]:
    predictions: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            if record.get("success") is not True:
                raise ValueError(f"{path}:{line_number}: sample is not successful")
            if "data_abbr" not in record or "id" not in record:
                raise ValueError(f"{path}:{line_number}: missing data_abbr or id")
            prediction = record.get("prediction")
            if not isinstance(prediction, str):
                raise ValueError(f"{path}:{line_number}: prediction must be a string")
            key = (str(record["data_abbr"]), str(record["id"]))
            if key in predictions:
                raise ValueError(f"{path}:{line_number}: duplicate sample {_sample_label(key)}")
            predictions[key] = prediction
    return predictions


def _load_accuracy_result(
    path: Path,
) -> tuple[list[Decimal], list[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: missing config object")
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError(f"{path}: missing rounds array")

    scores: list[Decimal] = []
    failures: list[str] = []
    for index, result in enumerate(rounds, start=1):
        if not isinstance(result, dict):
            failures.append(f"round {index}: result is not an object")
            continue
        if result.get("status") != "ok":
            failures.append(f"round {index}: status={result.get('status')!r}")
            continue
        if "accuracy" not in result:
            failures.append(f"round {index}: accuracy is missing")
            continue
        try:
            scores.append(_normalise_score(result["accuracy"]))
        except ValueError as exc:
            failures.append(f"round {index}: {exc}")
    return scores, failures, config


def _discover_predictions(
    outputs_root: Path, started_after: Path
) -> list[Path]:
    marker_mtime = started_after.stat().st_mtime
    experiments: dict[Path, list[Path]] = {}
    for path in outputs_root.glob(
        "*/predictions/vllm-api-stream-chat/*.jsonl"
    ):
        if path.stat().st_mtime < marker_mtime:
            continue
        experiment = path.parents[2]
        experiments.setdefault(experiment, []).append(path.resolve())

    selected: list[Path] = []
    for experiment, paths in experiments.items():
        successful = [path for path in paths if "failed" not in path.name.lower()]
        if len(successful) != 1:
            raise ValueError(
                f"{experiment}: expected exactly one non-failed prediction JSONL, "
                f"found {len(successful)}"
            )
        selected.append(successful[0])
    return sorted(selected, key=lambda path: (path.stat().st_mtime, str(path)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate vllm-ascend-aisbench-accuracy output and exact Math500 "
            "prediction text across rounds."
        )
    )
    parser.add_argument(
        "--accuracy-result",
        required=True,
        help="Raw JSON emitted by the common accuracy skill (raw_result path).",
    )
    prediction_group = parser.add_mutually_exclusive_group(required=True)
    prediction_group.add_argument(
        "--prediction-file",
        action="append",
        dest="prediction_files",
        help="One prediction JSONL per round, in round order; repeat this option.",
    )
    prediction_group.add_argument(
        "--outputs-root",
        help="AISBench outputs/default directory; requires --started-after.",
    )
    parser.add_argument(
        "--started-after",
        help="Marker file created immediately before the common skill invocation.",
    )
    parser.add_argument("--expected-rounds", type=int, default=10)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--accuracy-threshold", default="93")
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-visible-devices", required=True)
    parser.add_argument("--expected-cann-dir", required=True)
    parser.add_argument("--expected-tp", type=int, required=True)
    parser.add_argument("--expected-dp", type=int, required=True)
    parser.add_argument("--expected-pp", type=int, required=True)
    parser.add_argument(
        "--expected-enable-expert-parallel",
        choices=["true", "false"],
        required=True,
    )
    parser.add_argument("--expected-dataset", required=True)
    parser.add_argument("--expected-dataset-kind", required=True)
    parser.add_argument("--expected-temperature", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument(
        "--expected-vllm-batch-invariant", choices=["1"], required=True
    )
    parser.add_argument(
        "--expected-prefix-caching", choices=["true", "false"], required=True
    )
    parser.add_argument("--expected-max-model-len", type=int, required=True)
    parser.add_argument("--expected-max-num-seqs", type=int, required=True)
    parser.add_argument("--expected-max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--expected-block-size", type=int, required=True)
    parser.add_argument(
        "--expected-enable-chunked-prefill",
        choices=["true", "false"],
        required=True,
    )
    parser.add_argument("--expected-distributed-executor-backend", required=True)
    parser.add_argument("--expected-generation-config", required=True)
    parser.add_argument("--expected-gpu-memory-utilization", required=True)
    parser.add_argument(
        "--expected-enforce-eager", choices=["true", "false"], required=True
    )
    parser.add_argument(
        "--expected-disable-log-stats", choices=["true", "false"], required=True
    )
    parser.add_argument("--expected-load-format", required=True)
    parser.add_argument(
        "--expected-async-scheduling", choices=["true", "false"], required=True
    )
    parser.add_argument("--max-mismatch-details", type=int, default=20)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    positive_expectations = {
        "expected rounds": args.expected_rounds,
        "expected sample count": args.expected_count,
        "expected TP": args.expected_tp,
        "expected DP": args.expected_dp,
        "expected PP": args.expected_pp,
        "expected max model length": args.expected_max_model_len,
        "expected max number of sequences": args.expected_max_num_seqs,
        "expected max number of batched tokens": args.expected_max_num_batched_tokens,
        "expected block size": args.expected_block_size,
    }
    for name, value in positive_expectations.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    scores, round_failures, config = _load_accuracy_result(
        Path(args.accuracy_result).resolve()
    )
    threshold = _normalise_score(args.accuracy_threshold)
    expected_temperature = _normalise_temperature(args.expected_temperature)
    if "temperature" not in config:
        raise ValueError("raw_result config is missing temperature")
    temperature = _normalise_temperature(config["temperature"])

    serve_options = _parse_serve_args(config.get("serve_args"))
    env = config.get("env")
    if not isinstance(env, dict):
        raise ValueError("raw_result config is missing env object")

    def required_config(name: str) -> Any:
        if name not in config:
            raise ValueError(f"raw_result config is missing {name}")
        return config[name]

    def required_serve_option(name: str) -> str | bool:
        value = _last_serve_option(serve_options, name)
        if value is None:
            raise ValueError(f"raw_result config.serve_args is missing --{name}")
        if value is True:
            raise ValueError(
                f"raw_result config.serve_args option --{name} is missing its value"
            )
        return value

    def required_serve_bool(name: str) -> bool:
        normalized_name = _normalise_option_name(name)
        positive = _last_serve_option(serve_options, normalized_name)
        negative = _last_serve_option(serve_options, f"no-{normalized_name}")
        if positive is None and negative is None:
            raise ValueError(
                "raw_result config.serve_args is missing "
                f"--{normalized_name}/--no-{normalized_name}"
            )
        return _serve_bool(serve_options, normalized_name)

    actual_service_values = {
        "max_model_len": int(required_serve_option("max-model-len")),
        "max_num_seqs": int(required_serve_option("max-num-seqs")),
        "max_num_batched_tokens": int(
            required_serve_option("max-num-batched-tokens")
        ),
        "block_size": int(required_serve_option("block-size")),
        "enable_chunked_prefill": required_serve_bool("enable-chunked-prefill"),
        "distributed_executor_backend": str(
            required_serve_option("distributed-executor-backend")
        ),
        "generation_config": str(required_serve_option("generation-config")),
        "gpu_memory_utilization": _normalise_temperature(
            required_serve_option("gpu-memory-utilization")
        ),
        "enforce_eager": required_serve_bool("enforce-eager"),
        "enable_prefix_caching": required_serve_bool("enable-prefix-caching"),
        # vLLM 0.23.1 exposes only the positive --disable-log-stats flag;
        # absence is the supported CLI representation of the False default.
        "disable_log_stats": _serve_bool(serve_options, "disable-log-stats"),
        "pipeline_parallel_size": int(
            _last_serve_option(serve_options, "pipeline-parallel-size") or 1
        ),
        "enable_expert_parallel": _serve_bool(
            serve_options, "enable-expert-parallel"
        ),
        "load_format": str(
            _last_serve_option(serve_options, "load-format") or "auto"
        ),
        "async_scheduling": required_serve_bool("async-scheduling"),
    }
    expected_service_values = {
        "max_model_len": args.expected_max_model_len,
        "max_num_seqs": args.expected_max_num_seqs,
        "max_num_batched_tokens": args.expected_max_num_batched_tokens,
        "block_size": args.expected_block_size,
        "enable_chunked_prefill": _normalise_bool(
            args.expected_enable_chunked_prefill,
            name="expected enable chunked prefill",
        ),
        "distributed_executor_backend": args.expected_distributed_executor_backend,
        "generation_config": args.expected_generation_config,
        "gpu_memory_utilization": _normalise_temperature(
            args.expected_gpu_memory_utilization
        ),
        "enforce_eager": _normalise_bool(
            args.expected_enforce_eager, name="expected enforce eager"
        ),
        "enable_prefix_caching": _normalise_bool(
            args.expected_prefix_caching, name="expected prefix caching"
        ),
        "disable_log_stats": _normalise_bool(
            args.expected_disable_log_stats, name="expected disable log stats"
        ),
        "pipeline_parallel_size": args.expected_pp,
        "enable_expert_parallel": _normalise_bool(
            args.expected_enable_expert_parallel,
            name="expected enable expert parallel",
        ),
        "load_format": args.expected_load_format,
        "async_scheduling": _normalise_bool(
            args.expected_async_scheduling, name="expected async scheduling"
        ),
    }

    scenario_checks = {
        "model": _scenario_check(args.expected_model, required_config("model")),
        "ASCEND_RT_VISIBLE_DEVICES": _scenario_check(
            args.expected_visible_devices,
            str(env.get("ASCEND_RT_VISIBLE_DEVICES", "")),
        ),
        "cann_dir": _scenario_check(
            args.expected_cann_dir, config.get("cann_dir", "/usr/local/Ascend")
        ),
        "tp": _scenario_check(args.expected_tp, required_config("tp")),
        "dp": _scenario_check(args.expected_dp, config.get("dp", 1)),
        "runs": _scenario_check(args.expected_rounds, required_config("runs")),
        "dataset": _scenario_check(
            args.expected_dataset, required_config("dataset")
        ),
        "dataset_kind": _scenario_check(
            args.expected_dataset_kind, required_config("dataset_kind")
        ),
        "temperature": _scenario_check(expected_temperature, temperature),
        "seed": _scenario_check(args.expected_seed, required_config("seed")),
        "VLLM_BATCH_INVARIANT": _scenario_check(
            args.expected_vllm_batch_invariant,
            str(env.get("VLLM_BATCH_INVARIANT", "")),
        ),
    }
    for name, expected in expected_service_values.items():
        scenario_checks[name] = _scenario_check(expected, actual_service_values[name])
    scenario_config_ok = all(check["pass"] for check in scenario_checks.values())

    if args.outputs_root:
        if not args.started_after:
            raise ValueError("--outputs-root requires --started-after")
        prediction_paths = _discover_predictions(
            Path(args.outputs_root).resolve(), Path(args.started_after).resolve()
        )
    else:
        prediction_paths = [Path(path).resolve() for path in args.prediction_files]

    duplicate_prediction_files = len(prediction_paths) != len(set(prediction_paths))
    round_count_ok = (
        len(scores) == args.expected_rounds
        and len(prediction_paths) == args.expected_rounds
        and not round_failures
        and not duplicate_prediction_files
    )
    score_threshold_ok = bool(scores) and all(score >= threshold for score in scores)
    score_consistency_ok = bool(scores) and len(set(scores)) == 1

    comparisons: list[dict[str, Any]] = []
    prediction_consistency_ok = (
        len(prediction_paths) == args.expected_rounds
        and not duplicate_prediction_files
    )
    baseline: dict[tuple[str, str], str] | None = None
    baseline_keys: set[tuple[str, str]] = set()

    for index, path in enumerate(prediction_paths, start=1):
        try:
            current = _load_predictions(path)
            current_keys = set(current)
            if baseline is None:
                baseline = current
                baseline_keys = current_keys
            missing = sorted(baseline_keys - current_keys)
            extra = sorted(current_keys - baseline_keys)
            mismatches = sorted(
                key
                for key in baseline_keys & current_keys
                if baseline[key] != current[key]
            )
            count_ok = len(current) == args.expected_count
            round_ok = count_ok and not missing and not extra and not mismatches
            prediction_consistency_ok = prediction_consistency_ok and round_ok
            comparisons.append(
                {
                    "round": index,
                    "status": "PASS" if round_ok else "FAIL",
                    "path": str(path),
                    "sample_count": len(current),
                    "file_sha256": _file_sha256(path),
                    "missing_count": len(missing),
                    "extra_count": len(extra),
                    "text_mismatch_count": len(mismatches),
                    "missing_samples": [
                        _sample_label(key)
                        for key in missing[: args.max_mismatch_details]
                    ],
                    "extra_samples": [
                        _sample_label(key)
                        for key in extra[: args.max_mismatch_details]
                    ],
                    "text_mismatches": [
                        {
                            "sample": _sample_label(key),
                            "baseline_prediction_sha256": _prediction_sha256(
                                baseline[key]
                            ),
                            "candidate_prediction_sha256": _prediction_sha256(
                                current[key]
                            ),
                        }
                        for key in mismatches[: args.max_mismatch_details]
                    ],
                }
            )
        except (OSError, ValueError) as exc:
            prediction_consistency_ok = False
            comparisons.append(
                {"round": index, "status": "ERROR", "path": str(path), "error": str(exc)}
            )

    passed = (
        round_count_ok
        and score_threshold_ok
        and score_consistency_ok
        and scenario_config_ok
        and prediction_consistency_ok
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "accuracy_result": str(Path(args.accuracy_result).resolve()),
        "expected_rounds": args.expected_rounds,
        "actual_score_rounds": len(scores),
        "actual_prediction_rounds": len(prediction_paths),
        "duplicate_prediction_files": duplicate_prediction_files,
        "round_count_pass": round_count_ok,
        "expected_sample_count": args.expected_count,
        "round_failures": round_failures,
        "scores": [str(score) for score in scores],
        "unique_scores": sorted(str(score) for score in set(scores)),
        "accuracy_threshold": str(threshold),
        "scenario_config_pass": scenario_config_ok,
        "scenario_config": scenario_checks,
        "serve_args": config.get("serve_args"),
        "score_threshold_pass": score_threshold_ok,
        "score_consistency_pass": score_consistency_ok,
        "prediction_consistency_pass": prediction_consistency_ok,
        "comparison": "exact raw prediction text by (data_abbr, id)",
        "rounds": comparisons,
    }
    return report, passed


def main() -> int:
    args = _parse_args()
    report_path = Path(args.report).resolve()
    try:
        report, passed = _validate(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {"status": "ERROR", "error": str(exc)}
        passed = False
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
