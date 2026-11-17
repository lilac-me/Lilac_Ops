#!/usr/bin/env python3
"""Create a non-destructive layer-trimmed view of a sharded HF checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
SCHEDULE_FIELDS = ("layer_types", "mlp_layer_types", "compress_ratios")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layer-count", required=True, type=int)
    parser.add_argument(
        "--expert-count",
        type=int,
        default=None,
        help="Keep only experts 0..N-1 per MoE layer (OOM fallback for single-machine "
        "verification). Must be >= num_experts_per_tok and divisible by ep_size.",
    )
    parser.add_argument("--include-mtp", action="store_true")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="FIELD=JSON_VALUE",
        help="Explicitly correct one top-level config field in the checkpoint view.",
    )
    return parser.parse_args()


def parse_config_override(expression: str) -> tuple[str, object]:
    if "=" not in expression:
        raise ValueError(f"Invalid --config-override {expression!r}; expected FIELD=JSON_VALUE")
    field, raw_value = expression.split("=", 1)
    if not field or "." in field:
        raise ValueError(f"Config override must name one top-level field: {field!r}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return field, value


def infer_checkpoint_expert_count(weight_map: dict[str, str]) -> int | None:
    expert_ids = set()
    for key in weight_map:
        match = re.match(r"(?:model\.)?layers\.0\..*\.experts\.(\d+)\.", key)
        if match:
            expert_ids.add(int(match.group(1)))
    if not expert_ids:
        return None
    expected = set(range(max(expert_ids) + 1))
    if expert_ids != expected:
        raise ValueError("Layer 0 expert ids are not a contiguous range starting at zero")
    return len(expert_ids)


def select_weight_keys(
    weight_map: dict[str, str], layer_count: int, include_mtp: bool,
    expert_count: int | None = None,
) -> dict[str, str]:
    selected = {}
    for key, filename in weight_map.items():
        layer_match = re.match(r"^(?:model\.)?layers\.(\d+)\.", key)
        if layer_match:
            if int(layer_match.group(1)) >= layer_count:
                continue
            if expert_count is not None:
                expert_match = re.search(r"\.experts\.(\d+)\.", key)
                if expert_match and int(expert_match.group(1)) >= expert_count:
                    continue
            selected[key] = filename
            continue
        if (key.startswith("mtp.") or key.startswith("model.mtp.")) and not include_mtp:
            continue
        selected[key] = filename
    return selected


def link_file(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve())


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.layer_count <= 0:
        raise ValueError("--layer-count must be positive")
    if source_dir == output_dir:
        raise ValueError("Checkpoint view must not overwrite the source checkpoint.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_layer_count = int(config.get("num_hidden_layers", 0))
    if not 1 <= args.layer_count <= source_layer_count:
        raise ValueError(
            f"layer_count={args.layer_count} outside source range [1,{source_layer_count}]"
        )

    index_path = next((source_dir / name for name in INDEX_NAMES if (source_dir / name).is_file()), None)
    if index_path is None:
        raise FileNotFoundError(f"No sharded checkpoint index found in {source_dir}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in {index_path}")
    selected_weight_map = select_weight_keys(
        weight_map, args.layer_count, args.include_mtp, args.expert_count
    )
    config_overrides = dict(parse_config_override(item) for item in args.config_override)
    config.update(config_overrides)
    if args.expert_count is not None:
        expert_field = next(
            (f for f in ("num_experts", "n_routed_experts", "num_local_experts") if f in config),
            None,
        )
        if expert_field is None:
            raise ValueError("--expert-count 需要 config 中存在 num_experts/n_routed_experts/num_local_experts 字段")
        if args.expert_count > int(config[expert_field]):
            raise ValueError(
                f"--expert-count {args.expert_count} 超过原始专家数 {config[expert_field]}"
            )
        config[expert_field] = args.expert_count
    checkpoint_expert_count = infer_checkpoint_expert_count(selected_weight_map)
    configured_expert_count = next(
        (
            config[field]
            for field in ("n_routed_experts", "num_local_experts", "num_experts")
            if config.get(field) is not None
        ),
        None,
    )
    if checkpoint_expert_count is not None and configured_expert_count is None:
        raise ValueError("Checkpoint contains experts but config declares no expert count field")
    if checkpoint_expert_count is not None and int(configured_expert_count) != checkpoint_expert_count:
        raise ValueError(
            "Checkpoint/config expert count mismatch: "
            f"checkpoint={checkpoint_expert_count}, config={configured_expert_count}. "
            "Pass an explicit --config-override after verifying the checkpoint."
        )
    required_files = sorted(set(selected_weight_map.values()))
    missing_required_files = [name for name in required_files if not (source_dir / name).is_file()]
    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_layer_count": source_layer_count,
        "selected_layer_count": args.layer_count,
        "selected_key_count": len(selected_weight_map),
        "omitted_key_count": len(weight_map) - len(selected_weight_map),
        "required_shards": required_files,
        "missing_required_shards": missing_required_files,
        "config_overrides": config_overrides,
        "selected_expert_count": args.expert_count,
        "checkpoint_expert_count": checkpoint_expert_count,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoint_view_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if missing_required_files:
        raise FileNotFoundError(
            "Required checkpoint shards are missing: " + ", ".join(missing_required_files)
        )

    for source_file in source_dir.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name in INDEX_NAMES or source_file.suffix in {".safetensors", ".bin"}:
            continue
        if source_file.name == "config.json":
            continue
        link_file(source_file, output_dir / source_file.name)
    for filename in required_files:
        link_file(source_dir / filename, output_dir / filename)

    config["num_hidden_layers"] = args.layer_count
    for field in SCHEDULE_FIELDS:
        values = config.get(field)
        if isinstance(values, list):
            config[field] = values[: args.layer_count]
    if "num_hash_layers" in config:
        config["num_hash_layers"] = min(int(config["num_hash_layers"]), args.layer_count)
    if not args.include_mtp:
        config["num_nextn_predict_layers"] = 0
    if args.expert_count is not None:
        for topk_field in ("num_experts_per_tok", "moe_topk", "top_k"):
            if topk_field in config and int(config[topk_field]) > args.expert_count:
                config[topk_field] = args.expert_count
    config["use_cache"] = False
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    index_metadata = dict(index.get("metadata") or {})
    index_metadata["checkpoint_view"] = {
        "source_dir": str(source_dir),
        "source_layer_count": source_layer_count,
        "selected_layer_count": args.layer_count,
        "include_mtp": args.include_mtp,
        "selected_expert_count": args.expert_count,
        "selected_key_count": len(selected_weight_map),
        "omitted_key_count": len(weight_map) - len(selected_weight_map),
    }
    (output_dir / index_path.name).write_text(
        json.dumps({"metadata": index_metadata, "weight_map": selected_weight_map}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
