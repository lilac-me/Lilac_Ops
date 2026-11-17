"""Deterministically derive a TP plan from visible devices and model metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _positive_divisors(value: int) -> list[int]:
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def _model_parallel_limits(config: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for key in ("num_attention_heads", "num_key_value_heads"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            values.append(value)
    return values


def plan_tp(visible_devices: list[int | str], model_config: dict[str, Any],
            *, runnable_tps: list[int] | None = None,
            requested_tps: list[int] | None = None) -> dict[str, Any]:
    """Return an auditable plan; never selects a TP incompatible with the model."""
    if not visible_devices or len(set(map(str, visible_devices))) != len(visible_devices):
        raise ValueError("visible_devices must be a non-empty unique list")
    count = len(visible_devices)
    candidates = _positive_divisors(count)
    limits = _model_parallel_limits(model_config)
    if not limits:
        raise ValueError("model parallel metadata is required before TP planning")
    compatible = [tp for tp in candidates if all(limit % tp == 0 for limit in limits)]
    excluded = [{"tp": tp, "reason": "MODEL_DIMENSION_INCOMPATIBLE"}
                for tp in candidates if tp not in compatible]
    if runnable_tps is not None:
        runnable = {int(tp) for tp in runnable_tps if int(tp) > 0}
        excluded.extend({"tp": tp, "reason": "MODEL_LOAD_UNAVAILABLE"}
                        for tp in compatible if tp not in runnable)
        compatible = [tp for tp in compatible if tp in runnable]
    if requested_tps is not None:
        if not requested_tps or not set(requested_tps) <= set(compatible):
            raise ValueError("explicit TP constraint is incompatible; user decision required")
        compatible = [tp for tp in compatible if tp in requested_tps]
    if not compatible:
        raise ValueError("no model-compatible runnable TP for the visible devices")
    selected = max(compatible)
    matrix = [tp for tp in compatible if tp in {1, 2, 4}]
    if not matrix:
        matrix = [selected]
    diagnostic = [tp for tp in reversed(compatible) if tp < selected][:2]
    return {
        "mode": "auto_from_visible_devices",
        "visible_device_count": count,
        "candidate_tps": candidates,
        "compatible_tps": compatible,
        "excluded_tps": excluded,
        "load_probe_status": "not_probed" if runnable_tps is None else "provided",
        "model_parallel_limits": limits,
        "selected_plan": {
            "phase1": selected,
            "phase2_long_output": selected,
            "phase2_long_input": selected,
            "phase2_matrix": matrix,
            "phase2_diagnosis": diagnostic or [selected],
            "phase3": selected,
        },
    }


def plan_from_model_path(visible_devices: list[int | str], model_path: str | Path,
                         *, runnable_tps: list[int] | None = None) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: JSON object required")
    return plan_tp(visible_devices, config, runnable_tps=runnable_tps)
