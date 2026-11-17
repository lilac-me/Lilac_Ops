"""Load plugin-local agents without allowing cross-plugin name collisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_agent_registry(plugin_dir: Path, platform: str = "codex") -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for agent_dir in sorted((plugin_dir / "agents").iterdir()):
        spec_path = agent_dir / "agent.yaml"
        if not spec_path.is_file():
            continue
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        name = str(spec.get("name", agent_dir.name))
        qualified = f"{plugin_dir.name}:{name}"
        schema_name = spec.get("schema")
        schema_path = None
        if schema_name == "stage_result_schema":
            schema_path = plugin_dir / "contracts" / "stage_result.schema.json"
        registry[qualified] = {
            "id": qualified,
            "name": name,
            "plugin": plugin_dir.name,
            "plugin_dir": plugin_dir,
            "agent_dir": agent_dir,
            "spec": spec,
            "schema_path": schema_path,
            "platform_config": spec.get(platform.replace("-", "_"), {}),
        }
    return registry
