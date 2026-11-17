# Agent Registry
# ==============
# Discovers and maps agent types across plugins and platforms.
# Reads agent.yaml files from plugins/<plugin>/agents/<agent>/agent.yaml
# and resolves them for the active platform backend.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# Agent spec loading
# ---------------------------------------------------------------------------

def load_agent_spec(agent_dir: Path) -> dict[str, Any] | None:
    """Load an agent.yaml or AGENT.md from an agent directory.

    Prefers agent.yaml (cross-platform spec).
    Falls back to AGENT.md (Claude Code legacy format).
    """
    # Prefer agent.yaml
    yaml_path = agent_dir / "agent.yaml"
    if yaml_path.exists() and yaml is not None:
        with yaml_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Fallback: parse AGENT.md frontmatter
    md_path = agent_dir / "AGENT.md"
    if md_path.exists():
        return _parse_agent_md_frontmatter(md_path)

    return None


def _parse_agent_md_frontmatter(md_path: Path) -> dict[str, Any]:
    """Parse YAML frontmatter from an AGENT.md file."""
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {"name": md_path.parent.name, "description": ""}

    # Extract frontmatter between --- markers
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {"name": md_path.parent.name, "description": ""}

    if yaml is None:
        return {"name": md_path.parent.name, "description": parts[1].strip()[:200]}

    try:
        fm = yaml.safe_load(parts[1])
        if not isinstance(fm, dict):
            return {"name": md_path.parent.name, "description": ""}
        return fm
    except Exception:
        return {"name": md_path.parent.name, "description": ""}


# ---------------------------------------------------------------------------
# Agent discovery
# ---------------------------------------------------------------------------

def discover_agents(plugin_dir: Path) -> dict[str, dict[str, Any]]:
    """Discover all agents in a plugin's agents/ directory.

    Returns: {agent_name: agent_spec, ...}
    """
    agents_dir = plugin_dir / "agents"
    if not agents_dir.is_dir():
        return {}

    agents = {}
    for agent_dir in sorted(agents_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        spec = load_agent_spec(agent_dir)
        if spec:
            name = spec.get("name", agent_dir.name)
            agents[name] = spec
    return agents


def load_agent_registry(repo_root: Path, platform: str = "claude-code") -> dict[str, dict]:
    """Load all agent specs from all plugins, resolved for a specific platform.

    Args:
        repo_root: Root of the MindSpeed-Agent repository.
        platform: 'claude-code' or 'codex'.

    Returns:
        {agent_name: {spec, plugin_dir, backend_config, ...}}
    """
    registry: dict[str, dict] = {}
    plugins_dir = repo_root / "plugins"
    if not plugins_dir.is_dir():
        return registry

    for plugin_dir in sorted(plugins_dir.iterdir()):
        if not plugin_dir.is_dir():
            continue
        agents = discover_agents(plugin_dir)
        for name, spec in agents.items():
            # Resolve platform-specific config
            backend_config = spec.get(platform.replace("-", "_"), {})
            registry[name] = {
                "spec": spec,
                "plugin": plugin_dir.name,
                "plugin_dir": plugin_dir,
                "platform_config": backend_config,
            }
    return registry
