# Codex Backend
# =============
# Codex uses a task/subtask system for agent orchestration.
# Agents are defined as tasks with model, tools, and context configuration.
#
# This backend wraps the Codex Python API to spawn agents and collect
# structured results, translating them into the common AgentResult format.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from engine.backends.base import AbstractBackend, AgentResult
from engine.agent_registry import load_agent_registry


class CodexBackend(AbstractBackend):
    """Codex backend — spawns agents via the Codex task/subtask system.

    Codex agents are configured in agent.yaml under the `codex:` key.
    The backend resolves agent specs, builds task prompts, and collects
    structured results.
    """

    def __init__(self, repo_root: Path | None = None):
        self._repo_root = repo_root or Path.cwd()
        self._registry = load_agent_registry(self._repo_root, platform="codex")

    def platform_name(self) -> str:
        return "codex"

    def available_agents(self) -> list[str]:
        return list(self._registry.keys())

    def spawn_agent(
        self,
        agent_type: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        workspace: Path | None = None,
        label: str = "",
    ) -> AgentResult:
        """Spawn an agent via the Codex task system.

        NOTE: This is a STUB implementation. Replace with actual Codex API
        calls when the Codex Python SDK is available.

        The expected flow:
          1. Load agent.yaml → get codex config (model, context, tools).
          2. Create a Codex task with the prompt.
          3. Poll for completion.
          4. Parse the result text for handoff + structured output.
        """
        agent_info = self._registry.get(agent_type)
        if agent_info is None:
            return AgentResult(
                success=False,
                error=f"Agent '{agent_type}' not found in registry",
                label=label,
            )

        platform_cfg = agent_info.get("platform_config", {})
        model = platform_cfg.get("model", "claude-sonnet-5")

        # TODO: Replace with actual Codex API call
        # result = codex_client.create_and_run_task(
        #     model=model,
        #     prompt=prompt,
        #     context=platform_cfg.get("context", "fork"),
        #     tools=resolve_codex_tools(agent_info["spec"].get("tools", [])),
        # )
        result_text = (
            f"[Codex stub] Agent {agent_type} ({model}) would run with prompt: "
            f"{prompt[:200]}..."
        )
        success = True

        handoff = _extract_codex_handoff(result_text)
        output = _parse_structured_output(result_text, schema)

        return AgentResult(
            success=success,
            handoff=handoff,
            output=output,
            raw_text=result_text,
            label=label,
        )


def _extract_codex_handoff(text: str) -> str:
    """Extract handoff line from Codex task result."""
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.startswith(("→ orchestrator:", "handoff:", "@aog-")):
            return s.replace("**", "")
    return ""


def _parse_structured_output(text: str, schema: dict | None) -> dict | None:
    """Try to extract structured JSON output from text."""
    if schema is None:
        return None
    # Find JSON block in text
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try parsing the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return None
