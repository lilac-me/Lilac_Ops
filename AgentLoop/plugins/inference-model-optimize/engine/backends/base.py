# Abstract Backend Interface
# ==========================
# Every agent platform (Claude Code, Codex) implements this interface.
# The workflow engine calls spawn_agent() to delegate work to agents, and the
# backend translates this into platform-specific invocation.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentResult:
    """Normalized result from any agent platform.

    After an agent completes, the backend packages its output into this
    dataclass so the workflow engine can extract the handoff line, validate
    structured output against the expected schema, and decide the next state.
    """

    success: bool
    handoff: str = ""
    output: dict[str, Any] | None = None
    raw_text: str = ""
    error: str | None = None
    label: str = ""


class AbstractBackend(ABC):
    """Each agent platform implements this interface.

    The workflow engine is backend-agnostic: it calls spawn_agent() and
    the backend handles the platform-specific mechanics (Workflow JS
    agent() for Claude Code, task/subtask for Codex).
    """

    @abstractmethod
    def spawn_agent(
        self,
        agent_type: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        workspace: Path | None = None,
        label: str = "",
    ) -> AgentResult:
        """Spawn an agent on this platform and wait for its result.

        Args:
            agent_type: Platform-qualified agent identifier.
                Format: ``<plugin>:<agent-dir>:<agent-name>`` for Claude Code,
                or ``<plugin>/<agent-name>`` for Codex.
            prompt: The full prompt text to send to the agent.
            schema: Optional JSON Schema for structured output.
            workspace: Path to the workspace directory (for file I/O).
            label: Human-readable label for display/progress.

        Returns:
            AgentResult with success status, handoff line, structured output,
            raw text, and optional error message.
        """

    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier: 'claude-code' | 'codex'."""

    @abstractmethod
    def available_agents(self) -> list[str]:
        """Return the list of agent types available on this platform."""

    def read_workspace_file(self, workspace: Path, relpath: str) -> str | None:
        """Read a file from the workspace. Returns None if absent."""
        target = (workspace / relpath).resolve()
        try:
            return target.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            return None

    def write_workspace_file(self, workspace: Path, relpath: str, content: str) -> None:
        """Write a file into the workspace. Creates parent directories."""
        target = (workspace / relpath).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def workspace_file_exists(self, workspace: Path, relpath: str) -> bool:
        """Check whether a file exists in the workspace."""
        return (workspace / relpath).is_file()
