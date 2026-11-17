# MindSpeed-Agent Platform Backends
# ==================================
# Each backend implements the AbstractBackend interface for its platform.
#
# Available backends:
#   ClaudeCodeBackend  — Claude Code (Workflow JS runtime)
#   CodexBackend       — Codex (task/subtask system)

from engine.backends.base import AbstractBackend, AgentResult  # noqa: F401
