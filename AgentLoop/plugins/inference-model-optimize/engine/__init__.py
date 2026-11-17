# MindSpeed-Agent Cross-Platform Workflow Engine
# ===============================================
# Platform-agnostic workflow engine supporting Claude Code and Codex.
#
# Usage:
#   from engine.state_machine import StateMachine
#   from engine.state_executor import StateExecutor
#   from engine.fsm_conditions import eval_condition
#
# Architecture:
#   engine/                    Platform-agnostic core
#   engine/backends/           Platform-specific adapters (Claude Code / Codex)
#   plugins/<plugin>/workflows/  Per-plugin YAML state machine definitions
#   plugins/<plugin>/agents/     Per-plugin agent specs

__version__ = "1.0.0"
