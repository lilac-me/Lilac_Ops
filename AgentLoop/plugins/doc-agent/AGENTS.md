# Doc-Agent — Codex orchestration

Doc-Agent uses the same deterministic engine, contracts, rules, skills, and role prompts on Claude Code and Codex. The public entry point is the `doc-orchestrator` skill.

## Runtime root

Before invoking any engine or linter command, resolve the installed Doc-Agent plugin root and call it `<DOC_AGENT_ROOT>`:

- In an installed Codex plugin, start from the directory containing the loaded `skills/doc-orchestrator/SKILL.md`; the plugin root is two directories above that `doc-orchestrator` directory.
- In a source checkout, use `plugins/doc-agent` under the repository root.
- Accept the root only when `<DOC_AGENT_ROOT>/engine/doc_pipeline.yaml` and `<DOC_AGENT_ROOT>/rules/rules.yaml` both exist.
- Convert it to an absolute path and substitute it into commands; never hard-code a Codex cache path.

Claude Code may use `CLAUDE_PLUGIN_ROOT`, but all shared instructions express the resolved location as `<DOC_AGENT_ROOT>`.

## Role dispatch

- Read the exact role prompt from `<DOC_AGENT_ROOT>/agents/<role>/agent.md` before dispatching that role.
- Use Codex native subtask/subagent support when available. Give each task only the target repository, run workspace, input artifact paths, role prompt path, and required output contract.
- Keep semantic reviewers independent: do not provide one reviewer with another reviewer's findings.
- If native subtask support is unavailable, execute the roles sequentially in the orchestrator while preserving the same input/output contracts and handoff tokens.
- The engine alone evaluates transitions, gates, iteration counts, and scores; role prompts must not emulate that logic.

Role selection and review depth are defined by the `doc-orchestrator` and `doc-quality-check` skills. Agent definitions remain in `agents/*/agent.md`; Codex does not interpret their Claude-specific `tools` frontmatter.
