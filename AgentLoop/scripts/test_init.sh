#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for tool in claude codex opencode; do
  target="$TMP/$tool project"
  mkdir "$target"
  bash "$ROOT/init.sh" project "$tool" "$target"
  bash "$ROOT/init.sh" project "$tool" "$target"
  case "$tool" in claude) rel=.claude/skills;; codex) rel=.agents/skills;; opencode) rel=.opencode/skills;; esac
  test -L "$target/$rel/agentloop-train-opt"
  test -f "$target/$rel/agentloop-train-opt/SKILL.md"
done
target="$TMP/conflict"
mkdir -p "$target/.claude/skills/agentloop-train-opt"
echo original > "$target/.claude/skills/agentloop-train-opt/SKILL.md"
if bash "$ROOT/init.sh" project claude "$target"; then
  echo "Expected collision to fail" >&2; exit 1
fi
test "$(cat "$target/.claude/skills/agentloop-train-opt/SKILL.md")" = original
echo "init smoke test passed"
