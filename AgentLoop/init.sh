#!/usr/bin/env bash
# Lightweight, project-only Skill projection. No network, no global config edits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
usage() {
  echo "Usage: bash init.sh project [claude|codex|opencode] [target_dir]"
  echo "Default: claude; target_dir = current working directory."
  echo "Registers AgentLoop skills only. Reuse separately installed CANNBot plugins."
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if (( $# > 3 )) || [[ "${1:-project}" != project ]]; then usage >&2; exit 2; fi
tool="${2:-claude}"
target="${3:-$PWD}"
case "$tool" in
  claude) rel=.claude/skills ;;
  codex) rel=.agents/skills ;;
  opencode) rel=.opencode/skills ;;
  *) usage >&2; exit 2 ;;
esac
[[ -d "$target" ]] || { echo "Target directory missing: $target" >&2; exit 2; }
target="$(cd "$target" && pwd -P)"
dest="$target/$rel"
sources=(
  "$ROOT/skills/agentloop-optimization-principles"
  "$ROOT/skills/agentloop-infer-opt"
  "$ROOT/skills/ops-skills/triton/profile-bottleneck-analyzer"
  "$ROOT/skills/inference-model-skills/deployment/vllm-ascend-serving"
  "$ROOT/skills/inference-model-skills/performance-optimization/vllm-ascend-benchmark"
  "$ROOT/skills/inference-model-skills/performance-optimization/ascend-profiling-collection"
)
# Check all inputs and name conflicts before creating any links.
for src in "${sources[@]}"; do
  name="${src##*/}"
  [[ -f "$src/SKILL.md" ]] || { echo "Missing skill: $src" >&2; exit 1; }
  if [[ -e "$dest/$name" || -L "$dest/$name" ]]; then
    [[ -L "$dest/$name" && "$(readlink "$dest/$name")" == "$src" ]] || {
      echo "Existing skill conflict, not overwritten: $dest/$name" >&2; exit 1;
    }
  fi
done
mkdir -p "$dest"
for src in "${sources[@]}"; do
  name="${src##*/}"
  [[ -L "$dest/$name" ]] || ln -s "$src" "$dest/$name"
done
echo "AgentLoop infer skills: $dest"
echo "Entry skill: agentloop-infer-opt"
echo "Mandatory policy skill: agentloop-optimization-principles"
echo "Next: cd into $target and start $tool."
echo "CANNBot dependencies (reuse existing installation):"
echo "  triton-op-generator"
echo "  tilelang2ascendc-ops-generator"
# Only check project-local hints: global/marketplace installs may not live here.
if [[ -f "$dest/triton-latency-optimizer/references/Index.md" ]]; then
  echo "  Triton latency index found: $dest/triton-latency-optimizer/references/Index.md"
else
  echo "  Triton project-local latency index not detected; check the installed plugin."
fi
if [[ -f "$dest/tilelang2ascend-translator/SKILL.md" || -f "$dest/tilelang2ascend-operator-project-init/SKILL.md" ]]; then
  echo "  Ascend C project-local skill detected."
else
  echo "  Ascend C project-local skills not detected; check the installed plugin."
fi
echo "No CANNBot installation or NPU tests were performed by this command."
