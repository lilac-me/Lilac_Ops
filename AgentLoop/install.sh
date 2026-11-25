#!/usr/bin/env bash
# Install the tuning dependencies; optionally expose only the required skills.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$ROOT/skills/ops-skills/vendored-triton-op"
LOCAL="$ROOT/skills/ops-skills/triton"
local_skills=(profile-bottleneck-analyzer triton-op-source-assembler profile-driven-triton-opt)
vendor_skills=(triton-op-verifier triton-latency-optimizer triton-simulator-optimizer triton-precision-debug npu-arch triton-mfu-analyzer)
if (( $# > 2 )); then
  echo "Usage: bash install.sh [model_root [claude|codex]]" >&2; exit 2
fi
target="${1:-}"
tool="${2:-claude}"
case "$tool" in
  claude) skill_rel=.claude/skills ;;
  codex) skill_rel=.agents/skills ;;
  *) echo "Unsupported Agent: $tool" >&2; exit 2 ;;
esac
if [[ -n "$target" && ! -d "$target" ]]; then
  echo "Model directory does not exist: $target" >&2; exit 2
fi
missing=0
for skill in "${vendor_skills[@]}"; do
  [[ -f "$VENDOR/$skill/SKILL.md" ]] || missing=1
done
[[ -f "$VENDOR/triton-latency-optimizer/references/Index.md" && -f "$VENDOR/render_session.py" ]] || missing=1
if (( missing )); then
  bash "$ROOT/scripts/vendor-triton-op-generator.sh"
fi
for skill in "${vendor_skills[@]}"; do
  [[ -f "$VENDOR/$skill/SKILL.md" ]] || { echo "Missing vendor skill: $skill" >&2; exit 1; }
done
[[ -f "$VENDOR/triton-latency-optimizer/references/Index.md" ]] || { echo "Missing latency Index.md" >&2; exit 1; }
if [[ -n "$target" ]]; then
  dest="$(cd "$target" && pwd)/$skill_rel"
  mkdir -p "$dest"
  # Preflight all paths before creating links; never overwrite existing skills.
  for src in "${local_skills[@]/#/$LOCAL/}" "${vendor_skills[@]/#/$VENDOR/}"; do
    name="${src##*/}"
    [[ -f "$src/SKILL.md" ]] || { echo "Missing skill: $src" >&2; exit 1; }
    if [[ -e "$dest/$name" || -L "$dest/$name" ]]; then
      [[ -L "$dest/$name" && "$(readlink "$dest/$name")" == "$src" ]] || {
        echo "Existing skill conflict: $dest/$name" >&2; exit 1;
      }
    fi
  done
  for src in "${local_skills[@]/#/$LOCAL/}" "${vendor_skills[@]/#/$VENDOR/}"; do
    name="${src##*/}"
    [[ -L "$dest/$name" ]] || ln -s "$src" "$dest/$name"
  done
  echo "Skills installed: $dest"
fi
echo "AgentLoop profile optimization dependencies ready."
