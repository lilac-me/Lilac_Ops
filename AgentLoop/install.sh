#!/bin/bash
# MindSpeed-Agent one-click install - multi-platform, core + on-demand units
# =========================================================================
# Usage:
#   bash install.sh                      # Claude Code + core（默认）
#   bash install.sh claude-code          # Claude Code + core
#   bash install.sh codex                # Codex + core
#   bash install.sh dsh                  # DSH + core
#   bash install.sh all                  # 全部平台 + core
#   bash install.sh claude-code all      # Claude Code + 全部 unit（历史全量行为）
#   bash install.sh claude-code mindspeed-llm        # Claude Code + core + mindspeed-llm
#   bash install.sh dsh mindspeed-llm ops            # DSH + core + 两个领域 unit
#
# 安装架构（v2，docs/install-on-demand-design.md）：
#   - 默认只装 core（mindspeed-agent：ecosystem-routing + catalog）；
#   - 领域 unit（mindspeed-llm / inference-model-optimize / ops / doc-agent）
#     在路由命中后按需安装（显式 unit 参数 或 tools/install-unit.py）。
#
# High-level flow:
#   1. validate the requested platform + units, source Skills, required CLI;
#   2. build plugin-local skill views only for the units being installed
#      (delegated to tools/install-unit.py link);
#   3. register this checkout as a marketplace and install the requested units;
#   4. dsh: project only the requested units' skills into .dsh/skills.
set -e

cd "$(dirname "$0")"

PLATFORM="${1:-claude-code}"
# Second argument selects units: core (default) | all | space-separated unit ids.
UNIT_ARG="${2:-core}"
echo "=== MindSpeed-Agent Setup - platform: $PLATFORM | units: $UNIT_ARG ==="
echo ""

case "$PLATFORM" in
    claude-code|codex|dsh|all) ;;
    *) echo "ERROR: Unknown platform '$PLATFORM'. Use: claude-code, codex, dsh, all"; exit 1 ;;
esac

# ---- Preflight: repo root -------------------------------------------
if [ ! -f "CLAUDE.md" ] && [ ! -f "AGENTS.md" ]; then
    echo "ERROR: CLAUDE.md/AGENTS.md not found - script must be in repo root."
    echo "  Found: $(pwd)"
    exit 1
fi

# ---- Preflight: Python (install-unit.py dependency) ------------------
if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: 'python' not found. install-unit.py requires Python 3."
    exit 1
fi

# ---- Resolve units to install ---------------------------------------
# core is always installed. "all" = core + every catalog unit.
resolve_units() {
    if [ "$UNIT_ARG" = "all" ]; then
        # every unit id declared in catalog/plugins.yaml (excluding core itself)
        python - <<'PY'
import sys
sys.path.insert(0, ".")
from catalog.loader import load_yaml
plugins = load_yaml(__import__("pathlib").Path("catalog/plugins.yaml")).get("plugins", [])
print(" ".join(p["id"] for p in plugins if (p.get("install") or {}).get("unit") != "core"))
PY
    elif [ "$UNIT_ARG" = "core" ]; then
        echo ""
    else
        echo "$UNIT_ARG"
    fi
}
UNITS="$(resolve_units)"
echo "  units to install: core${UNITS:+ + $UNITS}"
echo ""

# ---- Preflight: skill directories exist ----------------------------
echo "[Preflight] Checking canonical skill directories..."
MISSING=""
for dir in \
    "skills/common/common/ecosystem-routing" \
    "skills/ops-skills/triton/ops-design" \
    "skills/ops-skills/triton/triton-op-dev" \
    "skills/ops-skills/tilelang/tilelang-design" \
    "skills/ops-skills/tilelang/tilelang-op-dev" \
    "skills/ops-skills/common_tools/ops-test-gen" \
    "skills/ops-skills/common_tools/ops-profiling" \
    "skills/ops-skills/common_tools/ops-boundary-survey" \
    "skills/inference-model-skills/model-analysis" \
    "skills/inference-model-skills/deployment" \
    "skills/inference-model-skills/performance-optimization" \
    "skills/inference-model-skills/batch-invariance-validation" \
    "skills/rl-model-skills/alignment/true-on-policy-align" \
    "skills/doc-skills/common/doc-orchestrator" \
    "skills/doc-skills/generation/doc-design" \
    "skills/doc-skills/generation/doc-generate" \
    "skills/doc-skills/generation/doc-update" \
    "skills/doc-skills/review/doc-quality-check"; do
    "skills/ecosystem-code-repository-npu-migration/model-adaptation/goal" \
    "skills/ecosystem-code-repository-npu-migration/model-adaptation/repair" \
    "skills/ecosystem-code-repository-npu-migration/model-analysis/feature-model" \
    "skills/ecosystem-code-repository-npu-migration/deployment/env-setup" \
    "skills/ecosystem-code-repository-npu-migration/deployment/vllm-ascend-env" \
    "skills/ecosystem-code-repository-npu-migration/common/local-assets" \
    "skills/ecosystem-code-repository-npu-migration/common/report-pr" \
    "skills/ecosystem-code-repository-npu-migration/common/docs-support" \
    "skills/veomni-skills/veomni-mfu-calculation" \
    "skills/veomni-skills/veomni-model-adaptation" \
    "skills/veomni-skills/veomni-model-analysis" \
    "skills/veomni-skills/veomni-npu-environment" \
    "skills/veomni-skills/veomni-parallel-recipe" \
    "skills/veomni-skills/veomni-performance-tuning" \
    "skills/veomni-skills/veomni-precision-alignment" \
    "skills/veomni-skills/veomni-quick-migration"; do
            exit 1
        fi
        ;;
esac
case "$PLATFORM" in
    codex|all)
        if ! command -v codex >/dev/null 2>&1; then
            echo "ERROR: 'codex' CLI not found in PATH."
            exit 1
        fi
        ;;
esac
case "$PLATFORM" in
    dsh|all)
        if ! command -v dsh >/dev/null 2>&1; then
            echo "ERROR: 'dsh' CLI not found in PATH."
            echo "  Install DeepSeek Harness: npm install -g @deepseek-ai/dsh"
            exit 1
        fi
        ;;
esac
echo ""

# ====================================================================
# Step 1/2: Build plugin-local skill views for the requested units
# ====================================================================
echo "[1/2] Building skill views for units..."
if [ -n "$UNITS" ]; then
    for unit in $UNITS; do
        echo "  -> link views for unit: $unit"
        python tools/install-unit.py link "$unit"
    done
else
    echo "  core only: no plugin-local skill views needed (root plugin reads ./skills/common)."
fi
echo ""

# ====================================================================
# DSH: flat projection of ONLY the requested units into .dsh/skills
# ====================================================================
case "$PLATFORM" in
    dsh|all)
        echo "[DSH] Projecting skills into .dsh/skills ..."
        # core is always installed first (root plugin skills/common)
        python tools/install-unit.py dsh mindspeed-agent
        if [ -n "$UNITS" ]; then
            for unit in $UNITS; do
                python tools/install-unit.py dsh "$unit"
            done
        fi
        echo ""
        ;;
esac

# ====================================================================
# Step 2/2: Install plugins
# ====================================================================
case "$PLATFORM" in
    claude-code|codex|all)
        echo ""
        echo "[2/2] Installing plugins for: $PLATFORM"
        ;;
esac

# --- Claude Code ---
case "$PLATFORM" in
    claude-code|all)
        echo "  --- Claude Code ---"

        echo "  Adding marketplace..."
        if ! claude plugin marketplace add "$PWD"; then
            echo "ERROR: Failed to add the MindSpeed-Agent marketplace for Claude Code."
            echo "  An existing marketplace may point to another checkout; resolve it explicitly and retry."
            exit 1
        fi

        echo "  Installing core (mindspeed-agent)..."
        claude plugin install mindspeed-agent@MindSpeed-Agent

        if [ -n "$UNITS" ]; then
            for unit in $UNITS; do
                echo "  Installing $unit..."
                if [ "$unit" = "mindspeed-scan-issue-agent" ]; then
                    python tools/install-unit.py claude "$unit" --run
                else
                    claude plugin install "$unit@MindSpeed-Agent"
                fi
            done
        fi

        # ecosystem-code-repository-npu-migration failure is reported explicitly, matching the Doc-Agent
        # handling above instead of relying on set -e alone.
        echo "  Installing ecosystem-code-repository-npu-migration..."
        if ! claude plugin install ecosystem-code-repository-npu-migration@MindSpeed-Agent; then
            echo "ERROR: Claude Code ecosystem-code-repository-npu-migration installation failed."
            exit 1
        fi

        echo "  Installing veomni-agent..."
        claude plugin install veomni-agent@MindSpeed-Agent

        echo "  Claude Code done."
        echo ""
        ;;
esac

# --- Codex ---
case "$PLATFORM" in
    codex|all)
        echo "  --- Codex ---"
        [ -f ".codex-plugin/plugin.json" ] && echo "  .codex-plugin/plugin.json found" || echo "  WARN: .codex-plugin/plugin.json not found"
        [ -f "AGENTS.md" ] && echo "  AGENTS.md found" || echo "  WARN: AGENTS.md not found"

        echo "  Adding marketplace..."
        if ! codex plugin marketplace add "$PWD"; then
            echo "ERROR: Failed to add the MindSpeed-Agent marketplace for Codex."
            echo "  An existing marketplace may point to another checkout; resolve it explicitly and retry."
            exit 1
        fi

        echo "  Installing core (mindspeed-agent)..."
        codex plugin add mindspeed-agent@MindSpeed-Agent || echo "  WARN: mindspeed-agent installation failed."

        if [ -n "$UNITS" ]; then
            for unit in $UNITS; do
                echo "  Installing $unit..."
                if [ "$unit" = "mindspeed-scan-issue-agent" ]; then
                    python tools/install-unit.py codex "$unit" --run
                else
                    codex plugin add "$unit@MindSpeed-Agent" || echo "  WARN: $unit installation failed."
                fi
            done
        fi

        # ecosystem-code-repository-npu-migration is a first-class Codex plugin and is required by this
        # installer (the Claude Code path installs it separately).
        echo "  Installing ecosystem-code-repository-npu-migration..."
        if ! codex plugin add ecosystem-code-repository-npu-migration@MindSpeed-Agent; then
            echo "ERROR: Codex ecosystem-code-repository-npu-migration installation failed."
            exit 1
        fi
        echo "  ecosystem-code-repository-npu-migration installed for Codex."

        echo "  Installing veomni-agent..."
        codex plugin add veomni-agent@MindSpeed-Agent || echo "  WARN: veomni-agent installation failed."

        echo "  Codex done."
        echo ""
        ;;
esac

# ====================================================================
# Common successful completion message
echo "=== Done. ==="
echo ""
case "$PLATFORM" in
    claude-code) echo "Claude Code: launch and describe your task" ;;
    codex)       echo "Codex:  launch codex . in this directory" ;;
    dsh)         echo "DSH:  launch dsh web in this directory" ;;
    all)         echo "Claude Code: launch and describe your task"
                 echo "Codex:        launch codex . in this directory"
                 echo "DSH:          launch dsh web in this directory" ;;
esac
echo ""
echo "提示：本次安装 unit: core${UNITS:+ + $UNITS}。"
echo "  - 领域 unit 按需安装：python tools/install-unit.py status"
echo "  - Claude Code/Codex 安装新插件后需【新开会话】方可加载其 skills。"
