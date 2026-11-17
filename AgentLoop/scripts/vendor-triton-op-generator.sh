#!/usr/bin/env bash
#
# vendor-triton-op-generator.sh — 通过 cannbot 统一安装器 vendor 性能调优 skill
#
# 上游只有一个仓：cannbot。它的 triton-op-generator 自带安装器：
#   plugins-official/triton-op-generator/init.sh
#     → script/bin/source-plugin-init.sh
#       → script/bin/cannbot.js install triton-op-generator --tool claude --target <dir>
# 由安装器自己完成「拉取其它仓库 + 安装插件」：
#   · 共享 skill 仓：cannbot 的 vendor/cannbot-skills submodule（可用 --skills <SHA|本地路径> 覆盖）
#   · 插件依赖仓：  plugin-install.json 声明的 cannbot-knowledge → <target>/.cannbot/dependencies/
#   · skill 以软链装到 <target>/.claude/skills/<name>
#   · 插件资产装到 <target>/.cannbot/plugins/triton-op-generator/（含 workflows/render_session.py）
# 本脚本**不直接 clone cannbot-skills**，只在安装完成后从安装产物取「性能调优模块」快照。
#
# 语义（对齐 docs/agent_workflow/profile-driven-triton-opt_design.md §4.4）：
#   - 只 vendor 调优 skill（verifier / latency-optimizer / simulator-optimizer /
#     precision-debug / npu-arch / mfu-analyzer）+ 会话渲染器；extractor/designer/coding 不入列。
#   - 幂等；可重跑覆盖旧副本；安装失败时明确报错并提示手动降级变量。
#   - **不改写任何 vendor 出来的内容**（字节保真，便于与上游逐字节核对）。
#
# 输入（环境变量，均可选）：
#   CANN_BOT_REPO        cannbot 仓远端，默认 https://gitcode.com/cann/cannbot.git
#   CANN_BOT_TAG         固定 ref（tag/sha/分支）；不设则跟随默认分支
#   CANN_BOT_ROOT        已检出的 cannbot 目录（设了则跳过自动拉取，离线可用）
#   CANN_BOT_DIR         自动拉取用的本地 cache，默认 <repo>/.vendor_src/cannbot
#   CANN_BOT_SKILLS      透传给安装器的 --skills：共享 skill 仓的 SHA 或本地路径
#                        （不设 = 用 cannbot 仓内 submodule 并跟随当前分支）
#   CANN_BOT_INSTALL_DIR 安装器落点（临时工程目录），默认 <repo>/.vendor_src/cannbot_install
#   CANNBOT_SKIP_DEPENDENCY_REPOS  透传给安装器；=1 时跳过 cannbot-knowledge 克隆
#
# 输出：
#   skills/ops-skills/vendored-triton-op/<skill>/      6 个调优 skill 快照
#   skills/ops-skills/vendored-triton-op/render_session.py
#   VENDORED_TRITON.json                               来源仓 + sha + 各 skill 版本 + 安装记录 + 时间

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/skills/ops-skills/vendored-triton-op"
PLUGIN_ID="triton-op-generator"
PLUGIN_REL="plugins-official/$PLUGIN_ID"

CANN_BOT_REPO="${CANN_BOT_REPO:-https://gitcode.com/cann/cannbot.git}"
CANN_BOT_TAG="${CANN_BOT_TAG:-}"
CANN_BOT_ROOT="${CANN_BOT_ROOT:-}"
CANN_BOT_DIR="${CANN_BOT_DIR:-$REPO_ROOT/.vendor_src/cannbot}"
CANN_BOT_SKILLS="${CANN_BOT_SKILLS:-}"
CANN_BOT_INSTALL_DIR="${CANN_BOT_INSTALL_DIR:-$REPO_ROOT/.vendor_src/cannbot_install}"

# 性能调优模块 skill——只做调优，不重新生成算子（extractor/designer/coding 不入列）
# triton-mfu-analyzer：metric_mode=mfu 时 S4 的 MFU 测定通道
SKILLS=(triton-op-verifier triton-latency-optimizer triton-simulator-optimizer triton-precision-debug npu-arch triton-mfu-analyzer)

# ── 1. 取得 cannbot 检出（唯一上游）─────────────────────────────────────────
MODE="auto"
if [ -n "$CANN_BOT_ROOT" ]; then
  MODE="manual"
  SRC="$CANN_BOT_ROOT"
  [ -d "$SRC/$PLUGIN_REL" ] || {
    echo "FAIL: CANN_BOT_ROOT=$SRC 下无 $PLUGIN_REL（不是 cannbot 检出？）" >&2
    exit 1
  }
else
  command -v git >/dev/null 2>&1 || { echo "FAIL: 未安装 git 且未设 CANN_BOT_ROOT" >&2; exit 1; }
  SRC="$CANN_BOT_DIR"
  mkdir -p "$(dirname "$SRC")"
  if [ ! -d "$SRC/.git" ]; then
    echo "[vendor] clone $CANN_BOT_REPO -> $SRC"
    git clone --depth 1 "$CANN_BOT_REPO" "$SRC" || {
      echo "FAIL: 自动拉取 cannbot 失败（网络/gitcode 不可达）" >&2
      echo "      可选降级：设 CANN_BOT_ROOT=<已检出 cannbot 目录> 复用本地" >&2
      exit 1
    }
  else
    echo "[vendor] update $SRC"
    if ! git -C "$SRC" fetch origin --depth 1 >/dev/null 2>&1; then
      echo "FAIL: 更新 cannbot cache 失败（网络不可达）" >&2
      echo "      可选降级：设 CANN_BOT_ROOT=<已检出 cannbot 目录> 复用本地" >&2
      exit 1
    fi
  fi
  if [ -n "$CANN_BOT_TAG" ]; then
    echo "[vendor] checkout $CANN_BOT_TAG"
    git -C "$SRC" fetch origin "$CANN_BOT_TAG" --depth 1 >/dev/null 2>&1 \
      && git -C "$SRC" checkout FETCH_HEAD >/dev/null 2>&1 \
      || { echo "FAIL: 检出 ref=$CANN_BOT_TAG 失败" >&2; exit 1; }
  else
    echo -n "[vendor] checkout default branch … "
    DEFAULT_BRANCH="$(git -C "$SRC" remote show origin 2>/dev/null | sed -n 's/  HEAD branch: //p')"
    DEFAULT_BRANCH="${DEFAULT_BRANCH:-$(git -C "$SRC" rev-parse --abbrev-ref HEAD)}"
    git -C "$SRC" checkout "$DEFAULT_BRANCH" >/dev/null 2>&1 && \
      git -C "$SRC" reset --hard origin/"$DEFAULT_BRANCH" >/dev/null 2>&1 || \
      git -C "$SRC" pull --ff-only >/dev/null 2>&1 || true
    echo "at $(git -C "$SRC" rev-parse --short HEAD)"
  fi
fi

INIT_SH="$SRC/$PLUGIN_REL/init.sh"
[ -f "$INIT_SH" ] || { echo "FAIL: 上游缺安装器 $INIT_SH" >&2; exit 1; }
command -v node >/dev/null 2>&1 || {
  echo "FAIL: cannbot 安装器需要 node（script/bin/cannbot.js）；请先安装 Node.js" >&2
  exit 1
}

# ── 2. 跑上游安装器（由它拉取 cannbot-skills / cannbot-knowledge 并安装插件）──
# 安装到独立临时工程目录，避免污染本仓 .claude/。
SKILLS_ARGS=()
if [ -n "$CANN_BOT_SKILLS" ]; then
  SKILLS_ARGS=(--skills "$CANN_BOT_SKILLS")
fi
mkdir -p "$CANN_BOT_INSTALL_DIR"
echo "[vendor] install $PLUGIN_ID (tool=claude) -> $CANN_BOT_INSTALL_DIR"
if ! bash "$INIT_SH" project claude "$CANN_BOT_INSTALL_DIR" ${SKILLS_ARGS[@]+"${SKILLS_ARGS[@]}"}; then
  echo "FAIL: cannbot 安装器执行失败（$INIT_SH）" >&2
  echo "      常见原因：node 缺失、共享 skill 仓拉取失败（可设 CANN_BOT_SKILLS=<本地路径>）、" >&2
  echo "      依赖仓网络不可达（可设 CANNBOT_SKIP_DEPENDENCY_REPOS=1 跳过 cannbot-knowledge）" >&2
  exit 1
fi

INSTALLED_SKILLS_DIR="$CANN_BOT_INSTALL_DIR/.claude/skills"
for s in "${SKILLS[@]}"; do
  [ -e "$INSTALLED_SKILLS_DIR/$s" ] || {
    echo "FAIL: 安装器未装出 skill $s（$INSTALLED_SKILLS_DIR/$s 不存在）" >&2
    exit 1
  }
done

# 渲染器优先取安装产物（上游 Phase 6 实际调用的路径），缺失时回退源码树
RENDER_SESSION_SRC="$CANN_BOT_INSTALL_DIR/.cannbot/plugins/$PLUGIN_ID/workflows/render_session.py"
if [ ! -f "$RENDER_SESSION_SRC" ]; then
  RENDER_SESSION_SRC="$SRC/$PLUGIN_REL/workflows/render_session.py"
fi
[ -f "$RENDER_SESSION_SRC" ] || { echo "FAIL: 找不到 render_session.py（安装产物与源码树均无）" >&2; exit 1; }

CANN_BOT_PIN="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unavailable)"
echo "[vendor] vendor @ cannbot=$CANN_BOT_PIN -> $VENDOR_DIR"

# ── 3. 取快照（解引用软链；不改写任何内容）──────────────────────────────────
rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR"
for s in "${SKILLS[@]}"; do
  cp -RL "$INSTALLED_SKILLS_DIR/$s" "$VENDOR_DIR/$s"
done
cp "$RENDER_SESSION_SRC" "$VENDOR_DIR/render_session.py"

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 每个 skill 的版本指纹 = 该 skill 在其源仓 HEAD 的 git tree sha；
# 非 git 源（手动拷贝目录）退化为文件内容 sha256，保证指纹仍可复现。
SKILL_PINS=""
for s in "${SKILLS[@]}"; do
  real="$(readlink -f "$INSTALLED_SKILLS_DIR/$s")"
  root="$(git -C "$real" rev-parse --show-toplevel 2>/dev/null || true)"
  tree=""
  if [ -n "$root" ]; then
    rel="${real#"$root"/}"
    tree="$(git -C "$root" ls-tree HEAD "$rel" 2>/dev/null | awk '{print $3}')"
  fi
  [ -n "$tree" ] || tree="$(
    cd "$real" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
  )"
  SKILL_PINS="$SKILL_PINS$s=$tree
"
done
RENDER_REL="$PLUGIN_REL/workflows/render_session.py"   # 上游源码树中的相对路径
RENDER_TREE="$(sha256sum "$RENDER_SESSION_SRC" | awk '{print $1}')"

# 安装器写出的插件登记表（记录 cannbot / cannbot-skills 的实际 revision）
REGISTRY="$CANN_BOT_INSTALL_DIR/.claude/cannbot-plugin.json"

python3 - "$CANN_BOT_REPO" "$MODE" "$CANN_BOT_PIN" "${CANN_BOT_TAG:-default-branch}" "$TIMESTAMP" \
  "$SKILL_PINS" "$RENDER_TREE" "$RENDER_REL" "$PLUGIN_ID" "$REGISTRY" \
  > "$REPO_ROOT/VENDORED_TRITON.json" <<'PY'
import json
import os
import sys

(repo, mode, pin, ref, ts, skill_pins, render_tree, render_src, plugin_id,
 registry_path) = sys.argv[1:11]

skills = {}
for line in skill_pins.strip().splitlines():
    name, tree = line.split("=", 1)
    skills[name] = {"pin": tree, "version": tree[:12]}

revisions = {}
if os.path.isfile(registry_path):
    try:
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
        for record in registry.get("plugins", []):
            if record.get("plugin") == plugin_id:
                revisions = (record.get("source") or {}).get("revisions") or {}
                break
    except (OSError, ValueError):
        revisions = {}

manifest = {
    "source_repo": repo,
    "vendor_via": {
        "installer": "plugins-official/triton-op-generator/init.sh",
        "tool": "claude",
        "registry": ".claude/cannbot-plugin.json",
    },
    "mode": mode,
    "pinned": pin,
    "requested_ref": ref,
    "vendored_at": ts,
    "skills_dir": "skills/ops-skills/vendored-triton-op",
    "skills_source": revisions.get("cannbot-skills", {}),
    "skills": skills,
    "tools": {
        "render_session.py": {
            "src": render_src,
            "pin": render_tree,
            "version": render_tree[:12],
        }
    },
}
json.dump(manifest, sys.stdout, indent=2, ensure_ascii=False)
sys.stdout.write("\n")
PY

cat > "$VENDOR_DIR/README.md" <<'EOF'
# vendored-triton-op

本目录是 **cannbot 性能调优模块** 的版本固定快照（vendor 副本），由
`scripts/vendor-triton-op-generator.sh` 生成，只保留调优相关 skill（verifier /
latency-optimizer / simulator-optimizer / precision-debug / npu-arch / mfu-analyzer）+ 会话
归档渲染器 `render_session.py`。**不重新生成算子**——extractor/designer/coding 不入列。

## 来源与生成方式

上游只有一个仓：`cannbot`。它的 `triton-op-generator` 自带统一安装器
（`plugins-official/triton-op-generator/init.sh` → `script/bin/cannbot.js`），由安装器负责：

- 拉取共享 skill 仓 `cannbot-skills`（cannbot 的 `vendor/cannbot-skills` submodule，
  可用 `--skills <SHA|本地路径>` 覆盖）；
- 克隆插件依赖仓 `cannbot-knowledge` 到 `<安装目录>/.cannbot/dependencies/`；
- 把 skill 软链装到 `<安装目录>/.claude/skills/`，插件资产装到
  `<安装目录>/.cannbot/plugins/triton-op-generator/`。

本目录的脚本只是：拉 `cannbot` → 跑它的安装器（装到 `.vendor_src/cannbot_install`）→
从安装产物取上述 skill 与渲染器快照。**不直接 clone cannbot-skills，也不改写 vendor 内容。**

- 运行时与上游完全解耦：S4 只读取本目录副本。
- 升级：重跑 `scripts/vendor-triton-op-generator.sh`（可设 `CANN_BOT_TAG` 固定 cannbot 版本、
  `CANN_BOT_SKILLS` 固定 skill 仓版本）。
- 来源仓 / 锁定 sha / 各 skill 版本 / 安装器记录见仓库根 `VENDORED_TRITON.json`。
- 知识库（KB）不在本目录：需要 KB 卡片或 `knowledge-query` 时设 `CANNBOT_KNOWLEDGE_ROOT`
  指向 cannbot-knowledge 的 `knowledge` 目录，或复用安装器克隆的
  `.vendor_src/cannbot_install/.cannbot/dependencies/triton-op-generator/cannbot-knowledge/knowledge`。

> 勿直接在本目录手工修改；如需临时改动请改后回到上游维护，再重跑脚本回填。
EOF

echo "[vendor] done: $((${#SKILLS[@]})) skills vendored"
echo "[vendor] 清单: $REPO_ROOT/VENDORED_TRITON.json"
