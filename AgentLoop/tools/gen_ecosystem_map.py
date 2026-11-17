# -*- coding: utf-8 -*-
"""Generate the MindSpeed ecosystem knowledge-graph map (docs/ecosystem-map.png).

Renders three swim lanes (community repos / MindSpeed-Agent plugins / skill
domains) plus a golden novice-utterance routing example on top.

Layout is computed, not hand-tuned: each lane auto-stacks its node boxes
top-down with equal gaps and asserts (a) no vertical overlap, (b) every box
stays inside its lane. Any font/text change that would collide fails loudly
instead of silently producing an unreadable image.

Usage:
    python tools/gen_ecosystem_map.py            # writes docs/ecosystem-map.png

Requirements: matplotlib with a CJK font (Microsoft YaHei / SimHei).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

AVAILABLE = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Microsoft YaHei", "SimHei", "DengXian", "SimSun"] if f in AVAILABLE), None)
if FONT:
    plt.rcParams["font.sans-serif"] = [FONT]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "docs", "ecosystem-map.png")

C_TXT = "#1F2733"
C_GRAY = "#8A93A3"
FIG_W_IN, DPI = 16.0, 150

def _glyph_w(ch, fs_pt, pp):
    return pp * fs_pt * (1.0 if ord(ch) > 0x2E80 or ch in "·—→↔" else 0.55)

def text_px(text, fs):
    pp = DPI / 72.0
    lines = text.split("\n")
    maxw = max(sum(_glyph_w(c, fs, pp) for c in ln) for ln in lines)
    h = len(lines) * fs * pp * 1.45
    return maxw, h

def box_px(w_frac, h_frac):
    return w_frac * FIG_W_IN * DPI, h_frac * FIG_W_IN * DPI

WARNINGS = []
BOXES = []  # (cx, cy, w, h) in figure fractions, for overlap assertions

def draw_box(ax, cx, cy, w, h, fc, ec, text, fs=10, tc=C_TXT, lw=1.3,
             weight="normal", style="round,pad=0.008,rounding_size=0.012"):
    tw, th = text_px(text, fs)
    bw, bh = box_px(w, h)
    if tw > bw * 0.95 or th > bh * 0.95:
        WARNINGS.append(f"TEXT OVERFLOW @({cx:.3f},{cy:.3f}) fs={fs} "
                        f"text=({tw:.0f}x{th:.0f}) box=({bw:.0f}x{bh:.0f}) :: {text!r}")
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight=weight, linespacing=1.45, zorder=4)
    BOXES.append((cx, cy, w, h))


def arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.1, ls="-", connectionstyle=None,
          shrinkA=1, shrinkB=1, zorder=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls, zorder=zorder,
                                 shrinkA=shrinkA, shrinkB=shrinkB,
                                 connectionstyle=connectionstyle or "arc3,rad=0"))


def stack_lane(ax, lx, lw, top, bot, header, nodes):
    """Auto-stack node boxes inside a lane; returns center-of records.

    nodes: list of (label, sub, fs, fc, ec). Box width = lane width - margin;
    box height = text height + padding; equal vertical gaps between boxes.
    """
    cx = lx + lw / 2
    w_box = lw - 0.035
    avail = top - bot
    # header occupies top 0.05 of lane
    header_h = 0.040
    body_top = top - header_h
    body = body_top - bot
    # precompute heights
    heights = []
    for label, sub, fs, fc, ec in nodes:
        text = label if sub is None else f"{label}\n({sub})"
        _, th = text_px(text, fs)
        hh = max(0.050, th / (FIG_W_IN * DPI) + 0.012)
        heights.append(hh)
    total_h = sum(heights)
    gap = (body - total_h) / (len(nodes) + 1) if len(nodes) > 1 else (body - total_h) / 2
    if gap < 0.006:
        WARNINGS.append(f"LANE STACK TOO TIGHT: {header} gap={gap:.4f} total_h={total_h:.4f} body={body:.4f}")
        gap = 0.006
    centers = []
    cy = body_top - gap
    for (label, sub, fs, fc, ec), hh in zip(nodes, heights):
        cy -= hh / 2
        text = label if sub is None else f"{label}\n({sub})"
        draw_box(ax, cx, cy, w_box, hh, fc, ec, text, fs=fs)
        centers.append((cy, label.split("\n")[0].strip()))
        cy -= hh / 2 + gap
    return cx, centers


def assert_no_overlap():
    bad = []
    for i, (cx1, cy1, w1, h1) in enumerate(BOXES):
        for cx2, cy2, w2, h2 in BOXES[i + 1:]:
            ox = min(cx1 + w1 / 2, cx2 + w2 / 2) - max(cx1 - w1 / 2, cx2 - w2 / 2)
            oy = min(cy1 + h1 / 2, cy2 + h2 / 2) - max(cy1 - h1 / 2, cy2 - h2 / 2)
            if ox > 0.002 and oy > 0.002:
                bad.append((cx1, cy1, cx2, cy2, ox, oy))
    return bad


def main():
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_W_IN * 0.60))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ============================== top golden example ========================
    q = 0.985
    draw_box(ax, 0.115, q, 0.165, 0.110, "#F5EEFF", "#7B3FF2",
             "新手一句话\n“我想优化\nmindspeed-llm 的模型”", fs=10, weight="bold")
    draw_box(ax, 0.345, q, 0.205, 0.110, "#FFF9DB", "#D29922",
             "① repo 消歧\naliases: ModelLink\n→ MindSpeed-LLM", fs=9.5)
    draw_box(ax, 0.600, q, 0.225, 0.110, "#FFF9DB", "#D29922",
             "② “优化”多义 → 反问\n(1) 训练/融合算子  (2) 模型迁入\n(3) 精度对齐", fs=8.6)
    draw_box(ax, 0.852, q, 0.215, 0.110, "#FFF9DB", "#D29922",
             "③ 命中插件 + skill\nmindspeed-agent →\nfsdp2-main-control 等", fs=9)
    for xa, xb in [(0.20, 0.237), (0.45, 0.485), (0.715, 0.743)]:
        arrow(ax, xa, q, xb, q, color="#B7791F", lw=1.5)
    ax.text(0.5, 1.0 - 0.016, "知识图谱驱动的意图路由示例", ha="center", va="center",
            fontsize=13.5, color=C_TXT, fontweight="bold")

    # ============================== swim lanes ================================
    lane_top, lane_bot = 0.805, 0.075
    lanes = [(0.012, 0.265, "#EAF2FE", "#1A73E8", "社区代码仓 Repo"),
             (0.330, 0.310, "#FFF4E0", "#F29900", "MindSpeed-Agent 插件 Plugin"),
             (0.665, 0.285, "#E9F7EC", "#2E7D32", "Skill 能力域 skills/")]
    for lx, lw, hf, he, hdr in lanes:
        ax.add_patch(FancyBboxPatch((lx, lane_bot), lw, lane_top - lane_bot,
                                    boxstyle="round,pad=0.004,rounding_size=0.005",
                                    fc="#FBFBFD", ec=C_GRAY, lw=1.0, zorder=1))
        draw_box(ax, lx + lw / 2, lane_top - 0.020, lw - 0.015, 0.040, hf, he, hdr, fs=10.5, weight="bold")

    # ---- lane A: repos ----
    repo_nodes = [
        ("MindSpeed-LLM\n(LLM 训练套件 · alias ModelLink)", "mcore / fsdp2 / mindspore", 8.6, "#EAF2FE", "#1A73E8"),
        ("MindSpeed-Bridge\n(开源模型桥接迁移)", "mcore · HF→NPU", 8.6, "#EAF2FE", "#1A73E8"),
        ("MindSpeed-MM\n(多模态训练 · 规划中)", "多模态", 8.6, "#EAF2FE", "#1A73E8"),
        ("MindSpeed\n(训练加速库)", "megatron 适配", 8.6, "#EAF2FE", "#1A73E8"),
        ("vllm-ascend\n(昇腾推理引擎)", "部署 / benchmark / profiling", 8.6, "#EAF2FE", "#1A73E8"),
        ("torch-npu / msprobe / triton-ascend\n等工具链", "环境 / 定位 / 算子", 8.4, "#EAF2FE", "#1A73E8"),
    ]
    ax_cx, repo_centers = stack_lane(ax, *lanes[0][:2], lane_top, lane_bot, "repo", repo_nodes)

    # ---- lane B: plugins (mindspeed-agent gets an intentionally tall box) ----
    # To keep the dominant plugin visually grouped, stack normal boxes first in
    # a compact lower band and give mindspeed-agent the upper band as one box.
    lx, lw = lanes[1][0], lanes[1][1]
    bx_cx = lx + lw / 2
    sub_nodes = [
        ("inference-model-optimize\n(推理优化流水线)", None, 8.2, "#FFF4E0", "#F29900"),
        ("ops\n(算子开发流水线)", None, 9.4, "#FFF4E0", "#F29900"),
        ("doc-agent\n(资料生成 / 审查)", None, 9.0, "#FFF4E0", "#F29900"),
    ]
    _bx_cx, sub_centers = stack_lane(ax, lx, lw, 0.475, lane_bot, "plugin-lower", sub_nodes)
    # mindspeed-agent tall box in the upper band (below the lane header box)
    _, thm = text_px("mindspeed-agent\n(根插件 · 通用入口)\n\nllm 训练技能族\nbridge 技能族\nvllm 通用技能族\n(agent 编排入口)", 9.2)
    tall_h = max(0.230, thm / (FIG_W_IN * DPI) + 0.030)
    tall_cy = lane_top - 0.042 - tall_h / 2  # header bottom sits at ~lane_top-0.040
    draw_box(ax, bx_cx, tall_cy, lw - 0.030, tall_h, "#FFF4E0", "#F29900",
             "mindspeed-agent\n(根插件 · 通用入口)\n\nllm 训练技能族\nbridge 技能族\nvllm 通用技能族\n(agent 编排入口)",
             fs=9.2)
    plugin_centers = [(tall_cy, "mindspeed-agent")] + sub_centers

    # ---- lane C: skill domains ----
    skill_nodes = [
        ("skills/llm-skills", "模型迁移 · FSDP2 适配\n精度对齐 · 训练验证", 8.6, "#E9F7EC", "#2E7D32"),
        ("skills/inference-model-skills", "部署 · 模型分析 · benchmark\nprofiling · autotune · 验收", 8.4, "#E9F7EC", "#2E7D32"),
        ("skills/ops-skills", "Ascend C / Triton / TileLang\n设计 · 测试 · profiling", 8.6, "#E9F7EC", "#2E7D32"),
        ("skills/doc-skills", "doc-orchestrator\n生成 / 更新 / 质量审查", 8.6, "#E9F7EC", "#2E7D32"),
    ]
    sx_cx, skill_centers = stack_lane(ax, *lanes[2][:2], lane_top, lane_bot, "skill", skill_nodes)

    # ================================ relations ===============================
    def center_of(name, coll):
        for cy, n in coll:
            if n == name:
                return cy
        return 0.5

    # repo -> plugin
    ax_r = lanes[0][0] + lanes[0][1] - 0.006
    bx_l = lanes[1][0] + 0.006
    for ry, label in repo_centers:
        if "LLM" in label or "Bridge" in label or "MM" in label or label == "MindSpeed":
            target = "mindspeed-agent"
        elif "vllm-ascend" in label:
            target = "inference-model-optimize"
        else:
            target = "ops"
        arrow(ax, ax_r, ry, bx_l, center_of(target, plugin_centers), color=C_GRAY, lw=0.9)
    # doc-agent serves ANY repo (dashed, from lane A top margin)
    arrow(ax, ax_r, lane_top - 0.050, bx_l, center_of("doc-agent", plugin_centers),
          color=C_GRAY, lw=0.8, ls=(0, (3, 3)))

    # plugin -> skill domain
    bx_r = lanes[1][0] + lanes[1][1] - 0.006
    sx_l = lanes[2][0] + 0.006
    ps_edges = [
        ("mindspeed-agent", "skills/llm-skills"),
        ("mindspeed-agent", "skills/inference-model-skills"),
        ("mindspeed-agent", "skills/ops-skills"),
        ("inference-model-optimize", "skills/inference-model-skills"),
        ("ops", "skills/ops-skills"),
        ("doc-agent", "skills/doc-skills"),
    ]
    for pname, sname in ps_edges:
        arrow(ax, bx_r, center_of(pname, plugin_centers), sx_l, center_of(sname, skill_centers),
              color=C_GRAY, lw=0.9)

    # golden-route highlight: user intent -> MindSpeed-LLM -> mindspeed-agent -> llm-skills
    arrow(ax, ax_r, center_of("MindSpeed-LLM", repo_centers),
          bx_l, center_of("mindspeed-agent", plugin_centers), color="#D29922", lw=2.4)
    arrow(ax, bx_r, center_of("mindspeed-agent", plugin_centers),
          sx_l, center_of("skills/llm-skills", skill_centers), color="#D29922", lw=2.4)

    ax.text(0.5, 0.040,
            "关系依据现有 plugin.json 与 SKILL.md 引用整理（示意）；权威清单见 catalog/repos.yaml（M0）与 docs/ecosystem-knowledge-graph-design.md",
            ha="center", va="center", fontsize=9, color=C_GRAY)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")

    print("written:", OUT)
    overlaps = assert_no_overlap()
    if overlaps:
        print("OVERLAPS:")
        for o in overlaps:
            print(" -", o)
    else:
        print("no box overlaps")
    if WARNINGS:
        print("WARNINGS:")
        for wmsg in WARNINGS:
            print(" -", wmsg)
    else:
        print("no text overflow / layout warnings")


if __name__ == "__main__":
    main()
