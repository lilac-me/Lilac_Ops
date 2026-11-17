#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# discover_fused_patchers.py — 定位框架内"手写"的 RMSNorm / RoPE
#
# 动机：框架常 vendor 自己的模块副本（类名/函数名与 transformers 同名但
# __module__ 不同）。按类名去改会打到框架实际不用的类 = 热环上的死代码，
# 端到端无收益。本脚本以【语义特征】把副本找全：
#   - RMSNorm 类：forward 体含手写 rsqrt 且不含 npu_rms_norm
#   - RoPE 函数：函数体含 rotate_half 且不含 npu_rotary_mul（含内联负切片形态）
#
# 用法：
#   python discover_fused_patchers.py --source-root <模型源码根>
#
# 输出是【定位清单】不是成品：文件:行号 + 签名 + RMSNorm 的 eps 属性名 +
# 与替换形态是否匹配。拿到后：
#   ① 按当前在跑的模型变体筛出真正在热环的（源码里常同时躺着多个 vendor 副本，
#      只有被加载的那个有收益）。不确定就在候选 forward 里临时加计数跑几步。
#   ② 直接改源码，两行的事（改法、数值口径、陷阱见 references/flow/2-optimize.md §4.3）：
#        RMSNorm:  x * torch.rsqrt(v + eps) * w   →  torch_npu.npu_rms_norm(x, w, epsilon=eps)[0]
#        RoPE:     (q*cos) + (rotate_half(q)*sin) →  torch_npu.npu_rotary_mul(q, cos, sin)
#
# 标 ✗ 的候选签名与替换形态不符（如 attention 函数被内联形态启发式捞进来），
# 改之前先看一眼原实现，不要照着套。
# ============================================================================
import argparse
import ast
import os
import sys

_FUSED_KERNELS = ("npu_rms_norm", "npu_rotary_mul")
# 首参的真实叫法比想象中杂：老版 transformers 用 x，LLaMA 融合版用 qkv，
# 自研常见 hidden_states / t。它不是硬门槛——不在表里只是记一条"疑似误报"。
_ROPE_FIRST_ARGS = ("q", "query", "query_states", "queries", "qkv",
                    "x", "t", "hidden_states", "inputs")


def _body_src(node) -> str:
    return "\n".join(ast.dump(n, include_attributes=False) for n in node.body)


def _is_handwritten_rmsnorm(node: ast.ClassDef) -> bool:
    body = _body_src(node)
    if any(k in body for k in _FUSED_KERNELS):
        return False
    # 只认 rsqrt。曾用 `sqrt + (mean|pow)` 兜 `x / torch.sqrt(...)` 变体，但语料 67 个模型实测
    # 真阳性 0、误报源 31（math.sqrt 与激活函数的 torch.pow 会撞上），故去掉。
    return "rsqrt" in body


def _is_handwritten_rope(node: ast.FunctionDef) -> bool:
    body = _body_src(node)
    if any(k in body for k in _FUSED_KERNELS):
        return False
    if "rotate_half" in body:
        return True
    # 内联形态（transformers 新版把 rotate_half 内联进 apply_rotary_pos_emb*）：
    # 仅当函数名含 rotary/rope/rotate 才启用 —— 否则按 cos/sin/chunk/cat 结构判
    # 会误报 timestep_embedding、大 forward，也会捞进 attention 函数。
    name = getattr(node, "name", "") or ""
    if ("rotary" in name.lower() or "rope" in name.lower()
            or "rotate" in name.lower()):
        return ("cos" in body and "sin" in body and "cat" in body
                and "USub" in body and ("Slice(" in body or "chunk" in body))
    return False


def _calls_named(node, names) -> bool:
    """函数体里是否调用了某个确切名字的函数（含 self.x / mod.x 形态）。
    用精确名而非子串 —— `_rotate_half_for_nemotron` 含 "rotate_half" 子串，
    但公式未必与标准 rotate_half 等价。"""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            fid = (f.id if isinstance(f, ast.Name)
                   else f.attr if isinstance(f, ast.Attribute) else None)
            if fid in names:
                return True
    return False


def _forward_args(cls: ast.ClassDef):
    for sub in cls.body:
        if isinstance(sub, ast.FunctionDef) and sub.name == "forward":
            return [a.arg for a in sub.args.args]
    return None


def _self_attrs(cls: ast.ClassDef):
    """读 __init__ 里 self.xxx = ... 的属性名 —— eps 属性名各框架不一
    （transformers 用 variance_epsilon，diffusers/自研常用 eps）。"""
    for sub in cls.body:
        if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
            names = {n.attr for n in ast.walk(sub)
                     if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                     and isinstance(n.value, ast.Name) and n.value.id == "self"}
            return sorted(names)
    return []


def _discover(source_root: str):
    """语义匹配（类名/函数名不是硬条件）。返回 [(relpath, lineno, name, kind, node)]"""
    if not source_root or not os.path.isdir(source_root):
        return []
    hits = []
    scan_root = source_root.rstrip("/\\")
    for dirpath, dirnames, filenames in os.walk(scan_root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, scan_root)
            try:
                with open(fp, encoding="utf-8") as f:
                    tree = ast.parse(f.read())
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if any(isinstance(s, ast.FunctionDef) and s.name == "forward"
                           for s in node.body) and _is_handwritten_rmsnorm(node):
                        hits.append((rel, node.lineno, node.name, "rmsnorm", node))
                elif isinstance(node, ast.FunctionDef):
                    if _is_handwritten_rope(node):
                        hits.append((rel, node.lineno, node.name, "rope", node))
    return sorted(hits, key=lambda h: (h[0], h[1]))


def _verdict(hit):
    """返回 (ok, 签名串, 备注)。ok=False 表示签名与替换形态不符，改前要先看原实现。"""
    _, _, name, kind, node = hit
    if kind == "rmsnorm":
        args = _forward_args(node)
        eps = [a for a in _self_attrs(node) if "eps" in a.lower()]
        note = f"eps 属性: self.{eps[0]}" if eps else "⚠ 未在 __init__ 找到 eps 相关属性"
        if args is None:
            return False, "?", "无 forward"
        ok = len(args) == 2          # (self, hidden_states) —— 替换函数同形态
        return ok, f"forward({', '.join(args)})", note
    args = [a.arg for a in node.args.args]
    sig = f"({', '.join(args)})" if args else "(*, ...)"
    if node.args.vararg or node.args.kwarg:
        return False, sig, "含 *args/**kwargs —— 不是定形参的 (q,cos,sin) 形态"
    # 3 参形态（x, cos, sin）也真实存在；只在函数名明说 rotary/rope 时接受，
    # 否则 3 参的普通函数太多，会淹掉真候选。
    named_rope = any(k in (getattr(node, "name", "") or "").lower()
                     for k in ("rotary", "rope", "rotate"))
    if len(args) < 3 or (len(args) < 4 and not named_rope):
        return False, sig, "参数不足 4 个（且函数名未含 rotary/rope）—— 不是 (q,k,cos,sin) 形态"
    if args[0].lower().lstrip("_") not in _ROPE_FIRST_ARGS:
        return False, sig, "首参不是输入张量（q/query/qkv/x/…）—— 疑似误报（如 attention 函数）"
    if not _calls_named(node, ("rotate_half",)):
        return True, sig, "⚠ 未调标准 rotate_half（内联或自定义实现）—— 替换前确认公式等价"
    # 「半宽 cos/sin」陷阱：上游 flash-attn 路径会把 cos/sin 砍成 chunk(2)[0]，
    # 而 npu_rotary_mul 要全宽（rotate_half 语义）——形状相同、用错静默算错。
    # 这类函数同时在解包里 chunk，是唯一能静态识别的信号。
    if "chunk" in _body_src(node):
        return True, sig, ("⚠ 函数体内含 chunk —— 确认 cos/sin 是否已被砍成半宽"
                           "（npu_rotary_mul 要全宽；详见 references/flow/1-migrate.md §3.4）")
    return True, sig, ""


def main():
    ap = argparse.ArgumentParser(
        description="Locate hand-written RMSNorm/RoPE in a model source tree "
                    "(semantic matching, not name-prefix).")
    ap.add_argument("--source-root", required=True,
                    help="model source root to scan (e.g. /path/to/cosmos_framework)")
    args = ap.parse_args()

    if not os.path.isdir(args.source_root):
        print(f"错误：目录不存在：{args.source_root}", file=sys.stderr)
        return 1

    hits = _discover(args.source_root)
    print(f"候选 {len(hits)} 个（目标：{os.path.abspath(args.source_root)}）")
    print("按当前在跑的模型变体筛出热环上的那些；✗ 的改前先看原实现。\n")

    n_ok = n_warn = n_bad = 0
    for hit in hits:
        rel, lineno, name, kind, _ = hit
        ok, sig, note = _verdict(hit)
        if not ok:
            n_bad += 1
        elif note.startswith("⚠"):
            n_warn += 1
        else:
            n_ok += 1
        print(f"  [{'✓' if ok else '✗'}] {rel}:{lineno}")
        print(f"      {name}  [{kind}]  {sig}" + (f"  {note}" if note else ""))

    print(f"\n可直接改 {n_ok}；需先确认公式 {n_warn}；签名不符 {n_bad}（共 {len(hits)}）。")
    print("改法：references/flow/2-optimize.md §4.3 —— 直接改源码，RMSNorm / RoPE 各两行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
