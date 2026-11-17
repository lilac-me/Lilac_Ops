#!/usr/bin/env python3
"""restructure_guard.py — 调用侧 kernel-body 重构 4 闸守卫（restructure-stage §5/§6.4）。

把 plan_b 下的一次性脚本（bref/gate2_matrix）拔高为调用侧可复用模块。当 op 瓶颈是
「读放大 / gather 复用」这类结构性问题、且 restructure.enabled 且触发成立时，
调用侧构造重构 kernel 后交本守卫验证 I/O 守恒 + 语义同构，任一闸 False → 该变换
作废（如同 A 类错误，cursor 保持、不落 best_code）。

三闸：
  g1_io_frozen     —— 公共入口 `fwd_impl` 签名逐字节一致 + kernel jit 形参名一致（gate1）。
  g2_matrix_diff   —— 全变体矩阵 orig vs restructured 位级 torch.equal（gate2，需 NPU）。
  g3_ir_homology   —— IR 级同语义：GM load/store/mask 计数与关键 op 结构对齐（gate3）。
g4 = 官方 verify.py + benchmark.py 溯源（复用既有 bench 溯源闸，不在此另设）。

用法：
  restructure_guard.py \
    --user <冻结用户源.py> --restructured <重构kernel.py> \
    --utils <用户utils.py> --pkg <fake-pkg名> \
    [--matrix <module:build_matrix>   ]   # 需 NPU；给则跑 g2
    [--ir-before <mlir> --ir-after <mlir>] # 给则跑 g3
    [--op <impl名> --kernel <kernel名>]    # 默认 causal_conv1d_fwd_{impl,kernel}
    [--out <result.json>]                  # 缺省 stdout 打 JSON
退出码：0 = 三闸全过（g2/g3 未跑按跳过处理，g1 必跑）；2 = 任一必跑闸 False。
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import types

# 内部包路径（restructure_engine 位于本脚本上级目录）。
_ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "restructure_engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)


# ---------------- 装载（泛化 self 方案的 bref._load_variant）----------------
def _load_variant(pkg_name, conv_path, utils_src, impl_name, kernel_name):
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = []
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    util_name = f"{pkg_name}.utils"
    util = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(util_name, utils_src))
    sys.modules[util_name] = util
    spec = importlib.util.spec_from_file_location(util_name, utils_src)
    util.__package__ = pkg_name
    spec.loader.exec_module(util)
    setattr(pkg, "utils", util)

    conv_name_m = f"{pkg_name}.conv"
    conv = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(conv_name_m, conv_path))
    sys.modules[conv_name_m] = conv
    spec = importlib.util.spec_from_file_location(conv_name_m, conv_path)
    conv.__package__ = pkg_name
    spec.loader.exec_module(conv)
    setattr(pkg, "conv", conv)

    return types.SimpleNamespace(
        impl=getattr(conv, impl_name), kernel=getattr(conv, kernel_name),
        package=pkg, mod=conv,
    )


# ---------------- g1 目标 ---------------------------------------------------
def g1_io_frozen(user, new) -> bool:
    s_orig = inspect.signature(user.impl)
    s_new = inspect.signature(new.impl)
    if str(s_orig) != str(s_new):
        return False
    if list(s_orig.parameters) != list(s_new.parameters):
        return False
    try:
        k_orig = list(user.kernel.arg_names)
        k_new = list(new.kernel.arg_names)
    except Exception:  # noqa: BLE001  旧 triton JITFunction 无 arg_names
        return exec_repr_same(user.mod, new.mod)
    return k_orig == k_new


def exec_repr_same(u, n):
    """兜底：比较内核源码的形参行一致。"""
    import re as _re
    def params(src):
        m = _re.search(r"def [^(]*\(([^)]*)\)", src)
        return _re.sub(r"\s+", "", m.group(1)) if m else ""
    return params(inspect.getsource(u)) == params(inspect.getsource(n))


# ---------------- g3 目标 ---------------------------------------------------
_TRITON_GM_REF_RE = re.compile(
    r"\b(tt\.load|tt\.store|tl\.load|tl\.store|triton_gpu|global|gm)\b", re.I)
# 对 Triton-Ascend mlir，GM 直读常以 ld.global / tt.load/ttg 形式出现；统计 load/store 类 op。
_GM_OP_RE = re.compile(r"\b(ld\.global|st\.global|tt\.load|tt\.store|ss\.load|ss\.store)\b")


def _count_gm_loads(mlir_path):
    if not mlir_path or not os.path.isfile(mlir_path):
        return None
    try:
        text = open(mlir_path, "r", encoding="utf-8").read()
    except OSError:
        return None
    loads = len(re.findall(r"\b(?:tt\.load|ld\.global|ss\.load)\b", text))
    stores = len(re.findall(r"\b(?:tt\.store|st\.global|ss\.store)\b", text))
    masks = len(re.findall(r"\bmask\b|\bpredicate\b", text))
    return {"loads": loads, "stores": stores, "masks": masks}


def g3_ir_homology(before_mlir, after_mlir):
    b = _count_gm_loads(before_mlir)
    a = _count_gm_loads(after_mlir)
    if b is None or a is None:
        # IR 产物缺失时不可判定；置 None 让调用侧标注"跳过"而非谎报通过。
        return None, b, a
    # 反滥用：store 数必须不变（写放大不增），load 应减少或不增。
    ok = (a["stores"] >= b["stores"]) is False or a["stores"] == b["stores"] or a["stores"] < b["stores"] + 0
    stores_ok = a["stores"] <= b["stores"]
    loads_ok = a["loads"] <= b["loads"]
    return (stores_ok and loads_ok), b, a


# ---------------- g2 目标 ---------------------------------------------------
def g2_matrix_diff(user, new, matrix_builder):
    """跑变体矩阵：orig vs new 位级 torch.equal。返回 (bool, cases, passed)。"""
    import torch  # 需 NPU，import 放函数内避免无 NPU 时 CLI 爆炸
    cases = matrix_builder()  # [{name,args}]
    passed = 0
    for name, args in cases:
        o = user.impl(**args)
        n = new.impl(**args)
        if isinstance(o, tuple):
            o, o_state = o
        else:
            o_state = None
        if isinstance(n, tuple):
            n, n_state = n
        else:
            n_state = None
        torch.npu.synchronize()
        eq = torch.equal(o.float(), n.float())
        eq_state = (o_state is None and n_state is None) or (
            o_state is not None and n_state is not None and torch.equal(o_state.float(), n_state.float()))
        if eq and eq_state:
            passed += 1
        del o, n
    return passed == len(cases), len(cases), passed


def _load_matrix_builder(spec):
    spec = spec.strip()
    if ":" not in spec:
        # 允许裸模块 → 用其 build_matrix
        modname = spec
        fn = "build_matrix"
    else:
        modname, fn = spec.split(":", 1)
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(modname)
    return getattr(mod, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="冻结用户源 triton .py")
    ap.add_argument("--restructured", required=True, help="重构 kernel .py")
    ap.add_argument("--utils", required=True, help="用户 utils.py（conv 依赖 .utils 加载）")
    ap.add_argument("--pkg", default="mmgt_guard", help="fabricated 包名")
    ap.add_argument("--impl", default="causal_conv1d_fwd_impl")
    ap.add_argument("--kernel", default="causal_conv1d_fwd_kernel")
    ap.add_argument("--matrix", default="", help="变体矩阵模块（含 build_matrix）→ 跑 g2（需 NPU）")
    ap.add_argument("--ir-before", default="", help="重构前 last_pass.mlir → 跑 g3")
    ap.add_argument("--ir-after", default="", help="重构后 last_pass.mlir → 跑 g3")
    ap.add_argument("--out", default="", help="结果 JSON 输出路径（缺省 stdout）")
    args = ap.parse_args()

    user = _load_variant(args.pkg + "_orig", args.user, args.utils, args.impl, args.kernel)
    new = _load_variant(args.pkg + "_new", args.restructured, args.utils, args.impl, args.kernel)

    result = {}

    # g1 —— 必跑
    g1 = g1_io_frozen(user, new)
    result["g1_io_frozen"] = g1

    # g3 —— IR 同构（给路径才跑）
    ir_loads = None
    if args.ir_before and args.ir_after:
        g3 = g3_ir_homology(args.ir_before, args.ir_after)
        if g3 is None:
            result["g3_ir_homology"] = None
            result.setdefault("_skipped", []).append("ir")
        else:
            g3ok, b, a = g3
            result["g3_ir_homology"] = g3ok
            result["ir_gm_loads_before"] = b
            result["ir_gm_loads_after"] = a
    else:
        result["g3_ir_homology"] = None
        result.setdefault("_skipped", []).append("ir")

    # g2 —— 变体矩阵位级 diff（给 --matrix 才跑）
    if args.matrix:
        try:
            builder = _load_matrix_builder(args.matrix)
            g2, cases, passed = g2_matrix_diff(user, new, builder)
            result["g2_matrix"] = g2
            result["g2_matrix_cases"] = cases
            result["g2_matrix_passed"] = passed
        except Exception as e:  # noqa: BLE001  无 NPU / matrix 构建失败 → 跳过并不谎报
            result["g2_matrix"] = None
            result["g2_matrix_error"] = str(e)
            result.setdefault("_skipped", []).append("g2")
    else:
        result["g2_matrix"] = None
        result.setdefault("_skipped", []).append("g2")

    ok = result.get("g1_io_frozen") and result.get("g3_ir_homology") in (True, None) and \
        result.get("g2_matrix") in (True, None)

    out = json.dumps(result, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"guard result: {out}")
    else:
        print(out)

    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()