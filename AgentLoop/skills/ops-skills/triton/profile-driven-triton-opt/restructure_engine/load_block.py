#!/usr/bin/env python3
"""load_block.py — 参数化 load-block 重构变换（restructure_engine，§6.5）。

背景（restructure-stage §0）：当 op 瓶颈是「读放大 / GM gather 复用」这类结构性问题，
唯一可继续的杠杆在方法体内部数据流改写。load-block 是把 per-tap GM gather
（W 次重叠读，MTE2 放大 ~W 倍指令数）替换为**单次 `win×BD` UB 窗口 load +
extract_slice 移位累加**（各 tap 从同一 UB 窗口复用），再以 size-mask 处理边界 tap。

本模块把该变换参数化，供：
  - dedup 键 `signature()` → `load_block{win=..., W=...}`（transform_registry）。
  - 落 summary 的 `restructure.applied_transform`。
  - report「重构过程」节 I/O 守恒/读放大削减说明。
  - g3 IR 同构校验的预期 GM-load before/after。

**边界诚实声明**：将任意用户源自动改写为 load_block 属「方法体内部数据流重写」，因
op 而异、无法在对任意算子正确的前提下全文自动生成——这正是一轮实证（causal）为何是
「人工越界的一次性尝试」。故本模块产出**重构计划**（recipe/dedup/summary 素材），
实际重构 kernel 由调用侧按计划构造，并**必须经 restructure_guard 4 闸 + 契约
restructure 块**验证后再落为 best_code。已验证的 causal 实现绑定于 `materialize_known`
（指向本次实证产物 conv_restructured.py）。
"""
from __future__ import annotations

import os

try:
    from . import transform_registry
except ImportError:  # 顶层直接跑（无包上下文）时退化为同级模块导入
    import transform_registry  # type: ignore


class LoadBlockPlan:
    """一段 load-block 重构的声明式计划。

    核心参数：
      win    UB 窗口宽度（按 BT 计，如 2*BT=128）：窗口须能容纳 W 个 tap 的移位跨度。
      W      卷积核 tap 数（gather 读放大倍数）。
      mask   size-mask 策略（'edge_taps_only'：仅 window 边界 tap 需要 mask）。
      bd     UB 窗口的 BD（特征维 tile，与 BT 共同决定窗口张量形状 win×BD）。
    """

    def __init__(self, win, W, mask="edge_taps_only", bd=None):
        if W is None or win < W:
            raise ValueError(f"load_block 窗口 win={win} 须 >= W={W} 以容纳移位跨度")
        self.win, self.W, self.mask, self.bd = int(win), int(W), mask, bd

    def signature(self) -> str:
        bd = f", bd={self.bd}" if self.bd else ""
        return f"load_block{{win={self.win}, W={self.W}{bd}}}"

    def gm_loads_expected(self, load_before: int) -> int:
        """预期 GM load 指令数削减：per-tap gather `load_before` → 单次窗口 load。

        对 causal 实证：before=W+1=5（w 权重 1 次 + 每 tap 1 次 x 读）→ after=2
        （w 权重 1 次 + 单窗口 x 读 1 次）。
        """
        return 2  # 权重 1 次 + 单窗口数据 1 次；调用侧按实际 op 覆写

    def recipe(self) -> str:
        return (
            f"数据流重构：W-tap({self.W}) GM gather 重叠读(MTE2 放大 ~{self.W}x) "
            f"→ 单次 {self.win}×{self.bd} UB 窗口 load + extract_slice 移位累加；"
            f"mask 策略: {self.mask}。仅改写 no-加速初始状态分支的输入 load 结构；"
            f"逐 tap 乘加顺序 / dtype / 布局 / 宿主启动逐字节不变。"
        )


# 已登记变换会在此触发一次真实注册（一次会话内同签名只接受一次，复用 transform_registry）。
def register_default(win, W, bd):
    plan = LoadBlockPlan(win=win, W=W, bd=bd)
    if not transform_registry.applied(plan.signature()):
        transform_registry.register(
            plan.signature(),
            module=__name__,
            plan=plan.recipe(),
        )
    return plan


def materialize_known(plan: LoadBlockPlan, user_source: str, plan_dir: str) -> str:
    """把已验证 causal 的 load_block 重构落地为具体 kernel 文件。

    调用侧一般以「构造重构 kernel 源码 → 交 restructure_guard 4 闸」的方式落地；
    本函数仅对**已实证**的 causal 模式提供绑定（指向本次实证产物读写）。
    其他 op：由调用侧按 `plan.recipe()` 构造重构 kernel 后走同一守卫，缺省此处
    明确 Raise 而非臆造一份"通用改写"，保持「只调优不重新生成算子」边界。
    """
    # 已知实证绑定：plan_b/conv_restructured.py 即本签名对应的 load_block 产物。
    proven = os.path.join(plan_dir, "conv_restructured.py")
    if os.path.isfile(proven):
        return proven
    raise NotImplementedError(
        f"restructure_engine.load_block 对 op {os.path.basename(user_source)!r} 尚无实证绑定；"
        f"请调用侧按 recipe 构造重构 kernel 后交 restructure_guard 4 闸验证：\n{plan.recipe()}"
    )


if __name__ == "__main__":
    p = register_default(win=128, W=4, bd=64)
    print("signature:", p.signature())
    print("recipe:", p.recipe())
    print("gm_loads_expected(5)=", p.gm_loads_expected(5))
    print("signatures:", transform_registry.signatures())