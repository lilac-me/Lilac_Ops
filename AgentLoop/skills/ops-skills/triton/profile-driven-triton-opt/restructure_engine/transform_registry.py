#!/usr/bin/env python3
"""transform_registry.py — restructure 变换签名 → 变换模块注册表（调用侧，§6.5）。

供两处复用：
  1) dedup：同一 `applied_transform` 签名只接受一次真实提升（restructure-stage §5.2），
     杜绝重复榨同一重构手段。
  2) 落 summary：`restructure.applied_transform` == 变换自身的 `signature()`，
     供契约 `_check_restructure` 与用户 report「重构过程」节溯源「改了什么」。

签名格式（建议）：`<transform>{k=v, ...}`，如 `load_block{win=128, W=4}`。
"""
from __future__ import annotations

from typing import Any, Optional

_REGISTRY: dict[str, Any] = {}


def register(signature: str, module: Any, plan: str) -> None:
    """注册一个变换：signature 为其去重键；module 是产重构 kernel 的入口模块；plan 是简述。"""
    if not signature or signature in _REGISTRY:
        raise ValueError(
            f"restructure 变换签名 {signature!r} 为空或已注册（重复注册）。"
            "同一签名只接受一次；如需新手段请用不同 signature。"
        )
    _REGISTRY[signature] = {"module": module, "plan": plan}


def lookup(signature: str) -> Optional[Any]:
    """按签名查变换模块；未注册返回 None。"""
    return _REGISTRY.get(signature)


def applied(signature: str) -> bool:
    """该变换签名是否已是本次会话中登记过的已应用变换（dedup 依据）。"""
    return signature in _REGISTRY


def signatures() -> list[str]:
    """全部已登记变换签名（升序）。"""
    return sorted(_REGISTRY)


def plan_of(signature: str) -> Optional[str]:
    entry = _REGISTRY.get(signature)
    return entry["plan"] if entry else None