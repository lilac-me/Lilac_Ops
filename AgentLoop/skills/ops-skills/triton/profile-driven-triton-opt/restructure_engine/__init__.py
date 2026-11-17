#!/usr/bin/env python3
"""restructure_engine 包：kernel-body 重构变换器（调用侧，§6.5）。

对外提供 `transform_registry`（去重/落 summary 用）与 `load_block`（参数化 load-block
重构计划）。本包 **不触碰 vendor**；重构 kernel 必须经 scripts/restructure_guard.py
4 闸验证后再作为 code_version 基线进入 S4 主循环。
"""