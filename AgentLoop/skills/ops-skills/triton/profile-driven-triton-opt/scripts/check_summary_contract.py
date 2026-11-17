#!/usr/bin/env python3
"""check_summary_contract.py — S4 出口产品契约强制校验。

防止两类顶包：
  A. 防止把 torch 参考当性能分母（`torch参考 / triton基线` 顶包 `speedup_vs_user`）。
  B. 防止用手写计时器（torch.npu Event / time.perf_counter 等）顶包官方 benchmark 数据
     —— 性能数字必须可溯源回 `scripts/benchmark.py`（torch_npu.profiler）的产物。

硬约束：
  1) `perf_baseline` 必须显式声明，且（给定 --triton-source 时）与其一致 → 分母唯一=用户源。
  2) 分母角色与真值角色分离：`perf_baseline`(用户 triton) ≠ `gate_ref`(torch 参考)。
  3) `REACHED` 必须 `optimizer_rounds > 0` 且有非空 `best_code`。
  4) **测量溯源闸（新增）**：`bench.tool` 必须是官方计时器标识；`bench.baseline_result_json` /
     `bench.optimized_result_json` 必须存在，且 summary 中的基线/优化 lat 与官方产物
     `result["implementation"]["avg_latency_ms"]` 一致。绕开官方脚本 = 契约 FAIL。
     显式 `bench.waiver = {reason}` 可将溯源放宽为「公示的偏离」：gate 打印 waiver 并标记
     PASS_WITH_WAIVER（deviation 被可见化，而非静默）。
  5) **调优执行者闸（新增）**：`optimizer_rounds>0` 时必须证明调优经 `triton-latency-optimizer`
     执行（`optimizer.tool` 白名单 + `points_hit`/`loaded_docs` 证物）——禁止自写 tiling
     扫描顶替调优循环；`optimizer.waiver = {reason}` 可将偏离公示放行（PASS_WITH_WAIVER）。
  6) **scan-trace 穷尽闸（新增，count-agnostic）**：`optimizer_rounds>0` 时必须带
     `points_considered`(N) + `scan_trace`（逐轮 `scan_from/hit/not_hit` 区间）。逐条校验区间
     是否等于编排器推导值（`hit=P → [cursor,P-1]` / `hit=None → [cursor,N]`）、`scan_from` 连续性
     （∈ {1, 上一命中点, 上一命中点+1}）、`scan_complete` 语义（须有 `scan_from=1 & hit=null`
     且覆盖 `[1,N]` 的整趟无命中轮）。**N 零常量**：点数未来增删只改 summary，脚本不写死。
  7) **patch 闸（新增，§10.5）**：`best_code` 非空（采纳了更优变体）时，`patch` 块必须存在且合法：
     `mode ∈ {launch_default, kernel_replace}`，`patch.path` 是真实文件，`apply_target == perf_baseline`
     （落点=用户源分母唯一），`resverified.passed==total>0`（patched 副本已重验）；`launch_default`
     需 `best_config_applied`。`patch.waiver={reason}` 可公示放行（如 kernel_replace 降级 advisory）。
  8) **指标口径闸（新增）**：`metric_mode` 缺省按 `speedup` 兼容旧契约；`mfu` 模式必须带数值
     `target_mfu` 与 `best_mfu`，`REACHED` 须 `best_mfu >= target_mfu`。
  KB 检索证物（`optimizer.kb_preflight`）缺失仅告警，不阻断。

用法：
  check_summary_contract.py <summary.json> [--triton-source <用户triton源路径>]
退出码：0 = 通过；2 = 契约违反（stderr 打印原因）。
"""

import argparse
import json
import os
import re
import sys

# 官方计时器唯一合法标识：scripts/benchmark.py（内部用 torch_npu.profiler 采集）。
CANONICAL_TOOLS = {"scripts/benchmark.py", "scripts/benchmark.py:torch_npu.profiler"}

# 调优循环唯一合法执行器标识：triton-latency-optimizer（按 references/Index.md 33 点有序扫描）。
# 禁止以自写 tiling 脚本顶替调优循环。
OPTIMIZER_TOOLS = {
    "triton-latency-optimizer",
    "vendored-triton-op/triton-latency-optimizer",
    "vendored-triton-op/triton-latency-optimizer:Index.md",
}


def normalize(p):
    return os.path.normpath(p) if p else ""


def _sum_dir(summary_path):
    return os.path.dirname(os.path.abspath(summary_path))


def _resolve_rel(base, p):
    """结果 JSON 路径可为绝对或相对路径（相对 summary 所在目录）。"""
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(base, p)


def _check_provenance(s, summary_path, errs, warnings):
    """测量溯源闸：强制性能数字来自官方 benchmark 产物，否则契约 FAIL。"""
    bench = s.get("bench") or {}
    tool = bench.get("tool")
    waiver = bench.get("waiver")

    if waiver:
        warnings.append(
            f"[WAIVER] bench.tool='{tool}' 且 waiver={json.dumps(waiver, ensure_ascii=False)} —— "
            "测量未走官方 benchmark 脚本，作为显式公示偏离记入报告，非静默放行。"
        )
        return  # waiver：放宽溯源，但 deviation 已在报告/报告侧公示

    if not tool:
        errs.append(
            "缺少 bench.tool：无法证明性能数字来自官方脚本 scripts/benchmark.py(torch_npu.profiler)。"
            "禁止用手写计时器（torch.npu Event / time.perf_counter）顶包官方数据。"
        )
        return

    if tool not in CANONICAL_TOOLS:
        errs.append(
            f"bench.tool='{tool}' 不在官方计时器白名单 {sorted(CANONICAL_TOOLS)} 中："
            "性能数字必须来自官方 benchmark 脚本，不得自写计时器。"
        )
        return

    base = _sum_dir(summary_path)
    baseline_json = _resolve_rel(base, bench.get("baseline_result_json"))
    optimized_json = _resolve_rel(base, bench.get("optimized_result_json"))

    def _read_impl_lat(path, label):
        if not path or not os.path.isfile(path):
            errs.append(f"bench.{label}_result_json 不存在：{path}。性能数字必须溯源官方产物。")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            errs.append(f"bench.{label}_result_json 读取/解析失败：{path} ({e})")
            return None
        impl = (data or {}).get("implementation") or {}
        lat = impl.get("avg_latency_ms")
        if lat is None:
            errs.append(f"bench.{label}_result_json 缺 implementation.avg_latency_ms({path})："
                        "非官方 benchmark 产物。")
        return lat

    base_lat = _read_impl_lat(baseline_json, "baseline")
    opt_lat = _read_impl_lat(optimized_json, "optimized")

    # summary 声称的 lat（中位数口径）须与官方产物一致（容差 1e-2 ms 防浮点）。
    claimed_base = bench.get("baseline_avg_ms")
    claimed_opt = bench.get("optimized_avg_ms")
    if base_lat is not None and claimed_base is not None:
        if abs(float(claimed_base) - float(base_lat)) > 1e-2:
            errs.append(
                f"bench.baseline_avg_ms({claimed_base}) 与官方产物 "
                f"implementation.avg_latency_ms({base_lat}) 不一致：测量数字必须复刻官方 benchmark 输出。"
            )
    if opt_lat is not None and claimed_opt is not None:
        if abs(float(claimed_opt) - float(opt_lat)) > 1e-2:
            errs.append(
                f"bench.optimized_avg_ms({claimed_opt}) 与官方产物 "
                f"implementation.avg_latency_ms({opt_lat}) 不一致：测量数字必须复刻官方 benchmark 输出。"
            )


def _resolve_rounds(s) -> int:
    """兼容 optimizer_rounds 单独字段，或嵌套在 bench 下。"""
    if "optimizer_rounds" in s:
        return int(s.get("optimizer_rounds") or 0)
    return int((s.get("bench") or {}).get("optimizer_rounds", 0))


_METRIC_MODES = {"speedup", "mfu"}


def _check_metric_mode(s, status, errs, warnings):
    """指标口径闸（metric_mode 双轨）。

    - speedup（缺省）：沿用既有契约，不新增字段要求（向后兼容旧 summary）。
    - mfu：必须显式 metric_mode/target_mfu 并带 best_mfu；REACHED 须
      `best_mfu >= target_mfu`（与 speedup 模式的 target_speedup 判定对称）。
    """
    mode = s.get("metric_mode", "speedup")
    if mode not in _METRIC_MODES:
        errs.append(f"metric_mode='{mode}' 非法：仅允许 {sorted(_METRIC_MODES)}。")
        return
    if mode != "mfu":
        return

    target = s.get("target_mfu")
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        errs.append("metric_mode=mfu 但缺数值 target_mfu：MFU 判定必须显式给目标（默认 60，不得省略）。")
        return

    if s.get("baseline_mfu") is None:
        warnings.append(
            "metric_mode=mfu 但缺 baseline_mfu：无法量化 MFU 提升幅度"
            "（建议补 triton-mfu-analyzer 的基线测定）。"
        )

    best = s.get("best_mfu")
    if best is None:
        if status == "REACHED":
            errs.append("metric_mode=mfu 且 status=REACHED 但缺 best_mfu：无 MFU 实测不得判定达标。")
        else:
            warnings.append(
                "metric_mode=mfu 但缺 best_mfu：建议补 triton-mfu-analyzer 的 mfu_decompose 结果。"
            )
        return

    if status == "REACHED" and float(best) < float(target):
        errs.append(
            f"status=REACHED 但 best_mfu({best}) < target_mfu({target})："
            "mfu 模式达标判据未满足。"
        )


def _check_optimizer_evidence(s, errs, warnings, index_path):
    """调优执行者闸（新增）：凡声称跑过调优轮，必须证明经由 triton-latency-optimizer。

    防止「自写 tiling 扫描顶替调优循环」：optimizer_rounds>0 却缺 optimizer.tool /
    points_hit / loaded_docs → 契约 FAIL。显式 optimizer.waiver = {reason} 可放宽为
    公示的偏离（PASS_WITH_WAIVER），不静默放行。
    """
    rounds = _resolve_rounds(s)
    if rounds <= 0:
        return  # 未跑调优轮无需证物；REACHED 前置另行校验
    opt = s.get("optimizer") or {}
    tool = opt.get("tool")
    waiver = opt.get("waiver")

    if waiver:
        warnings.append(
            f"[WAIVER] optimizer.tool='{tool}' 且 waiver={json.dumps(waiver, ensure_ascii=False)} —— "
            "调优未走 triton-latency-optimizer Index 扫描通道，作为显式公示偏离记入报告，非静默放行。"
        )
        return

    if not tool:
        errs.append(
            "optimizer_rounds>0 但缺 optimizer.tool：无法证明调优经 triton-latency-optimizer 执行。"
            "禁止以自写 tiling 脚本顶替调优循环（须按 references/Index.md 点数 N 有序扫描，N 当场读取）。"
        )
        return

    if tool not in OPTIMIZER_TOOLS:
        errs.append(
            f"optimizer.tool='{tool}' 不在调优器白名单 {sorted(OPTIMIZER_TOOLS)} 中："
            "调优必须走 triton-latency-optimizer 的唯一执行通道。"
        )
        return

    if "points_hit" not in opt or "loaded_docs" not in opt:
        errs.append(
            f"optimizer.tool='{tool}' 合法但缺 optimizer.points_hit / optimizer.loaded_docs："
            "未记录命中的优化点编号与真实加载的参考文档，违反『命中即加载参考文档』纪律，"
            "无法证明经过 Index 有序扫描。"
        )
        return

    # presence 通过后追加 count-agnostic 穷尽闸（scan-trace）
    _check_scan_trace(s, errs, warnings)
    # 追加强制点闸：每轮必须执行的优化点（如 31 IR 分析）不得被区间折叠跳过
    _check_mandatory_points(s, errs, warnings, index_path)

    # KB 检索证物（非阻断）：latency-optimizer 现为「每次调用前强制 preflight」的单步函数，
    # 摘要缺失即偏离其纪律；公示告警但不阻断（兼容旧 summary / KB 不可用场景）。
    if not opt.get("kb_preflight"):
        warnings.append(
            "optimizer_rounds>0 但缺 optimizer.kb_preflight："
            "latency-optimizer 已改为『每次调用前强制 KB preflight + 盒装摘要』，"
            "建议记录检索摘要（called/cards/points/available）；缺省仅告警。"
        )


# 【强制点】Index.md 机器可读标记：每轮必须执行的优化点编号集合。禁止 self-declare，
# 一律从 triton-latency-optimizer 的 references/Index.md 读取（单一事实源）。
_MANDATORY_RE = re.compile(r"MANDATORY_PER_ROUND\s*:\s*([0-9,\s]+)")
# 标记缺失时的兜底：直接从 Index.md 表格里「每轮作为最后一个优化点必须执行」那一行取点号。
# 上游 Index.md 不带机器可读标记，而 vendor 内容必须字节保真（禁止为校验而改写上游），
# 因此校验侧按表格语义派生，点数与编号仍是 count-agnostic。
_MANDATORY_TABLE_RE = re.compile(
    r"^\|\s*(\d+)\s*\|[^|\n]*\|[^|\n]*\|\s*每轮作为最后一个优化点必须执行", re.MULTILINE)
_INTERVAL_RE = re.compile(r"\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*")


def _parse_interval(s):
    m = _INTERVAL_RE.fullmatch(s or "")
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _check_scan_trace(s, errs, warnings):
    """（count-agnostic）scan-trace 穷尽闸：把「我合并判断了 3-31」式手写声明，变成可数值自洽的
    逐轮扫描区间。N 不写死，取 summary 自报的 points_considered。
    """
    rounds = _resolve_rounds(s)
    if rounds <= 0:
        return
    opt = s.get("optimizer") or {}
    if opt.get("waiver"):
        return  # presence 闸已公示 waiver 放行；穷尽闸不二次拦

    N = opt.get("points_considered")
    trace = opt.get("scan_trace")

    if not isinstance(N, int) or N <= 0:
        errs.append(
            "optimizer_rounds>0 但 optimizer.points_considered 缺失/非法："
            "无法做 scan-trace 穷尽闸。N = 本次从 Index.md 表格当场读取的最大点号（count-agnostic），"
            "不得硬编码常点。"
        )
        return
    if not isinstance(trace, list) or not trace:
        errs.append(
            "optimizer_rounds>0 但 optimizer.scan_trace 缺失/为空：禁止用 points_hit=[...] 手写式声明"
            "顶替逐轮扫描区间。scan_trace 必须逐轮记录 scan_from / hit / not_hit 区间。"
        )
        return

    claimed_complete = bool(opt.get("scan_complete"))
    found_full_pass = False
    prev = None  # (scan_from, hit)

    for i, rec in enumerate(trace):
        if not isinstance(rec, dict):
            errs.append(f"scan_trace[{i}] 非对象，无法解析。")
            continue
        sf = rec.get("scan_from")
        hit = rec.get("hit")
        nothit_str = rec.get("not_hit")

        if not isinstance(sf, int) or sf <= 0:
            errs.append(f"scan_trace[{i}] scan_from={sf!r} 非法（须为正整数）。")
            continue

        iv = _parse_interval(nothit_str)
        if iv is None:
            errs.append(f"scan_trace[{i}] not_hit={nothit_str!r} 非 [a,b] 格式。")
            continue
        lo, hi = iv

        # 推导 expect 区间（count-agnostic：上界一律取 N，命中即在 scan_from 处）
        exp = None
        if hit is None:
            exp = (sf, N)
            if sf == 1 and lo == 1 and hi == N:
                found_full_pass = True
        elif isinstance(hit, int):
            exp = (sf, hit - 1)
        else:
            errs.append(f"scan_trace[{i}] hit={hit!r} 非法（须为整数点号或 null）。")
        if exp is None:
            continue
        if (lo, hi) != exp:
            errs.append(
                f"scan_trace[{i}] 区间不自洽：scan_from={sf}, hit={hit} → 推导 not_hit={list(exp)}，"
                f"但记录 not_hit={nothit_str}。扫描区间必须由编排器按 cursor+hit 推导，不得手写。"
            )
        if isinstance(hit, int) and hit < sf:
            errs.append(f"scan_trace[{i}] hit={hit} < scan_from={sf}：命中点必须在扫描起点及之后。")

        # scan_from 连续性（跳过首条；只约束非终结轮）
        if prev is not None:
            _p_sf, p_hit = prev
            allowed = {1}
            if isinstance(p_hit, int):
                allowed.add(p_hit)
                allowed.add(p_hit + 1)
            if sf not in allowed:
                errs.append(
                    f"scan_trace 扫描起点不连续：round{i-1} hit={p_hit!r} → round{i} scan_from={sf} 非法"
                    f"（允许 {sorted(allowed)} = 1 / 上一命中点 / 上一命中点+1）。"
                )
        prev = (sf, hit)

    if claimed_complete and not found_full_pass:
        errs.append(
            "optimizer.scan_complete=true 但 scan_trace 中无「scan_from=1、hit=null、not_hit=[1,N]」"
            "的整趟无命中轮：scan_complete 必须由编排器从 cursor=1 走完全趟仍无命中推导，不得自报。"
        )
    if not claimed_complete and found_full_pass:
        warnings.append(
            "scan_trace 已含整趟无命中轮但 optimizer.scan_complete=false：语义不一致，建议补 true。"
        )


def _read_mandatory_points(index_path, errs):
    """从 triton-latency-optimizer references/Index.md 派生每轮必须执行的优化点集。

    优先读机器可读标记 `MANDATORY_PER_ROUND`；上游未提供标记时按表格行
    「每轮作为最后一个优化点必须执行」派生（count-agnostic，且无需改写 vendor 内容）。
    单一事实源=拉取的 vendor skill；禁止 self-declare / 硬编码。返回 set[int]，读取失败返回 None。
    """
    if not index_path:
        errs.append(
            "optimizer_rounds>0 但缺 --index：每轮必须执行的优化点集必须从 "
            "triton-latency-optimizer skill 的 references/Index.md 读取，禁止 self-declare/硬编码。"
        )
        return None
    if not os.path.isfile(index_path):
        errs.append(f"--index 指向的 Index.md 不存在：{index_path}")
        return None
    try:
        text = open(index_path, "r", encoding="utf-8").read()
    except OSError as e:
        errs.append(f"--index 读取失败：{index_path} ({e})")
        return None
    m = _MANDATORY_RE.search(text)
    if not m:
        # 兜底：按表格行派生（上游 Index.md 无标记；不改写 vendor 内容）
        rows = _MANDATORY_TABLE_RE.findall(text)
        pts = {int(x) for x in rows}
        if not pts:
            errs.append(
                f"Index.md 既无 MANDATORY_PER_ROUND 标记、也无「每轮作为最后一个优化点必须执行」"
                f"表格行（{index_path}）：无法派生每轮必须执行的优化点集，禁止 self-declare。"
            )
            return None
        return pts
    pts = {int(x) for x in re.split(r"[,\s]+", m.group(1).strip()) if x.strip().isdigit()}
    if not pts:
        errs.append("Index.md MANDATORY_PER_ROUND 标记解析为空：强制点集不能为空。")
        return None
    return pts


def _check_mandatory_points(s, errs, warnings, index_path):
    """（新增）强制点闸：每轮必须执行的优化点（Index.md MANDATORY_PER_ROUND，如 31 IR 分析）
    不得被 scan_trace 区间折叠静默跳过。

    编排器在每轮收尾须真实执行全部强制点并记入该轮 record.mandatory。此处要求：任何
    `hit` 非 null（应用了改动）的轮次，其 record.mandatory 必须覆盖强制集 M；缺任一轮即 FAIL。
    仅有 `optimizer.waiver` 可公示放行（PASS_WITH_WAIVER）。
    """
    if _resolve_rounds(s) <= 0:
        return
    opt = s.get("optimizer") or {}
    if opt.get("waiver"):
        return  # presence 闸已公示 waiver 放行；强制点闸不二次拦
    M = _read_mandatory_points(index_path, errs)
    if M is None:
        return  # 缺 --index / 标记：errs 已记录 FAIL 理由
    if not M:
        return
    trace = opt.get("scan_trace") or []
    if not isinstance(trace, list) or not trace:
        errs.append(
            f"optimizer_rounds>0 但 optimizer.scan_trace 为空，无法校验强制点 {sorted(M)} 是否每轮执行。"
        )
        return
    sorted_m = sorted(M)
    for i, rec in enumerate(trace):
        if not isinstance(rec, dict) or rec.get("hit") is None:
            continue  # 整趟复查轮（hit=null）豁免；rule 仍建议执行并记录
        mandatory = rec.get("mandatory") or []
        if not isinstance(mandatory, list) or not all(isinstance(x, int) for x in mandatory):
            errs.append(f"scan_trace[{i}] record.mandatory 非法（须为点号 int 列表）。")
            continue
        miss = sorted(M - set(mandatory))
        if miss:
            errs.append(
                f"scan_trace[{i}] 应用轮(hit={rec.get('hit')})缺强制点执行 {miss}："
                f"每轮收尾必须执行全部强制点 {sorted_m}（来源 Index.md MANDATORY_PER_ROUND，禁止 self-declare）"
                "并记入 record.mandatory；缺则补充执行或加 optimizer.waiver 公示。"
            )


def _check_patch(s, summary_path, errs, warnings):
    """（§10.5）patch 闸：采纳更优 best_code 时，必须交付一份字节派生、经 verify 重验的补丁。

    防止「有优化却不给可落地方案」：best_code 非空却没有可应用的 patch / 落点不是用户源 /
    patched 副本未重验 → 契约 FAIL。显式 patch.waiver={reason} 可公示放行。
    """
    if not s.get("best_code"):
        return  # 无采纳优化，不要求 patch

    perf_baseline = s.get("perf_baseline") or ""
    patch = s.get("patch") or {}

    if patch.get("waiver"):
        warnings.append(
            f"[WAIVER] patch.waiver={json.dumps(patch['waiver'], ensure_ascii=False)} —— "
            "采纳 best_code 但未交付可应用 patch，作为显式公示偏离记入报告，非静默放行。"
        )
        return

    mode = patch.get("mode")
    if mode not in ("launch_default", "kernel_replace"):
        errs.append(
            f"best_code 非空但 patch.mode={mode!r} 非法（须为 launch_default / kernel_replace）："
            "采纳更优变体后必须产出可应用的补丁。"
        )

    opname = s.get("op_name", "op")
    ppath = _resolve_rel(_sum_dir(summary_path), patch.get("path"))
    if not ppath or not os.path.isfile(ppath):
        errs.append(
            f"best_code 非空但 patch.path 缺失或文件不存在（{ppath}）：必须产出 patch/{opname}.patch。"
        )

    diff_from = _resolve_rel(_sum_dir(summary_path), patch.get("diff_from"))
    if diff_from and not os.path.isfile(diff_from):
        errs.append(
            f"patch.diff_from 指向的文件不存在：{diff_from}（应为冻结用户源副本 official/{opname}.py）。"
        )

    at = normalize(str(patch.get("apply_target")))
    pb = normalize(str(perf_baseline))
    if at and pb and at != pb:
        errs.append(
            f"patch.apply_target({at}) != perf_baseline({pb})：patch 必须落在用户源分母唯一上。"
        )

    rv = patch.get("resverified") or {}
    passed, total = rv.get("passed"), rv.get("total")
    if not (isinstance(passed, int) and isinstance(total, int) and total > 0 and passed == total):
        errs.append(
            f"patch.resverified 缺失或未全过（passed={passed!r} / total={total!r}）："
            "patched 副本必须经 verify.py passed==total 重验后才交付。"
        )

    if mode == "launch_default" and not patch.get("best_config_applied"):
        errs.append("patch.mode='launch_default' 但缺 patch.best_config_applied：需记录覆盖的启动默认值。")

    # —— 责任声明写入生成报告（§10.6，强制）——
    if mode in ("launch_default", "kernel_replace"):
        _check_report_statement(summary_path, errs, warnings)


REPORT_STMT = ("算子级归因", "替换回图")  # 《Patch 应用与责任声明》必需片段

# —— 用户 report.md 必含记录片段(§8,与 summary 字段对齐)——
# 存在 restructure 块时,report.md「重构过程」节必含(§8.2/§8.3)。
RESTRUCTURE_REPORT_FRAG = ("重构过程", "g1", "g2", "g3", "speedup_vs_user")
# optimizer_rounds>0 时,report.md「调优过程」节必含(§8.3)。
OPTIMIZATION_REPORT_FRAG = ("调优过程", "triton-latency-optimizer", "scan_complete")


def _check_report_statement(summary_path, errs, warnings):
    """责任声明必须写进 per-op report.md（FAIL）与 batch_report.md 尾部（WARN）。

    per-op 为强制：patch 要求时 report.md 缺声明即视为未交付。
    batch_report 为批次级、S5 后落地，存在但缺声明 → 告警（避免逐 op 校验与批次写入的顺序耦合）。
    """
    op_dir = _sum_dir(summary_path)
    report = os.path.join(op_dir, "report.md")
    if os.path.isfile(report):
        try:
            text = open(report, "r", encoding="utf-8").read()
        except OSError:
            text = ""
        miss = [t for t in REPORT_STMT if t not in text]
        if miss:
            errs.append(
                f"采纳 best_code 但 per-op report.md 缺责任声明片段 {miss}："
                "《Patch 应用与责任声明》必须写入生成报告（§10.6），否则视为未交付。"
            )
    else:
        errs.append(
            f"采纳 best_code 但 per-op report.md 不存在（{report}）："
            "报告须含《Patch 应用与责任声明》小节（§10.6）。"
        )

    # batch_report 尾部（沿用既有文案，仅告警）
    run_root = os.path.dirname(os.path.abspath(op_dir))  # op_<name>/ 的父 = RUN_DIR
    breport = os.path.join(run_root, "batch_report.md")
    if os.path.isfile(breport):
        try:
            btext = open(breport, "r", encoding="utf-8").read()
        except OSError:
            btext = ""
        bmiss = [t for t in REPORT_STMT if t not in btext]
        if bmiss:
            warnings.append(
                f"batch_report.md 缺责任声明片段 {bmiss}：报告尾部须明示替换回图=用户责任（§10.6）。"
            )


def _check_restructure(s, errs, warnings, summary_path):
    """（§6.3/§7）可选 restructure 块校验；缺省→直接 return，纯 tune 契约逐字节不变。

    块存在时校验：
      - trigger / applied_transform 非空；engine 为真实文件。
      - gates.{g1_io_frozen,g2_matrix,g3_ir_homology} 全 true（g4=官方 verify/benchmark，
        复用既有 bench 溯源闸，不在此另设）。
    反滥用硬断言：restructure 存在但 `optimizer_rounds<=0` → FAIL（重构后必须回到 vendor
    扫描跑过 tune 轮，阻止「重构+直接判 REACHED」）。restructure 只作为新 code_version 的基线。
    """
    rs = s.get("restructure")
    if not rs:
        return  # 纯 tune（无重构）契约路径不变

    trigger = rs.get("trigger")
    applied = rs.get("applied_transform")
    engine = rs.get("engine")
    gates = rs.get("gates") or {}

    if not trigger:
        errs.append(
            "restructure.trigger 为空：须记录触发来源（ir_point31_gather_reuse / "
            "structural_gather_detector）。"
        )
    if not applied:
        errs.append(
            "restructure.applied_transform 为空：须记录去重签名（如 load_block{win=2*BT,W=4}）。"
        )

    eng = _resolve_rel(_sum_dir(summary_path), engine)
    if not eng or not os.path.isfile(eng):
        errs.append(
            f"restructure.engine 非真实文件：{engine}。须指向 restructure_engine/<transform>.py。"
        )

    g1 = gates.get("g1_io_frozen") is True
    g2 = gates.get("g2_matrix") is True
    g3 = gates.get("g3_ir_homology") is True
    if not (g1 and g2 and g3):
        errs.append(
            f"restructure 4 闸未全过：g1_io_frozen={g1}, g2_matrix={g2}, "
            f"g3_ir_homology={g3}。任一闸 False → 该变换作废，不得作为 best_code。"
            "（g4=官方 verify/benchmark，复用 bench 溯源闸，不在此另设）"
        )

    rounds = _resolve_rounds(s)
    if rounds <= 0:
        errs.append(
            "restructure 块存在但 optimizer_rounds<=0：重构后必须回到 vendor 扫描跑过 tune 轮，"
            "禁止「重构+直接判 REACHED」。restructure 只作为新 code_version 基线进入 S4 主循环。"
        )


def _read_report_text(summary_path):
    report = os.path.join(_sum_dir(summary_path), "report.md")
    if not os.path.isfile(report):
        return None, report
    try:
        return open(report, "r", encoding="utf-8").read(), report
    except OSError:
        return "", report


def _check_report_records(s, errs, warnings, summary_path):
    """（§8.3）report 落点闸：重构与调优记录都必须写进用户 report.md。

    report.md 是用户最终阅读的落点；契约只校验 summary 无法让用户看到过程。缺 report
    记录 = 视为未交付（与 §10.6 责任声明同级别强制）。
      - 存在 restructure 块 → report.md 必含「重构过程」节及必含字段。
      - optimizer_rounds>0     → report.md 必含「调优过程」记录。
    """
    rounds = _resolve_rounds(s)
    has_rs = bool(s.get("restructure"))
    if not has_rs and rounds <= 0:
        return

    text, report = _read_report_text(summary_path)
    if text is None:
        if has_rs:
            errs.append(f"存在 restructure 块但 per-op report.md 不存在（{report}）："
                        "「重构过程」记录必须写入用户 report.md（§8）。")
        if rounds > 0:
            errs.append(f"optimizer_rounds={rounds}>0 但 per-op report.md 不存在（{report}）："
                        "「调优过程」记录必须写入用户 report.md（§8.3）。")
        return

    if has_rs:
        miss = [t for t in RESTRUCTURE_REPORT_FRAG if t not in text]
        if miss:
            errs.append(f"存在 restructure 块但 report.md 缺「重构过程」记录片段 {miss}："
                        "用户报告须含「重构过程」节及必含字段（§8.2/§8.3）。")
    if rounds > 0:
        miss = [t for t in OPTIMIZATION_REPORT_FRAG if t not in text]
        if miss:
            errs.append(f"optimizer_rounds={rounds}>0 但 report.md 缺「调优过程」记录片段 {miss}："
                        "用户报告须含「调优过程」节（§8.3）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--triton-source", default="", help="用户提供的 triton 源文件路径（唯一分母）")
    ap.add_argument("--index", default="", help="triton-latency-optimizer 的 references/Index.md 路径，"
                    "从中派生每轮必须执行的优化点集（MANDATORY_PER_ROUND 标记，缺失时按表格行派生），"
                    "单一事实源、禁止 self-declare")
    args = ap.parse_args()

    with open(args.summary, "r", encoding="utf-8") as f:
        s = json.load(f)

    errs = []
    warnings = []

    # (1) perf_baseline 必须显式声明
    perf_baseline = s.get("perf_baseline")
    if not perf_baseline:
        errs.append("缺少 perf_baseline：性能分母未显式指向用户源。禁止用 gate_ref(torch 参考) 当分母。")

    # (2) 分母 = 用户源 triton（若提供了）
    if args.triton_source:
        expected = normalize(args.triton_source)
        actual = normalize(str(perf_baseline))
        if not actual:
            errs.append("perf_baseline 为空但 --triton-source 已指定。")
        elif not (actual == expected or actual.endswith(expected) or expected.endswith(actual)):
            errs.append(
                f"perf_baseline({actual}) 与 --triton-source 用户源({expected}) 不一致："
                "性能分母必须唯一=用户源，不得换成其他实现。"
            )

    # (3) 测量溯源闸（所有 status 均强制，不限于 REACHED）
    _check_provenance(s, args.summary, errs, warnings)

    # (3b) 调优执行者闸（所有 status 均强制：凡 optimizer_rounds>0 须证明经 triton-latency-optimizer）
    _check_optimizer_evidence(s, errs, warnings, args.index)

    # (3c) patch 闸（所有 status 均强制：best_code 非空须有可应用补丁）
    _check_patch(s, args.summary, errs, warnings)

    # (3d) restructure 块（可选/正交：纯 tune 缺省即跳过，契约结果逐字节不变）
    _check_restructure(s, errs, warnings, args.summary)

    # (3e) report 落点闸（§8.3：重构/调优记录必须写进用户 report.md，否则视为未交付）
    _check_report_records(s, errs, warnings, args.summary)

    # (3f) 指标口径闸（metric_mode 双轨：speedup 兼容旧契约，mfu 校验 target_mfu/best_mfu）
    _check_metric_mode(s, s.get("status"), errs, warnings)

    # (4) REACHED 前置
    status = s.get("status")
    if status == "REACHED":
        rounds = _resolve_rounds(s)
        best_code = s.get("best_code")
        if rounds <= 0:
            errs.append("REACHED 但 optimizer_rounds<=0：未跑调优轮不得判定达标。")
        if not best_code:
            errs.append("REACHED 但缺 best_code：无优化后产物不得判定达标。")

    for w in warnings:
        print(f"[WARN] {w}", file=sys.stderr)

    if errs:
        print("CONTRACT FAIL:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 2

    if warnings:
        print("CONTRACT PASS (WITH WAIVER)", file=sys.stderr)
        return 0

    print("CONTRACT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())