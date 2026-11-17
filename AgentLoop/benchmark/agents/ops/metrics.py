"""Metric extraction for ops-agent benchmark runs.

Reads a completed run workspace (``outputs/<op>/``) and extracts the raw,
deterministic evidence that ``scoring.py`` turns into the five-dimension score.

Every extractor is defensive: it returns ``None`` (or an empty collection) on a
missing / unparseable file instead of raising, so a partial run still scores
(and is reported as partial) rather than crashing the whole batch.

Evidence sources (all optional, best-effort):

* ``run_meta.json``    — harness/model/token accounting written by the runner's
                         harness adapter (Claude Code / Codex / DSH). This is the
                         canonical source for cost metrics.
* ``state_transitions.jsonl`` — append-only state log produced by
                         ``plugins/ops/engine/state_machine.py``; the canonical
                         source for iteration counts and wall-clock span.
* ``test_result.md``   — L1-L5 pass/fail counts and bit-exact markers.
* ``speedup.md``       — speedup matrix (Triton vs torch reference).
* ``bottleneck.md``    — bottleneck classification + optimisation suggestions.
* ``opt_history.md``   — per-round optimisation evidence + convergence reason.
* ``design.md``        — design quality (section count) + declared targets.
* ``kernel.py``        — anti-cheat check (must be a Triton-native kernel).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .console import force_utf8_stdout

# Anti-cheat: a **call** into a torch functional path, however it is spelled.
# Hoisted to module scope because it is a constant pattern, not per-run state.
_CALL_RE = re.compile(
    r"(?:F|functional|nn\.functional|torch\.nn\.functional)\s*\.\s*[a-zA-Z_]\w*\s*\(")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return path.read_text(encoding=enc)
        except (OSError, UnicodeDecodeError):
            continue
    return None


def _json(path: Optional[Path]) -> Optional[Any]:
    txt = _read(path)
    if txt is None:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def _read_lines(path: Optional[Path]) -> list[str]:
    txt = _read(path)
    if txt is None:
        return []
    return txt.splitlines()


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Cost metrics (tokens / wall clock / iterations)
# ---------------------------------------------------------------------------

def cost_metrics(workspace: Path) -> dict[str, Any]:
    """Extract cost evidence: tokens, wall clock, iteration counts."""
    meta = _json(workspace / "run_meta.json") or {}
    cost: dict[str, Any] = {
        "harness": meta.get("harness"),
        "model": meta.get("model"),
        "provider": meta.get("provider"),
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "total_tokens": meta.get("total_tokens"),
        "cache_read_tokens": meta.get("cache_read_tokens"),
        "wall_ms": meta.get("wall_ms"),
        "tool_calls": meta.get("tool_calls"),
        "spawns": meta.get("spawns"),
        "tokens_captured": bool(meta.get("total_tokens") is not None),
    }

    # Iterations from the state machine log.
    transitions = read_transitions(workspace / "state_transitions.jsonl")
    iters = {}
    for entry in transitions:
        for stage, n in (entry.get("iteration") or {}).items():
            iters[stage] = max(iters.get(stage, 0), n)
    cost["iterations"] = iters or None
    cost["reach_stage"] = transitions[-1].get("to") if transitions else None

    # Wall-clock proxy: timestamps inside the transition log.
    if cost["wall_ms"] is None and len(transitions) >= 2:
        t0 = _parse_ts(transitions[0].get("timestamp"))
        t1 = _parse_ts(transitions[-1].get("timestamp"))
        if t0 and t1 and t1 >= t0:
            cost["wall_ms"] = int((t1 - t0).total_seconds() * 1000)
            cost["wall_from_transitions"] = True

    return cost


def read_transitions(path: Optional[Path]) -> list[dict]:
    """Parse the append-only ``state_transitions.jsonl`` log."""
    out: list[dict] = []
    for line in _read_lines(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Correctness / perf / convergence evidence
# ---------------------------------------------------------------------------

def test_outcome(workspace: Path) -> dict[str, Any]:
    """Parse ``test_result.md`` for L1-L5 pass/fail and bit-exact markers."""
    txt = _read(workspace / "test_result.md") or ""
    m = re.search(r"(\d+)\s+passed,\s*(\d+)\s+failed", txt)
    passed = int(m.group(1)) if m else None
    failed = int(m.group(2)) if m else None
    bit_exact = bool(re.search(r"bit[\s-]?exact", txt, re.IGNORECASE))
    has_counts = passed is not None and failed is not None
    return {
        "passed": passed,
        "failed": failed,
        "total": (passed + failed) if has_counts else None,
        "bit_exact": bit_exact,
        "has_test_result": bool(txt),
    }


def bench_rows(workspace: Path) -> tuple[list[dict[str, Any]], str]:
    """Return ``(rows, source)`` for the per-config speedup evidence.

    Prefers the **structured** ``bench_result.json`` that ``bench.py`` emits, and
    only falls back to parsing ``speedup.md`` Markdown when it is absent (older
    runs such as ``outputs/relu``). Structured evidence is the contract; Markdown
    parsing is the fragile legacy path (bold ``**1.04×**`` cells, prose rows).
    """
    data = _json(workspace / "bench_result.json")
    if isinstance(data, list):
        rows: list[dict[str, Any]] = []
        for r in data:
            if not isinstance(r, dict):
                continue
            spd = r.get("speedup")
            if spd is None:
                continue
            try:
                spd = float(spd)
            except (TypeError, ValueError):
                continue
            shape = r.get("shape") or r.get("ashape")
            rows.append({"shape": _norm_shape(shape), "dtype": r.get("dtype"),
                         "speedup": spd,
                         # V4: repeat-measurement spread, carried as evidence
                         "speedup_std": r.get("speedup_std"),
                         "runs": r.get("runs")})
        if rows:
            if _is_stale(workspace / "bench_result.json", workspace / "kernel.py"):
                # Structured evidence predates the current kernel: it measures a
                # different implementation, so trust the reviewed report instead.
                return speedup_rows(workspace), "speedup.md (bench_result.json stale)"
            return rows, "bench_result.json"
    return speedup_rows(workspace), "speedup.md"


def _is_stale(evidence: Path, source: Path) -> bool:
    """True when ``evidence`` was written before ``source`` (i.e. is out of date)."""
    try:
        return evidence.stat().st_mtime < source.stat().st_mtime
    except OSError:
        return False


def _norm_shape(shape: Any) -> Optional[tuple[int, ...]]:
    """Normalise a shape to a tuple of ints (accepts list, tuple, or '[a, b]')."""
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)):
        try:
            return tuple(int(v) for v in shape)
        except (TypeError, ValueError):
            return None
    nums = re.findall(r"\d+", str(shape))
    return tuple(int(n) for n in nums) if nums else None


def shape_coverage(rows: list[dict[str, Any]], case: Optional[dict]) -> dict[str, Any]:
    """Check the case's ``required_shapes`` against what the run actually produced.

    This is the **scalability gate**: a case declares the shapes a solved run must
    cover (e.g. attention at S=128/256, not just a toy S=64). A shape that never
    ran (UB overflow, crash, not attempted) counts as *uncovered*, so a kernel that
    is fast only on toy sizes cannot score as if it solved the case.
    """
    required = (case or {}).get("required_shapes") or []
    if not required:
        return {"gate": 1.0, "covered": None, "total": 0, "missing": [],
                "note": "case declares no required_shapes; gate disabled"}
    have = {tuple(r["shape"]) for r in rows if r.get("shape")}
    missing = []
    covered = 0
    for req in required:
        t = _norm_shape(req)
        if t in have:
            covered += 1
        else:
            missing.append(list(t) if t else req)
    total = len(required)
    return {"gate": covered / total, "covered": covered, "total": total,
            "missing": missing,
            "note": f"{covered}/{total} required shapes present"}


def speedup_deciding(rows: list[dict[str, Any]],
                     case: Optional[dict] = None) -> Optional[float]:
    """Pick the 'deciding' speedup, conservatively.

    Scope: the largest shape the **case requires** (``required_shapes``), so an
    extra exploratory shape a run happens to measure cannot change the verdict;
    falls back to the largest measured shape when no requirements are declared.

    Within that shape take the **worst** dtype, not the best: a run should not
    look good because one dtype won while another lagged (matmul fp16 0.51x vs
    fp32 0.80x).
    """
    if not rows:
        return None
    required = [tuple(t) for t in (_norm_shape(r) for r in
                                   ((case or {}).get("required_shapes") or [])) if t]
    pool = [r for r in rows if r.get("shape") in required] if required else rows
    if not pool:
        pool = rows
    maxsize = max(_shape_size(r.get("shape")) for r in pool)
    biggest = [float(r["speedup"]) for r in pool
               if _shape_size(r.get("shape")) == maxsize]
    return min(biggest)


def _shape_size(shape: Any) -> int:
    t = _norm_shape(shape)
    if not t:
        return 0
    prod = 1
    for n in t:
        prod *= n
    return prod


def speedup_rows(workspace: Path) -> list[dict[str, Any]]:
    """Parse ``speedup.md`` for per-config speedup values (legacy fallback).

    Rows look like ``| [16384, 16384] | fp16 | 0.86959 | 0.84111 | 0.97× | ... |``.
    The speedup cell is the one that carries a ``×`` / ``x`` suffix; durations
    (which are also numeric) are ignored.
    """
    rows: list[dict[str, Any]] = []
    for line in _read_lines(workspace / "speedup.md"):
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # A speedup cell ends with × / x and parses as a float in (0, 1000).
        # Markdown bold (``**1.04×**``) is stripped before the suffix test.
        spd = None
        for c in cells:
            cc = c.strip().replace("*", "").strip()
            if cc.endswith(("×", "x", "X")):
                core = cc.rstrip("×xX ").strip()
                if not core:
                    continue
                try:
                    v = float(core)
                except ValueError:
                    continue
                if 0.0 < v < 1000.0:
                    spd = v
                    break
        if spd is None:
            continue
        shape = None
        for c in cells:
            # Take only the FIRST bracketed group so matmul's "[M,K]@[K,N]"
            # shape cell normalises to the A shape, not all four dims.
            m_shape = re.match(r"^\[([^\]]*)\]", c)
            if m_shape:
                shape = m_shape.group(1)
                break
        dtype = next((c for c in cells if c in ("fp16", "fp32", "bf16", "fp8")), None)
        if shape is None and dtype is None:
            # Not a data row (e.g. a prose/legend line that happened to hold a
            # trailing "×"); every real speedup row carries a shape.
            continue
        rows.append({"shape": _norm_shape(shape), "dtype": dtype, "speedup": spd})
    return rows


def bottleneck_structured(workspace: Path) -> dict[str, Any]:
    """Read the machine-readable ``bottleneck.json`` convergence contract.

    Designed to replace keyword matching for D4 (convergence), which was
    wording-sensitive: a report that wrote "轮 N" instead of "P0/P1/P2/P3" lost
    20 points for expressing the same thing. Runs that emit this file are scored
    on structure; runs that do not fall back to the keyword heuristic and are
    flagged ``present=False`` so the weaker evidence source is visible.

    Expected shape::

        {"classification": "memory_bound",
         "bandwidth": {"measured": 1.29, "unit": "TB/s", "utilization_pct": 81},
         "suggestions": [{"id": "S1", "priority": "P1", "actionable": false}],
         "convergence": {"reason": "NO_IMPROVEMENT", "judgement": "..."}}
    """
    d = _json(workspace / "bottleneck.json")
    if not isinstance(d, dict):
        return {"present": False}
    bw = d.get("bandwidth")
    conv = d.get("convergence")
    sug = d.get("suggestions")
    return {
        "present": True,
        "classification": d.get("classification"),
        "bandwidth": bw if isinstance(bw, dict) else {},
        "suggestions": sug if isinstance(sug, list) else [],
        "convergence": conv if isinstance(conv, dict) else {},
    }


def bottleneck_evidence(workspace: Path) -> dict[str, Any]:
    """Extract bottleneck classification and suggestion grades."""
    txt = _read(workspace / "bottleneck.md") or ""
    opt = _read(workspace / "opt_history.md") or ""
    joined = txt + "\n" + opt
    lower = joined.lower()
    classification = None
    if "memory-bound" in lower or "memory bound" in lower or "bandwidth" in lower:
        classification = "memory_bound"
    elif "compute-bound" in lower or "compute bound" in lower:
        classification = "compute_bound"
    if classification is None and "launch" in lower and "bound" in lower:
        classification = "launch_bound"
    has_suggestion_grading = any(
        p in lower for p in ("p0", "p1", "p2", "p3"))
    has_bandwidth = bool(re.search(r"(TB/s|GB/s|带宽)", joined))
    return {
        "classification": classification,
        "has_suggestion_grading": has_suggestion_grading,
        "has_bandwidth": has_bandwidth,
    }


def convergence_evidence(workspace: Path) -> dict[str, Any]:
    """Extract convergence honesty evidence."""
    transitions = read_transitions(workspace / "state_transitions.jsonl")
    opt = _read(workspace / "opt_history.md") or ""
    delivery = _read(workspace / "delivery.md") or ""
    handoffs = [e.get("handoff", "") for e in transitions]
    done_handoff = next((h for h in handoffs if "OPTIMIZE_DONE" in h), None)
    reached_done = bool(transitions and transitions[-1].get("to") == "done")
    reason = None
    handoff_text = " ".join(handoffs)
    for token in ("NO_SUGGESTIONS", "HARDWARE_LIMIT", "TARGET_MET",
                  "NO_IMPROVEMENT", "CAP"):
        if token in handoff_text or token in (done_handoff or ""):
            reason = token
            break
    has_judgement = bool(re.search(r"收敛|converge|收敛判断", opt + delivery))
    return {
        "reached_done": reached_done,
        "done_handoff": done_handoff,
        "convergence_reason": reason,
        "has_judgement": has_judgement,
    }


def deliverable_files(workspace: Path) -> list[dict[str, Any]]:
    """Check presence / size of the mandatory deliverable set."""
    required = [
        ("design.md", 500),
        ("kernel.py", 1),
        ("test.py", 1),
        ("test_result.md", 200),
        ("speedup.md", 200),
        ("bottleneck.md", 200),
        ("bottleneck.json", 50),
        ("execution_log.md", 100),
        ("delivery.md", 200),
        ("env_snapshot.md", 1),
    ]
    result = []
    for name, min_bytes in required:
        p = workspace / name
        ok = p.exists() and p.stat().st_size >= min_bytes
        result.append({"file": name, "present": ok,
                       "bytes": p.stat().st_size if p.exists() else 0})
    return result


def anti_cheat(workspace: Path) -> dict[str, Any]:
    """Verify the kernel is a genuine Triton implementation, not a torch wrapper.

    Only **calls** to a torch functional/op path are treated as delegation, and
    only in *code* — docstrings/comments are stripped first, so a docstring that
    names ``torch.nn.functional.gelu(...)`` as the reference is not a wrapper.
    """
    txt = _read(workspace / "kernel.py")
    if txt is None:
        return {"kernel_found": False, "native_triton": False,
                "reasons": ["kernel.py missing"]}
    lower = txt.lower()
    code = _code_only(txt)
    reasons: list[str] = []
    native = ("triton" in lower) and (
        "tl.load" in code or "tl.store" in code or "tl.arange" in code)
    if not native:
        reasons.append("kernel.py is not Triton-native "
                       "(missing @triton.jit / tl.load / tl.store)")

    # (1) direct call into a torch functional path
    if _CALL_RE.search(code):
        reasons.append("kernel.py delegates to a torch.nn.functional op "
                       "(forbidden wrapper)")

    # (2) aliased / indirect delegation - the obvious bypasses of (1).
    #     V5 adversarial testing showed all three of these slipped through the
    #     first version when a dummy @triton.jit kernel was also present.
    alias_re = r"import\s+torch\.nn\.functional\s+as\s+([A-Za-z_]\w*)"
    for alias in re.findall(alias_re, code):
        if re.search(rf"\b{re.escape(alias)}\s*\.", code):
            reasons.append("kernel.py delegates via aliased module "
                           f"'{alias}' (forbidden wrapper)")
    if re.search(r"from\s+torch\.nn\.functional\s+import\s", code):
        reasons.append("kernel.py imports names from torch.nn.functional "
                       "(forbidden wrapper)")
    if re.search(r"getattr\s*\(\s*torch", code):
        reasons.append("kernel.py uses getattr() indirection into torch "
                       "(forbidden wrapper)")
    if re.search(r"\beval\s*\(", code) and "torch" in code:
        reasons.append("kernel.py uses eval() with torch (indirect wrapper)")
    if re.search(r"\bimportlib\b", code) and "torch" in code:
        reasons.append("kernel.py uses importlib with torch (indirect wrapper)")

    # (3) dummy kernel: declares @triton.jit but never launches it while also
    #     pulling in torch, i.e. the Triton code is decoration and the real work
    #     happens in torch. Guarded on "torch is present" so a kernel-only file
    #     (launcher lives in test.py) is not falsely accused.
    jit_names = re.findall(r"@triton\.jit\b[^\n]*\n\s*def\s+([A-Za-z_]\w*)", code)
    if not jit_names:
        reasons.append("kernel.py is missing the @triton.jit decorator")
    elif ("torch" in code
          and not any(re.search(rf"\b{re.escape(n)}\s*\[", code) for n in jit_names)):
        reasons.append(
            "kernel.py declares @triton.jit kernel(s) but never launches one "
            "(dummy kernel; real work is delegated)")

    return {
        "kernel_found": True,
        "native_triton": native,
        "jit_kernels": jit_names,
        "reasons": reasons,
        "pass": native and not reasons,
    }


def _code_only(text: str) -> str:
    """Return the source with comments and string literals (docstrings) blanked.

    Blanks the *spans* of COMMENT/STRING tokens in place rather than rejoining
    the surviving tokens, so original whitespace and formatting survive
    (rejoining produced ``defk`` instead of ``def k`` and broke identifier
    regexes). Falls back to the raw text if tokenising fails.
    """
    try:
        import io
        import tokenize
        spans = [(tok.start, tok.end)
                 for tok in tokenize.generate_tokens(io.StringIO(text).readline)
                 if tok.type in (tokenize.COMMENT, tokenize.STRING)]
        if not spans:
            return text
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        chars = list(text)
        for (sr, sc), (er, ec) in spans:
            start = offsets[sr - 1] + sc
            end = offsets[er - 1] + ec
            for i in range(start, min(end, len(chars))):
                if chars[i] != "\n":
                    chars[i] = " "
        return "".join(chars)
    except Exception:
        return text


def design_quality(workspace: Path) -> dict[str, Any]:
    """Measure design completeness via section count and size."""
    lines = _read_lines(workspace / "design.md")
    sections = [ln for ln in lines if ln.startswith("## ")]
    return {
        "lines": len(lines),
        "sections": len(sections),
        "bytes": (workspace / "design.md").stat().st_size
        if (workspace / "design.md").exists() else 0,
    }


def extract(workspace: Path, case: Optional[dict] = None) -> dict[str, Any]:
    """Collect every raw metric for a run workspace.

    ``case`` (the manifest entry) supplies ``required_shapes`` for the
    scalability gate; without it the gate is disabled (gate = 1.0).
    """
    ws = Path(workspace)
    rows, source = bench_rows(ws)
    return {
        "workspace": str(ws),
        "cost": cost_metrics(ws),
        "test": test_outcome(ws),
        "speedup": {
            "deciding": speedup_deciding(rows, case),
            "rows": rows,
            "source": source,
            "coverage": shape_coverage(rows, case),
        },
        "bottleneck": bottleneck_evidence(ws),
        "bottleneck_structured": bottleneck_structured(ws),
        "convergence": convergence_evidence(ws),
        "deliverables": deliverable_files(ws),
        "anti_cheat": anti_cheat(ws),
        "design": design_quality(ws),
    }


if __name__ == "__main__":
    import sys
    force_utf8_stdout()

    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/relu")
    print(json.dumps(extract(ws), ensure_ascii=False, indent=2))
