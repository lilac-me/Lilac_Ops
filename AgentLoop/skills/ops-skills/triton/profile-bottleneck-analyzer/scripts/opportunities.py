"""Identify evidence-backed optimization opportunities, not proven fusion legality."""
import csv
import math
import re
from collections import defaultdict

MATMUL = re.compile(r"matmul|mat_mul|gemm|grouped[_ ]?mm|batch[_ ]?mm|(?:^|_)bmm(?:_|$)", re.I)
VECTOR = re.compile(r"gelu|silu|swish|relu|sigmoid|tanh|exp|rsqrt|sqrt|cast|add|sub|mul|div|clamp|maximum|minimum|square|neg|activation", re.I)
VECTOR_CORES = {"AI_VECTOR_CORE", "AIV", "VECTOR_CORE"}
CUBE_CORES = {"AI_CORE", "AIC", "MIX_AIC", "MIX_AIV"}


def number(value):
    try:
        n = float(str(value).strip())
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def norm(row):
    return {k.strip().replace("μ", "u").replace("µ", "u"): (v or "").strip()
            for k, v in row.items() if k is not None}


def known(value):
    return bool(value and value.lower() not in ("n/a", "none", "-", "nan"))


def identify(path, steps, threshold=0.03, max_gap_us=5.0, max_shapes=5):
    steady = {str(int(s)) for s in steps}
    totals = defaultdict(float)
    matmuls = defaultdict(list)
    streams = defaultdict(list)
    missing_timing = False
    with open(path, encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            r = norm(raw)
            st, dur = number(r.get("Step Id")), number(r.get("Duration(us)"))
            if st is None or str(int(st)) not in steady or dur is None or dur <= 0:
                continue
            step = str(int(st))
            totals[step] += dur
            name, kind = r.get("Name", ""), r.get("Type", "")
            core = r.get("Accelerator Core", "").upper()
            shape, dtype = r.get("Input Shapes", ""), r.get("Input Data Types", "")
            output, outdtype = r.get("Output Shapes", ""), r.get("Output Data Types", "")
            start = number(r.get("Start Time(us)", r.get("Task Start Time(us)")))
            stream = r.get("Stream ID", "")
            item = {"name": name, "type": kind, "step": step, "stream": stream,
                    "start_us": start, "duration_us": dur, "shape": shape, "dtype": dtype,
                    "output_shape": output, "output_dtype": outdtype, "core": core,
                    "task_id": r.get("Task ID", ""),
                    "pipe": {k: number(r[k]) for k in ("aic_mac_ratio", "aic_mte2_ratio",
                              "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
                              "cube_utilization(%)", "cube_utilization( %)")
                             if k in r and number(r[k]) is not None}}
            is_matmul = bool(MATMUL.search(name + " " + kind))
            if is_matmul:
                matmuls[name].append(item)
            item["vector"] = (not is_matmul and core in VECTOR_CORES
                              and bool(VECTOR.search(name + " " + kind))
                              and known(shape) and known(dtype))
            if start is None or not known(stream):
                missing_timing = True
            else:
                streams[(step, stream)].append(item)
    def ratio(rows):
        by_step = defaultdict(float)
        for row in rows:
            by_step[row["step"]] += row["duration_us"]
        return sum(by_step[s] / totals[s] if totals[s] else 0 for s in steady) / max(1, len(steady))

    mm = []
    for name, rows in matmuls.items():
        share = ratio(rows)
        if share < threshold:
            continue
        shapes = {}
        for row in rows:
            if known(row["shape"]) and known(row["dtype"]):
                key = (row["shape"], row["dtype"], row["core"])
                shapes.setdefault(key, {"shape": key[0], "dtype": key[1], "core": key[2]})
        mm.append({"name": name, "route": "matmul_mfu", "mean_op_ratio": share,
                   "calls": len(rows), "shapes": list(shapes.values())[:max_shapes],
                   "evidence": rows[:max_shapes], "status": "NEEDS_SOURCE_AND_FLOPS",
                   "note": "Pipe metrics are hints, not MFU; resolve GEMM semantics and FLOPs, then benchmark."})
    groups = defaultdict(list)
    # If chronology is incomplete, do not infer adjacency by skipping unknown rows.
    if not missing_timing:
        for key, rows in streams.items():
            rows.sort(key=lambda r: r["start_us"])
            chain = []
            def flush():
                if len(chain) >= 2:
                    signature = tuple((r["name"], r["shape"], r["dtype"]) for r in chain)
                    groups[signature].append(list(chain))
                chain.clear()
            for row in rows:
                if not row["vector"]:
                    flush()
                    continue
                if chain:
                    prev = chain[-1]
                    gap = row["start_us"] - prev["start_us"] - prev["duration_us"]
                    # Matching shape is weak evidence only, never a proof of producer/consumer.
                    if (not 0 <= gap <= max_gap_us or len(chain) >= 8
                            or not known(prev["output_shape"])
                            or prev["output_shape"] != row["shape"]
                            or not known(prev["output_dtype"])
                            or prev["output_dtype"] != row["dtype"]):
                        flush()
                chain.append(row)
            flush()
    fusion = []
    for signature, chains in groups.items():
        share = ratio([r for chain in chains for r in chain])
        if share < threshold:
            continue
        fusion.append({"route": "vector_fusion", "names": [x[0] for x in signature],
                       "mean_op_ratio": share, "occurrences": len(chains),
                       "evidence": chains[:max_shapes], "status": "NEEDS_DATAFLOW_REVIEW",
                       "note": "Same stream/step, close timing and compatible shapes are not proof of dependency. Check real graph, aliasing, autograd and external consumers before generating a fused op."})
    return {"matmul_mfu": sorted(mm, key=lambda x: -x["mean_op_ratio"]),
            "vector_fusion": sorted(fusion, key=lambda x: -x["mean_op_ratio"]),
            "fusion_scan_complete": not missing_timing,
            "note": "Kernel-duration shares are not critical-path wall time. Builtin candidates remain advisory."}
