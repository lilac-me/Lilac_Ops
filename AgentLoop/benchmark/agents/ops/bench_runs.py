"""Generic repeated-measurement bench harness for ops cases.

Run with the **case directory as the working directory** so that ``import kernel``
resolves to that case's ``kernel.py`` — this repo's copy is runnable from anywhere
(the script injects the cwd on ``sys.path`` below), so nothing has to be copied next
to the kernel::

    # inside the NPU container, in the case workspace
    cd outputs/gelu
    ASCEND_RT_VISIBLE_DEVICES=0 \\
        python /path/to/repo/benchmark/agents/ops/bench_runs.py --case gelu --runs 3

Writes ``bench_result.json`` with the **median** speedup per shape plus
``speedup_std`` / ``speedup_samples`` / ``runs`` (single samples are not trustworthy —
fp32 showed a 13% spread, gelu's largest shape drifted ~7% between runs).
Timing uses ``torch_npu.profiler(Level1)`` -> ``kernel_details.csv`` Duration(us);
``time.perf_counter`` is forbidden by the pipeline gate.

Before measuring, confirm the target card is idle (``npu-smi info``): a card shared
with another job once pushed the repeat-measurement std to 32%.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import statistics
import sys

import torch
import torch_npu  # noqa: F401

# ``import kernel`` in ``main()`` below must resolve to the kernel.py in the *current
# directory*. Running ``python /path/to/bench_runs.py`` puts the script's own directory
# first on sys.path, so the cwd has to be moved **to the front** — not merely added. A
# plain "is it in sys.path already?" guard is not enough: the cwd can already appear
# further back (containers often inject it), and then a stray `kernel.py` sitting next
# to the script silently wins. That is exactly what happened the first time this was
# tested: `/tmp/kernel.py` shadowed the case's kernel, the wrong operator got measured.
#
# The injection happens *after* the torch imports above on purpose: putting the cwd
# first before importing torch would let a stray `torch.py` shadow the real framework.
_CWD = os.getcwd()
if _CWD in sys.path:
    sys.path.remove(_CWD)
sys.path.insert(0, _CWD)


def _prof(fn, tag, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    d = f"./prof_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(d),
    ):
        for _ in range(repeat):
            fn()
            torch.npu.synchronize()
    dur = []
    for f in glob.glob(os.path.join(d, "**", "kernel_details.csv"), recursive=True):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                n = row.get("Name", "")
                v = float(row.get("Duration(us)", 0))
                if v > 0 and not any(x in n for x in ("Free", "Cast")):
                    dur.append(v)
    return torch.tensor(dur).mean().item() if dur else float("nan")


def _make(shape, dtype, n):
    """n identical input tensors of `shape`/`dtype` on npu (seeded)."""
    torch.manual_seed(0)
    return tuple(torch.randn(shape, device="npu", dtype=dtype) for _ in range(n))


def case_config(case, kernel):
    """-> (run_fn, ref_fn, entries, ) where entries =
    [(report_shape, dtype, args_factory)]; ref_fn(*args) is the ground truth.

    Each case builds its own inputs: rmsnorm needs a 1-D weight, matmul needs
    B=[K,N] (not the same shape as A), sdpa needs three [B,H,S,D] tensors.

    ⚠️ **The shapes and dtypes here must stay in sync with the case's
    ``required_shapes`` / ``dtypes`` in ``testcases/manifest.yaml``.** The two
    cannot read each other (this module runs inside the NPU container, alone, with
    no repo around it), so the duplication is structural — if you add a dtype or a
    shape to one, add it to the other. A drift shows up as either a coverage-gate
    miss (a required shape never measured) or a D2 deciding value computed over a
    different dtype set than the case declares.
    """
    functional = torch.nn.functional
    fp16 = torch.float16
    fp32 = torch.float32

    def vec(shape, dtype, n=1):
        return lambda: _make(shape, dtype, n)

    if case == "relu":
        return (kernel.relu, lambda x: functional.relu(x), [
            ((4096, 4096), fp32, vec((4096, 4096), fp32)),
            ((4096, 4096), fp16, vec((4096, 4096), fp16)),
            ((2048, 8192), fp16, vec((2048, 8192), fp16)),
            ((16384, 16384), fp16, vec((16384, 16384), fp16)),
        ])
    if case == "gelu":
        return (kernel.gelu, lambda x: functional.gelu(x, approximate="tanh"), [
            ((4096, 4096), fp32, vec((4096, 4096), fp32)),
            ((4096, 4096), fp16, vec((4096, 4096), fp16)),
            ((2048, 8192), fp16, vec((2048, 8192), fp16)),
            ((16384, 16384), fp16, vec((16384, 16384), fp16)),
        ])
    if case == "swiglu":
        return (kernel.swiglu, lambda g, u: functional.silu(g) * u, [
            ((4096, 4096), fp32, vec((4096, 4096), fp32, 2)),
            ((4096, 4096), fp16, vec((4096, 4096), fp16, 2)),
            ((2048, 8192), fp16, vec((2048, 8192), fp16, 2)),
            ((16384, 16384), fp16, vec((16384, 16384), fp16, 2)),
        ])
    if case == "rmsnorm":
        # [16384,16384] cannot complete (naive tl.sum reduction is 5-9x slow);
        # omitted, and the scoring coverage gate records the miss.
        def rms(shape, dtype):
            def f():
                torch.manual_seed(0)
                x = torch.randn(shape, device="npu", dtype=dtype)
                w = torch.randn(shape[-1], device="npu", dtype=dtype)   # 1-D weight
                return (x, w)
            return f

        def ref_rmsnorm(x, w):
            return functional.rms_norm(x, [x.shape[-1]], w, eps=1e-6)

        return (kernel.rmsnorm, ref_rmsnorm, [
            ((4096, 4096), fp32, rms((4096, 4096), fp32)),
            ((4096, 4096), fp16, rms((4096, 4096), fp16)),
            ((2048, 8192), fp16, rms((2048, 8192), fp16)),
        ])
    if case == "matmul":
        def mm(shape, dtype):
            m, k, n = shape

            def f():
                torch.manual_seed(0)
                return (torch.randn(m, k, device="npu", dtype=dtype),
                        torch.randn(k, n, device="npu", dtype=dtype))
            return f
        return (kernel.matmul, lambda a, b: a @ b, [
            ((1024, 1024), fp16, mm((1024, 1024, 1024), fp16)),
            ((2048, 1024), fp16, mm((2048, 1024, 2048), fp16)),
            ((4096, 4096), fp16, mm((4096, 4096, 4096), fp16)),
            ((4096, 4096), fp32, mm((4096, 4096, 4096), fp32)),
        ])
    if case == "sdpa":
        def s3(shape, dtype):
            return lambda: _make(shape, dtype, 3)

        def ref_sdpa(q, k, v):
            return functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # flash/causal version: S>=128 is now measurable (naive overflowed UB)
        return (kernel.sdpa, ref_sdpa, [
            ((1, 8, 64, 64), fp16, s3((1, 8, 64, 64), fp16)),
            ((1, 8, 128, 64), fp16, s3((1, 8, 128, 64), fp16)),
            ((1, 8, 256, 64), fp16, s3((1, 8, 256, 64), fp16)),
            ((2, 4, 512, 64), fp16, s3((2, 4, 512, 64), fp16)),
        ])
    raise SystemExit(f"unknown case {case!r}")


def bench_case(case, kernel, runs=3, quiet=False):
    """Measure every entry of ``case`` for ``kernel`` -> rows for bench_result.json.

    This is the **only** implementation of "how a case is measured", so a number
    produced here is comparable no matter who produced it: the per-case pipeline
    runs, a re-measurement after a kernel change, or a run in a different
    environment all come from the same code. Returns the list schema that
    ``metrics.bench_rows`` consumes.
    """
    run_fn, ref_fn, entries = case_config(case, kernel)

    results = []
    if not quiet:
        print(f"{'Shape':<22} {'Dtype':<8} {'Triton(us)':>11} {'torch(us)':>11} "
              f"{'Speedup':>9} {'std':>7} {'n':>3}", flush=True)
    for report_shape, dtype, factory in entries:
        tag = f"{str(report_shape):<22} {str(dtype).replace('torch.', ''):<8}"
        tris, tors, spds = [], [], []
        failure = None
        for _ in range(max(1, runs)):
            args_in = factory()
            try:
                # Bind the arguments as defaults: a bare `lambda: run_fn(*args_in)`
                # closes over the loop variable, which is correct only because
                # _prof calls it synchronously. Binding removes the footgun.
                t = _prof(lambda a=args_in: run_fn(*a), "tri")
                r = _prof(lambda a=args_in: ref_fn(*a), "torch")
            except Exception as exc:                      # noqa: BLE001
                failure = f"{type(exc).__name__}: {str(exc)[:160]}"
                break
            if t != t or r != r:
                continue
            tris.append(t)
            tors.append(r)
            spds.append(r / t)
        if failure is not None:
            results.append({"shape": list(report_shape), "dtype": str(dtype),
                            "error": failure})
            if not quiet:
                print(f"{tag}  ERROR {failure[:80]}", flush=True)
            continue
        if not spds:
            if not quiet:
                print(f"{tag}  NO_DATA (not measurable)", flush=True)
            continue
        med = statistics.median(spds)
        std = statistics.pstdev(spds) if len(spds) > 1 else 0.0
        results.append({
            "shape": list(report_shape), "dtype": str(dtype),
            "triton_us": statistics.median(tris),
            "torch_us": statistics.median(tors),
            "speedup": med, "speedup_std": std, "runs": len(spds),
            "speedup_samples": spds,
        })
        if not quiet:
            print(f"{tag} {statistics.median(tris):>11.2f} "
                  f"{statistics.median(tors):>11.2f} {med:>8.3f}x "
                  f"{std:>7.4f} {len(spds):>3}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    import kernel
    results = bench_case(args.case, kernel, runs=args.runs)

    with open("bench_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> bench_result.json ({len(results)} shapes, runs={args.runs})")


if __name__ == "__main__":
    main()
