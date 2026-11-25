#!/usr/bin/env python3
"""Run an inference/benchmark command while collecting SMA/NPU utilization logs.

This wrapper starts scripts/sma.py in parallel, streams the benchmark command to
stdout, writes benchmark/SMA/combined logs, then stops SMA and appends a summary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--devices",
        required=True,
        help="Comma-separated device ids; force-injected as ASCEND_RT_VISIBLE_DEVICES, e.g. 4,5,6,7",
    )
    p.add_argument("--sma-script", default=str(Path(__file__).with_name("sma.py")))
    p.add_argument("--npu-util-script", default=str(Path(__file__).with_name("npu_util.py")))
    p.add_argument("--log-dir", default=".")
    p.add_argument("--prefix", default="run")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--no-sma", action="store_true", help="Run command without starting SMA monitor")
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = p.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        p.error("missing command; use: run_with_sma.py [opts] -- <command...>")
    devices = [part.strip() for part in args.devices.split(",")]
    if not devices or any(not part.isdigit() for part in devices):
        p.error("--devices must be a comma-separated list of numeric device ids, e.g. 4,5,6,7")
    args.devices = ",".join(devices)
    return args


def pump(prefix: str, proc: subprocess.Popen, targets: list):
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        stamped = f"{dt.datetime.now().isoformat(timespec='seconds')} {prefix} {line}"
        print(stamped, flush=True)
        for f in targets:
            f.write(stamped + "\n")
            f.flush()


def terminate(proc: subprocess.Popen | None, timeout: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=timeout)
    except Exception:
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    child_env = os.environ.copy()
    child_env["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bench_log = log_dir / f"{args.prefix}-bench-{ts}.log"
    sma_log = log_dir / f"{args.prefix}-sma-{ts}.log"
    combined_log = log_dir / f"{args.prefix}-combined-{ts}.log"

    bench_file = bench_log.open("w", encoding="utf-8")
    sma_file = sma_log.open("w", encoding="utf-8")
    combined_file = combined_log.open("w", encoding="utf-8")

    header = [
        f"start_time={dt.datetime.now().isoformat(timespec='seconds')}",
        f"command={' '.join(args.command)}",
        f"ASCEND_RT_VISIBLE_DEVICES={args.devices}",
        f"bench_log={bench_log}",
        f"sma_log={sma_log}",
        f"combined_log={combined_log}",
    ]
    for line in header:
        print(line)
        combined_file.write(line + "\n")

    sma_proc = None
    bench_proc = None
    rc = 1
    threads: list[threading.Thread] = []
    try:
        if not args.no_sma:
            # Run in log_dir so sma.py's fixed npu_util.log lands beside the logs.
            sma_proc = subprocess.Popen(
                [args.python, str(Path(args.sma_script).resolve())],
                cwd=str(log_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=child_env,
                start_new_session=(os.name != "nt"),
            )
            t = threading.Thread(target=pump, args=("[SMA]", sma_proc, [sma_file, combined_file]), daemon=True)
            t.start()
            threads.append(t)

        bench_proc = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
            start_new_session=(os.name != "nt"),
        )
        t = threading.Thread(target=pump, args=("[BENCH]", bench_proc, [bench_file, combined_file]), daemon=True)
        t.start()
        threads.append(t)
        rc = bench_proc.wait()
    finally:
        terminate(sma_proc)
        for t in threads:
            t.join(timeout=2)

    summary = ""
    npu_log = log_dir / "npu_util.log"
    if npu_log.exists() and Path(args.npu_util_script).exists():
        try:
            summary_proc = subprocess.run(
                [args.python, str(Path(args.npu_util_script).resolve())],
                cwd=str(log_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
            summary = summary_proc.stdout.strip()
        except Exception as e:  # noqa: BLE001
            summary = f"npu_util summary failed: {e!r}"
    else:
        summary = "npu_util.log not found; SMA summary unavailable"

    footer = [
        f"end_time={dt.datetime.now().isoformat(timespec='seconds')}",
        f"returncode={rc}",
        f"sma_summary={summary}",
    ]
    for line in footer:
        print(line)
        combined_file.write(line + "\n")

    bench_file.close()
    sma_file.close()
    combined_file.close()
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
