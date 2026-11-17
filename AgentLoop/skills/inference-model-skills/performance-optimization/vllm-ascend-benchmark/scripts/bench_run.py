#!/usr/bin/env python3
"""Run vllm bench serve benchmarks on a workspace-managed remote container.

Supports single-run and multi-run (warm-service) modes.  In multi-run mode
the service is started once and multiple benchmark iterations run against the
same warm service, with optional warmup runs excluded from the aggregated
statistics.

Usage examples:

    # Minimal single run
    python3 bench_run.py --host 173.131.1.2 --container my-container --model /home/weights/Qwen3.5-35B

    # Multi-run with warmup (start service once, run 3 times, discard first)
    python3 bench_run.py --host 173.131.1.2 --container my-container \\
        --model /home/weights/Qwen3.5-35B --runs 3 --warmup-runs 1 --tp 4

    # With explicit serve and bench args
    python3 bench_run.py --host 173.131.1.2 --container my-container \\
        --model /home/weights/Qwen3.5-35B --tp 4 \\
        --serve-args --async-scheduling --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \\
        --bench-args --num-prompts 128 --max-concurrency 32 --output-len 1500


Progress on stderr as __WORKSPACE_BENCHMARK_PROGRESS__=<json>.
Final result on stdout as a single JSON object.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _common import (
    assemble_config,
    call_serve_start,
    call_serve_stop,
    emit_progress,
    extract_metrics,
    now_utc,
    print_json,
    resolve_bench_target,
    run_bench_on_remote,
    save_bench_log,
    write_local_result,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run vllm bench serve benchmarks (single or multi-run).",
        allow_abbrev=False,
    )
    # ---- direct SSH connection ----
    p.add_argument("--employee-id", required=True, help="传给 serving 的个人工号；benchmark 客户端仍在 /tmp 运行")
    p.add_argument("--host", required=True, help="direct SSH host IP")
    p.add_argument("--ssh-port", type=int, default=22, help="SSH port (default: 22)")
    p.add_argument("--user", default="root", help="SSH user (default: root)")
    p.add_argument("--password", help="SSH password (if not using key)")
    p.add_argument("--key-file", help="SSH private key file path")
    p.add_argument("--container", help="Docker container name to exec into")
    p.add_argument("--conda-env", help="conda environment name on the remote host "
                   "(mutually exclusive with --container)")
    p.add_argument("--mode", "--connection-mode", dest="connection_mode",
                   choices=["docker", "conda", "direct"], default="docker",
                   help="连接模式: docker / conda / direct — 默认: docker")
    p.add_argument("--cann-dir", default="/usr/local/Ascend",
                   help="CANN 安装根目录 (default: /usr/local/Ascend)")
    # --------------------------------------------------------
    p.add_argument("--model", required=True, help="remote model weight path")
    p.add_argument("--tp", "--tensor-parallel-size", type=int, default=None)
    p.add_argument("--dp", "--data-parallel-size", type=int, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--devices", default=None,
                   help="ASCEND_RT_VISIBLE_DEVICES, e.g. 8,9,10,11 (overrides auto-detection)")
    p.add_argument("--extra-env", action="append", default=None,
                   help="KEY=VALUE env vars for the service (repeatable)")

    p.add_argument("--runs", type=int, default=1,
                   help="number of benchmark iterations against the same warm service (default: 1)")
    p.add_argument("--warmup-runs", type=int, default=0,
                   help="number of initial runs to discard from aggregated statistics (default: 0)")
    # ---- multi-node (optional) ----
    p.add_argument("--worker-hosts", help="comma-separated worker IPs for multi-node deployment")
    p.add_argument("--master-port", type=int, help="DP RPC port (default: auto-allocate)")
    p.add_argument("--worker-key-file", help="SSH key for worker nodes (default: same as head)")
    p.add_argument("--worker-password", help="SSH password for worker nodes (default: same as head)")
    p.add_argument("--worker-container", help="Docker container for worker nodes (default: same as head)")
    p.add_argument("--worker-conda-env", help="Conda env for worker nodes (default: same as head)")
    p.add_argument("--worker-extra-env", action="append", default=None,
                   help="extra env vars for worker nodes (repeatable)")
    return p


def _split_sections(argv: list[str]) -> tuple[list[str], list[str] | None, list[str] | None]:
    """Split argv into (main_args, serve_args, bench_args).

    Recognizes --serve-args and --bench-args as section delimiters in any order.
    """
    delimiters = {"--serve-args", "--bench-args"}
    sections: dict[str, list[str]] = {}
    main_args: list[str] = []
    current_key: str | None = None

    for token in argv:
        if token in delimiters:
            if token in sections:
                raise SystemExit(
                    f"duplicate section delimiter {token}: "
                    f"previously collected {len(sections[token])} arg(s) would be lost; "
                    f"each section delimiter may appear at most once."
                )
            current_key = token
            sections[current_key] = []
        elif current_key is not None:
            sections[current_key].append(token)
        else:
            main_args.append(token)

    return (
        main_args,
        sections.get("--serve-args"),
        sections.get("--bench-args"),
    )


def _aggregate_metrics(
    all_runs: list[dict[str, Any]],
    warmup: int,
) -> dict[str, Any]:
    """Compute mean/stddev over the statistical runs (excluding warmup)."""
    stat_runs = all_runs[warmup:]
    if not stat_runs:
        return {}

    metric_keys = set()
    for m in stat_runs:
        metric_keys.update(m.keys())

    agg: dict[str, Any] = {"count": len(stat_runs)}
    for key in sorted(metric_keys):
        vals: list[float] = []
        for m in stat_runs:
            if key in m:
                try:
                    vals.append(float(m[key]))
                except (TypeError, ValueError):
                    continue
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            stddev = variance ** 0.5
        else:
            stddev = 0.0
        agg[key] = {"mean": round(mean, 4), "stddev": round(stddev, 4), "values": vals}

    return agg


def main(argv: list[str] | None = None) -> int:
    raw_argv = argv if argv is not None else sys.argv[1:]
    main_argv, manual_serve_args, manual_bench_args = _split_sections(raw_argv)

    args = build_parser().parse_args(main_argv)

    total_runs: int = max(1, args.runs)
    warmup_runs: int = max(0, min(args.warmup_runs, total_runs - 1))

    serve_args = manual_serve_args if manual_serve_args is not None else getattr(args, "serve_args", None)
    bench_args = manual_bench_args if manual_bench_args is not None else getattr(args, "bench_args", None)

    # merge --devices into extra_env as ASCEND_RT_VISIBLE_DEVICES
    extra_env = list(args.extra_env) if args.extra_env else []
    if args.devices:
        extra_env.append(f"ASCEND_RT_VISIBLE_DEVICES={args.devices}")

    try:
        config = assemble_config(
            host=args.host,
            employee_id=args.employee_id,
            ssh_port=args.ssh_port,
            user=args.user,
            password=args.password,
            key_file=args.key_file,
            container=args.container,
            conda_env=getattr(args, "conda_env", None),
            connection_mode=getattr(args, "connection_mode", "docker"),
            cann_dir=getattr(args, "cann_dir", "/usr/local/Ascend"),
            model=args.model,
            tp=args.tp,
            dp=args.dp,
            port=args.port,
            serve_args=serve_args,
            bench_args=bench_args,
            extra_env=extra_env or None,
            worker_hosts=getattr(args, "worker_hosts", None),
            master_port=getattr(args, "master_port", None),
            worker_key_file=getattr(args, "worker_key_file", None),
            worker_password=getattr(args, "worker_password", None),
            worker_container=getattr(args, "worker_container", None),
            worker_conda_env=getattr(args, "worker_conda_env", None),
            worker_extra_env=list(getattr(args, "worker_extra_env", None) or []),
        )

        emit_progress("start", "launching vllm service")
        start_result = call_serve_start(config)

        if start_result.get("status") != "ready":
            cleanup_result = call_serve_stop(config, force=True)
            fail_json = {
                "status": "failed",
                "phase": "serve_start",
                "error": start_result.get("error", "service did not become ready"),
                "serve_result": start_result,
                "cleanup_result": cleanup_result,
                "config": config.summary_dict(),
                "timestamp": now_utc(),
            }
            write_local_result(config, fail_json)
            print_json(fail_json)
            return 1

        base_url = start_result["base_url"]
        served_model = start_result.get("served_model_name", Path(args.model).name)
        bench_target = resolve_bench_target(config)

        all_metrics: list[dict[str, Any]] = []
        all_raw: list[dict[str, Any]] = []
        bench_log_paths: list[str] = []

        for i in range(total_runs):
            run_label = f"run {i + 1}/{total_runs}"
            is_warmup = i < warmup_runs
            tag = " (warmup)" if is_warmup else ""
            emit_progress("bench", f"{run_label}{tag}: running vllm bench serve")
            raw_stdout: str = ""
            try:
                raw_result, raw_stdout = run_bench_on_remote(
                    config, base_url, served_model, bench_target,
                )
                log_path = save_bench_log(config, i, raw_stdout)
                bench_log_paths.append(str(log_path))
            except Exception as e:
                emit_progress("bench", f"{run_label}: benchmark failed: {e}")
                call_serve_stop(config, force=True)
                fail_json = {
                    "status": "failed",
                    "phase": "bench_run",
                    "run": i + 1,
                    "error": str(e),
                    "completed_runs": [
                        {"run": j + 1, "warmup": j < warmup_runs, "metrics": m}
                        for j, m in enumerate(all_metrics)
                    ],
                    "config": config.summary_dict(),
                    "timestamp": now_utc(),
                }
                write_local_result(config, fail_json)
                print_json(fail_json)
                return 1

            metrics = extract_metrics(raw_result)
            all_metrics.append(metrics)
            all_raw.append(raw_result)
            throughput = metrics.get("output_throughput", "N/A")
            emit_progress("bench", f"{run_label}{tag}: throughput={throughput}")

        emit_progress("stop", "stopping service")
        stop_result = call_serve_stop(config)
        cleanup_warning: str | None = None
        if stop_result.get("status") not in ("stopped", "not_found"):
            emit_progress("stop", "graceful stop failed, retrying with force")
            stop_result = call_serve_stop(config, force=True)
            if stop_result.get("status") not in ("stopped", "not_found"):
                cleanup_warning = f"service may still be running: {stop_result}"

        if total_runs == 1:
            emit_progress("done", f"benchmark complete, throughput={all_metrics[0].get('output_throughput', 'N/A')}")
            result_json: dict[str, Any] = {
                "status": "ok",
                "host": args.host,
                "model": args.model,
                "metrics": all_metrics[0],
                "bench_log": bench_log_paths[0] if bench_log_paths else None,
                "config": config.summary_dict(),
                "raw_result": all_raw[0],
                "timestamp": now_utc(),
            }
            if cleanup_warning:
                result_json["cleanup_warning"] = cleanup_warning
            write_local_result(config, result_json)
            print_json(result_json)
        else:
            aggregated = _aggregate_metrics(all_metrics, warmup_runs)
            emit_progress(
                "done",
                f"benchmark complete: {total_runs} runs ({warmup_runs} warmup), "
                f"mean throughput={aggregated.get('output_throughput', {}).get('mean', 'N/A')}",
            )
            result_json = {
                "status": "ok",
                "host": args.host,
                "model": args.model,
                "runs": total_runs,
                "warmup_runs": warmup_runs,
                "aggregated": aggregated,
                "per_run": [
                    {
                        "run": j + 1,
                        "warmup": j < warmup_runs,
                        "metrics": m,
                        "bench_log": bench_log_paths[j] if j < len(bench_log_paths) else None,
                    }
                    for j, m in enumerate(all_metrics)
                ],
                "config": config.summary_dict(),
                "timestamp": now_utc(),
            }
            if cleanup_warning:
                result_json["cleanup_warning"] = cleanup_warning
            write_local_result(config, result_json)
            print_json(result_json)
        return 0

    except Exception as e:
        try:
            if "config" in locals():
                call_serve_stop(config, force=True)
        except Exception:
            pass
        fail_json = {
            "status": "failed",
            "phase": "unexpected",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": now_utc(),
        }
        if "config" in locals():
            fail_json["config"] = config.summary_dict()
            write_local_result(config, fail_json)
        print_json(fail_json)
        return 2


if __name__ == "__main__":
    sys.exit(main())
