#!/usr/bin/env python3
"""Shared utilities for vllm-ascend-benchmark scripts.

SSH primitives and target resolution are imported from the serving skill's
``_common`` so we do not maintain a second copy.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
SERVING_SCRIPTS = ROOT / "skills" / "inference-model-skills" / "deployment" / "vllm-ascend-serving" / "scripts"

# TODO: Move it to a shared common directory once one is established
#       These are currently scattered across each skill and should be
#       consolidated into a single shared directory.

if str(SERVING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SERVING_SCRIPTS))

from workspace_validate import require_env_name  # noqa: E402
from workspace_output import load_serving_common, emit_progress as _emit_progress, print_json, now_utc  # noqa: E402

BENCHMARK_STATE_DIR = ROOT / ".workspace-local" / "benchmark"
PROGRESS_SENTINEL = "__WORKSPACE_BENCHMARK_PROGRESS__="


SERVING = load_serving_common(
    module_alias="workspace_bench_serving_common",
    serving_scripts_dir=SERVING_SCRIPTS,
)
SshEndpoint = SERVING.SshEndpoint
ssh_exec = SERVING.ssh_exec
ssh_exec_target = SERVING.ssh_exec_target
_container_wrap = SERVING._container_wrap
resolve_direct_target = SERVING.resolve_direct_target
ExecutionTarget = SERVING.ExecutionTarget
safe_token = SERVING.safe_token


emit_progress = partial(_emit_progress, sentinel=PROGRESS_SENTINEL)


def benchmark_runs_dir(config: "BenchConfig") -> Path:
    target = safe_token(config.host or "legacy")
    return BENCHMARK_STATE_DIR / target / "runs"


def write_local_result(config: "BenchConfig", result: dict[str, Any]) -> Path:
    runs_dir = benchmark_runs_dir(config)
    runs_dir.mkdir(parents=True, exist_ok=True)
    target_token = safe_token(config.host or "benchmark")
    filename = (
        f"{now_utc().replace(':', '-')}_{target_token}_"
        f"{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
    )
    result_path = runs_dir / filename
    result["result_path"] = str(result_path)
    result["run_dir"] = str(runs_dir)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def _run_json_command_streaming(
    cmd: list[str],
    *,
    progress_markers: tuple[str, ...] = (),
) -> tuple[int, dict[str, Any] | None, str, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
    )
    stderr_lines: list[str] = []

    def relay_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            if not progress_markers or any(marker in line for marker in progress_markers):
                sys.stderr.write(line)
                sys.stderr.flush()

    thread = threading.Thread(target=relay_stderr, daemon=True)
    thread.start()
    assert proc.stdout is not None
    stdout = proc.stdout.read()
    returncode = proc.wait()
    thread.join(timeout=1)
    stderr = "".join(stderr_lines)
    payload = None
    if stdout.strip():
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = None
    return returncode, payload, stdout, stderr



# ---------------------------------------------------------------------------
# Configuration assembly
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    """Assembled benchmark configuration ready for execution."""
    # ---- direct SSH connection ----
    employee_id: str = ""
    host: str | None = None
    ssh_port: int = 22
    user: str = "root"
    password: str | None = None
    key_file: str | None = None
    container: str | None = None
    conda_env: str | None = None
    connection_mode: str = "docker"
    cann_dir: str = "/usr/local/Ascend"
    # ---- multi-node (optional) ----
    worker_hosts: str | None = None
    master_port: int | None = None
    worker_key_file: str | None = None
    worker_password: str | None = None
    worker_container: str | None = None
    worker_conda_env: str | None = None
    worker_extra_env: list[str] = field(default_factory=list)
    # --------------------------------------------------------
    model: str = ""
    tp: int | None = None
    dp: int | None = None
    port: int | None = None
    serve_args: list[str] = field(default_factory=list)
    bench_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def to_serve_start_args(self) -> list[str]:
        """Build CLI args for serve_start.py."""
        args: list[str] = ["--employee-id", self.employee_id]
        # ---- target selection ----
        args.extend(["--host", self.host])
        if self.ssh_port != 22:
            args.extend(["--ssh-port", str(self.ssh_port)])
        if self.user != "root":
            args.extend(["--user", self.user])
        if self.password:
            args.extend(["--password", self.password])
        if self.key_file:
            args.extend(["--key-file", self.key_file])
        if self.container:
            args.extend(["--container", self.container])
        if self.conda_env:
            args.extend(["--conda-env", self.conda_env])
        if self.connection_mode != "docker":
            args.extend(["--mode", self.connection_mode])
        if self.cann_dir != "/usr/local/Ascend":
            args.extend(["--cann-dir", self.cann_dir])
        # ---- serve params ----
        args.extend(["--model", self.model])
        if self.tp is not None:
            args.extend(["--tp", str(self.tp)])
        if self.dp is not None:
            args.extend(["--dp", str(self.dp)])
        if self.port is not None:
            args.extend(["--port", str(self.port)])
        for k, v in self.env.items():
            args.extend(["--extra-env", f"{k}={v}"])

        # ---- multi-node ----
        if self.worker_hosts:
            args.extend(["--worker-hosts", self.worker_hosts])
        if self.master_port is not None:
            args.extend(["--master-port", str(self.master_port)])
        if self.worker_key_file:
            args.extend(["--worker-key-file", self.worker_key_file])
        if self.worker_password:
            args.extend(["--worker-password", self.worker_password])
        if self.worker_container:
            args.extend(["--worker-container", self.worker_container])
        if self.worker_conda_env:
            args.extend(["--worker-conda-env", self.worker_conda_env])
        for item in self.worker_extra_env:
            args.extend(["--worker-extra-env", item])

        if self.serve_args:
            args.append("--")
            args.extend(self.serve_args)
        return args

    @property
    def is_direct(self) -> bool:
        """True when direct connection params are used."""
        return bool(self.host)

    def to_bench_serve_args(
        self, base_url: str, served_model_name: str,
    ) -> list[str]:
        """Build CLI args for `vllm bench serve`."""
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = str(parsed.port or 8000)

        args = [
            "vllm", "bench", "serve",
            "--backend", "vllm",
            "--host", host,
            "--port", port,
            "--model", "",
            "--tokenizer", self.model,
            "--save-result",
        ]
        has_num_prompts = any(a.startswith("--num-prompts") for a in self.bench_args)
        has_concurrency = any(a.startswith("--max-concurrency") for a in self.bench_args)

        if not has_num_prompts:
            args.extend(["--num-prompts", "64"])
        if not has_concurrency:
            args.extend(["--max-concurrency", "16"])

        args.extend(self.bench_args)
        return args

    def summary_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "employee_id": self.employee_id}
        d["host"] = self.host
        d["ssh_port"] = self.ssh_port
        if self.container:
            d["container"] = self.container
        if self.conda_env:
            d["conda_env"] = self.conda_env
        if self.cann_dir != "/usr/local/Ascend":
            d["cann_dir"] = self.cann_dir
        if self.tp is not None:
            d["tp"] = self.tp
        if self.serve_args:
            d["serve_args"] = self.serve_args
        if self.bench_args:
            d["bench_args"] = self.bench_args
        if self.env:
            d["env"] = self.env
        if self.worker_hosts:
            d["worker_hosts"] = self.worker_hosts
            d["nnodes"] = 1 + self.worker_hosts.count(",") + (1 if self.worker_hosts else 0)
        return d


def assemble_config(
    *,
    host: str,
    employee_id: str,
    ssh_port: int = 22,
    user: str = "root",
    password: str | None = None,
    key_file: str | None = None,
    container: str | None = None,
    conda_env: str | None = None,
    connection_mode: str = "docker",
    cann_dir: str = "/usr/local/Ascend",
    model: str,
    tp: int | None = None,
    dp: int | None = None,
    port: int | None = None,
    serve_args: list[str] | None = None,
    bench_args: list[str] | None = None,
    extra_env: list[str] | None = None,
    worker_hosts: str | None = None,
    master_port: int | None = None,
    worker_key_file: str | None = None,
    worker_password: str | None = None,
    worker_container: str | None = None,
    worker_conda_env: str | None = None,
    worker_extra_env: list[str] | None = None,
) -> BenchConfig:
    """Assemble a BenchConfig from user-provided parameters."""
    if not host:
        raise RuntimeError("--host is required")

    SERVING.employee_runtime_base(employee_id)
    cfg = BenchConfig(
        employee_id=employee_id,
        host=host,
        ssh_port=ssh_port,
        user=user,
        password=password,
        key_file=key_file,
        container=container,
        conda_env=conda_env if not container else None,
        connection_mode=connection_mode,
        cann_dir=cann_dir,
        model=model,
        worker_hosts=worker_hosts,
        master_port=master_port,
        worker_key_file=worker_key_file,
        worker_password=worker_password,
        worker_container=worker_container,
        worker_conda_env=worker_conda_env,
        worker_extra_env=list(worker_extra_env) if worker_extra_env else [],
    )

    if tp is not None:
        cfg.tp = tp

    if dp is not None:
        cfg.dp = dp

    cfg.port = port

    if serve_args:
        cfg.serve_args = list(serve_args)

    if bench_args:
        cfg.bench_args = list(bench_args)

    env: dict[str, str] = {}
    if extra_env:
        for item in extra_env:
            if "=" not in item:
                raise ValueError(f"bad --extra-env {item!r}, expected KEY=VALUE")
            k, v = item.split("=", 1)
            env[require_env_name(k)] = v
    cfg.env = env

    return cfg


# ---------------------------------------------------------------------------
# Serving skill wrappers
# ---------------------------------------------------------------------------

def call_serve_start(config: BenchConfig) -> dict[str, Any]:
    """Call serve_start.py and return its JSON output."""
    script = str(SERVING_SCRIPTS / "serve_start.py")
    cmd = [sys.executable, script] + config.to_serve_start_args()

    emit_progress("serve_start", f"starting service: {config.model}")
    returncode, data, stdout, stderr = _run_json_command_streaming(
        cmd,
        progress_markers=(SERVING.PROGRESS_SENTINEL,),
    )

    if not stdout.strip():
        raise RuntimeError(
            f"serve_start.py produced no output (rc={returncode}):\n"
            f"{stderr[:2000]}"
        )
    if data is None:
        raise RuntimeError(
            f"serve_start.py output is not JSON (rc={returncode}):\n"
            f"stdout: {stdout[:1000]}\nstderr: {stderr[:1000]}"
        )
    return data


def call_serve_stop(config: BenchConfig, force: bool = False) -> dict[str, Any]:
    """Call serve_stop.py and return its JSON output."""
    script = str(SERVING_SCRIPTS / "serve_stop.py")
    cmd = [sys.executable, script]
    cmd.extend(["--host", config.host])
    if config.ssh_port != 22:
        cmd.extend(["--ssh-port", str(config.ssh_port)])
    if config.user != "root":
        cmd.extend(["--user", config.user])
    if config.password:
        cmd.extend(["--password", config.password])
    if config.key_file:
        cmd.extend(["--key-file", config.key_file])
    if config.container:
        cmd.extend(["--container", config.container])
    if config.conda_env:
        cmd.extend(["--conda-env", config.conda_env])
    if config.connection_mode != "docker":
        cmd.extend(["--mode", config.connection_mode])
    if config.worker_key_file:
        cmd.extend(["--worker-key-file", config.worker_key_file])
    if config.worker_password:
        cmd.extend(["--worker-password", config.worker_password])
    if config.worker_container:
        cmd.extend(["--worker-container", config.worker_container])
    if config.worker_conda_env:
        cmd.extend(["--worker-conda-env", config.worker_conda_env])
    if force:
        cmd.append("--force")

    emit_progress("serve_stop", "stopping service")
    returncode, data, stdout, _stderr = _run_json_command_streaming(cmd)
    if not stdout.strip():
        return {"status": "failed", "returncode": returncode,
                "message": "no output from serve_stop"}
    if data is None:
        return {"status": "failed", "returncode": returncode,
                "message": stdout[:500]}
    if returncode != 0:
        data.setdefault("_returncode", returncode)
    return data


# ---------------------------------------------------------------------------
# Remote benchmark execution
# ---------------------------------------------------------------------------

def resolve_bench_target(config: BenchConfig) -> ExecutionTarget:
    """Resolve the execution target for the benchmark.

    Uses direct connection params from config.
    """
    return resolve_direct_target(
        host=config.host,
        employee_id=config.employee_id,
        port=config.ssh_port,
        user=config.user,
        password=config.password,
        key_file=config.key_file,
        container=config.container,
        conda_env=config.conda_env,
        connection_mode=config.connection_mode,
    )


def run_bench_on_remote(
    config: BenchConfig,
    base_url: str,
    served_model_name: str,
    target: ExecutionTarget,
) -> tuple[dict[str, Any], str]:
    """Run vllm bench serve on the remote container via SSH.

    Uses ``ssh_exec`` with proper auth support (password, key_file, or
    BatchMode) and automatic ``docker exec`` wrapping when a container
    name is set on the target.

    Returns:
        (parsed_json_result, raw_stdout) — raw_stdout preserves the
        complete console output (including progress bars, per-request
        summaries, and JSON) for diagnostic logging.
    """
    bench_cmd_parts = config.to_bench_serve_args(base_url, served_model_name)
    target_token = safe_token(config.host or "benchmark")
    result_filename = (
        f"result_bench_{target_token}_{now_utc().replace(':', '-')}_"
        f"{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
    )
    bench_cmd_parts.extend(["--result-filename", result_filename])

    # Wrap with conda run when conda_env is set (no container)
    if config.conda_env and not config.container:
        bench_cmd_parts = [
            "conda", "run", "-n", config.conda_env,
            "--no-capture-output",
        ] + bench_cmd_parts

    bench_cmd = " ".join(shlex.quote(str(s)) for s in bench_cmd_parts)

    # Export user-configured env vars before running bench (e.g. VLLM_VERSION)
    env_exports = "".join(
        f"export {shlex.quote(k)}={shlex.quote(v)}; "
        for k, v in config.env.items()
    )

    remote_script = (
        SERVING.ascend_env_preamble(config.cann_dir)
        + env_exports
        + f"cd /tmp && {bench_cmd} 2>&1 && cat /tmp/{result_filename}"
    )

    emit_progress("bench_run", f"running vllm bench serve on {target_token}")
    proc = ssh_exec_target(target, remote_script, check=True)

    raw_stdout: str = proc.stdout
    json_start = raw_stdout.rfind("\n{")
    if json_start == -1:
        json_start = 0 if raw_stdout.startswith("{") else -1
    else:
        json_start += 1

    if json_start == -1:
        raise RuntimeError(
            f"cannot find JSON result in bench output:\n{raw_stdout[-2000:]}"
        )

    try:
        result_data = json.loads(raw_stdout[json_start:])
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"cannot parse bench result JSON: {e}\n{raw_stdout[json_start:json_start+500]}"
        )

    return result_data, raw_stdout


def save_bench_log(
    config: BenchConfig,
    run_index: int,
    raw_stdout: str,
) -> Path:
    """Save raw benchmark console output to a ``.log`` file.

    The log file is written alongside benchmark result JSONs in the same
    run directory so the two can be correlated by timestamp + target token.

    Returns the absolute path to the written log file.
    """
    runs_dir = benchmark_runs_dir(config)
    runs_dir.mkdir(parents=True, exist_ok=True)
    target_token = safe_token(config.host or "benchmark")
    filename = (
        f"{now_utc().replace(':', '-')}_{target_token}_"
        f"{os.getpid()}_{uuid.uuid4().hex[:8]}_run{run_index + 1}.log"
    )
    log_path = runs_dir / filename
    log_path.write_text(raw_stdout, encoding="utf-8")
    emit_progress("log_saved", f"bench console log saved: {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def extract_metrics(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from vllm bench serve result JSON."""
    metrics: dict[str, Any] = {}

    for key in ("output_throughput", "mean_tpot_ms", "mean_ttft_ms",
                "median_tpot_ms", "median_ttft_ms", "acceptance_rate",
                "total_input", "total_output", "request_throughput",
                "mean_e2el_ms", "median_e2el_ms"):
        if key in raw_result:
            val = raw_result[key]
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    pass
            metrics[key] = val

    return metrics
