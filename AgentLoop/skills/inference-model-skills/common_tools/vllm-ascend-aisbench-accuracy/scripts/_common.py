#!/usr/bin/env python3
"""Shared utilities for vllm-ascend-aisbench-accuracy scripts.

SSH primitives and target resolution are imported from the serving skill's
``_common`` so we do not maintain a second copy.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
SERVING_SCRIPTS = ROOT / "skills" / "inference-model-skills" / "deployment" / "vllm-ascend-serving" / "scripts"

if str(SERVING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SERVING_SCRIPTS))

from workspace_validate import require_env_name  # noqa: E402
from workspace_output import load_serving_common, emit_progress as _emit_progress, print_json, now_utc  # noqa: E402

ACCURACY_STATE_DIR = ROOT / ".workspace-local" / "aisbench-accuracy"
PROGRESS_SENTINEL = "__WORKSPACE_ACCURACY_PROGRESS__="

# ---- Constants ----
AIS_BENCH_GIT_URL = "https://github.com/AISBench/benchmark.git"
AIS_BENCH_CLONE_DIR = "/home/vllm/benchmark"
AIS_BENCH_WORK_DIR = "/home/vllm"  # working directory for running ais_bench
DEFAULT_MATH_URL = "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip"
TEXTVQA_DATASET_ID = "vllm-ascend/textvqa_subset"
CONFIG_REL_PATH = "benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
DATASET_DIR_REL_PATH = "datasets"
TEXTVQA_IMAGE_SUBDIR = "train_images"

# ---- Lazy-load serving primitives ----
SERVING = load_serving_common(
    module_alias="workspace_accuracy_serving_common",
    serving_scripts_dir=SERVING_SCRIPTS,
)
SshEndpoint = SERVING.SshEndpoint
ssh_exec = SERVING.ssh_exec
ssh_exec_target = SERVING.ssh_exec_target
resolve_direct_target = SERVING.resolve_direct_target
ExecutionTarget = SERVING.ExecutionTarget
safe_token = SERVING.safe_token
ascend_env_preamble = SERVING.ascend_env_preamble

emit_progress = partial(_emit_progress, sentinel=PROGRESS_SENTINEL)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AccuracyConfig:
    """Assembled accuracy-test configuration ready for execution."""
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
    # ---- model / serving ----
    model: str = ""
    tp: int | None = None
    dp: int | None = None
    port: int | None = None
    serve_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # ---- network / proxy (optional) ----
    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: str | None = None
    git_insecure: bool = False
    # ---- accuracy test ----
    runs: int = 10
    dataset: str = "math500_gen_0_shot_cot_chat_prompt"
    dataset_kind: str = "auto"  # auto|math|textvqa
    batch_size: int = 1024
    temperature: float = 0.0
    seed: int = 1234
    max_out_len: int = 20480
    # ---- resolved target (populated after assemble) ----
    target: ExecutionTarget | None = field(default=None, repr=False)

    def proxy_env_dict(self) -> dict[str, str]:
        """Return proxy environment variables (both lower- and upper-case)."""
        env: dict[str, str] = {}
        if self.http_proxy:
            env["http_proxy"] = self.http_proxy
            env["HTTP_PROXY"] = self.http_proxy
        if self.https_proxy:
            env["https_proxy"] = self.https_proxy
            env["HTTPS_PROXY"] = self.https_proxy
        if self.no_proxy:
            env["no_proxy"] = self.no_proxy
            env["NO_PROXY"] = self.no_proxy
        return env

    def proxy_bash_preamble(self) -> str:
        """Bash lines exporting proxy env vars, or an empty string."""
        lines: list[str] = []
        for key, value in self.proxy_env_dict().items():
            lines.append(f'export {key}="{value}"')
        return ("\n".join(lines) + "\n") if lines else ""

    def proxy_unset_preamble(self) -> str:
        """Bash lines unsetting any configured proxy env vars, or empty string.

        Used when running the actual ais_bench accuracy test so requests go
        directly to the vLLM service on the internal network, rather than being
        routed through http_proxy/https_proxy (which are only needed during
        environment preparation: git clone / pip install / dataset download).
        """
        keys = list(self.proxy_env_dict().keys())
        if not keys:
            return ""
        return "unset " + " ".join(keys) + "\n"

    def max_model_len(self) -> int | None:
        """Return the ``--max-model-len`` value from serve_args, or None if
        not explicitly set by the caller. Handles both ``--max-model-len N``
        and ``--max-model-len=N`` forms."""
        args = self.serve_args
        for i, a in enumerate(args):
            if a == "--max-model-len" and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    return None
        for a in args:
            if a.startswith("--max-model-len="):
                try:
                    return int(a.split("=", 1)[1])
                except ValueError:
                    return None
        return None

    def _worker_head_record(self) -> dict[str, Any]:
        """Build a head record for worker target config inheritance."""
        return {
            "host": {"port": self.ssh_port, "user": self.user},
            "employee_id": self.employee_id,
            "_key_file": self.key_file,
            "_password": self.password,
            "container_name": self.container,
            "conda_env": self.conda_env,
            "_connection_mode": self.connection_mode,
        }

    def worker_targets(self) -> list[ExecutionTarget]:
        """Resolve worker ExecutionTargets for a multi-node deployment."""
        if not self.worker_hosts:
            return []
        worker_hosts = SERVING.parse_worker_hosts(self.worker_hosts)
        head_record = self._worker_head_record()
        targets: list[ExecutionTarget] = []
        for wh in worker_hosts:
            targets.append(
                SERVING.resolve_worker_target(
                    wh,
                    head_record=head_record,
                    ssh_port=self.ssh_port,
                    user=self.user,
                    key_file=self.worker_key_file,
                    password=self.worker_password,
                    container=self.worker_container,
                    conda_env=self.worker_conda_env,
                    connection_mode=self.connection_mode,
                )
            )
        return targets

    def all_targets(self) -> list[ExecutionTarget]:
        """Return [head] plus worker targets for a multi-node deployment."""
        targets: list[ExecutionTarget] = []
        if self.target is not None:
            targets.append(self.target)
        targets.extend(self.worker_targets())
        return targets

    def to_serve_start_args(self) -> list[str]:
        """Build CLI args for serve_start.py."""
        args: list[str] = ["--employee-id", self.employee_id]
        args.extend(["--host", self.host or ""])
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
        serve_args = list(self.serve_args)
        if self.dataset_kind == "textvqa":
            # vLLM refuses to load local image files (file://) unless the media
            # root is whitelisted via --allowed-local-media-path. Auto-inject it
            # for the textvqa scenario unless the caller already supplied it.
            has_media = any(
                a == "--allowed-local-media-path"
                or a.startswith("--allowed-local-media-path=")
                for a in serve_args
            )
            if not has_media:
                serve_args += ["--allowed-local-media-path", self.textvqa_media_root()]
        if serve_args:
            args.append("--")
            args.extend(serve_args)
        return args

    def textvqa_media_root(self) -> str:
        """Media root shared by preparation and the vLLM allowlist."""
        return f"{AIS_BENCH_CLONE_DIR}/ais_bench/{DATASET_DIR_REL_PATH}/textvqa"

    def summary_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "employee_id": self.employee_id,
            "model": self.model,
            "host": self.host,
            "ssh_port": self.ssh_port,
            "runs": self.runs,
            "dataset": self.dataset,
            "dataset_kind": self.dataset_kind,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if self.container:
            d["container"] = self.container
        if self.conda_env:
            d["conda_env"] = self.conda_env
        if self.cann_dir != "/usr/local/Ascend":
            d["cann_dir"] = self.cann_dir
        if self.tp is not None:
            d["tp"] = self.tp
        if self.dp is not None:
            d["dp"] = self.dp
        if self.serve_args:
            d["serve_args"] = self.serve_args
        if self.env:
            d["env"] = self.env
        if self.worker_hosts:
            d["worker_hosts"] = self.worker_hosts
            d["nnodes"] = 1 + self.worker_hosts.count(",") + (1 if self.worker_hosts else 0)
        if self.http_proxy or self.https_proxy or self.no_proxy:
            d["proxy"] = self.proxy_env_dict()
        if self.git_insecure:
            d["git_insecure"] = True
        return d


def assemble_config(args: Any, serve_args: list[str]) -> AccuracyConfig:
    """Build an AccuracyConfig from parsed CLI args."""
    connection_mode = getattr(args, "connection_mode", "docker")
    host = args.host
    if connection_mode == "local":
        if getattr(args, "worker_hosts", None):
            raise ValueError("--mode local supports single-node evaluation only")
        if args.container or args.conda_env:
            raise ValueError("--mode local does not accept --container or --conda-env")
        if args.password or args.key_file:
            raise ValueError("--mode local does not accept SSH authentication options")
        host = host or "127.0.0.1"
    elif not host:
        raise ValueError("--host is required unless --mode local is used")

    employee_id = getattr(args, "employee_id", None)
    SERVING.employee_runtime_base(employee_id)
    cfg = AccuracyConfig(
        employee_id=employee_id,
        host=host,
        ssh_port=args.ssh_port,
        user=args.user,
        password=args.password,
        key_file=args.key_file,
        container=args.container,
        conda_env=args.conda_env if not args.container else None,
        connection_mode=connection_mode,
        cann_dir=args.cann_dir,
        model=args.model,
        tp=args.tp,
        dp=args.dp,
        port=args.port,
        serve_args=serve_args,
        runs=args.runs,
        dataset=args.dataset,
        dataset_kind=getattr(args, "dataset_kind", "auto"),
        worker_hosts=getattr(args, "worker_hosts", None),
        master_port=getattr(args, "master_port", None),
        worker_key_file=getattr(args, "worker_key_file", None),
        worker_password=getattr(args, "worker_password", None),
        worker_container=getattr(args, "worker_container", None),
        worker_conda_env=getattr(args, "worker_conda_env", None),
        worker_extra_env=list(getattr(args, "worker_extra_env", None) or []),
        http_proxy=getattr(args, "http_proxy", None),
        https_proxy=getattr(args, "https_proxy", None),
        no_proxy=getattr(args, "no_proxy", None),
        git_insecure=bool(getattr(args, "git_insecure", False)),
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed,
        max_out_len=getattr(args, "max_out_len", 20480),
    )

    # Parse env overrides
    env: dict[str, str] = {}
    for item in getattr(args, "extra_env", []) or []:
        if "=" not in item:
            raise ValueError(f"bad --extra-env {item!r}, expected KEY=VALUE")
        k, v = item.split("=", 1)
        env[require_env_name(k)] = v
    cfg.env = env

    # Auto-infer dataset preparation kind from the dataset name unless the
    # caller explicitly chose one.
    if cfg.dataset_kind == "auto":
        cfg.dataset_kind = "textvqa" if "textvqa" in cfg.dataset.lower() else "math"

    # Resolve execution target
    cfg.target = resolve_direct_target(
        host=cfg.host,
        employee_id=cfg.employee_id,
        port=cfg.ssh_port,
        user=cfg.user,
        password=cfg.password,
        key_file=cfg.key_file,
        container=cfg.container,
        conda_env=cfg.conda_env,
        connection_mode=cfg.connection_mode,
    )

    return cfg


# ---------------------------------------------------------------------------
# Serving skill wrappers (shared pattern with vllm-ascend-benchmark)
# ---------------------------------------------------------------------------

def _run_json_command_streaming(
    cmd: list[str],
    *,
    progress_markers: tuple[str, ...] = (),
) -> tuple[int, dict[str, Any] | None, str, str]:
    """Run a subprocess, relay stderr, and parse stdout as JSON."""
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


def call_serve_start(config: AccuracyConfig) -> dict[str, Any]:
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
            f"serve_start.py produced no output (rc={returncode}):\n{stderr[:2000]}"
        )
    if data is None:
        raise RuntimeError(
            f"serve_start.py output is not JSON (rc={returncode}):\n"
            f"stdout: {stdout[:1000]}\nstderr: {stderr[:1000]}"
        )
    return data


def call_serve_stop(config: AccuracyConfig, force: bool = False) -> dict[str, Any]:
    """Call serve_stop.py and return its JSON output."""
    script = str(SERVING_SCRIPTS / "serve_stop.py")
    cmd = [sys.executable, script]
    cmd.extend(["--host", config.host or ""])
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
        return {"status": "failed", "returncode": returncode, "message": "no output from serve_stop"}
    if data is None:
        return {"status": "failed", "returncode": returncode, "message": stdout[:500]}
    if returncode != 0:
        data.setdefault("_returncode", returncode)
    return data


# ---------------------------------------------------------------------------
# Remote command execution
# ---------------------------------------------------------------------------

def run_remote_cmd(
    target: ExecutionTarget,
    script: str,
    *,
    check: bool = False,
) -> dict[str, Any]:
    """Execute a shell script on the remote target via SSH.

    Returns a dict with ``returncode``, ``stdout``, and ``stderr``.
    """
    try:
        proc = ssh_exec_target(target, script, check=check)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


# ---------------------------------------------------------------------------
# Local result persistence
# ---------------------------------------------------------------------------

def write_local_result(
    output_dir: Path,
    filename: str,
    data: dict[str, Any],
) -> Path:
    """Write a JSON result file to the local output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / filename
    result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path
