#!/usr/bin/env python3
"""同步本地仓库并在免密 SSH 远端启动 MindSpeed-LLM 验证脚本。"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SKILL_ROOT.parents[1]
DEFAULT_PRIVATE_CONFIG = SKILL_ROOT / "config" / "remote_config.local.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
RUN_DIRECTORY_PREFIX = "remote-validation-"
RUN_DIRECTORY_RE = re.compile(rf"^{re.escape(RUN_DIRECTORY_PREFIX)}(\d{{4,}})$")
NPU_PROCESS_TABLE_HEADER_RE = re.compile(r"^\|\s*NPU\s+Chip\s*\|\s*Process id\b.*$", re.MULTILINE)
NPU_PROCESS_ROW_RE = re.compile(r"^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|", re.MULTILINE)
NPU_IDLE_ROW_RE = re.compile(r"^\|\s*No running processes found in NPU\s+\d+\b")


class ConfigError(ValueError):
    """配置或本地输入不合法。"""


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("remote config 顶层必须是 JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} 必须是 JSON object")
    return value


def require_string(payload: dict[str, Any], key: str, field: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} 必须是非空字符串")
    if "\n" in value or "\r" in value:
        raise ConfigError(f"{field} 不能包含换行")
    return value.strip()


def validate_remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ConfigError("repository.remote_path 必须是无 .. 的绝对路径")
    if str(path) in {"/", "/home", "/root", "/tmp"}:
        raise ConfigError("repository.remote_path 过于宽泛，必须指向具体仓库目录")
    return str(path)


def validate_relative_script(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError("execution.script 必须是仓库内不含 .. 的相对路径")
    return str(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--remote-config", dest="config", default=str(DEFAULT_PRIVATE_CONFIG))
    parser.add_argument("--mode", choices=("plan", "preflight", "run"), default="run")
    parser.add_argument("--repo-root", "--local-repo", dest="local_repo")
    parser.add_argument("--task", help="总控传入的 task JSON，仅记录来源")
    parser.add_argument("--host")
    parser.add_argument("--user")
    parser.add_argument("--port", type=int)
    parser.add_argument("--remote-repo")
    parser.add_argument("--script")
    parser.add_argument("--conda-env")
    parser.add_argument("--cann-env-script", action="append", dest="cann_env_scripts")
    parser.add_argument("--output-dir", "--artifact-root", dest="output_dir")
    parser.add_argument("--run-id", help="仅用于 run 模式；指定尚未使用的正整数编号")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--skip-sync", action="store_true")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(
            f"remote config 不存在：{config_path}；请复制 config/remote_config.example.json 为私有 *.local.json"
        )
    raw = load_json(config_path)
    if raw.get("schema_version", 1) != 1:
        raise ConfigError("仅支持 remote config schema_version: 1")
    connection = require_mapping(raw, "connection")
    repository = require_mapping(raw, "repository")
    execution = require_mapping(raw, "execution")
    sync = raw.get("sync", {})
    if not isinstance(sync, dict):
        raise ConfigError("sync 必须是 JSON object")
    output = raw.get("output", {})
    if not isinstance(output, dict):
        raise ConfigError("output 必须是 JSON object")

    host = args.host or require_string(connection, "host", "connection.host")
    user = args.user or require_string(connection, "user", "connection.user")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", host):
        raise ConfigError("connection.host 包含不支持的字符")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", user):
        raise ConfigError("connection.user 包含不支持的字符")
    port = args.port if args.port is not None else connection.get("port", 22)
    connect_timeout = connection.get("connect_timeout_seconds", 10)
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError("connection.port 必须在 1..65535")
    if not isinstance(connect_timeout, int) or connect_timeout < 1:
        raise ConfigError("connection.connect_timeout_seconds 必须为正整数")

    local_value = args.local_repo or require_string(repository, "local_path", "repository.local_path")
    local_repo = Path(local_value).expanduser().resolve()
    if not local_repo.is_dir():
        raise ConfigError(f"本地仓库不存在：{local_repo}")
    remote_repo = validate_remote_path(
        args.remote_repo or require_string(repository, "remote_path", "repository.remote_path")
    )
    script = validate_relative_script(args.script or require_string(execution, "script", "execution.script"))
    if not (local_repo / script).is_file():
        raise ConfigError(f"本地训练脚本不存在：{local_repo / script}")

    conda_env = args.conda_env or require_string(execution, "conda_env", "execution.conda_env")
    if "\n" in conda_env or "\r" in conda_env:
        raise ConfigError("execution.conda_env 不能包含换行")
    cann_scripts = args.cann_env_scripts if args.cann_env_scripts is not None else execution.get("cann_env_scripts")
    if not isinstance(cann_scripts, list) or not cann_scripts:
        raise ConfigError("execution.cann_env_scripts 必须是非空数组")
    for item in cann_scripts:
        if not isinstance(item, str) or not PurePosixPath(item).is_absolute() or "\n" in item:
            raise ConfigError("每个 CANN 环境脚本都必须是无换行的远端绝对路径")

    script_args = execution.get("script_args", [])
    if not isinstance(script_args, list) or not all(isinstance(item, str) for item in script_args):
        raise ConfigError("execution.script_args 必须是字符串数组")
    timeout = args.timeout_seconds if args.timeout_seconds is not None else execution.get("timeout_seconds", 7200)
    if not isinstance(timeout, int) or timeout < 1:
        raise ConfigError("execution.timeout_seconds 必须为正整数")
    excludes = sync.get("exclude", [])
    if not isinstance(excludes, list) or not all(
        isinstance(item, str) and "\n" not in item and "\r" not in item for item in excludes
    ):
        raise ConfigError("sync.exclude 必须是字符串数组")
    if bool(sync.get("delete", False)) and len(PurePosixPath(remote_repo).parts) < 4:
        raise ConfigError("启用 sync.delete 时，repository.remote_path 必须至少包含三级具体目录")

    output_value = args.output_dir or output.get("local_path") or str(DEFAULT_OUTPUT_ROOT)
    if not isinstance(output_value, str) or not output_value.strip():
        raise ConfigError("output.local_path 必须是非空字符串")
    if "\n" in output_value or "\r" in output_value:
        raise ConfigError("output.local_path 不能包含换行")
    output_root = Path(output_value).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    if output_root == Path(output_root.anchor):
        raise ConfigError("output.local_path 不能指向文件系统根目录")

    task_path = None
    if args.task:
        task = Path(args.task).expanduser().resolve()
        if not task.is_file():
            raise ConfigError(f"task JSON 不存在：{task}")
        task_path = str(task)

    return {
        "config_path": str(config_path),
        "task_path": task_path,
        "connection": {
            "host": host,
            "user": user,
            "port": port,
            "connect_timeout_seconds": connect_timeout,
        },
        "repository": {"local_path": str(local_repo), "remote_path": remote_repo},
        "execution": {
            "script": script,
            "script_args": script_args,
            "conda_env": conda_env,
            "cann_env_scripts": cann_scripts,
            "timeout_seconds": timeout,
            "capture_npu_smi": bool(execution.get("capture_npu_smi", True)),
        },
        "sync": {"delete": bool(sync.get("delete", False)), "exclude": excludes},
        "output": {"local_path": str(output_root)},
    }


def ssh_base(config: dict[str, Any]) -> list[str]:
    connection = config["connection"]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connection['connect_timeout_seconds']}",
        "-p",
        str(connection["port"]),
        f"{connection['user']}@{connection['host']}",
    ]


def remote_command(config: dict[str, Any], body: str) -> list[str]:
    return [*ssh_base(config), f"bash -lc {shlex.quote(body)}"]


def environment_prefix(config: dict[str, Any]) -> list[str]:
    execution = config["execution"]
    lines = [
        "set -eo pipefail",
        "CONDA_BASE=\"$(bash -ic 'conda info --base' 2>/dev/null | tail -n 1 || true)\"",
        'test -n "$CONDA_BASE" || { printf "无法通过远端交互式 Bash 解析 conda\\n" >&2; exit 127; }',
        'source "$CONDA_BASE/etc/profile.d/conda.sh"',
        f"conda activate {shlex.quote(execution['conda_env'])}",
    ]
    lines.extend(f"source {shlex.quote(path)}" for path in execution["cann_env_scripts"])
    lines.append(f"cd {shlex.quote(config['repository']['remote_path'])}")
    return lines


def build_preflight(config: dict[str, Any]) -> list[str]:
    script = config["execution"]["script"]
    lines = environment_prefix(config)
    lines.extend(
        [
            f"test -f {shlex.quote(script)}",
            'printf "remote_repo=%s\\n" "$PWD"',
            'printf "conda_env=%s\\n" "$CONDA_DEFAULT_ENV"',
            f"printf 'training_script=%s\\n' {shlex.quote(script)}",
        ]
    )
    return remote_command(config, "; ".join(lines))


def build_training(config: dict[str, Any]) -> list[str]:
    execution = config["execution"]
    script_command = ["bash", "-e", "-o", "pipefail", execution["script"], *execution["script_args"]]
    quoted_script = " ".join(shlex.quote(item) for item in script_command)
    lines = environment_prefix(config)
    lines.append(
        f"exec timeout --signal=TERM --kill-after=30s {execution['timeout_seconds']}s {quoted_script}"
    )
    return remote_command(config, "; ".join(lines))


def build_sync(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    remote_repo = config["repository"]["remote_path"]
    mkdir = remote_command(config, f"set -e; mkdir -p -- {shlex.quote(remote_repo)}")
    connection = config["connection"]
    ssh_transport = (
        f"ssh -o BatchMode=yes -o ConnectTimeout={connection['connect_timeout_seconds']} "
        f"-p {connection['port']}"
    )
    command = ["rsync", "-az", "--stats", "-e", ssh_transport]
    if config["sync"]["delete"]:
        command.append("--delete")
    for pattern in config["sync"]["exclude"]:
        command.extend(["--exclude", pattern])
    command.extend(
        [
            f"{config['repository']['local_path']}/",
            f"{connection['user']}@{connection['host']}:{remote_repo}/",
        ]
    )
    return mkdir, command


def command_result(command: list[str], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-8000:],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": 124,
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "timed_out": True,
        }


def inspect_npu_usage(output: str) -> dict[str, Any]:
    """解析 npu-smi info 的进程表；无法确认空闲时拒绝继续训练。"""
    header = NPU_PROCESS_TABLE_HEADER_RE.search(output)
    if header is None:
        raise ConfigError("npu-smi info 输出中缺少进程表")
    process_table = output[header.end() :]
    process_ids = sorted({int(process_id) for process_id in NPU_PROCESS_ROW_RE.findall(process_table)})
    if process_ids:
        return {"idle": False, "process_ids": process_ids}
    table_rows = [line for line in process_table.splitlines() if line.startswith("|")]
    if table_rows and all(NPU_IDLE_ROW_RE.match(line) for line in table_rows):
        return {"idle": True, "process_ids": []}
    raise ConfigError("无法从 npu-smi info 输出确认 NPU 是否空闲")


def stream_training(command: list[str], log_path: Path, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    tail: deque[str] = deque(maxlen=200)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
                tail.append(line)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exit_code = 124
        reader.join(timeout=10)
    return {
        "command": command,
        "exit_code": exit_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "timed_out": timed_out or exit_code == 124,
        "log_path": str(log_path),
        "output_tail": "".join(tail)[-12000:],
    }


def build_plan(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mkdir, sync = build_sync(config)
    return {
        "connection": config["connection"],
        "repository": config["repository"],
        "execution": config["execution"],
        "sync": config["sync"],
        "output": config["output"],
        "skip_sync": args.skip_sync,
        "commands": {
            "create_remote_repo": mkdir,
            "sync": sync,
            "preflight": build_preflight(config),
            "resource_check": remote_command(config, "npu-smi info"),
            "training": build_training(config),
        },
    }


def allocate_run_directory(output_root: Path, mode: str, requested_run_id: str | None) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    if mode != "run":
        if requested_run_id is not None:
            raise ConfigError("--run-id 仅可用于 run 模式")
        run_id = f".{mode}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    if requested_run_id is not None:
        if not re.fullmatch(r"[1-9]\d*", requested_run_id):
            raise ConfigError("run 模式的 --run-id 必须是正整数")
        run_id = f"{RUN_DIRECTORY_PREFIX}{int(requested_run_id):04d}"
        run_dir = output_root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise ConfigError(f"输出目录已存在：{run_dir}") from exc
        return run_id, run_dir

    existing_ids = []
    for path in output_root.iterdir():
        match = RUN_DIRECTORY_RE.fullmatch(path.name)
        if path.is_dir() and match is not None:
            existing_ids.append(int(match.group(1)))
    next_id = max(existing_ids, default=0) + 1
    while True:
        run_id = f"{RUN_DIRECTORY_PREFIX}{next_id:04d}"
        run_dir = output_root / run_id
        try:
            run_dir.mkdir()
            return run_id, run_dir
        except FileExistsError:
            next_id += 1


def initial_report(mode: str, run_id: str, run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "validation_report.json"
    return {
        "schema_version": 1,
        "workflow": "mindspeed-llm-remote-validation",
        "mode": mode,
        "run_id": run_id,
        "passed": False,
        "stage": "configuration",
        "errors": [],
        "plan": {},
        "artifacts": {
            "run_dir": str(run_dir),
            "validation_report": str(report_path),
            "training_log": str(run_dir / "training.log"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if shutil.which("ssh") is None or shutil.which("rsync") is None:
            raise ConfigError("本地必须安装 ssh 和 rsync")
        config = resolve_config(args)
        run_id, run_dir = allocate_run_directory(
            Path(config["output"]["local_path"]), args.mode, args.run_id
        )
        report = initial_report(args.mode, run_id, run_dir)
        report_path = run_dir / "validation_report.json"
        report["plan"] = build_plan(config, args)
    except (ConfigError, json.JSONDecodeError, OSError) as exc:
        error_dir = DEFAULT_OUTPUT_ROOT / ".configuration_error"
        error_dir.mkdir(parents=True, exist_ok=True)
        report = initial_report(args.mode, ".configuration_error", error_dir)
        report_path = error_dir / "validation_report.json"
        report["errors"] = [str(exc)]
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 2

    if args.mode == "plan":
        report["stage"] = "planned"
        report["passed"] = True
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    if args.mode == "run" and not args.skip_sync:
        report["stage"] = "sync"
        mkdir_command, sync_command = build_sync(config)
        mkdir_result = command_result(mkdir_command, 60)
        report["create_remote_repo_result"] = mkdir_result
        if mkdir_result["exit_code"] != 0:
            report["errors"] = ["创建远端仓库目录失败"]
            write_json(report_path, report)
            print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
            return 3
        sync_result = command_result(sync_command, max(300, config["execution"]["timeout_seconds"]))
        report["sync_result"] = sync_result
        if sync_result["exit_code"] != 0:
            report["errors"] = ["本地仓库同步失败"]
            write_json(report_path, report)
            print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
            return 3

    report["stage"] = "preflight"
    preflight_result = command_result(build_preflight(config), 120)
    report["preflight_result"] = preflight_result
    if preflight_result["exit_code"] != 0:
        report["errors"] = ["远端 SSH、Conda、CANN、仓库或训练脚本预检失败"]
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 3

    if args.mode == "preflight":
        report["stage"] = "preflight_completed"
        report["passed"] = True
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    report["stage"] = "resource_check"
    resource_snapshot = command_result(remote_command(config, "npu-smi info"), 60)
    report["resource_snapshot"] = resource_snapshot
    if resource_snapshot["exit_code"] != 0:
        report["errors"] = ["npu-smi info 资源检查失败，未启动训练"]
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 3
    try:
        usage = inspect_npu_usage(resource_snapshot["stdout"])
    except ConfigError as exc:
        report["errors"] = [f"{exc}，未启动训练"]
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 3
    resource_snapshot.update(usage)
    if not usage["idle"]:
        report["stage"] = "resource_busy"
        process_ids = ", ".join(str(process_id) for process_id in usage["process_ids"])
        report["errors"] = [f"检测到 NPU 正被进程占用（process ids: {process_ids}），未启动训练"]
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 5

    report["stage"] = "training"
    training_result = stream_training(
        build_training(config), run_dir / "training.log", config["execution"]["timeout_seconds"]
    )
    report["training_result"] = training_result
    if training_result["exit_code"] == 0:
        report["stage"] = "completed"
        report["passed"] = True
        exit_code = 0
    elif training_result["exit_code"] == 124:
        report["stage"] = "timed_out"
        report["errors"] = ["远端训练超时"]
        exit_code = 124
    else:
        report["stage"] = "training_failed"
        report["errors"] = [f"远端训练脚本失败，exit code: {training_result['exit_code']}"]
        exit_code = 4
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
