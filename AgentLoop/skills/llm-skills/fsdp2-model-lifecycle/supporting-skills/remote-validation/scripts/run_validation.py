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
RESERVED_ENVIRONMENT_KEYS = {
    "REMOTE_VALIDATION_OUTPUT_DIR",
    "REMOTE_VALIDATION_REPO",
    "REMOTE_VALIDATION_RUN_ID",
    "REPO_REVISION",
}


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
    if not re.fullmatch(r"/[A-Za-z0-9._/@+\-]+", value):
        raise ConfigError("repository.remote_path 只能包含安全的 POSIX 路径字符")
    if str(path) in {"/", "/home", "/root", "/tmp"}:
        raise ConfigError("repository.remote_path 过于宽泛，必须指向具体仓库目录")
    return str(path)


def validate_remote_subdir(value: str, field: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise ConfigError(f"{field} 必须是仓库内不含 .. 的非空相对路径")
    if not re.fullmatch(r"[A-Za-z0-9._/@+\-]+", value):
        raise ConfigError(f"{field} 只能包含安全的 POSIX 路径字符")
    return str(path)


def validate_relative_script(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError("execution.script 必须是仓库内不含 .. 的相对路径")
    return str(path)


def repository_revision(local_repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(local_repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        return {"base_commit": None, "dirty": None, "revision": "unversioned"}
    status = subprocess.run(
        ["git", "-C", str(local_repo), "status", "--porcelain=v1"],
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise ConfigError(f"无法读取本地仓库状态：{local_repo}")
    base_commit = commit.stdout.strip()
    dirty = bool(status.stdout.strip())
    return {
        "base_commit": base_commit,
        "dirty": dirty,
        "revision": f"{base_commit}-dirty" if dirty else base_commit,
    }


def render_remote_template(value: str, config: dict[str, Any], run_id: str) -> str:
    try:
        return value.format(
            remote_repo=config["repository"]["remote_path"],
            run_id=run_id,
        )
    except (KeyError, ValueError) as exc:
        raise ConfigError(
            f"远端模板只允许使用 {{remote_repo}} 和 {{run_id}}：{value!r}"
        ) from exc


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
        raise ConfigError(f"本地执行脚本不存在：{local_repo / script}")

    task_type = execution.get("task_type", "training")
    if not isinstance(task_type, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", task_type):
        raise ConfigError("execution.task_type 必须是小写字母开头的 kebab-case 字符串")
    environment = execution.get("environment", {})
    if not isinstance(environment, dict):
        raise ConfigError("execution.environment 必须是 JSON object")
    normalized_environment: dict[str, str] = {}
    for key, value in environment.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise ConfigError("execution.environment 的 key 必须是大写 shell 环境变量名")
        if not isinstance(value, str) or "\n" in value or "\r" in value:
            raise ConfigError(f"execution.environment.{key} 必须是无换行字符串")
        if key in RESERVED_ENVIRONMENT_KEYS:
            raise ConfigError(f"execution.environment.{key} 是远端控制器保留变量")
        normalized_environment[key] = value

    required_remote_paths = execution.get("required_remote_paths", [])
    if not isinstance(required_remote_paths, list):
        raise ConfigError("execution.required_remote_paths 必须是数组")
    normalized_required_paths = []
    for index, item in enumerate(required_remote_paths):
        if not isinstance(item, dict):
            raise ConfigError(f"execution.required_remote_paths[{index}] 必须是 JSON object")
        path = item.get("path")
        kind = item.get("kind", "exists")
        if not isinstance(path, str) or not path or "\n" in path or "\r" in path:
            raise ConfigError(f"execution.required_remote_paths[{index}].path 必须是无换行字符串")
        if kind not in {"exists", "file", "directory", "executable"}:
            raise ConfigError(
                f"execution.required_remote_paths[{index}].kind 必须是 exists/file/directory/executable"
            )
        normalized_required_paths.append({"path": path, "kind": kind})

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
    silence_timeout = execution.get("silence_timeout_seconds", 900)
    if not isinstance(silence_timeout, int) or silence_timeout < 0:
        raise ConfigError("execution.silence_timeout_seconds 必须为非负整数（0 表示关闭静默看门狗）")
    excludes = sync.get("exclude", [])
    if not isinstance(excludes, list) or not all(
        isinstance(item, str) and "\n" not in item and "\r" not in item for item in excludes
    ):
        raise ConfigError("sync.exclude 必须是字符串数组")
    if bool(sync.get("delete", False)) and len(PurePosixPath(remote_repo).parts) < 4:
        raise ConfigError("启用 sync.delete 时，repository.remote_path 必须至少包含三级具体目录")

    runtime_bundles = sync.get("runtime_bundles", [])
    if not isinstance(runtime_bundles, list):
        raise ConfigError("sync.runtime_bundles 必须是数组")
    normalized_bundles = []
    seen_remote_subdirs: list[PurePosixPath] = []
    for index, item in enumerate(runtime_bundles):
        if not isinstance(item, dict):
            raise ConfigError(f"sync.runtime_bundles[{index}] 必须是 JSON object")
        local_value = item.get("local_path")
        if not isinstance(local_value, str) or not local_value.strip():
            raise ConfigError(f"sync.runtime_bundles[{index}].local_path 必须是非空字符串")
        local_path = Path(local_value).expanduser()
        if not local_path.is_absolute():
            local_path = config_path.parent / local_path
        local_path = local_path.resolve()
        if not local_path.is_dir():
            raise ConfigError(f"runtime bundle 本地目录不存在：{local_path}")
        remote_subdir = validate_remote_subdir(
            str(item.get("remote_subdir", "")),
            f"sync.runtime_bundles[{index}].remote_subdir",
        )
        remote_subdir_path = PurePosixPath(remote_subdir)
        if any(
            remote_subdir_path == existing
            or remote_subdir_path in existing.parents
            or existing in remote_subdir_path.parents
            for existing in seen_remote_subdirs
        ):
            raise ConfigError(f"runtime bundle 远端目录重复或嵌套：{remote_subdir}")
        seen_remote_subdirs.append(remote_subdir_path)
        bundle_excludes = item.get("exclude", ["__pycache__", "*.pyc"])
        if not isinstance(bundle_excludes, list) or not all(
            isinstance(pattern, str) and "\n" not in pattern and "\r" not in pattern
            for pattern in bundle_excludes
        ):
            raise ConfigError(f"sync.runtime_bundles[{index}].exclude 必须是字符串数组")
        bundle_includes = item.get("include")
        if bundle_includes is not None and (
            not isinstance(bundle_includes, list)
            or not bundle_includes
            or not all(
                isinstance(pattern, str)
                and pattern
                and "\n" not in pattern
                and "\r" not in pattern
                for pattern in bundle_includes
            )
        ):
            raise ConfigError(
                f"sync.runtime_bundles[{index}].include 必须是非空字符串数组"
            )
        normalized_bundles.append(
            {
                "local_path": str(local_path),
                "remote_subdir": remote_subdir,
                "delete": bool(item.get("delete", True)),
                "include": bundle_includes,
                "exclude": bundle_excludes,
            }
        )

    artifacts = raw.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ConfigError("artifacts 必须是 JSON object")
    artifact_remote_path = artifacts.get("remote_path")
    if artifact_remote_path is not None and (
        not isinstance(artifact_remote_path, str)
        or not artifact_remote_path.strip()
        or "\n" in artifact_remote_path
        or "\r" in artifact_remote_path
    ):
        raise ConfigError("artifacts.remote_path 必须是无换行的非空字符串")
    artifact_includes = artifacts.get("include", ["*.json", "*.md", "*.log"])
    artifact_excludes = artifacts.get("exclude", ["*.pt"])
    for field, patterns in (("include", artifact_includes), ("exclude", artifact_excludes)):
        if not isinstance(patterns, list) or not all(
            isinstance(pattern, str) and pattern and "\n" not in pattern and "\r" not in pattern
            for pattern in patterns
        ):
            raise ConfigError(f"artifacts.{field} 必须是字符串数组")
    if artifact_remote_path is not None and not artifact_includes:
        raise ConfigError("配置 artifacts.remote_path 时，artifacts.include 不能为空")

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

    revision = repository_revision(local_repo)
    resolved = {
        "config_path": str(config_path),
        "task_path": task_path,
        "connection": {
            "host": host,
            "user": user,
            "port": port,
            "connect_timeout_seconds": connect_timeout,
        },
        "repository": {
            "local_path": str(local_repo),
            "remote_path": remote_repo,
            **revision,
        },
        "execution": {
            "task_type": task_type,
            "script": script,
            "script_args": script_args,
            "environment": normalized_environment,
            "required_remote_paths": normalized_required_paths,
            "conda_env": conda_env,
            "cann_env_scripts": cann_scripts,
            "timeout_seconds": timeout,
            "silence_timeout_seconds": silence_timeout,
            "capture_npu_smi": bool(execution.get("capture_npu_smi", True)),
        },
        "sync": {
            "delete": bool(sync.get("delete", False)),
            "exclude": excludes,
            "runtime_bundles": normalized_bundles,
        },
        "output": {"local_path": str(output_root)},
        "artifacts": {
            "remote_path": artifact_remote_path,
            "include": artifact_includes,
            "exclude": artifact_excludes,
            "pull_on_failure": bool(artifacts.get("pull_on_failure", True)),
        },
    }
    for value in normalized_environment.values():
        render_remote_template(value, resolved, "preview")
    for item in normalized_required_paths:
        render_remote_template(item["path"], resolved, "preview")
    if artifact_remote_path is not None:
        rendered_artifacts = render_remote_template(artifact_remote_path, resolved, "preview")
        artifact_path = PurePosixPath(rendered_artifacts)
        remote_root = PurePosixPath(remote_repo)
        if not artifact_path.is_absolute() or remote_root not in artifact_path.parents:
            raise ConfigError("artifacts.remote_path 渲染后必须位于 repository.remote_path 内部")
        if not re.fullmatch(r"/[A-Za-z0-9._/@+\-]+", rendered_artifacts):
            raise ConfigError("artifacts.remote_path 渲染后只能包含安全的 POSIX 路径字符")
    return resolved


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


def execution_environment(config: dict[str, Any], run_id: str) -> dict[str, str]:
    values = {
        key: render_remote_template(value, config, run_id)
        for key, value in config["execution"]["environment"].items()
    }
    values["REMOTE_VALIDATION_RUN_ID"] = run_id
    values["REMOTE_VALIDATION_REPO"] = config["repository"]["remote_path"]
    values["REPO_REVISION"] = config["repository"]["revision"]
    if config["artifacts"]["remote_path"] is not None:
        values["REMOTE_VALIDATION_OUTPUT_DIR"] = render_remote_template(
            config["artifacts"]["remote_path"], config, run_id
        )
    return values


def environment_exports(config: dict[str, Any], run_id: str) -> list[str]:
    return [
        f"export {key}={shlex.quote(value)}"
        for key, value in sorted(execution_environment(config, run_id).items())
    ]


def required_path_check(path: str, kind: str) -> str:
    operator = {
        "exists": "-e",
        "file": "-f",
        "directory": "-d",
        "executable": "-x",
    }[kind]
    return f"test {operator} {shlex.quote(path)}"


def build_preflight(config: dict[str, Any], run_id: str = ".preflight") -> list[str]:
    script = config["execution"]["script"]
    lines = environment_prefix(config)
    lines.extend(environment_exports(config, run_id))
    lines.extend(
        [
            f"test -f {shlex.quote(script)}",
            'printf "remote_repo=%s\\n" "$PWD"',
            'printf "conda_env=%s\\n" "$CONDA_DEFAULT_ENV"',
            f"printf 'task_type=%s\\n' {shlex.quote(config['execution']['task_type'])}",
            f"printf 'execution_script=%s\\n' {shlex.quote(script)}",
        ]
    )
    for bundle in config["sync"]["runtime_bundles"]:
        remote_path = str(
            PurePosixPath(config["repository"]["remote_path"]) / bundle["remote_subdir"]
        )
        lines.append(required_path_check(remote_path, "directory"))
    for item in config["execution"]["required_remote_paths"]:
        rendered = render_remote_template(item["path"], config, run_id)
        lines.append(required_path_check(rendered, item["kind"]))
    return remote_command(config, "; ".join(lines))


def build_training(config: dict[str, Any], run_id: str = ".run") -> list[str]:
    execution = config["execution"]
    script_command = ["bash", "-e", "-o", "pipefail", execution["script"], *execution["script_args"]]
    quoted_script = " ".join(shlex.quote(item) for item in script_command)
    lines = environment_prefix(config)
    lines.extend(environment_exports(config, run_id))
    lines.append(
        f"exec timeout --signal=TERM --kill-after=30s {execution['timeout_seconds']}s {quoted_script}"
    )
    return remote_command(config, "; ".join(lines))


def build_artifact_dir_check(config: dict[str, Any], run_id: str) -> list[str] | None:
    """precision-alignment 的远端产物目录必须全新：旧目录残留会让远端 run_alignment
    报 FileExistsError 白跑一轮，更糟的是可能读到旧目录的过期数据。启动前快速失败。"""
    if config["execution"]["task_type"] != "precision-alignment":
        return None
    artifacts = config.get("artifacts") or {}
    remote_path = artifacts.get("remote_path")
    if not remote_path:
        return None
    rendered = render_remote_template(remote_path, config, run_id)
    return remote_command(
        config,
        f"if [ -d {shlex.quote(rendered)} ] && [ -n \"$(ls -A {shlex.quote(rendered)} 2>/dev/null)\" ]; then "
        f"printf '远端产物目录已存在且非空：%s\\n' {shlex.quote(rendered)} >&2; exit 1; fi",
    )


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
    for pattern in [
        ".git",
        ".git/**",
        ".validation-runtime/",
        ".validation-artifacts/",
        "__pycache__",
        "*.pyc",
        *config["sync"]["exclude"],
    ]:
        command.extend(["--exclude", pattern])
    command.extend(
        [
            f"{config['repository']['local_path']}/",
            f"{connection['user']}@{connection['host']}:{remote_repo}/",
        ]
    )
    return mkdir, command


def build_runtime_syncs(config: dict[str, Any]) -> list[dict[str, Any]]:
    connection = config["connection"]
    remote_repo = config["repository"]["remote_path"]
    ssh_transport = (
        f"ssh -o BatchMode=yes -o ConnectTimeout={connection['connect_timeout_seconds']} "
        f"-p {connection['port']}"
    )
    result = []
    for bundle in config["sync"]["runtime_bundles"]:
        remote_path = str(PurePosixPath(remote_repo) / bundle["remote_subdir"])
        mkdir = remote_command(config, f"set -e; mkdir -p -- {shlex.quote(remote_path)}")
        command = ["rsync", "-az", "--stats", "-e", ssh_transport]
        if bundle["delete"]:
            command.append("--delete")
            if bundle["include"] is not None:
                command.append("--delete-excluded")
        for pattern in bundle["exclude"]:
            command.extend(["--exclude", pattern])
        if bundle["include"] is not None:
            for pattern in bundle["include"]:
                command.extend(["--include", pattern])
            command.extend(["--exclude", "*"])
        command.extend(
            [
                f"{bundle['local_path']}/",
                f"{connection['user']}@{connection['host']}:{remote_path}/",
            ]
        )
        result.append(
            {
                "local_path": bundle["local_path"],
                "remote_path": remote_path,
                "create_remote_dir": mkdir,
                "sync": command,
            }
        )
    return result


def build_artifact_sync(config: dict[str, Any], run_id: str, local_dir: Path) -> list[str] | None:
    remote_template = config["artifacts"]["remote_path"]
    if remote_template is None:
        return None
    remote_path = render_remote_template(remote_template, config, run_id)
    connection = config["connection"]
    ssh_transport = (
        f"ssh -o BatchMode=yes -o ConnectTimeout={connection['connect_timeout_seconds']} "
        f"-p {connection['port']}"
    )
    command = ["rsync", "-az", "--stats", "--prune-empty-dirs", "-e", ssh_transport]
    command.append("--include=*/")
    for pattern in config["artifacts"]["exclude"]:
        command.extend(["--exclude", pattern])
    for pattern in config["artifacts"]["include"]:
        command.extend(["--include", pattern])
    command.extend(["--exclude", "*"])
    command.extend(
        [
            f"{connection['user']}@{connection['host']}:{remote_path}/",
            f"{local_dir}/",
        ]
    )
    return command


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


def stream_training(
    command: list[str], log_path: Path, timeout: int, silence_timeout: int = 900
) -> dict[str, Any]:
    """流式执行远端 launcher。

    两道保险：
    - 硬超时 timeout（远端 launcher 自身也有 timeout 包裹）。
    - 静默看门狗 silence_timeout：远端进程死亡但 ssh 通道保持打开时，输出会永久静默，
      超过该时长无任何新输出即判定挂死并终止，返回 exit_code=125（stalled）。
    """
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

        last_activity = [time.monotonic()]

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
                tail.append(line)
                last_activity[0] = time.monotonic()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        timed_out = False
        stalled = False
        deadline = started + timeout + 60
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                break
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                break
            if silence_timeout > 0 and now - last_activity[0] >= silence_timeout:
                stalled = True
                break
            time.sleep(2)
        if timed_out or stalled:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exit_code = 124 if timed_out else 125
        reader.join(timeout=10)
    return {
        "command": command,
        "exit_code": exit_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "timed_out": timed_out or exit_code == 124,
        "stalled": stalled,
        "log_path": str(log_path),
        "output_tail": "".join(tail)[-12000:],
    }


def build_plan(config: dict[str, Any], args: argparse.Namespace, run_id: str = ".plan") -> dict[str, Any]:
    mkdir, sync = build_sync(config)
    execution_command = build_training(config, run_id)
    commands = {
        "create_remote_repo": mkdir,
        "sync": sync,
        "runtime_syncs": build_runtime_syncs(config),
        "preflight": build_preflight(config, run_id),
        "resource_check": remote_command(config, "npu-smi info"),
        "execution": execution_command,
    }
    if config["execution"]["task_type"] == "training":
        commands["training"] = execution_command
    return {
        "connection": config["connection"],
        "repository": config["repository"],
        "execution": config["execution"],
        "sync": config["sync"],
        "output": config["output"],
        "artifacts": config["artifacts"],
        "skip_sync": args.skip_sync,
        "commands": commands,
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


def initial_report(
    mode: str,
    run_id: str,
    run_dir: Path,
    task_type: str = "unknown",
) -> dict[str, Any]:
    report_path = run_dir / "validation_report.json"
    execution_log = run_dir / ("training.log" if task_type == "training" else "execution.log")
    artifacts = {
        "run_dir": str(run_dir),
        "validation_report": str(report_path),
        "execution_log": str(execution_log),
        "remote_artifacts": str(run_dir / "remote_artifacts"),
    }
    if task_type == "training":
        artifacts["training_log"] = str(execution_log)
    return {
        "schema_version": 1,
        "workflow": "mindspeed-llm-remote-validation",
        "task_type": task_type,
        "mode": mode,
        "run_id": run_id,
        "passed": False,
        "stage": "configuration",
        "errors": [],
        "plan": {},
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if shutil.which("ssh") is None or shutil.which("rsync") is None:
            raise ConfigError("本地必须安装 ssh 和 rsync")
        config = resolve_config(args)
        run_id, run_dir = allocate_run_directory(
            Path(config["output"]["local_path"]), args.mode, args.run_id
        )
        report = initial_report(
            args.mode,
            run_id,
            run_dir,
            config["execution"]["task_type"],
        )
        report_path = run_dir / "validation_report.json"
        report["plan"] = build_plan(config, args, run_id)
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
        runtime_results = []
        for bundle in build_runtime_syncs(config):
            mkdir_result = command_result(bundle["create_remote_dir"], 60)
            bundle_result = {
                "local_path": bundle["local_path"],
                "remote_path": bundle["remote_path"],
                "create_remote_dir_result": mkdir_result,
            }
            if mkdir_result["exit_code"] == 0:
                bundle_result["sync_result"] = command_result(
                    bundle["sync"],
                    max(300, config["execution"]["timeout_seconds"]),
                )
            runtime_results.append(bundle_result)
            if mkdir_result["exit_code"] != 0 or bundle_result.get("sync_result", {}).get(
                "exit_code"
            ) != 0:
                report["runtime_sync_results"] = runtime_results
                report["errors"] = [f"runtime bundle 同步失败：{bundle['local_path']}"]
                write_json(report_path, report)
                print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
                return 3
        report["runtime_sync_results"] = runtime_results

    report["stage"] = "preflight"
    preflight_result = command_result(build_preflight(config, run_id), 120)
    report["preflight_result"] = preflight_result
    if preflight_result["exit_code"] != 0:
        report["errors"] = ["远端 SSH、Conda、CANN、仓库、runtime 或执行脚本预检失败"]
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
        report["errors"] = ["npu-smi info 资源检查失败，未启动远端任务"]
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 3
    try:
        usage = inspect_npu_usage(resource_snapshot["stdout"])
    except ConfigError as exc:
        report["errors"] = [f"{exc}，未启动远端任务"]
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 3
    resource_snapshot.update(usage)
    if not usage["idle"]:
        report["stage"] = "resource_busy"
        process_ids = ", ".join(str(process_id) for process_id in usage["process_ids"])
        report["errors"] = [f"检测到 NPU 正被进程占用（process ids: {process_ids}），未启动远端任务"]
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 5

    report["stage"] = "execution"
    artifact_dir_command = build_artifact_dir_check(config, run_id)
    if artifact_dir_command is not None:
        artifact_dir_result = command_result(artifact_dir_command, 60)
        report["artifact_dir_check"] = artifact_dir_result
        if artifact_dir_result["exit_code"] != 0:
            report["stage"] = "artifact_dir_conflict"
            report["errors"] = [
                "远端产物目录已存在且非空（见 artifact_dir_check.stderr）；"
                "请先清理或归档远端旧产物，避免读到过期数据"
            ]
            write_json(report_path, report)
            print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
            return 3

    execution_log = (
        run_dir / "training.log"
        if config["execution"]["task_type"] == "training"
        else run_dir / "execution.log"
    )
    report["artifacts"]["execution_log"] = str(execution_log)
    execution_result = stream_training(
        build_training(config, run_id),
        execution_log,
        config["execution"]["timeout_seconds"],
        config["execution"].get("silence_timeout_seconds", 900),
    )
    report["execution_result"] = execution_result
    if config["execution"]["task_type"] == "training":
        report["training_result"] = execution_result
        report["artifacts"]["training_log"] = str(execution_log)

    artifact_command = None
    should_pull_artifacts = execution_result["exit_code"] == 0 or config["artifacts"][
        "pull_on_failure"
    ]
    if should_pull_artifacts:
        remote_artifact_dir = run_dir / "remote_artifacts"
        artifact_command = build_artifact_sync(config, run_id, remote_artifact_dir)
        if artifact_command is not None:
            remote_artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_result = command_result(
                artifact_command,
                max(300, config["execution"]["timeout_seconds"]),
            )
            report["artifact_sync_result"] = artifact_result
        else:
            artifact_result = None
    else:
        artifact_result = None

    if execution_result["exit_code"] == 0 and (
        artifact_result is None or artifact_result["exit_code"] == 0
    ):
        report["stage"] = "completed"
        report["passed"] = True
        exit_code = 0
    elif execution_result["exit_code"] == 124:
        report["stage"] = "timed_out"
        report["errors"] = ["远端执行超时"]
        exit_code = 124
    elif execution_result["exit_code"] == 125:
        report["stage"] = "stalled"
        report["errors"] = [
            f"远端输出静默超过 {config['execution'].get('silence_timeout_seconds', 900)}s，判定挂死并终止"
        ]
        exit_code = 3
    elif execution_result["exit_code"] == 0 and artifact_result is not None:
        report["stage"] = "artifact_sync_failed"
        report["errors"] = ["远端执行成功，但产物回收失败"]
        exit_code = 3
    else:
        report["stage"] = "execution_failed"
        report["errors"] = [f"远端执行脚本失败，exit code: {execution_result['exit_code']}"]
        if artifact_result is not None and artifact_result["exit_code"] != 0:
            report["errors"].append("失败产物回收同时失败")
        exit_code = 4
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
