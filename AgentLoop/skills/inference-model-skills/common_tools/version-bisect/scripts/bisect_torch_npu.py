#!/usr/bin/env python3
"""torch_npu version binary search bisect script."""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ssh_utils import SSHClient, DockerExecutor, CondaExecutor


@dataclass
class TestResult:
    version: str
    status: str = "abnormal"
    detail: str = ""
    timestamp: str = ""


@dataclass
class BisectConfig:
    local_pkg_dir: str
    remote_host: str
    remote_port: int = 22
    remote_user: str = ""
    remote_password: str = ""
    remote_ssh_key: str = ""
    remote_upload_dir: str = ""
    cann_path: str = ""
    env_type: str = "conda"
    env_name: str = ""
    launch_script: str = ""
    verify_script: str = ""
    verify_criteria: str = "log_contains"
    verify_expected: str = "choices"
    timeout_install: int = 120
    timeout_launch: int = 600
    timeout_verify: int = 120


def parse_folder_version(folder_name: str) -> int:
    m = re.search(r"\d+", folder_name)
    return int(m.group()) if m else 0


def scan_package_dirs(local_pkg_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(local_pkg_dir):
        print(json.dumps({"error": f"Directory not found: {local_pkg_dir}"}))
        sys.exit(1)
    entries = []
    for name in os.listdir(local_pkg_dir):
        dir_path = os.path.join(local_pkg_dir, name)
        if not os.path.isdir(dir_path):
            continue
        whl_files = [f for f in os.listdir(dir_path) if f.endswith(".whl")]
        if not whl_files:
            continue
        entries.append((name, dir_path))
    if not entries:
        print(json.dumps({"error": f"No torch_npu .whl files found in {local_pkg_dir}"}))
        sys.exit(1)
    entries.sort(key=lambda e: parse_folder_version(e[0]))
    print(json.dumps({"versions": [v for v, _ in entries]}))
    return entries


def _create_executor(config: BisectConfig, ssh: SSHClient):
    if config.env_type == "docker":
        return DockerExecutor(ssh, config.env_name)
    return CondaExecutor(ssh, config.env_name, config.cann_path)


def is_good(config: BisectConfig, pkg_dir: str, version: str) -> TestResult:
    result = TestResult(version=version, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    ssh = SSHClient(config.remote_host, config.remote_port, config.remote_user, config.remote_password, config.remote_ssh_key)
    try:
        ssh.connect()
        executor = _create_executor(config, ssh)
        whl_files = [f for f in os.listdir(pkg_dir) if f.endswith(".whl")]
        whl_local = os.path.join(pkg_dir, whl_files[0])
        upload_dir = f"{config.remote_upload_dir}/{version}"
        rc, _, _ = executor.exec(f"mkdir -p {upload_dir}")
        if rc != 0:
            return TestResult(version=version, status="fail_upload", detail="mkdir failed")
        executor.upload(whl_local, f"{upload_dir}/{whl_files[0]}")
        whl_remote = f"{upload_dir}/{whl_files[0]}"
        rc, stdout, stderr = executor.exec(
            f"pip uninstall torch-npu -y 2>/dev/null; pip install {whl_remote} && pip list | grep torch",
            timeout=config.timeout_install)
        if rc != 0:
            rc2, _, _ = executor.exec(f"pip install --force-reinstall {whl_remote}", timeout=config.timeout_install)
            if rc2 != 0:
                result.status = "fail_install"
                result.detail = f"pip install failed: {stderr[:300]}"
                return result
        rc, stdout, stderr = executor.exec(f"bash {config.launch_script}", timeout=config.timeout_launch)
        if config.verify_criteria == "exit_code":
            expected = int(config.verify_expected)
            is_normal = (rc == expected)
            result.detail = f"exit_code={rc}"
        elif config.verify_criteria == "log_contains":
            is_normal = config.verify_expected in stdout + stderr
            result.detail = f"contains {config.verify_expected}: {is_normal}"
        else:
            is_normal = (rc == 0)
            result.detail = f"exit_code={rc}"
        result.status = "normal" if is_normal else "abnormal"
    except Exception as e:
        result.detail = str(e)
    finally:
        ssh.close()
    return result


def run_bisect(config: BisectConfig):
    versions = scan_package_dirs(config.local_pkg_dir)
    n = len(versions)
    results = []
    low, high = 0, n - 1
    r0 = is_good(config, versions[0][1], versions[0][0])
    results.append(r0)
    print(json.dumps({"test": versions[0][0], "status": r0.status, "detail": r0.detail}))
    if r0.status == "abnormal":
        print(json.dumps({"result": "earliest_bad", "version": versions[0][0]}))
        return results, versions
    if n > 1:
        rn = is_good(config, versions[-1][1], versions[-1][0])
        results.append(rn)
        print(json.dumps({"test": versions[-1][0], "status": rn.status, "detail": rn.detail}))
        if rn.status == "normal":
            print(json.dumps({"result": "all_good"}))
            return results, versions
        high = n - 1
    while low < high:
        mid = (low + high) // 2
        tested = {r.version for r in results}
        while mid <= high and versions[mid][0] in tested:
            mid += 1
        if mid > high:
            break
        r = is_good(config, versions[mid][1], versions[mid][0])
        results.append(r)
        print(json.dumps({"test": versions[mid][0], "status": r.status, "detail": r.detail}))
        if r.status in ("fail_upload", "fail_install"):
            low = mid + 1
            continue
        if r.status == "normal":
            low = mid + 1
        else:
            high = mid
    return results, versions


def main():
    parser = argparse.ArgumentParser(description="torch_npu Version Bisect")
    parser.add_argument("--local-pkg-dir", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=22)
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--remote-password")
    parser.add_argument("--remote-ssh-key")
    parser.add_argument("--cann-path", required=True)
    parser.add_argument("--remote-upload-dir", required=True)
    parser.add_argument("--env-type", choices=["conda", "docker"], default="conda")
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--launch-script", required=True)
    parser.add_argument("--verify-script")
    parser.add_argument("--verify-criteria", default="log_contains")
    parser.add_argument("--verify-expected", default="choices")
    args = parser.parse_args()
    config = BisectConfig(**{k: v for k, v in vars(args).items() if v is not None})
    results, versions = run_bisect(config)
    print(json.dumps({"results": [asdict(r) for r in results], "versions": [v for v, _ in versions]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
