#!/usr/bin/env python3
"""vllm-ascend commit binary search bisect script."""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ssh_utils import SSHClient, DockerExecutor, CondaExecutor


@dataclass
class TestResult:
    commit: str
    vllm_commit: str = ""
    message: str = ""
    status: str = "abnormal"
    detail: str = ""
    timestamp: str = ""


@dataclass
class BisectConfig:
    vllm_ascend_path: str
    vllm_path: str
    remote_host: str
    remote_port: int = 22
    remote_user: str = ""
    remote_password: str = ""
    remote_ssh_key: str = ""
    env_type: str = "conda"
    env_name: str = ""
    launch_script: str = ""
    verify_script: str = ""
    verify_criteria: str = "log_contains"
    verify_expected: str = "choices"
    timeout_launch: int = 600
    timeout_verify: int = 120


VLLM_COMMIT_RE = re.compile(r"vllm-project/vllm/commit/([a-f0-9]{40})")


def get_commits(ascend_path: str, ssh: SSHClient) -> List[Tuple[str, str, str]]:
    rc, stdout, _ = ssh.exec(f"cd {ascend_path} && git log --oneline --format='%H %s'")
    if rc != 0:
        print(json.dumps({"error": "Failed to get git log"}))
        sys.exit(1)
    entries = []
    for line in stdout.strip().split("\n"):
        parts = line.strip().split(" ", 1)
        if len(parts) < 2:
            continue
        commit_id = parts[0]
        msg = parts[1] if len(parts) > 1 else ""
        rc2, stdout2, _ = ssh.exec(f"cd {ascend_path} && git log --format='%B' -1 {commit_id}")
        vllm_commit = ""
        if rc2 == 0:
            m = VLLM_COMMIT_RE.search(stdout2)
            if m:
                vllm_commit = m.group(1)
        if vllm_commit:
            entries.append((commit_id, vllm_commit, msg))
    return entries


def switch_commit(ssh: SSHClient, config: BisectConfig, commit_id: str, vllm_commit: str):
    ssh.exec(f"cd {config.vllm_ascend_path} && git checkout -f {commit_id}")
    ssh.exec(f"cd {config.vllm_path} && git checkout -f {vllm_commit}")


def _create_executor(config: BisectConfig, ssh: SSHClient):
    if config.env_type == "docker":
        return DockerExecutor(ssh, config.env_name)
    return CondaExecutor(ssh, config.env_name)


def is_good(config: BisectConfig, ssh: SSHClient, commit_id: str, vllm_commit: str) -> TestResult:
    result = TestResult(commit=commit_id, vllm_commit=vllm_commit, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        switch_commit(ssh, config, commit_id, vllm_commit)
        executor = _create_executor(config, ssh)
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
    return result


def run_bisect(config: BisectConfig):
    ssh = SSHClient(config.remote_host, config.remote_port, config.remote_user, config.remote_password, config.remote_ssh_key)
    try:
        ssh.connect()
        commits = get_commits(config.vllm_ascend_path, ssh)
        if not commits:
            print(json.dumps({"error": "No commits found with vllm commit info"}))
            return
        print(json.dumps({"commits": [c[0][:8] for c in commits]}))
        n = len(commits)
        results = []
        low, high = 0, n - 1
        r0 = is_good(config, ssh, commits[0][0], commits[0][1])
        results.append(r0)
        print(json.dumps({"test": commits[0][0][:8], "status": r0.status, "detail": r0.detail}))
        if r0.status == "abnormal":
            print(json.dumps({"result": "earliest_bad", "commit": commits[0][0]}))
            return results, commits
        if n > 1:
            rn = is_good(config, ssh, commits[-1][0], commits[-1][1])
            results.append(rn)
            print(json.dumps({"test": commits[-1][0][:8], "status": rn.status, "detail": rn.detail}))
            if rn.status == "normal":
                print(json.dumps({"result": "all_good"}))
                return results, commits
            high = n - 1
        while low < high:
            mid = (low + high) // 2
            tested = {r.commit for r in results}
            while mid <= high and commits[mid][0] in tested:
                mid += 1
            if mid > high:
                break
            r = is_good(config, ssh, commits[mid][0], commits[mid][1])
            results.append(r)
            print(json.dumps({"test": commits[mid][0][:8], "status": r.status, "detail": r.detail}))
            if r.status == "normal":
                low = mid + 1
            else:
                high = mid
    finally:
        ssh.close()
    return results, commits


def main():
    parser = argparse.ArgumentParser(description="vllm-ascend Commit Bisect")
    parser.add_argument("--vllm-ascend-path", required=True)
    parser.add_argument("--vllm-path", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=22)
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--remote-password")
    parser.add_argument("--remote-ssh-key")
    parser.add_argument("--env-type", choices=["conda", "docker"], default="conda")
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--launch-script", required=True)
    parser.add_argument("--verify-script")
    parser.add_argument("--verify-criteria", default="log_contains")
    parser.add_argument("--verify-expected", default="choices")
    args = parser.parse_args()
    config = BisectConfig(**{k: v for k, v in vars(args).items() if v is not None})
    results, commits = run_bisect(config)
    print(json.dumps({
        "results": [{"commit": r.commit[:8], "vllm_commit": r.vllm_commit[:8] if r.vllm_commit else "", "status": r.status, "detail": r.detail} for r in results],
        "commits": [{"ascend": c[0][:8], "vllm": c[1][:8], "msg": c[2][:60]} for c in commits]
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
