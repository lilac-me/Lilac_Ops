from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .models import RepositoryRef
from .local_settings import LocalSettings, read_token


OBSERVATORY_REPOSITORY = "https://gitcode.com/ye_qm/mindspeed-issue-observatory.git"
OBSERVATORY_BRANCH = "codex/agent-settings-v0.4"
SUPPORTED_REPOSITORIES = {
    "ascend/mindspeed-mm": "MindSpeed-MM",
    "ascend/mindspeed-llm": "MindSpeed-LLM",
    "ascend/mindspeed": "MindSpeed",
    "ascend/fsdpturbo": "FSDPTurbo",
    "ascend/mindspeed-bridge": "MindSpeed-Bridge",
}


def _command(name: str) -> str:
    value = shutil.which(name)
    if not value and os.name == "nt":
        value = shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
    if not value:
        raise RuntimeError(f"required command is not available: {name}")
    return value


def _node_major(node: str) -> int:
    completed = subprocess.run([node, "--version"], check=True, capture_output=True, text=True)
    match = re.search(r"v?(\d+)", completed.stdout.strip())
    if not match:
        raise RuntimeError("could not determine Node.js version")
    return int(match.group(1))


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _run(command: list[str], cwd: Path | None = None, capture_output: bool = False) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, cwd=cwd, check=True, capture_output=capture_output, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail[-500:]}" if detail else ""
        raise RuntimeError(f"command failed: {Path(command[0]).name} {' '.join(command[1:])}{suffix}") from exc


class WebObservatory:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir).expanduser().resolve() / "web-observatory"
        self.app = self.root / "app"
        self.runtime = self.root / "runtime"
        self.state_path = self.runtime / "service.json"
        self.log_path = self.runtime / "service.log"

    def _state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    @staticmethod
    def _healthy(url: str, timeout: float = 2.0) -> bool:
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.status == 200 and b"MindSpeed" in response.read(65536)
        except (OSError, URLError):
            return False

    def status(self) -> dict:
        state = self._state()
        url = str(state.get("url") or "")
        running = bool(url and self._healthy(url))
        return {
            "installed": (self.app / "package.json").is_file(),
            "running": running,
            "repo": state.get("repo") or "not configured",
            "url": url or "not running",
            "pid": state.get("pid") if running else None,
            "version": state.get("version") or "unknown",
            "log": str(self.log_path),
            "app_dir": str(self.app),
        }

    def install(self, refresh: bool = False) -> dict:
        git = _command("git")
        node = _command("node")
        npm = _command("npm")
        major = _node_major(node)
        if major < 24:
            raise RuntimeError(f"MindSpeed Issue Observatory requires Node.js >=24; found {major}")
        self.root.mkdir(parents=True, exist_ok=True)
        changed_branch = False
        if not self.app.exists():
            _run([git, "clone", "--depth", "1", "--branch", OBSERVATORY_BRANCH,
                  OBSERVATORY_REPOSITORY, str(self.app)])
        elif not (self.app / ".git").is_dir() or not (self.app / "package.json").is_file():
            raise RuntimeError(f"refusing to reuse invalid Observatory directory: {self.app}")
        else:
            origin = _run([git, "remote", "get-url", "origin"], cwd=self.app,
                          capture_output=True).stdout.strip().rstrip("/")
            if origin.removesuffix(".git").lower() != OBSERVATORY_REPOSITORY.removesuffix(".git").lower():
                raise RuntimeError(f"refusing to refresh unexpected Observatory origin: {origin}")
            dirty = _run([git, "status", "--porcelain"], cwd=self.app,
                         capture_output=True).stdout.strip()
            if dirty:
                raise RuntimeError("Observatory checkout has local changes; refusing to refresh")
            current = _run([git, "branch", "--show-current"], cwd=self.app,
                           capture_output=True).stdout.strip()
            if current != OBSERVATORY_BRANCH:
                _run([git, "fetch", "--depth", "1", "origin",
                      f"{OBSERVATORY_BRANCH}:refs/remotes/origin/{OBSERVATORY_BRANCH}"], cwd=self.app)
                _run([git, "switch", "-c", OBSERVATORY_BRANCH,
                      "--track", f"origin/{OBSERVATORY_BRANCH}"], cwd=self.app)
                changed_branch = True
            elif refresh:
                _run([git, "pull", "--ff-only", "origin", OBSERVATORY_BRANCH], cwd=self.app)
        needs_dependencies = refresh or changed_branch or not (self.app / "node_modules").is_dir()
        needs_build = refresh or changed_branch or needs_dependencies or not (self.app / ".next").is_dir()
        if needs_dependencies:
            _run([npm, "ci"], cwd=self.app)
        if needs_build:
            _run([npm, "run", "build"], cwd=self.app)
        return {"installed": True, "app_dir": str(self.app), "node_major": major}

    def start(self, repo: RepositoryRef, port: int = 5173, sync_minutes: int = 60,
              open_browser: bool = True, refresh: bool = False) -> dict:
        if repo.host != "gitcode.com":
            raise ValueError("the full Observatory currently supports GitCode repositories only")
        repo_id = SUPPORTED_REPOSITORIES.get(repo.full_name.lower(), repo.full_name)
        LocalSettings(self.root.parent).ensure_repository(repo)
        if port < 1 or port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if sync_minutes < 0 or sync_minutes > 1440:
            raise ValueError("sync interval must be between 0 and 1440 minutes")
        url = f"http://127.0.0.1:{port}/"
        current = self.status()
        if current["running"]:
            if current["version"] != "0.4.0":
                self.stop()
                current = self.status()
            else:
                if current["url"] != url:
                    raise RuntimeError(f"Observatory is already running at {current['url']}")
                if open_browser:
                    webbrowser.open(url + ("agent-settings" if not read_token() else ""))
                return {**current, "reused": True}
        if not _port_available(port):
            raise RuntimeError(f"port {port} is already in use by another process")
        installation = self.install(refresh=refresh)
        self.runtime.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        token = read_token()
        if token:
            env["GITCODE_TOKEN"] = token
        env["AGENT_CLI"] = str(Path(__file__).resolve().parents[2] / "scripts" / "issue_agent.py")
        env["AGENT_PYTHON"] = sys.executable
        env["AGENT_DATA_DIR"] = str(self.root.parent)
        env.update({"HOST": "127.0.0.1", "PORT": str(port),
                    "SYNC_INTERVAL_MINUTES": str(sync_minutes)})
        node = _command("node")
        log = self.log_path.open("ab")
        kwargs: dict = {
            "cwd": self.app,
            "env": env,
            "stdin": subprocess.DEVNULL,
            "stdout": log,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
            )
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen([node, "scripts/serve.mjs"], **kwargs)
        log.close()
        state = {"pid": process.pid, "repo": repo.full_name, "repo_id": repo_id,
                 "version": "0.4.0",
                 "url": url, "port": port, "started_at": time.time()}
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Observatory exited during startup; inspect {self.log_path}")
            if self._healthy(url):
                if open_browser:
                    webbrowser.open(url + ("agent-settings" if not read_token() else ""))
                return {**installation, **state, "running": True, "reused": False,
                        "log": str(self.log_path), "settings_url": url + "agent-settings"}
            time.sleep(0.5)
        raise RuntimeError(f"Observatory did not become ready within 90 seconds; inspect {self.log_path}")

    def stop(self) -> dict:
        state = self._state()
        pid = state.get("pid")
        if not isinstance(pid, int) or pid < 1:
            return {"stopped": False, "reason": "no managed Observatory process"}
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=True,
                               capture_output=True)
            else:
                os.killpg(pid, signal.SIGTERM)
        except (OSError, subprocess.CalledProcessError):
            if self.status()["running"]:
                raise RuntimeError(f"failed to stop managed Observatory process {pid}")
        self.state_path.unlink(missing_ok=True)
        return {"stopped": True, "pid": pid}
