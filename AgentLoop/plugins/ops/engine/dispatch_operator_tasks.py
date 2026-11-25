#!/usr/bin/env python3
"""Dispatch a reviewed operator task to its configured CANNBot Agent CLI.

This is execution orchestration only. CLI exit zero is not precision/performance
acceptance, and the runner never applies patches to the model.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

PLUGINS = {"triton-op-generator", "tilelang2ascendc-ops-generator"}
ACTIONS = {"optimize_existing", "generate_fused", "generate_replacement"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dispatch(run, task_name, config, task_root):
    if not re.fullmatch(r"task_[0-9]+", task_name):
        raise ValueError("select an explicit reviewed task_NNN")
    run = Path(run).resolve(strict=True)
    base = (run / task_root).resolve(strict=True)
    base.relative_to(run)
    task = (base / task_name).resolve(strict=True)
    task.relative_to(base)
    request_path = task / "request.json"
    prompt_path = task / "prompt.txt"
    request = read(request_path)
    route = request["route"]
    plugin = route.get("plugin")
    if plugin not in PLUGINS or route.get("action") not in ACTIONS:
        raise ValueError("task is not routable to a CANNBot operator plugin")
    if Path(request["output_dir"]).resolve() != task:
        raise ValueError("request output_dir does not match selected task")
    model = Path(request["model_root"]).resolve(strict=True)
    if not model.is_dir():
        raise ValueError("model_root must exist")
    commands = config.get("commands", {})
    argv = commands.get(plugin)
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
        raise ValueError(f"configure the actual installed Agent CLI for {plugin}")
    if not any("{request}" in a or "{prompt}" in a or "{prompt_file}" in a for a in argv):
        raise ValueError("command must receive request or prompt")
    timeout = config.get("timeout_seconds", 7200)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout_seconds must be positive integer")
    prompt = prompt_path.read_text(encoding="utf-8")
    replacements = {"{request}": str(request_path), "{prompt}": prompt,
                    "{prompt_file}": str(prompt_path), "{plugin}": plugin,
                    "{output_dir}": str(task), "{model_root}": str(model)}
    # Single-pass placeholder substitution: never expand placeholders embedded
    # in the prompt or source paths themselves.
    pattern = re.compile("|".join(re.escape(k) for k in replacements))
    command = [pattern.sub(lambda m: replacements[m.group(0)], a) for a in argv]
    attempt = task / ("dispatch_" + uuid.uuid4().hex[:12])
    attempt.mkdir()
    env = dict(os.environ, AGENTLOOP_CANNBOT_PLUGIN=plugin,
               AGENTLOOP_OPERATOR_REQUEST=str(request_path),
               AGENTLOOP_OPERATOR_OUTPUT_DIR=str(task))
    result = {"plugin": plugin, "action": route["action"], "request": str(request_path),
              "status": "CLI_FAILED", "acceptance": "NOT_VERIFIED"}
    start = time.monotonic()
    try:
        with (attempt / "command.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=task, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
                result["exit_code"] = code
                if code == 0:
                    result["status"] = "CLI_COMPLETED_UNVERIFIED"
            except BaseException:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                except ProcessLookupError:
                    pass
                raise
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        (attempt / "dispatch.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result, attempt


def main(task_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True, help="actual installed Agent CLI argv per plugin")
    parser.add_argument("--task", required=True, action="append",
                        help="reviewed task_NNN, repeat to run multiple tasks serially")
    args = parser.parse_args()
    config = read(args.config)
    success = True
    for task in args.task:
        result, attempt = dispatch(args.run, task, config, task_root)
        print(attempt / "dispatch.json", flush=True)
        success = success and result["status"] == "CLI_COMPLETED_UNVERIFIED"
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main("generation_requests"))
