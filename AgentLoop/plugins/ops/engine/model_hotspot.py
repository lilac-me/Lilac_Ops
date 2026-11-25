#!/usr/bin/env python3
"""一次 profiling → 固定候选 → 逐算子调优 → 一次整网回归。无需整网 MFU 目标。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid

from model_metrics import aggregate, compare, number, validate_metrics
from training_backend.base import normalize, phase_env

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / "skills/ops-skills/triton/profile-driven-triton-opt"
ANALYZER = ROOT / "skills/ops-skills/triton/profile-bottleneck-analyzer/scripts/parse_profile.py"
CHECKER = SKILL / "scripts/check_summary_contract.py"
INDEX = ROOT / "skills/ops-skills/vendored-triton-op/triton-latency-optimizer/references/Index.md"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    path.relative_to(root)
    if ".git" in path.relative_to(root).parts:
        raise ValueError("产物或补丁不得指向 .git")
    return path


@contextmanager
def locked(run):
    lock = run / ".lock"
    lock.mkdir()  # 拒绝同时 finish / tune，防止重复应用。
    try:
        yield
    finally:
        lock.rmdir()


def load_config(path):
    path = Path(path).resolve()
    cfg = normalize(read(path))
    cfg["model_root"] = str((path.parent / cfg["model_root"]).resolve())
    if not Path(cfg["model_root"]).is_dir():
        raise ValueError("model_root 不存在")
    for name in ("profile_command", "benchmark_command"):
        argv = cfg.get(name)
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            raise ValueError(f"{name} 必须为非空 argv 数组")
    workload = cfg.get("workload", {})
    for name in ("model", "checkpoint", "data", "seed", "precision", "parallelism", "batch_size", "sequence_length"):
        if name not in workload:
            raise ValueError(f"workload 缺 {name}，必须固定整网对照配置")
    identity = workload if "training" not in cfg else {"workload": workload, "training": cfg["training"], "profiling": cfg["profiling"]}
    cfg["workload_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cfg.setdefault("timeout_seconds", 7200)
    number(cfg["timeout_seconds"], "timeout_seconds", positive=True)
    cfg.setdefault("min_steps", 5)
    if type(cfg["min_steps"]) is not int or cfg["min_steps"] < 1:
        raise ValueError("min_steps 必须为正整数")
    for key, default in (("loss_atol", 1e-5), ("loss_rtol", 1e-3), ("max_step_regression_pct", 0)):
        cfg.setdefault(key, default)
        if number(cfg[key], key) < 0:
            raise ValueError(f"{key} 不得为负数")
    cfg.setdefault("source_map", {})
    cfg["latency_index"] = str((path.parent / cfg.get("latency_index", str(INDEX))).resolve())
    return cfg


def execute(argv, cfg, run, phase, directory, extra=None):
    directory.mkdir(parents=True, exist_ok=True)
    substitutions = {"agentloop_root": str(ROOT), "model_root": cfg["model_root"],
                     "run_dir": str(run), **(extra or {})}
    command = []
    for arg in argv:
        for key, value in substitutions.items():
            arg = arg.replace("{" + key + "}", str(value))
        command.append(arg)
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    env.update({"AGENTLOOP_RUN_ID": run.name, "AGENTLOOP_PHASE": phase,
                "AGENTLOOP_RUN_DIR": str(run), "AGENTLOOP_MODEL_ROOT": cfg["model_root"],
                "AGENTLOOP_PROFILE_DIR": str(run / "profile"),
                "AGENTLOOP_METRICS_PATH": str(directory / "metrics.json"),
                "AGENTLOOP_WORKLOAD_ID": cfg["workload_id"],
                "AGENTLOOP_PROFILE_ENABLED": "1" if phase == "profile" else "0"})
    backend_env = phase_env(cfg, phase)
    if backend_env:
        env.update(backend_env)
        # Training processes import the hook from the engine, including torchrun children.
        engine = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = engine + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        for key, value in backend_env.items():
            command = [arg.replace("{" + key.removeprefix("AGENTLOOP_").lower() + "}", value) for arg in command]
    with (directory / "command.log").open("w") as stream:
        process = subprocess.Popen(command, cwd=cfg["model_root"], env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=cfg["timeout_seconds"])
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    write(directory / "execution.json", {"phase": phase, "exit_code": code,
                                         "workload_id": cfg["workload_id"]})
    if code:
        raise RuntimeError(f"{phase} 命令失败 ({code})，见 {directory / 'command.log'}")


def metrics(run, cfg, phase):
    data = validate_metrics(read(run / phase / "metrics.json"), run.name, phase,
                            cfg["workload_id"], cfg["min_steps"])
    if "training" in cfg:
        t = cfg["training"]
        expected = t["max_steps"] - t["benchmark_warmup_steps"]
        if len(data["samples"]) != expected:
            raise ValueError(f"expected {expected} benchmark samples")
        if t["integration"] == "hook" and [s["step"] for s in data["samples"]] != list(range(t["benchmark_warmup_steps"] + 1, t["max_steps"] + 1)):
            raise ValueError("hook benchmark step window mismatch")
    return data


def unchanged(state):
    root = Path(state["config"]["model_root"])
    for rel, snapshot in state["sources"].items():
        if digest(inside(root, rel)) != snapshot["sha256"]:
            raise ValueError(f"基线源码已改变，拒绝继续应用补丁: {rel}")


def report(run, state):
    write(run / "state.json", state)
    write(run / "model_report.json", state)
    lines = ["# 整网热点算子优化结果", "", f"状态：{state['status']}",
             "", "本轮仅采集一次热点；算子内部迭代；整网只做一次优化后回归。", ""]
    result = state.get("comparison")
    if result:
        lines += ["| 指标 | 基线 | 优化后 |", "| --- | ---: | ---: |"]
        for label, key in (("平均 step（ms）", "mean_step_ms"), ("中位 step（ms）", "median_step_ms"),
                           ("吞吐（tokens/s）", "tokens_per_second")):
            lines.append(f"| {label} | {result['baseline'][key]} | {result['optimized'][key]} |")
        lines += ["", f"实测整网加速比：{result['step_speedup']:.4f}×；"
                  f"loss 对照通过：{result['accuracy_passed']}；补丁保留：{result['accepted']}。", ""]
    elif "baseline" in state:
        lines += [f"基线平均 step：{aggregate(state['baseline'])['mean_step_ms']:.4f} ms。",
                  "未产生优化后整网测量，不填写推算收益。", ""]
    if state.get("error"):
        lines += [f"失败原因：{state['error']}", ""]
    lines += ["| 算子 | 算子加速比 | 处理结果 | 原因 |", "| --- | ---: | --- | --- |"]
    for task in state.get("tasks", []):
        cells = [str(task.get(k, "")).replace("|", "\\|").replace("\n", " ")
                 for k in ("name", "operator_speedup", "status", "reason")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "热点占比来自 kernel 耗时聚合，不作为整网实际时间占比；整网收益只使用对照实测。", ""]
    (run / "model_report.md").write_text("\n".join(lines))


def prepare(config_path, out):
    cfg = load_config(config_path)
    run = Path(out).resolve() / ("model_opt_" + uuid.uuid4().hex[:12])
    run.mkdir(parents=True)
    state = {"run_id": run.name, "status": "PREPARING", "config": cfg, "sources": {}, "tasks": []}
    report(run, state)
    try:
        for entry in cfg["source_map"].values():
            source = inside(cfg["model_root"], entry["source"])
            if not source.is_file():
                continue
            rel = str(source.relative_to(cfg["model_root"]))
            frozen = run / "original" / rel
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, frozen)
            state["sources"][rel] = {"sha256": digest(source), "snapshot": str(frozen)}
        execute(cfg["profile_command"], cfg, run, "profile", run / "profile")
        profile = inside(run / "profile", cfg.get("profile_subdir", "."))
        if cfg.get("training", {}).get("integration") == "hook":
            p, t = cfg["profiling"], cfg["training"]
            done = read(profile / "agentloop_profile_done.json")
            expected = {"run_id": run.name, "rank": p["ranks"][0], "steps": t["max_steps"],
                        "skip_first": p["skip_first"], "warmup": p["warmup"], "active": p["active"]}
            if done != expected:
                raise ValueError("hook profiling completion marker mismatch")
        if len(list(profile.rglob("kernel_details.csv"))) != 1:
            raise ValueError("请选择仅含一个 kernel_details.csv 的 profile_subdir（显式选择统计 rank）")
        analyzer_config = run / "analyzer_config.json"
        write(analyzer_config, cfg.get("analyzer", {"bottleneck_threshold": 0.03}))
        subprocess.run([sys.executable, str(ANALYZER), str(profile), "-o", str(run / "analysis"),
                        "--config", str(analyzer_config)], check=True, stdout=subprocess.DEVNULL)
        candidate_path = run / "analysis/candidates.json"
        state["candidates_sha256"] = digest(candidate_path)
        execute(cfg["benchmark_command"], cfg, run, "baseline", run / "baseline")
        state["baseline"] = metrics(run, cfg, "baseline")
        unchanged(state)
        for i, candidate in enumerate(read(candidate_path)["candidates"]):
            task = {"name": candidate["name"], "candidate": candidate, "status": "PENDING"}
            state["tasks"].append(task)
            if candidate["tier"] != "strong":
                task.update(status="NEEDS_REVIEW", reason="弱候选不自动调优")
                continue
            mapping = cfg["source_map"].get(candidate["name"])
            if not mapping:
                task.update(status="SKIPPED_NO_SOURCE", reason="缺少源码映射")
                continue
            source = inside(cfg["model_root"], mapping["source"])
            rel = str(source.relative_to(cfg["model_root"]))
            if rel not in state["sources"]:
                task.update(status="SKIPPED_NO_SOURCE", reason="源码不存在")
                continue
            directory = run / "operators" / f"op_{i:03d}"
            directory.mkdir(parents=True)
            task.update(directory=str(directory), source=rel)
            request = {"run_id": run.name, "candidate": candidate, "source": str(source),
                       "frozen_source": state["sources"][rel]["snapshot"], "mode": mapping.get("mode", "B"),
                       "output_dir": str(directory), "summary_path": str(directory / "summary.json"),
                       "operator_config": cfg.get("operator_config", {}), "latency_index": cfg["latency_index"]}
            write(directory / "request.json", request)
            task["request_sha256"] = digest(directory / "request.json")
            prompt = (f"使用 {SKILL / 'SKILL.md'} 的 S3/S4-S6 算子工作流处理 {directory / 'request.json'}。\n"
                      "本次只处理该候选；候选清单与源码基线已冻结。所有结果写入 output_dir。\n"
                      "读取 source 和 frozen_source；仅修改工作目录副本，禁止修改模型仓源码。\n"
                      "逐轮调用 cannbot latency/simulator skills，沿用算子层退出条件，不设置整网目标。\n"
                      "交付 summary.json、report.md、验证/benchmark 证据及针对 source 的补丁。\n"
                      "summary 增加 run_id 与 op_name，perf_baseline 指向 source；无源或失败如实报告。\n"
                      "不得复用历史结果；不得自行创建计时器顶替原有 benchmark。")
            (directory / "prompt.txt").write_text(prompt)
        state["status"] = "READY"
        report(run, state)
    except BaseException as exc:
        state.update(status="FAILED", error=str(exc))
        report(run, state)
        raise
    return run


def tune(run):
    with locked(run):
        state = read(run / "state.json")
        if state["status"] != "READY":
            raise ValueError("只能调优 READY 状态的运行")
        cfg = state["config"]
        argv = cfg.get("tune_command")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError("自动运行需要 tune_command argv；也可手动执行各 prompt.txt 后 finish")
        for task in state["tasks"]:
            if task["status"] != "PENDING":
                continue
            directory = Path(task["directory"])
            try:
                unchanged(state)
                execute(argv, cfg, run, "tune", directory,
                        {"request": directory / "request.json", "prompt": (directory / "prompt.txt").read_text()})
                task["status"] = "TUNED"
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                task.update(status="FAILED", reason=str(exc))
            unchanged(state)
            report(run, state)


def artifact(directory, path):
    result = inside(directory, path)
    if not result.is_file():
        raise ValueError(f"缺少本轮产物: {result}")
    return result


def validated_patch(run, state, task):
    cfg, directory = state["config"], Path(task["directory"])
    if digest(directory / "request.json") != task["request_sha256"]:
        raise ValueError("任务输入已改变")
    summary_path = directory / "summary.json"
    summary = read(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("summary 必须是 JSON 对象")
    if summary.get("run_id") != run.name or summary.get("op_name") != task["name"]:
        raise ValueError("summary 不属于本轮候选")
    if summary.get("status") not in ("REACHED", "BELOW") or not summary.get("best_code"):
        raise ValueError("没有可采纳的优化版本")
    if type(summary.get("optimizer_rounds")) is not int or summary["optimizer_rounds"] <= 0:
        raise ValueError("没有实际算子调优轮次")
    task["operator_status"] = summary["status"]
    source = inside(cfg["model_root"], task["source"])
    if Path(summary.get("perf_baseline", "")).resolve() != source:
        raise ValueError("性能基线不是本轮用户源码")
    for block in ("bench", "optimizer", "patch"):
        if not isinstance(summary.get(block), dict):
            raise ValueError(f"summary 缺少合法 {block} 对象")
        if summary[block].get("waiver"):
            raise ValueError(f"自动整网集成不采纳带 {block}.waiver 的补丁")
    artifact(directory, summary["best_code"])
    bench = summary["bench"]
    times = [number(read(artifact(directory, bench[key]))["implementation"]["avg_latency_ms"], key, True)
             for key in ("baseline_result_json", "optimized_result_json")]
    if times[1] >= times[0]:
        raise ValueError("真实 benchmark 未显示单算子收益")
    patch = summary["patch"]
    if Path(patch.get("apply_target", "")).resolve() != source or patch.get("apply_rel") != task["source"]:
        raise ValueError("补丁落点与候选源码不一致")
    frozen = artifact(directory, patch["diff_from"])
    target = artifact(directory, patch["diff_target"])
    if digest(frozen) != state["sources"][task["source"]]["sha256"]:
        raise ValueError("补丁未从本轮冻结源码派生")
    patch_path = artifact(directory, patch["path"])
    check = subprocess.run([sys.executable, str(CHECKER), str(summary_path), "--triton-source", str(source),
                            "--index", cfg["latency_index"]], capture_output=True, text=True)
    (directory / "contract.log").write_text(check.stdout + check.stderr)
    if check.returncode:
        raise ValueError("算子产物契约未通过，见 contract.log")
    task["operator_speedup"] = times[0] / times[1]
    return patch_path, target


def apply_in_sandbox(sandbox, patch, rel):
    result = subprocess.run(["git", "apply", "--numstat", "-z", str(patch)], cwd=sandbox,
                            check=True, capture_output=True)
    paths = [item.split(b"\t", 2)[2].decode() for item in result.stdout.split(b"\0") if item]
    if paths != [rel]:
        raise ValueError("补丁必须仅修改当前候选源码，禁止额外文件/重命名")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=sandbox, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=sandbox, check=True, capture_output=True)


def finish(run):
    with locked(run):
        state = read(run / "state.json")
        if state["status"] != "READY":
            raise ValueError("finish 仅允许 READY 状态；已结束的运行不得重复应用补丁")
        cfg = state["config"]
        applied = []
        try:
            unchanged(state)
            if digest(run / "analysis/candidates.json") != state["candidates_sha256"]:
                raise ValueError("热点清单已改变")
            accepted = []
            with tempfile.TemporaryDirectory(prefix="agentloop_patch_") as tmp:
                sandbox = Path(tmp)
                for rel, meta in state["sources"].items():
                    dest = sandbox / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(meta["snapshot"], dest)
                for task in state["tasks"]:
                    if task["status"] not in ("PENDING", "TUNED"):
                        continue
                    rel = task["source"]
                    before = (sandbox / rel).read_bytes()
                    try:
                        patch, target = validated_patch(run, state, task)
                        # 先独立验证补丁与已复验的目标字节一致，再测试整批累积应用。
                        with tempfile.TemporaryDirectory(prefix="agentloop_single_") as one:
                            dest = Path(one) / rel
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(state["sources"][rel]["snapshot"], dest)
                            apply_in_sandbox(one, patch, rel)
                            if dest.read_bytes() != target.read_bytes():
                                raise ValueError("补丁结果与已复验的 diff_target 不一致")
                        apply_in_sandbox(sandbox, patch, rel)
                        accepted.append(task)
                        task.update(status="SELECTED", reason="契约通过且单算子有收益")
                    except (ValueError, TypeError, KeyError, OSError, subprocess.CalledProcessError) as exc:
                        (sandbox / rel).write_bytes(before)
                        task.update(status="REJECTED", reason=str(exc))
                if not accepted:
                    state["status"] = "NO_CHANGES"
                    report(run, state)
                    return state
                unchanged(state)
                state["status"] = "APPLYING"
                report(run, state)
                for rel in sorted({t["source"] for t in accepted}):
                    applied.append(rel)
                    shutil.copy2(sandbox / rel, inside(cfg["model_root"], rel))
                state["applied_sources"] = applied
                state["status"] = "REGRESSION"
                report(run, state)
            execute(cfg["benchmark_command"], cfg, run, "regression", run / "regression")
            state["comparison"] = compare(state["baseline"], metrics(run, cfg, "regression"),
                                          cfg["loss_atol"], cfg["loss_rtol"], cfg["max_step_regression_pct"])
            keep = state["comparison"]["accepted"]
            if not keep:
                for rel in applied:
                    shutil.copy2(state["sources"][rel]["snapshot"], inside(cfg["model_root"], rel))
            for task in accepted:
                task["status"] = "APPLIED" if keep else "ROLLED_BACK"
            state["status"] = "COMPLETED" if keep else "ROLLED_BACK"
        except BaseException as exc:
            for rel in applied:
                shutil.copy2(state["sources"][rel]["snapshot"], inside(cfg["model_root"], rel))
            for task in state["tasks"]:
                if task["status"] == "SELECTED":
                    task["status"] = "ROLLED_BACK"
            state.update(status="FAILED", error=str(exc))
            report(run, state)
            raise
        report(run, state)
        return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run"):
        p = sub.add_parser(action)
        p.add_argument("--config", required=True)
        p.add_argument("--out", required=True)
    for action in ("tune", "finish"):
        sub.add_parser(action).add_argument("--run", required=True)
    args = ap.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"收到信号 {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.action in ("prepare", "run"):
            if args.action == "run":
                cfg = load_config(args.config)
                if not cfg.get("tune_command") or not Path(cfg["latency_index"]).is_file():
                    raise ValueError("自动运行前需配置 tune_command 并安装 cannbot skills（缺 latency Index.md）")
            run = prepare(args.config, args.out)
            print(run, flush=True)
        else:
            run = Path(args.run).resolve()
        if args.action in ("run", "tune"):
            tune(run)
        if args.action in ("run", "finish"):
            state = finish(run)
            print(run / "model_report.md")
            return 0 if state["status"] in ("COMPLETED", "NO_CHANGES") else 2
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
