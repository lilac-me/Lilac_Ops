"""Normalize the two backend families into a single training launch contract."""
from pathlib import Path
import sys
from .profile_contract import PROFILE_SETTINGS

ENGINE = Path(__file__).resolve().parents[1]


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def normalize(cfg):
    """Keep legacy command-based configurations unchanged."""
    if "training" not in cfg:
        return cfg
    if "profile_command" in cfg or "benchmark_command" in cfg:
        raise ValueError("training.launch_command cannot be mixed with legacy phase commands")
    t = cfg["training"]
    p = cfg.setdefault("profiling", {})
    if not isinstance(t, dict) or not isinstance(p, dict):
        raise ValueError("training and profiling must be objects")
    backend = t.get("backend")
    if backend not in ("megatron", "fsdp"):
        raise ValueError("training.backend must be megatron or fsdp")
    t.setdefault("integration", "native" if backend == "megatron" else "hook")
    if t["integration"] not in ("native", "hook"):
        raise ValueError("training.integration must be native or hook")
    command = t.get("launch_command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError("training.launch_command must be a nonempty argv array")
    if "model_root" in t:
        if "model_root" in cfg and cfg["model_root"] != t["model_root"]:
            raise ValueError("conflicting model_root")
        cfg["model_root"] = t["model_root"]
    if "model_root" not in cfg:
        raise ValueError("training.model_root is required")
    for key, default, minimum in (
        ("max_steps", 20, 1), ("benchmark_warmup_steps", 10, 0),
        ("metric_rank", 0, 0),
    ):
        t[key] = integer(t.get(key, default), key, minimum)
    for key, default, minimum in (
        ("skip_first", 10, 0), ("warmup", 1, 0), ("active", 3, 1),
    ):
        p[key] = integer(p.get(key, default), key, minimum)
    if p.get("trigger", "fixed") != "fixed" or p.get("repeat", 1) != 1:
        raise ValueError("v1 supports one fixed profiling window only")
    p["repeat"] = 1
    for key, required in PROFILE_SETTINGS.items():
        if key in p and p[key] != required:
            raise ValueError(f"profiling.{key} must be {required!r} for CANNBot")
        p[key] = required
    ranks = p.setdefault("ranks", [0])
    if not isinstance(ranks, list) or len(ranks) != 1:
        raise ValueError("v1 requires exactly one profiling rank")
    integer(ranks[0], "profiling rank")
    if p["skip_first"] + p["warmup"] + p["active"] > t["max_steps"]:
        raise ValueError("profiling window exceeds max_steps")
    minimum = integer(cfg.get("min_steps", 5), "min_steps", 1)
    if t["max_steps"] - t["benchmark_warmup_steps"] < minimum:
        raise ValueError("insufficient benchmark steps after warmup")
    t.setdefault("metrics", "megatron_log" if backend == "megatron" and t["integration"] == "native" else "json")
    if t["metrics"] not in ("json", "megatron_log"):
        raise ValueError("unsupported metrics adapter")
    if t["integration"] == "hook" and t["metrics"] != "json":
        raise ValueError("hook integration writes JSON metrics")
    if t["metrics"] == "megatron_log" and backend != "megatron":
        raise ValueError("megatron_log is only for Megatron-format logs")
    cfg["profile_command"] = list(command)
    cfg["benchmark_command"] = list(command)
    if t["metrics"] == "megatron_log":
        wrapper = [sys.executable, str(ENGINE / "training_benchmark.py"),
                   "--warmup-steps", str(t["benchmark_warmup_steps"])]
        if "tokens_per_step" in t:
            wrapper += ["--tokens-per-step", str(integer(t["tokens_per_step"], "tokens_per_step", 1))]
        cfg["benchmark_command"] = wrapper + ["--"] + command
    if t["integration"] == "hook":
        subdir = f"rank_{ranks[0]}"
        if cfg.get("profile_subdir", subdir) != subdir:
            raise ValueError("hook profile_subdir is determined by profiling rank")
        cfg["profile_subdir"] = subdir
    return cfg


def phase_env(cfg, phase):
    if "training" not in cfg or phase not in ("profile", "baseline", "regression"):
        return {}
    t, p = cfg["training"], cfg["profiling"]
    return {
        "AGENTLOOP_BACKEND": t["backend"],
        "AGENTLOOP_INTEGRATION": t["integration"],
        "AGENTLOOP_MAX_STEPS": str(t["max_steps"]),
        "AGENTLOOP_BENCHMARK_WARMUP_STEPS": str(t["benchmark_warmup_steps"]),
        "AGENTLOOP_METRIC_RANK": str(t["metric_rank"]),
        "AGENTLOOP_PROFILE_RANK": str(p["ranks"][0]),
        "AGENTLOOP_PROFILE_SKIP_FIRST": str(p["skip_first"]),
        "AGENTLOOP_PROFILE_WARMUP": str(p["warmup"]),
        "AGENTLOOP_PROFILE_ACTIVE": str(p["active"]),
        **{"AGENTLOOP_PROFILE_" + k.upper(): str(v).lower() if isinstance(v, bool) else str(v)
           for k, v in PROFILE_SETTINGS.items()},
    }
