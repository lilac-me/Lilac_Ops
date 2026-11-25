"""Framework-neutral iteration hook. Importable without torch/torch_npu."""
import json
import os
from pathlib import Path
from contextlib import nullcontext
from model_metrics import validate_metrics, number


def make_profiler(directory, skip, warmup, active):
    import torch_npu
    p = torch_npu.profiler
    directory.mkdir(parents=True, exist_ok=True)
    return p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        schedule=p.schedule(wait=0, skip_first=skip, warmup=warmup,
                            active=active, repeat=1),
        on_trace_ready=p.tensorboard_trace_handler(str(directory)),
        record_shapes=True,
        with_stack=True,
        profile_memory=False,
        with_modules=False,
        experimental_config=p._ExperimentalConfig(
            profiler_level=p.ProfilerLevel.Level1,
            aic_metrics=p.AiCMetrics.PipeUtilization,
            export_type=p.ExportType.Text),
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False) + "\n")
    tmp.replace(path)


class TrainingHook:
    """Call step once AFTER each full optimizer iteration, on every rank.

    Caller supplies actual globally reduced loss and measured step latency on
    the metrics rank in benchmark phases. No added device synchronization.
    Return True means all ranks should exit at this iteration boundary.
    Outside an AgentLoop run the hook is a no-op.
    """
    def __init__(self, rank=None, env=None, profiler_factory=None):
        self.env = os.environ if env is None else env
        e = self.env
        self.phase = e.get("AGENTLOOP_PHASE")
        self.enabled = self.phase in ("profile", "baseline", "regression")
        self.count = 0
        self.samples = []
        self.context = nullcontext()
        self.profiler = None
        if not self.enabled:
            return
        if e.get("AGENTLOOP_INTEGRATION") != "hook":
            raise ValueError("TrainingHook requires training.integration=hook")
        if rank is None:
            if "RANK" not in e and int(e.get("WORLD_SIZE", "1")) != 1:
                raise ValueError("distributed training must supply global rank")
            rank = int(e.get("RANK", "0"))
        self.rank = rank
        if type(rank) is not int or rank < 0:
            raise ValueError("invalid global rank")
        self.max_steps = int(e["AGENTLOOP_MAX_STEPS"])
        self.warmup = int(e["AGENTLOOP_BENCHMARK_WARMUP_STEPS"])
        self.metric_rank = int(e["AGENTLOOP_METRIC_RANK"])
        self.profile_rank = int(e["AGENTLOOP_PROFILE_RANK"])
        self.profile_enabled = e["AGENTLOOP_PROFILE_ENABLED"] == "1"
        if self.profile_enabled != (self.phase == "profile"):
            raise ValueError("profiling phase mismatch")
        self.directory = Path(e["AGENTLOOP_PROFILE_DIR"]) / f"rank_{self.profile_rank}"
        if self.profile_enabled and rank == self.profile_rank:
            factory = profiler_factory or make_profiler
            self.context = factory(self.directory,
                                   int(e["AGENTLOOP_PROFILE_SKIP_FIRST"]),
                                   int(e["AGENTLOOP_PROFILE_WARMUP"]),
                                   int(e["AGENTLOOP_PROFILE_ACTIVE"]))

    def __enter__(self):
        self.profiler = self.context.__enter__()
        return self

    @property
    def needs_metrics(self):
        return (self.enabled and not self.profile_enabled
                and self.rank == self.metric_rank and self.count >= self.warmup
                and self.count < self.max_steps)

    def step(self, *, loss=None, step_time_ms=None, tokens=None):
        if not self.enabled:
            return False
        if self.count >= self.max_steps:
            raise ValueError("training exceeded max_steps; all ranks must honor stop")
        collect = self.needs_metrics
        self.count += 1
        if self.profiler is not None:
            self.profiler.step()
        if collect:
            sample = {"step": self.count,
                      "step_time_ms": number(step_time_ms, "step_time_ms", positive=True),
                      "loss": number(loss, "loss")}
            if tokens is not None:
                sample["tokens"] = number(tokens, "tokens", positive=True)
            self.samples.append(sample)
        return self.count == self.max_steps

    def __exit__(self, kind, value, traceback):
        self.context.__exit__(kind, value, traceback)
        if kind is not None or not self.enabled:
            return False
        if self.count != self.max_steps:
            raise ValueError(f"training ended after {self.count}, expected {self.max_steps} steps")
        e = self.env
        if self.profile_enabled and self.rank == self.profile_rank:
            atomic_json(self.directory / "agentloop_profile_done.json", {
                "run_id": e["AGENTLOOP_RUN_ID"], "rank": self.rank, "steps": self.count,
                "skip_first": int(e["AGENTLOOP_PROFILE_SKIP_FIRST"]),
                "warmup": int(e["AGENTLOOP_PROFILE_WARMUP"]),
                "active": int(e["AGENTLOOP_PROFILE_ACTIVE"]),
            })
        if not self.profile_enabled and self.rank == self.metric_rank:
            data = {"run_id": e["AGENTLOOP_RUN_ID"], "phase": self.phase,
                    "workload_id": e["AGENTLOOP_WORKLOAD_ID"],
                    "profiling_enabled": False, "samples": self.samples}
            validate_metrics(data, data["run_id"], self.phase, data["workload_id"],
                             self.max_steps - self.warmup)
            atomic_json(e["AGENTLOOP_METRICS_PATH"], data)
        return False
