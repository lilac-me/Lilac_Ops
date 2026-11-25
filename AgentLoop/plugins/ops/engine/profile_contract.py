"""Training profiling data contract for CANNBot hand-off (no torch dependency)."""
import csv
import json
import math
from pathlib import Path

PROFILE_SETTINGS = {
    "profiler_level": "Level1", "export_type": "Text",
    "aic_metrics": "PipeUtilization", "record_shapes": True,
    "with_stack": True, "with_modules": False, "profile_memory": False,
}
PIPE_COLUMNS = ("aic_mac_ratio", "aic_mte2_ratio", "aiv_vec_ratio",
                "aiv_mte2_ratio", "aic_mac_time(us)", "aiv_vec_time(us)",
                "aicore_time(us)", "aiv_time(us)")


def normalize_header(value):
    return value.strip().replace("μ", "u").replace("µ", "u")


def finite(value):
    try:
        return math.isfinite(float(str(value).strip()))
    except (TypeError, ValueError):
        return False


def check_profile(directory, require_step_trace=False):
    """Reject insufficient export; validate observed fields, not just requested flags."""
    root = Path(directory)
    files = list(root.rglob("kernel_details.csv"))
    if len(files) != 1:
        raise ValueError("expected exactly one kernel_details.csv for selected rank")
    kernel = files[0]
    traces = list(root.rglob("trace_view.json"))
    steps = list(root.rglob("step_trace_time.csv"))
    if len(traces) != 1 or len(steps) > 1 or (require_step_trace and not steps):
        raise ValueError("expected one trace_view.json and at most one step_trace_time.csv")
    with kernel.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {normalize_header(k) for k in (reader.fieldnames or [])}
        required = {"Step Id", "Name", "Duration(us)", "Accelerator Core",
                    "Input Shapes", "Input Data Types", "Stream ID"}
        missing = required - fields
        if missing or not ({"Start Time(us)", "Task Start Time(us)"} & fields):
            raise ValueError(f"profile missing required timing/shape columns: {sorted(missing)}")
        pipes = set(PIPE_COLUMNS) & fields
        if not pipes:
            raise ValueError("missing Level1 PipeUtilization columns")
        shape_rows = pipe_rows = 0
        for raw in reader:
            row = {normalize_header(k): v for k, v in raw.items() if k is not None}
            if not finite(row.get("Step Id")) or not finite(row.get("Duration(us)")):
                continue
            def available(k):
                return str(row.get(k) or "").strip().lower() not in ("", "n/a", "none", "nan", "-")
            if available("Input Shapes") and available("Input Data Types"):
                shape_rows += 1
            if any(finite(row.get(k)) for k in pipes):
                pipe_rows += 1
        if not shape_rows or not pipe_rows:
            raise ValueError("profile has no usable shape/dtype or numeric AI Core metric rows")
    # A valid trace is required for source/dataflow review; do not load huge trace into memory here.
    if traces[0].stat().st_size == 0 or (steps and steps[0].stat().st_size == 0):
        raise ValueError("empty trace or step CSV")
    return {"settings_required": PROFILE_SETTINGS, "kernel_details": str(kernel),
            "trace_view": str(traces[0]), "step_trace_time": str(steps[0]) if steps else None,
            "shape_rows": shape_rows, "pipe_rows": pipe_rows,
            "pipe_columns": sorted(pipes)}
