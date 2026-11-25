#!/usr/bin/env python3
"""Analyze a single copied vLLM-Ascend rank profile, no training dependencies."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from profile_contract import check_profile
from operator_routing import route_plan, generation_prompt

ROOT = Path(__file__).resolve().parents[3]
ANALYZER = ROOT / "skills/ops-skills/triton/profile-bottleneck-analyzer/scripts/parse_profile.py"


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True, help="one rank's extracted local profile")
    p.add_argument("--phase", choices=["prefill", "decode", "mixed", "unknown"], default="unknown")
    p.add_argument("--phase-evidence", help="path to actual scheduler/worker evidence; phase is supplied by caller, not inferred")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--model-root", required=True)
    p.add_argument("--source-map", help="JSON kernel name -> mapping")
    p.add_argument("--analyzer-config")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.rank < 0:
        p.error("rank must be nonnegative")
    if args.phase != "unknown" and not args.phase_evidence:
        p.error("phase requires actual scheduler/worker evidence; otherwise use unknown")
    evidence = Path(args.phase_evidence).resolve(strict=True) if args.phase_evidence else None
    if evidence and not evidence.is_file():
        p.error("phase evidence must be a file")
    model = Path(args.model_root).resolve(strict=True)
    if not model.is_dir():
        p.error("model-root must be a directory")
    profile = Path(args.profile).resolve(strict=True)
    contract = check_profile(profile)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "profile_contract.json", contract)
    command = [sys.executable, str(ANALYZER), str(profile), "-o", str(out)]
    if args.analyzer_config:
        command += ["--config", args.analyzer_config]
    subprocess.run(command, check=True)
    candidates = json.loads((out / "candidates.json").read_text())
    if not candidates["steady_steps"]:
        raise ValueError("no usable worker iteration IDs; adapt profiler export first")
    opportunities = json.loads((out / "opportunities.json").read_text())
    mapping = json.loads(Path(args.source_map).read_text()) if args.source_map else {}
    sources = {}
    for name, entry in mapping.items():
        if entry.get("source"):
            path = (model / entry["source"]).resolve()
            rel = str(path.relative_to(model))
            if path.is_file():
                sources[rel] = {"snapshot": str(path)}
    routes = route_plan(opportunities, candidates["candidates"], mapping, sources, model)
    # Training's MatMul MFU default is not appropriate for small-M decode.
    # Keep MFU as diagnostic evidence; do not assume compute-bound without roofline.
    for i, route in enumerate(routes):
        if route["objective"] == "matmul_mfu" and args.phase != "prefill":
            route["objective"] = "matmul_latency_or_mfu"
            route["note"] = "Select latency/bandwidth or MFU from actual shape and arithmetic intensity; no fixed MFU target."
        route["phase"] = args.phase
        route["rank"] = args.rank
        if route.get("plugin") and route["action"] != "resolve_implementation":
            task = out / "requests" / f"task_{i:03d}"
            task.mkdir(parents=True)
            write(task / "request.json", {"route": route, "model_root": str(model),
                   "source_map": mapping, "profile_dir": str(profile),
                   "output_dir": str(task), "phase_evidence": str(evidence) if evidence else None})
            (task / "prompt.txt").write_text(generation_prompt(route) +
                "\nThis is inference: preserve real serving shape, request semantics, KV cache, graph mode, quantization and state. Do not apply changes automatically.\n")
            route["request"] = str(task / "request.json")
    write(out / "operator_routes.json", routes)
    write(out / "analysis_manifest.json", {"status": "ANALYZED_NOT_OPTIMIZED",
          "phase": args.phase, "phase_evidence": str(evidence) if evidence else None,
          "phase_verified_by_tool": False, "rank": args.rank, "profile": str(profile),
          "model_root": str(model), "route_count": len(routes)})
    print(out / "analysis_manifest.json")


if __name__ == "__main__":
    main()
