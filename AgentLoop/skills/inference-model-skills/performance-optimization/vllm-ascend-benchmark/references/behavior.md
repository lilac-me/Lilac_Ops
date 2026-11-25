# Benchmark Skill Behavior

## Remote execution

This skill owns the benchmark lifecycle; its scripts handle all remote
execution for managed workspace sessions. Direct Docker connections (`--host`,
`--container`, `--password` / `--key-file`) are supported — when used the
SSH commands are automatically
wrapped with ``docker exec``.

## Lifecycle

1. **Resolve target** from direct connection params
   (`--host` / `--container` / `--password` / `--key-file`).
2. **Assemble config** from user args.
3. **Stop existing service** on the target if any.
4. **Start service** via `serve_start.py`. If startup returns non-ready, call `serve_stop.py --force` for the same target before failing.
5. **Run benchmark iterations** via SSH on the remote container — all against the same warm service.
6. **Stop service** via `serve_stop.py`.
7. **Output structured JSON** on stdout.

## Configuration Priority

All parameters are explicitly provided by the user via CLI arguments.

## Multi-Run (Warm-Service) Mode

When `--runs N` is given with N > 1, the service starts once and all N iterations run sequentially against the same warm service instance. The service is never restarted between runs.

`--warmup-runs M` excludes the first M runs from the aggregated statistics. This accounts for JIT compilation, graph capture, and other one-time costs that skew initial measurements.

### Why warm-service matters

Restarting the service between runs means every run pays the full startup cost (model loading, graph capture, JIT). The "discard first run" strategy only works when subsequent runs hit the already-warm service. If the service restarts each time, there are no warm runs to keep.

### Aggregation

The output JSON includes:

- `per_run`: every run's metrics, tagged with `warmup: true/false`
- `aggregated`: mean + sample stddev over the non-warmup runs, for each metric key
- `aggregated.count`: number of runs included in the statistics

## Multi-State Comparison

Comparing multiple code states (baseline, PR, modified) is **not** handled by a single benchmark script. Instead:

1. The agent switches the local workspace to each code state.
2. The agent calls `bench_run.py` once per state (each call does service lifecycle + N runs).
3. The agent compares the returned JSON metrics across states.

This keeps the benchmark script focused on one thing (reliable measurement) and leaves orchestration to the agent, which is better equipped to handle git worktrees, cross-fork commits, and submodule complexity.

### Comparison contract

For performance regression comparisons, all runs must use identical core benchmark parameters (`--serve-args`, `--bench-args`, `--extra-env`, `--tp`). Only the code state should change between runs. If any configuration parameter differs, the agent must explicitly record the difference in its output and treat the result as a **configuration comparison**, not a pure regression comparison.

### Regression

Given baseline throughput `T_b` and patched throughput `T_p`, compute the ratio `r = T_p / T_b`. If `r < 0.97`, the patched version is considered a throughput regression. TTFT and TPOT regressions use inverted comparison (`r = T_b / T_p`) since lower is better for latency metrics.

## Remote Execution

`vllm bench serve` runs inside the container via SSH. When `--container` is
set (direct Docker mode) every command is automatically wrapped with
``docker exec -i <container> bash -c …``. The result JSON file is written
to `/tmp/` with a target token, local process id, and random suffix in the
file name, then `cat`-ed back through the SSH session. The script parses
the last JSON object from stdout. The unique file name matters because
containers can share the host `/tmp` mount on the same machine.

## Defaults

When the user provides no bench args:

- `--num-prompts 64`
- `--max-concurrency 16`

These are conservative defaults suitable for a quick smoke test. For production benchmarking, users should specify explicit parameters.

## State Management

Benchmark results are not persisted locally by default. The structured JSON is returned on stdout for the agent or user to consume. The serving skill handles its own state under `.workspace-local/serving/`.
