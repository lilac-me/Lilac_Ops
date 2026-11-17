# Module 1: Execution Readiness

Prepare a user-confirmed local or remote environment without assuming a training
framework, inference engine, accelerator vendor, container runtime or metric name.

This module proves that the declared baseline command can run and that the agent can
safely own its process lifecycle. It does not collect alignment evidence or diagnose a
difference.

## Exit gate

Complete Module 1 only when:

1. the execution target and ready environment are confirmed;
2. training, inference/rollout and hardware backends are recorded;
3. the resource-inspection command shows the approved resources are available;
4. the user-provided baseline command reaches its declared smoke-success signal;
5. the leader PID, process group, command and log are recorded;
6. liveness, process-tree, log-growth and backend resource smoke evidence agree;
7. stale control-plane/cache/port/shared-memory state has been recorded or ruled out;
8. cleanup stops only the recorded job or leaves it running by explicit user choice;
9. the handoff below is complete.

If the environment cannot be executed, switch to `Local-analyze` only when comparable
artifacts were supplied. Do not mark Module 1 complete.

## One authorization and environment bundle

Ask all foreseeable questions together before any remote mutation:

| Area | Required information |
| --- | --- |
| Access | local/SSH target, user, key/password handling and jump host |
| Environment | activation command and optional container/session name |
| Backends | training framework, inference engine and hardware/runtime |
| Commands | workdir, baseline command and any separate inference service command |
| Logs | exact path or permission to inspect the launch script and discover it |
| Resources | inspection command, approved device IDs and ownership rule |
| Smoke gate | success line/regex and timeout; do not hardcode a metric |
| E2E metric | optional name, target and evidence line |
| Lifecycle | permission to start, stop and force-stop only the recorded job |
| Later modules | edit, replay, isolated repair worktree, dependency and cleanup scope |
| Long horizon | expected steps, historical failure step, first-failure capture and final model scope |
| Dependencies | package mirror choice and sanitized origin when installation is authorized |

Use the `AUTH_*` fields in `SKILL.md`. Credentials must never enter the workspace, Skill,
report or casebook.

## Backend and adapter record

Read [references/backend-adapters.md](references/backend-adapters.md) and record:

```text
TRAINING_BACKEND=
INFERENCE_BACKEND=
HARDWARE_BACKEND=
EVIDENCE_PROVIDER=
RESOURCE_INSPECT_COMMAND=
SMOKE_SUCCESS_PATTERN=
E2E_METRIC_NAME=
E2E_METRIC_TARGET=
```

Examples of resource inspection include `nvidia-smi`, `npu-smi info`, `rocm-smi`, a
cluster scheduler query, or a user-provided CPU/TPU command. Examples are not defaults.
Run only the command confirmed for the target.

## Read-only preflight

1. Prove the connection with a harmless command.
2. Confirm environment/container, workdir, launch files, model and data paths.
3. Record repository revisions and pre-existing dirty paths.
4. Run the confirmed resource-inspection command.
5. If any approved resource is occupied, stop and ask; never terminate another job.
6. Confirm that the baseline command and smoke pattern belong to the same backend path
   that later modules will diagnose.
7. Inspect backend-appropriate stale workers/control plane, ports/sockets, shared memory,
   compile/graph cache, prefix cache, task queue and weight generation without deleting
   anything.

Use non-interactive container execution. On Windows, use the LF/base64 helper in
[references/operational-reliability.md](references/operational-reliability.md) when
quoting becomes complex; do not send CRLF heredocs or embed credentials.

## Launch and smoke gate

Launch the exact user-confirmed command. When safe and approved, give it a dedicated
process group. On POSIX `/proc`, prefer `scripts/managed_job.py`. Immediately record:

```text
JOB_LEADER_PID=
JOB_PGID=
JOB_COMMAND=
JOB_PROCESS_EVIDENCE=
LOG_FILE=
```

Follow the log until the declared smoke pattern appears or the timeout expires. A
framework-specific metric such as `training/rollout_probs_diff_mean` may be used when the
user selects it, but Module 1 requires only the agreed baseline success signal.

Do not report a successful launch until the dedicated PID/PGID identity, liveness grace,
expected process tree, growing log, smoke pattern and accelerator evidence are all
consistent. Store them under one run directory with `job.json`; a detached PID alone is
not launch proof.

On failure, preserve the relevant log tail and report the blocking backend capability.
Do not redesign the environment or install packages unless covered by authorization.

## Stop only the recorded job

Before stopping, show the leader, PGID, command and current members. Signal a process
group only when it is proven dedicated to this launch and `leader_pid == pgid`.

```bash
ps -eo pid,ppid,pgid,args | awk -v pgid="$JOB_PGID" '$3 == pgid'
kill -TERM -- "-$JOB_PGID"
```

Use `KILL` only after an approved grace period. Never use broad `pkill`, unscoped
`xargs kill`, PID 1, or a detached worker merely because its name resembles the job.
Re-run the confirmed resource-inspection command after cleanup.

## Handoff

```text
EXECUTION_TARGET=
REMOTE_SSH_TARGET=
REMOTE_ENV_ACTIVATE=
REMOTE_CONTAINER=
REMOTE_WORKDIR=
BASELINE_COMMAND=
INFERENCE_SERVICE_COMMAND=
REMOTE_LOG_FILE=
TRAINING_BACKEND=
INFERENCE_BACKEND=
HARDWARE_BACKEND=
EVIDENCE_PROVIDER=
RESOURCE_INSPECT_COMMAND=
SMOKE_SUCCESS_PATTERN=
SMOKE_EVIDENCE_LINE=
E2E_METRIC_NAME=
E2E_METRIC_TARGET=
BASELINE_E2E_VALUE=
JOB_LEADER_PID=
JOB_PGID=
JOB_PROCESS_EVIDENCE=
JOB_LIVENESS_EVIDENCE=
RESOURCE_POST_LAUNCH_EVIDENCE=
CONTROL_PLANE_STATE=
COMPILE_CACHE_STATE=
PREFIX_CACHE_STATE=
TASK_QUEUE_STATE=
WEIGHT_GENERATION=
PACKAGE_MIRROR=
PACKAGE_INDEX_ORIGIN_SANITIZED=
```

Module 1 success proves executability, not train/infer alignment.

## Safety

- Do not modify system, driver, network, account or scheduler configuration.
- Do not guess a backend command, resource rule or success metric.
- Clean only recorded session files and approved artifact directories.
- Preserve user-owned dirty changes.
- Never hardcode a vendor package index. Use the user-selected/configured mirror, record
  only a sanitized origin, and verify the installed version/import afterward.
