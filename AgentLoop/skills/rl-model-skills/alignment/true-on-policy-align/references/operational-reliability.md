# Operational Reliability

Use this reference for remote command transport, tracked jobs, resumable state, stale
runtime checks, artifact layout and progress reporting. These controls prevent an
apparently launched or partially validated job from being mistaken for success.

## One session tree

Keep task files under one declared root instead of creating scripts at the repository
top level:

```text
alignment-session/
  state_card.json
  commands/                 # LF scripts; no credentials
  runs/<run-id>/            # job.json, stdout.log, audit.json, first_nonzero.json
  discovery/epoch-<NNN>/    # immutable registries and Difference Packets
  replay/                   # private session-only manifests and payloads
  repair/<repair-unit>/     # plans and candidate patches
  reports/                  # four generated report views
  package/                  # production-only patches and sanitized manifests
```

Initialize and validate the resumable state card:

```bash
python scripts/validate_state_card.py --init alignment-session/state_card.json
python scripts/validate_state_card.py --state alignment-session/state_card.json
```

Update it after every gate, launch, failure, registry freeze/reopen and validation run.
On resume, verify recorded artifact hashes, repository revisions and live PID identity;
do not reconstruct state from chat memory or an old log tail.

Use the timed event helper instead of estimating durations later:

```bash
python scripts/record_state_event.py start \
  --state alignment-session/state_card.json \
  --event-id module-2 --event minimal_evidence_case --module 2

python scripts/record_state_event.py finish \
  --state alignment-session/state_card.json \
  --event-id module-2 --status pass --evidence discovery/epoch-001/manifest.json
```

## Windows to SSH/container transport

Put complex commands in an LF UTF-8 script that contains no password, token, key or
private prompt. Generate a base64-safe command instead of nesting PowerShell, SSH,
`docker exec` and shell quoting manually:

```powershell
python scripts/prepare_remote_script.py `
  --script alignment-session/commands/run.sh `
  --ssh-target user@host `
  --container training-container `
  --workdir /workspace/project `
  --output alignment-session/commands/run.transport.json
```

Use the emitted `ssh_argv` directly. Base64 is transport encoding, not encryption. The
helper rejects common embedded credential forms; credentials stay in the interactive
SSH/askpass mechanism and never enter scripts, logs, reports or commits.

## Read-only stale-runtime preflight

Before each clean validation run, record backend-appropriate evidence for:

- accelerator ownership and available memory;
- existing training/rollout workers and control-plane services;
- stale scheduler/Ray/session metadata, ports and sockets;
- shared-memory capacity and stale IPC objects;
- compile/graph cache and prefix-cache state;
- old weight versions, worker generations and task queues;
- repository revisions, dirty paths, effective config and model paths.

Never delete or kill from a name match. If a resource belongs to another job, stop and
report it. Clean only the exact recorded session resources already authorized.

## Tracked launch gate

On a POSIX `/proc` server, launch a dedicated process group with the included helper:

```bash
python scripts/managed_job.py start \
  --run-dir alignment-session/runs/V001 \
  --cwd /workspace/project \
  --smoke-pattern 'declared backend-ready pattern' \
  --liveness-seconds 10 \
  --smoke-timeout 180 \
  -- bash run_validation.sh
```

The helper refuses existing run records/logs, credential-like commands and platforms
without `/proc`. It records command, PID, PGID, process start ticks, log and smoke
evidence. Do not tell the user a job is running until both identity/liveness and the
declared smoke gate pass.

Inspect or stop only that record:

```bash
python scripts/managed_job.py status --record alignment-session/runs/V001/job.json
python scripts/managed_job.py stop --record alignment-session/runs/V001/job.json
```

`status` separates the persisted launch status from the current `/proc` observation and
lists all recorded-PGID members. `stop` rechecks PID start time and
`leader_pid == pgid`, signals only that group, and waits for the entire PGID to empty—not
merely for the leader to disappear. If the leader is gone while members remain and their
identities cannot be proven, the helper reports `orphaned_group_unverified` or refuses a
force kill instead of claiming success. Use `--force-kill` only after the recorded grace
period and existing authorization.

After launch, independently check the accelerator, process tree and growing log. A PID
without the correct worker/NPU/smoke evidence is not a successful launch.

## Configurable package mirror

If dependencies are authorized, record a sanitized mirror choice in the session state:

```text
PACKAGE_MIRROR=vendor|default|custom
PACKAGE_INDEX_ORIGIN=<scheme and host only; no credentials/query>
PACKAGE_MIRROR_REASON=
```

Prefer the user-requested or environment-local vendor mirror when available, but never
hardcode a vendor URL in the generic Skill. Verify the installed version and import from
the active environment after installation.

## Progress contract

For work longer than one minute, report concise updates at gate transitions and at least
once per minute while actively operating:

```text
current module and objective
what is running: run ID, verified PID/PGID, log path
latest proved evidence, not guesses
next gate and whether the final objective is still open
```

Do not repeatedly ask for an item already covered by the authorization bundle. Do not
announce completion while a required validation-matrix row is missing or incomplete.

## Timing ledger

Record timestamps in `state_card.history` for every module, remote job, repair wave,
integration and audit. Derive wall time from timestamps; do not estimate from memory.
Module 6 copies these values into the report timing block.
