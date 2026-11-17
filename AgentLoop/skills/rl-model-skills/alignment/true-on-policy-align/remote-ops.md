# Module 3: Discovery and Repair Orchestrator

Own the remote closed loop after Modules 1 and 2. Keep remote process control and the
canonical worktrees under the primary agent. Separate the work into:

- **Discovery Sweep**: find every independent semantic prefill difference in the declared
  coverage, replay it with the training-side Tensor, and continue downstream.
- **Repair Waves**: freeze/group differences, let isolated repair agents prepare
  candidates, then integrate and verify those candidates serially.
- **Stateful/Transfer Validation**: prove the repair across reload/warm-state transitions,
  long-horizon execution, representative reduction and the user's requested model.

Do not analyze decode or TP-sharded collective paths. Reduce the minimal diagnostic case
to TP=1 before discovery, then return to the user's requested topology for transfer and
final validation.

## Entry gate

Require:

```text
REMOTE_SSH_TARGET=
REMOTE_DOCKER=
REMOTE_WORKDIR=
DUMP_SCRIPT=
DUMP_ROOT=
DUMP_TRAIN_PATH=
DUMP_INFER_PATH=
DUMP_SAMPLE_ID=
DUMP_PROMPT_TOKEN_LENGTH=
DUMP_TRAIN_PROMPT_SLICE=
AUTHORIZATION_RECORD=
BASELINE_REVISIONS=
PREEXISTING_DIRTY_PATHS=
```

Authorization must cover the actions actually used: iterative worktree edits, diagnostic
Tensor replay, isolated repair worktrees/subagents, exact job stop scope and dump cleanup.
Missing authority is not a technical failure; ask the user once for the missing scope.

## Single-writer rule

The primary agent alone may:

- start or stop target accelerator jobs;
- change the canonical train/infer worktrees;
- change the active replay chain;
- clean dump output;
- integrate or rollback a candidate;
- declare a real-path result.

Repair agents work only in isolated git worktrees or return patch files. They may not
operate shared jobs or canonical state.

## Evidence layout

Use one task-owned root below the approved `DUMP_ROOT`:

```text
alignment_evidence/
  state_card.json
  commands/
  runs/<run-id>/
    job.json
    stdout.log
    audit.json
    first_nonzero.json
  discovery/epoch-001/
    discovery_registry.json
    replay_chain.json
    triage/
      semantic_mapping.json
      triage_observations.json
      structural_triage.json
    differences/
      D001_<boundary>/
        difference.json
        pair_manifest.json
        compare_report.json
        stack.train.json
        stack.infer.json
        source.train.txt
        source.infer.txt
        replay_entry.json
        replay_activation.json
        iteration.md
  repair/
    R001_<topic>/
      repair_unit.json
      agent_<id>/
        fix_plan.json
        candidate.patch
        probe_adapter.py
        notes.md
      integration/
        selected_plan.json
        applied.patch
        target_verify.json
        rollback.json
  reports/
  package/
```

Keep raw Tensor payloads only in the private session evidence directory. Never copy them
to the Skill casebook or public project.

## Phase A: Discovery Sweep

### State machine

```text
DISCOVERY_INIT
  -> ENSURE_COMPARABLE_DUMPS
  -> OPTIONAL_STRUCTURAL_TRIAGE
  -> CALL_MODULE_4
       EXACT_ALIGNED -> DISCOVERY_COVERAGE_COMPLETE
       NEED_*        -> NARROW_REDUMP -> CALL_MODULE_4
       FIRST_DIVERGENCE_FOUND -> FREEZE_DIFFERENCE_PACKET
       STATE_BOUNDARY_FOUND -> FREEZE_DIFFERENCE_PACKET_V2
  -> BUILD_REPLAY_ENTRY
  -> VALIDATE_REPLAY_MANIFEST
  -> ACTIVATE_REPLAY
  -> VERIFY_REPLAY_EXACT
  -> APPEND_REPLAY_CHAIN
  -> RECOLLECT_MINIMAL_PREFILL
  -> CALL_MODULE_4
  -> FREEZE_DISCOVERY_REGISTRY
```

During this phase, do not apply behavior fixes for RMSNorm, SwiGLU, attention, logits or
other model math. Diagnostic hooks and replay configuration are allowed, but they must be
session-local and reversible.

Optional triage is a read-only acceleration step. A missing input or script failure
records `TRIAGE_STATUS=skipped|failed` and routes directly to the original Module 4
exact path. Candidate priority may guide source inspection or narrow re-dump requests,
but `CALL_MODULE_4` still walks proved semantic execution order.

### Reuse and recollection

- Freeze one training-side reference for the same model/sample/config when determinism
  and provenance remain valid.
- Recollect the inference-side prefill after the replay chain changes. If the environment
  cannot run inference alone, run the already approved minimal dump case and retain only
  the required new evidence.
- Request the smallest operator scope that proves the next boundary. Do not repeatedly
  collect a full dump when a narrow scope is sufficient.
- Before cleanup, archive the Difference Packet, required neighbor Tensors, JSON, stacks,
  source excerpts and hashes.

### Difference Packet gate

Call Module 4 and validate its output with:

```bash
python scripts/validate_difference_packet.py --packet difference.json
```

A packet represents one independent semantic difference only when:

1. provenance and prompt slicing are explicit;
2. all earlier covered outputs and all causal inputs are exact;
3. a Tensor boundary has finite, same-shape/dtype, nonzero output; a non-Tensor v2
   boundary has immutable before/after state digests, observed transition and positive
   downstream impact;
4. active train/infer source paths are recorded;
5. every upstream diagnostic replay is listed;
6. the observation is labeled `native` or `conditional`.

Do not catalog every downstream Tensor contaminated by one upstream error. Replay the
first boundary, then discover the next independent difference from exact inputs.

### Replay construction

Use [references/tensor-replay.md](references/tensor-replay.md). Create one
`replay_entry.json`, append it to `replay_chain.json`, then validate:

```bash
python scripts/validate_replay_manifest.py --manifest replay_chain.json
```

Mandatory rules:

- canonical side is `train`;
- phase is `prefill`;
- load only a session-created file with `weights_only=True`;
- match model/config/sample digests, prompt slice, layer, semantic boundary, symbol and
  call index;
- require exact shape and dtype; device movement is allowed, numeric casting is not;
- no wildcard call matching, broadcast or guessed reshape;
- cover auxiliary returns, mutated state and other causal side effects.

If an internal FA/native operator cannot satisfy the side-effect gate, move the replay to
the nearest stable outer Attention boundary. Record the failed inner boundary and why it
was unsafe.

### Replay activation gate

Before continuing downstream, prove:

```text
hook_active_exactly_once=true
loaded_hash_matches=true
injected_shape_dtype_match=true
torch.equal(injected_output, train_reference)=true
max_abs_diff=0
immediate_downstream_input_exact=true
```

Failure means the replay contract is invalid. Repair the diagnostic boundary or raise it;
do not continue and do not call the underlying difference solved.

### Freeze or reopen a discovery epoch

Discovery is complete only for the declared coverage. Freeze:

- ordered Difference Packets;
- the replay dependency chain;
- uncovered/opaque boundaries;
- repeated occurrences that share a likely implementation;
- total collection/recollection time and accelerator run count.

Set the current epoch `registry_status=frozen`. No repair agent starts before this
default gate. Never rewrite the frozen registry. Reopen only when a repair agent returns
a precise missing-evidence request or an integration/stateful run exposes a genuinely
new state, call-context or earlier boundary. Increment `discovery_epoch`, retain the old
hash and record `reopen_reason`.

## Phase B: Repair Waves

### Build Repair Units

Group differences only when current evidence shows the same:

- active inference symbol or adjacent atomic load hook;
- mathematical/root-cause delta;
- applicability guards;
- verification pair.

Do not create one unit per layer when one shared RMSNorm/SwiGLU implementation caused all
occurrences. Do not combine unrelated residual, activation, attention and logprob fixes.

Each `repair_unit.json` contains:

```text
repair_unit_id
member_diff_ids
dependency_diff_ids
baseline_revisions
immutable_evidence_files
allowed_worktree
forbidden_actions
required_fix_plan
required_probe
target_verify_pairs
```

### Start repair agents

Use the available bounded worker pool; keep the primary agent free to orchestrate.
Parallelize only units without overlapping files/symbols or dependency edges.

Give each agent the immutable Repair Unit and raw evidence references. Ask it to challenge
the diagnosis, re-derive the current-model change and produce exactly one causal plan.
Do not tell an agent to copy an earlier Qwen patch.

Accepted agent statuses:

```text
PLAN_READY
NEED_EVIDENCE
HYPOTHESIS_REJECTED
NO_SAFE_FIX
DEPENDENCY_REQUIRED
```

See [references/agent-orchestration.md](references/agent-orchestration.md).

### Candidate selection and conflict gate

The primary agent checks:

1. `fix_plan.json` passes `scripts/validate_fix_plan.py`;
2. candidate baseline matches the frozen revision;
3. changed paths are inside the assigned worktree and do not overwrite user dirt;
4. no selected candidate conflicts with another unit's symbol/path;
5. probe reproduces both original outputs before judging the candidate;
6. rollback is path-scoped and reversible.

Overlapping candidates are serialized or regrouped; they are never merged blindly.

### Serial integration

Integrate upstream-to-downstream:

```text
SELECT_CANDIDATE
  -> PREFLIGHT
  -> PROBE_REPRODUCE
  -> PROBE_CANDIDATE_EXACT
  -> APPLY_TO_CANONICAL_WORKTREE
  -> WORKER_ACTIVATION_PROOF
  -> REMOVE_ONLY_TARGET_REPLAY
  -> NARROW_REAL_PATH_PREFILL_REDUMP
  -> TARGET_NATIVE_EXACT
  -> CONTINUE_DOWNSTREAM
  -> KEEP_OR_ROLLBACK
```

If the real path is not exact, restore the target replay, rollback only the session patch,
archive the failed evidence and try another evidence-backed plan. When no new evidence
supports another attempt, mark the difference `unresolved_replayed` and continue.

Attention/FA is allowed to finish in that state. A micro-probe that makes the training
operator exact is not enough if active-worker prefill remains nonzero.

## Phase C: Stateful, transfer and final verification

Run the prefill gates first, then every additional gate required by the immutable
objective. Read
[references/stateful-validation.md](references/stateful-validation.md).

### Native prefill

Disable all replay entries. Run the fixed minimal prefill and compare the entire declared
coverage. Set:

- `prefill_native_exact` only if every pair is exact;
- otherwise `prefill_not_exact` and list unresolved native boundaries.

### Conditional prefill

Enable only the final unresolved replay entries. Require all downstream declared pairs
exact. If so, set `prefill_exact_after_unresolved_replay` and list the replay IDs.

### Long-horizon and first-failure gate

When the issue was late-onset or static exactness contradicts E2E, the normal training
run is mandatory, not optional. Run enough complete steps to cross the historical
failure point (or the user-requested horizon), and use:

```bash
python scripts/audit_e2e_log.py \
  --log <tracked-log> \
  --metric <user-metric> \
  --expected-steps <N> \
  --historical-failure-step <step-if-known> \
  --trigger <run-dir>/first_nonzero.json \
  --output <run-dir>/audit.json
```

Pre-arm the backend adapter's bounded capture before launch. The trigger is emitted only
after the metric line exists, so it can notify that pre-armed capture but cannot recover
an already-discarded failing Tensor. Without pre-armed capture, use it only to identify a
rerun step and keep Tensor truth at `evidence_missing`. With `--follow-pid`, log silence
while the process is alive is not EOF; an explicitly configured idle timeout is an audit
error. A log with missing steps, duplicates, nonfinite values, any nonzero value or a
horizon that stops before the historical failure step fails closed.

### Transfer and requested-model gate

After the minimized case passes, run a semantically representative reduced model and
then the requested model/runtime. Record prompt/response/shape coverage. A one-layer or
response-length-1 pass cannot close a broader model/shape objective.

### Clean production and performance gate

Start a fresh process, freshly load weights, disable diagnostic replay/capture/forced
response/fingerprints, prove the narrow production alignment profile activated, and run
the requested regression. Audit every production patch with
`scripts/validate_production_patch.py`. Record throughput/latency/memory/compile changes
on the same scope.

Write all rungs to `validation_matrix`; `prefill_native_exact` and a zero E2E metric stay
separate truth axes. The task completes only when every `acceptance_contract.required_gates`
row passes with retained evidence.

## Iteration record

Update after every discovery or integration action:

```text
OPS_PHASE=DISCOVERY|REPAIR|INTEGRATION|STATEFUL_VERIFY|TRANSFER_VERIFY|FINAL_VERIFY
OPS_STATUS=
OPS_DIFF_ID=
OPS_REPAIR_UNIT_ID=
OPS_DUMP_TRAIN_PATH=
OPS_DUMP_INFER_PATH=
OPS_MANIFEST=
OPS_COMPARE_REPORT=
OPS_TRIAGE_STATUS=
OPS_TRIAGE_ARTIFACT=
OPS_REPLAY_CHAIN=
OPS_FIX_PLAN=
OPS_CHANGED_FILES=
OPS_ROLLBACK=
OPS_EVIDENCE_DIR=
OPS_RUN_LOG=
OPS_METRIC_LINE=
OPS_VALIDATION_RUN_ID=
OPS_EXPECTED_STEPS=
OPS_OBSERVED_STEPS=
OPS_FIRST_NONZERO_TRIGGER=
OPS_DISCOVERY_EPOCH=
OPS_REGISTRY_REOPEN_REASON=
OPS_JOB_LEADER_PID=
OPS_JOB_PGID=
OPS_STARTED_AT=
OPS_ENDED_AT=
OPS_DURATION_SECONDS=
OPS_NEXT_ACTION=
```

## Cleanup

- Clean only approved `DUMP_ROOT/train`, `DUMP_ROOT/infer` outputs and recorded session
  temp files.
- Stop only the recorded job group or explicitly approved exact PIDs.
- Preserve frozen Difference Packets, replay manifests, selected/rejected fix plans,
  integration reports, final patches and reports.
- Remove diagnostic replay from production patch artifacts. Raw replay Tensors remain
  private session evidence and are never published.

## Local-analyze fallback

With supplied artifacts but no executable target accelerator, complete Module 4, build Difference Packets and
prepare Repair Units. Do not claim replay activation, active-worker repair, native
prefill exactness or E2E verification.
