# Stateful and Long-Horizon Validation

Use this reference when a static one-forward prefill case is exact but the user reports a
late, warm-state, reload-time, or multi-step E2E difference. The deterministic minimized
prefill Tensor path, normally reduced to TP=1 to isolate non-collective mathematics,
remains the discovery truth source; this layer proves that the same repair survives state
transitions and the requested topology and end-to-end run.

## Never shrink the objective

Copy the user's final objective into `state_card.json` before minimization. A one-layer,
one-token, one-step run is an intermediate diagnostic gate, not a replacement objective.
Do not stop while a required transfer, long-horizon, clean-runtime, or requested-model
gate remains `not_run`, `fail`, or `evidence_missing`.

Use these independent truth axes:

| Axis | What it proves | Typical evidence |
| --- | --- | --- |
| Static Tensor prefill | Same declared prefill math for one fixed forward | exact pair manifests and downstream continuation |
| Weight sync | Actor/training weights and rollout worker weights match at the declared transition | before/after parameter manifests and reload events |
| Stateful training | Exactness survives optimizer, scheduler, reload, worker and queue transitions | cold/warm/reload matrix and first-failure packet |
| Long-horizon E2E | User metric remains exact for the required duration | strict step-complete log audit |
| Transfer/final model | Repair generalizes from the minimized case to the requested model scope | reduced and final validation runs |

An exact value on one axis must never silently promote another axis.

## Escalation triggers

Enter this workflow immediately when any of these is true:

- the user says the issue appears only after many steps;
- response length 1 or a cold forward is nonzero despite static boundaries being exact;
- cold startup passes but warm, reload, post-update or worker-restart execution fails;
- the actor and rollout paths exchange, reshard, reload or asynchronously publish weights;
- an E2E metric becomes nonzero after a previously exact step;
- a compiler, graph, prefix cache, task queue or kernel choice changes over time.

## Tiered acceptance ladder

Run the smallest applicable gates first, but close the task only at the user's requested
gate:

1. deterministic replay/probe reproduction;
2. native static Tensor prefill exactness;
3. minimized-model state-transition matrix;
4. minimized-model long-horizon run beyond the historical failure step;
5. semantically representative reduced-model transfer;
6. requested-model/requested-runtime final run;
7. clean-process production activation and performance regression gate.

Record every rung in `validation_matrix`. A pass requires a complete run, zero nonfinite
samples, zero nonzero samples, and `max_abs_diff=0`. If a historical failure step is
known, the passing run must cross it. Never infer completion from a still-running PID,
an isolated metric line, or the absence of a recent error.

## Semantic-preserving minimization

Never edit the source model configuration in place. Create an immutable diagnostic copy,
record the source and effective config SHA256 values, and prove the checkpoint/effective
model identity seen by both backends.

Minimize one dimension at a time while preserving every semantic family implicated by
the target architecture. Maintain a coverage matrix such as:

| Semantic family | Preserved? | Evidence |
| --- | --- | --- |
| text embedding/final norm/logit head | yes/no | effective config + source path |
| attention/QKV/RoPE/mask | yes/no | active module map |
| recurrent/GDN or other non-attention block | yes/no | layer-type schedule |
| RMSNorm/residual/activation | yes/no | active call trace |
| dense or MoE experts/router | yes/no | effective expert/top-k config |
| vision/projector/multimodal merge | yes/no/N/A | fixed multimodal sample trace |
| actor-to-rollout weight reload | yes/no | runtime event trace |

A one-layer case that removes the failing operator family is not representative. Expand
the minimum until the failing call-context exists, then reduce experts, layers, sequence
length and batch independently. Final claims still require the requested model scope.

## State-transition matrix

Capture the same fixed sample at these states when applicable:

```text
cold process before any request
warm process after one request
after optimizer update but before rollout publication
immediately before and after actor-to-rollout weight reload
after worker restart/reconnect
compile/cache miss and compile/cache hit
prefix-cache disabled and enabled
task-queue empty and under the reproducing schedule
```

For every transition record model/config/sample digests, worker identity, weight version,
reload generation, compile graph/cache key, prefix-cache state, task/request ID, seed,
stream and active source revision. A static Tensor difference uses
`boundary_kind=tensor`; a proved transition gap uses Difference Packet v2 with
`boundary_kind=weight_reload|state_transition|control_plane`. For a state packet,
compare the recorded `expected_state_digest` with `target_after_state_digest`; do not
require the target's before/after digests to differ. A skipped reload can correctly leave
the target unchanged while still proving a mismatch against the expected post-transition
state.

## First-failure trigger

Run the strict auditor beside or after the long job:

```bash
python scripts/audit_e2e_log.py \
  --log runs/V003/stdout.log \
  --metric training/rollout_probs_diff_mean \
  --valid-metric training/rollout_probs_diff_valid \
  --expected-steps 1000 \
  --expected-first-step 1 \
  --historical-failure-step 777 \
  --trigger runs/V003/first_nonzero.json \
  --output runs/V003/audit.json
```

When monitoring a live job, add `--follow-pid <recorded-leader-pid>`. Live silence is not
EOF: the default waits while the recorded process remains alive. Set `--idle-timeout`
only when the backend has a justified heartbeat bound; expiry fails the audit instead of
claiming a complete run.

The auditor sees a nonzero value only after its metric line is written. Its trigger is a
control-plane notification and cannot recover a failing Tensor that was never retained.
Before starting the long run, pre-arm a bounded session-local ring buffer or an in-process
first-failure callback that can atomically preserve the implicated state. Wire that
capture to the trigger so the first failure records:

- nearest exact/nonzero semantic Tensor boundaries;
- before/after weight and runtime-state fingerprints;
- operator call context (`M/N/K`, token/padding shape, kernel/fusion, layout, dtype and
  accumulation dtype);
- compile/cache/prefix-cache/task-queue provenance.

If capture was not pre-armed, use the trigger only to identify the failing step and rerun
the fixed sample/state transition; mark Tensor evidence as `evidence_missing` until that
rerun succeeds. Do not claim that the trigger retroactively captured an already-lost
Tensor, and do not wait for the job to finish before preserving available state.

## Coverage matrix

Vary prompt/response/shape only after the fixed reproducer is understood. At minimum,
record prompt length, response length, batch, modality, padded/unpadded shapes and
whether each case is cold, warm or reload-time. A response-length-1 pass does not cover a
different response or shape class. Decode Tensor alignment remains outside this Skill;
such runs are E2E regression evidence only.

## Performance gate

Production alignment features must be independently feature-gated and measurable.
Record baseline/candidate throughput, latency, memory, compile time and kernel choice on
the same scope. Do not trade an unbounded regression for numerical exactness without the
user's explicit decision. Diagnostic capture and replay are never performance candidates.

## Registry reopening

Freeze each discovery epoch. Reopen it only when new evidence proves a new state,
call-context or earlier semantic boundary. Increment `discovery_epoch`, retain the
previous immutable registry, record `reopen_reason`, and never rewrite an old Difference
Packet in place.
