# Module 5: Repair Unit Strategy

Consume one frozen Repair Unit created after the discovery sweep. Re-derive exactly one
causal, reversible repair plan for the current model and active prefill path.

This module may run inside a repair subagent. It does not control remote jobs, edit the
canonical worktree, activate replay, clean dumps, integrate patches or declare success.

## Entry gate

Require:

```text
REPAIR_UNIT_ID=
MEMBER_DIFF_IDS=
IMMUTABLE_DIFFERENCE_PACKETS=
DEPENDENCY_DIFF_IDS=
BASELINE_REVISIONS=
ALLOWED_ISOLATED_WORKTREE=
PREEXISTING_DIRTY_PATHS=
TARGET_VERIFY_PAIRS=
```

Every member Difference Packet must pass
`scripts/validate_difference_packet.py`. At least one packet must have root-cause
confidence `CONFIRMED` or `SUPPORTED`. Otherwise return `NEED_EVIDENCE`.

## Statuses

Return exactly one:

| Status | Meaning |
| --- | --- |
| `PLAN_READY` | One evidence-backed, reversible repair plan and candidate are ready |
| `NEED_EVIDENCE` | A precise source/Tensor/activation fact is missing |
| `HYPOTHESIS_REJECTED` | The assigned cause is falsified by the evidence |
| `NO_SAFE_FIX` | No isolated reversible implementation is available |
| `DEPENDENCY_REQUIRED` | A compatible required package/operator is absent |
| `UNSUPPORTED_VERSION` | Active source/signature/load order does not support the proposed hook |

`NO_SAFE_FIX` is a valid result. The primary agent marks the underlying differences
`unresolved_replayed` after confirming that no new evidence-backed alternative remains.

## Subagent contract

A repair agent may:

- read the frozen evidence and current source;
- challenge the root-cause classification;
- edit only its assigned isolated worktree;
- build a fixed-artifact micro-probe;
- emit a path-scoped candidate patch, plan and rollback instructions.

It may not:

- use or print credentials;
- modify the canonical train/infer worktree;
- start, stop or inspect unrelated accelerator jobs;
- clean shared dumps;
- merge its candidate;
- remove replay from the active chain;
- declare active-worker/native prefill exactness.

See [references/agent-orchestration.md](references/agent-orchestration.md).

## Re-derive; do not copy

Historical patches and casebook records are analogies only. For every plan, independently
record:

1. current model architecture and revision;
2. minimized prefill semantic boundary and its diagnostic topology;
3. active train/infer source and runtime path;
4. exact training formula/order/dtype/layout that must be reproduced;
5. inference interface invariants;
6. applicability guards;
7. why all member differences share the same repair.

Set:

```text
rederived_for_current_model=true
copied_historical_patch=false
```

Reject plans that merely fit a known Qwen patch to a similar function name.

## Repair selection

```text
frozen Repair Unit
  -> effective config alone explains the boundary? -> config
  -> Python symbol safely replaceable before workers cache it? -> runtime experiment
  -> runtime load order/native dispatch defeats it? -> guarded source patch
  -> first difference is weight loading/conversion? -> framework/reshard plan
  -> no isolated rollback or activation proof? -> NO_SAFE_FIX
```

Prefer making inference reproduce training mathematics. Training/shared-side changes are
exceptional and require explicit repository approval plus proof that inference-side repair
is invalid. Effective-vocabulary masking during training is an example that must be
classified as a framework/representation boundary, not hidden as an inference op patch.

## Model-agnostic pattern questions

### Residual and RMSNorm

Check whether one side uses a fused add+RMSNorm output while the other performs:

```python
combined = hidden + residual
normalized = rms_norm(combined, weight, eps)
```

Match addition order, accumulation/cast placement, epsilon, weight dtype and which fused
return field is consumed. A probe that simply returns a saved training Tensor is replay,
not a repair. A durable fix changes the active inference formula/selection and remains
exact for repeated prefill calls.

### SwiGLU/gating

Derive the actual gate/up packing order. Compare the fused activation with the training
formula, for example:

```python
gate, up = torch.chunk(gate_up, 2, dim=-1)
output = torch.nn.functional.silu(gate) * up
```

Match split axis/order and BF16/FP32 cast placement. Do not assume all gated MLPs use
Qwen-style `[gate, up]`.

### Attention/FA prefill

Check Q/K/V, mask predicate, scale, sparse mode, layout and active operator. A training
operator candidate that is exact only in a fixed-artifact probe remains experimental.
If the real worker path is still nonzero or native side effects cannot be reproduced,
return `NO_SAFE_FIX` or a narrow missing-evidence request. Do not spend unbounded rounds
on FA; preserve it as an unresolved replay boundary and continue.

### Logits, logprob and vocabulary

After final hidden/projection input is exact, check:

- effective vocabulary versus padded vocabulary width;
- padded-tail masking before log-softmax/cross-entropy;
- raw versus processed rollout logprobs;
- max-subtraction and accumulation dtype;
- selected token and causal position.

Do not patch sampling just because the final metric differs.

### Generic op/config boundary

For residual, activation, linear, RoPE, mask or other dense boundaries, match only the
proved mathematical delta. Preserve shape, dtype, device, return schema, aliasing,
scheduler/request behavior and no-backward execution.

## One causal change

One plan may touch adjacent files only when they form one atomic activation/load hook.
Record:

- hypothesis and root-cause class;
- member Difference Packet IDs;
- exact repository, baseline, paths and symbols;
- code-level mathematical delta;
- applicability guards and excluded paths;
- worker activation proof;
- micro-probe;
- target exact pairs;
- failure routes and path-scoped rollback;
- performance risk;
- expected throughput/latency/memory/compile impact and the matching performance gate;
- a narrow production alignment profile/feature flag and activation evidence;
- whether the candidate replaces replay or only adds instrumentation.

Production and diagnostic controls must be separate. A repair may use a scoped
`ALIGNMENT_PROFILE` or one feature flag for its mathematical change. Tensor capture,
forced responses, replay, stack tracing and weight fingerprints use diagnostic-only
controls and are removed before production packaging. Never make a broad debug flag the
activation mechanism for a production repair.

Do not combine RMSNorm, SwiGLU, attention and logprob rewrites in one plan.

## Probe gate

Use [references/verification-probe.md](references/verification-probe.md). First reproduce:

```text
train callable == original train dump
infer baseline == original infer dump
train callable != infer baseline
```

Then require the candidate:

```text
same_shape=true
same_dtype=true
no_nonfinite=true
torch.equal=true
max_abs_diff=0
```

The repair agent may report probe exactness. Only the primary agent may prove worker
activation and native real-path prefill exactness.

## Fix plan contract

Write `fix_plan.json` with one top-level `causal_change` and validate:

```bash
python scripts/validate_fix_plan.py --plan fix_plan.json
```

Required conceptual fields:

```json
{
  "status": "PATCH_PROPOSAL_READY",
  "repair_unit_id": "R001",
  "member_diff_ids": ["D002", "D006"],
  "analysis": {
    "status": "FIRST_DIVERGENCE_FOUND",
    "root_cause_class": "Op path",
    "root_cause_confidence": "CONFIRMED",
    "first_divergence": "post_attention_rmsnorm",
    "first_divergence_proof": "all inputs exact; output nonzero",
    "evidence_files": ["difference.json"]
  },
  "generalization": {
    "current_model": {"family": "...", "architecture": "...", "revision": "..."},
    "architecture_evidence": ["..."],
    "active_path_evidence": ["..."],
    "historical_analogy": "pattern only or none",
    "applicability_limits": ["prefill semantic boundary", "TP collective behavior not evaluated"],
    "rederived_for_current_model": true,
    "copied_historical_patch": false
  },
  "causal_change": {
    "id": "R001.C1",
    "pattern": "residual-rmsnorm",
    "hypothesis": "...",
    "method": "config|runtime_monkey_patch|source_patch|weight_or_reshard_patch",
    "target": {"side": "infer", "repository": "...", "baseline_revision": "...", "active_runtime_source": "...", "preexisting_dirty_paths": [], "symbols": ["..."], "changed_paths": ["..."]},
    "train_reference": {"repository": "...", "revision": "...", "path": "...", "symbol": "...", "semantic_boundary": "..."},
    "mathematical_delta": "...",
    "interface_invariants": ["..."],
    "applicability_guards": ["prefill", "minimal evidence proved", "requested topology requires final validation"],
    "activation_probe": "...",
    "preflight_checks": ["..."],
    "micro_probe": {"mode": "single_process_ab|dual_process_replay|in_situ_targeted|not_applicable", "adapter": "...", "case_dir": "...", "captured_inputs": ["sha256:..."], "state_dependencies": ["..."], "semantic_transforms": ["..."], "train_entry": "...", "infer_baseline_entry": "...", "infer_candidate_entry": "...", "reproduction_expected": {"train_matches_train_dump": true, "infer_matches_infer_dump": true, "baseline_difference_present": true}, "candidate_expected": {"same_shape": true, "same_dtype": true, "no_nonfinite": true, "torch_equal": true, "max_abs_diff": 0}, "real_path_integration_gate": "primary-agent narrow prefill re-dump", "cleanup": "...", "limitations": ["..."]},
    "targeted_verify_pair": {"manifest": "...", "pair_name": "...", "expected": {"same_shape": true, "same_dtype": true, "no_nonfinite": true, "torch_equal": true, "max_abs_diff": 0}},
    "success_gates": ["probe exact", "worker active", "target native exact", "continue downstream"],
    "failure_routes": {"preflight_failure": "...", "activation_failure": "...", "no_effect": "...", "target_worse": "...", "downstream_difference": "...", "e2e_regression": "..."},
    "rollback": {"scope": "...", "steps": ["..."], "restart_required": true},
    "performance_risk": "...",
    "performance_gate": {"baseline": "...", "candidate": "...", "allowed_regression": "...", "evidence": "..."},
    "production_alignment_profile": {"profile_id": "...", "feature": "...", "default_enabled": false, "activation_probe": "..."},
    "diagnostic_controls_removed": true,
    "packaging_disposition": "...",
    "case_candidate_path": "..."
  },
  "requires_explicit_approval": false,
  "next_action": "primary_agent_review"
}
```

The existing validator status names remain compatible: `PLAN_READY` at the agent
orchestration layer corresponds to a validator-clean `PATCH_PROPOSAL_READY` or
`CONFIG_ONLY_CHANGE` plan.

## Failure discipline

- No effect, worse diff, non-finite output or activation failure: reject/rollback the
  candidate and keep the evidence.
- No new evidence-backed alternative: return `NO_SAFE_FIX`; do not tune indefinitely.
- A candidate that resolves several member occurrences still needs one real-path exact
  check per declared verification boundary.
- Failed hypotheses stay session-local and never enter the verified casebook.
