# Prefill Verification Probe

Build a case-specific micro-probe for one frozen Repair Unit. It accelerates candidate
development; it does not replace active-worker prefill verification.

## Trust sequence

The probe must answer in order:

1. Can the captured case reproduce the original training output exactly?
2. Can it reproduce the original inference output exactly?
3. Is the original nonzero difference still present?
4. Does one candidate reproduce the training output exactly?

Never judge question 4 until 1～3 pass.

## Modes

| Mode | Use when |
| --- | --- |
| `single_process_ab` | Train/infer callables coexist and all prefill state is explicit |
| `dual_process_replay` | Environments or registrations are incompatible |
| `in_situ_targeted` | Worker load order, native dispatch or hidden state cannot be isolated |
| `not_applicable` | No probe can preserve the causal state; primary agent uses a narrow real-path gate |

This Skill does not build decode/KV-cache or TP-distributed probes.

## Capture the minimum real case

Keep only:

- exact causal input Tensor(s) and SHA256 hashes;
- weights, bias, mask, position IDs, scales or auxiliary state that affects the boundary;
- original train and infer outputs;
- dtype, layout, prompt slice and causal-shift transform IDs;
- effective config and active source revisions;
- the upstream replay IDs under which the difference was observed.

Use actual captured artifacts, not random tensors. Load only session-created tensor files
with `torch.load(path, map_location="cpu", weights_only=True)`. If the installed PyTorch
does not support the safe option, stop and request a compatible loader rather than
silently falling back to unrestricted pickle loading.

## Adapter contract

```python
def probe_metadata() -> dict:
    return {
        "case_id": "R001.post_attention_rmsnorm",
        "repair_unit_id": "R001",
        "member_diff_ids": ["D002"],
        "phase": "prefill",
        "model": {"family": "...", "architecture": "...", "revision": "..."},
        "topology": {"tp": 1, "pp": 1, "ep": 1, "rank": 0},
        "source_revisions": {"train": "...", "infer": "..."},
        "input_hashes": ["sha256:..."],
        "upstream_replay_ids": ["D001"],
        "state_dependencies": ["residual", "epsilon"],
        "semantic_transforms": ["prompt slice ID"],
        "applicability_limits": [
            "prefill boundary",
            "TP collective behavior not covered; requested topology requires final validation",
        ],
    }

def load_probe_case(case_dir):
    ...

def load_expected_outputs(case):
    return {"train": ..., "infer": ...}

def run_train_reference(case):
    ...

def run_infer_baseline(case):
    ...

def run_infer_candidate(case):
    ...
```

The candidate must implement the proposed mathematics. Returning the saved training
Tensor is diagnostic replay and cannot validate a repair.

## Run

```bash
python scripts/run_verification_probe.py reproduce \
  --adapter /session/R001/probe_adapter.py \
  --case-dir /session/R001/probe_case \
  --report /session/R001/probe_reproduce.json

python scripts/run_verification_probe.py verify \
  --adapter /session/R001/probe_adapter.py \
  --case-dir /session/R001/probe_case \
  --report /session/R001/probe_candidate.json
```

## Gates

`PROBE_REPRODUCED` requires:

```text
train reference equals original train dump
infer baseline equals original infer dump
train reference differs from infer baseline
input/source/config hashes match the Difference Packet
```

`PROBE_CANDIDATE_EXACT` additionally requires:

```text
same_shape=true
same_dtype=true
no_nonfinite=true
torch.equal=true
max_abs_diff=0
```

When reproduction fails, enumerate missing state and use a richer or in-situ probe. Do
not tune a candidate against a non-reproducing harness.

## Real-path handoff

After probe exactness, return the candidate to the primary agent. The primary agent:

1. preflights the canonical worktree and worker load path;
2. applies the candidate;
3. proves all intended workers use it;
4. removes only the target difference's replay entry;
5. collects a narrow real prefill dump;
6. requires the target pair native exact;
7. continues downstream or rolls back.

A probe-exact attention training-operator substitution remains unresolved if the active
worker path does not become exact.

## Evidence and privacy

Retain adapter, compact metadata, hashes, reproduction/candidate reports, selected patch
and rollback record. Keep minimal Tensor payloads only in the private session directory.
Never publish raw Tensors, dumps, prompts, credentials, hostnames or private paths.
