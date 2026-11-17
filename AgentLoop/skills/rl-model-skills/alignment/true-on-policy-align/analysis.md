# Module 4: Prefill Difference Discovery

Act as the read-only diagnostic subroutine called by Module 3. Consume one comparable,
provider-neutral train/infer artifact pair from Module 2's minimized evidence case and
return either a proved first semantic prefill divergence, exact declared coverage, or an
executable narrow re-dump request.

Do not start/stop jobs, clean dumps, edit model code, activate replay, or propose a fix.

## Inputs

Require and never invent:

```text
TRAIN_DUMP_JSON=
INFER_DUMP_JSON=
TRAIN_STACK_JSON=
INFER_STACK_JSON=
TRAIN_SOURCE_REVISION=
INFER_SOURCE_REVISION=
MODEL_AND_WEIGHT_DIGESTS=
SAMPLE_AND_TOKEN_PROVENANCE=
PROMPT_TOKEN_LENGTH=
TRAIN_PROMPT_SLICE=
LOGIT_CAUSAL_SHIFT=
TRAIN_AND_INFER_CONFIG_DIGESTS=
STEP_CALL_RANGE_AND_CREATED_TIME=
TOPOLOGY=tp1_pp1_ep1
UPSTREAM_REPLAY_IDS=[]
REPLAY_ACTIVATION_EVIDENCE=[]
OPTIONAL_SEMANTIC_MAPPING=
OPTIONAL_TRIAGE_OBSERVATIONS=
OPTIONAL_STATE_MANIFEST=
OPTIONAL_FIRST_FAILURE_TRIGGER=
```

For first-divergence discovery, Module 2 is expected to reduce TP to 1. If the artifact
pair is not TP=1, return `NOT_COMPARABLE` and ask Module 3 to complete the minimal-case
reduction. This does not reject the user's original topology, which belongs to transfer
and final validation. Do not add TP reconstruction analysis.

## Statuses

| Status | Route |
| --- | --- |
| `EXACT_ALIGNED` | Declared manifest coverage is exact; Module 3 may widen/finish discovery |
| `FIRST_DIVERGENCE_FOUND` | Difference Packet is complete and replay-ready |
| `STATE_BOUNDARY_FOUND` | Weight/reload/state/control-plane Difference Packet v2 is complete |
| `NEED_NARROW_REDUMP` | Execute the exact returned capture request |
| `NOT_COMPARABLE` | Repair provenance, prompt slicing or semantic mapping |
| `MISSING_TENSOR` | Re-dump required `.pt` payloads |
| `AMBIGUOUS_PAIR` | Inspect source/stack or capture a closer boundary |
| `INVALID_DUMP` | Recollect a fresh valid case |
| `NONFINITE_TENSOR` | Localize the first NaN/Inf origin |
| `REPLAY_PREFIX_INVALID` | An upstream replay was not activated/exact; stop downstream conclusions |

Only `FIRST_DIVERGENCE_FOUND` or `STATE_BOUNDARY_FOUND` creates a Difference Packet.
Module 5 is not called during the discovery sweep.

## State machine

```text
VALIDATE_PROVENANCE
  -> VALIDATE_REPLAY_PREFIX
  -> INVENTORY
  -> OPTIONAL_STRUCTURAL_TRIAGE
  -> BUILD_EXPLICIT_SEMANTIC_MANIFEST
  -> LOAD_AND_NORMALIZE
  -> EXACT_COMPARE
  -> PROVE_FIRST_DIVERGENCE
     OR PROVE_FIRST_STATE_BOUNDARY
  -> LOCATE_ACTIVE_SOURCE
  -> CLASSIFY
  -> WRITE_DIFFERENCE_PACKET
```

Statistics only screen Tensor candidates. Exact Tensor evidence is mandatory for
`boundary_kind=tensor`; non-Tensor boundaries require immutable before/after state
manifests, event provenance and a positive downstream impact proof.

## Optional structural triage

Read [references/structural-triage.md](references/structural-triage.md) when stack,
module, summary or structural metadata can help build mappings or narrow a re-dump.
Default to `TRIAGE_MODE=auto`.

Triage is optional:

- if inputs are sufficient, validate `semantic_mapping.json` and generate
  `structural_triage.json`;
- if inputs are absent, mark `TRIAGE_STATUS=skipped`;
- if validation or analysis fails, mark `TRIAGE_STATUS=failed`, retain the error and
  continue the original exact pair-manifest flow;
- never return `NOT_COMPARABLE` only because triage-only metadata is missing.

Use `coverage_order` for exact comparison. Candidate rank is only an investigative
priority for mapping inspection, narrow re-dump and explanation. It cannot skip an
earlier unproved semantic boundary, establish an exact prefix, create a Difference
Packet or freeze discovery.

## Comparability and prompt slicing

Prove the same:

1. model/checkpoint/weight conversion;
2. sample, prompt token IDs, positions and effective config;
3. prefill call and semantic step;
4. topology matches the minimized evidence contract, normally TP=1 for TP isolation;
5. dump creation after the recorded launch;
6. declared upstream replay chain.

Training commonly consumes `prompt + response` while rollout prefill consumes `prompt`.
Do not change the data path. Use manifest transforms backed by recorded token provenance,
for example:

```json
{
  "train": {
    "selector": "output.0",
    "transforms": ["select:0,0", "slice:0,0,196"],
    "semantic_basis": "training length 197 contains the same 196-token prompt plus one response token"
  }
}
```

For logits/logprob, separately prove which position predicts the selected next token.
Never reuse an activation slice blindly at a causally shifted output.

## Semantic pairing

Use evidence in this priority:

1. active source and `stack.json`;
2. model semantics, layer identity and tensor role;
3. shape/dtype and execution neighborhood;
4. name similarity only as a hint.

One train operator may correspond to an infer operator chain. Pair shared semantic
boundaries rather than forcing physical one-to-one matches. Repeated calls need exact
layer/call identity.

Every ordered manifest pair records:

```text
name
execution_order
semantic_node
tensor_role
phase=prefill
train key/selector/transforms/semantic_basis
infer key/selector/transforms/semantic_basis
origin=native|conditional
required_upstream_replays
```

An optional semantic mapping may describe `1:1`, `1:N` or `N:1` module relationships,
but the exact pair manifest still names the concrete normalized Tensor boundary to
compare. Derive execution order from runtime/stack/source/semantic-dependency evidence,
never JSON insertion order.

The comparison helper supports explicit `squeeze`, `reshape`, `permute`, `transpose`,
`flatten`, `slice`, `select` and `contiguous`. Each transform needs a semantic basis.
Do not choose a transform because element counts happen to match.

## Exact comparison

When `EVIDENCE_PROVIDER=msprobe`, run:

```bash
python scripts/compare_msprobe_dumps.py compare \
  --manifest pair_manifest.json \
  --output compare_report.json
```

For any other provider, use its manifest-native comparator or a session-local converter
that emits the same exact pair result fields. Do not convert a backend choice into a
truth-gate difference.

Exact means:

```text
same shape
same dtype
no NaN or Inf
torch.equal == True
max_abs_diff == 0
```

`allclose`, equal Norm/Mean/Max or a guessed reshape is not evidence.

## Replay prefix gate

When `UPSTREAM_REPLAY_IDS` is non-empty, require an activation record for every ID:

```text
hook_active_exactly_once=true
loaded_hash_matches=true
post_injection_torch_equal=true
post_injection_max_abs_diff=0
immediate_downstream_input_exact=true
```

Missing or failed evidence returns `REPLAY_PREFIX_INVALID`. A downstream difference
found after an invalid replay cannot enter the registry.

## Prove an independent first divergence

All must hold:

1. manifest starts at a declared exact/native boundary or a verified replay boundary;
2. every earlier semantic output in coverage is exact;
3. every causal input of the candidate is present and exact;
4. output is finite, same shape/dtype and has `max_abs_diff > 0`;
5. residual, bias, mask, position, branch input and auxiliary state are not omitted;
6. active train and infer source/stack frames are known.

If an earlier available boundary already differs, return `NEED_NARROW_REDUMP`; call it an
earliest observed candidate, not the true first divergence.

Do not record downstream manifestations of the same upstream error. The discovery loop
replays the proved boundary before searching again.

When static Tensor coverage is exact but a late run fails, do not fabricate a Tensor
first divergence. Prove the earliest weight reload, runtime-state or control-plane
transition whose before/after state differs and whose downstream impact is nonzero. Use
the same fixed model/config/sample, aligned prefix, finite-value and active-source rules,
then emit Difference Packet v2 with a non-Tensor `boundary_kind`. Capture operator call
context (`M/N/K`, padding, kernel/fusion, layout and dtype), compile/cache, prefix-cache,
task queue, worker generation and weight version.

## Root-cause classes

Use one primary class and record remaining alternatives:

| Class | Minimum evidence |
| --- | --- |
| `Op path` | Exact inputs; differing output; active paths show different kernel/formula/order |
| `Input precision` | First difference is a cast, scale, dtype, layout or accumulation boundary |
| `Config` | Active branches are known and effective values differ |
| `Missing step` | One path omits/adds a causal operation |
| `Framework bug` | Checkpoint agrees but load/conversion/framework state changes semantics |
| `Test setup mismatch` | Provenance differs before mathematical comparison |
| `Weight sync` | Actor/rollout parameter manifests or reload generations diverge |
| `State transition` | Cold/warm/update/reload state first changes downstream semantics |
| `Control plane` | Worker, queue, cache or request routing changes the active computation |

Confidence is `CONFIRMED`, `SUPPORTED` or `CANDIDATE`. A Difference Packet may be
replay-ready with a bounded `CANDIDATE` cause, but Module 5 later requires
`CONFIRMED|SUPPORTED` before a code repair plan.

## Replayability analysis

Analysis does not activate replay, but it must define the safest candidate boundary:

- enumerate all output fields and auxiliary returns;
- identify mutated buffers or state visible downstream;
- state whether the boundary is pure enough to replace;
- if an internal fused/native boundary is unsafe, name the nearest outer semantic
  boundary expected to cover all effects;
- specify shape, dtype, token slice, layer, symbol and call index guards.

Return `replay.recommended_boundary` and `replay.side_effects_to_cover`. Module 3 verifies
the actual replay contract.

## Difference Packet

Write `difference.json` and validate it:

```bash
python scripts/validate_difference_packet.py --packet difference.json
```

Difference Packet v1 remains valid for existing Tensor cases. New packets use v2 and
declare `boundary_kind=tensor|weight_reload|state_transition|control_plane`. A Tensor
packet keeps the exact-input/nonzero-output proof below. A non-Tensor packet instead
requires recorded source/target state, different before/after SHA256 state digests, an
observed transition and a positive finite downstream impact metric. Use
`references/templates/state_difference_packet.json` for that form.

Conceptual Tensor schema:

```json
{
  "schema_version": 2,
  "diff_id": "D001",
  "phase": "prefill",
  "boundary_kind": "tensor",
  "execution_order": 10,
  "observation": "native",
  "semantic_boundary": "decoder.layer0.attention.context",
  "upstream_replay_ids": [],
  "proof": {
    "aligned_prefix": true,
    "all_causal_inputs_exact": true,
    "same_shape": true,
    "same_dtype": true,
    "no_nonfinite": true,
    "output_nonzero": true,
    "max_abs_diff": 0.000244140625
  },
  "root_cause": {"class": "Op path", "confidence": "CONFIRMED"},
  "source": {"train": "...", "infer": "..."},
  "evidence": {
    "manifest": "pair_manifest.json",
    "compare_report": "compare_report.json",
    "train_stack": "stack.train.json",
    "infer_stack": "stack.infer.json",
    "tensor_hashes": ["sha256:..."]
  },
  "replay": {
    "recommended_boundary": "decoder.layer0.attention.output",
    "side_effects_to_cover": ["context output"],
    "status": "proposed"
  },
  "repair": {"group_hint": "attention-prefill", "status": "not_started"}
}
```

## Narrow re-dump request

Return executable fields:

```text
REDUMP_REASON=
REDUMP_PHASE=prefill
REDUMP_SEMANTIC_BOUNDARY=
REDUMP_TRAIN_KEYS_OR_SCOPE=
REDUMP_INFER_KEYS_OR_SCOPE=
REDUMP_REQUIRED_ROLES=input|output|parameter|aux_state
REDUMP_MODE=statistics|tensor|stack|all
REDUMP_UPSTREAM_NEIGHBORS=
REDUMP_DOWNSTREAM_NEIGHBORS=
REDUMP_CUSTOM_API_TARGET=none|module.api
REDUMP_KEEP_FROM_CURRENT=
REDUMP_SUCCESS_GATE=
```

## Output contract

```text
ANALYSIS_STATUS=
ANALYSIS_COVERAGE=
ANALYSIS_MANIFEST=
ANALYSIS_COMPARE_REPORT=
ANALYSIS_FIRST_DIVERGENCE=
ANALYSIS_BOUNDARY_KIND=tensor|weight_reload|state_transition|control_plane
ANALYSIS_FIRST_DIVERGENCE_PROOF=
ANALYSIS_MAX_ABS_DIFF=
ANALYSIS_TRAIN_STACK=
ANALYSIS_INFER_STACK=
ANALYSIS_TRAIN_SOURCE=
ANALYSIS_INFER_SOURCE=
ANALYSIS_ROOT_CAUSE_CLASS=
ANALYSIS_ROOT_CAUSE_CONFIDENCE=
ANALYSIS_ALTERNATIVES=
ANALYSIS_UPSTREAM_REPLAY_IDS=
ANALYSIS_DIFFERENCE_PACKET=
ANALYSIS_REPLAY_BOUNDARY=
ANALYSIS_REDUMP_REQUEST=
ANALYSIS_TRIAGE_STATUS=
ANALYSIS_TRIAGE_ARTIFACT=
ANALYSIS_TRIAGE_LIMITATION=
ANALYSIS_STATE_MANIFEST=
ANALYSIS_OPERATOR_CALL_CONTEXT=
ANALYSIS_NEXT_ACTION=freeze_difference|continue_downstream|redump|repair_case
```

Module 4 closes for one invocation when this record and its evidence are complete. It
never declares native prefill or overall task success.
