# Module 2: Minimal Evidence Case

Build the smallest deterministic **prefill** case that emits comparable training-side and
inference-side Tensor evidence. The core workflow is independent of framework, engine,
hardware and dump tool.

For first-divergence discovery, reduce tensor parallelism to TP=1 as one minimization
step so collective and TP-reassembly behavior cannot confound non-collective operator
mathematics. This diagnostic topology is not the Skill's overall scope and does not
replace final validation on the user's requested topology.

Do not diagnose operators, implement `PROMPTS_ONLY`, or claim alignment in this module.

## Exit gate

Complete Module 2 only when:

1. the backend adapter contract is recorded;
2. one source model/checkpoint, immutable diagnostic copy, sample and effective config
   are fixed and fingerprinted on both backends;
3. training and inference both execute the same declared prompt prefill;
4. the diagnostic topology is TP=1 and the sample batch is one;
5. both sides produce exact-value Tensor artifacts with provenance;
6. prompt slice and next-token causal shift are explicit;
7. all artifacts are fresh, parseable and handed to Module 3;
8. the minimization coverage matrix proves that every relevant semantic family remains.

If a backend cannot expose comparable exact Tensors or a safe replay boundary, report
`ADAPTER_CAPABILITY_MISSING` with the missing capability.

## Select the evidence provider

Read [references/backend-adapters.md](references/backend-adapters.md). Prefer an existing
backend dump API, then scoped hooks, then a session-local wrapper. Saved artifacts are
valid for local analysis when their provenance is complete.

`msprobe` is one optional provider for Ascend stacks. It is not required. When it is used,
`scripts/compare_msprobe_dumps.py` can inspect its layout. Other providers may emit the
pair manifest directly or use a session-local converter.

Record:

```text
TRAINING_BACKEND=
INFERENCE_BACKEND=
HARDWARE_BACKEND=
EVIDENCE_PROVIDER=
TRAIN_CAPTURE_ENTRY=
INFER_CAPTURE_ENTRY=
ACTIVE_TRAIN_SOURCE=
ACTIVE_INFER_SOURCE=
```

## Minimal deterministic case

Derive a separate diagnostic launch artifact and immutable model/config copy; do not
overwrite the user baseline or edit the source model directory in place.

- batch/sample count: one;
- topology: TP=1, with other model-parallel dimensions reduced when the backend permits;
- generation: enough to make the training logprob path observable, commonly one response
  token, while the compared phase remains prompt prefill;
- steps: one relevant training/logprob forward;
- data: fixed sample ID/index, prompt tokens, positions and seed;
- execution: eager/non-compiled mode only when the selected evidence provider requires it;
- model reduction: minimize one dimension at a time and preserve every active semantic
  family needed to reproduce the target architecture and failing call-context.

Layer reduction must affect both sides consistently. If the training and inference
backends consume different configs, prove their effective layer counts rather than
changing only one side.

Record `SOURCE_CONFIG_SHA256`, `EFFECTIVE_TRAIN_CONFIG_SHA256`,
`EFFECTIVE_INFER_CONFIG_SHA256`, `CHECKPOINT_MANIFEST_SHA256`, instantiated layer types,
parameter inventory and active call trace. A matching source `config.json` is not proof
that both workers instantiated or loaded the same effective model.

Maintain the semantic coverage matrix in
[references/stateful-validation.md](references/stateful-validation.md). For MoE,
multimodal, recurrent/GDN or mixed-layer models, do not minimize away the relevant
router/expert, vision/projector or layer-type path. The diagnostic minimum is an
intermediate rung; retain a representative reduced model and the requested model for
transfer/final validation.

This Skill does not analyze distributed collectives or TP reassembly. If the backends
cannot run the minimal diagnostic path at TP=1, stop Module 2 with
`NOT_COMPARABLE_TOPOLOGY`; do not relabel the user's requested topology as unsupported.
After repair, validate the candidate on that requested topology.

## Prompt and causal provenance

Do not modify the data path to remove the response. Record:

```text
PROMPT_TOKEN_LENGTH=
TRAIN_INPUT_TOKEN_LENGTH=
INFER_PREFILL_TOKEN_LENGTH=
TRAIN_PROMPT_SLICE=
LOGIT_CAUSAL_SHIFT=
SAMPLE_DIGEST=
TOKEN_DIGEST=
```

Training may contain `prompt + response` while inference prefill contains only prompt.
Module 4 may use only manifest-backed `select`/`slice` transforms. Logits and logprobs
must separately prove the next-token causal offset.

Create a prompt/response/shape coverage matrix for the requested final gate. At minimum
record prompt length, response length, batch, modality, padded shapes and cold/warm/
reload state. A response-length-1 case covers only its declared cell.

## Provider-neutral artifact contract

Each comparable item must identify:

```text
side=train|infer
semantic_boundary=
role=input|output|parameter|aux_state
tensor_file=
sha256=
shape=
dtype=
layer_or_block=
symbol=
call_index=
execution_order=
source_file=
source_revision=
semantic_transforms=[]
```

Capture stack/source evidence when available. Statistical summaries may help triage but
cannot prove exact equality.

The pair manifest, not directory names, defines train/infer correspondence.

When inexpensive and available, also retain provider-emitted module order, input/
parameter/output summaries, unmatched-node metadata and structural fusion hints for the
optional triage described in
[references/structural-triage.md](references/structural-triage.md). These fields are not
part of the Module 2 exit gate. Their absence must not make an otherwise exact-comparable
case unsupported.

## Collection procedure

1. Validate artifact-root and cleanup paths are inside the approved work area.
2. Clear only the prior run outputs; preserve configs and unrelated artifacts.
3. Launch one recorded diagnostic job using Module 1 lifecycle rules.
4. Confirm exactly one job instance and both capture hooks/providers activated.
5. Stop after the declared forward/step.
6. Verify both sides contain fresh non-empty artifacts and metadata.
7. Hash final configs, model/checkpoint identity, sample and Tensor files.
8. Preserve the evidence needed by discovery; clean only recorded session temporaries.

If one side is empty, debug the provider integration for that backend. Do not wait
indefinitely after the smoke forward has completed.

## Handoff

```text
CASE_LAUNCH_ARTIFACT=
ARTIFACT_ROOT=
TRAIN_ARTIFACT_PATH=
INFER_ARTIFACT_PATH=
PAIR_MANIFEST=
OPTIONAL_SEMANTIC_MAPPING=
OPTIONAL_TRIAGE_OBSERVATIONS=
TRAINING_BACKEND=
INFERENCE_BACKEND=
HARDWARE_BACKEND=
EVIDENCE_PROVIDER=
MODEL_DIGEST=
SAMPLE_DIGEST=
CONFIG_DIGEST=
SOURCE_CONFIG_SHA256=
EFFECTIVE_TRAIN_CONFIG_SHA256=
EFFECTIVE_INFER_CONFIG_SHA256=
CHECKPOINT_MANIFEST_SHA256=
IMMUTABLE_DIAGNOSTIC_MODEL=
MINIMIZATION_COVERAGE_MATRIX=
PROMPT_RESPONSE_SHAPE_MATRIX=
PROMPT_TOKEN_LENGTH=
TRAIN_PROMPT_SLICE=
LOGIT_CAUSAL_SHIFT=
RUN_LEADER_PID=
RUN_PGID=
CAPTURE_PATCH=
CAPTURE_PATCH_ROLLBACK=
```

Module 2 success proves comparable evidence exists. It does not prove the paths align.

## Safety

- Do not install a provider without authorization.
- Do not publish raw Tensors, prompts, tokens, credentials or private paths.
- Do not delete a whole worktree or artifact root with a broad glob.
- Do not treat provider instrumentation as a production repair.
