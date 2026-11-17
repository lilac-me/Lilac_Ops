# Backend Adapter Contract

The alignment workflow is backend-agnostic. Treat these as four independent axes:

```text
training_backend       # e.g. Megatron, FSDP, DeepSpeed, JAX, custom
inference_backend      # e.g. vLLM, SGLang, Transformers, TensorRT-LLM, custom
hardware_backend       # e.g. CUDA, ROCm, Ascend, TPU, CPU
evidence_provider      # e.g. framework hooks, custom dump, msprobe, saved artifacts
```

Examples are not an allowlist. A backend is supported when the session can satisfy the
contract below without guessing.

## Required contract

Before Module 2, record:

1. how to launch one deterministic training-side forward and one inference-side prefill;
2. how to fix model/checkpoint, sample, token IDs, config, seed and topology;
3. how to capture named semantic Tensor boundaries, parameters and auxiliary state;
4. how to prove the active source file, symbol, call index and runtime revision;
5. how to inject a guarded training-side replay at a Python-visible inference boundary;
6. how to restart/reload the target runtime and prove the candidate path activated;
7. how to inspect resource ownership and stop only the recorded job;
8. the optional E2E metric name, target and log evidence.
9. how to fingerprint actor/rollout weights and prove publish/reload generations;
10. how to capture cold/warm/reload worker state plus compile, prefix-cache and task-queue
    provenance;
11. how a bounded session-local capture is pre-armed before launch and how the
    post-metric first-failure trigger freezes it without changing production behavior;
12. how to record operator call context, including logical/physical shapes, GEMM M/N/K,
    padding, kernel/fusion, layout, dtype and accumulation dtype.

If any required capability is missing, report it as `ADAPTER_CAPABILITY_MISSING`. Do not
pretend that a framework is supported merely because Tensor files exist.

Structural-triage metadata is optional and is not part of this required adapter
contract. A backend that can satisfy exact Tensor capture and comparison remains
supported when it cannot emit module summaries, unmatched-node inventories or fusion
hints.

For replay, define a safe backend-native serialization/loader that cannot execute
arbitrary code, verifies the payload hash before loading and rejects shape/dtype/context
mismatch. `scripts/tensor_replay.py` and `validate_replay_manifest.py` are the included
PyTorch adapter. Other runtimes must provide equivalent session-local implementations;
never route an untrusted payload through a permissive deserializer.

## Evidence provider choices

Prefer the least invasive provider that preserves exact Tensor values and provenance:

1. existing framework/backend dump API;
2. scoped module/operator hooks;
3. session-local wrapper or monkey patch with activation evidence;
4. pre-existing saved artifacts supplied by the user.

The artifact layout may differ. Normalize it through an explicit pair manifest containing
side, semantic boundary, file, dtype, shape, call index, transforms, source symbol and
hash. `scripts/compare_msprobe_dumps.py` is an optional adapter for msprobe layouts, not a
core dependency. Other providers may emit the same comparison manifest directly or use a
session-local converter.

When available, normalize additional module mapping and summary evidence through the
provider-neutral optional contracts in
[structural-triage.md](structural-triage.md). Never make those contracts mandatory for
the exact pair-manifest path.

## Adapter-specific configuration

Keep backend-specific launch flags, environment variables, operator names and known bugs
in the session record or a dedicated adapter reference. Never place them in the generic
truth gates.

If package installation is authorized, expose an optional sanitized package-mirror
setting. Do not hardcode a vendor mirror in the generic adapter; use the user's or
environment's configured origin, omit credentials/query strings from records, and verify
the installed version/import from the active runtime.

For the included Qwen3 case:

```text
training_backend=Megatron/MindSpeed
inference_backend=vLLM-Ascend
hardware_backend=Ascend NPU
evidence_provider=msprobe plus session-local fixed-artifact probes
```

That case demonstrates the workflow; it does not limit the Skill to those backends.

For an evidence-confirmed Qwen3.5-family Ascend stack, optionally consult
[adapters/qwen35-ascend.md](adapters/qwen35-ascend.md). Its branches are evidence
checklists only and must not be copied as fixes.

## Invariants across adapters

- compare semantics, not filenames or framework module names;
- preserve exact values: no lossy summaries for an exactness claim;
- make prompt slicing and causal shift explicit;
- verify replay hash, activation and immediate downstream input;
- distinguish fixed-artifact probes from real-runtime activation;
- keep diagnostic replay out of production patches;
- retain backend names and revisions in Module 6 provenance.
- keep static Tensor, weight reload, stateful, long-horizon and final-model claims on
  separate truth axes;
- distinguish production alignment features from diagnostic replay/capture controls.
