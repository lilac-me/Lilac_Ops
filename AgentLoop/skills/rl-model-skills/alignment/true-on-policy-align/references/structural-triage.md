# Optional Structural Triage

Use this reference only when Module 4 has enough module/stack/summary metadata to benefit
from pre-triage. Triage is an optional acceleration layer. It must never reduce the set of
cases handled by the exact comparison path.

## Capability monotonicity

Apply these invariants:

1. A case accepted by the original exact pair-manifest path remains accepted.
2. Missing triage-only metadata never becomes `ADAPTER_CAPABILITY_MISSING`.
3. A triage script or mapping failure records `TRIAGE_STATUS=failed` and falls back to
   the original Module 4 path.
4. Candidate rank never changes semantic execution order.
5. Statistics, names and structural hints never prove equality or a root cause.
6. Only the existing Module 4 exact gates can create a Difference Packet.

Use:

```text
TRIAGE_MODE=auto|off
TRIAGE_STATUS=complete|partial|skipped|failed
TRIAGE_REASON=
```

Default to `auto`. Select `skipped` when an explicit pair manifest already provides a
clear exact-comparison path or when only raw Tensor pairs are available. `off` is an
explicit diagnostic fallback and does not change truth gates.

## Inputs and artifacts

Triage consumes two provider-neutral JSON files:

- `semantic_mapping.json`: ordered semantic train/infer relationships;
- `triage_observations.json`: input/parameter/output/auxiliary observations, unmatched
  nodes, structural hints and hashes.

Start from:

- [templates/semantic_mapping.json](templates/semantic_mapping.json)
- [templates/triage_observations.json](templates/triage_observations.json)
- [templates/structural_triage.json](templates/structural_triage.json)

Validate and run:

```bash
python scripts/validate_semantic_mapping.py \
  --mapping semantic_mapping.json

python scripts/triage_module_artifacts.py \
  --mapping semantic_mapping.json \
  --observations triage_observations.json \
  --expect-input-hash train_artifacts=sha256:<current-train-digest> \
  --expect-input-hash infer_artifacts=sha256:<current-infer-digest> \
  --output structural_triage.json
```

The mapping validator prints `mapping_file_sha256`; record that value in the observation
file. The observation file binds itself to the current mapping SHA256, and the output
records both input hashes. Stale evidence must fail validation instead of silently
reusing an old report. When Module 2 has current artifact digests, pass them through
`--expect-input-hash`; a mismatch makes triage fail-soft and routes back to exact
discovery.

## Semantic mapping rules

Support `1:1`, `1:N` and `N:1` mappings. Do not force fused and split implementations
into physical one-to-one pairs.

Every mapping records:

```text
mapping_id
semantic_node
relation
execution_order
execution_order_evidence.kind/detail
tensor_roles
train endpoints
infer endpoints
semantic_basis
confidence
optional structural_relation
```

Derive execution order from call indices, runtime sequence, stack evidence, source
control flow or proved semantic dependencies. JSON, dictionary or mapping-file insertion
order is not execution evidence.

Transforms and fusion relationships need a source/stack/model-semantic basis. Equal
element counts and similar names are insufficient.

## Observation states

`exact_tensor` observations may produce:

```text
exact_equal
exact_difference
shape_mismatch
dtype_mismatch
nonfinite
```

Statistics and metadata may produce only triage states:

```text
statistics_same
statistics_difference
metadata_only
unknown
missing_train
missing_infer
missing_both
```

`statistics_same` never means aligned. Missing parameter evidence never means a
parameter mismatch. Partial kwargs or missing causal inputs never mean exact input.

## Candidate analysis

Use output-level evidence before internal-only evidence. For a differing output:

- exact causal input and exact parameter suggest an in-module implementation candidate;
- exact input and differing parameter suggest a parameter/checkpoint candidate;
- differing input suggests upstream propagation;
- unmatched nodes after the previous exact output boundary suggest a missing/extra
  operation candidate;
- fusion/split or opaque-boundary hints suggest a structural false-positive candidate.

These remain hypotheses. Operator-name terms such as `silu`, `swiglu`, `activation` or
`qkv` may help a human inspect source, but cannot trigger a backend/model conclusion.

If a module output is exactly equal while an internal observation differs, list it under
`excluded_candidates` for the current declared output coverage. Do not erase the
internal evidence; it may matter under wider coverage.

The previous boundary may be called exact only when backed by exact Tensor evidence.
Never promote statistics-aligned output to an exact boundary.

## Ranking versus coverage order

`structural_triage.json` contains two distinct orders:

- `coverage_order`: the semantic execution order used by Module 4;
- `candidates`: an investigative priority used for mapping review, narrow re-dump and
  explanation.

Module 4 must walk `coverage_order` when proving the first divergence. A later
high-priority candidate cannot leap over an earlier unproved boundary.

## Handoff to exact discovery

Use triage output to:

- refine semantic mappings;
- request the smallest missing input/parameter/output evidence;
- inspect structural fusion/split relationships;
- explain kept and excluded candidates;
- prioritize source inspection.

Then run the existing exact comparator and first-divergence proof. Triage cannot:

- mark declared coverage exact;
- freeze the discovery registry;
- activate replay;
- create or validate a Difference Packet;
- classify a repair as resolved.

Pass the triage status, artifact paths, candidate/exclusion summaries and next checks to
Module 6. Preserve `truth_role=triage_only` in every report.
