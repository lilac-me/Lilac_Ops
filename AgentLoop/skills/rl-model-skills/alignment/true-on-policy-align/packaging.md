# Module 6: Reporting and Packaging

Run after the requested discovery/repair work stops at a truthful terminal state. Module
6 is required for success, partial success, unresolved replay boundaries and user abort.

Do not package diagnostic replay code/Tensors as production fixes.

## Outputs

Use one structured source and generate four synchronized views:

```text
reports/
  alignment_report.json       # source of truth
  alignment_summary.txt       # compact UTF-8 terminal view
  alignment_report.md         # repository/user detailed report
  alignment_report.html       # responsive self-contained Windows view
```

Initialize a new fact source from the Skill-owned template:

```bash
python scripts/generate_alignment_report.py \
  --init-source report_source.json
```

Fill the initialized source with this session's facts and evidence. Never create a new
schema ad hoc and never use an old case report as the template. Existing files are not
overwritten unless `--force` is supplied explicitly.

Then generate all views:

```bash
python scripts/generate_alignment_report.py \
  --input report_source.json \
  --output-dir reports
```

The generator must not invent missing evidence. Use `unknown`, `not_run` or an explicit
limitation.

## Language and layout

Module 6 reports are Chinese-first:

- Every new source starts from
  `references/templates/alignment_report_source.zh-CN.json`; generator and template
  changes must be covered by regression tests before publication.
- TXT, Markdown and HTML use Simplified Chinese for all human-facing titles, labels,
  statuses, explanations and case narratives.
- JSON keeps stable English schema keys and enum values so validators and downstream
  automation remain compatible; human narrative values are Simplified Chinese.
- The UTF-8 TXT view wraps by terminal display width, including double-width Chinese
  characters, and no generated line may exceed 96 display columns.
- The Markdown summary table contains only short comparable fields. Long summaries,
  evidence, repair results and limitations belong in separate detail sections.
- HTML uses `lang="zh-CN"`, responsive cards and tables, wraps long paths, and opens
  only the first difference by default so a long report remains scannable.

`report_source.json` may include an optional `triage` object with
`truth_role=triage_only`, status, ranked candidates, structural false-positive
candidates, excluded candidates and next checks. Omission preserves the original report
path. Triage content must never change native/conditional truth labels.

It may also include `metric_progression` with:

```text
metric_name
truth_role=e2e_impact_only
headline_stage_id
evidence
note
entries[].stage_id
entries[].label
entries[].run_scope=standard_training|minimal_one_layer|full_rollout
entries[].step
entries[].value
entries[].related_difference_ids
```

Use this block for staged impact measurements. Mark cumulative changes explicitly and
keep different run scopes separate. The headline stage must match
`results.e2e_metric_name` and `results.e2e_metric_value`. User-supplied values may be
published when attributed as such, but disclose when raw logs are not included.

For every new stateful/long-horizon task also fill:

```text
acceptance_contract                 # immutable final objective and required gates
verification_axes                   # Tensor, weight, stateful, horizon, transfer/final,
                                    # patch hygiene, clean runtime and performance
validation_matrix.entries[]         # one evidence-backed row per validation rung/run
production_alignment_profile        # production features only; diagnostics false
clean_run_gate                       # fresh process/weights and activation evidence
```

A passing validation row requires `complete=true`, all expected steps observed,
`max_abs_diff=0`, `nonzero_count=0`, `nonfinite_count=0`, and traversal beyond a declared
historical failure step. Record incomplete and failed runs instead of omitting them. For
each resolved difference retain `discovery_max_abs_diff` and
`verification_max_abs_diff=0`; do not erase the pre-fix value.
The generator rejects `acceptance_contract.status=pass` unless every
`acceptance_contract.required_gates` value has a passing validation row.

## Truth labels

Keep these separate:

```text
native_prefill_status=exact|nonzero|not_run|evidence_missing
conditional_prefill_status=exact|nonzero|not_run|evidence_missing
unresolved_replay_ids=[]
e2e_metric_status=zero|nonzero|not_run|evidence_missing
```

- `prefill_native_exact`: no replay enabled and all declared prefill pairs exact.
- `prefill_exact_after_unresolved_replay`: only declared unresolved replays enabled and
  all downstream pairs exact.
- `evidence_missing`: the action may have occurred, but retained artifacts cannot prove
  the required gate. Do not rewrite it as `exact`, `nonzero` or `not_run`.
- A user-declared normal-training difference metric is an E2E impact check. Record its
  name, target and fresh line/value if run; `training/rollout_probs_diff_mean` is one
  example. Do not replace prefill proof with it.

Never write “fully aligned” when an attention/FA or other replay remains active.

The report generator enforces evidence gates:

- `native_prefill_status=exact` requires `native_exact_gate` with shape/dtype/finite/
  `torch_equal`/downstream/real-path activation proof and `max_abs_diff=0`;
- `conditional_prefill_status=exact` requires the same exact gate plus replay activation
  proof and declared `unresolved_replay_ids` unless native prefill is already exact;
- every `resolved` difference requires a real-path `verification_gate`;
- every `unresolved_replayed` difference requires a replay manifest, SHA256 verification,
  activation proof and an exact immediate-downstream input gate;
- `zero`/`nonzero` E2E labels must agree with a numeric metric, while `not_run` and
  `evidence_missing` require a null value.

If retained artifacts cannot fill these fields, use `pending`, `excluded_known_gap` or
`evidence_missing`. A fixed-artifact probe alone is never sufficient.

## Compact summary requirements

The TXT and top of MD/HTML show at a glance:

- model, phase, topology, declared coverage, training backend, inference backend,
  hardware backend and evidence provider;
- native/conditional/E2E result;
- count of `resolved`, `unresolved_replayed`, `excluded_known_gap`,
  `rejected_hypothesis` and `pending`;
- one row per independent Difference Packet;
- optional triage status and candidate/exclusion counts, clearly separated from proved
  Difference Packets;
- total wall time, discovery time, repair-wave wall time, summed agent time,
  integration time, accelerator time and run count;
- patch/report locations and key limitations.

## Detailed difference requirements

For every difference include:

```text
diff_id and Repair Unit
native or conditional observation
semantic boundary and execution order
root-cause class/confidence
exact proof and max_abs_diff
upstream replay dependency IDs
train/infer active source paths and symbols
short code excerpts or path-scoped code diff
repair hypothesis and actual implemented change
probe result and active-worker/native result
rejected attempts and why
replay boundary/hash metadata without Tensor values
status and limitations
time spent
```

When triage ran, add a separate section containing:

```text
triage status and input hashes
coverage order versus investigative candidate rank
previous exact boundary for each candidate
unmatched train/infer nodes between boundaries
structural false-positive candidates
kept and excluded candidates with reasons
alternative hypotheses and next checks
formal exact verification still required
```

Code excerpts must be short enough to review and must not contain credentials, prompts,
private host paths or full copyrighted source files. A path-scoped patch artifact may be
linked separately.

## Timing model

Record ISO timestamps and numeric seconds:

```text
session_started_at
session_ended_at
total_wall_seconds                 # null/unknown unless actual session start/end were recorded
minimum_observed_window_seconds    # optional lower bound from retained artifact timestamps
module_1_seconds ... module_6_seconds
discovery_seconds
repair_wave_wall_seconds
summed_repair_agent_seconds
repair_queue_wait_seconds
integration_seconds
accelerator_run_seconds
accelerator_run_count
```

If reporting estimated parallel saving, show the formula:

```text
estimated_parallel_saving_seconds =
  summed_repair_agent_seconds - repair_wave_wall_seconds
```

Do not claim savings when agent durations overlap unknown work or fields are missing.

## Patch packaging

For each target repository:

1. record baseline/tested revision and pre-existing dirty paths;
2. list only files changed by this session;
3. run `git diff --check`;
4. create a path-scoped source patch or guarded runtime module;
5. include complete rollback/applicability notes;
6. test `git apply --check` in a matching clean worktree when available.
7. audit the added lines and clean runtime:

```bash
python scripts/validate_production_patch.py \
  --patch package/<repo>.patch \
  --env-json runs/final/clean_env.json \
  --profile-json runs/final/alignment_profile.json \
  --require-clean-runtime \
  --output package/<repo>.production_audit.json
```

The default audit fails on diagnostic capture/replay/forced-response/fingerprint
controls, forward hooks, Tensor writes, replay code, private paths, credentials and
private hosts in added lines. A rare legitimate match requires a narrow reviewed
`--allow-pattern`; retain the allowed finding in the audit rather than weakening the
global rule.

Exclude:

- replay adapters and `replay_chain.json` from production patches;
- raw `.pt`, full dump trees, prompts/tokens, credentials, hosts and private absolute
  paths;
- failed candidate patches from the recommended apply instructions.

Create patches from explicit repository/file allowlists. Do not build a multi-repository
patch from the parent workspace or include untracked session artifacts. Record for each
patch: repository root, baseline/tested revision, included paths, patch SHA256,
`git diff --check`, `git apply --check`, production audit and rollback.

Failed/rejected candidates may be listed by hash and summary in the report.

## Public case sanitization

For a public GitHub case, preserve:

- model architecture and public repository revisions;
- declared training/inference/hardware/evidence backends;
- semantic difference, code-level formula/diff excerpt and result status;
- Tensor/dump/report hashes where useful;
- time and workflow statistics;
- truthful scope/limitations.

Remove:

- raw Tensor/dump/prompt data;
- IPs, users, passwords, container-private paths and proprietary dataset details;
- statements not backed by retained evidence.

Build the public package only from an explicit allowlist under `package/`, scan the
result for the same private/credential patterns, and emit a SHA256 manifest. When files
move between remote and local machines, compare the two manifests before publication.
Keep `runs/`, replay payloads and raw discovery artifacts outside the public allowlist.

```bash
python scripts/build_public_package.py build \
  --source-root alignment-session \
  --allowlist public_package_allowlist.json \
  --output-dir alignment-session/package/public

python scripts/build_public_package.py verify \
  --package-dir alignment-session/package/public \
  --manifest alignment-session/package/public/sha256_manifest.json
```

Run `verify` on both the remote and local copy; both must pass the same manifest. The
builder refuses an existing output directory, path escape, symlink, raw Tensor/model
suffix, oversized/non-UTF-8 file, private host/path or credential-like content.

An unresolved FA boundary may be included as a useful result. Label downstream results
as conditional on its replay/exclusion.

## Casebook promotion

Use [references/fix-casebook.md](references/fix-casebook.md). Promote an individual fix
only when its target boundary is native exact, downstream comparison continued and the
change survived final prefill/regression checks. A session may contain unresolved replay
boundaries while still promoting an unrelated, independently native-exact fix, provided
the case records its upstream observation conditions and limitations.

During an alignment task, export a sanitized promotion candidate and report its pending
status. Do not write into the installed Skill. Repository maintainers may promote the
candidate later through the explicit maintenance command in the casebook reference.
Do not promote replay-only, probe-only, failed or no-effect changes.

## Required final modification summary

```markdown
## Outcome
- native prefill: <exact|nonzero|not_run|evidence_missing>
- conditional prefill: <exact|nonzero|not_run|evidence_missing> under replay IDs <...>
- E2E <metric_name>: <value|not_run|evidence_missing>
- weight/stateful/long-horizon/transfer/final/clean-runtime/performance: <status + evidence>

## Differences
1. <D001 boundary — status — short result>

## Modifications
1. <repo-relative file>::<symbol> — <why>

## Timing
- total wall: ...
- discovery / repair agents / integration / accelerator: ...

## Artifacts
- JSON/TXT/MD/HTML: <paths>
- patches: <paths>
- apply check: <result>
- production hygiene and clean-runtime audit: <paths + result>
- validation matrix: <path>
- casebook: <IDs or pending reason>
```

## Cleanup

After reports and patches are verified, clean only recorded session temp files and old
approved dump outputs. Keep the final reports, selected patches, frozen Difference
Packets, selected/rejected plan summaries and user-requested evidence. Raw replay Tensors
remain private and may be deleted only under the approved evidence-retention policy.
