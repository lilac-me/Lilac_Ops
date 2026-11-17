# Verified Fix Casebook

Use this casebook to accumulate compact, model-specific train/infer alignment lessons
without turning one successful model patch into a universal template.

## Contents

- [Lifecycle](#lifecycle)
- [Promotion gate](#promotion-gate)
- [Case contents](#case-contents)
- [Privacy and storage](#privacy-and-storage)
- [Search before proposing](#search-before-proposing)
- [Record after success](#record-after-success)
- [Generalization policy](#generalization-policy)
- [Failure lessons](#failure-lessons)

## Lifecycle

```text
Module 4 proves first divergence
  -> Module 5 searches casebook for analogies
  -> Module 5 re-derives one current-model plan
  -> Module 3 applies and verifies target pair
  -> write session-local case_candidate.json
  -> later target boundary is native exact and final prefill/regression checks are recorded
  -> Module 6 exports sanitized verified_fix_case.json
  -> a repository maintainer reviews and explicitly promotes it to references/fix-cases/
```

Do not add candidates or failed hypotheses to the global casebook. Keep them in the
session evidence until overall success or later review.

## Promotion gate

Promote a contributing fix only when all are true:

1. Module 4 proved the semantic first divergence and recorded evidence hashes.
2. The plan was re-derived for the current model and did not copy historical code.
3. The targeted pair reached same shape/dtype, no non-finite values,
   `torch.equal == True`, and `max_abs_diff == 0`.
4. Downstream comparison continued and the fix remained active.
5. Supported model/phase/request guards showed no known regression.
6. Final native/conditional prefill truth labels and any fresh E2E metric are recorded.
7. The target repair itself does not rely on replay; any unresolved upstream replay is
   listed as an observation condition and limitation.
8. Repository baselines/tested revisions and packaged patch artifact are recorded.

If the target boundary never becomes native exact, keep a candidate artifact but do not
promote it. A session-wide nonzero E2E metric caused by a separately documented unresolved
boundary does not automatically invalidate an independently native-exact fix.

## Case contents

Store one compact JSON record per verified fix. Required conceptual fields:

```json
{
  "schema_version": 2,
  "lifecycle": "verified",
  "title": "<short semantic lesson>",
  "root_cause_class": "<Module-4 class>",
  "pattern": "<model-agnostic pattern card>",
  "semantic_boundary": "<architecture-level boundary>",
  "model": {
    "family": "<family>",
    "architecture": "<dense/MoE + attention structure>",
    "revision": "<model/config revision>",
    "attention_type": "<MHA/GQA/MLA/etc>",
    "structure": "<dense/MoE and relevant details>",
    "quantization": "<none/type>"
  },
  "topology": {"tp": 1, "pp": 1, "ep": 1, "notes": "<rank mapping>"},
  "diagnosis": {
    "first_divergence_proof": "<compact proof>",
    "evidence_hashes": ["sha256:<hash>"]
  },
  "repair": {
    "method": "<config/runtime/source/reshard>",
    "mathematical_delta": "<semantic change, no full code body>",
    "interface_invariants": ["<preserved contract>"],
    "applicability_guards": ["<model/backend/phase guard>"],
    "implementation_summary": "<short summary>",
    "rederived_for_current_model": true,
    "copied_historical_patch": false
  },
  "verification": {
    "target_native_exact": true,
    "same_shape": true,
    "same_dtype": true,
    "no_nonfinite": true,
    "torch_equal": true,
    "max_abs_diff": 0,
    "downstream_checked": true,
    "upstream_replay_ids": ["<unresolved upstream condition, or empty>"],
    "final_native_prefill_status": "exact|nonzero|not_run",
    "final_conditional_prefill_status": "exact|nonzero|not_run",
    "e2e_metric_status": "zero|nonzero|not_run",
    "e2e_metric_name": "<user-declared zero-target difference metric>",
    "e2e_metric_value": 0,
    "e2e_evidence": "sha256:<log-or-snippet-hash>",
    "regressions": []
  },
  "rejected_alternatives": ["<falsified hypothesis + evidence summary>"],
  "provenance": {
    "repositories": [{"name": "<repo>", "baseline": "<rev>", "tested": "<rev>"}],
    "recorded_at_utc": "<ISO-8601>",
    "verified_at_utc": "<ISO-8601>",
    "session_id": "<non-secret ID>",
    "patch_artifact": "<packaged relative path or hash>"
  },
  "storage": {
    "raw_tensor_included": false,
    "raw_dump_included": false,
    "prompt_included": false,
    "full_source_or_patch_body_included": false
  },
  "generalization": {
    "level": "single_model_verified",
    "limitations": ["<known limits>"],
    "independent_case_ids": []
  }
}
```

The casebook tool computes `case_id` and `pattern_fingerprint`; do not invent them.

## Privacy and storage

Never store:

- raw `.pt` tensors or complete `dump_tensor_data` trees;
- complete `dump.json` / `stack.json` contents;
- prompt/token text, credentials, hostnames, usernames, or private absolute paths;
- full source files or full patch bodies;
- environment dumps containing secrets.

Store hashes, repository-relative packaged artifact references, compact mathematical
summaries, architecture facts, and applicability limits. Keep each record small. The
casebook complements packaged patches; it does not duplicate them.

## Search before proposing

Search only after Module 4 has independently classified the current first divergence:

```bash
python scripts/fix_casebook.py search \
  --root-cause "Op path" \
  --pattern "attention" \
  --architecture "GQA"
```

Load only the few returned case files relevant to the current boundary. Treat matches as
historical analogies. Reopen current source, verify current architecture and active path,
and keep `copied_historical_patch=false`.

No casebook match is evidence that a patch applies. A mismatch between the case and
current architecture is a reason to ignore the case.

## Record after success

Validate a session-local candidate at any time:

```bash
python scripts/fix_casebook.py validate --case case_candidate.json
```

After the promotion gate is satisfied, create a sanitized verified record in the session
package. An alignment task stops there and reports promotion as pending; it never modifies
the installed Skill by default.

A repository maintainer may later promote the record from a writable source checkout:

```bash
python scripts/fix_casebook.py add \
  --case verified_fix_case.json \
  --casebook-root <maintainer-checkout>/references/fix-cases \
  --confirm-maintainer-update
```

`add` is idempotent by computed case ID. It requires both an explicit destination and an
explicit maintainer confirmation so an Agent cannot silently write into a shared or
read-only installation. Ordinary tasks only generate the sanitized candidate and do not
call `add`.

## Generalization policy

Every new case starts as `single_model_verified`. Do not label it model-agnostic.

The shared `pattern_fingerprint` lets later runs find independent observations. Only
after multiple verified cases cover materially different model architectures or backends
may a later review create a generalized pattern. Even then, retain applicability limits
and require current-case Module-4 proof.

## Failure lessons

Failed hypotheses may be useful but can poison recommendations if promoted. Keep them in
the session change log. When a final verified case is promoted, add only compact,
evidence-backed rejected alternatives to that case. Never create a recommended global
case whose repair failed, had no effect, introduced a known regression, relied on replay
at its own target boundary, or lacked the exact target gate.
