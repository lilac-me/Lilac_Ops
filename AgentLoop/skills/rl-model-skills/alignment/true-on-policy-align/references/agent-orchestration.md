# Repair Agent Orchestration

Use after the prefill discovery registry is frozen. The goal is parallel candidate
development with one authoritative integration path.

## Roles

### Primary orchestrator

- owns authorization, credentials and the state card;
- owns remote accelerator jobs, canonical worktrees, artifacts and replay chain;
- freezes Difference Packets and builds Repair Units;
- starts, monitors and stops repair-agent tasks;
- resolves conflicts and integrates candidates serially;
- runs real-path verification and Module 6 reporting.

### Repair agent

- consumes exactly one immutable Repair Unit;
- independently reviews raw evidence and root-cause classification;
- works only in an assigned isolated worktree or produces a patch artifact;
- returns one plan, probe and candidate or a precise non-ready status;
- never changes shared runtime state.

## Scheduling

Keep one orchestration slot free. Use the remaining available agent capacity as a bounded
worker pool. Do not hardcode an agent count in the Skill.

Build a dependency DAG:

```text
R001 attention (unresolved replay)
  -> R002 post-attention residual/RMSNorm
  -> R003 SwiGLU
  -> R004 logits/logprob
```

`depends_on` means the difference was observed under those upstream replay/fix conditions;
it does not automatically mean the code patches overlap. Start units in parallel only
when:

- their changed-path/symbol forecasts do not overlap;
- neither depends on the other's native integration result;
- they do not need the same mutable isolated test resource.

Discovery remains serial by default because every new independent difference depends on
a verified replay prefix.

## Repair Unit envelope

Give an agent a fixed envelope:

```json
{
  "schema_version": 1,
  "repair_unit_id": "R002",
  "member_diff_ids": ["D002", "D006"],
  "depends_on": ["D001"],
  "baseline_revisions": {"inference_backend": "<sha>"},
  "evidence_files": ["discovery/D002/difference.json"],
  "allowed_worktree": "<isolated absolute path>",
  "allowed_paths": ["inference_backend/ops/layernorm.py"],
  "forbidden_actions": [
    "start_or_stop_accelerator_jobs",
    "edit_canonical_worktree",
    "clean_shared_dump",
    "merge_candidate",
    "declare_real_path_success"
  ],
  "required_outputs": [
    "fix_plan.json",
    "candidate.patch",
    "probe_adapter.py",
    "agent_result.json"
  ]
}
```

Pass raw evidence paths plus the diagnosis. Do not pass an intended patch body as the
answer. The agent must be able to reject the assigned hypothesis.

## Agent result

```json
{
  "repair_unit_id": "R002",
  "status": "PLAN_READY|NEED_EVIDENCE|HYPOTHESIS_REJECTED|NO_SAFE_FIX|DEPENDENCY_REQUIRED|UNSUPPORTED_VERSION",
  "summary": "...",
  "evidence_reviewed": ["..."],
  "changed_paths": ["..."],
  "changed_symbols": ["..."],
  "artifacts": {
    "fix_plan": "fix_plan.json",
    "candidate_patch": "candidate.patch",
    "probe_adapter": "probe_adapter.py",
    "probe_report": "probe_candidate.json"
  },
  "missing_evidence_request": null,
  "integration_risks": ["..."],
  "duration_seconds": 0
}
```

## Collaboration protocol

1. The primary agent announces the frozen registry and Repair Unit IDs.
2. Start one agent per independent unit up to available capacity.
3. Require periodic status only at meaningful gates; do not let agents edit a shared
   status file concurrently.
4. Agents return immutable artifacts to their own directory.
5. The primary agent validates every result and records it in the repair queue.
6. If an agent requests evidence, the primary agent decides whether to reopen a narrow
   discovery capture. The agent does not launch the capture itself.
7. Conflicting candidates are serialized, regrouped or one is rejected.

## Conflict rules

Treat candidates as conflicting when they touch:

- the same file and overlapping symbol;
- the same dispatcher/config selector;
- adjacent code whose load order is causal;
- shared environment variables with incompatible meanings.

Before integration, verify candidate baseline revisions and run `git diff --check`.
Never rebase by discarding user changes. If the canonical worktree moved, regenerate the
candidate against the new baseline or reject it.

## Integration order

Prefer model execution/dependency order. For each selected candidate:

1. re-resolve active source and signature;
2. apply only the path-scoped candidate;
3. prove worker activation;
4. remove only its replay entry;
5. run narrow native prefill verification;
6. keep or rollback;
7. re-evaluate remaining candidates against the updated baseline.

One patch may resolve several Difference Packets. Mark them resolved only after each
declared target pair is native exact.

## Fallback without subagents

If the runtime cannot start subagents, preserve the same Repair Unit contracts and run
Module 5 sequentially. Reporting must state `repair_execution=sequential_fallback` rather
than implying parallel execution.

## Timing

Record per-agent wall time, summed agent work time, repair-wave wall time, queue wait,
integration time and accelerator verification time. Parallel time saved may be reported as an
estimate only when its formula is shown, for example:

```text
estimated_parallel_saving = sum(agent_duration) - repair_wave_wall_duration
```
