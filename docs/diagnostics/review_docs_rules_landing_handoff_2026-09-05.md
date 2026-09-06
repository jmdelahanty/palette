# Review documentation and collaboration rules: local landing handoff

## Authorized PR preparation — September 6

The user authorized Palette-only implementation/review, commits, pushes, and CI
for the relevant PRs and this separate documentation work. Main merges,
repository-rule administration, deployment, `/groups` updates, worktree cleanup,
dependency installation, and production activation remain separate.

Preparation preserves the 30-path draft below and the original checkout. The
two read-only historical scripts retain their selection, membership, and JSON
output logic; their descriptions now state the heuristic/non-authoritative
limits, and the training probe's console label says recording/model overlap
candidates rather than claiming evaluation contamination. No measurement or
production-data sweep was rerun, and historical numerical reports were not
rewritten. The AGENTS addition is exactly the agreed 107-line collaboration
contract; current main's other instructions are preserved.

At 2026-09-06 11:00 UTC, #140 head
`74926b21cb2e587b1caae151aaa9c4df91e592b8` passed all 24 checks in
[run 34028051914](https://github.com/jmdelahanty/palette/actions/runs/34028051914)
and included current main `3d017867e79b14d11ddca3ee1916d50ac6499c78`.
The refreshed #143 head `eb224a216ac7774770b33de021c599b741ce7927` has
fresh CI pending and is not incorporated here. The local clock-publication
and ingestion-enforcement changes also remain separate. These observations
supersede older status snapshots below without altering their evidence.

The subsequent draft PR records its exact final commit, integration
prerequisites, and candidate CI. Local checks do not satisfy its full required
checks; all 23 existing contexts plus `ci-required` must succeed if the validated
gate prerequisite is incorporated. This document grants no merge or activation.

The local docs-only commit is `648d42b15a1b810d4b3e136214cec26023d77ffd`.
The clean owned branch was then combined, without conflicts, with the exact
green #140 prerequisite above. Newer main runtime/source/test changes are
preserved; none are copied from an unvalidated worker branch. Generated module
counts are refreshed for the combined tree. The resulting candidate remains
incomplete until its own required CI succeeds, and its PR records that SHA.

Fresh combined local validation: 151 host-policy, documentation-freshness,
AGENTS-policy, and CI-gate tests passed in 1.01 seconds; script compilation,
generated census, file-size, freshness, and working/index whitespace checks
passed. No live diagnostic sweep, database acceptance, or scientific rerun was
performed. Earlier local counts below are historical snapshots.

Historical checkpoint, 2026-09-05. **Uncommitted preparation only; not merge-ready.** No source
implementation, commit, push, merge, worktree removal, production activation,
registry/store write, or `/groups` update was performed for this preparation.

Documentation-only reconciliation added 2026-09-06; see the
[dated follow-up](#september-6-audit-reconciliation). The September 5 checkpoint
and validation below retain their original scope and do not attest the newer
uncommitted contents.

## Exact source and destination

- Original worktree: `/home/delahantyj@hhmi.org/gitrepos/palette`.
- Original branch: `agent/palette/refined-assignment-rebinding-gaze-20260831`.
- Original HEAD: `73bee0d5194c662e3b7e535be3de92db7ef53f63`.
- Draft worktree: `/tmp/palette-review-docs-rules-20260905`.
- Draft branch: `agent/palette/review-docs-rules-20260905`.
- Draft HEAD/base: `6af66c5a6ba3b35ea0bf00cfc74add7bb22da2b2`.
- Verified baseline main CI: [run 34004329735](https://github.com/jmdelahanty/palette/actions/runs/34004329735), all 23 jobs successful.
- Local main was not advanced; no existing worktree's branch was switched.

The baseline includes #117, #134, #110 and the subsequently merged #137.
These are observations at preparation time, not a claim that remote main will
remain fixed. Recheck main and active worktree owners before later integration.

## Verified preservation checkpoint

Local checkpoint:
`/home/delahantyj@hhmi.org/gitrepos/palette-preservation-20260905-DhRVvF/checkpoint-v1`.

It contains an all-refs Git bundle, a tested restored mirror, manifest, and
separate tracked patches plus untracked/changed-file archives for seven dirty
worktrees. All 398 refs and all 63 original worktree HEADs were verified in
the restored mirror. Full Git integrity checks passed. File archive contents,
modes, hashes/symlink targets, refs, and worktree status were checked for
stability across the capture. The source checkout was not cleaned or staged.

Bundle: `palette-all-refs.bundle`, 45,149,749 bytes, SHA-256
`a8d3b95aa154ed2b47177b46cb684d8cd2a7cffb0adc7102c249a1c4000a4798`.

This is a same-workstation recovery checkpoint, not an off-device disaster
backup. It excludes ignored caches, live databases/stores, external deployment
repositories, LFS payloads, unreachable/reflog-only history, and agent process
state. The parent's earlier partial `worktrees/` attempt is not the verified
checkpoint; use `checkpoint-v1/manifest.json`. Nothing was removed.

Adding the draft made 64 workstation worktree registrations at that moment.
The canary worktree `/tmp/palette-core-chaser-canary-main-20260906` had dirty
work captured in this checkpoint. There is no blanket clearance to retire
other agents' worktrees or the separately managed `/groups` deployments.

## Draft scope and historical interpretation

Imported the 18 docs/two-script-wave changed paths from the original branch
(16 Markdown files and two read-only scripts), plus its four untracked review
reports. Those 18 paths were initially verified byte-identical to `73bee0d5`.
Existing modified docs had no conflicting edits on the selected main base.
This is a working-tree document/script port, not integration of the old branch
history or a claim that its commits passed current CI.

Added 107 collaboration-rule lines to main's AGENTS.md. Removing that inserted
section reproduces main's AGENTS.md exactly, including its newer restrictions
on pytest hosts and LSF. Existing contracts were not weakened.

Added historical-status banners to both old briefs and a dated-state qualifier
to the artifact storage map. Original bodies remain evidence, not fresh
verification or implementation/deletion authority. In particular, the old
wave-1 READY status and the asserted single-defect diagnosis must be reconciled
with the second opinion and final main before implementation assignment.

Added the [server-side CI plan](server_side_ci_enforcement_plan_2026-09-05.md)
and [preliminary instruction audit](agent_instruction_audit_preliminary_2026-09-05.md).
The latter records a separate user-requested Codex config toggle; no skill or
AGENTS.md cleanup was applied in response to that audit.

Running the tracked storage-census generator changed only the scanned module
count from 1706 to 1708 in `zarr_array_schema_census.json` and
`zarr_production_writer_census.json`, accounting for the two added scripts.
No array schemas, writer entries, ratchet baselines, `src/`, `tests/`, or
`.github/` workflows were changed.

## Local validation and outstanding CI

Passed: exact-copy/AGENTS preservation checks; in-memory compilation of the two
scripts; contract freshness; file-size ratchet; explicit Zarr metadata-mode
ratchet; observed-metadata literal check; regenerated storage-census check;
and nine focused host-policy/contract-freshness tests outside the sandbox.
Python resolved through the draft's `scripts/py` to palette-py311 and imported
fisheye from the draft source tree. No live-data diagnostic script was rerun.

Focused command:

```sh
scripts/py -B -m pytest tests/unit/test_pytest_host_policy.py tests/unit/fisheye/test_check_contract_freshness.py -q -p no:cacheprovider
```

There is **no candidate commit and no candidate CI run**. Every required
server-side check is unrun for these uncommitted contents: generated artifacts,
import boundaries, file-size ratchet, Zarr open metadata modes, observed
metadata literals, active contract freshness, package and collection, and
non-GPU test shards 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.
Local checks and green baseline CI do not satisfy that missing evidence.

## Next authorization and sequence

Review this draft, its dated dispositions, the completed instruction audit,
and the joint plan before authorizing a commit/push/PR. Require all current CI
on that exact candidate and the resulting integration commit. Phase A of the
linked CI plan is now active; do not recreate it. Any further administrative
change needs separate approval and fresh exact-base/check evidence.
Reconcile the architecture implementation packages against the final green
main; then retire only individually approved, preserved, owner-cleared
worktrees. No step here authorizes cleanup or production promotion.

## September 6 audit reconciliation

This is a documentation-only correction of the September 1–6 audit set, not a
new fix queue, source implementation, fresh scientific analysis, or repeated
live-store measurement. Status remains in the
[authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md)
and existing specialist queues. The clock-publication worker owns the queue
and its clock interface; this parallel documentation worker did not edit it.

### Ownership and preserved work

- Documentation owner: parallel audit-reconciliation worker, coordinated with
  the clock-publication implementation owner.
- Worktree/branch: `/tmp/palette-review-docs-rules-20260905`,
  `agent/palette/review-docs-rules-20260905`.
- HEAD/base remains `6af66c5a6ba3b35ea0bf00cfc74add7bb22da2b2`; no commit,
  rebase, reset, switch, push, PR update, merge, or deployment was performed.
- All 28 pre-existing dirty paths were preserved. Added the completed
  [instruction audit](agent_instruction_audit_2026-09-05.md) and
  [joint plan](architecture_review_joint_plan_2026-09-05.md) from the original
  checkout, yielding 30 dirty paths. Each copy preserves the report body and
  adds a dated landing/status note. The preliminary audit now explicitly links
  to its completed successor instead of leaving scope confirmation pending.
- Added scoped disposition notes to affected reports/briefs and a scientific
  interpretation note to the existing roadmap. Corrected the wave-1 banner:
  this was an uncommitted main-based port, not work imported onto main.
- Preserved source, tests, scripts, AGENTS.md, generated JSON, historical report
  evidence/probe code, numerical values, historical measurements, and the
  original checkout. No scientific default, persisted digest/receipt grammar,
  supplier rule, selector, registry row, or runtime interface changed here.
  The brief's READY label is now explicitly historical; the landing sequence
  distinguishes already active CI rules from the unmerged gate follow-up.
  Two broken links to `validation_receipt_audit_2026-08-17.md` are retained as
  explicitly unavailable historical references; that absent report was not
  imported, and no live replacement link was invented.
- The old preservation checkpoint remains valid for its captured contents,
  not for all later edits. No worktree or deployment cleanup clearance follows.

### Dated current-state evidence

The coordinating cross-check observed the following at approximately
2026-09-06 08:46 UTC. These are pinned observations, not a live dashboard;
refresh exact heads, required checks, and ownership before later integration.

| Surface | Exact observed state | Disposition |
| --- | --- | --- |
| Main | `1bf9d9195fb026f7bb426c47ee01fdff99cb698d` | Newer than this draft's base; reconcile against the final integration base without overwriting newer work. |
| [Ingestion source corrections, PR #143](https://github.com/jmdelahanty/palette/pull/143) | `3a85a8c9215894945685eaba6f0055730392db44`; 23/23 successful checks; open, draft, CLEAN | F2, F8, and the integer/order portion of F9 are corrected only on this unmerged branch. F3 clock-publication ordering is not closed by it. |
| [Ingestion audit, PR #142](https://github.com/jmdelahanty/palette/pull/142) | `0cbca7f5be5e8f2c1a21660147b831606295bf4e`; 23/23 successful checks; open, draft, BEHIND | Evidence, not an implementation prerequisite or completed all-stage audit. |
| [Success-only CI gate, PR #140](https://github.com/jmdelahanty/palette/pull/140) | `6b87f860348a893ad358f9b46d19e74fccc3db3f`; 24/24 successful checks; open, non-draft, BEHIND | Implemented and checked at that head, but unmerged, not activated, and missing current-base integration evidence. |
| Effective main rules | All 23 existing contexts required, strict current-base checks, GitHub Actions source app 15368 | The historical “main unprotected” finding is superseded. `ci-required` is not yet on main or required; skipped/neutral semantics remain weaker than Palette's success-only policy. |

The [pinned ingestion audit](https://github.com/jmdelahanty/palette/blob/0cbca7f5be5e8f2c1a21660147b831606295bf4e/docs/diagnostics/pipeline_evidence_ingestion_audit_2026-09-06.md)
is authoritative only as dated review evidence: its downstream index is not
proof that every indexed stage was deeply audited. Clock publication ordering,
dispatch acknowledgment/recovery, crop-ledger integrity, and parent-level
clipped intake remain open in that snapshot. The new clock task is tracked by
its owner separately; this document does not anticipate a successful outcome.
The [source-correction handoff](https://github.com/jmdelahanty/palette/blob/3a85a8c9215894945685eaba6f0055730392db44/docs/diagnostics/ingestion_validation_handoff_2026-09-06.md)
records what PR #143 preserves, including valid digest bytes and unchanged
source-ID origin policy. None of these observations activates production data.

### Findings to retain, supersede, or constrain

1. **Unsafe proposals remain rejected.** The
   [second opinion](review_wave_second_opinion_2026-09-04.md) owns the correction
   to blanket `latest` backfills, registry-only authority, and a universal
   receipt/serializer rewrite. Missing selectors may represent deliberate
   ineligibility; exact-path sufficient suppliers need not acquire a selector
   or second upstream/human gate. Preserve product-specific scientific and
   stable identity contracts before sharing mechanics. The old wave-1 brief
   and its copied checklist sequencing are not executable unchanged.
2. **Reproduced boundary defects remain scoped and actionable.** Thirteen of
   fourteen historical characterization cases still reproduce: silent array
   read omission, stale same-key viewer cache, empty crop-result success,
   mixed-subject numeric range matching, mutable training membership, partial
   training/model commits, two output-option override forms, two permissive
   metadata type comparisons, two task-name disagreements, and same-valued
   successor rollback. The thirteen passes confirm undesirable behavior, not
   correctness, live-data corruption, or completed fixes. The
   [consolidation review](shared_helper_consolidation_review_2026-09-04.md)
   identifies preservation tests and interface owners before implementation.
3. **Runtime closure has advanced, but is not universal.** At main `1bf9d919`,
   `runtime_verification.py` registers tracks, track kinematics, swim bouts,
   eye angles, subject shape, and tail kinematics. The executor calls
   `_revalidate_reused_stage_inputs` before commands execute. The old
   metadata-only swim-bout assertion now fails because the candidate is
   refused; two maintained executor refusal tests pass. The generic no-verifier
   branch still returns discovery availability. Thus the old one-/two-verifier
   counts and blanket reuse-bypass claims are partly superseded, not evidence
   that every stage now has complete admission.
4. **Historical measurements require narrower claims.** The second opinion's
   correction is 28 RedScare plus four GoodCopBadCop selector mismatches, four
   registry backups, and remaining `/nvme1` references, including four active
   missing frame-index locators in that survey. Five recording/model overlaps
   on four cameras establish neither exact-example overlap nor downstream
   held-out evaluation leakage, and unresolved membership prevents a claim
   that all other recordings are clean. The numerical tables and mutation
   history remain historical evidence; no backup/store sweep was repeated.
5. **Scientific conclusions stay exploratory.** The strategy/export results
   are recording-level, not subject/batch-cluster-adjusted confirmatory
   evidence. “Innate avoidance” and “passive-coping state” remain hypotheses;
   post minutes 1–3 are a candidate window, not a validated standard. Preserve
   the reported numerical results; this is an inference-limit correction,
   not a rerun or rejection of the observations.
6. **Documentation and integration claims are now separated.** The completed
   instruction audit/joint plan are present in this local draft, but their
   recommendations are not implemented by being copied. The main instruction
   and ingestion follow-up worktrees remain separately owned. Current-base CI,
   combined-commit CI, deployment, and authority activation remain distinct.

### Cross-check reproduction evidence

The coordinating reviewer ran the following outside the sandbox on the
workstation at `/tmp/palette-ingestion-validation-20260906`, exact head
`3a85a8c9215894945685eaba6f0055730392db44`:

```sh
scripts/py -B -m pytest -vv -p no:cacheprovider \
  /tmp/palette-design-review-20260904-zuhe29/test_design_probes.py \
  /tmp/palette-consolidation-review-20260904-L0DOux/test_consolidation_probes.py
```

Result: **13 passed, 1 failed in 4.59 s**, intentionally testing historical bad
behavior. The one failing historical assertion expects acceptance of the now
refused swim-bout candidate; it is not a required-CI regression. These exact
probe sources are preserved in the two linked design/consolidation reports.
PR #143 differs from main only in the clock/map source owners, not these
probed modules; that establishes the stated main-equivalent scope.

```sh
scripts/py -B -m pytest -q -p no:cacheprovider \
  tests/unit/fisheye/test_analysis_workflow_execution.py::test_apply_verifies_completed_run_before_reporting_success \
  tests/unit/fisheye/test_analysis_workflow_execution.py::test_apply_blocks_mutation_when_reused_input_fails_dynamic_admission
```

Result: **2 passed in 1.11 s**. These are maintained behavior tests. They do not
replace the draft's required CI or prove all-stage acceptance. This
documentation worker independently re-read the six verifier registrations and
the executor preflight in main's exact Git object; it did not rerun these
characterization or scientific analyses while editing the reports.

### Remaining landing gate

**Implemented locally:** dated documentation dispositions and missing-report
copies only. **Integrated, deployed, activated:** none. This draft has no
candidate commit and no candidate CI; every required check listed above is
unrun for its current uncommitted contents. It remains **not merge-ready**.
Local documentation validation is recorded below after the edit checks; prior
baseline or PR checks do not validate this combined draft. Compatibility work,
instruction/helper fixes, historical remediation, and broader caller migration
are still separate scopes requiring their own exact-base evidence and authority.

### Documentation validation on September 6

Passed on this unchanged draft base plus the uncommitted corrections:

- SHA-256 preservation checks for all nine pre-existing non-target paths:
  AGENTS.md, artifact storage map, both cluster guides, detect-decode benchmark
  todo, both generated JSON censuses, and both diagnostic scripts. Report-body
  checks preserve the copied reports and historical evidence except the
  explicitly documented notes, brief label/banner, and two link qualifications.
- Local Markdown target/heading and trailing-whitespace checks across all 21
  documents touched by this pass. Both absent historical links are explicitly
  qualified, not silently treated as resolved. `git diff --check` also passes;
  the separate document scan includes the untracked reports it would omit.
- `scripts/py -B scripts/check_contract_freshness.py` (48 managed contracts),
  `scripts/py -B scripts/check_file_size_ratchet.py`,
  `scripts/py -B -m fisheye.diagnostics.zarr_storage_census --check`,
  `scripts/py -B scripts/check_zarr_open_group_modes.py`, and
  `scripts/py -B scripts/check_observed_metadata_literals.py`.
- The focused command in the original validation section was rerun outside
  the sandbox on this workstation: **9 passed in 0.11 s**, no skips/failures.
  `scripts/py` resolved to the `palette-py311` Python 3.11.14 environment.

These local checks do not satisfy any missing candidate server-side evidence.
All 23 required checks remain unrun for the current uncommitted draft, as
listed in the original validation section. No full-suite, GPU, cluster, live
registry acceptance, data migration, or scientific-analysis rerun was performed.
