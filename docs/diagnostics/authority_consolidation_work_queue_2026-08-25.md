# Authority Consolidation Active Work Queue

<!-- contract-meta
version: 1
status: active
last_verified: 2026-08-27
implementation: partial
-->

**Established:** 2026-08-25

**Status:** active plan of record for authority-resolution, producer-admission,
and related source-of-truth consolidation work.

This file owns implementation status and ordering. The linked audits preserve
observations, decisions, and acceptance evidence; they do not independently
track whether a work item is current, complete, or safe to activate. An audit
finding is not an executable gate.

This queue does not replace scientific-model, labeling, or performance work
queues. It also does not fold the clipped downstream-admission work into
recording-identity migration 73. Those are separate implementation packages
that share the same consolidation rules.

The authority-versus-scientific-acceptance divergence discovered during the
chaser body-frame work is specified in
[`authority_acceptance_implementation_checklist_2026-08-27.md`](authority_acceptance_implementation_checklist_2026-08-27.md).
That checklist owns the detailed receipt schema, stage adapters, compatibility
classification, tests, and collection policy. This queue owns the cross-cutting
implementation status and ordering through `ACC-001`--`ACC-003` below.

## Governing execution contract

Planning is always read-only and returns a state for every node:

- `admitted`: all concrete required inputs passed their full-strength profile
  resolver;
- `pending_receipt`: an upstream producer is statically supported, but its
  concrete immutable output receipt does not exist yet;
- `blocked`: a required producer/profile is unsupported, its evidence is
  unknown or conflicting, or its resolver rejected the artifact.

A node cannot be marked reusable, runnable, or submittable, and cannot create
scratch or output state, until all required inputs are `admitted`. Planning
must never invent a future digest, placeholder authority record, selector
snapshot, or metadata generation.

Admission has two stages owned by the same executable profile declaration:

1. **Static producer capability:** the selected producer declares its output
   profile, receipt schema, lifecycle, resolver, and boundary-test identity.
2. **Dynamic artifact evidence:** after publication, the declared resolver
   validates the concrete receipt, run/path, digests, row/time identity,
   selector state, and published metadata generation.

The planner does not reimplement profile validation. The generated inventory
and future proof walker consume the same producer/profile declarations. An
unknown producer blocks the requested workload's transitive dependency
closure; the repository CI ratchet separately rejects newly introduced
maintained production entry points that lack declarations and boundary-test
coverage.

## Current safety state

- The four original Sleepyfish recovered keypoint aggregates remain unchanged
  legacy evidence. Four new selector-ineligible canonical coordinate
  successors and four selector-ineligible assignment-keypoint rebinding
  receipts were published from commit `3383675d`; no existing selector moved.
- The rebinding receipts are evidence-only equivalence certificates. They do
  not copy keypoint or mask arrays, alter bundle lineage, or create a permanent
  rebinding stage for future workloads.
- Do not rerun eye angles or dependent products from the historical assignment
  paths directly. Subject shape must first consume the exact bundle plus its
  camera-matched rebinding receipt and publish a fresh canonical proof.
- Do not mutate, rename, restamp, promote, or reinterpret the recovered runs.
- Read-only planning, inspection, proof-sufficiency checks, and
  selector-ineligible canaries remain allowed.
- A working-tree implementation makes the ordinary shard finalizer
  type-incompatible with canonical clipped dependencies, but that broader
  planner wiring has not yet been integrated and required-CI accepted. Treat
  the production planner on `main` as not yet globally hardened.
- The targeted direct-hybrid successor/rebinding branch passed its required CI,
  but PR 64 remains a separate integration event. The pinned deployment is
  suitable for read-only planning and the already-reviewed publications; this
  document does not authorize new selector activation.
- No production selector, registry authority, shared checkout, or canonical
  publication may be activated from this work until all applicable required CI
  checks pass.

## Active queue

Status vocabulary: `open`, `in_progress`, `blocked`, `complete`. `complete`
requires the acceptance evidence named in the row; implementation or local
tests alone are insufficient when required CI is part of the gate.

| ID | Track | Status | Work and acceptance gate | Source evidence |
|---|---|---|---|---|
| GOV-001 | Plan governance | in_progress | Consolidate status here; stamp overlapping audits/queues as evidence or scoped companion work; integrate the documentation through a clean docs change with required CI green. | This document and the companion roster below. |
| ADM-001 | Shared admission | open | Define executable producer/profile declarations and one profile-neutral resolution result. Static capability and dynamic artifact checks must be methods of the same declaration, not duplicated planner grammar. | Clipped-eye audit Phases 1–2; source-of-truth plan Steps 1 and 6. |
| ADM-002 | Shared admission | open | Add `admitted`, `pending_receipt`, and typed `blocked` node states. Reuse, scratch creation, submission, and publication require concrete admission; downstream nodes wait for real upstream receipts. | Clipped-eye audit Phase 2. |
| ADM-003 | Inventory/proof | open | Generate entry point -> producer -> output profile -> resolver -> boundary test edges from executable declarations. At runtime, unknown blocks only the requested closure; CI rejects newly undeclared maintained production entry points. Reuse the graph schema for the future proof walker. | Pipeline survey, clipped-eye audit, source-of-truth plan Step 1. |
| ACC-001 | Scientific acceptance | open | Define one immutable, digest-bound, use-scoped scientific acceptance receipt separate from technical completion, canonical validation, authority selection, and legacy review-status pointers. Preserve product-specific bundle manifests and use shared receipt mechanics. | Authority and scientific acceptance checklist Phases 1–2. |
| ACC-002 | Scientific acceptance | open | Add one normalized use-aware resolution result and stage policy. Technical consumers may resolve exact model-output or bundle authority; acceptance-required consumers must validate the matching receipt and intended use without falling back after an explicit authority fails. | Authority and scientific acceptance checklist Phases 3–7. |
| ACC-003 | Cohort release | open | Add deterministic QC for every member, reproducible stratified review plus exception handling, and one immutable collection receipt binding accepted, excluded, quarantined, blocked, and inapplicable members. This gates accepted release, not selector-ineligible evidence generation. | Authority and scientific acceptance checklist Phases 8–10. |
| PROD-KPT-001 | Producer wiring | in_progress | Wire the maintained clipped strict-v2 keypoint fragment into every active canonical clipped planner. `finalize_keypoint_shards` may remain only as an explicitly labeled compatibility producer and must be type-incompatible with canonical dependencies. A generated-plan regression must prove the ordinary finalizer cannot be admitted. | Working-tree implementation and tests exist on `agent/palette/recording-identity-evidence-20260825`; integration and required CI remain open. |
| PROD-DET-001 | Producer wiring | open | Make artifact-first recording-level detection the explicit production default; compatibility detection must be named and unable to satisfy canonical edges. | Clipped-eye audit Phase 3. |
| RES-CROP-001 | Shared resolvers | open | Add the sealed geometry-only crop profile to the shared position-authority resolver with full manifest validation; retain materialized and split hybrid profiles; migrate consumers and then remove the process-global adapter. | Crop-contract audit decision and tripwires. |
| PROD-MASK-001 | Producer wiring | open | Make subject-mask inference and recording-bundle assignment ingress require an admitted keypoint/crop authority. Remove path/completion and latest/sorted fallback from canonical mode while retaining explicit compatibility reads. | Clipped-eye audit producer inventory and Phase 3. |
| RES-TRACK-001 | Shared resolvers | open | Route tracking and track-kinematics through the profile-neutral position surface, add the sealed crop-v2 motion-lineage mode, and bind temporal authority without consumer-local grammar. | Crop-contract audit tripwires and source-of-truth plan Step 6. |
| RES-SHAPE-001 | Shared resolvers | in_progress | Declare the historical-v4 and recording-bundle-v5 subject-shape profiles, decide their normal-consumption lifecycle, and require assignment-keypoint admission before materialization or reuse. | PR 62 added bundle+rebinding consumption; the four-camera admission/publication boundary and the broader profile consolidation remain open. |
| RES-EYE-001 | Shared resolvers | in_progress | Replace direct `.context` assumptions with the shared assignment-keypoint resolver; make assignment proof mandatory in preflight, staging, atomic publication, and consolidated reopen. | PR 62 removed the bundle `.context` failure and seals rebinding lineage; the four-camera subject-shape -> eye execution proof remains open. |
| RES-TAIL-001 | Shared resolvers | open | Bind tail-kinematics and swim-bout inputs to admitted subject-shape, position/motion, and temporal authorities and propagate those exact proofs into their publications. | Clipped-eye audit downstream inventory and source-of-truth plan. |
| SLP-KPT-001 | Targeted recovery | complete | Run a read-only proof-sufficiency check over all 220 surviving shards. Verify provider record, crop-v2 geometry, pose binding, preprocessing, exact row/key/frame coverage, gap/overlap state, and scientific arrays. Return either a sealed proof or typed `unmigratable`; do not infer missing evidence. | All four results were `migratable`; proof payload digests are preserved in the clipped-eye audit and operation receipts. |
| SLP-KPT-002 | Targeted recovery | complete | Publish through the supported general direct-hybrid terminal-evidence profile and maintained strict-v2 path without GPU inference; expose each result through an immutable selector-ineligible canonical coordinate successor. | Four coordinate successors were published under the 2026-08-26 v002 operation; direct and consolidated metadata agree and selectors did not move. |
| RES-ASSIGN-001 | Shared resolvers | in_progress | Implement one assignment-keypoint resolution interface for supported subject-shape profiles, returning explicit `used`/`not_used` and one normalized proof. A selected-profile failure must not fall through to another profile. | PR 62 implements the bundle+rebinding route used by this recovery; profile-neutral consolidation and repository-wide admission states remain open. |
| SLP-REBIND-001 | Targeted recovery | complete | Publish immutable evidence proving that each historical bundle assignment is exactly equivalent to its canonical keypoint successor. Reuse the existing arrays in place; never retarget sealed manifests or make rebinding a routine future DAG stage. | Four complete selector-ineligible rebinding receipts were published from `3383675d`; payload digests are recorded below. |
| SLP-SHAPE-001 | Targeted recovery | in_progress | Produce four read-only subject-shape admission plans from the exact bundle+rebinding pairs. After review, publish fresh access-aware subject-shape runs and dynamically reopen their canonical receipts before any eye node becomes runnable. | All four inputs are admitted by the commit-pinned reports below; execution/publication remains open. |
| SLP-EYE-001 | Targeted recovery | blocked | After each subject-shape receipt exists, plan and run eye angles with the exact canonical keypoint assertion; then bind the resulting proofs through reuse, reporting, and handoff. | `pending_receipt` on `SLP-SHAPE-001`; no eye scratch or output may be created before that transition. |
| RES-BOUT-001 | Shared resolvers | open | Close the bout-kinematics source-admission and promotion boundary so completion/layout alone cannot authorize publication. | Clipped-eye audit and pipeline survey. |
| REP-001 | Reporting/reuse | open | Make availability, reuse, registry readiness, visualization, export, and campaign handoff consume sealed admission results instead of path, name, `latest`, completion, or forced availability. | Pipeline survey and clipped-eye audit Phases 7–9. |
| VIS-CHASER-001 | Visualization/reuse | in_progress | Replace literal minimal-versus-receipt-bound child-object equality with one closed-profile exact-identity validator shared by Marimo discovery and loading; prove the live v4 smoke and 80-recording metadata discovery without selector, legacy, candidate, or unconsolidated fallback. Mount missing persisted views and decide any additive digest-bound interactive descriptor in later packages; do not mutate or recompute the completed scientific cohort. | [`chaser_exact_successor_marimo_status_2026-08-26.md`](chaser_exact_successor_marimo_status_2026-08-26.md) and [`chaser_exact_successor_interactive_visualization_implementation_checklist_2026-08-27.md`](chaser_exact_successor_interactive_visualization_implementation_checklist_2026-08-27.md). |
| RID-001 | Recording identity | in_progress | Finish the current-v2 registry writer boundary, receipt/consolidation races, durable operational evidence, subtraction, canary, and required CI in the ordered packages retained in the source-of-truth plan. Do not expand migration 73 with downstream admission semantics. | Source-of-truth consolidation plan §4.7. |
| TEST-001 | Boundary tests | in_progress | Add real production writer -> publisher -> full-strength resolver -> unpatched consumer tests for crop, keypoint, assignment/eye, track/motion/bout, and reporting/reuse boundaries, plus adversarial tamper and lifecycle cases. Make the inventory/boundary workflow required CI. | PRs 61–62 and 64 cover the targeted keypoint/rebinding/eye boundary; crop, track/motion/bout, reporting, and the dedicated CI gate remain open. |
| NAME-001 | Terminology/lint | open | Reserve authority-claiming run-name tokens such as `canonical` and `authority` for producers declaring the matching output profile, and post-validate the claim. Do not globally reserve `v2`. Replace ambiguous prose `signed hybrid provider` with `provider-record-bound` or `digest- and row-signature-bound` except where quoting an existing schema identifier. | The misleading `keypoints_geometry_authority_*` incident and issuer-authentication review. |
| SUB-001 | Subtraction | open | Remove adapters, fallbacks, duplicate selectors, and superseded validators only after their callers resolve through the supported shared interface and the deletion gates pass. | Redundancy campaign and subtraction queue. |

`blocked` above means a named dependency is unresolved, not that work should be
silently skipped. When a dependency closes, update this queue before treating
the dependent item as active.

## Targeted four-camera recovery plan — 2026-08-27

This is an execution slice of the active queue, not a second plan of record.
It uses the governing `admitted` / `pending_receipt` / `blocked` vocabulary and
the commit-pinned Palette deployment:

```text
/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/direct-hybrid-successor-20260825-3383675d
commit=3383675d344d2d9d68b87b5bcc41f4df5eea497d
```

The operation root is:

```text
/groups/johnson/johnsonlab/jeremy/operations/sleepyfish_2026_08_06_direct_hybrid_keypoint_canonicalization_20260826_v002
```

PR 64 is open and mergeable at the exact deployment commit. Every reported
required check passed in GitHub Actions run `32942597875`; this is green branch
evidence, not a claim that the PR has merged or that the shared checkout moved.

Per-camera materializer plans are written below
`downstream_admission_plans_3383675d/`. Planning omits `--apply`; it may write
only an operation report and must not create scratch, output arrays, selectors,
or registry state. The plans also omit
`--allow-inactive-subject-mask-bundle`: all four selected bundles must remain
active at admission and execution.

The aggregate machine-readable plan is
`downstream_admission_plans_3383675d/campaign_plan.json`, SHA-256
`f635a57669018c17d005a67ba0fe48925b240e24f0bd307f7f5231e1078283eb`.
Its state is `subject_shape_admitted_eye_pending_receipt`; it references the
resolver-generated reports rather than duplicating their validation grammar.

| Camera | Canonical keypoint successor | Bundle assignment evidence | Subject-shape target | Eye-angle target |
|---|---|---|---|---|
| `2010093` | `keypoints_coordinate_successor_sleepyfish_2026_08_06_direct_hybrid_20260826_v002_sleepyfish_cam2010093` | `assignment_keypoint_rebinding_sleepyfish_2026_08_06_direct_hybrid_20260826_v002_sleepyfish_cam2010093` | `subject_shape_sleepyfish_2026_08_06_complete_kinematics_20260827_v001_sleepyfish_cam2010093` | `eye_angles_sleepyfish_2026_08_06_complete_kinematics_20260827_v001_sleepyfish_cam2010093` |
| `2010094` | `keypoints_coordinate_successor_sleepyfish_2026_08_06_direct_hybrid_20260826_v002_sleepyfish_cam2010094` | `assignment_keypoint_rebinding_sleepyfish_2026_08_06_direct_hybrid_20260826_v002_sleepyfish_cam2010094` | `subject_shape_sleepyfish_2026_08_06_complete_kinematics_20260827_v001_sleepyfish_cam2010094` | `eye_angles_sleepyfish_2026_08_06_complete_kinematics_20260827_v001_sleepyfish_cam2010094` |
| `2010095` | `keypoints_coordinate_successor_sleepyfish_2026_08_06_direct_hybrid_20260826_v002_sleepyfish_cam2010095` | `assignment_keypoint_rebinding_sleepyfish_2026_08_06_direct_hybrid_20260826_v002_sleepyfish_cam2010095` | `subject_shape_sleepyfish_2026_08_06_complete_kinematics_20260827_v001_sleepyfish_cam2010095` | `eye_angles_sleepyfish_2026_08_06_complete_kinematics_20260827_v001_sleepyfish_cam2010095` |
| `2010096` | `keypoints_coordinate_successor_sleepyfish_2026_08_06_direct_hybrid_20260826_v002_sleepyfish_cam2010096` | `assignment_keypoint_rebinding_sleepyfish_2026_08_06_direct_hybrid_20260826_v002_sleepyfish_cam2010096` | `subject_shape_sleepyfish_2026_08_06_complete_kinematics_20260827_v001_sleepyfish_cam2010096` | `eye_angles_sleepyfish_2026_08_06_complete_kinematics_20260827_v001_sleepyfish_cam2010096` |

The rebinding receipts seal these immutable payload digests:

| Camera | Rows | Rebinding payload digest |
|---|---:|---|
| `2010093` | 2,936,291 | `afc0076f032bd07242e105ae754ae0cbc1623eec2eac2b32569e76e4894f4528` |
| `2010094` | 2,745,488 | `4f6b27a950b5e64c40a615a47e321aa709205a0c1a2dc42d3e7078c87c14c93e` |
| `2010095` | 2,873,607 | `1bc20c51f4e5ae4eb22205c03b0260b068486b8590256e8f94db3be11402530c` |
| `2010096` | 2,914,539 | `b55f2eb9060f6c098ab988037db5affae76e576cbca4c8b987f642c913dd2061` |

The corrected subject-shape admission reports were independently schema- and
state-checked after generation:

| Camera | Report SHA-256 | Estimated scratch bytes | Result |
|---|---|---:|---|
| `2010093` | `74900241f57bb1a23f3a47d79e99e2348b9a922c1c620a1b0e353c37bb62434a` | 25,127,837,696 | `admitted` |
| `2010094` | `fc822e4e678dc0d49f9709cf0633361732616581b9a95d15a8dfaaf2d65ae903` | 23,564,779,520 | `admitted` |
| `2010095` | `ac564f90e3a80d133f7d2d45df509679c2d4ba660c550a85f7e82be49447079b` | 24,614,330,368 | `admitted` |
| `2010096` | `0e99fcfd221d189cd6ab4372700e74556dfdddf251a614691af5b1ced596357c` | 24,949,645,312 | `admitted` |

All four reports require `bundle_active=true`, dense `masks_roi`, exact
`component_names=[subject_body, swim_bladder, eye_left, eye_right]`, the
camera-matched rebinding payload, and
`allow_inactive_subject_mask_bundle=false`. No scratch directory, analysis
output, selector, or registry state was created.

### Node transitions and execution gates

1. **Keypoint authority — admitted.** Reopen the exact coordinate successor,
   require complete direct and consolidated metadata agreement, and require
   selector-ineligible exact-path consumption. No selector fallback is allowed.
2. **Historical assignment equivalence — admitted.** Reopen the camera-matched
   rebinding receipt and both live authorities. Require `assignment_state=used`,
   exact rows, instance keys, acquisition frames, success, labels, crop
   authority, and normalized keypoint values.
3. **Subject shape — admitted, execution not yet authorized.** The exact bundle
   and rebinding resolved before payload computation for all four cameras. The
   reviewed reports make these nodes runnable, but this planning step does not
   itself authorize submission. Publish through `subject_shape_access_aware_v1`,
   then reopen the direct and consolidated canonical publication before
   releasing its eye dependency.
4. **Eye angles — pending receipt.** Before subject shape exists, record only
   the supported producer, target run, and dependency. Do not predict its
   digest. After subject-shape publication, build a fresh read-only eye plan
   with `--subject-shape-run` and the exact canonical `--keypoint-run`
   assertion. Only a fully admitted plan may create scratch or submit work.
5. **Tail and bout kinematics — blocked on scoped admission, not on eye
   computation.** Validate and reuse existing track/swim-bout publications
   where their lineage closes. Publish refreshed tail/bout products only after
   their subject-shape, track-motion, and temporal proofs resolve.
6. **Handoff — blocked on publication receipts.** Produce one dataset manifest
   listing each selected run, authority/rebinding digest, units, coordinate
   system, component scientific-failure annotations, reader, commit, atomic
   publication receipt, registry finalization, and consolidated generation.

For future canonical workloads, step 2 disappears: the subject-mask assignment
must seal the canonical keypoint authority when it is created. Rebinding is a
compatibility admission profile for immutable historical assignments, not a
new extraction layer or a recurring DAG node.

### Targeted implementation checklist

- [x] Prove all 220 historical shards migratable without reconstructing missing
  lineage from names or proximity (`SLP-KPT-001`).
- [x] Publish four strict direct-hybrid keypoint successors without GPU rerun
  and without moving selectors (`SLP-KPT-002`).
- [x] Publish four evidence-only assignment rebindings and verify direct versus
  consolidated metadata (`SLP-REBIND-001`).
- [x] Land the bundle+rebinding subject-shape and eye consumer support with
  required CI green (PR 62).
- [x] Finish and review all four subject-shape admission-only reports.
- [ ] Submit and atomically publish all four subject-shape runs; dynamically
  reopen each receipt before planning its eye node.
- [ ] Finish and review all four eye-angle admission-only reports.
- [ ] Submit and atomically publish all four eye-angle runs; validate their
  source-contract, keypoint-authority, consolidated-generation, and registry
  receipts.
- [ ] Admit existing track and swim-bout publications; compute only missing or
  stale tail/bout products.
- [ ] Write and independently validate the four-camera colleague handoff
  manifest.

### Comparison with the broader documents

| Broader document/workstream | Reconciled status | Effect on this delivery |
|---|---|---|
| Clipped-eye incident audit | Phase 4 and the evidence-only part of Phase 5 are complete. PR 62 implements the targeted Phase 6–7 consumer path. Phase 9 is this execution slice. The generic planner-state and repo-wide inventory work from Phases 1–3 remains open. | The four-camera chain may advance one admitted receipt at a time; it does not wait for the global producer inventory. |
| Crop-contract split audit | `RES-CROP-001` and `RES-TRACK-001` remain open, including adapter retirement and the shared tracking position surface. | It does not block subject shape/eyes because their keypoint successor carries a closed crop authority. It does block claiming the general tracking profile consolidation complete. |
| Source-of-truth consolidation plan | Recording identity (`RID-001`) remains a separate in-progress package. Executable profile declarations and typed DAG admission (`ADM-001`–`003`) remain general hardening work. | Neither migration 73 nor unrelated catalog unknowns serialize this already-proven four-camera closure. |
| Pipeline survey and redundancy campaign | Reporting/reuse (`REP-001`), duplicate validation removal, and scan optimization remain open. | Computation can proceed through strict writers; final handoff must still use explicit selected runs and receipts rather than loose discovery. |
| Track/motion reader optimization | Read amplification and validation-performance work remains deferred; source admission must not be weakened for speed. | Reuse existing track/swim publications if their proofs close. Recompute is a fallback, not the default plan. |
| Sleepyfish data handoff | The 2026-08-20 status table predates the canonical keypoint successors and rebindings. | Update it only after the new subject-shape/eye publications and final dataset manifest exist. |
| Future clipped planner hardening | `PROD-KPT-001` has a working-tree implementation but is not yet integrated with required CI evidence. | It prevents recurrence only after integration; the current recovery is safe because every selected source is exact and receipt-admitted. |

## Stage-by-stage implementation checklist

The active table above owns status. The checkboxes below are the required
engineering and acceptance clauses for each stage; checking one does not mark
its work item complete until the table's full gate, including required CI, is
satisfied.

### Stage 0 — Plan governance and operational freeze (`GOV-001`)

Primary surfaces: this file, the clipped-eye audit, the crop-contract audit,
and the source-of-truth consolidation plan.

- [x] Keep the four recovered keypoint aggregates and sealed dependents
  unmodified and unavailable to canonical downstream execution.
- [x] Preserve read-only planning, inspection, proof-sufficiency checks, and
  selector-ineligible canaries.
- [ ] Integrate the audit corpus from a clean docs branch based on current
  `origin/main`; do not include unrelated implementation or working-tree files.
- [ ] Retain this file as the only cross-cutting status source and update old
  plans only with evidence/disposition links.

### Stage 1 — Source import and recording identity (`RID-001`)

Primary surfaces: `shared/source_recording_identity.py`,
`shared/recording_import_receipt.py`,
`registry/recording_identity_authority.py`, `registry/shadow_publish.py`, and
migration 73 in `registry/migration_bodies.py`.

- [ ] Finish the designated-writer boundary and read-only registry facade from
  source-of-truth plan §4.7 without adding downstream admission to migration 73.
- [ ] Close source/consolidated-generation races and preserve durable accepted
  and rejected operation evidence without making telemetry authoritative.
- [ ] Route all current-v2 identity writers/readers, subtract superseded paths,
  and run the quarantined physical writer -> receipt -> projection -> verified
  reader canary.
- [ ] Pass the focused identity tests and every required CI check before any
  current-v2 production activation.

### Stage 2 — Detection, quality, and refinement (`PROD-DET-001`, `ADM-001`)

Primary surfaces: `cluster/clipped_inference.py`,
`cluster/clipped_detection_evidence.py`, and the artifact-first detection,
quality, and refined-detection publishers they compose.

- [ ] Declare the artifact-first recording-level detection producer, its
  quality/refinement receipts, output profiles, lifecycle, and resolver.
- [ ] Make that declaration the explicit production default in the clipped DAG;
  place the legacy fragment behind a named compatibility mode.
- [ ] Make compatibility detection type-incompatible with canonical downstream
  dependencies rather than relying on a flag or run-name convention.
- [ ] Add generated-plan and real writer -> unpatched resolver tests for the
  complete detection/quality/refinement chain.

### Stage 3 — Crop geometry and pixel authority (`RES-CROP-001`)

Primary surfaces: `shared/observation_coordinate_publication.py`,
`shared/zarr/crop_shadow.py`, `shared/zarr/crop_snapshot_publication.py`,
`shared/zarr/historical_geometry_only_crop_adapter.py`, and
`analysis/track_kinematics.py`.

- [ ] Declare materialized crop and sealed geometry-only crop as supported crop
  publication profiles; declare the provider-record-bound hybrid source as the
  pixel-input half of the split profile, not a third crop publication.
- [ ] Add the geometry-only branch to the shared position-surface resolver using
  full `open_persisted_crop_geometry_publication` validation, exact false
  eligibility polarity, mutually exclusive dispatch, and defined zero-row
  behavior.
- [ ] Bind the provider record to live acquisition-camera temporal authority and
  verify digest-algorithm parity for row, coordinate, and temporal seals.
- [ ] Add the corresponding track-motion manifest lineage mode and preserve all
  fields required by `_require_stage_source_surface`.
- [ ] Migrate keypoint, mask, tracking, and visualization consumers to the shared
  resolver; then remove the process-global geometry adapter and neutralize
  false historical/future labels.
- [ ] Add real publisher -> unpatched position/tracking consumer coverage for
  materialized and geometry-only profiles plus mixed-grammar rejection.

### Stage 4 — Keypoint inference and strict finalization
(`PROD-KPT-001`, `SLP-KPT-001`, `SLP-KPT-002`)

Primary surfaces: `cluster/keypoints/clipped_collection.py`,
`cluster/clipped_inference.py`, `cluster/keypoints/v2_finalization.py`,
`utils/write_keypoint_clip_terminal_receipt.py`,
`utils/finalize_clipped_keypoint_v2_bundle.py`, and
`utils/finalize_keypoint_shards.py`.

- [ ] Wire `build_clipped_keypoint_v2_finalization_fragment` into both active
  clipped planners and pass the terminal geometry crop explicitly.
- [ ] Produce `legacy_noncanonical` clip-local shards only as immutable terminal
  evidence; publish recording-wide raw, quality, refined, and body-frame v2
  artifacts through the maintained strict finalizer.
- [ ] Make `finalize_keypoint_shards` an explicit compatibility producer that
  cannot satisfy a canonical dependency; remove the stale legacy refinement
  command from canonical plans.
- [x] Run the 220-shard proof-sufficiency checker and emit exactly one sealed
  `migratable` or `unmigratable` result.
- [x] For `migratable`, add the general direct-hybrid terminal-evidence profile;
  for `unmigratable`, rerun the admitted inference producer. Never mint
  placeholder work-package evidence or a four-run name-based bridge.
- [x] Prove normal-consumption lifecycle, tamper/reorder/gap/overlap rejection,
  and real writer -> strict finalizer -> unpatched keypoint consumer behavior.

### Stage 5 — Subject masks and assignment collections
(`PROD-MASK-001`, `RES-ASSIGN-001`, `SLP-REBIND-001`)

Primary surfaces: `cluster/clipped_inference.py`,
`shared/subject_mask_worker_receipt.py`,
`shared/zarr/subject_mask_bundle_publication.py`,
`shared/zarr/subject_mask_bundle_coordinate_authority.py`, and
`cluster/subject_masks/publish_recording_bundle.py`.

- [ ] Make canonical assignment ingress resolve the exact admitted keypoint and
  crop authorities; remove ordinary `keypoints_runs`, path/completion, and
  latest/sorted fallback from canonical mode.
- [ ] Require one recording-wide keypoint authority initially; reject mixed
  worker runs, gaps, overlaps, row/time reorder, and label/schema disagreement.
- [x] Publish explicit `used` or `not_used` assignment state and retain
  component-level scientific failures as annotations rather than confusing
  them with authority-admission failures.
- [x] For a historical immutable assignment, publish an evidence-only
  equivalence rebinding when every required field closes; otherwise republish
  the dependent artifacts. Never retarget old sealed manifests in place.
- [x] Add real bundle -> rebinding -> subject-shape/eye resolver tests for the
  accepted canonical successor and rejected mismatch/tamper cases.

### Stage 6 — Tracking and track kinematics (`RES-TRACK-001`)

Primary surfaces: `shared/observation_coordinate_publication.py`,
`analysis/track_kinematics.py`,
`analysis_workflows/materializers/track_kinematics.py`, and
`analysis_workflows/materializers/track_kinematics_candidate.py`.

- [ ] Make tracking request one profile-neutral position surface and eliminate
  crop-profile grammar from tracking consumers.
- [ ] Support materialized, collection-successor compatibility, and sealed
  geometry-only lineage through full-strength resolver branches with exactly
  one selected branch.
- [ ] Extend motion-manifest publication and reload to preserve the new lineage,
  coordinate descriptor, row identity, and temporal authority field for field.
- [ ] Un-stub at least one currently monkeypatched track-kinematics boundary
  test and add real crop-v2 publication -> track publication -> reload coverage.
- [ ] Keep performance projections bounded and receipt-validated; optimization
  cannot introduce a skip-validation reader mode.

### Stage 7 — Subject shape (`RES-SHAPE-001`, `RES-ASSIGN-001`)

Primary surfaces: `analysis_workflows/materializers/subject_shape.py`,
`shared/subject_shape_coordinate_publication.py`,
`shared/subject_shape_storage.py`, and
`shared/zarr/subject_shape_bundle_source.py`.

- [ ] Declare the historical-v4 and bundle-v5 publication/lifecycle profiles and
  decide which exact forms are eligible for normal consumption.
- [x] Resolve assignment keypoints before subject-shape payload work and block
  candidate-only or `not_used` sources before scratch creation.
- [x] Make the materializer and publisher consume the same normalized assignment
  proof and preserve it in the immutable subject-shape publication.
- [ ] Publish the four fresh Sleepyfish subject-shape generations only after
  their exact bundle+rebinding pairs are admitted.
- [ ] Add positive tests for both supported profiles and negative lifecycle,
  mixed-run, missing-assignment, and stale-selector cases.

### Stage 8 — Eye angles (`RES-EYE-001`, `SLP-EYE-001`)

Primary surfaces: `shared/eye_geometry_source.py`,
`analysis/eye_angle_analysis.py`, and
`analysis_workflows/materializers/eye_angles.py`.

- [x] Replace `publication.source.context` assumptions with the shared
  assignment-keypoint resolver for every supported subject-shape profile.
- [x] Reject null/missing assignment authority during metadata preflight, before
  exhaustive subject-shape scans or scratch/output creation.
- [x] Pass an exact keypoint run as an assertion, not as an alternate selector,
  and require staged subject-shape/keypoint receipts to match the resolver
  result field for field.
- [ ] Seal the assignment, coordinate, row, temporal, and source-contract
  digests into the atomic eye-angle publication and consolidated reopen.
- [x] Add the real bundle writer -> subject-shape publisher -> strict loader ->
  unpatched assignment resolver -> unpatched eye consumer regression, including
  semantic `not_used` rejection instead of raw `AttributeError`.

### Stage 9 — Tail kinematics, swim bouts, and bout kinematics
(`RES-TAIL-001`, `RES-BOUT-001`)

Primary surfaces: `analysis_workflows/materializers/tail_kinematics.py`,
`analysis_workflows/materializers/swim_bouts.py`,
`analysis_workflows/materializers/bout_kinematics.py`,
`analysis/tail_kinematics_runs.py`, `analysis/swim_bout_io.py`, and
`analysis/bout_kinematics.py`.

- [ ] Resolve exact admitted subject-shape, position/motion, keypoint, and
  temporal inputs required by each product; do not assume upstream group
  completion establishes authority.
- [ ] Persist those exact authority identities in tail, swim-bout, and
  bout-kinematics candidate/publication receipts.
- [ ] Make bout-kinematics promotion re-prove the source admission instead of
  accepting completion/layout alone.
- [ ] Preserve valid partial scientific outputs and component-failure annotations
  without weakening input-authority admission.
- [ ] Add real writer -> promotion -> reload tests plus source-tamper,
  selector-movement, and row/time mismatch failures.

### Stage 10 — Reporting, reuse, visualization, registry, and handoff (`REP-001`)

Primary surfaces: `analysis_workflows/availability.py`,
`analysis_workflows/registry_finalize.py`, `reporting/`, `visualization/`,
`utils/plan_analysis_workflow.py`, and `utils/execute_analysis_workflow.py`.

- [ ] Replace path, name, `latest`, completion-only, and forced-availability
  decisions with the same sealed admission result used by execution.
- [ ] Bind availability/reuse identities to the exact producer profile, receipt,
  artifact/run, selector snapshot, and consolidated generation.
- [ ] Make registry readiness, visualizations, exports, and campaign handoff
  preserve the normalized authority chain rather than reconstructing it.
- [ ] Require a successful serial registry-finalization receipt for campaign
  completion; physical complete/eligible outputs alone are insufficient.
- [ ] Add generated-plan and report/reuse boundary tests that fail on unsupported
  profiles, missing receipts, and selector or metadata-generation drift.

### Stage 11 — Cross-cutting tests, CI, and production execution
(`ADM-002`, `ADM-003`, `TEST-001`, `NAME-001`, `SUB-001`)

Primary surfaces: generated workflow plans, `tests/unit/fisheye/`, required CI
workflows, and commit-pinned deployment/submission helpers.

- [ ] Generate the producer/profile/resolver/boundary-test inventory from
  executable declarations and use the same graph schema for proof walking.
- [ ] Add planner tests for `admitted`, `pending_receipt`, and typed `blocked`,
  including the canonical rejection of `finalize_keypoint_shards`.
- [ ] Add the authority-claiming name lint and post-publication profile check;
  do not reserve generic `v2`.
- [ ] Run focused outside-sandbox tests, then every required CI check. Failed,
  skipped, cancelled, or unrun required checks remain blocking.
- [ ] Use a clean commit-pinned cluster deployment and preserve an admission-only
  dry-run plan before rerunning the four-recording workload.
- [ ] Remove adapters/fallbacks only after all supported callers and boundary
  tests use their replacements; record the actual subtraction.

## Parallel execution order

Two tracks proceed without globally serializing the Sleepyfish recovery behind
the full repository census:

1. **Targeted Sleepyfish recovery:** retain the freeze; implement the
   proof-sufficiency result; publish or rerun keypoints through an admitted
   producer; rebind immutable dependents; close the assignment/eye resolver;
   run boundary tests and required CI; perform an admission-only dry run; then
   rerun the downstream chain.
2. **General hardening:** implement executable profile declarations, planner
   states, generated inventory/proof edges, canonical producer wiring, shared
   resolvers, adapter retirement, and repository-wide boundary CI.

Only dependencies in the requested DAG block targeted recovery. Production
activation still requires the chain-specific boundary tests and every required
CI check for the change.

## Supervisor-agent review disposition

Two supervisor-agent reviews of the initial clipped-eye audit were delivered on
2026-08-25 after initial document commit `d9cbcc42`. Their exact agent names were
not included in the handoff, so this record uses stable local labels rather than
inventing attribution. The matrices below preserve every actionable finding and
the accepted disposition. They are the durable review record; the raw chat text
is not treated as a repository artifact or an implementation authority.

Both reviews affirmed the incident's central conclusions: complete/eligible is
not equivalent to admitted authority; immutable migration or rerun is required
instead of restamping old runs; subject-shape assignment needs one resolver with
no fallback after selected-profile failure; and acceptance requires a real
writer -> publisher -> strict loader -> unpatched consumer boundary test.

Disposition vocabulary:

- `accepted`: adopted as stated;
- `accepted with modification`: adopted with the recorded architectural
  correction;
- `accepted and resolved`: the requested investigation was performed and its
  result is now evidence, although implementation work may remain open.

### Supervisor review A — planning consolidation and recovery architecture

| Finding | Disposition | Decision and rationale | Work/evidence mapping |
|---|---|---|---|
| A1. Seven overlapping active plans reproduce the multiple-sources-of-truth defect at the planning layer. | accepted with modification | One active queue now owns status, but the incident audit was not made a mutable spine. Audits remain immutable evidence; this smaller queue owns ordering and state. | `GOV-001`; companion dispositions in this file. |
| A2. Do not serialize the four-recording recovery behind the complete global inventory. | accepted | Targeted Sleepyfish recovery and general hardening run in parallel. Only the requested DAG's transitive dependency closure blocks recovery. | Parallel execution order; `SLP-*`, `ADM-*`. |
| A3. Check whether the maintained strict-v2 finalizer can consume the original 55 shards per camera before building a bridge. | accepted and resolved | All 220 shards are complete and `legacy_noncanonical`, but all lack `source_crop_pixel_work_package_manifest`. The current terminal-receipt profile requires that exact path and digest, so it cannot consume them unchanged. A proof-sufficient direct-hybrid profile or an admitted rerun is required; the benchmark aggregate adapter is not a production substitute. | `SLP-KPT-001`, `SLP-KPT-002`; clipped-eye audit Phase 4. |
| A4. Admission and resolution must be one mechanism, not independently encoded layers. | accepted | Static capability and dynamic artifact evidence are operations of the same executable profile declaration. The planner consumes the normalized result and does not own another grammar. | `ADM-001`, `ADM-002`. |
| A5. The producer inventory and future proof walker should share one graph/data model. | accepted | Executable declarations generate entry point -> producer -> profile -> resolver -> boundary-test edges; runtime proof walking reuses those stable identities while verifying concrete evidence. | `ADM-003`. |
| A6. Add a lint for authority-claiming names because names steer humans and agents even though admission ignores them. | accepted with modification | Reserve specific claims such as `canonical` and `authority` against producer output profiles and post-validate them. Do not globally reserve `v2`, which may name an unrelated model or schema version. | `NAME-001`. |
| A7. Consolidate the forked documentation before more agents fan out. | accepted | Documentation must move through a clean branch from current `origin/main`; the mixed recording-identity branch must not be presented as docs-only, and required CI remains mandatory. | `GOV-001`; documentation integration state. |
| A8. Enact the procedural freeze, stamp absorbed plans, and resolve strict-finalizer feasibility before implementation fans out. | accepted and resolved | The freeze and plan dispositions are recorded; unchanged strict-finalizer feasibility was resolved negatively because the package evidence is absent. Clean-branch documentation integration and executable enforcement remain open and are not described as complete. | Current safety state; `GOV-001`, `SLP-KPT-001`, `PROD-KPT-001`. |

### Supervisor review B — admission semantics and reproducibility

| Finding | Disposition | Decision and rationale | Work/evidence mapping |
|---|---|---|---|
| B1. Planning must be allowed to report failure; invalid inputs should block execution rather than prevent read-only planning. | accepted | Planning always returns a DAG with `admitted`, `pending_receipt`, or typed `blocked` nodes. Only admitted nodes can be reused, submitted, or create scratch/output state. | Governing execution contract; `ADM-002`. |
| B2. Admission needs static producer capability followed by dynamic concrete-receipt verification. | accepted | Future outputs carry only producer/profile/receipt expectations and remain pending. No predicted digest, selector snapshot, or placeholder authority is allowed. | `ADM-001`, `ADM-002`; clipped-eye audit Phase 2. |
| B3. The generated inventory must not become a third authority, and unrelated unknowns must not block a valid workload. | accepted | The inventory is a projection of executable declarations. Runtime blocking is dependency-closure-local; CI separately ratchets newly introduced maintained production entry points. | `ADM-003`. |
| B4. Migration requires an explicit `unmigratable` result before any bridge is built. | accepted with modification | Proof sufficiency returns exactly `migratable` or `unmigratable`. `source_detect_run=unknown` is not reconstructed and is not independently fatal if every required lineage claim is proven by the sealed crop-v2 lineage and exact row/frame/geometry binding. | `SLP-KPT-001`, `SLP-KPT-002`; clipped-eye audit Phase 4. |
| B5. Separate immediate recovery from broad producer cleanup. | accepted | Targeted recovery and general hardening are parallel tracks with explicit dependencies and shared gates. | Parallel execution order; stage checklist. |
| B6. Record the audited/document commits, observation mode/time, external manifest digest, exact artifact/report/status paths, and additional code anchors. | accepted | The incident audit now records code `6969043e`, initial document `d9cbcc42`, the exact recovery-manifest digest, bounded observation timing, direct metadata-file mode, four artifact paths, report/status evidence, and expanded source map. | Clipped-eye audit evidence identity, reproducibility evidence, and source map. |
| B7. "Signed hybrid provider" overstates issuer authentication. | accepted | Prose now uses `provider-record-bound` or `digest- and row-signature-bound`; existing schema identifiers remain quoted unchanged. | `NAME-001`; crop-contract and clipped-eye audits. |
| B8. Keep this work out of migration 73/current-v2 recording-identity semantics. | accepted | Downstream producer admission is a companion workstream. Migration 73 retains its bounded source-identity purpose. | `RID-001`; source-of-truth plan disposition. |

No review finding is marked implemented merely because its documentation was
accepted. Implementation status remains the active table above, and required
CI remains part of every applicable completion gate.

## Companion evidence and disposition

- `clipped_eye_assignment_authority_failure_2026-08-25.md`: incident evidence
  and chain-specific acceptance requirements; status tracked here.
- `crop_contract_split_audit_2026-08-24.md`: crop-profile decision and resolver
  tripwires; status tracked here.
- `pipeline_survey_2026-08-24.md`: end-to-end evidence and finding catalog;
  authority work tracked here, scientific/performance findings remain evidence
  for their scoped queues.
- `redundancy_campaign_2026-08-24.md`: redundancy catalog and deletion evidence;
  authority work tracked here.
- `contract_enforcement_divergence_review_2026-08-21.md`: historical evidence;
  authority work tracked here.
- `source_of_truth_consolidation_plan_2026-08-25.md`: governing architecture and
  recording-identity implementation record; cross-cutting execution status is
  tracked here, while its §4.7 ordered identity packages remain authoritative
  for that bounded implementation.
- `design_review_findings_2026-08-09.md` and
  `subtraction_queue_2026-08-21.md`: historical/scoped queues; overlapping
  authority items are tracked here, while unrelated scientific and deletion
  findings retain their own evidence.

## Documentation integration state

The 2026-08-24 audit wave begins at `236a9be6`; the clipped-eye audit was first
committed at `d9cbcc42`. Neither commit is on `origin/main` as observed on
2026-08-25. The source-of-truth and track-reader documents also have unrelated
working-tree changes on `agent/palette/recording-identity-evidence-20260825`.
Integrate the documentation from a clean branch based on current `origin/main`;
do not present the mixed implementation branch as a docs-only change. Required
CI remains mandatory even for documentation-only integration.
