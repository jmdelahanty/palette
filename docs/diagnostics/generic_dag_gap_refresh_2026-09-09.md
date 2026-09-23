# Generic DAG gap refresh — September 9, 2026

Follow-up: the first ranked slice was subsequently implemented as uncommitted
work; see the [exact dependency-pinning handoff](dag_motion_dependency_pinning_handoff_2026-09-09.md).
The review snapshot and its validation limits below describe the pre-fix baseline.

## Scope and ownership

This is read-only runtime-review evidence, not another status authority or an
implementation/merge handoff. The owning queue remains
[authority_consolidation_work_queue_2026-08-25.md](authority_consolidation_work_queue_2026-08-25.md).
The September 7 contract/chaser review remains the broader historical snapshot.
Recording-level segmentation, scientific classifier selection, new intake
formats, and production activation are outside this task.

- Reviewed baseline: `fc6fea5fd937759f5ed21c79d13279b25c309e13`, also the GitHub
  main head verified during this review.
- Active worktree: `/tmp/palette-generic-dag-gap-review-20260909`.
- Branch: `agent/palette/generic-dag-gap-review-20260909`; owner: root review
  agent. Five user-requested Luna/xhigh agents reviewed bounded areas; the root
  agent reconciled their findings against the pinned code.
- The worktree is Git-locked for active review. Its only intended uncommitted
  change is this evidence document. The lock does not prevent deliberate
  force/removal outside Git.
- No runtime edits, commits, pushes, merges, deployments, submissions, dependency
  installations, registry writes, or selector changes were performed.

## What already works and must be preserved

The generic executor is not missing wholesale. It resolves output names,
revalidates reused inputs before launching commands, blocks descendants after
failure, supports a receipt-bound eye-angle admission/apply phase, and verifies
outputs before advancing. See `utils/execute_analysis_workflow.py:229`, `:301`,
and `:547` under `src/fisheye/`.

Strict tracking, full-motion, swim-bout, subject-shape, eye-angle, and tail
readers already exist. The defect discussed below is not an absence of those
scientific/publication validators; it is incomplete composition with the DAG's
selected dependencies and lifecycle.

Reused validated suppliers may close structural branches. Do not globally move
the blocked-ancestor check ahead of reuse in `analysis_workflows/dag.py:156`.
The closure at `:239` and the absent-result handling in
`utils/execute_analysis_workflow.py:246` are intentional. Compare explicitly
resolved dependencies when present; do not invent missing upstream authorities
or human-review gates for sealed lineage.

Likewise, `_verify_tracks` dropping its local dependency argument is not by
itself evidence of a lineage bypass: its preceding discovery gate invokes
`_tracking_authority_availability`, which compares the selected keypoint crop
lineage (`analysis_workflows/availability.py:644`, `:1042`).

## Ranked implementation slices

### 1. Exact tracking selection must survive motion execution and reuse

Confirmed code-level enforcement defect:

- `analysis_workflows/execution.py:156` passes `--tracking-run` only for the
  canary lifecycle (`:187`). Production passes the keypoint run but drops the
  workflow's exact tracking dependency.
- `tracking/single_subject_per_arena.py:544` accepts an omitted tracking run.
  After validating matching source lineage, it selects the matching `latest`
  or sole match (`:675`). Thus two valid tracking runs with the same required
  upstream lineage can lead to a different selected run than the DAG records.
- `analysis_workflows/runtime_verification.py:435` validates the full-motion
  publication but discards `dependency_runs`; it does not compare the sealed
  tracking/keypoint source references against explicitly resolved plan inputs.
  This affects both newly computed outputs and reuse.

Recommended first change: an **enforcement correction**, not a scientific or
persisted-grammar migration. Always pass the exact tracking dependency; retain
the existing strict motion loader and compare its validated source bindings
against applicable resolved plan dependencies. Motion derivation source refs
already include `source_keypoint_path` and `source_tracking_path`
(`analysis/track_kinematics.py:975`, `:1011`); reuse their existing grammar.

Required preservation/regression cases before changing implementation:

1. Matching exact source bindings remain accepted, in production and canary.
2. A selected older tracking run remains the actual input even when another
   matching valid run is `latest`.
3. Internally valid motion from a different explicitly selected tracking or
   keypoint source is rejected on both post-producer verification and reuse.
4. Missing structural ancestors remain permissible when an admitted sealed
   supplier closes the branch; no new unconditional upstream gate.
5. Malformed, tampered, stale, incomplete, and wrong-lifecycle artifacts remain
   rejected by the existing strict loader.

The existing production argv test checks the keypoint pin but not the tracking
pin (`tests/unit/fisheye/test_analysis_workflow_execution.py:875`). The canary
test does check tracking (`:906`) but does not execute a real producer.

### 2. Declare only lifecycle paths the generic producer actually implements

`WorkflowExecutionProfile.unsupported_producer_stage_ids` defaults to empty,
and both installed profiles leave it empty
(`analysis_workflows/execution_profiles.py:28`). The existing planner and
execution-builder refusal branches therefore do not fence off any installed
stage/profile combination (`dag.py:190`, `execution.py:902`).

The shape, eye, and bout adapters do not implement the same candidate lifecycle
as their direct supported candidate paths:

- Shape's generic builder selects the legacy-explicit or supported storage
  profile, not the candidate profile (`execution.py:440`).
- Eye's shared admission/apply argv uses Dask and omits candidate profile/owner
  bindings (`execution.py:373`); its direct candidate path has more specific
  source-owner and serial-execution requirements.
- Bout's generic command always invokes compute/apply without a lifecycle
  option (`execution.py:317`). Its compute publisher is promoting
  (`materializers/bout_kinematics.py:609`, `:792`).

These are unsupported/uncomposed generic canary paths, not demonstrated selector
mutations through a valid unmodified canary executor. The reconciled refusal
boundaries are:

- Shape's legacy source planner calls the eligible-only refined-mask coordinate
  loader (`materializers/subject_shape.py:375`,
  `shared/refined_subject_mask_coordinate_publication.py:5773`). It rejects an
  ineligible source before scratch setup (`materializers/subject_shape.py:1060`).
- Eye's admission without a candidate owner enters the normal strict shape
  reader (`shared/eye_geometry_source.py:1859`, `:1479`) and rejects an ineligible
  shape before apply/scratch creation. The owner-bound candidate plan is a
  distinct supported path, not wired by the generic argv.
- Bout first rejects candidate motion through its production-default motion
  read (`analysis/bout_kinematics.py:2952`, `analysis/track_kinematics_io.py:552`).
  Its later eligible-only eye read is another refusal boundary
  (`analysis/eye_angle_io.py:611`, `:1373`), before bout run creation
  (`analysis/bout_kinematics.py:3144`, `:3155`). Its materializer may already have
  created node-local scratch, retained after failure
  (`materializers/bout_kinematics.py:749`, `:808`). The promoting publisher remains
  a latent hazard if candidate-capable source reading is later wired without
  propagating lifecycle policy.

Forcing eligible inputs into a direct writer is not proof that they pass the
canary executor's prior reuse revalidation. That distinction corrects overly
broad preliminary reviewer claims of valid-canary selector mutation.

Recommended next change: test and populate the existing unsupported-producer
fence, while retaining strict reuse of already supported exact candidates.
Then compose each supported candidate producer individually, proving unchanged
direct/consolidated selectors, no registry mutation, wrong-source refusal,
ownership-loss handling, failed publication, and retry behavior. Do not require
a new universal storage profile or rewrite historical receipts.

### 3. Close the bout runtime admission gap, then inventory custom stages

`bout_kinematics` has a command adapter and storage contract but is absent from
`_RUNTIME_STAGE_VERIFIERS` (`runtime_verification.py:550`). Verification falls
back to generic completion/eligibility metadata (`:581`; `availability.py:1057`)
instead of re-proving payload/schema and exact selected motion/bout/eye sources.
Fresh writer validation is not equivalent to a reuse gate.

Add a stage-owned strict bout verifier through the existing shared gate, with
wrong-lineage and payload-tamper tests. Inventory other executable custom-profile
stages lacking this gate: stimulus response, classification, and presentation
adapters. Presentation validation should remain distinct from numerical product
validity; a plot failure must not erase valid numerical evidence.

### 4. Finish admission declarations and consumer adoption incrementally

The queue's broad `ADM-001`–`ADM-003` remain open. Existing pieces include
`stage_catalog`, `storage_contract_catalog`, `storage_candidate_catalog`, typed
candidate-execution adapters, lifecycle profiles, runtime verifiers, and
`StageCommand.admission`. Only eye angles currently have a generic admission
command (`execution.py:837`). No unified executable declaration currently
generates the complete entrypoint/producer/profile/resolver/boundary-test graph.

Use those existing owners as the starting point, not a competing helper catalog.
In particular, keep physical storage candidates distinct from lifecycle and
scientific authority. Any admission-state work should initially be internal or
an explicitly compatible planner/report extension, not a renaming of persisted
run statuses. Planned future outputs are pending concrete receipts, not admitted
because an output name is known.

The shared core roster and chaser consumer adoption are real, but the current
core-authority ratchet is scoped to the composable chaser planner and eight
descendants (`scripts/check_paradigm_core_authority_access.py:12`). It is not the
repository-wide generated inventory promised by `ADM-003`.

For the next consumer migration, stimulus response is a bounded candidate,
followed by activity/spatial and classifier composite adapters. Bind only their
declared selected suppliers; keep standalone consumers standalone. Do not impose
a universal core roster, body-frame, arena geometry, NN, or human-review gate.
The generic command adapters also embed track `0` in several products; broader
multi-track claims need explicit track selection and complete bout mappings.

### 5. Repair the planning-only availability override's identity ambiguity

`utils/plan_analysis_workflow.py:38` stores an overridden artifact path but no
exact `run_name`. The structural planner can fall back to profile `latest`
(`dag.py:212`), while later dependency discovery requires a nonempty run name.
This makes the documented `--available-stage STAGE=PATH` modeling example
ambiguous. Choose and test an explicit contract: exact parsed/validated identity,
or clearly non-executable hypothetical availability. Do not treat a path or
forced boolean as runtime admission.

### 6. Compose modern origin suppliers without relabeling the crop-row path

The packaged core profile builds tracking from refined-keypoint crop rows and
does not have a subject-position node. Availability explicitly refuses
`subject_position_run` tracking for that producer (`availability.py:662`). This
is an intentional source-contract fence, not a reason to weaken the check.

Modern position/tracking/body-frame motion components already exist, and
`utils/materialize_provider_behavior_chain.py:175` composes an exact named,
selector-ineligible provider chain outside the generic DAG. A later bounded
extension should reuse those producer/handle contracts with an explicitly
selected origin recipe. Keep the existing crop-row production path intact;
do not infer hardware review or new intake prerequisites for an already admitted
archive, and do not call this extension implemented merely because its direct
materializers exist.

Separately, strict-v2 clipped **full** scope deliberately refuses before falling
back to a compatibility finalizer (`cluster/clipped_inference.py:1976`). Its
keypoints/downstream scopes and the generic recording-analysis DAG are distinct
execution paths. This review does not resume the excluded recording-level
segmentation integration work.

## Branch reconciliation and excluded dependencies

- Main is still `fc6fea5f` for this review; the older primary workstation
  checkout at `73bee0d5` is not the runtime baseline.
- Parent intake PR 149 head is
  `0653e75e5faad6b05a4ec715621cc6d9ee727f5d`. All 24 required checks were successful
  in the fresh September 9 check, but it remains unintegrated. Older prose saying
  those checks are unrun is historical, not current evidence. Its merge-base
  with main is `fc6fea5f`; its diff changes neither `analysis_workflows`, tracking,
  nor `cluster/clipped_inference.py`, so it does not close the runtime gaps above.
- The dirty clipped-analytics worktree at `d8cb0c21` modifies the same generic
  execution file only to cap swim-bout level workers at two; it is not a tracking
  pin or candidate-lifecycle fix. Preserve that independent performance work.
- Dirty subject-shape optimization, deterministic-ellipse finalizer, and targeted
  keypoint worktrees also remain preserved. Their older bases must not overwrite
  stronger current-main validators or be treated as integrated fixes.
- Transfer watching and the proposed unified-H5 intake adapter are separate
  orchestration/intake work, not prerequisites for correcting the generic DAG on
  an already admitted recording archive.

## Validation evidence and limits

All pytest commands used the workstation `scripts/py` outside the sandbox,
synthetic temporary inputs, and disabled pytest's repository cache. No production
archive or cluster test execution was used.

From the protected worktree, this command passed **80 tests in 6.35 seconds**:

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-dag-pycache scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_analysis_workflow_dag.py \
  tests/unit/fisheye/test_analysis_workflow_execution.py \
  tests/unit/fisheye/test_check_paradigm_core_authority_access.py \
  tests/unit/fisheye/test_core_paradigm_authority.py \
  tests/unit/fisheye/test_validated_behavior_core_chaser_contracts.py -q
```

The following existing tests were subsequently run individually with the same
command prefix and `-q`, and passed:

- `tests/unit/fisheye/test_subject_mask_recording_bundle_publication.py::test_recording_bundle_publishes_coordinate_bound_members_and_subject_shape_v5`:
  1 passed, 9 Zarr warnings, 93.19 seconds; includes wrong-bundle and tamper refusal.
- `tests/unit/fisheye/test_eye_angle_materializer.py::test_materializer_stages_computes_shards_and_publishes_with_provenance`:
  1 passed, 1 Zarr warning, 8.64 seconds; includes wrong-shape refusal.

Before those individual runs, a three-test invocation selected the subject-shape
test below followed by those two tests. It was interrupted after 78.07 seconds
in the first test, with `KeyboardInterrupt` in `pathlib.py:71` and no tests
completed. It was not a passing check or an assertion failure; the cause was not
diagnosed, and subject-shape candidate validation remains deferred. Exact
three-test command:

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-dag-pycache scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_subject_shape_runs.py::test_subject_shape_byte_planned_candidate_is_complete_ineligible_and_pointer_free \
  tests/unit/fisheye/test_subject_mask_recording_bundle_publication.py::test_recording_bundle_publishes_coordinate_bound_members_and_subject_shape_v5 \
  tests/unit/fisheye/test_eye_angle_materializer.py::test_materializer_stages_computes_shards_and_publishes_with_provenance -q
```

To obtain the missing local evidence, run only the first node ID with that
workstation prefix after investigating the stall; do not silently mark it green.

Most generic DAG tests are metadata fixtures or command-rendering assertions;
apply tests monkeypatch subprocesses and the verifier. The two real temporary
producer/publication tests strengthen specific boundaries but do not establish a
full generic CLI producer-to-unpatched-consumer chain. No new negative regression
tests were added. Full repository suites, new-change required CI, benchmarks,
and a new DAG canary were not run. This review does not make an implementation
complete, merge-ready, deployed, scientifically accepted, or activated.
