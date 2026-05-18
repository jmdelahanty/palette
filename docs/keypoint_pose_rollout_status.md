# Keypoint Pose Rollout Status

Date anchored: 2026-04-17

Status addendum: 2026-05-17

Purpose: document the current implementation status of Palette's modern
keypoint pose stack after the heading-semantics cutover and the first packaged
pose-heuristic rollout, and define the next concrete implementation sequence.

## Executive Summary

The repo is past the "design only" phase for modern keypoint pose semantics.

What is now materially true:

- heading semantics are metadata-driven through
  `pose_schema.metadata.heading_computation`
- runtime heading evaluation is centralized in
  `src/fisheye/pose/heading.py`
- historical keypoint/refined-keypoint runs can be backfilled to current
  heading fields and metadata
- traditional raw detector/tuner geometry and blob-assignment defaults now load
  packaged heuristic profiles from
  `configs/fisheye/pose_heuristics/traditional_pose/`
- traditional refinement, failure-review, and crop-patch paths now use the
  same packaged geometry defaults, with stage-local params treated as explicit
  overrides instead of competing baseline defaults
- active raw and refined keypoint writers now persist explicit skeleton
  identity attrs (`skeleton_id`, `kpt_shape`, `pose_schema`) on new outputs
- historical keypoint/refined-keypoint runs can now be normalized to the same
  explicit skeleton-identity contract with dedicated audit and backfill tools
- keypoint storage/read contracts now use dense dynamic `K` arrays
  (`(N, K, 2)` and `(N, K)`) rather than treating `K=3` as the shared stage
  invariant
- the first major consumer/training surfaces now resolve labels from run
  metadata and reject missing or mixed signatures rather than silently assuming
  the starter skeleton
- YOLO keypoint inference can now be explicitly stamped with a packaged pose
  schema via `detect_keypoints_yolo --pose-schema`, and the writer validates
  the model keypoint count against that schema before writing dynamic `K`
  arrays

The main remaining work is not "decide the architecture." The architecture is
clear enough now. The remaining work is:

1. finish the packaged-heuristics rollout beyond geometry defaults
2. remove the remaining producer-specific and secondary fixed-`K=3` assumptions
3. decide override policy and remaining reader hardening around explicit
   skeleton identity

## What Is Implemented

### 1. Heading semantics are now metadata-driven

Canonical contract:

- `docs/keypoint_heading_computation_contract.md`

Current runtime helper:

- `src/fisheye/pose/heading.py`

Current behavior:

- readers resolve heading semantics in this order:
  1. `heading_computation_override`
  2. `pose_schema.metadata.heading_computation`
  3. deprecated run attr `heading_computation`
- heading computation is no longer defined by ad hoc fixed-label assumptions in
  each consumer

This is the right long-term boundary:

- pose meaning lives with the skeleton
- heuristics do not define heading semantics

### 2. Shared runtime heading evaluation exists

Implemented in:

- `src/fisheye/pose/heading.py`

Current scope:

- resolves heading precedence from attrs/schema
- evaluates `keypoint` and `midpoint` expressions
- computes:
  - heading scalar
  - heading origin
  - explicit dependent labels

Current write-path adoption:

- raw YOLO keypoint detection
- raw traditional keypoint detection
- refined keypoint writes
- manual correction / failure review paths

### 3. Heading-field migration and backfill utilities exist

Implemented utilities:

- `src/fisheye/utils/backfill_keypoint_heading_computation.py`
- `src/fisheye/utils/backfill_keypoint_heading_fields.py`
- `src/fisheye/utils/backfill_keypoint_label_names.py`

Current status:

- these tools now support dry-run/apply
- `backfill_keypoint_heading_fields` supports `--log-dir` JSONL reporting
- run discovery and live-child reads were hardened for mixed or stale zarr
  metadata cases
- legacy label alias reconciliation (`bladder` -> `swim_bladder`) is handled in
  the maintenance path

### 3a. Explicit skeleton identity on new runs is now hardened

Implemented in:

- `src/fisheye/pose/schema.py`
- `src/fisheye/detection/detect_keypoints_traditional.py`
- `src/fisheye/detection/detect_keypoints_yolo.py`
- `src/fisheye/refinement/refine_keypoints.py`
- `src/fisheye/utils/audit_keypoint_skeleton_attrs.py`
- `src/fisheye/utils/backfill_keypoint_skeleton_attrs.py`

Current behavior:

- shared runtime precedence is now:
  1. explicit run attr `skeleton_id`
  2. `pose_schema.skeleton_id`
  3. fallback `pose_schema:<name>`
- `kpt_shape` resolves from explicit run attr first, then `pose_schema.kpt_shape`,
  then runtime keypoint count where needed
- new raw and refined keypoint runs now persist explicit `skeleton_id` and
  `kpt_shape`, not only `pose_schema`
- the audit utility can report runs still missing explicit attrs
- historical runs can be normalized in place with a dedicated skeleton-attr
  backfill utility
- current operator validation on the maintained recording corpus converged to
  zero missing explicit skeleton attrs after backfill

### 4. Packaged pose-heuristic profiles now exist

Contract:

- `docs/pose_heuristic_profile_contract.md`

Design note:

- `docs/pose_schema_heuristics_split_proposal.md`

Packaged defaults:

- `configs/fisheye/pose_heuristics/traditional_pose/traditional_v1.json`
- `configs/fisheye/pose_heuristics/traditional_pose/traditional_v2.json`

Runtime loader:

- `src/fisheye/pose/heuristics.py`

Current supported policy families:

- `blob_assignment`
- `geometry_qc`
- `heading_qc`
- `flip_detection`

### 5. First packaged-heuristics rollout is complete

Current runtime adoption:

- `src/fisheye/detection/detect_keypoints_traditional.py`
- `src/fisheye/tune/keypoint_tuner.py`
- `src/fisheye/refinement/refine_keypoints.py`
- `src/fisheye/tune/keypoint_failure_review.py`
- `src/fisheye/utils/patch_keypoints_from_crops.py`

Current behavior:

- traditional blob assignment is no longer hardcoded as magic rules in those
  modules
- geometry-QC defaults are read from the packaged profile instead of local
  literals
- tuner slider defaults and saved tuning defaults now start from the packaged
  baseline
- refinement, failure-review, and crop-patch flows now also resolve
  traditional geometry defaults from the packaged profile before applying any
  stage-local overrides

Important current limitation:

- these raw traditional tools still target the starter 3-point skeleton, so
  they currently resolve the `traditional_v1` packaged profile
- the presence of `traditional_v2` packaged defaults does not mean the raw
  detector is already multi-skeleton
- the presence of a direct `traditional_v2` YOLO model does not mean that model
  is the recommended automatic label source for new PyNvVC-luma crops; the
  current recommended path is 3-point inference plus `traditional_v2` seed
  promotion and manual completion

## What Is Not Yet Finished

### 1. Packaged heuristics are not yet the repo-wide default policy surface

Still pending:

- any other repair/retry/manual paths that still carry their own geometry or
  flip defaults
- heading-temporal/QC consumers still use their own local threshold defaults
  instead of packaged `heading_qc`

Current gap:

- the repo now has a packaged policy surface across the main traditional raw,
  refine, review, and patch paths
- it is not yet the default policy surface for every repair/retry/manual path
  or for heading-temporal QC thresholds

### 2. Multi-skeleton runtime support is only partial

Policy and migration direction are real, but runtime generalization is still
incomplete.

Still incomplete:

- the shared stage-array specs now model keypoint coordinate/confidence arrays
  with symbolic `n_keypoints`, but some producer-specific paths still allocate
  traditional-v1 arrays directly
- some consumers still assume positional eye/bladder indexing instead of label
  resolution, but the first eye-mask and eye-angle consumer cluster now uses
  shared label-to-index resolution from run attrs
- `src/fisheye/utils/keypoint_retry.py` now resolves source-run labels,
  dynamic `K`, and metadata-driven heading, but other repair/manual paths have
  not all been brought onto that same helper surface yet
- training/export surfaces no longer silently inject starter labels:
  `src/fisheye/utils/export_keypoint_training_zarr.py`,
  `src/fisheye/training/zarr_yolo_dataset_loader.py`, and
  `src/fisheye/training/train_pose.py` now resolve labels from run attrs /
  `pose_schema` and reject mixed or missing signatures
- dynamic `K` support is not yet the default invariant across keypoint runtime
  modules

Primary tracker:

- `docs/keypoint_multi_skeleton_todo.md`

### 2a. Remaining fixed-3 surfaces are now mostly producer-specific

The current fixed-shape audit no longer shows shared storage contracts or the
main training/export readers as the primary blocker. Remaining fixed-3 hits
fall into these categories:

- raw `traditional_pose` detector output is intentionally a 3-point
  `traditional_v1` producer for now
- the YOLO writer has a schema-aware dynamic-`K` path, but richer skeletons
  still require model-specific validation and operator policy before being used
  as production automatic label sources
- `triangle_angles`, `triangle_angles_raw`, and `triangle_area` remain
  triangle-QC compatibility arrays, not the general skeleton geometry contract
- visualization snippets that index `kp[0]` / `kp[1]` are reading x/y
  coordinates from one already-resolved point, not selecting semantic keypoints
- homography and image-channel reshapes using `3` are unrelated to keypoint
  skeleton cardinality

Next producer work should therefore be explicit: decide when raw YOLO and/or
traditional output contracts should grow beyond traditional-v1, rather than
treating their fixed `3` allocations as accidental reader bugs.

### 3. Explicit skeleton identity is improved but not yet universal

Still pending:

- readers should prefer explicit skeleton identity over schema-name fallback
  wherever possible
- secondary writers and maintenance tools should converge on the same helper
  surface where they still carry bespoke fallback code

Current state:

- the active raw and refined writers now set explicit attrs on new outputs
- the main reader precedence is centralized, but some downstream reader code
  still carries local skeleton-resolution logic
- historical runs can now be normalized in place, but reader tolerance for
  unnormalized external archives still exists for compatibility

### 4. Heuristic-profile overrides are not yet defined beyond stage-local tuning

Current state:

- packaged profiles act as shared defaults
- stage-local tuning metadata still exists, for example
  `analysis_metadata.attrs["keypoint_tuning"]`

Still open:

- whether there should be a formal run-level heuristic override surface
- or whether stage-local tuned attrs remain the only override mechanism

### 5. Some secondary docs and consumers may still lag

The main contracts are now aligned, but secondary surfaces can still lag:

- secondary docs that describe geometry thresholds or traditional behavior
- downstream readers that should remain semantic-only and not absorb heuristic
  policy
- helper/review utilities that still carry hardcoded starter-skeleton logic

## Recommended Next Implementation Sequence

### Phase A: Finish packaged-heuristics rollout

Goal:

- make the packaged heuristic profile the shared default policy source for the
  traditional pose family, not just for the raw detector/tuner

Checklist:

- [x] Audit remaining traditional geometry/flip defaults in:
  - `src/fisheye/refinement/refine_keypoints.py`
  - `src/fisheye/tune/keypoint_failure_review.py`
  - `src/fisheye/utils/patch_keypoints_from_crops.py`
- [x] Decide which defaults belong in packaged heuristic profiles versus which
      are truly stage-local
- [x] Load packaged profiles in the remaining shared traditional paths
- [x] Keep stage-local tuned params as explicit overrides, not silent competing
      default systems
- [x] Add focused tests proving packaged defaults are actually used in those
      paths
- [ ] Extend packaged-heuristic adoption into remaining retry/manual helpers
- [ ] Decide whether `heading_qc` should migrate into the runtime temporal
      heading/QC helpers in a follow-up phase

### Phase B: Harden skeleton identity on new runs

Goal:

- stop relying on schema-name inference where a writer can persist the explicit
  skeleton signature directly

Checklist:

- [x] Audit all keypoint and refined-keypoint writers for:
  - `skeleton_id`
  - `kpt_shape`
  - `pose_schema`
- [x] Ensure all new runs write those attrs explicitly
- [x] Document the exact reader precedence for skeleton identity
- [x] Add maintenance checks that report missing explicit skeleton attrs on new
      outputs
- [x] Provide a maintenance backfill for historical runs missing explicit
      skeleton attrs
- [x] Validate the historical-maintenance path against large real archives and
      confirm convergence with the audit utility
- [ ] Continue migrating downstream readers to the shared helper where they
      still carry local precedence code

### Phase C: Remove remaining fixed-`K=3` runtime assumptions

Goal:

- make richer skeletons a first-class runtime case rather than a migration
  exception

Checklist:

- storage-contract note:
  - keep keypoint storage dense-array based
  - convert consumers from fixed positional indexing to label-to-index helper
    resolution
  - do not redesign the datastore around per-row key/value keypoint objects
- [x] Convert shared keypoint/refined-keypoint stage-array specs from fixed
      `3` to symbolic `n_keypoints` for keypoint coordinate/confidence arrays
- [x] Centralize skeleton-edge resolution, including the legacy 3-point triangle
      fallback, in `src/fisheye/pose/schema.py`
- [x] Make pose-training collation preserve runtime `K` for empty/no-label
      batches instead of constructing implicit 3-keypoint tensors
- [x] Update `src/fisheye/utils/patch_keypoints_from_crops.py` so traditional
      patch output maps into the run's label order and preserves labels the
      traditional detector does not emit
- [x] Convert the first eye-mask / eye-angle consumer slice from positional
      indexing to label resolution:
  - `src/fisheye/refinement/refine_eye_masks.py`
  - `src/fisheye/tune/eye_mask_tuner.py`
  - `src/fisheye/utils/materialize_refined_eye_masks_compat.py`
  - `src/fisheye/analysis/eye_angle_analysis.py`
  - `src/fisheye/visualization/visualize_eye_angle_overlays.py`
- [x] Update `src/fisheye/utils/keypoint_retry.py` to resolve dynamic `K`,
      source-run label metadata, and metadata-driven heading instead of
      hardcoded starter-skeleton assumptions
- [x] Remove starter-label fallback defaults from the current training/export
      surfaces:
  - `src/fisheye/utils/export_keypoint_training_zarr.py`
  - `src/fisheye/training/zarr_yolo_dataset_loader.py`
  - `src/fisheye/training/train_pose.py`
- [ ] Continue replacing positional eye/bladder indexing across the rest of the
      runtime consumer surface
- [ ] Fail clearly when a required label is absent, rather than silently
      assuming the starter skeleton
- [ ] Decide when raw `traditional_pose` outputs should grow beyond
      traditional-v1. The YOLO writer can now write dynamic schema-stamped
      outputs, but each richer YOLO model still needs representation-specific
      validation before promotion.
- [ ] Re-check manual/review UIs for dynamic keypoint count behavior
- [ ] Re-check remaining patch/manual/review helper paths for dynamic `K`

### Phase D: Decide override policy

Goal:

- define one coherent policy for packaged defaults versus local deviation

Checklist:

- [ ] Decide whether there will be a formal run-level heuristic override
      contract
- [ ] If no, document that packaged profiles plus stage-local tuning attrs are
      the only supported model
- [ ] If yes, define precedence and writer guidance explicitly
- [ ] Ensure downstream readers do not confuse heuristic overrides with
      skeleton semantics

### Phase E: Validation and operator closure

Goal:

- treat the modern pose stack as a maintained contract rather than a one-off
  migration

Checklist:

- [ ] Keep in-memory/fake-group unit coverage for zarr-heavy maintenance tools
- [ ] Add mixed-skeleton fixtures that exercise coexisting 3-point and 5-point
      runs in one lineage
- [ ] Add a documented local validation recipe for real archives
- [ ] Periodically audit secondary docs for stale starter-skeleton assumptions

## Suggested Acceptance Criteria For The Next Milestone

This rollout should be considered materially complete when all of the following
are true:

- packaged heuristic profiles are the shared default source for traditional
  detector, refine, review, and patch paths
- all new keypoint/refined-keypoint runs write explicit `skeleton_id` and
  `kpt_shape`
- historical keypoint/refined-keypoint runs have a supported audit/backfill
  path to reach the same explicit identity contract
- the main runtime stack no longer assumes `(N,3,2)` except in intentionally
  starter-skeleton-specific producers
- label-based consumers fail clearly when required labels are missing
- the main docs agree on:
  - heading semantics in `pose_schema.metadata`
  - heuristics in packaged profiles
  - run-local tuning as local override rather than pose meaning

## Related Docs

- `docs/keypoint_heading_computation_contract.md`
- `docs/pose_heuristic_profile_contract.md`
- `docs/pose_schema_heuristics_split_proposal.md`
- `docs/keypoint_multi_skeleton_todo.md`
- `docs/keypoint_heading_validity_todo.md`
- `docs/keypoint_temporal_heading_heuristic_todo.md`
- `src/fisheye/docs/zarr_structure.md`
