# Keypoint Multi-Skeleton Recording TODO

## Goal

Support multiple keypoint skeleton identities within the same recording lineage,
including a migration path to create a new skeleton run by copying existing
keypoints and adding new keypoints.

Keep current safety rule: a single training set/export must resolve to exactly
one skeleton identity (`skeleton_id` + `kpt_shape`).

## Why This Is Needed

- Current workflows assume one simple starter skeleton.
- We now want iterative schema growth (for example, keep old 3-point labels and
  add more points) without duplicating whole recordings into separate archives.

## Current State (As Implemented)

- Multiple keypoint runs can already coexist in one recording zarr, and run
  selectors exist (`latest`, `latest_traditional`, `latest_yolo`, explicit run).
- Training/export paths fail fast on mixed skeleton identities across selected
  datasets (intended behavior).
- Shared keypoint/refined-keypoint stage specs and primary training/export
  readers now use dynamic `K` contracts for coordinate/confidence arrays.
- Remaining fixed-3 surfaces are mostly producer-specific traditional-v1
  outputs or compatibility/QC arrays, not the shared storage contract.
- `traditional_v2` seed runs can now be created from `traditional_v1` refined
  runs and completed manually.
- `traditional_v3` is now packaged as a 10-point schema that extends
  `traditional_v2` with `mid_tail` and pectoral-fin insertion/tip landmarks.
  It is intended for new manual labeling/model training and should be seeded
  as a new run, not applied by mutating existing `traditional_v2` runs.
- schema-driven derived metrics can now be stored on refined runs and surfaced
  in keypoint profile payloads.
- historical keypoint/refined-keypoint runs can now be normalized in place to
  explicit `skeleton_id` / `kpt_shape` attrs with dedicated audit/backfill
  tooling.
- registry/query projection of those derived metrics is intentionally deferred
  pending a cross-skeleton query policy.

## Non-Goals

- Allow mixed skeleton identities inside one training set.
- Remove current single-skeleton guardrails in preflight/export/validation.

## Phase 0: Contracts and Policy

- [ ] Add explicit policy section to keypoint contracts:
  - recording-level: multiple skeleton runs allowed
  - training-set-level: single skeleton required
- [x] Define packaged heuristic profiles as a separate config surface from
      `pose_schema.metadata`.
- [x] Define canonical skeleton identity source precedence:
  1) explicit `skeleton_id` attr
  2) `pose_schema.skeleton_id`
  3) fallback `pose_schema:<name>`
- [x] Define required attrs on keypoint/refined runs:
  - `skeleton_id`
  - `kpt_shape`
  - `pose_schema` (nodes/edges/metadata)
  - optional `heading_computation_override`
- [x] Define heading metadata precedence:
  1) run attr `heading_computation_override`
  2) `pose_schema.metadata.heading_computation`
  3) deprecated run attr `heading_computation`
  4) disabled / unavailable

## Phase 1: Data Model Hardening

- [x] Ensure writers set `skeleton_id` and `kpt_shape` explicitly on new
      `keypoints_runs/*` and `refined_keypoints_runs/*` (not only `pose_schema`).
- [x] Ensure new runs that persist meaningful `heading` set canonical heading
      semantics in `pose_schema.metadata.heading_computation`.
- [ ] Only use `heading_computation_override` for explicit run-level divergence
      or disable behavior.
- [x] Add maintenance/backfill utility to populate missing
      `pose_schema.metadata.heading_computation` on existing keypoint runs.
- [x] Add audit utility to report runs missing explicit `skeleton_id` /
      `kpt_shape` attrs.
- [x] Add maintenance/backfill utility to populate missing explicit
      `skeleton_id` / `kpt_shape` attrs on historical keypoint runs.
- [ ] Verify registry ingestion continues to map skeleton specs into
      `pose_skeleton_specs` and `keypoint_data_profile`.

## Phase 2: Migration Utility (Starter -> Extended Skeleton)

- [x] Add utility to create a new keypoint run from a source run:
  - copies lineage arrays (`frame_indices`, `detection_indices`, `frame_counts`)
  - copies existing points into mapped indices
  - initializes new points as `NaN` + failure labels
  - writes updated `keypoint_labels`, `kpt_shape`, `pose_schema`, `skeleton_id`
- [x] Add optional mode to create corresponding refined run seed.
- [x] Emit JSON report with counts:
  - rows copied
  - rows requiring manual completion
  - mapping used (`old_idx -> new_idx`)

## Phase 3: Runtime Generalization (Remove K=3 Assumptions)

- Storage-contract note:
  - keep keypoints as dense arrays (`(N,K,2)` and related `(N,K)` arrays)
  - Phase 3 is about replacing fixed positional access with label-to-index
    resolution, not about switching the datastore to per-row key/value storage

- [x] Generalize keypoint stage array contracts from fixed `(N,3,2)` to dynamic
      `(N,K,2)` where valid.
- [x] Generalize the shared keypoint/refined-keypoint stage-array specs for
      coordinate/confidence arrays to symbolic `n_keypoints`.
- [x] Centralize skeleton-edge resolution in `src/fisheye/pose/schema.py` so
      legacy 3-point triangle fallback is compatibility-only and shared.
- [x] Update pose-training collation to preserve runtime `K` for empty/no-label
      batches instead of constructing implicit 3-keypoint tensors.
- [x] Update `src/fisheye/utils/patch_keypoints_from_crops.py` to map
      traditional detector output into the run's label order and preserve
      non-traditional labels.
- [ ] Update keypoint detect/refine code paths that allocate fixed 3-point
      tensors.
- [x] Update YOLO keypoint detection to accept an explicit packaged
      `--pose-schema`, allocate dynamic keypoint/confidence arrays, and fail
      when the model keypoint count does not match the selected schema.
- [x] Switch the traditional raw detector and interactive tuner from hardcoded
      geometry/blob-assignment defaults to packaged `pose_heuristics`
      profiles.
- [x] Update the first downstream consumer slice that indexed fixed eye
      landmarks by position:
  - `src/fisheye/refinement/refine_eye_masks.py`
  - `src/fisheye/tune/eye_mask_tuner.py`
  - `src/fisheye/utils/materialize_refined_eye_masks_compat.py`
  - `src/fisheye/analysis/eye_angle_analysis.py`
  - `src/fisheye/visualization/visualize_eye_angle_overlays.py`
- [x] Update `src/fisheye/utils/keypoint_retry.py` to resolve dynamic `K`,
      source-run label metadata, and metadata-driven heading from the source
      run instead of assuming the starter 3-point contract.
- [x] Remove starter-label fallback defaults from the current training/export
      surfaces:
  - `src/fisheye/utils/export_keypoint_training_zarr.py`
  - `src/fisheye/training/zarr_yolo_dataset_loader.py`
  - `src/fisheye/training/train_pose.py`
- [ ] Continue migrating the remaining downstream consumers from fixed eye
      positions to label-based resolution (`swim_bladder`, `eye_left`,
      `eye_right`) and clear missing-label failures.
- [ ] Confirm eye-mask-dependent logic works when skeleton has extra points.

### Remaining Fixed-3 Categories

- raw `traditional_pose` detector: intentionally emits traditional-v1
  3-point runs until the producer contract changes
- YOLO keypoint writer wrapper: now schema-aware and dynamic-`K`, but richer
  YOLO models still require representation-specific validation before they are
  treated as production label sources
- triangle QC arrays: `triangle_angles`, `triangle_angles_raw`, and
  `triangle_area` remain compatibility diagnostics for the traditional
  triangle, not the general skeleton geometry surface
- non-keypoint `3`s: homography matrices, image channels, and x/y coordinate
  indexing are not keypoint cardinality assumptions

## Phase 4: CLI and Operator Ergonomics

- [ ] Add optional skeleton-aware selectors where helpful:
  - `--skeleton-id` filters for query/report tooling
  - clearer display of selected skeleton signature in prepare/export summaries
- [ ] Add review/check outputs that print per-run skeleton identity.
- [x] Update manual keypoint review to support dynamic keypoint selection beyond
      the starter 3-point layout.

## Phase 5: Validation and Tests

- [ ] Unit tests:
  - migration utility copies + extends correctly
  - mixed skeleton training-set failures remain enforced
  - single-skeleton selected subsets pass
  - label-based landmark resolution behavior
- [x] Focused tests for migration utility, dynamic manual-review selection, and
      derived-metric backfill/profile aggregation.
- [ ] Add/refresh fixtures with at least two skeleton identities in one
      recording.
- [ ] Add a local operator validation recipe for large real zarrs.

## Acceptance Criteria

- [ ] A recording can contain two keypoint runs with different skeleton
      identities and both remain queryable/profiled.
- [ ] Extended skeleton run can be created from starter skeleton run with a
      deterministic mapping report.
- [ ] Training preflight/export continues to hard-fail mixed skeleton sets.
- [ ] All keypoint/eye-mask critical pipelines run successfully with an
      extended skeleton where required labels are present.

## Risks

- Hidden fixed-index assumptions outside keypoint modules.
- Incomplete skeleton metadata on legacy runs causing ambiguous selection.
- Manual-review tooling may need updates for larger `K`.
- Registry/query surfacing of derived metrics may become inconsistent if we do
  not define how skeleton-specific metric sets should be compared or filtered.

## Deferred Query-Surface Work

Derived metrics such as `total_length`, `tail_length`, `head_length`, and
`eye_span` now exist in:

- `refined_keypoints_runs/<run>/derived_metric_*`
- `analysis/keypoint_profile_runs/<run>.attrs["profile_summary"]["geometry"]["derived_metrics"]`

We are intentionally deferring:

- registry SQL projection for those metrics
- query CLI surfaces that filter or sort on those metrics

Reason: not every skeleton will define the same named metrics, so the query
surface needs a careful skeleton-aware policy before we denormalize them.

## Related Docs

- `docs/keypoint_pose_rollout_status.md`
- `docs/keypoint_training_data_card_contract.md`
- `docs/pose_heuristic_profile_contract.md`
- `docs/archive/pose_schema_heuristics_split_proposal.md`
- `docs/archive/traditional_v2_keypoint_migration_design.md`
- `docs/keypoint_derived_metric_schema_contract.md`
- `docs/keypoint_data_profile_schema_contract.md`
- `docs/training_quality_gate_contract.md`
- `src/fisheye/docs/zarr_structure.md`
