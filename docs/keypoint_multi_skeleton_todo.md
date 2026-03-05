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
- Several runtime surfaces are still hard-coded to 3 keypoints (`(N,3,2)`),
  which blocks first-class support for richer skeletons.

## Non-Goals

- Allow mixed skeleton identities inside one training set.
- Remove current single-skeleton guardrails in preflight/export/validation.

## Phase 0: Contracts and Policy

- [ ] Add explicit policy section to keypoint contracts:
  - recording-level: multiple skeleton runs allowed
  - training-set-level: single skeleton required
- [ ] Define canonical skeleton identity source precedence:
  1) explicit `skeleton_id` attr
  2) `pose_schema.skeleton_id`
  3) fallback `pose_schema:<name>`
- [ ] Define required attrs on keypoint/refined runs:
  - `skeleton_id`
  - `kpt_shape`
  - `pose_schema` (nodes/edges/metadata)

## Phase 1: Data Model Hardening

- [ ] Ensure writers set `skeleton_id` and `kpt_shape` explicitly on new
      `keypoints_runs/*` and `refined_keypoints_runs/*` (not only `pose_schema`).
- [ ] Add maintenance/backfill utility to populate missing run attrs from
      existing `pose_schema`.
- [ ] Verify registry ingestion continues to map skeleton specs into
      `pose_skeleton_specs` and `keypoint_data_profile`.

## Phase 2: Migration Utility (Starter -> Extended Skeleton)

- [ ] Add utility to create a new keypoint run from a source run:
  - copies lineage arrays (`frame_indices`, `detection_indices`, `frame_counts`)
  - copies existing points into mapped indices
  - initializes new points as `NaN` + failure labels
  - writes updated `keypoint_labels`, `kpt_shape`, `pose_schema`, `skeleton_id`
- [ ] Add optional mode to create corresponding refined run seed.
- [ ] Emit JSON report with counts:
  - rows copied
  - rows requiring manual completion
  - mapping used (`old_idx -> new_idx`)

## Phase 3: Runtime Generalization (Remove K=3 Assumptions)

- [ ] Generalize keypoint stage array contracts from fixed `(N,3,2)` to dynamic
      `(N,K,2)` where valid.
- [ ] Update keypoint detect/refine code paths that allocate fixed 3-point
      tensors.
- [ ] Update downstream consumers that index fixed eye landmarks by position:
  - use label-based resolution (`swim_bladder`, `eye_left`, `eye_right`) where required
  - fail with clear errors when required labels are missing
- [ ] Confirm eye-mask-dependent logic works when skeleton has extra points.

## Phase 4: CLI and Operator Ergonomics

- [ ] Add optional skeleton-aware selectors where helpful:
  - `--skeleton-id` filters for query/report tooling
  - clearer display of selected skeleton signature in prepare/export summaries
- [ ] Add review/check outputs that print per-run skeleton identity.

## Phase 5: Validation and Tests

- [ ] Unit tests:
  - migration utility copies + extends correctly
  - mixed skeleton training-set failures remain enforced
  - single-skeleton selected subsets pass
  - label-based landmark resolution behavior
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

## Related Docs

- `docs/keypoint_training_data_card_contract.md`
- `docs/keypoint_data_profile_schema_contract.md`
- `docs/training_quality_gate_contract.md`
- `src/fisheye/docs/zarr_structure.md`
