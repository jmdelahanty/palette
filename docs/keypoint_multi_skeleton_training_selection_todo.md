# Keypoint Multi-Skeleton Training Selection TODO

Purpose: define how training/export should behave once a single analysis or
training zarr can contain multiple keypoint skeleton identities and both raw and
refined annotation runs.

## Goal

Support archives that contain multiple keypoint skeletons, while keeping a hard
rule that any one training/export job must resolve to exactly one skeleton
identity and one effective annotation source.

This is the broad policy needed to handle:

- legacy `traditional_v1` raw runs
- migrated `traditional_v2_seed` refined runs
- `traditional_v3` seed/refined runs with appended mid-tail and pectoral-fin
  landmarks
- future richer skeletons
- archives where multiple skeletons coexist intentionally

## Historical Problem

Earlier training preflight/export flow treated the selected raw
`keypoints_runs/<run>` as the skeleton authority, even when review-gated row
selection and exported coordinates came from a refined run.

That broke the intended `traditional_v2` migration workflow:

- raw source run may still be `traditional_v1`
- refined reviewed run may already be `traditional_v2`
- old preflight could still report `traditional_v1`
- old export could switch coordinate arrays to refined keypoints but still
  carry raw/manifest skeleton metadata

## Current Status

The main preflight/export path now resolves the effective annotation source and
persists manifest fields such as `annotation_source_kind`,
`annotation_source_parent`, `annotation_source_run`, `skeleton_id`, and
`kpt_shape`. When a refined run supplies the exported coordinates and row gate,
that refined run is the skeleton authority.

This TODO remains useful for open policy and validation work, but the core
"raw run overrides refined run" bug is no longer the expected behavior. The
remaining risks are operational:

- stale registry quality rows can still point preflight at the wrong refined
  sibling until the tie-break fix is completed
- mixed-skeleton selections must continue to fail closed
- operators still need explicit `--skeleton-id` selection once several reviewed
  skeletons coexist in the same candidate pool

## Proposed Model

Separate three concepts explicitly:

1. raw source lineage
   - where the original keypoint run came from
2. effective annotation source
   - the run that supplies coordinates and row gating for export/training
3. skeleton identity
   - the `(skeleton_id, kpt_shape)` attached to the effective annotation source

In other words:

- provenance may point back to a raw 3-point run
- but if training is exporting from a refined 5-point run, the training
  skeleton must be the 5-point skeleton

## Archive-Level Policy

Allowed:

- one zarr may contain multiple `keypoints_runs/*`
- one zarr may contain multiple `refined_keypoints_runs/*`
- those runs may have different `skeleton_id` values

Required:

- every run must carry explicit
  - `skeleton_id`
  - `kpt_shape`
  - `pose_schema`

Non-goal:

- no archive-level restriction that “all keypoint runs in one zarr must share a
  skeleton”

## Training-Level Policy

Hard rule:

- one training/export job must resolve to exactly one skeleton identity

That means:

- mixed-skeleton selection remains a hard error
- but the selected skeleton must come from the effective annotation source, not
  blindly from the raw source run

## Effective Annotation Source

Use explicit terminology and manifest fields:

- `annotation_source_kind = raw | refined`
- `annotation_source_parent = keypoints_runs | refined_keypoints_runs`
- `annotation_source_run = <run_name>`
- `annotation_skeleton_id = <skeleton_id>`

Training/export should reason from this effective annotation source.

If row gating resolves to refined usable keypoints and coordinates are exported
from a refined run, then:

- `annotation_source_kind = refined`
- `annotation_source_parent = refined_keypoints_runs`
- `annotation_source_run = <refined run>`
- training skeleton identity comes from that refined run

Raw source lineage still matters for provenance, but it should not override the
effective training skeleton.

## CLI Direction

Current selectors like `--keypoint-run latest_traditional` are no longer enough
once multiple skeletons can coexist.

Desired additional selectors:

- `--skeleton-id <id>`
- `--annotation-source raw|refined`
- `--refined-run-selector latest_approved|explicit|latest_matching`
- `--keypoint-method traditional_pose|yolo_pose`

The exact CLI can evolve, but the missing capability is explicit skeleton-aware
selection.

## Required Patches

### 1. Preflight Skeleton Identity Must Follow Effective Annotation Source

Target:
- [prepare_keypoint_training_from_registry.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/prepare_keypoint_training_from_registry.py)

Status: implemented for the primary registry-driven path.

Required behavior:

- when quality/review gating resolves a refined run as the effective annotation
  source, derive:
  - `skeleton_id`
  - `kpt_shape`
  - `skeleton_signature`
  from the refined run, not from the raw `keypoints_runs/<source_run>` group

### 2. Export Skeleton Identity Must Match Exported Coordinate Array

Target:
- [export_keypoint_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_keypoint_training_zarr.py)

Status: implemented for the primary merged export path.

Required behavior:

- if export switches `keypoints_path` to `refined_keypoints_runs/.../keypoints_roi`,
  then skeleton identity resolution must use that refined run’s metadata
- manifest/raw fallback metadata must not override the actual exported
  coordinate source

### 3. Manifest Should Record Effective Annotation Source Explicitly

Status: implemented for generated keypoint-training manifests.

Manifest/dataset payload additions:

- `annotation_source_kind`
- `annotation_source_parent`
- `annotation_source_run`
- `annotation_skeleton_id`
- `annotation_kpt_shape`

These should describe the actual exported annotation source, not only the raw
source lineage.

### 4. Registry/Query Surfaces Should Become Skeleton-Aware

Longer-term:

- allow filtering training candidates by `skeleton_id`
- show effective annotation source vs raw lineage
- avoid ambiguous “latest” semantics when multiple skeletons coexist

## Acceptance Criteria

- A zarr may contain both `traditional_v1` and `traditional_v2` runs.
- Training preflight for a reviewed `traditional_v2` refined run reports the
  `traditional_v2` skeleton identity.
- Exported merged training zarr carries the skeleton identity of the actual
  exported keypoint arrays.
- Mixed-skeleton selection still fails closed.
- Operator can explicitly choose a skeleton when multiple are available.

## Related Docs

- [keypoint_multi_skeleton_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_multi_skeleton_todo.md)
- [traditional_v2_keypoint_migration_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/traditional_v2_keypoint_migration_design.md)
- [keypoint_training_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_training_workflow.md)
- [keypoint_training_refined_run_tie_fix_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_training_refined_run_tie_fix_todo.md)
