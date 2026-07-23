# Traditional V2 Keypoint Migration Design

## Goal

Add a new 5-point fish pose skeleton that extends `traditional_v1` with:

- `snout_tip`
- `tail_tip`

without rewriting existing 3-point recordings in place.

The intended workflow is:

1. keep existing `traditional_v1` keypoint runs intact
2. create a new `traditional_v2` keypoint run by copying the old run
3. manually or semi-automatically fill the two new landmarks
4. keep training/export guardrails that require one skeleton identity per dataset

## Current Status

The following pieces are now implemented:

- `configs/fisheye/pose_schemas/traditional_v2.json`
- `scripts/py -m fisheye.utils.extend_keypoint_skeleton`
- dynamic manual keypoint review selection for runs with more than 3 keypoints
- `scripts/py -m fisheye.detection.detect_keypoints_yolo --pose-schema ...`,
  including dynamic output array shape and model/schema keypoint-count checks
- schema-driven derived metric storage on `refined_keypoints_runs`
- `scripts/py -m fisheye.utils.backfill_keypoint_derived_metrics`
- keypoint profile aggregation of derived metrics in
  `analysis/keypoint_profile_runs/<run>.attrs["profile_summary"]`

Current recommended canary flow:

1. run a reliable `traditional_v1` keypoint model over the target crop
   representation
2. refine that 3-point run
3. extend the `traditional_v1` refined run into a `traditional_v2` seed run
4. manually label `snout_tip` and `tail_tip`
5. approve the completed refined run for `intended_use=training`
6. only then export/train a `traditional_v2` pose model

Do not treat an auto-generated `traditional_v2` seed as a reviewed label
source. The seed intentionally has missing new landmarks and should keep
`refined_success=false` / `usable_keypoints=false` until manual completion.

Direct `traditional_v2` YOLO inference is supported by the runtime, but the
current available 5-point model is not approved as an automatic label source
for the new PyNvVC-luma crop representation. A 2026-05-17 smoke with
`pose_cedar_shadow_pose_traditional_v2_v002/.../weights/best.pt` showed very
low normal-confidence coverage on the new sampled `sickyfish`/`sleepyfish`
training crops. Forcing near-zero confidence can produce arrays, but those
outputs should remain experimental and should not become `latest` or be used
for training without manual review.

After manual completion:

1. run `backfill_keypoint_derived_metrics` to populate named derived metrics and
   finalize migration status
2. backfill/write keypoint profiles from the completed refined run

For the current `sickyfish` PyNvVC-luma training zarrs, the local working
pattern is:

- keep `keypoints_runs/latest` on the reliable 3-point
  `keypoints_pynvvc_luma_v1_20260517_cam...` source run
- set `refined_keypoints_runs/latest` to the corresponding
  `refined_keypoints_traditional_v2_seed_pynvvc_luma_v1_20260517_cam...` seed
  while it awaits manual completion of `snout_tip` and `tail_tip`
- keep any low-confidence direct 5-point runs clearly experimental unless they
  are manually corrected and promoted

Still intentionally deferred:

- registry SQL projection of skeleton-specific derived metrics
- registry query/report surfaces for those derived metrics

That deferral is intentional because not every skeleton will expose the same
named metrics, and the query surface needs a clearer cross-skeleton policy
before denormalizing them.

## New Skeleton

Schema file:

- `configs/fisheye/pose_schemas/traditional_v2.json`

Skeleton identity:

- `schema_name = "traditional_v2"`
- `skeleton_id = "pose_skel_traditional_v2"`

Keypoint order:

1. `swim_bladder`
2. `eye_left`
3. `eye_right`
4. `snout_tip`
5. `tail_tip`

Edges:

- `swim_bladder -> eye_left`
- `swim_bladder -> eye_right`
- `eye_left -> eye_right`
- `snout_tip -> eye_left`
- `snout_tip -> eye_right`
- `swim_bladder -> tail_tip`

This keeps the original 3-point indices stable and appends new points at the end.

## Old-To-New Mapping

`traditional_v1 -> traditional_v2`

- `0 -> 0` (`swim_bladder`)
- `1 -> 1` (`eye_left`)
- `2 -> 2` (`eye_right`)

New target indices:

- `3` (`snout_tip`) initialized missing
- `4` (`tail_tip`) initialized missing

## Migration Utility Requirements

The migration utility should create a new run rather than mutating the source run.

Suggested CLI shape:

```bash
scripts/py -m fisheye.utils.extend_keypoint_skeleton \
  /path/to/recording_training.zarr \
  --source-run refined_keypoints_... \
  --target-run refined_keypoints_traditional_v2_seed_001 \
  --target-schema traditional_v2
```

Required behavior:

- resolve source run from `keypoints_runs` or `refined_keypoints_runs`
- copy lineage arrays:
  - `frame_indices`
  - `frame_counts`
  - `n_rois` where present
  - `detection_indices`
  - `detection_source`
- allocate target coordinate arrays with shape `(n_rois, 5, 2)`
- copy source coordinates into indices `0:3`
- initialize `snout_tip` and `tail_tip` to `NaN`
- carry forward source confidences for the first 3 points when available
- initialize new point confidences as missing / invalid
- write:
  - `keypoint_labels`
  - `kpt_shape`
  - `skeleton_id`
  - `pose_schema`
  - `source_skeleton_id`
  - `source_kpt_shape`
  - `source_pose_schema`

Suggested JSON report:

- `source_run`
- `target_run`
- `source_skeleton_id`
- `target_skeleton_id`
- `rows_copied`
- `new_keypoints_initialized`
- `index_mapping`

Implemented utility:

```bash
scripts/py -m fisheye.utils.extend_keypoint_skeleton \
  /path/to/recording_training.zarr \
  --source-run refined_keypoints_... \
  --source-parent refined_keypoints_runs \
  --target-run refined_keypoints_traditional_v2_seed_001 \
  --apply
```

The utility currently:

- copies the source run into a new sibling run
- appends `snout_tip` and `tail_tip` as missing
- preserves the source run untouched
- writes migration metadata:
  - `migration_status = "needs_keypoint_completion"`
  - `migration_completion_required_keypoints`
  - `migration_source_run`
  - `migration_target_schema`

After manual completion, `scripts/py -m fisheye.utils.backfill_keypoint_derived_metrics`
can mark the migration complete.

Implemented batch utility:

```bash
scripts/py -m fisheye.utils.batch_extend_keypoint_skeleton \
  /nvme1/recordings \
  --recursive \
  --zarr-use training \
  --source-parent refined_keypoints_runs \
  --apply
```

Batch behavior:

- scans matching zarrs
- defaults to the latest run in the selected source parent when `--source-run`
  is not provided
- creates a `traditional_v2` seed sibling run per archive
- leaves existing source runs untouched
- skips existing seed runs unless `--overwrite` is passed

## Important Runtime Constraint

The new schema file alone is not enough for full first-class support.

Resolved blockers:

- `src/fisheye/shared/zarr/stage_arrays.py` now models raw and refined
  keypoint coordinate/confidence arrays with dynamic `n_keypoints`
- `src/fisheye/docs/zarr_structure.md` now documents dynamic
  `(n_rois, n_keypoints, 2)` / `(N, K, 2)` keypoint arrays
- `src/fisheye/detection/detect_keypoints_yolo.py` can now write schema-stamped
  dynamic-`K` YOLO keypoint runs

Remaining constraint:

- several geometry and refinement fields are still explicitly triangle-based
  compatibility metrics for the 3-point starter skeleton
- downstream consumers that need semantic landmarks must continue migrating to
  label-based resolution and clear missing-label failures

So the first migrated `traditional_v2` runs should still be treated as
seed/canary labeling runs until the new landmarks are manually completed and
the target downstream consumer has been validated.

## What Must Generalize Next

### Phase 1: Metadata Hardening

Every keypoint and refined-keypoint writer should explicitly set:

- `skeleton_id`
- `kpt_shape`
- `pose_schema`
- `keypoint_labels`

Status: mostly implemented for active raw/refined writers; keep this as a
maintenance requirement for any new writer.

### Phase 2: Dynamic Keypoint Array Specs

Replace fixed `(n_rois, 3, 2)` assumptions with `(n_rois, K, 2)` where possible.

Status: implemented for the shared stage-array contracts and the YOLO writer.
Remaining work is producer/consumer-specific, not the core storage contract.

Priority surfaces:

- `keypoints_runs`
- `refined_keypoints_runs`
- helper docs and validation

### Phase 3: Label-Based Consumer Resolution

Any downstream consumer that needs a specific landmark should resolve by label, not hard-coded index.

Examples:

- eye-mask tools should resolve `eye_left` / `eye_right`
- subject-mask SAM prompting should resolve labels from run metadata
- future tail/body geometry should resolve `tail_tip` by label

See also:

- `docs/keypoint_derived_metric_schema_contract.md`

That contract defines the planned named-metric layer for skeleton-aware
distances such as `total_length` and `tail_length`.

### Phase 4: New Review/Labeling Flow

Once array specs are generalized:

- seed a `traditional_v2` run from a `traditional_v1` source
- manually annotate `snout_tip` and `tail_tip`
- keep training exports single-skeleton only

## KPT Shape Policy

Runtime coordinate arrays remain `(n_rois, 5, 2)`.

Training/export identity should use the repo's existing `kpt_shape` convention for pose datasets. For the new skeleton, that should resolve to:

- `kpt_shape = [5, 3]`

where the second dimension remains the model/export convention already used in pose training.

## Recommendation

Implement in this order:

1. keep using the implemented canary flow for manual `traditional_v2` labeling
2. generalize keypoint stage specs from `3` to `K`
3. update fixed-index consumers to label-based resolution
4. decide how skeleton-specific derived metrics should surface in registry/query tools
5. only then expand broader operator/reporting support

This keeps the rollout incremental and preserves the current 3-point workflow while the 5-point path is being proven.
