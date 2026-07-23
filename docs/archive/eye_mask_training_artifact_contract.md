<!-- ARCHIVED 2026-07-04: documents eye-mask code deleted in commit 4a85e5d (eye-mask stage severance). Retained for history only; NOT current. Live replacement: docs/archive/eye_subject_mask_unification_design.md. -->

# Eye Mask Training Artifact Contract (v1)
<!-- contract-meta
version: 1
status: active
last_verified: 2026-04-22
-->

This document defines the merged eye-mask training artifact expected by:

- `fisheye.utils.export_eye_mask_training_zarr`
- `fisheye.utils.validate_eye_mask_training_zarr`

The artifact is a single `.zarr` store with row-aligned crop and eye-mask arrays,
plus split and source-index metadata.

## Scope

- Purpose: stable, versioned training dataset for eye-mask model training.
- Canonical use: curated `_training.zarr` sources.
- One merged output contains one active run under `crop_runs/` and `eye_masks_runs/`.

## Root Contract

Required root attrs:

- `zarr_purpose = "training"`
- `training_task = "eye_masks"` (recommended, validated when present)
- `training_export` dict with:
  - `tool`
  - `created_at_utc`
  - `input_format` (`gray` or `rgb`)
  - `label_mode` (`lr` or `union`)
  - `source_stage` (`refined_subject_masks_runs`, `refined_eye_masks_runs`,
    `eye_masks_runs`, or `mixed`)
  - `source_stage_role` (`canonical`, `derived_compat`, or `mixed`)
  - `source_stage_label`
  - `source_authority_stage`
  - `source_eye_run`
  - `source_refined_subject_masks_run` when canonical refined-subject geometry
    was used or a compatibility source maps to it
  - `source_crop_run`
  - `split_seed`

Required root groups:

- `crop_runs/`
- `eye_masks_runs/`
- `splits/`
- `source_index/`

## `crop_runs/<latest>`

Required arrays:

- `roi_images`
  - shape: `(N, H, W)` for gray or `(N, H, W, 3)` for rgb
- `bbox_norm_coords`
  - shape: `(N, 4)`
- `crop_bbox_norm_coords`
  - shape: `(N, 4)`
- `frame_indices`
  - shape: `(N,)`
  - required to be local indexing `0..N-1`
- `detection_source`
  - shape: `(N,)`
  - integer codes; currently expected in `{0, 1}`

Required attrs:

- `source_crop_run`
- `source_zarr_path`

## `eye_masks_runs/<latest>`

Required arrays:

- `masks_roi`
  - shape: `(N, C, H, W)`
  - binary values `{0, 1}`
  - `C == 2` for `label_mode=lr`, `C == 1` for `label_mode=union`
- `ellipse_params`
  - shape: `(N, C, 5)`
  - `[cx, cy, major, minor, angle]`
- `ellipse_success`
  - shape: `(N, C)`
- `eye_separation`
  - shape: `(N,)`
- `frame_indices`
  - shape: `(N,)`
  - required to be local indexing `0..N-1`
- `detection_source`
  - shape: `(N,)`
  - expected to match `crop_runs/<latest>/detection_source`

Recommended arrays:

- `reason`
  - shape: `(N,)`
  - pipe-delimited tags (for review/history/debug)
- `mask_probs_roi_refined` or `mask_probs_roi`
  - shape: `(N, C, H, W)`
  - finite values in `[0, 1]`

Required successful-ellipse invariant:

- for every `(n, c)` where `ellipse_success[n, c]` is true:
  - `major` and `minor` are finite and positive
  - `major >= minor`

Required attrs:

- `source_eye_stage`
- `source_eye_stage_role`
- `source_eye_stage_label`
- `source_eye_authority_stage`
- `source_eye_run`
- `source_refined_subject_masks_run` when canonical refined-subject geometry
  was used or mapped
- `source_crop_run`
- `source_zarr_path`
- `label_mode`

## `splits/`

Required arrays:

- `train_indices`
- `val_indices`
- `test_indices` (empty allowed)

All split arrays must:

- be 1D integer arrays
- have indices in `[0, N-1]`
- have no duplicates internally
- be pairwise disjoint
- exactly cover all `N` rows across train/val/test

## `source_index/`

Required arrays:

- `source_dataset_idx`
  - shape: `(N,)`
  - integer indices into dataset mapping arrays
- `source_frame_idx`
  - shape: `(N,)`
  - source frame index from original dataset
- `source_dataset_id`
  - shape: `(M,)`
  - dataset IDs
- `source_zarr_path`
  - shape: `(M,)`
  - source zarr paths

Recommended array:

- `source_roi_idx`
  - shape: `(N,)`
  - source ROI row index
- `source_refined_row_ids`
  - shape: `(N,)`
  - stable refined-detection row identity when available, or `-1`
- `source_detect_row_index`
  - shape: `(N,)`
  - raw detect row lineage when available, or `-1`

Required constraints:

- `source_dataset_idx` values in `[0, M-1]`
- `source_frame_idx` non-negative
- when present, `source_roi_idx` must be in `[0, N-1]`
- when present, `source_refined_row_ids` and `source_detect_row_index`
  must be 1D integer arrays of length `N`
- `source_dataset_id` and `source_zarr_path` must be same length `M`

Required attrs:

- `mapping_version`
- `source_count` (`M`)

## Source Selection

The merged training artifact still writes its supervised masks under
`eye_masks_runs/<latest>` so existing eye-mask trainers continue to load a
self-contained artifact.

Source selection is separate from output layout:

- `--eye-stage auto` first uses canonical `refined_subject_masks_runs/<run>`
  eye geometry when the run contains `eye_left` and `eye_right` components.
- If no canonical refined-subject eye geometry is available, `auto` falls back
  to `refined_eye_masks_runs/<run>` and then `eye_masks_runs/<run>`.
- Explicit `--eye-stage refined_subject_masks_runs` requires canonical
  refined-subject eye geometry.
- Explicit `--eye-stage refined_eye_masks_runs` is compatibility behavior. If
  the selected refined-eye run maps to a canonical refined-subject run, the
  exporter uses the canonical source and records that authority in metadata.

`refined_eye_masks_runs` should therefore be read as a compatibility or
historical source unless the export was intentionally built from a legacy
archive.

## Validator Entrypoints

Use either:

- `scripts/py -m fisheye.utils.validate_eye_mask_training_zarr <merged>.zarr`
- `scripts/py -m fisheye.utils.export_eye_mask_training_zarr <source>.zarr <merged>.zarr` (runs validation by default)

## Notes

- This contract is intentionally parallel to detect/keypoint merged training contracts.
- Additional fields are allowed as long as required fields and invariants remain valid.
