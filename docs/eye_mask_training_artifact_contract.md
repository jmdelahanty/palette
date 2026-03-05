# Eye Mask Training Artifact Contract (v1)
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
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
  - `source_stage` (`eye_masks_runs` or `refined_eye_masks_runs`)
  - `source_eye_run`
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
- `source_eye_run`
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

Required constraints:

- `source_dataset_idx` values in `[0, M-1]`
- `source_frame_idx` non-negative
- when present, `source_roi_idx` must be in `[0, N-1]`
- `source_dataset_id` and `source_zarr_path` must be same length `M`

Required attrs:

- `mapping_version`
- `source_count` (`M`)

## Validator Entrypoints

Use either:

- `scripts/py -m fisheye.utils.validate_eye_mask_training_zarr <merged>.zarr`
- `scripts/py -m fisheye.utils.export_eye_mask_training_zarr <source>.zarr <merged>.zarr` (runs validation by default)

## Notes

- This contract is intentionally parallel to detect/keypoint merged training contracts.
- Additional fields are allowed as long as required fields and invariants remain valid.
