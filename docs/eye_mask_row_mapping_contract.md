# Eye Mask Row-Mapping Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
stage_arrays_spec: EYE_MASKS_SPEC
-->

Purpose: define the canonical contract for ROI lineage arrays on eye-mask runs so
Palette and Crimson consumers can align rows safely.

## Scope

Applies to:

- `eye_masks_runs/<run>`
- `refined_eye_masks_runs/<run>`

Lineage arrays covered by this contract:

- `frame_indices` (`(n_rois,)`)
- `detection_indices` (`(n_rois,)`, optional only for legacy compatibility)
- `frame_counts` (`(n_frames,)`)
- `source_refined_row_ids` (`(n_rois,)`, optional when refined detect lineage
  exists)
- `source_detect_row_index` (`(n_rois,)`, optional when raw detect lineage
  exists)

## Canonical Source of Truth

Eye masks are generated over crop ROIs, so row lineage is anchored to the crop
run, not keypoint math:

- Canonical source: `crop_runs/<source_crop_run>`
- Cross-check source: resolved keypoint run (`keypoints_runs/<run>` or
  `refined_keypoints_runs/<run>`)

Rationale: keypoints can be derived/transformed, but ROI identity originates in
crop outputs.

## Producer Contract

For segmentation producers (`traditional`, `yolo`, `unet`):

1. Copy lineage arrays from `crop_runs/<source_crop_run>` when present.
2. If the resolved keypoint run also has the same arrays, verify keypoint vs.
   crop equality before write; fail fast on mismatch.
3. If a legacy crop run is missing an array but keypoints provide it, allow a
   keypoint fallback copy with a warning.
4. If neither crop nor keypoints provide an array, omit it with a warning
   (legacy compatibility mode).

For refinement producer (`refined_eye_masks_runs`):

1. Copy lineage arrays from the source eye-mask run (`source_eye_masks_run`).
2. Do not recompute or reindex lineage arrays during refinement.

## Consumer Contract

For current and new runs, consumers should expect lineage arrays to be present.
For legacy runs, consumers must still tolerate missing lineage arrays and warn.

Recommended strict-mode checks:

- `len(frame_indices) == n_rois`
- if present: `len(detection_indices) == n_rois`
- if present: `len(source_refined_row_ids) == n_rois`
- if present: `len(source_detect_row_index) == n_rois`
- if present: `sum(frame_counts) == n_rois`
- if both crop and keypoint lineage arrays are available during write-time
  validation: values must match exactly.

## Archive Policy

This contract applies equally to:

- `_analysis.zarr`
- `_training.zarr`

Legacy compatibility behavior exists for historical archives only; it is not the
target state for new stage outputs.
