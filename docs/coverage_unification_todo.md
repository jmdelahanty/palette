# Coverage Unification TODO

Goal: standardize coverage/success metadata across detection, refinement, crops,
keypoints, masks, tracking, and diagnostics.

## Why

Current coverage stats are inconsistent across stages:
- `summary_statistics` (detect, crops, keypoints) uses stage-specific keys.
- `coverage_comparison` + `coverage_frames_*` (refine_detect) uses a different
  schema.
- `coverage_stats` (detect_quality) and `coverage_percent` (online refine) are
  yet another shape.

This makes diagnostics and reporting harder to reason about and encourages
duplicate or inconsistent coverage calculations.

## Proposed standard coverage block (common across stages)

Add the following attrs to each run group:
- `coverage_frames_total`: int
- `coverage_frame_source`: `"full"` or `"sampled"`
- `coverage_frames_full`: int (only when sampled)
- `coverage_percent`: float (0–100)
- `frames_with_data`: int (numerator for coverage)

Stage-specific metrics remain in their current `summary_statistics` or other
stage-specific attributes.

## Affected producers (write coverage attrs)

- Detection runs: `detect_traditional.py`, `detect_yolo.py`
- Refined detection: `refine_detect.py`, `refine_online_detect.py`
- Crops: `tracking/crop.py`
- Keypoints: `detect_keypoints_traditional.py`, `refine_keypoints.py`
- Eye masks: `refine_eye_masks.py` (and any mask review/tune step)
- Tracking: `tracking/assign_ids.py` (optional if track coverage is needed)

## Current keypoints coverage/success attrs (for reference)

`detect_keypoints_traditional.py`:
- `summary_statistics` includes:
  - `total_rois`
  - `successful_detections` / `failed_detections`
  - `success_rate_percent`
  - `frames_with_keypoints` (frames with ≥1 successful keypoint detection)
- Also sets:
  - `success_rate` (percent, duplicate of `success_rate_percent`)
  - `keypoints_processed` (total ROIs)

`refine_keypoints.py`:
- `summary_statistics` includes:
  - `total_rois`
  - `source_success` / `source_failures`
  - `refined_success`
  - `pass_rate_percent`
  - `usable_keypoints` and geometry/confidence counters
- No explicit `coverage_frames_*` attributes today.

## Current detection/crop coverage attrs (for reference)

`detect_traditional.py`:
- `summary_statistics` includes:
  - `total_frames`
  - `frames_with_detections`
  - `percent_frames_with_detections`
  - `total_detections` + distribution stats

`detect_yolo.py`:
- `summary_statistics` includes:
  - `frames_with_detections`
  - `percent_frames_with_detections`
  - `frames_with_zero_detections` / `frames_with_multiple_detections`
  - `mean_detections_per_frame`

`refine_detect.py`:
- `coverage_comparison` includes original/filtered/interpolated coverage.
- `coverage_frames_total`, `coverage_frame_source`, `coverage_frames_full` set on run.

`tracking/crop.py`:
- `summary_statistics` includes:
  - `total_frames`
  - `frames_with_crops`
  - `percent_frames_with_crops`
  - `total_rois_cropped`

## Affected consumers (read coverage attrs)

- `utils/check_recording_steps.py`
- `diagnostics/crop_dry_run.py`
- `utils/zarr_inspector.py`
- Visualization: `visualize_refined_detections.py`, `visualize_detect_quality.py`,
  `visualize_online_movement.py`

## Implementation notes

- Keep coverage computation in the stage that generates the data, but reuse a
  shared helper for the math so every stage applies the same rules.
- Diagnostics should read stored coverage attrs by default and only recompute
  if missing or when a `--recompute` flag is provided.
- Preserve existing stage-specific stats to avoid breaking downstream consumers.
