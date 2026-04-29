# Subject Shape Snout-Centerline Workflow
<!-- contract-meta
version: 1
status: active
last_verified: 2026-04-29
-->

Purpose: document the current subject-shape workflow produced by Palette for
snout-anchored body centerlines, B-splines, tail samples, and review overlays.

This document is the practical implementation/runbook view. The schema-level
contract remains [subject_shape_runs_contract.md](subject_shape_runs_contract.md).
The landmark naming conventions are in
[subject_shape_landmark_conventions.md](subject_shape_landmark_conventions.md).

## Current Implementation

Current writer:

- `schema_id = "analysis.subject_shape_runs"`
- `schema_version = 3`
- `method = "subject_shape_from_refined_masks_v8"`
- `method_version = 8`
- `centerline_method = "snout_anchored_skeleton_longest_endpoint_path_v1"`
- `centerline_skeleton_method = "skeleton_longest_endpoint_path_v1"`
- `centerline_snout_extension_method = "prepend_mask_path_to_body_frame_guided_join_v1"`
- `centerline_snout_join_method = "body_frame_lateral_min_head_region_v1"`
- `head_endpoint_semantics = "validated_snout_tip"`

The important semantic guarantee is:

```text
if centerline_valid is true, head_endpoint_xy is the validated snout_tip_xy
```

Older v2/v5 style runs used `head_endpoint_xy` as the skeleton-derived anterior
endpoint and only measured the gap to `snout_tip_xy`. Current v3/v8 runs use
`snout_tip_xy` as the actual anterior endpoint for valid centerlines.

## Inputs

The workflow consumes one refined subject-mask run:

```text
refined_subject_masks_runs/<run>/masks_roi
refined_subject_masks_runs/<run>.attrs["mask_labels"]
```

Required components for the full current body/tail workflow:

- `subject_body`
- `swim_bladder`
- `eye_left`
- `eye_right`

Component channels must always be resolved through `mask_labels`. Consumers
must not assume fixed channel indices.

Optional but important upstream inputs:

- `components/subject_body/qc/severe_qc_failure`
- `components/subject_body/qc/requires_review`
- `components/subject_body/qc/reason_bytes`
- component contours under refined subject-mask components

When severe subject-body QC is present, subject-shape fails closed for that row
instead of fitting body geometry over a known-bad mask.

## Processing Steps

The current row-local algorithm is:

1. Read refined masks for `subject_body`, `swim_bladder`, `eye_left`, and
   `eye_right`.
2. Compute component-local summaries: mask presence, area, centroid, bbox,
   principal axis for body, and ellipses for eyes/swim bladder.
3. Build the body frame from swim bladder and eye components.
4. Estimate `snout_tip_xy` from the subject-body contour point with maximum
   projection along the body-frame forward axis, tie-broken toward the body
   midline/lateral axis.
5. Estimate the caudal swim-bladder contour anchor from the swim-bladder
   contour point furthest caudally in the body frame.
6. Skeletonize the subject-body mask and extract the longest endpoint path.
7. Orient the skeleton path using body-frame polarity.
8. Select a medial head-region skeleton join point using body-frame lateral
   coordinates. This avoids joining the snout to lateral head-side skeleton
   branches.
9. Build a bounded mask path from `snout_tip_xy` to the selected skeleton join
   point. If the path is too long or cannot be found inside the body mask, the
   row is invalid.
10. Prepend that path to the remaining skeleton path and resample the full body
    centerline.
11. Project the caudal swim-bladder anchor onto the centerline to define
    `tail_base_xy`.
12. Fit a B-spline over the ordered centerline.
13. Sample the tail segment from `tail_base_xy` to `tail_tip_xy`, including
    tangents, normals, and curvature.

## Stored Outputs

Main arrays under:

```text
analysis/subject_shape_runs/<run>/components/subject_body/
```

Snout and endpoint arrays:

- `snout_tip_xy`
- `snout_tip_valid`
- `snout_tip_failure_reason_bytes`
- `head_endpoint_xy`
- `head_endpoint_to_snout_distance_px`
- `centerline_reaches_snout`
- `centerline_snout_check_reason_bytes`

Centerline and length arrays:

- `centerline_xy`
- `centerline_valid`
- `centerline_failure_reason_bytes`
- `body_arclength_px`
- `tail_tip_xy`
- `tail_base_xy`
- `tail_base_valid`
- `tail_base_arclength_px`
- `tail_base_failure_reason_bytes`
- `tail_segment_arclength_px`

B-spline and tail-sampling arrays:

- `bspline_control_points_xy`
- `bspline_knots`
- `bspline_degree_used`
- `bspline_sample_xy`
- `bspline_valid`
- `bspline_failure_reason_bytes`
- `bspline_arc_length_px`
- `tail_sample_s`
- `tail_sample_xy`
- `tail_tangent_xy`
- `tail_normal_xy`
- `tail_curvature_px_inv`
- `tail_sample_valid`
- `tail_sample_failure_reason_bytes`

The refined subject-mask run remains the canonical mask-pixel authority. These
subject-shape arrays are derived analysis products and should be regenerated
after mask edits rather than manually edited in place.

## Failure Reasons

Common row-local failure reasons:

- `source_body_mask_qc_failed`
- `missing_body_frame`
- `missing_subject_body_mask`
- `fragmented_subject_body_mask`
- `missing_subject_body_contour`
- `rostral_projection_failed`
- `missing_snout_tip`
- `skeleton_empty`
- `skeleton_endpoint_ambiguous`
- `centerline_order_failed`
- `endpoint_orientation_failed`
- `ambiguous_polarity`
- `snout_extension_too_long`
- `snout_extension_endpoint_outside_mask`
- `snout_extension_no_mask_path`
- `snout_extension_path_too_indirect`
- `missing_tail_anchor`
- `tail_base_projection_failed`
- `spline_fit_failed`
- `tail_segment_too_short`

`snout_extension_too_long` is currently the main remaining non-QC failure on the
feeding canary. That generally means the proposed snout-to-medial-skeleton
bridge is too long to trust as a local head correction.

## Canary Status

The current canary used during implementation:

```text
/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr
```

Source refined run:

```text
refined_subject_masks_smart_finalizer_dask_processes48_c64_canary_2026-04-26
```

Latest v8 canary run:

```text
subject_shape_v3_snout_medialjoin_canary_20260429
```

Observed summary:

- ROIs: `19235`
- `snout_tip_valid`: `17874`
- `centerline_valid`: `17496`
- `centerline_reaches_snout`: `17496`
- `bspline_valid`: `17496`
- `tail_base_valid`: `17496`
- `tail_sample_valid`: `17495`
- source body severe QC failures: `1360`
- remaining non-QC centerline failures: `378 snout_extension_too_long`

The v8 medial-join method fixed the row-46 issue where the B-spline followed a
head-side skeleton offshoot. It does this by joining the snout bridge into a
body-frame-guided medial head-region point instead of blindly using the first
skeleton endpoint.

## Running The Workflow

Example command:

```bash
scripts/py -m fisheye.analysis.subject_shape_runs \
  /nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr \
  --refined-run refined_subject_masks_smart_finalizer_dask_processes48_c64_canary_2026-04-26 \
  --run-name subject_shape_v3_snout_medialjoin_canary_20260429 \
  --chunk-size 256 \
  --execution-backend dask_worker_chunks \
  --scheduler processes \
  --num-workers 12 \
  --json
```

Per `AGENTS.md`, real Zarr runs and Zarr-heavy tests should be run outside the
Codex sandbox when possible.

## Review Visualizations

Generate overlays:

```bash
scripts/py -m fisheye.visualization.visualize_subject_shape_overlays \
  /nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr \
  --shape-run subject_shape_v3_snout_medialjoin_canary_20260429 \
  --refined-run refined_subject_masks_smart_finalizer_dask_processes48_c64_canary_2026-04-26 \
  --rows 0 21 46 424 929 1000 \
  --show-bspline \
  --show-spline-control-points \
  --show-tail-samples \
  --show-tail-normals \
  --output-dir /tmp/palette_subject_shape_v3_snout_medialjoin_overlays
```

Build a contact sheet:

```bash
montage /tmp/palette_subject_shape_v3_snout_medialjoin_overlays/subject_shape_overlay_subject_shape_v3_snout_medialjoin_canary_20260429_row_*.png \
  -tile 3x2 -geometry +8+8 \
  /tmp/palette_subject_shape_v3_snout_medialjoin_overlays/contact_sheet.png
```

Open it:

```bash
display /tmp/palette_subject_shape_v3_snout_medialjoin_overlays/contact_sheet.png
```

Overlay interpretation:

- White outline: subject-body contour.
- Cyan/red/blue outlines: swim bladder, left eye, right eye.
- Orange dot: `snout_tip_xy`. In v3/v8 valid rows this also equals
  `head_endpoint_xy`.
- Green line: resampled centerline.
- Pink line: B-spline sample.
- Pink diamonds: B-spline control points.
- Yellow star: caudal swim-bladder contour anchor.
- Yellow dot: `tail_base_xy`.
- Magenta `x`: `tail_tip_xy`.
- Cyan dots/yellow short lines: tail samples and tail normals.

The visualizer suppresses a separate head-endpoint marker when it exactly
overlaps the snout, because showing both markers made the point look ambiguous.

## Consumer Guidance

Consumers should:

- Treat `refined_subject_masks_runs` as the mask-pixel authority.
- Treat `analysis/subject_shape_runs` as derived geometry.
- Use run attrs to branch on `schema_version`, `method`, and
  `head_endpoint_semantics`.
- For current v3/v8 runs, use `head_endpoint_xy` as the snout-anchored anterior
  endpoint when `centerline_valid` is true.
- For older v2/v5 runs, do not assume `head_endpoint_xy` is the snout. Use
  `snout_tip_xy` and `head_endpoint_to_snout_distance_px` to audit the gap.
- Use `centerline_valid`, `bspline_valid`, and `tail_sample_valid` before
  consuming downstream arrays.
- Expect future multi-subject support to require track-aware row joins, but the
  current canary and present data are single-fish per arena.

## Known Limits

- Snout estimation is contour/body-frame based, not a learned nose keypoint.
- Severe body-mask artifacts can still drive false geometry unless the body QC
  layer catches them first.
- `snout_extension_too_long` rows need review before loosening thresholds.
- B-spline fitting is currently interpolating (`smoothing = 0.0`), so it follows
  the resampled centerline closely. If the centerline is noisy, the spline will
  be noisy too.
- Tail samples are geometric outputs, not yet a behavioral classifier. They are
  intended to support future Stytra, ZebraZoom, Megabouts, or Palette-native
  tail-analysis adapters.

## Next Work

Recommended next slices:

1. Add summary plots for body length, spline length, tail length, and failure
   reasons per subject-shape run.
2. Review the remaining `snout_extension_too_long` rows to decide whether they
   are mask/QC failures or threshold tuning cases.
3. Add persisted visualization artifacts under
   `analysis/subject_shape_runs/<run>/visualizations`.
4. Add tail-angle time-series and bout-aligned tail features from
   `tail_sample_xy`, `tail_tangent_xy`, and `tail_curvature_px_inv`.
   The first frame-level metric surface is specified in
   [tail_kinematics_run_design.md](tail_kinematics_run_design.md).
5. Add a Crimson read contract for subject-shape overlays after the current
   arrays stabilize.
