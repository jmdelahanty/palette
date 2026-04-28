# Eye-Angle Metrics: Data Layout and Computation

This note summarizes where the eye-angle products are written inside a Palette archive and how each quantity is derived from the upstream detections, keypoints, and refined eye geometry. It reflects the post-`1.4` pipeline where deltas and frame-level series are also emitted.

## Where the data lives

Eye-angle analysis runs are stored under:

```
analysis/eye_angle_runs/<run>/
    angles/
        roi/            # per-detection signals
        frame/          # optional frame-aligned signals
    qa/
        roi/
        frame/
    support/
```

Key datasets:

- `angles/roi/left_deg`, `right_deg`, `vergence_deg`, etc. hold the **unsigned** magnitudes (0–180° after the 2025-10 update) for each primary axis. Signed counterparts append `_signed_deg`.
- `angles/roi/left_centroid_deg`, `right_centroid_deg`, `vergence_centroid_deg` hold the **centroid-based** angles (paper-comparable methodology).
- `_delta_deg` and `_delta_deg_smoothed` arrays contain absolute frame-to-frame changes.
- `qa/roi/valid_left`, `valid_right`, `valid_frame`, and `reason_codes` provide flags and bitmasks that explain any exclusions.
- `support/time_seconds`, `frame_indices`, `ellipse_*` expose timing metadata and ellipse diagnostics used by the visualizations.

Current runs resolve eye geometry through `fisheye.shared.eye_geometry_source`.
The preferred source is `analysis/subject_shape_runs/<run>` when it contains
`eye_left` and `eye_right` component ellipse geometry. If no subject-shape
geometry is available, the resolver falls back to
`refined_subject_masks_runs/<run>` with eye component geometry, then to
historical `refined_eye_masks_runs/<run>` data as a compatibility fallback.

The referenced sources are captured in run attributes:

- `schema_id = "analysis.eye_angle_runs"` and `schema_version = 1`: stable
  run-level contract for this analysis product.
- `method = "ellipse_and_centroid_eye_angles"`: the writer computes both
  ellipse-axis and centroid-position eye-angle families.
- `row_axis = "keypoint_detection_rows"`: ROI outputs are row-aligned to the
  refined keypoint/eye-geometry detection rows.
- `eye_angle_output_schema`: machine-readable summary of output groups,
  row axes, units, suffix conventions, derivative outputs, and QA reason-code
  linkage.
- `source_eye_geometry_stage` and `source_eye_geometry_run`: the actual stage
  and run used for geometry.
- `source_geometry_kind`: normalized geometry role, one of
  `subject_shape_eye_geometry`, `refined_subject_eye_geometry`, or
  `legacy_refined_eye_geometry`; unknown future stages are recorded as
  `unknown_eye_geometry`.
- `source_subject_shape_run`: subject-shape source when analysis-facing shape
  geometry was used.
- `source_refined_subject_masks_run`: canonical refined-subject source when
  available.
- `source_refined_eye_run`: compatibility refined-eye source when one was used
  or mapped.
- `source_keypoints_run`: canonical refined keypoint source. The legacy
  `source_keypoint_run` alias may be mirrored during migration.

The raw ROIs sampled by the viewer live under `keypoints_runs/<run>/roi_images`.

Schema boundary:

- `schema_id` / `schema_version` identify the run family contract.
- `eye_angle_output_schema` describes the current output layout, units, suffix
  conventions, derivative arrays, reason-code links, and mixed support row
  axes. It is not a replacement for source lineage attrs.
- `source_geometry_kind` records which eye-geometry authority was actually
  consumed so readers do not have to infer semantics from path names alone.

## Execution model

`fisheye.analysis.eye_angle_analysis` writes base ROI outputs directly into the
target zarr run in row chunks. The default `serial_driver` backend processes
those chunks in the driver process. `--execution-backend dask_worker_chunks`
lets Dask workers reopen the archive and write disjoint ROI row chunks; use
`--scheduler threads`, `--scheduler processes`, or `--scheduler distributed`
depending on the workload.

Smoothing, deltas, speed, acceleration, and frame-level projection are computed
after the base ROI pass. Those second-pass products are intentionally
driver-side for now because smoothing and derivatives need adjacent rows across
chunk boundaries.

## Angle conventions

Angles are generated inside `fisheye.analysis.eye_angle_analysis._process_chunk`, which receives:

1. Keypoint ROIs (swim bladder, left/right eye centers).
2. Eye-geometry ellipse fits, preferably from `analysis/subject_shape_runs`.
3. Heading estimates exported by the keypoint run.

### Per-eye angles

For each detection and eye:

1. We build the **head direction** as the unit vector from the swim bladder to the midpoint between left and right eyes (`head_axis`).
2. The ellipse major-axis direction (`theta_deg`) is converted to a unit vector. To remove the 180° ambiguity, the vector is flipped so it points **temporally**—i.e., away from the midline. The sign of the resulting angle encodes temporal (positive) vs. nasal (negative) rotation.
3. The unsigned magnitude is simply `arccos(major_axis ⋅ head_axis)`, yielding values in `[0°, 180°]`. We no longer clamp at 90°, so values beyond right angles remain visible.
4. The minor-axis direction is produced by rotating the aligned major axis by 90°. It is flipped toward the temporal direction before the dot product with the head vector, producing the minor signed angles.

Invalid or near-circular fits are rejected early; reason bits (`REASON_*`) mark any failure so consumers can down-weight those detections.

### Binocular aggregates

Once left and right signed angles are available:

- We reinterpret the temporal-positive angles as nasal rotations by negating them (`left_nasal = -left_signed`). This keeps convergence defined as *both eyes turning nasally*.
- **Vergence (signed)** is `left_nasal + right_nasal`, recorded in metadata as `-(left_signed_deg + right_signed_deg)`. The unsigned magnitude is stored separately as `abs(vergence_signed_deg)`.
- **Version (signed)** is `0.5 * (left_nasal - right_nasal)`, recorded in metadata as `0.5*(-left_signed_deg + right_signed_deg)`.
- Minor axis variants follow the same algebra: `vergence_minor_signed_deg = -(left_minor_signed_deg + right_minor_signed_deg)` and `version_minor_deg = 0.5*(-left_minor_signed_deg + right_minor_signed_deg)`.

### Centroid-based angles (paper-comparable)

In addition to the ellipse-based angles, we compute **centroid-based** angles following the methodology of Johnson et al. (2020). These measure the *position* of each eye centroid relative to the fish's heading, rather than the *orientation* of the ellipse major axis.

For each detection:

1. Compute `head_center = mean(swim_bladder, left_eye, right_eye)`.
2. Build vectors from `head_center` to each eye centroid.
3. Convert to math coordinates (`y` flipped to point up) to match the heading convention.
4. Rotate into the fish frame by `-heading_rad` so the heading aligns with `+x`.
5. Compute per-eye angles: `theta_L = atan2(Ly, Lx)`, `theta_R = atan2(Ry, Rx)`.
6. Compute vergence: `vergence_centroid = |theta_L| + |theta_R|`.

This centroid-based vergence is directly comparable to the paper's definition and can be used with thresholds like 24° for hunting-state classification in downstream bout analysis.

Outputs:
- `angles/roi/left_centroid_deg`, `right_centroid_deg`, `vergence_centroid_deg`
- Smoothed and delta variants follow the same naming pattern as ellipse-based angles.
- Frame-level equivalents in `angles/frame/`.

### Deltas and smoothing

Absolute per-step changes (`*_delta_deg`) are computed with `_compute_delta`, preserving NaNs. When smoothing windows are configured, rolling averages are applied after the base computation, and the same delta routine is run on the smoothed series.

## Auxiliary products

- `support/ellipse_major`, `ellipse_minor`, and `ellipse_ratio` capture the geometric properties of the fitted ellipses and are useful for QA thresholds.
- Heading (in degrees) is re-serialized alongside the angles so downstream viewers can overlay the fish’s forward axis (`heading_deg`).
- Frame-level outputs repeat the same signals after the detections are resampled onto the video frame timeline; they live under `angles/frame/` with matching schema.

## Visual tools

- `fisheye.visualization.visualize_eye_angles` renders dashboards from an eye-angle run, including the unsigned/signed series, delta plots, and QA summaries.
- `fisheye.visualization.visualize_eye_angle_overlays` overlays masks, headings, and signed/unsigned values on the original ROIs.

Both tools automatically pick up the 0–180° range and the temporal-positive convention introduced here, so large rotations and backward-looking poses remain visible without additional configuration.
