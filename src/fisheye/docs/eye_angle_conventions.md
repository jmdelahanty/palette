# Eye-Angle Metrics: Data Layout and Computation

This note summarizes where the eye-angle products are written inside a Palette archive and how each quantity is derived from the upstream detections, keypoints, and refined eye-mask fits. It reflects the post-`1.4` pipeline where deltas and frame-level series are also emitted.

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
- `_delta_deg` and `_delta_deg_smoothed` arrays contain absolute frame-to-frame changes.
- `qa/roi/valid_left`, `valid_right`, `valid_frame`, and `reason_codes` provide flags and bitmasks that explain any exclusions.
- `support/time_seconds`, `frame_indices`, `ellipse_*` expose timing metadata and ellipse diagnostics used by the visualizations.

The refined keypoint and eye-mask runs referenced by the analysis are captured in the run attributes (`source_keypoint_run`, `source_refined_eye_run`), and the raw ROIs sampled by the viewer live under `keypoints_runs/<run>/roi_images`.

## Angle conventions

Angles are generated inside `fisheye.analysis.eye_angle_analysis._process_chunk`, which receives:

1. Keypoint ROIs (swim bladder, left/right eye centers).
2. Refined eye-mask ellipse fits and (optionally) Feret axes.
3. Heading estimates exported by the keypoint run.

### Per-eye angles

For each detection and eye:

1. We build the **head direction** as the unit vector from the swim bladder to the midpoint between left and right eyes (`head_axis`).
2. The ellipse major-axis direction (`theta_deg`) is converted to a unit vector. To remove the 180° ambiguity, the vector is flipped so it points **temporally**—i.e., away from the midline. The sign of the resulting angle encodes temporal (positive) vs. nasal (negative) rotation.
3. The unsigned magnitude is simply `arccos(major_axis ⋅ head_axis)`, yielding values in `[0°, 180°]`. We no longer clamp at 90°, so values beyond right angles remain visible.
4. The minor-axis direction is produced by rotating the aligned major axis by 90°. It is flipped toward the temporal direction before the dot product with the head vector, producing the minor signed angles.
5. If Feret major/minor axes are available, they undergo the same alignment (temporal-positive) before the angle computation.

Invalid or near-circular fits are rejected early; reason bits (`REASON_*`) mark any failure so consumers can down-weight those detections.

### Binocular aggregates

Once left and right signed angles are available:

- We reinterpret the temporal-positive angles as nasal rotations by negating them (`left_nasal = -left_signed`). This keeps convergence defined as *both eyes turning nasally*.
- **Vergence (signed)** is `left_nasal + right_nasal`. The unsigned magnitude is stored separately (`vergence_deg`).
- **Version (signed)** is `0.5 * (left_nasal - right_nasal)`.
- Minor and Feret variants follow the same algebra (`left_minor_signed`, `left_feret_major_signed`, …).

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
