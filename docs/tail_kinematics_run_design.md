# Tail Kinematics Run Design
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-29
-->

Purpose: define the first Palette-native tail-angle, tail-deflection, and
tail-curvature metric surface derived from ordered subject-shape tail samples,
while keeping future Megabouts/ZebraZoom/Stytra/BEAST adapters compatible but
non-canonical.

## Design Decision

Palette should add a dedicated frame-level run family:

```text
analysis/tail_kinematics_runs/<run>
```

This run should consume an exact tail-geometry source, usually:

```text
analysis/subject_shape_runs/<shape_run>/components/subject_body/
```

and produce tail posture metrics that are easier to plot, compare, export, and
summarize than raw geometry arrays.

`analysis/subject_shape_runs` remains the geometry authority:

- snout/head endpoint
- tail base and tail tip
- ordered centerline
- B-spline samples
- normalized tail samples
- tangents, normals, curvature

`analysis/tail_kinematics_runs` owns behavior-facing derived traces:

- tail angles
- lateral deflections
- scalar bend/curvature summaries
- optional temporal derivatives or frequency summaries in later versions

It should not mutate subject-shape runs, refined masks, swim-bout
segmentations, or external classifier outputs.

## Source Requirements

The first implementation should require a valid subject-shape run with:

```text
components/subject_body/tail_sample_s
components/subject_body/tail_sample_xy
components/subject_body/tail_tangent_xy
components/subject_body/tail_curvature_px_inv
components/subject_body/tail_sample_valid
components/subject_body/tail_base_xy
components/subject_body/tail_tip_xy
body_frame/forward_axis_xy
body_frame/left_axis_xy
body_frame/valid
```

Rows should be valid only when the source row has a valid body frame and valid
tail samples. Source failures should propagate into
`failure_reason_bytes` rather than being silently interpolated.

## Coordinate And Sign Convention

Palette's body frame currently defines:

- `forward_axis_xy`: posterior-to-anterior direction.
- `left_axis_xy`: anatomical left.

Tail samples are ordered from `tail_base_xy` to `tail_tip_xy`, so their natural
tail direction is caudal/posterior. Tail angles should therefore use:

```text
caudal_axis_xy = -forward_axis_xy
```

Recommended signed angle convention:

```text
tail_angle_rad =
  atan2(dot(tail_tangent_xy, left_axis_xy),
        dot(tail_tangent_xy, caudal_axis_xy))
```

This gives:

- `0`: tail tangent points straight caudally.
- positive values: bend toward anatomical left.
- negative values: bend toward anatomical right.

This convention should be recorded in attrs, not inferred from array names.
Degree arrays are useful for plotting, but the mathematical convention should
be defined once in radians.

## Frame-Level Metric Set

The initial run should expose one trace group for the selected geometry source:

```text
analysis/tail_kinematics_runs/<run>/
  attrs:
    schema_id                         "analysis.tail_kinematics_runs"
    schema_version                    1
    method                            "tail_metrics_from_subject_shape"
    method_version                    1
    row_axis                          "roi_rows"
    source_subject_shape_run
    source_refined_subject_masks_run
    source_tail_geometry_kind         "subject_shape_tail_samples"
    body_frame_convention
    tail_angle_reference_axis         "caudal_axis=-forward_axis"
    tail_angle_positive_direction     "anatomical_left"
    tail_angle_units_primary          "rad"
    tail_sample_domain                "tail_segment_normalized_arclength"
    tail_sample_count
    curvature_source                  "subject_shape.tail_curvature_px_inv"
    created_at_utc

  frame_index                         (N,)
  time_s                              (N,) optional
  valid                               (N,)
  failure_reason_bytes                (N, width)

  tail_sample_s                       (K,)
  tail_angle_rad                      (N, K)
  tail_angle_deg                      (N, K) optional plotting mirror
  tail_tip_angle_rad                  (N,)
  tail_tip_angle_deg                  (N,) optional plotting mirror

  tail_lateral_deflection_px          (N, K)
  tail_tip_lateral_deflection_px      (N,)
  tail_lateral_deflection_mm          (N, K) optional when calibrated
  tail_tip_lateral_deflection_mm      (N,) optional when calibrated

  max_abs_tail_angle_rad              (N,)
  max_abs_tail_angle_deg              (N,) optional plotting mirror
  tail_angle_rms_rad                  (N,)
  tail_angle_rms_deg                  (N,) optional plotting mirror
  integrated_abs_tail_angle_rad       (N,)

  tail_curvature_px_inv               (N, K)
  max_abs_tail_curvature_px_inv       (N,)
  integrated_abs_tail_curvature       (N,)
```

`tail_lateral_deflection_px` should be computed from each tail sample relative
to the tail base in body-frame coordinates:

```text
tail_lateral_deflection_px =
  dot(tail_sample_xy - tail_base_xy, left_axis_xy)
```

This is a signed spatial deflection, not an angle. It is useful because some
users reason about tail-tip displacement more naturally than tangent angle.

`integrated_abs_tail_angle_rad` should integrate over normalized tail arclength
using `tail_sample_s`; it is a compact "how bent is the tail now?" scalar, not
a frequency or movement classifier.

## What Not To Add Yet

Do not add these to v1 unless there is an immediate analysis need:

- tail-beat frequency
- phase
- dominant frequency
- temporal derivatives
- tail vigor
- bout-aligned tail arrays

Those metrics depend on temporal windows, smoothing, gap policy, and bout
selection. They are valid and important, but they should be added as explicit
method-versioned extensions rather than smuggled into the first geometry-to-
kinematics pass.

## Bout-Level Relationship

`analysis/tail_kinematics_runs` should be frame-level and independent of bout
segmentation.

Bout summaries should be written downstream, linked to exact sources:

```text
analysis/bout_kinematics_runs/<run>/
  attrs:
    source_tail_kinematics_run
    source_swim_bout_run
    source_swim_bout_speed_level

  tail/per_bout_metrics/
    bout_id
    source_start_frame
    source_end_frame
    source_start_time_s
    source_end_time_s
    max_abs_tail_angle_deg
    tail_tip_angle_peak_to_peak_deg
    tail_tip_lateral_deflection_peak_to_peak_px
    integrated_abs_tail_angle_mean_rad
    max_abs_tail_curvature_px_inv
    valid
    failure_reason_bytes
```

This keeps raw bout segmentation, frame-level tail traces, and per-bout
biological summaries separable and re-runnable.

## Megabouts Compatibility

Megabouts should be treated as an adapter and classifier consumer, not as the
canonical Palette schema.

The key boundary is:

```text
Palette owns reusable tail primitives.
Megabouts consumes a mapped view of those primitives.
Megabouts outputs return as imported classifier results.
```

Palette should therefore compute the general signals that Megabouts-like tools
need, such as ordered tail points, body-frame tail angles, tail-tip deflection,
curvature, and validity masks. It should not rename its canonical arrays or
change its sign/unit conventions just to match a specific external package.
External-tool conventions belong in explicit adapter attrs and export manifests.

Palette-native inputs for Megabouts can be generated from:

- `analysis/track_kinematics_runs` for `head_x`, `head_y`, `head_yaw`,
  position, and trajectory.
- `analysis/subject_shape_runs` for `tail_x`, `tail_y`.
- `analysis/tail_kinematics_runs` for `tail_angle`.
- `analysis/swim_bout_runs` when using Palette-selected bout windows.

The preferred first step is an export or view, not duplicated permanent arrays:

```text
Palette sources
  -> Megabouts adapter/export manifest
  -> Megabouts runtime
  -> imported/classifier output run
```

If we persist a Megabouts-ready view inside Zarr, it should be clearly marked as
a tool view:

```text
analysis/tail_kinematics_runs/<run>/tool_views/megabouts/
  attrs:
    tool_name                         "megabouts"
    tool_version
    source_tail_kinematics_run
    source_subject_shape_run
    source_track_kinematics_run
    fps
    units_xy
    invalid_row_policy                "nan"
    export_hash

  head_x
  head_y
  head_yaw
  tail_x
  tail_y
  tail_angle
  tracking_valid
```

This view is a compatibility artifact. It should be regenerated from Palette
sources when needed.

Megabouts outputs should land in classifier/import runs:

```text
analysis/bout_classification_runs/<run>/
  attrs:
    schema_id                         "analysis.bout_classification_runs"
    schema_version                    1
    classifier_family                 "megabouts"
    classifier_version
    source_tail_kinematics_run
    source_track_kinematics_run
    source_swim_bout_run              optional
    megabouts_config_json
    megabouts_export_hash

  per_bout/
    source_bout_id                    optional if Palette bouts used
    start_frame
    end_frame
    start_time_s
    end_time_s
    class_id
    class_label_bytes
    confidence                        optional
    valid
    failure_reason_bytes

  features/                           optional compact model-specific features
```

If Megabouts segments its own bouts, those boundaries belong in the
classification/import run. They should not overwrite `analysis/swim_bout_runs`.

This keeps Megabouts useful without making Palette dependent on Megabouts'
internal schema, model versions, dependency stack, or classifier taxonomy.

## Implementation Checklist

- [ ] Implement `analysis/tail_kinematics_runs` writer from subject-shape tail
  samples.
- [ ] Add unit tests for sign convention, straight-tail zero angle, left/right
  sign, and invalid-row propagation.
- [ ] Run the writer on the feeding canary subject-shape run.
- [ ] Persist PNG summaries for tail angles, tail-tip deflection, curvature,
  and validity/failure reasons.
- [ ] Add tail traces to the Marimo kinematics explorer after the Zarr schema
  stabilizes.
- [ ] Add per-bout tail summaries under `analysis/bout_kinematics_runs`.
- [ ] Prototype a Megabouts export manifest/view from the canary.
- [ ] Decide whether Megabouts execution should be Palette-owned CLI,
  user-run external tool, or both.
- [ ] Define `analysis/bout_classification_runs` once we have first real
  Megabouts output.

## Open Questions

- Should v1 store both radians and degrees, or store radians canonically and
  let plotting code convert to degrees?
- Should tail-angle arrays use one angle per tail sample `(K)` or one angle per
  segment `(K - 1)`? The first implementation can use sample tangents `(K)`
  because `subject_shape_runs` already stores `tail_tangent_xy`.
- Should curvature be mirrored into `tail_kinematics_runs`, or referenced from
  `subject_shape_runs` only? Mirroring is convenient and small; referencing is
  cleaner. The first implementation can mirror with explicit source attrs.
- Should the first Megabouts integration persist a Zarr tool view or only write
  external export files plus a manifest?
