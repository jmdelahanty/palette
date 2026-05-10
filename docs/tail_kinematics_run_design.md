# Tail Kinematics Run Design
<!-- contract-meta
version: 1
status: active
last_verified: 2026-05-01
-->

Purpose: define the first Palette-native tail-angle, tail-deflection, and
tail-curvature metric surface derived from ordered subject-shape tail samples,
while keeping future Megabouts/ZebraZoom/Stytra/BEAST adapters compatible but
non-canonical.

## Design Decision

Palette has a dedicated frame-level run family:

```text
analysis/tail_kinematics_runs/<run>
```

This run consumes an exact tail-geometry source, usually:

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

The current implementation requires a valid subject-shape run with:

```text
components/subject_body/bspline_sample_xy
components/subject_body/bspline_valid
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

Rows are valid only when the source row has a valid body frame and valid tail
geometry. The v1 behavior resamples from valid B-spline/tail geometry into the
lower-dimensional `tail_angle_sample_*` surface. Source failures propagate into
`failure_reason_bytes` rather than being
silently interpolated.

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

## Sampling Dimensionality

Tail kinematics should be lower-dimensional than the dense subject-shape
geometry surface.

Subject-shape runs may store:

- dense `bspline_sample_xy` for whole-body geometry and visualization
- compact `bspline_control_points_xy`
- subject-shape `tail_sample_xy` geometry samples, currently schema-v3
  geometry outputs
- dense or moderately dense curvature samples for geometry/QC

Tail-kinematics runs store behavior-facing tail samples separately. The default
is:

```text
tail_angle_sample_count = 10
tail_angle_sample_s = linspace(0.0, 1.0, 10)
```

where `0.0` is the tail base and `1.0` is the tail tip. These samples are the
markers that should drive the default Palette tail-angle/deflection vectors
shown to users and used by Palette-native summaries. External adapters may use
a different sample count, but they must record it explicitly.

Megabouts keypoint input uses a related but different count: 11 ordered
tail-curve points produce 10 Megabouts cumulative angle segments. Palette may
therefore generate a K=11 tail-kinematics candidate for comparison or adapter
symmetry, but the Megabouts adapter can also resample directly from
`subject_shape_runs` without changing Palette's default K=10 behavior-facing
tail-angle surface.

This split avoids using hundreds of dense spline evaluation points as a
behavior feature vector while preserving dense geometry for measurements that
need it.

## Frame-Level Metric Set

The current run exposes one trace group for the selected geometry source:

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
    source_tail_geometry_kind         "subject_shape_bspline_tail_resample"
    body_frame_convention
    tail_angle_reference_axis         "caudal_axis=-forward_axis"
    tail_angle_positive_direction     "anatomical_left"
    tail_angle_units_primary          "rad"
    tail_sample_domain                "tail_segment_normalized_arclength"
    tail_angle_sample_count           10 by default
    source_geometry_tail_sample_count optional
    curvature_source                  "subject_shape.tail_curvature_px_inv"
    created_at_utc

  frame_index                         (N,)
  time_s                              (N,) optional
  valid                               (N,)
  failure_reason_bytes                (N, width)

  tail_angle_sample_s                 (K,)
  tail_angle_sample_xy                (N, K, 2)
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

  source_refined_subject_masks/       optional copied source-revision snapshot
    row_revision                      (N, C)
    row_revision_available            (C,)
```

`tail_lateral_deflection_px` should be computed from each behavior-facing tail
angle sample relative to the tail base in body-frame coordinates:

```text
tail_lateral_deflection_px =
  dot(tail_angle_sample_xy - tail_base_xy, left_axis_xy)
```

This is a signed spatial deflection, not an angle. It is useful because some
users reason about tail-tip displacement more naturally than tangent angle.

`integrated_abs_tail_angle_rad` should integrate over normalized tail arclength
using `tail_angle_sample_s`; it is a compact "how bent is the tail now?" scalar,
not a frequency or movement classifier.

If the source subject-shape run has
`source_refined_subject_masks/row_revision`, the tail-kinematics writer should
copy that snapshot into its own `source_refined_subject_masks/` group. The tail
run is still downstream of the subject-shape run, but this copied revision table
keeps the refined-mask lineage auditable even if a consumer only has the tail
run selected.

Schema policy:

- `analysis.tail_kinematics_runs` schema v1 should define this low-dimensional
  `tail_angle_sample_*` behavior-facing surface before the first implementation
  ships.
- Existing `analysis.subject_shape_runs` schema v3 does not need to change if
  tail kinematics resamples from valid subject-shape geometry.
- If subject-shape itself changes the meaning or default dimensionality of
  `components/subject_body/tail_sample_xy`, then subject-shape should bump to
  schema v4 and a new subject-shape method version.

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
- `analysis/tail_kinematics_runs` for Palette-native tangent-angle review
  and comparison.
- `analysis/swim_bout_runs` when using Palette-selected bout windows.

Megabouts-compatible `tail_angle` should not be assumed to be Palette
`tail_angle_rad`. Palette `tail_angle_rad` is a local body-frame tangent angle
sampled along the tail. Megabouts `tail_angle` is a cumulative segment-angle
trace derived from ordered keypoints. The first Megabouts adapter should
therefore derive its `tail_angle` from `K=11` subject-shape tail keypoints via
Megabouts' own keypoint conversion, then compare it against Palette
`tail_angle_rad` only as an audit.

The implemented first step is a derived view, not duplicated permanent arrays
inside the native tail-kinematics run:

```text
Palette sources
  -> analysis/tail_posture_view_runs/<run>
  -> Megabouts runtime
  -> imported/classifier output run
```

`analysis/tail_posture_view_runs` is a compatibility artifact. It is
regenerated from Palette sources when needed and does not redefine Palette's
native `analysis/tail_kinematics_runs` schema.

Current v1 structure:

```text
analysis/tail_posture_view_runs/<run>/
  attrs:
    schema_id                         "analysis.tail_posture_view_runs"
    schema_version                    1
    method                            "tail_posture_view_from_subject_shape"
    method_version                    1
    row_axis                          "roi_rows"
    view_family                       "megabouts_compatible"
    compatible_tool                   "megabouts"
    dependency_policy                 "no_megabouts_dependency_required"
    source_subject_shape_run
    source_subject_shape_path
    source_refined_subject_masks_run
    source_tail_kinematics_run        optional comparison source
    source_tail_geometry_kind         "subject_shape_tail_curve_resample"
    head_source                       "head_endpoint_xy" | "snout_tip_xy"
    keypoint_count                    11
    angle_count                       10
    angle_convention                  "megabouts_cumulative_segment_angle"
    keypoint_order                    "tail_base_to_tail_tip"
    frame_index_source
    row_lineage_copied
    row_lineage_missing
    algorithm_provenance

  frame_index
  row_index/
    frame_indices                     copied when available
    detection_indices                 copied when available
    source_refined_row_ids            copied when available
    source_detect_row_index           copied when available
  valid
  failure_reason_bytes
  head_xy                             (N, 2)
  head_yaw_rad                        (N,)
  tail_keypoints_xy                   (N, 11, 2)
  tail_angle_rad                      (N, 10)
  tail_angle_deg                      (N, 10)
```

The first canary run was:

```text
tail_posture_view_megabouts_compatible_canary_20260501
source_subject_shape_run: subject_shape_v3_snout_medialjoin_canary_20260429
source_tail_kinematics_run: tail_kinematics_k10_canary_20260430
rows: 19,235
valid rows: 17,495
invalid rows: 1,740
duration: about 4.2 s
```

This run stores a Megabouts-compatible geometric view but does not run
Megabouts preprocessing, segmentation, or classification.

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

- [x] Implement `analysis/tail_kinematics_runs` writer that resamples valid
  subject-shape tail geometry into default `K=10` behavior-facing tail-angle
  samples.
- [x] Add unit tests for sign convention, straight-tail zero angle, left/right
  sign, and invalid-row propagation.
- [x] Run the writer on the feeding canary subject-shape run:
  `tail_kinematics_k10_canary_20260430` from
  `subject_shape_v3_snout_medialjoin_canary_20260429` wrote 17,495 valid rows
  and 1,740 invalid rows from 19,235 ROI rows.
- [ ] Persist PNG summaries for tail angles, tail-tip deflection, curvature,
  and validity/failure reasons.
- [ ] Add tail traces to the Marimo kinematics explorer after the Zarr schema
  stabilizes.
- [ ] Add per-bout tail summaries under `analysis/bout_kinematics_runs`.
- [x] Prototype a Megabouts-compatible posture view from the canary:
  `tail_posture_view_megabouts_compatible_canary_20260501` wrote 17,495 valid
  rows and 1,740 invalid rows from 19,235 ROI rows.
- [x] Add a Palette-owned optional Megabouts classifier adapter CLI. It records
  classifier outputs into Palette-native `analysis/bout_classification_runs`
  while keeping Megabouts itself an optional dependency.
- [x] Define `analysis/bout_classification_runs` for the first Megabouts
  classifier output. See
  [bout_classification_runs_contract.md](bout_classification_runs_contract.md).

## Open Questions

- Resolved for v1: store radians canonically and also write degree mirrors for
  plotting/review convenience. Radians remain the primary units.
- Resolved for first Megabouts-compatible view: do not pass Palette native
  tangent-angle samples directly as Megabouts `tail_angle`. Instead,
  `analysis/tail_posture_view_runs` resamples subject-shape tail geometry to
  `K=11` ordered tail keypoints and writes the `K=10` cumulative segment-angle
  representation expected by Megabouts-like tooling.
- Resolved for v1: mirror curvature into `tail_kinematics_runs` at the same
  low-dimensional `tail_angle_sample_s` positions and record the source in attrs.
- Resolved for v1: persist a sibling `analysis/tail_posture_view_runs` family
  rather than nesting external-tool arrays under `tail_kinematics_runs` or
  requiring external export files.
