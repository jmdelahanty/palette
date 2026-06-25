# GoodCopBadCop CRA Near-Field Avoidance Design
<!-- design-meta
status: draft
last_updated: 2026-06-21
-->

Purpose: define a modular near-field avoidance component for floor-projection
GoodCopBadCop recordings. This component extends the existing CRA primary
endpoint by measuring the lower tail of fish-to-object distance distributions:
close approaches, near-zone dwell, and near-zone entry events. It is a sibling
analysis product under the chaser-distance run, not a replacement for the
primary endpoint.

The motivating observation is that the avoidance effect may live within a few
millimetres of the projected object. Coarse quadrant occupancy and whole-arena
mean distance can dilute that effect because they integrate over much larger
spatial scales.

## Main Correction To The Initial Prompt

Do not ask this module to start from raw `fish_xy[t]` and `object_xy[t]` arrays
in millimetres.

The current stable dependency is the already-materialized GoodCopBadCop
chaser-distance and CRA stack:

```text
analysis/chaser_distance_runs/<run>/
  positions/fish_centroid_arena_xy
  positions/chaser_arena_xy
  positions/fish_valid
  positions/chaser_valid
  distances/distance_mm
  distances/distance_px
  frames/stimulus_epoch_window_id
  epoch_summary/*
  cra_primary_endpoint/<component>/
    objects/*
    phases/*
    object_phase/*
    per_object_phase/*
    summary/*
```

This matters because that stack has already done the hard, fragile work:

- refined fish detections have been mapped from camera image coordinates into
  the same arena-relative canvas frame as the stimulus chaser positions;
- fish-to-chaser distances have already been converted to millimetres using
  the run's `pixels_per_mm_projector`;
- GoodCopBadCop object roles have already been relabelled into
  `aggressive` and `benign`;
- `pre_static` and `post_static` effective windows have already been resolved,
  including the post-settle trim;
- primary endpoint dropout and quadrant summaries already exist and should be
  reused as companion and QC fields.

The near-field module should therefore consume the existing derived products
and fail loudly if they are missing or incompatible. It should not silently
recompute homography alignment or invent new phase windows.

## Current Data Contract

The inspected `/groups` GoodCopBadCop archive has:

```text
analysis/chaser_distance_runs/goodcopbadcop_chaser_distance_v1_20260617
  attrs.coordinate_frame = "arena_relative_canvas_px"
  attrs.coordinate_origin = "top_left_of_active_arena"
  attrs.pixels_per_mm_projector = 4.169337749481201
  attrs.fps = 100.0
  positions/chaser_arena_xy        shape (frame, object, xy), float32
  positions/chaser_valid           shape (frame, object), bool
  positions/fish_centroid_arena_xy shape (frame, xy), float32
  positions/fish_valid             shape (frame,), bool
  distances/distance_mm            shape (frame, object), float32
```

Its CRA primary endpoint component has:

```text
cra_primary_endpoint/object_relative_pre_post_v1
  attrs.schema_id = "palette.goodcopbadcop.cra_primary_endpoint.v1"
  attrs.status = "computed"
  objects/object_index
  objects/object_role_code
  objects/object_role_label_bytes
  objects/raw_color_rgba
  objects/raw_color_hex_bytes
  phases/phase_label_bytes              # pre_static, post_static
  phases/effective_start_frame
  phases/effective_end_frame
  object_phase/object_x_px
  object_phase/object_y_px
  object_phase/object_x_mm
  object_phase/object_y_mm
  per_object_phase/median_distance_mm
  per_object_phase/mean_distance_mm
  per_object_phase/occupancy_fraction
  per_object_phase/tracking_dropout_fraction
  summary/delta_agg
  summary/delta_benign
  summary/specificity_distance
  summary/delta_occ_agg
  summary/delta_occ_benign
  summary/specificity_occupancy
```

The near-field component should use those arrays as source authority.

## Storage Decision

Recommended component location:

```text
analysis/chaser_distance_runs/<run>/cra_near_field/<component_name>/
```

Recommended initial component name:

```text
object_relative_near_field_v1
```

This mirrors the existing protocol-specific component pattern under
`analysis/chaser_distance_runs/<run>`:

- `cra_primary_endpoint/<component>` stores object-relative pre/post occupancy
  and distance readouts;
- `egocentric_bearing/<component>` stores heading-relative object bearing
  readouts;
- `cra_near_field/<component>` should store lower-tail object-distance
  readouts.

## Source Preconditions

The writer should require:

- a complete chaser-distance run under `analysis/chaser_distance_runs/<run>`;
- a complete CRA primary endpoint component under that run;
- `coordinate_frame == "arena_relative_canvas_px"`;
- `coordinate_origin == "top_left_of_active_arena"`;
- finite positive `pixels_per_mm_projector`;
- `distances/distance_mm` with shape `(frame, object)`;
- `positions/fish_centroid_arena_xy` and `positions/fish_valid`;
- `positions/chaser_arena_xy` and `positions/chaser_valid`;
- exactly one `aggressive` and one `benign` object in the source CRA component
  for v1;
- `pre_static` and `post_static` phase rows from the source CRA component.

The writer should reject `training` for all near-field metrics. Training is
stimulus delivery, not the pre/post avoidance readout.

## Coordinate And Unit Policy

Distance-based metrics should read `distances/distance_mm` directly. That is
the authoritative fish-to-object distance in millimetres for this workflow.

Position-based diagnostics should read:

```text
positions/fish_centroid_arena_xy
positions/chaser_arena_xy
```

These arrays are arena-relative canvas pixels. They can be converted to
millimetres only for derived geometry that requires physical units, using:

```text
distance_or_position_delta_mm = delta_px / pixels_per_mm_projector
```

Do not reinterpret `fish_centroid_arena_xy` as physical millimetres. The name
means the fish centroid has been mapped into the arena-local canvas coordinate
frame.

## Metrics

All metrics are computed per fish recording, per phase, per object role, with
phase axis:

```text
pre_static
post_static
```

and object role axis:

```text
aggressive
benign
```

### Close-Approach Distance

For each `(phase, object)`, compute lower percentiles of valid
`distance_mm` samples:

```text
approach_p05_mm
approach_p10_mm
```

The percentile list should be configurable. The initial default should include
5 and 10 percent.

### Near-Zone Occupancy

For each `(phase, object)`, compute:

```text
near_zone_occupancy_fraction = count(distance_mm <= r_zone_mm) / valid_distance_count
near_zone_occupancy_fraction_of_epoch = count(distance_mm <= r_zone_mm) / total_frame_count
near_zone_dwell_s = count(distance_mm <= r_zone_mm) / fps
```

The near-zone radius should be a configuration value, not a tuned constant.
The first implementation can default to a conservative few-mm value, but every
written component must store the exact configured value.

### Area-Normalized Near-Zone Occupancy

When a valid arena geometry is available, also compute an area-normalized
near-zone density:

```text
near_zone_density_per_mm2 =
  near_zone_occupancy_fraction / available_near_zone_area_mm2
```

The available area must be the intersection of the object-centred zone with
the fish-accessible arena. Do not use raw circle area if the zone crosses a
wall.

For the current GoodCopBadCop archives, prefer the circular experimental-area
metadata from:

```text
analysis/stimulus_runs/<run>/calibration/arena_geometry.attrs[
  experimental_area_shape,
  experimental_area_center_x_px,
  experimental_area_center_y_px,
  experimental_area_radius_px,
  experimental_area_radius_mm
]
```

The inspected `/groups` archive reports `experimental_area_shape = "CIRCLE"`,
center `(172, 172)` in arena-relative canvas pixels, and radius `166` px.
Use this circular disk for area-normalized radial densities and wall-band
thigmotaxis QC.

If dish center/radius or a trusted arena polygon is missing, the writer should
either:

- fail in strict geometry mode, or
- write `arena_geometry_status = "rectangular_approximation"` and mark
  area-normalized values as diagnostic, not confirmatory.

It should not silently pretend the arena rectangle is the dish.

### Close-Approach Entries

For each `(phase, object)`, count distinct entries into the near zone using
hysteresis:

```text
enter when distance_mm < r_in_mm
exit  when distance_mm > r_out_mm
```

with `r_out_mm > r_in_mm`.

Store:

```text
near_zone_entry_count
near_zone_entry_rate_per_min
near_zone_visit_median_dwell_s
near_zone_visit_total_dwell_s
```

Visits should ignore invalid-distance gaps conservatively. V1 should close any
active visit across a sufficiently long invalid gap and record the invalid-gap
policy in parameters.

### Radial Occupancy Density Diagnostic

For each `(phase, object, radial_bin)`, bin `distance_mm` into annuli.

Store both raw and area-normalized diagnostics:

```text
radial_count
radial_fraction
radial_density_per_mm2
radial_available_area_mm2
```

The bin edges should be configurable. Default bins should be fine near the
object, for example 1 to 2 mm bins, and can be coarser farther away.

Area normalization should use the same geometry policy as near-zone density.
If only rectangular bounds are available, label the density output as an
approximation.

### Distance CDF Diagnostic

For each `(phase, object, threshold)`, compute:

```text
cdf_fraction = P(distance_mm <= threshold_mm)
```

Thresholds should be configurable and written to the component.

### Thigmotaxis QC

If trusted arena geometry is available, compute fraction of phase frames within
a configurable wall band:

```text
thigmotaxis_fraction
thigmotaxis_dwell_s
```

Use fish positions in arena-relative canvas pixels and convert the wall-band
width from millimetres to pixels. If only rectangular bounds are available,
label the result as rectangular perimeter occupancy. If no trusted geometry is
available, write missing values and a QC warning rather than fabricating a
dish-wall covariate.

## Suggested Zarr Layout

```text
analysis/chaser_distance_runs/<run>/cra_near_field/<component_name>/
  zarr.json
  config/
    percentile_values
    radial_bin_edges_mm
    radial_bin_centers_mm
    cdf_thresholds_mm
  objects/
    object_index
    object_role_code
    object_role_label_bytes
    raw_color_rgba
    raw_color_hex_bytes
  phases/
    phase_index
    phase_label_bytes
    effective_start_frame
    effective_end_frame
    total_frame_count
  per_object_phase/
    approach_percentile_mm              # shape (phase, object, percentile)
    near_zone_occupancy_fraction        # shape (phase, object)
    near_zone_occupancy_fraction_of_epoch
    near_zone_dwell_s
    near_zone_density_per_mm2
    near_zone_available_area_mm2
    near_zone_entry_count
    near_zone_entry_rate_per_min
    near_zone_visit_median_dwell_s
    near_zone_visit_total_dwell_s
    valid_distance_count
    missing_frame_count
    tracking_dropout_fraction
  radial_density/
    radial_count                        # shape (phase, object, radial_bin)
    radial_fraction
    radial_density_per_mm2
    radial_available_area_mm2
  distance_cdf/
    cdf_fraction                        # shape (phase, object, threshold)
  thigmotaxis/
    thigmotaxis_fraction                # shape (phase,)
    thigmotaxis_dwell_s
    geometry_status_bytes
  summary/
    fish_id_bytes
    recording_id_bytes
    aggressive_color_bytes
    benign_color_bytes
    approach_p05_pre_agg
    approach_p05_post_agg
    approach_p05_delta_agg
    approach_p05_pre_benign
    approach_p05_post_benign
    approach_p05_delta_benign
    approach_p05_specificity
    approach_p10_pre_agg
    approach_p10_post_agg
    approach_p10_delta_agg
    approach_p10_pre_benign
    approach_p10_post_benign
    approach_p10_delta_benign
    approach_p10_specificity
    nearzone_occ_pre_agg
    nearzone_occ_post_agg
    nearzone_occ_delta_agg
    nearzone_occ_pre_benign
    nearzone_occ_post_benign
    nearzone_occ_delta_benign
    nearzone_occ_specificity
    nearzone_entry_rate_pre_agg
    nearzone_entry_rate_post_agg
    nearzone_entry_rate_delta_agg
    nearzone_entry_rate_pre_benign
    nearzone_entry_rate_post_benign
    nearzone_entry_rate_delta_benign
    nearzone_entry_rate_specificity
    delta_occ_agg                       # copied from CRA primary endpoint
    delta_occ_benign                    # copied from CRA primary endpoint
    specificity_occupancy               # copied from CRA primary endpoint
    thigmotaxis_frac_pre
    thigmotaxis_frac_post
    frac_tracking_dropout_pre
    frac_tracking_dropout_post
    qc_warnings_json_bytes
  visualizations/
    cra_near_field_radial_density_png
    cra_near_field_distance_cdf_png
    cra_near_field_summary_png
    cra_near_field_interactive
```

The exact scalar summary field names may be expanded by percentile value. Keep
the long table stable and machine-readable, even if the marimo viewer renders
a friendlier layout.

## Component Attributes

The component attrs should include:

```text
schema_id = "palette.goodcopbadcop.cra_near_field.v1"
schema_version = 1
method = "goodcopbadcop_object_relative_near_field"
method_version = "1"
status = "computed" | "failed_qc" | "skipped"
row_axis = "fish_recording"
coordinate_frame = "arena_relative_canvas_px"
coordinate_origin = "top_left_of_active_arena"
pixels_per_mm_projector = <float>
source_refs = {
  source_chaser_distance_run,
  source_chaser_distance_path,
  source_cra_primary_endpoint_component,
  source_cra_primary_endpoint_path,
  source_stimulus_run,
  source_stimulus_path,
  source_stimulus_epoch_run,
  source_stimulus_epoch_path
}
parameters = {
  r_zone_mm,
  r_in_mm,
  r_out_mm,
  percentile_values,
  radial_bin_edges_mm,
  cdf_thresholds_mm,
  perimeter_band_mm,
  geometry_mode,
  invalid_gap_policy
}
summary = {...}
qc_warnings = [...]
provenance = {...}
```

Use the same run-lineage fingerprint pattern used by the existing
`cra_primary_endpoint` and `egocentric_bearing` components.

## Group Export Design

The cross-recording export should add tables alongside the existing
GoodCopBadCop CRA tables:

```text
goodcopbadcop_cra_near_field_summary
goodcopbadcop_cra_near_field_object_phase
goodcopbadcop_cra_near_field_radial_density
goodcopbadcop_cra_near_field_cdf
```

Recommended table grain:

- `summary`: one row per fish recording;
- `object_phase`: one row per fish recording, phase, object role;
- `radial_density`: one row per fish recording, phase, object role, radial bin;
- `cdf`: one row per fish recording, phase, object role, distance threshold.

Every exported row should carry explicit provenance fields:

```text
recording_id
zarr_path
chaser_distance_run_name
chaser_distance_run_path
cra_primary_endpoint_component_name
cra_primary_endpoint_component_path
cra_near_field_component_name
cra_near_field_component_path
schema_id
method
method_version
parameters_json
```

## Statistics And Viewer Design

Statistics should use fish as the unit of analysis. For each selected metric,
compute:

```text
delta_agg = post_agg - pre_agg
delta_benign = post_benign - pre_benign
specificity = delta_agg - delta_benign
```

Use the existing GoodCopBadCop group-statistics infrastructure for:

- paired Wilcoxon signed-rank tests;
- matched-pairs rank-biserial effect size;
- bootstrap confidence interval on the median paired difference.

Viewer support should be added at two levels:

- per-recording marimo Palette Explorer: radial density, distance CDF,
  pre/post near-zone summary, and the scalar per-recording table;
- exported group viewer: pooled radial density/CDF by object role and phase,
  paired pre-to-post summary plots, and a table of effect sizes and confidence
  intervals.

## Parameter Guardrail

Expose these as configuration:

```text
r_zone_mm
r_in_mm
r_out_mm
percentile_values
radial_bin_edges_mm
cdf_thresholds_mm
perimeter_band_mm
geometry_mode
```

Do not tune these parameters to minimize p-values or maximize effects. The
diagnostic radial-density and CDF plots are for human inspection and later
pre-specification. If we want discovery and confirmation in the same dataset,
the held-out split belongs in the group/export analysis layer, not inside the
per-recording zarr component.

## Implementation Checklist

- [ ] Confirm source paths and attrs on one `/groups` GoodCopBadCop zarr.
- [ ] Add unit-tested compute helpers for percentiles, near-zone occupancy,
  hysteresis entries, radial density, CDF, and thigmotaxis geometry status.
- [ ] Add `src/fisheye/analysis/cra_near_field.py` with a dataclass result,
  source resolver, compute function, zarr writer, PNG renderers, and CLI.
- [ ] Reuse CRA primary endpoint objects/phases instead of redefining object
  roles or pre/post windows.
- [ ] Store source refs, parameters, QC warnings, and run-lineage attrs.
- [ ] Add focused in-memory tests for all compute helpers and writer layout.
- [ ] Run the writer on one real `/groups` GoodCopBadCop archive outside the
  Codex sandbox.
- [ ] Inspect the written zarr component and exported PNGs.
- [ ] Add a registry wrapper to backfill eligible `/groups` GoodCopBadCop
  recordings.
- [ ] Add cross-recording export tables for summary, object-phase,
  radial-density, and CDF outputs.
- [ ] Extend GoodCopBadCop group statistics to include near-field summary
  metrics.
- [ ] Add per-recording marimo Palette Explorer panels for the near-field
  component.
- [ ] Add group viewer panels for exported near-field pooled distributions and
  statistics.
- [ ] Backfill all eligible `/groups` GoodCopBadCop zarrs once one-recording
  validation passes.

## Acceptance Criteria

1. The per-recording near-field component can be written using only the
   existing chaser-distance run and CRA primary endpoint component.
2. The component never recomputes camera-to-projector registration.
3. The component excludes training and uses only `pre_static` and
   `post_static`.
4. Metrics are role-resolved for aggressive and benign objects before
   aggregation.
5. Lower-tail metrics, radial diagnostics, CDF diagnostics, and thigmotaxis QC
   are persisted with parameters and provenance.
6. Cross-recording exports expose both scalar summaries and pooled diagnostic
   distributions.
7. Per-recording and grouped viewers can visualize the near-field outputs
   without recomputing the analysis.
