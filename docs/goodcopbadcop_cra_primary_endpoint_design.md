# GoodCopBadCop CRA Primary Endpoint Design
<!-- design-meta
status: draft
last_updated: 2026-06-21
-->

Purpose: define the primary, pre-specified CRA occupancy/distance endpoint for
floor-projection GoodCopBadCop recordings. This endpoint is intentionally
narrow: it measures how the fish relates to the aggressive and benign projected
objects during pre and post phases. Training is stimulus delivery and is not
part of the confirmatory readout.

This document is an implementation design/checklist, not an implemented
schema. It deliberately does not attempt to mirror paper STAR Methods
parameters yet; Palette already has the local frame rate and event windows, and
the first goal is to encode the assay-specific endpoint correctly from our
imported data.

## Current Data Inventory

The required object metadata already exists in imported H5-derived stimulus
runs.

Example inspected archive:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop/zarr/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop_analysis.zarr
```

Relevant surfaces:

```text
analysis/stimulus_runs/<stimulus_run>/zarr.json
analysis/stimulus_runs/<stimulus_run>/tracking_data/chaser_states/
analysis/stimulus_runs/<stimulus_run>/stimulus_coordinates/arena_1/
analysis/stimulus_epoch_runs/<epoch_run>/windows/
analysis/chaser_distance_runs/<chaser_distance_run>/
```

The stimulus run attrs include `protocol_json`. For GoodCopBadCop this contains:

- `steps[].parameters.chasers[]`
- per-chaser `enable_chase`
- per-chaser `behavior_mode`
- per-chaser `color_r`, `color_g`, `color_b`, `color_a`
- per-chaser `start_position_preset`, `start_position_x`, `start_position_y`
- per-chaser `end_position_preset`, `end_position_x`, `end_position_y`
- shared `pre_period_duration_s`, `training_period_duration_s`,
  `post_period_duration_s`
- shared `position_transition_duration_s`
- shared `pixels_per_mm`

The inspected recording has:

```text
chaser 0: enable_chase=true,  start=top_left,  end=bottom_right
chaser 1: enable_chase=false, start=top_right, end=bottom_left
```

The imported `tracking_data/chaser_states` group stores columnar frame-level
object state. Inspected attrs report:

```text
coordinate_frame = arena_relative_canvas_px
coordinate_origin = top_left_of_active_arena
position_fields = chaser_pos_x,chaser_pos_y,target_pos_x,target_pos_y,target_clamped_pos_x,target_clamped_pos_y
units = px
x_axis_direction = right
y_axis_direction = down
```

Key chaser-state arrays include:

```text
stimulus_frame_num
timestamp_ns_session
chaser_index
is_chasing
chaser_pos_x
chaser_pos_y
trial_state
chase_sequence_active
pixels_per_mm
```

The existing chaser-distance run has already resolved the heavy coordinate
join:

```text
analysis/chaser_distance_runs/<run>/
  positions/fish_centroid_arena_xy
  positions/chaser_arena_xy
  positions/fish_valid
  positions/chaser_valid
  distances/distance_mm
  frames/stimulus_epoch_window_id
  epoch_summary/*
```

It stores:

```text
coordinate_frame = arena_relative_canvas_px
coordinate_origin = top_left_of_active_arena
pixels_per_mm_projector = <float>
source_stimulus_epoch_path = analysis/stimulus_epoch_runs/<epoch_run>
```

## Design Decision

Implement the primary endpoint as a component under the existing
`analysis/chaser_distance_runs/<run>` surface, not as a standalone visualization
run family.

Recommended location:

```text
analysis/chaser_distance_runs/<run>/cra_primary_endpoint/<component_name>/
```

Rationale:

- the endpoint's direct inputs are already aligned inside the chaser-distance
  run: fish position, object position, validity masks, distance in mm, and
  stimulus epoch ids;
- the endpoint is object-relative and protocol-specific, so it should not be
  confused with generic fixed spatial occupancy zones;
- keeping it as a component follows the existing modular pattern used for
  egocentric bearing components under chaser-distance runs;
- visualizations can render from this component, but the component itself is
  the scientific analysis product.

## Coordinate Contract

All endpoint calculations use the chaser-distance run coordinate frame:

```text
coordinate_frame = arena_relative_canvas_px
coordinate_origin = top_left_of_active_arena
x_axis_direction = right
y_axis_direction = down
distance_unit = mm
```

Distances should use `distances/distance_mm` from the source chaser-distance
run whenever possible. Occupancy should use `positions/fish_centroid_arena_xy`
and the per-phase object location from `positions/chaser_arena_xy`.

Quadrant definitions must be in the same arena-relative canvas frame. The
preferred bounds source is:

```text
analysis/stimulus_runs/<stimulus_run>/stimulus_coordinates/arena_1.attrs[
  texture_width_px,
  texture_height_px,
  texture_origin
]
```

For the inspected data this is a top-left-origin `344 x 344` arena texture.
Quadrants are split at `width / 2` and `height / 2`. Midline points follow the
existing deterministic convention: right/bottom owns the midpoint.

## Object Role Resolution

Relabel raw chaser indices into object roles before aggregation:

```text
enable_chase == true  -> aggressive
enable_chase == false -> benign
```

V1 expects exactly one aggressive and one benign object. If the protocol has a
different number or ambiguous flags, the endpoint writer should fail for that
recording unless an explicit role override is supplied.

Store raw identity for QC:

```text
object_index
object_role              # aggressive | benign
raw_color_rgba
raw_color_hex
start_position_preset
end_position_preset
enable_chase
behavior_mode
```

Color counterbalancing is a QC dimension only. Group-level analyses operate on
`object_role`, not raw color.

## Phase Windows

Input windows come from:

```text
analysis/stimulus_epoch_runs/<epoch_run>/windows/
```

Use:

```text
pre_event
post_event
```

Exclude:

```text
training_event
```

V1 phase policy:

- `pre_static`: use the full `pre_event` window.
- `post_static`: use `post_event` after object settling.

The post-settle exclusion should use the protocol parameter:

```text
position_transition_duration_s
```

Frame policy:

```text
post_static_start_frame =
  post_event.start_frame + ceil(position_transition_duration_s * fps)
```

If `position_transition_duration_s` is missing, use `0` and record a diagnostic.
The effective post window must remain non-empty after trimming.

## Static Object Position Policy

The design assumption is:

- chasers are static in pre;
- chasers settle at the beginning of post;
- chasers are static after settling.

For each phase and object, compute the phase object position as the median of
valid framewise `chaser_arena_xy` samples in the effective phase window.

Record drift diagnostics:

```text
object_phase_x_px
object_phase_y_px
object_phase_x_mm
object_phase_y_mm
object_position_sample_count
object_max_drift_px
object_max_drift_mm
object_median_drift_mm
```

The first implementation should report drift rather than exclude on drift.
Later, a pre-registered static-position tolerance can promote large drift to a
QC failure.

## Primary Metrics

Per fish/recording, phase, and object role:

```text
median_distance_mm = median(distance_mm[effective_phase, object])
mean_distance_mm = mean(distance_mm[effective_phase, object])
occupancy_fraction = valid fish frames in object's phase quadrant / valid fish frames
occupancy_fraction_of_epoch = valid fish frames in object's phase quadrant / total effective phase frames
valid_frame_count
total_frame_count
missing_frame_count
tracking_dropout_fraction
```

Use `occupancy_fraction` for the primary object-quadrant readout because it
matches spatial preference among tracked frames. Also store
`occupancy_fraction_of_epoch` so missing detections remain visible.

Object-quadrant occupancy is phase-relative:

```text
pre aggressive occupancy  = fish in aggressive object's pre quadrant during pre_static
post aggressive occupancy = fish in aggressive object's post quadrant during post_static
pre benign occupancy      = fish in benign object's pre quadrant during pre_static
post benign occupancy     = fish in benign object's post quadrant during post_static
```

Do not compare fixed quadrants across phases for the primary endpoint.

## Derived Wide Confirmatory Row

One row per fish/recording should expose the paper-facing endpoint shape:

```text
fish_id
recording_id
zarr_path
dpf
aggressive_color
benign_color

d_pre_agg
d_post_agg
delta_agg
d_pre_benign
d_post_benign
delta_benign
specificity_distance

occ_pre_agg
occ_post_agg
delta_occ_agg
occ_pre_benign
occ_post_benign
delta_occ_benign
specificity_occupancy

n_valid_frames_pre
n_valid_frames_post
frac_tracking_dropout_pre
frac_tracking_dropout_post

pre_aggressive_quadrant
post_aggressive_quadrant
pre_benign_quadrant
post_benign_quadrant

endpoint_status
diagnostics_json
```

Derived definitions:

```text
delta_agg = d_post_agg - d_pre_agg
delta_benign = d_post_benign - d_pre_benign
specificity_distance = delta_agg - delta_benign

delta_occ_agg = occ_post_agg - occ_pre_agg
delta_occ_benign = occ_post_benign - occ_pre_benign
specificity_occupancy = delta_occ_agg - delta_occ_benign
```

For interpretability, group plots may also expose avoidance-positive occupancy:

```text
avoidance_delta_occ_agg = -delta_occ_agg
avoidance_specificity_occupancy = -specificity_occupancy
```

but these should be clearly labeled as sign-flipped display variables.

## Recommended Component Storage

```text
analysis/chaser_distance_runs/<run>/cra_primary_endpoint/<component_name>/
  zarr.json
  objects/
    object_index
    object_role_code
    object_role_label_bytes
    raw_color_rgba
    raw_color_hex_bytes
    enable_chase
    behavior_mode
    start_position_preset_bytes
    end_position_preset_bytes
  phases/
    phase_index
    phase_label_bytes              # pre_static, post_static
    source_window_label_bytes      # pre_event, post_event
    source_start_frame
    source_end_frame
    effective_start_frame
    effective_end_frame
    settle_excluded_frame_count
  object_phase/
    object_x_px                    # phase x object
    object_y_px
    object_x_mm
    object_y_mm
    object_quadrant_code
    object_quadrant_label_bytes
    object_position_sample_count
    object_max_drift_mm
    object_median_drift_mm
  per_object_phase/
    median_distance_mm             # phase x object
    mean_distance_mm
    occupancy_fraction
    occupancy_fraction_of_epoch
    valid_frame_count
    total_frame_count
    missing_frame_count
    tracking_dropout_fraction
  summary/
    fish_id_bytes
    recording_id_bytes
    d_pre_agg
    d_post_agg
    delta_agg
    d_pre_benign
    d_post_benign
    delta_benign
    specificity_distance
    occ_pre_agg
    occ_post_agg
    delta_occ_agg
    occ_pre_benign
    occ_post_benign
    delta_occ_benign
    specificity_occupancy
  visualizations/
    cra_primary_endpoint_overview_png
    cra_primary_endpoint_interactive
```

Run attrs should include:

```text
schema_id = palette.goodcopbadcop.cra_primary_endpoint.v1
schema_version = 1
method = goodcopbadcop_object_relative_pre_post_endpoint
method_version = 1
row_axis = fish_recording
source_chaser_distance_run = <run>
source_chaser_distance_path = analysis/chaser_distance_runs/<run>
source_stimulus_run = <stimulus_run>
source_stimulus_path = analysis/stimulus_runs/<stimulus_run>
source_stimulus_epoch_run = <epoch_run>
source_stimulus_epoch_path = analysis/stimulus_epoch_runs/<epoch_run>
coordinate_frame = arena_relative_canvas_px
coordinate_origin = top_left_of_active_arena
quadrant_bounds_source = stimulus_coordinates/arena_1
post_settle_policy = trim_position_transition_duration_s
dropout_exclusion_policy = report_only
```

## QC Policy

Initial v1 should not hard-exclude on tracking dropout because the threshold is
not selected yet. Instead:

- compute dropout fractions for pre and post;
- mark `endpoint_status = computed` if minimum required source data exists;
- include `qc_warnings` for high dropout using a configurable warning threshold;
- leave `qc_excluded = false` unless a user supplies a hard threshold.

Proposed config fields:

```text
dropout_warning_fraction = 0.20
dropout_exclusion_fraction = null
static_object_drift_warning_mm = 1.0
post_settle_duration_policy = protocol_position_transition_duration_s
```

Before this endpoint becomes final confirmatory analysis, choose and freeze the
dropout exclusion threshold in a protocol-level analysis config.

## Group Export

Add compact export tables after the per-recording component is stable:

```text
goodcopbadcop_cra_primary_endpoint_summary
goodcopbadcop_cra_primary_endpoint_object_phase
```

The summary table is one row per fish/recording. The object-phase table is long
format:

```text
recording x fish x phase x object_role
```

The group-statistics layer should then compute:

- paired Wilcoxon signed-rank for `delta_agg`;
- paired Wilcoxon signed-rank for `delta_occ_agg`;
- paired Wilcoxon signed-rank for `specificity_distance`;
- paired Wilcoxon signed-rank for `specificity_occupancy`;
- benign deltas as negative-control summaries.

The current group statistics module uses sign-flip tests. The CRA primary
endpoint needs a Wilcoxon method before final confirmatory reporting.

## Visualization Surfaces

Visualization is required at two levels: per-recording inspection and exported
group/cohort inspection. Both should render from persisted endpoint arrays or
Parquet export tables. They should not recompute endpoint values in the UI.

The implementation has two separate visualization contracts:

1. **Per-recording contract:** inspect one recording's persisted CRA endpoint
   component inside the Palette marimo explorer.
2. **Exported/group contract:** inspect merged CRA endpoint tables and
   provenance-linked statistics inside the group analytics viewer.

These are intentionally different surfaces. The marimo explorer answers "does
this recording's object-relative endpoint look correct?" The group viewer
answers "what does this endpoint look like across animals/recordings?" Both
must use the same stored endpoint schema so a value shown for a single
recording can be traced into the exported dataset.

### Explicit Visualization Contract

The source-of-truth boundary is:

```text
single recording:
  zarr component -> marimo panel

merged cohort:
  exported Parquet tables -> group analytics viewer

statistics:
  exported Parquet tables -> group statistics artifact -> group analytics viewer
```

The per-recording marimo panel may read zarr arrays directly because it is an
inspection tool for one selected recording. It must read from the persisted CRA
component and linked source arrays; it must not recompute object roles, phase
trimming, object quadrants, endpoint deltas, or specificity contrasts in the
notebook.

The group analytics viewer must not read individual recording zarrs. It should
only read the merged export tables and matching statistics artifacts. This keeps
group plots reproducible from the exported artifact alone and makes it possible
to share or archive cohort-level results without depending on live `/groups`
recording paths.

The run-level mirrored interactive spec, if present, is only an index/discovery
handle for marimo. The canonical scientific artifact remains:

```text
analysis/chaser_distance_runs/<run>/cra_primary_endpoint/<component_name>/
```

Values shown in any UI should be explainable by one of these stored tables:

```text
component/summary/*                         -> per-recording summary cards/table
component/object_phase/*                    -> object positions, quadrants, drift
component/per_object_phase/*                -> phase x role distance/occupancy rows
goodcopbadcop_cra_primary_endpoint_summary  -> group deltas/specificity
goodcopbadcop_cra_primary_endpoint_object_phase -> group phase x role plots/tables
goodcopbadcop_group_statistical_summary     -> Wilcoxon/effect-size overlays
```

This means the implementation should include an explicit "value parity" check:
for a fixture recording, the values rendered by the marimo helper functions must
match the component arrays, and the exported rows must match the same component
values after export.

### Explicit Deliverables

Deliverable 1 is the persisted per-recording component. It should create the
canonical zarr component with roles, phases, object-phase arrays, per-object
phase metrics, a wide summary row, source refs, QC warnings, and provenance.

Deliverable 2 is the per-recording inspection surface. It should add a CRA
Primary Endpoint panel to the existing GoodCopBadCop section of
`apps/marimo/palette_explorer.py`, driven by the selected recording and
selected chaser-distance run. The panel should show summary values, pre/post
role-specific distance and occupancy, object positions/quadrants, post-settle
trimming, and drift/QC diagnostics from the stored component.

Deliverable 3 is the merged export surface. It should add the CRA summary and
object-phase tables to the cross-recording export, preserving source component
paths and provenance columns needed to trace every group row back to one zarr
component.

Deliverable 4 is the cohort viewer surface. It should add API/query/UI support
for the exported CRA tables in the group analytics viewer, including paired
pre/post views, delta/specificity distributions, summary/object-phase tables,
and QC filters.

Deliverable 5 is the statistics surface. It should compute Wilcoxon signed-rank
tests, rank-biserial effects, and bootstrap median CIs from the exported summary
table, then expose those provenance-linked results in the group viewer only when
the selected stats artifact references the selected export.

### Per-Recording Visualization

Per-recording views should read:

```text
analysis/chaser_distance_runs/<run>/cra_primary_endpoint/<component_name>/
analysis/chaser_distance_runs/<run>/positions/*
analysis/detection_occupancy_runs/<run>/heatmaps/*          # optional density backing
```

The per-recording view must be discoverable from the selected top-level
GoodCopBadCop recording in `apps/marimo/palette_explorer.py`. It should not
require the user to manually type a chaser-distance run path. If multiple
chaser-distance runs exist, the UI should use the run marked latest by default
and expose an explicit run selector.

Required first-pass views:

- endpoint summary table with distance, occupancy, specificity, frame counts,
  and QC warnings;
- pre/post arena panels showing fish occupancy or density;
- aggressive and benign object positions marked in each phase;
- the object-relative quadrant used for each phase/object highlighted;
- object-role legend using raw colors plus role labels;
- post-settle trim annotation so the user can see that early post transition
  frames were excluded;
- object drift diagnostic table.

Recommended stored artifacts:

```text
analysis/chaser_distance_runs/<run>/cra_primary_endpoint/<component_name>/visualizations/cra_primary_endpoint_overview_png
analysis/chaser_distance_runs/<run>/cra_primary_endpoint/<component_name>/visualizations/cra_primary_endpoint_interactive
```

If the current marimo discovery machinery also needs run-level visualization
specs, the writer may mirror a lightweight interactive spec at:

```text
analysis/chaser_distance_runs/<run>/visualizations/cra_primary_endpoint_interactive
```

The component-local artifact is the canonical artifact. Any mirrored run-level
spec must point back to the component path and must not contain independently
computed endpoint values.

The existing Palette marimo explorer should expose a CRA Primary Endpoint panel
inside the GoodCopBadCop module. It should discover the component from the
selected chaser-distance run and render the stored summary/phase arrays. The
per-recording marimo panel is the main tool for checking that object-relative
quadrants are correct before group export.

Per-recording acceptance criteria:

- changing the selected recording changes every CRA endpoint panel from the
  newly selected recording's stored component;
- the displayed pre/post object positions match the component's stored
  `object_phase` arrays;
- highlighted quadrants are derived from stored object quadrants, not from
  fixed quadrant names;
- post panels visibly indicate the excluded settle frames;
- the summary values displayed in marimo exactly match the component arrays;
- the panel remains informative when optional heatmap/density backing is
  missing.

### Exported Group Visualization

The group analytics viewer should read the exported tables:

```text
goodcopbadcop_cra_primary_endpoint_summary
goodcopbadcop_cra_primary_endpoint_object_phase
goodcopbadcop_group_statistical_summary      # after Wilcoxon stats exist
```

The export should preserve enough identifiers to round-trip from a group row
back to the per-recording component:

```text
source_zarr_path
source_chaser_distance_run
source_cra_primary_endpoint_component
source_cra_primary_endpoint_path
source_component_schema_id
source_component_fingerprint
export_run_id
export_created_at
```

Required first-pass views:

- paired pre-to-post plots for aggressive distance and occupancy;
- paired pre-to-post plots for benign distance and occupancy as negative
  controls;
- delta/specificity distributions across recordings;
- summary table with one row per fish/recording;
- object-phase table with phase, role, quadrant, distance, occupancy, and
  dropout;
- QC filters/tables for dropout, object drift, endpoint status, and raw color;
- provenance panel linking the export run, source collection, and stats run.

The group viewer can show descriptive summaries immediately. Confirmatory
p-values and effect sizes should appear only after the Wilcoxon statistics
extension writes a provenance-linked stats artifact for the same source export.

Exported/group acceptance criteria:

- every row in `goodcopbadcop_cra_primary_endpoint_summary` maps to exactly one
  source component and one fish/recording;
- every row in `goodcopbadcop_cra_primary_endpoint_object_phase` maps to one
  source component, phase, and object role;
- group plots use exported Parquet rows only, not direct zarr reads;
- summary and object-phase tables expose provenance columns by default;
- statistical overlays are hidden or marked unavailable when the selected
  export has no matching Wilcoxon statistics artifact;
- filters for endpoint status, dropout, drift, raw color, and role are applied
  consistently across plots and tables.

## Implementation Checklist

- [x] Confirm the endpoint storage location under
      `analysis/chaser_distance_runs/<run>/cra_primary_endpoint/`.
- [x] Treat
      `analysis/chaser_distance_runs/<run>/cra_primary_endpoint/<component_name>/`
      as the canonical source for all per-recording CRA endpoint values.
- [x] Store per-recording visualization artifacts under the component-local
      `visualizations/` group; mirror run-level specs only for marimo discovery
      if needed.
- [x] Add a protocol-role resolver that reads `protocol_json` and returns one
      aggressive and one benign object with raw colors and start/end metadata.
- [x] Add source validation for required chaser-distance arrays:
      `fish_centroid_arena_xy`, `chaser_arena_xy`, `fish_valid`,
      `chaser_valid`, `distance_mm`, and epoch frame/window references.
- [x] Add quadrant-bounds resolver from
      `stimulus_coordinates/arena_1.texture_width_px` and
      `texture_height_px`, with attrs recorded in the endpoint component.
- [x] Add effective phase resolver:
      - [x] `pre_static = pre_event`.
      - [x] `post_static = post_event` after
            `position_transition_duration_s`.
      - [x] exclude `training_event`.
- [x] Add static object-position summarizer using median object position within
      each effective phase and object.
- [x] Add object drift diagnostics.
- [x] Add per-object phase metrics:
      - [x] median distance in mm.
      - [x] mean distance in mm.
      - [x] object-quadrant occupancy fraction over valid fish frames.
      - [x] object-quadrant occupancy fraction over all phase frames.
      - [x] valid, missing, and dropout counts.
- [x] Add wide confirmatory summary row arrays with distance, occupancy, and
      specificity contrasts.
- [x] Add run attrs/source refs/provenance and lineage fingerprint attrs.
- [x] Add focused in-memory/unit tests for:
      - [x] role resolution from protocol JSON.
      - [x] post-settle trimming.
      - [x] object-relative quadrant assignment when object quadrant changes.
      - [x] aggressive/benign specificity calculations.
      - [x] dropout report-only behavior.
- [x] Add one real-zarr smoke command for a single `/groups` GoodCopBadCop
      recording, run outside the sandbox.
- [x] Add per-recording visualization artifacts:
      - [x] `cra_primary_endpoint_overview_png`.
      - [x] `cra_primary_endpoint_interactive`.
      - [x] summary table, phase panels, object markers, object-relative
            quadrant overlays, post-settle trim annotation, and drift/QC table.
- [x] Add a GoodCopBadCop marimo panel that discovers and renders the stored
      CRA endpoint component for the selected recording.
- [x] Add per-recording acceptance checks that compare displayed marimo values
      against the stored component arrays for at least one fixture recording.
- [x] Add a registry batch wrapper only after the single-recording writer is
      validated.
- [x] Extend cross-recording export with the summary/object-phase tables.
- [x] Include source component path, schema id, and fingerprint/provenance
      columns in both exported tables.
- [x] Add group viewer endpoints for:
      - [x] `goodcopbadcop_cra_primary_endpoint_summary`.
      - [x] `goodcopbadcop_cra_primary_endpoint_object_phase`.
- [x] Add group viewer UI for:
      - [x] aggressive and benign paired pre/post plots.
      - [x] distance and occupancy deltas.
      - [x] distance and occupancy specificity.
      - [x] one-row-per-fish summary table.
      - [x] object-phase table.
      - [x] dropout/object-drift/endpoint-status QC tables.
      - [x] raw-color counterbalancing QC.
- [x] Add group viewer acceptance checks that prove plots/tables render from
      exported Parquet tables without direct zarr reads.
- [x] Extend group statistics with Wilcoxon signed-rank and rank-biserial
      effect sizes for the primary endpoint.
- [x] Overlay provenance-linked Wilcoxon statistics in the group viewer only
      when the matching stats export references the selected CRA endpoint
      export.

## Deferred

- STAR Methods parity audit for exact paper parameters.
- Final hard dropout/exclusion threshold.
- KDE density map implementation and bandwidth policy.
- Multi-fish identity handling beyond the current one-fish-per-recording
  GoodCopBadCop recordings.
- Generalizing the endpoint beyond chaser-style projected objects.
