# Bout Kinematics Run Design

Date anchored: 2026-04-27

Purpose: define a derived analysis layer for per-bout heading and movement
metrics without mutating swim-bout segmentation outputs.

## Executive Summary

`analysis/swim_bout_runs/<run>/<speed_level>/` should remain the immutable
segmentation artifact: it answers where bouts start and stop for a particular
source speed trace and parameter set.

Richer per-bout measurements should live in a linked derived analysis run:

```text
analysis/bout_kinematics_runs/<run_name>/
```

This layer should consume:

- one exact `analysis/track_kinematics_runs/<scope>/<run>` source
- one exact `analysis/swim_bout_runs/<run>/<speed_level>` source
- a declared heading source from the track-kinematics run
- a declared physical movement source from the track-kinematics run
- explicit pre-bout and post-bout window parameters

It should produce per-bout metrics such as net heading change, within-bout
heading excursion, within-bout heading path length, physical active-motion
duration, physical path length, and optional within-bout dominant frequency.
By default it also joins frame-level eye-gaze outputs from an exact
`analysis/eye_angle_runs/<run>` source and writes bout-aligned eye summaries
without mutating either source run. A compatibility run may explicitly disable
that join with `include_eye_gaze=False` or `--no-include-eye-gaze`; missing eye
inputs must otherwise fail rather than silently producing a reduced contract.

## Use-Case Boundary

Use `analysis/swim_bout_runs/<run>/<speed_level>/` when the question is about
bout segmentation itself:

- which speed trace and thresholding parameters define a bout candidate
- where each bout starts and ends
- what core and expanded boundaries were selected
- what segmentation-time movement summaries were available from the source
  track, such as duration, path length, net displacement, and gap coverage

Use `analysis/bout_kinematics_runs/<run>/<heading_level>/per_bout_metrics/`
when the question is about measurements derived from a frozen bout candidate:

- what heading change the bout produced
- what stable pre/post position displacement the bout produced
- how much within-bout heading excursion or oscillation occurred
- which heading source, pre/post epoch policy, and measurement parameters were
  used

Use `analysis/bout_kinematics_runs/<run>/movement/per_bout_metrics/` when the
question is about physical movement measured for those same frozen bout rows:

- what detector-window duration was inherited from the source bout candidate
- what physical active-motion duration was measured from a declared physical
  speed source
- what path length, mean speed, and peak speed were measured from that physical
  source
- which boundary policy, boundary constraint, threshold, and search margin were
  used

Use `analysis/bout_kinematics_runs/<run>/eye_gaze/per_bout_metrics/` when the
question is how eye convergence or gaze state relates to those same frozen bout
boundaries. The source eye trace remains owned by `analysis/eye_angle_runs`; the
bout-kinematics run stores only windowed per-bout summaries and lineage.

New compact-v2 runs store the same logical levels in direct tables
(`movement_metrics`, `heading_metrics`, and `eye_gaze_metrics`) rather than
subgroup-per-level tables. `eye_gaze_metrics` is absent only for an explicitly
opted-out compatibility run. Readers should use
`resolve_bout_kinematics_tables(...)` instead of hard-coding physical paths.
See `docs/bout_kinematics_compact_v2_layout.md` for the compact layout
contract.

This boundary lets operators tune and compare segmentation candidates without
rewriting downstream biological measurements, and lets analysts recompute
kinematics without changing the bout-definition artifact.

## Why This Should Not Mutate `swim_bout_runs`

The tempting shortcut would be to add heading-change columns directly to:

```text
analysis/swim_bout_runs/<run>/<speed_level>/bouts
```

That would be a semantic mistake. `swim_bout_runs` is the segmentation output.
It currently stores bout boundaries and basic segmentation-derived summaries,
but those fields are part of the bout-detection artifact. Heading analysis is a
downstream interpretation of those boundaries plus a heading time series.

If heading metrics were written into the segmentation table, then improving
heading logic would require either:

- editing an existing segmentation artifact after creation
- regenerating segmentation just to update downstream metrics
- leaving old segmentation runs with mixed metric-schema generations

Instead, `bout_kinematics_runs` should be independently recomputable and should
link back to the exact segmentation candidate it used.

## Relationship To Johnson-Style Bout Analysis

For repeatability with published bout-centric larval zebrafish analyses, Palette
should support a net heading-change metric computed from stable interbout
windows:

- average heading over a pre-bout measurement epoch
- average heading over a post-bout measurement epoch
- subtract the two circular means after unwrapping or using circular arithmetic

This metric answers:

```text
What net reorientation did this bout produce?
```

It should not be replaced by a start-frame/end-frame heading subtraction, because
boundary frames can be noisy and the fish can still be in the motor act.

The writer should support two pre/post epoch modes:

- `fixed_window`: use fixed-duration windows immediately before bout start and
  immediately after bout end. This is useful for short, controlled audits and
  for recordings where interbout epoch boundaries are not yet trusted.
- `interbout_epoch`: use the full preceding interbout epoch and the full
  following interbout epoch, derived from adjacent bouts in the selected
  segmentation candidate. This is the Johnson-parity mode.

For `interbout_epoch` mode, the first bout has no preceding interbout epoch and
the last bout has no following interbout epoch. The initial policy should mark
those sides invalid by default rather than inventing archive-edge epochs. Edge
handling can be added later as an explicit parameter if needed.

## Physical Movement Metrics

`swim_bout_runs` stores detector outputs. Its `duration_s`,
`observed_duration_s`, and `core_duration_s` fields describe the event window
chosen by the detector and its boundary policy. Those fields are intentionally
preserved when the detector uses a transformed response such as
`speed_exponential`, even if that response has a broader tail than measured
fish motion.

`bout_kinematics_runs/<run>/movement/per_bout_metrics/` is the first-class
physical estimator layer. It measures physical active motion from a declared
track-kinematics speed source, currently one of:

- `speed_raw_mm`
- `speed_filtered_mm`
- `speed_smoothed_mm`

The default is `speed_filtered_mm`, because it suppresses sub-threshold jitter
without using the causal exponential detector response as if it were measured
speed.

The physical-active policy is factored into two concepts:

- `physical_active_boundary_policy = "physical_active"`: the metric measures
  first/last above-threshold samples on the physical speed source.
- `physical_active_boundary_constraint`: how far the measurement search may
  move relative to the detector window. Current values are
  `clip_to_detector`, `search_with_margin`, and `allow_extension`.

For `search_with_margin`, the writer records both the requested
`physical_active_boundary_margin_s` and the resolved frame count. The search is
bounded by adjacent source bouts so one source bout cannot consume samples that
belong to a neighboring source bout.

The movement table preserves both detector and physical-estimator duration:

- `detector_duration_s`, `detector_observed_duration_s`, and
  `detector_core_duration_s` copy source detector-boundary durations.
- `physical_active_duration_s` is the sampled first-to-last active-frame
  duration on the physical speed source.
- `physical_active_duration_s_interpolated` is the optional threshold-crossing
  duration estimated between adjacent samples when both boundaries are valid.
- `physical_active_observed_duration_s` sums valid transition durations across
  the active sampled span.

Physical path length and mean speed are computed from the matching
`frame_path_distance_<level>_*` arrays when present. Peak speed is measured from
the selected physical speed source within the physical-active search window.
The causal exponential/convolved detector response should not be used for these
physical estimator fields.

### Deferred Detector-Response Diagnostics

A future optional sibling surface may store detector-response summaries:

```text
bout_kinematics_runs/<run>/detector_response/per_bout_metrics/
```

That surface would be diagnostic, not the primary physical estimator. It would
summarize the exact detector signal declared by the source swim-bout candidate,
for example `speed_exponential_mm`, and would keep response-derived quantities
clearly labeled:

- `response_peak_value`
- `response_area`
- `response_width_s`
- `response_duration_s`
- `response_rise_time_s`
- `response_decay_time_s`

This can be useful for QA, method comparison, and classifier features. It should
not write fields named as physical quantities such as `path_length_mm` or
`mean_speed_mm_s` unless the field name explicitly says it is a detector
response proxy. The source detector metadata already declares the transform
family and parameters, so this deferred surface should reference that metadata
rather than duplicating it ad hoc.

## Within-Bout Heading Metrics

Net reorientation is not enough for high-speed recordings or bouts where the
head oscillates during the swim. A fish may have near-zero net heading change
while still exhibiting meaningful within-bout head wiggle.

The schema should therefore include both net and within-bout metrics:

```text
per_bout_metrics/
  bout_id
  source_start_frame
  source_end_frame
  source_core_start_frame                         optional
  source_core_end_frame                           optional
  source_core_start_time_s_interpolated           optional
  source_core_end_time_s_interpolated             optional
  source_core_duration_s_interpolated             optional
  source_core_start_time_interpolated_valid       optional
  source_core_end_time_interpolated_valid         optional
  pre_epoch_start_frame
  pre_epoch_end_frame
  post_epoch_start_frame
  post_epoch_end_frame

  pre_heading_mean_deg
  post_heading_mean_deg
  net_delta_heading_deg
  abs_net_delta_heading_deg

  pre_position_mean_x_mm
  pre_position_mean_y_mm
  post_position_mean_x_mm
  post_position_mean_y_mm
  interbout_epoch_displacement_mm
  pre_position_mean_x_px
  pre_position_mean_y_px
  post_position_mean_x_px
  post_position_mean_y_px
  interbout_epoch_displacement_px

  within_heading_range_deg
  within_heading_peak_to_peak_deg
  within_heading_path_deg
  within_heading_std_deg
  within_heading_zero_crossings
  within_heading_dominant_frequency_hz            optional, NaN when not computed
  within_angular_velocity_mean_deg_s
  within_angular_speed_mean_deg_s
  within_angular_speed_max_deg_s
  within_angular_velocity_std_deg_s

  pre_window_valid
  post_window_valid
  pre_position_valid
  post_position_valid
  within_window_valid
  within_angular_velocity_valid
  dominant_frequency_valid
  pre_window_sample_count
  post_window_sample_count
  pre_position_sample_count
  post_position_sample_count
  within_window_sample_count
  within_angular_velocity_transition_count
  failure_reason_bytes                            optional preferred string encoding
```

Recommended initial semantics:

- `net_delta_heading_deg`: post-window circular mean minus pre-window circular
  mean, signed in the configured heading convention.
- `abs_net_delta_heading_deg`: absolute value of `net_delta_heading_deg`.
- `pre_epoch_*` / `post_epoch_*`: the resolved frame intervals used for
  pre/post heading and position means. These are fixed-size windows in
  `fixed_window` mode and adjacent interbout epochs in `interbout_epoch` mode.
  Stored start/end frame fields are inclusive source-frame bounds, with `-1`
  used when the side has no resolved epoch.
- `pre_position_mean_*` / `post_position_mean_*`: coordinate means over the
  same resolved pre/post epochs. Pixel fields should be populated whenever
  pixel positions are available; millimeter fields should be `NaN` when
  calibrated positions are unavailable.
- `interbout_epoch_displacement_*`: Euclidean distance between post- and
  pre-position means. This is distinct from segmentation-time `path_length_*`
  and `net_displacement_*`, because it measures stable interbout mean-position
  displacement rather than frame-boundary or within-bout trajectory distance.
- `within_heading_range_deg`: max minus min unwrapped heading during the bout.
- `within_heading_peak_to_peak_deg`: alias-level semantic metric for
  within-bout oscillation amplitude. Keep this as a distinct schema field even
  when the first implementation computes it identically to
  `within_heading_range_deg`, so future robust peak-to-peak estimators do not
  require a schema break.
- `within_heading_path_deg`: sum of absolute frame-to-frame heading changes
  during the bout.
- `within_heading_std_deg`: circular or unwrap-aware heading variability within
  the bout.
- `within_heading_zero_crossings`: count of sign changes in within-bout heading
  velocity after the configured derivative threshold.
- `within_heading_dominant_frequency_hz`: optional frequency estimate. This is
  only meaningful when the bout has enough samples and the recording frame rate
  supports the desired frequency band.

Angular-velocity summaries are derived from validated framewise heading
transitions, not by mutating the source swim-bout segmentation:

- `within_angular_velocity_mean_deg_s`: signed mean turning rate over valid
  within-bout transitions. This can cancel when left/right turns occur in the
  same bout, so it should not be the only magnitude summary.
- `within_angular_speed_mean_deg_s`: mean absolute turning rate over valid
  within-bout transitions.
- `within_angular_speed_max_deg_s`: peak absolute turning rate over valid
  within-bout transitions.
- `within_angular_velocity_std_deg_s`: variability of signed turning rate over
  valid within-bout transitions.

The default heading source for these summaries should be smoothed heading,
with raw-heading variants allowed when high-frequency framewise motion is the
scientific target. Values should be `NaN` when the bout crosses invalid track
transitions, nonpositive time deltas, or heading gaps.

## Dominant Frequency Policy

Dominant frequency should be part of the schema from the beginning, but optional
as an output.

Reasons:

- standard 60 fps recordings may have too few samples per bout
- short bouts may not support a stable spectral estimate
- high-speed recordings may make this metric valuable later
- the absence of a frequency estimate should not invalidate the rest of the run

The writer should store `NaN` plus `dominant_frequency_valid=false` when the
metric is not computed or not trustworthy.

## Heading Levels

Bout kinematics should mirror the candidate-style thinking used by
`detect_bouts_multi_level`, but for heading traces rather than speed traces.
The same swim-bout segmentation candidate may be measured against multiple
heading inputs:

```text
heading_smoothed/
  per_bout_metrics/
heading_raw/
  per_bout_metrics/
```

The default heading level should be `heading_smoothed`, because both pre/post
heading means and within-bout oscillation metrics are sensitive to frame-level
heading noise. Raw heading should remain available as a parallel candidate for
auditing and high-speed recordings where smoothing could hide fast within-bout
structure.

Recommended initial levels:

- `heading_smoothed`: uses `smoothed_heading_degrees`
- `heading_raw`: uses `heading_degrees`

The run should record `default_heading_level = "heading_smoothed"`.

## Current Storage Shape

```text
analysis/bout_kinematics_runs/<run_name>/
  attrs:
    schema_id: "analysis.bout_kinematics_runs"
    schema_version: 7
    method: "heading_window_and_within_bout_metrics"
    method_version: "bout_kinematics.v7"
    status: "running" | "complete" | "failed"
    created_at_utc
    row_axis: "swim_bout_rows"
    source_refs:
      zarr_path
      source_track_kinematics_run
      source_track_kinematics_track_path
      source_swim_bout_run
      source_swim_bout_speed_level
      source_swim_bout_path
      source_peak_events_path                    optional, when source has peak_events
      source_track_id
      source_heading_arrays:
        heading_smoothed: <track path>/smoothed_heading_degrees
        heading_raw: <track path>/heading_degrees
      source_movement_arrays:
        physical_active_speed: <track path>/speed_filtered_mm
        physical_active_path_distance_mm: <track path>/frame_path_distance_filtered_mm
        physical_active_path_distance_px: <track path>/frame_path_distance_filtered_px
      source_validity_arrays:
        delta_seconds: <track path>/delta_seconds
        transition_valid: <track path>/transition_valid
        sample_valid: <track path>/sample_valid
      source_eye_angle_run                      optional, when eye_gaze is enabled
      source_eye_angle_path                     optional
      source_eye_angle_schema_version           optional
      source_eye_angle_family                   optional
      source_eye_angle_arrays                   optional
    failure_stage                               optional, when status == "failed"
    failure_reason                              optional, when status == "failed"
    parameters:
      default_heading_level: "heading_smoothed"
      heading_levels: ["heading_smoothed", "heading_raw"]
      pre_post_mode: "fixed_window" | "interbout_epoch"
      pre_window_s
      post_window_s
      resolved_pre_window_frames
      resolved_post_window_frames
      within_window: "bout_start_end" | "core_start_end"
      heading_units: "degrees"
      heading_unwrap_policy
      physical_active:
        enabled: true
        boundary_policy: "physical_active"
        boundary_constraint: "search_with_margin"
        boundary_margin_s
        resolved_boundary_margin_frames
        threshold_mm_s
        measurement_signal_level: "speed_filtered"
        measurement_signal_array: "speed_filtered_mm"
      source_interpolated_threshold_fields
      source_peak_event_fields
      zero_crossing_derivative_threshold_deg_s
      dominant_frequency:
        enabled
        min_samples
        method
        detrend
      eye_gaze:
        enabled
        eye_angle_run
        eye_angle_family: "gaze"
        eye_validity_min_fraction
        vergence_threshold_deg                  optional
  heading_smoothed/
    per_bout_metrics/
      bout_id
      source_start_frame
      source_end_frame
      ...
  heading_raw/
    per_bout_metrics/
      bout_id
      source_start_frame
      source_end_frame
      ...
  movement/
    per_bout_metrics/
      bout_id
      source_start_frame
      source_end_frame
      source_core_start_frame
      source_core_end_frame
      detector_duration_s
      detector_observed_duration_s
      detector_core_duration_s
      physical_active_start_frame
      physical_active_end_frame
      physical_active_start_time_s
      physical_active_end_time_s
      physical_active_duration_s
      physical_active_observed_duration_s
      physical_active_start_time_s_interpolated
      physical_active_end_time_s_interpolated
      physical_active_duration_s_interpolated
      physical_active_start_time_interpolated_valid
      physical_active_end_time_interpolated_valid
      physical_active_sample_count
      physical_active_valid_transition_count
      physical_active_valid_transition_fraction
      physical_active_path_length_mm
      physical_active_path_length_px
      physical_active_mean_speed_mm_s
      physical_active_peak_speed_mm_s
      physical_active_threshold_mm_s
      physical_active_boundary_margin_s
      physical_active_boundary_policy_bytes
      physical_active_boundary_constraint_bytes
      physical_active_valid
      failure_reason_bytes
  eye_gaze/                                    optional
    per_bout_metrics/
      bout_id
      source_start_frame
      source_end_frame
      source_core_start_frame
      source_core_end_frame
      pre_epoch_start_frame
      pre_epoch_end_frame
      post_epoch_start_frame
      post_epoch_end_frame
      within_epoch_start_frame
      within_epoch_end_frame
      pre_left_gaze_mean_deg
      pre_right_gaze_mean_deg
      pre_vergence_gaze_mean_deg
      pre_vergence_gaze_signed_mean_deg
      pre_vergence_gaze_std_deg
      pre_vergence_gaze_valid_fraction
      pre_converged_fraction
      post_left_gaze_mean_deg
      post_right_gaze_mean_deg
      post_vergence_gaze_mean_deg
      post_vergence_gaze_signed_mean_deg
      post_vergence_gaze_std_deg
      post_vergence_gaze_valid_fraction
      post_converged_fraction
      within_bout_left_gaze_mean_deg
      within_bout_right_gaze_mean_deg
      within_bout_vergence_gaze_mean_deg
      within_bout_vergence_gaze_signed_mean_deg
      within_bout_vergence_gaze_max_deg
      within_bout_vergence_gaze_range_deg
      within_bout_vergence_gaze_std_deg
      within_bout_vergence_gaze_valid_fraction
      within_bout_converged_fraction
      pre_eye_window_valid
      post_eye_window_valid
      within_eye_window_valid
      pre_eye_sample_count
      post_eye_sample_count
      within_eye_sample_count
      failure_reason_bytes
```

The run should also record enough source identity to validate that the
`bout_id` values still refer to the same source bout table. At minimum:

- source archive path
- source swim-bout run name
- source speed level
- source track id

When the source swim-bout schema provides interpolated core-threshold timing,
`bout_kinematics_runs` should copy those source fields into
`per_bout_metrics`. They are source-boundary annotations, not new segmentation
or row-alignment keys. Frame fields remain authoritative for slicing heading and
position arrays.

When the source swim-bout speed subgroup provides an aligned `peak_events`
table, `bout_kinematics_runs` should also copy the peak-event boundary context
into `per_bout_metrics`. These fields preserve the signal-derived boundary
estimate beside the integer frame-boundary contract:

- `source_peak_frame`, `source_peak_time_s`
- `source_peak_signal_value_mm_s`, `source_peak_prominence_mm_s`
- `source_peak_width_s`, `source_peak_width_height_mm_s`
- `source_peak_left_width_frame_interpolated`
- `source_peak_right_width_frame_interpolated`
- `source_peak_left_width_time_s`
- `source_peak_right_width_time_s`
- `source_peak_boundary_mode_bytes`
- `source_peak_shape_split_policy_bytes`

These fields are copied provenance/review annotations. Current heading and
position metrics still use integer `source_start_frame` / `source_end_frame`
boundaries for array slicing. Fractional-time heading or position interpolation
would be a separate future analysis mode.
- source track-kinematics run
- source heading arrays used for each heading level
- segmentation parameter snapshot or source provenance hash when available

Top-level `source_refs` is the authoritative source snapshot for the run. Child
heading-level attrs such as `heading_source_array` may mirror part of this
information for local convenience, but readers should treat `source_refs` as the
canonical provenance mapping if the two ever disagree.

The optional `eye_gaze` subgroup follows the same provenance rule. It consumes
logical frame-aligned gaze arrays from `analysis/eye_angle_runs/<run>` through
`fisheye.analysis.eye_angle_io`, so the source can be either hierarchical-v1
`angles/frame/*_gaze_*` arrays or compact-dense-v2 backing channels. Values are
aligned by source frame index to the selected track-kinematics frames. Its
`pre_*`, `post_*`, and `within_*` windows are the same resolved windows used for
heading metrics. If a `vergence_threshold_deg` is configured,
`*_converged_fraction` records the fraction of valid samples meeting or
exceeding that threshold; otherwise those fields are `NaN`.

## Visualization Policy

Visual summaries should keep signed net heading changes separate from
within-bout magnitude/path metrics. Net heading change is a wrapped signed angle
and should be plotted on a fixed `[-180, 180]` degree x-axis. Within-bout range,
peak-to-peak, path length, and standard deviation are nonnegative excursion
metrics and may legitimately exceed 180 degrees, so they should use independent
positive axes.

The writer should persist a separate physical movement visualization pair under
`visualizations/`:

- `bout_movement_summary_track_<id>_png`
- `bout_movement_summary_track_<id>_interactive`

This plot spec uses `palette.plot_spec.bout_movement_summary.v1` and should
focus on `movement/per_bout_metrics`, including detector duration, physical
active duration, interpolated physical active duration, physical path length,
mean speed, and peak speed. Keeping this separate from the heading plot avoids
mixing physical movement units with angular units.

Its dedicated renderer ID is `palette-bout-kinematics-movement-v1`. The
heading summary uses `palette-bout-kinematics-heading-v1`, and the optional
eye-gaze summary below uses `palette-bout-kinematics-eye-gaze-v1`. Older runs
may persist the generic `matplotlib_static_plotly_spec.v1` renderer. Readers
must recognize that legacy value only when it is paired with one of the exact
bout plot schema IDs; the generic renderer alone is not a bout contract.

When the run contains `eye_gaze/per_bout_metrics`, the writer should also persist
a separate eye-gaze visualization pair under `visualizations/`:

- `bout_eye_gaze_summary_track_<id>_png`
- `bout_eye_gaze_summary_track_<id>_interactive`

This plot spec uses `palette.plot_spec.bout_eye_gaze_summary.v1` and should
focus on bout-aligned eye summaries such as pre/post vergence, within-bout mean
and maximum vergence, within-bout vergence range, and optional converged
fractions. Keeping this separate from the heading plot avoids mixing different
biological quantities into one overloaded artifact.

The Recording Explorer groups these companion specs into one Bout kinematics
provider per run. Heading, movement, and eye-gaze entries are exposed only when
their corresponding persisted contracts exist. The bounded PNG declared by
`snapshot_artifact` remains available as an optional reference; displaying the
interactive summary does not read the full per-bout table. Provenance remains
available from the interactive spec and artifact attributes.

The interactive distribution renderer treats the spec's histogram metrics as
an allowlist. It reads only the selected metric column, the compact heading
level index when required, and metric-specific validity columns. Histogram and
cumulative-distribution bins are computed server-side; raw per-bout values are
not embedded in the Plotly payload. Users may change the metric, heading
representation, bin count, and validity filter, and may reveal the persisted
PNG as an optional reference. When a materialized run retains source paths that
name its parent run, the reader resolves the declared run-relative suffix
against the actual owning run while preserving the original spec for
provenance.

## Window Parameters

User-facing fixed pre/post windows should be specified in seconds. Writers
should persist:

- selected `pre_post_mode`
- requested `pre_window_s`
- requested `post_window_s`
- resolved frame counts for the run FPS
- actual valid sample counts per bout and heading level

This keeps the command interface stable across recordings with different frame
rates while preserving enough detail to audit the exact sample windows used.

When `pre_post_mode="fixed_window"`, the resolved pre/post epochs are:

- `[bout_start - resolved_pre_window_frames, bout_start)`
- `(bout_end, bout_end + resolved_post_window_frames]`

using the selected track-kinematics frame index and marking the side invalid if
the full requested window is not available or contains gaps.

When `pre_post_mode="interbout_epoch"`, the resolved pre/post epochs are:

- preceding epoch: frames after the previous bout end and before this bout start
- following epoch: frames after this bout end and before the next bout start

The first bout has no preceding epoch and the last bout has no following epoch;
those sides should be marked invalid by default. This avoids silently mixing
archive-edge behavior into Johnson-style measurements.

## Track Scope

The first writer should support one selected track per run. Current canary data
is effectively single-subject per recording, and one-track output is simpler to
validate.

The design must still avoid blocking future multi-animal recordings. Required
future-compatible constraints:

- persist `source_track_id` at run level for the v1 single-track writer
- avoid assuming track `0` in schema names or dataset paths
- keep metric row identity tied to source bout rows, not physical row order
- reserve a future layout such as:

```text
analysis/bout_kinematics_runs/<run_name>/
  tracks/
    id_<track_id>/
      heading_smoothed/per_bout_metrics/
      heading_raw/per_bout_metrics/
```

Multi-track support should be a storage extension, not a semantic rewrite.

## Validity And Failure Reasons

Per-bout metrics should prefer explicit validity over implicit sentinel values.

Recommended failure tags:

- `missing_heading_source`
- `missing_position_source`
- `insufficient_pre_window`
- `insufficient_post_window`
- `insufficient_pre_position`
- `insufficient_post_position`
- `insufficient_within_bout_samples`
- `heading_contains_gap`
- `dominant_frequency_disabled`
- `dominant_frequency_insufficient_samples`
- `source_bout_missing`

Floating-point metrics may use `NaN`, but consumers should read validity arrays
and reason tags rather than infer failure state from `NaN` alone.

## Completion And Visualization Failure Policy

`analysis/bout_kinematics_runs.attrs["latest"]` should only point at a run after
the writer has completed all requested outputs. This includes optional
`--write-zarr-artifacts` output. A failure while writing optional visualization
artifacts may leave useful numeric tables in the run group, but the run must be
marked `status = "failed"` and must not be promoted to `latest`.

Persisted visualization artifacts are non-interactive PNG/spec products. The
writer should use a non-GUI matplotlib backend so CLI artifact generation is not
coupled to workstation display state, Tk/Qt availability, or GUI teardown
behavior.

## Naming Policy For `_runs`

The `_runs` suffix is useful in durable Zarr parent group names because it
communicates that the group contains multiple versioned attempts plus run-level
metadata, for example:

```text
analysis/swim_bout_runs/<run>
analysis/bout_kinematics_runs/<run>
analysis/track_kinematics_runs/<scope>/<run>
```

However, carrying `_runs` into every local variable or UI label creates noise.
Recommended naming:

- persistent parent group: `swim_bout_runs`, `bout_kinematics_runs`
- one concrete run: `swim_bout_run`, `bout_kinematics_run`
- local parent variable: `bout_parent` or `kinematics_parent`
- selected option in UI: `bout_candidate`
- source reference attr: `source_swim_bout_run`
- exact level reference: `source_swim_bout_speed_level`

UI labels should use biological or operator-facing language such as "derived
swim-bout candidate" rather than exposing storage suffixes.

## Resolved Initial Decisions

- `within_heading_peak_to_peak_deg` remains a distinct schema field; v1 may
  compute it identically to `within_heading_range_deg`.
- `heading_smoothed` is the default heading level.
- `heading_raw` should be computable side by side with `heading_smoothed`.
- pre/post windows are user-facing seconds with resolved frame counts persisted.
- v1 writes one selected track per run while reserving a future multi-track
  `tracks/id_<track_id>/...` layout.

## Related Documents

- [derived_analysis_run_contract.md](derived_analysis_run_contract.md)
- [track_kinematics_bout_status.md](track_kinematics_bout_status.md)
- [pose_kinematics_run_design.md](pose_kinematics_run_design.md)
- [keypoint_heading_computation_contract.md](keypoint_heading_computation_contract.md)
