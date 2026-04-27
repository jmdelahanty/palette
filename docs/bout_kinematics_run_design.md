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
- explicit pre-bout and post-bout window parameters

It should produce per-bout metrics such as net heading change, within-bout
heading excursion, within-bout heading path length, and optional within-bout
dominant frequency.

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

- average heading over a pre-bout window
- average heading over a post-bout window
- subtract the two circular means after unwrapping or using circular arithmetic

This metric answers:

```text
What net reorientation did this bout produce?
```

It should not be replaced by a start-frame/end-frame heading subtraction, because
boundary frames can be noisy and the fish can still be in the motor act.

## Within-Bout Heading Metrics

Net reorientation is not enough for high-speed recordings or bouts where the
head oscillates during the swim. A fish may have near-zero net heading change
while still exhibiting meaningful within-bout head wiggle.

The first schema should therefore include both net and within-bout metrics:

```text
per_bout_metrics/
  bout_id
  source_start_frame
  source_end_frame
  source_core_start_frame                         optional
  source_core_end_frame                           optional

  pre_heading_mean_deg
  post_heading_mean_deg
  net_delta_heading_deg
  abs_net_delta_heading_deg

  within_heading_range_deg
  within_heading_peak_to_peak_deg
  within_heading_path_deg
  within_heading_std_deg
  within_heading_zero_crossings
  within_heading_dominant_frequency_hz            optional, NaN when not computed

  pre_window_valid
  post_window_valid
  within_window_valid
  dominant_frequency_valid
  failure_reason_bytes                            optional preferred string encoding
```

Recommended initial semantics:

- `net_delta_heading_deg`: post-window circular mean minus pre-window circular
  mean, signed in the configured heading convention.
- `abs_net_delta_heading_deg`: absolute value of `net_delta_heading_deg`.
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

## Proposed Storage Shape

```text
analysis/bout_kinematics_runs/<run_name>/
  attrs:
    schema_id: "analysis.bout_kinematics_runs"
    schema_version: 1
    method: "heading_window_and_within_bout_metrics"
    method_version: "<implementation version>"
    created_at_utc
    row_axis: "swim_bout_rows"
    source_refs:
      zarr_path
      source_track_kinematics_run
      source_track_kinematics_track_path
      source_swim_bout_run
      source_swim_bout_speed_level
      source_swim_bout_path
      source_track_id
      source_heading_arrays:
        heading_smoothed: <track path>/smoothed_heading_degrees
        heading_raw: <track path>/heading_degrees
    parameters:
      default_heading_level: "heading_smoothed"
      heading_levels: ["heading_smoothed", "heading_raw"]
      pre_window_s
      post_window_s
      resolved_pre_window_frames
      resolved_post_window_frames
      within_window: "bout_start_end" | "core_start_end"
      heading_units: "degrees"
      heading_unwrap_policy
      zero_crossing_derivative_threshold_deg_s
      dominant_frequency:
        enabled
        min_samples
        method
        detrend
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
```

The run should also record enough source identity to validate that the
`bout_id` values still refer to the same source bout table. At minimum:

- source archive path
- source swim-bout run name
- source speed level
- source track id
- source track-kinematics run
- source heading arrays used for each heading level
- segmentation parameter snapshot or source provenance hash when available

Top-level `source_refs` is the authoritative source snapshot for the run. Child
heading-level attrs such as `heading_source_array` may mirror part of this
information for local convenience, but readers should treat `source_refs` as the
canonical provenance mapping if the two ever disagree.

## Visualization Policy

Visual summaries should keep signed net heading changes separate from
within-bout magnitude/path metrics. Net heading change is a wrapped signed angle
and should be plotted on a fixed `[-180, 180]` degree x-axis. Within-bout range,
peak-to-peak, path length, and standard deviation are nonnegative excursion
metrics and may legitimately exceed 180 degrees, so they should use independent
positive axes.

## Window Parameters

User-facing pre/post windows should be specified in seconds. Writers should
persist:

- requested `pre_window_s`
- requested `post_window_s`
- resolved frame counts for the run FPS
- actual valid sample counts per bout and heading level

This keeps the command interface stable across recordings with different frame
rates while preserving enough detail to audit the exact sample windows used.

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
- `insufficient_pre_window`
- `insufficient_post_window`
- `insufficient_within_bout_samples`
- `heading_contains_gap`
- `dominant_frequency_disabled`
- `dominant_frequency_insufficient_samples`
- `source_bout_missing`

Floating-point metrics may use `NaN`, but consumers should read validity arrays
and reason tags rather than infer failure state from `NaN` alone.

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
