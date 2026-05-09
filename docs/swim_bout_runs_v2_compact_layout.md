# Swim-Bout Runs V2 Compact Layout

## Status

Design note for deferred implementation. No existing `analysis/swim_bout_runs`
archives should be migrated or rewritten as part of this design slice.

Implementation note, 2026-05-08: the first compatibility resolver now lives in
`src/fisheye/analysis/swim_bout_io.py`. It reads current v1 hierarchical runs
and exposes normalized candidate/signal/table objects so consumers can migrate
before a compact v2 writer exists. The cross-recording analytics exporter and
`bout_kinematics.py` now use this resolver for swim-bout table loading.
`visualization/interactive_track_kinematics.py`, which backs the Marimo track
kinematics explorer, also uses the resolver for swim-bout option discovery and
payload loading. `analysis/plot_track_kinematics.py` uses the same resolver for
static swim-bout overlay span loading.

## Motivation

The object-count audit of `/nvme1/recordings` on 2026-05-08 found
`analysis/swim_bout_runs` as the largest metadata-object contributor:

```text
analysis/swim_bout_runs zarr.json files: 36,538
largest single analysis store: 10,358 swim_bout_runs zarr.json files
```

The current v1 layout is useful for canary exploration, but it is too
tree-shaped for long-lived production archives:

```text
analysis/swim_bout_runs/<run>/<speed_level>/<table>/<column_array>
```

Each parameter candidate becomes a run group, each speed or detector signal
level becomes a subgroup, and every structured table field becomes a separate
Zarr array. The result is high filesystem metadata fanout on NFS-like storage,
even though the underlying data are mostly ordinary tables.

The v2 goal is not to remove Zarr from swim-bout analysis. The goal is to store
accepted per-recording bout candidates as compact tables with explicit variant
IDs, while keeping broad parameter sweeps in scratch outputs or Parquet
sidecars until a candidate is promoted.

## Current V1 Shape

`src/fisheye/analysis/detect_bouts_multi_level.py` currently writes:

```text
analysis/swim_bout_runs/<run>/
  attrs:
    schema_id = "palette.swim_bout_runs"
    schema_version = 6
    detection_method
    default_level
    source_track_kinematics_run
    track_id
    provenance/lineage attrs

  speed_raw/
  speed_filtered/
  speed_smoothed/
  speed_averaged/
  speed_exponential/
    attrs:
      speed_level
      detection_signal_* attrs
      path_distance_source_level
      method parameters repeated from run attrs

    bouts/
    peak_events/
    inter_bout_intervals/
    inter_bout_interval_histogram/
    global_metrics/
    bout_points/
    detection_signal_mm_s        # exponential level only
    frame_indices                # exponential level only
```

The main downstream readers historically assumed this physical shape:

- `bout_kinematics.py` resolved `analysis/swim_bout_runs/<run>/<speed_level>`.
- `plot_track_kinematics.py` overlaid spans from `<speed_level>/bouts`.
- `interactive_track_kinematics.py` discovered candidates and read
  `<speed_level>` tables for Marimo.
- `export_cross_recording_analytics.py` exports rows from the latest run's
  `default_level`.
- Crimson's read contract currently names `<run>/<speed_level>` as the
  canonical candidate surface.

That means v2 needs a logical resolver layer before the writer can safely
change defaults.

## Design Goals

- Reduce Zarr group and array count for accepted swim-bout outputs.
- Preserve detector-vs-estimator semantics.
- Preserve exact bout boundaries, peak events, inter-bout intervals, and
  movement metrics.
- Make signal variants and parameter candidates selectable through tables, not
  through path names.
- Keep strict-JSON attrs and table payloads: non-finite metadata values must be
  represented as `null` in JSON strings/attrs, never `NaN` or `Infinity`.
- Keep v1 archives readable through adapters.
- Avoid broad sweep persistence in the canonical Zarr tree.

## Proposed V2 Layout

Use one compact run group per accepted or promoted swim-bout analysis:

```text
analysis/swim_bout_runs/<run>/
  attrs:
    schema_id = "palette.swim_bout_runs"
    schema_version = 7
    layout = "compact_tabular_v2"
    source_track_kinematics_run
    track_id
    default_candidate_id
    default_signal_id
    provenance/lineage attrs

  indexes/
    candidates/
    signal_variants/

  tables/
    bouts/
    peak_events/
    inter_bout_intervals/
    summary_metrics/
    histograms/                  # optional
    bout_points/                 # optional

  signals/
    detector_signal_mm_s         # optional dense (S,F) array
    frame_indices                # required if detector_signal_mm_s exists
```

Use the existing columnar-table writer style for v2 tables, but collapse
variant identity into table columns instead of child groups. This still creates
one Zarr array per column, but it avoids multiplying that table by every speed
level and every candidate branch.

### `indexes/candidates`

One row per detector parameterization promoted into the run.

Required columns:

```text
candidate_id                    int32 stable within run
candidate_name                  fixed/string
is_default                      bool
detection_method                enum/string: threshold | peak | peak_event
boundary_mode                   enum/string
boundary_window_s               float64
boundary_constraint             enum/string or empty
gap_merge_policy                enum/string
min_bout_duration_s             float64
min_gap_duration_s              float64
min_gap_frames                  int32, -1 when unset
parameter_hash                  fixed/string
parameters_json                 strict JSON string
provenance_json                 strict JSON string or empty
```

`parameters_json` stores full detector parameters once per candidate, including
peak-event thresholds, exponential parameters, gap handling, rounding policy,
and interpolation policy. Numeric "not applicable" values should serialize as
JSON `null`.

### `indexes/signal_variants`

One row per signal representation used by the candidate.

Required columns:

```text
signal_id                       int32 stable within run
signal_name                     fixed/string: raw | filtered | smoothed | averaged | exponential
role                            enum/string: physical_estimator | detector_response
source_level                    enum/string
transform_type                  enum/string: identity | exponential
transform_source_signal_id      int32, -1 when none
tau_s                           float64, NaN allowed in array payload only
units                           fixed/string
path_distance_source_level      enum/string
parameters_json                 strict JSON string
```

`speed_exponential` should be represented as a detector response whose movement
metrics are measured from a physical estimator signal. This keeps the existing
detector-vs-estimator contract explicit.

### `tables/bouts`

One row per detected bout per candidate and signal variant.

Required identity columns:

```text
bout_id                         int64 stable within candidate
candidate_id                    int32
signal_id                       int32 detector signal that produced the boundary
estimator_signal_id             int32 physical signal used for movement metrics
track_id                        int32
```

Required boundary columns:

```text
start_frame
end_frame
core_start_frame
core_end_frame
start_time_s
end_time_s
duration_s
observed_duration_s
core_start_time_s_interpolated
core_end_time_s_interpolated
core_duration_s_interpolated
```

Required physical metric columns:

```text
path_length_mm
path_length_px
net_displacement_mm
net_displacement_px
mean_speed_mm_s
mean_speed_px_s
peak_physical_speed_mm_s
valid_transition_fraction
gap_censored
```

Required detector metric columns:

```text
peak_detection_signal_mm_s
peak_detection_signal_px_s
peak_frame
peak_time_s
threshold_crossing_valid
```

V2 should keep physical metrics and detector metrics in the same bout row only
when the source IDs identify each value's provenance. That is the practical
compromise between compact storage and clear semantics.

### `tables/peak_events`

One row per accepted detector peak. Rows align to bouts when possible through
`bout_id`.

Required columns:

```text
peak_event_id
bout_id                         -1 when unassigned
candidate_id
signal_id
peak_frame
peak_time_s
peak_signal_value_mm_s
peak_prominence_mm_s
peak_width_s
peak_width_height_mm_s
boundary_mode
accepted
rejection_reason
```

### `tables/inter_bout_intervals`

One row per interval between successive bouts.

Required columns:

```text
interval_id
candidate_id
signal_id
prev_bout_id
next_bout_id
prev_end_frame
next_start_frame
prev_end_time_s
next_start_time_s
interval_s
valid
```

### `tables/summary_metrics`

One row per scalar summary metric. This replaces one-row structured
`global_metrics` groups and avoids adding columns for every future summary.

Required columns:

```text
candidate_id
signal_id
metric_name
value
units
source_table
```

Examples:

```text
n_bouts
total_bout_time_s
mean_bout_duration_s
mean_bout_peak_detection_signal_mm_s
mean_bout_peak_physical_speed_mm_s
total_path_length_mm
inter_bout_interval_mean_s
```

### `tables/histograms`

Optional. Histograms are useful for quick UI summaries, but they are
derivable from bout rows and inter-bout intervals. V2 should only write this
table when a caller explicitly requests persisted histograms.

Required columns:

```text
candidate_id
signal_id
metric_name
bin_left
bin_right
count
density
units
```

### `tables/bout_points`

Optional. `bout_points` is useful for overlays, but it can become large and is
also derivable from `bouts` plus track kinematics. V2 should write this only
when interactive consumers need it and should mark it as a cache/visualization
support table.

Required columns:

```text
candidate_id
signal_id
bout_id
point_role                      enum/string: start | end | peak
frame
time_s
x_px
y_px
x_mm
y_mm
```

### `signals/detector_signal_mm_s`

Optional dense detector response array:

```text
detector_signal_mm_s            (S,F)
frame_indices                   (F,)
```

`S` indexes rows in `indexes/signal_variants` where
`role == "detector_response"`. For the current exponential detector this will
usually be one signal. Do not store physical speed traces here; they belong to
`analysis/track_kinematics_runs`.

## Expected Object-Count Impact

The current v1 shape creates a repeated table set for every speed level:

```text
5 levels * 6 table families * one array per field
```

V2 writes each table family once per run, with `candidate_id` and `signal_id`
columns. A single accepted candidate with five signal variants should be tens
of metadata nodes, not hundreds. The largest feeding canary currently has
10,358 `analysis/swim_bout_runs` metadata objects; a comparable compact v2
target should be under a few hundred objects for accepted runs, excluding
deliberate scratch sweeps.

The exact reduction depends on which optional tables are written, but the main
win is structural: adding another signal variant adds rows to `signal_variants`
and `bouts`, not another subtree of tables.

## Compatibility Plan

Add a resolver API before changing writer defaults:

```text
discover_swim_bout_candidates(root) -> list[SwimBoutCandidate]
load_swim_bout_tables(root, run, candidate_id=None, signal_id=None) -> SwimBoutTables
load_default_swim_bout_tables(root, run="latest") -> SwimBoutTables
```

Resolver behavior:

- For v1 hierarchical runs, synthesize candidate and signal rows from run attrs
  and `<speed_level>` groups.
- For v2 compact runs, read `indexes/*` and `tables/*` directly.
- Preserve existing labels such as `filtered`, `smoothed`, `exponential`, and
  `default` in UI-facing option objects.
- Return normalized in-memory tables so Marimo, exporters, and
  `bout_kinematics.py` stop depending on physical path shape.

Keep old physical paths readable. Do not add v2 compatibility aliases that
materialize the full v1 hierarchy unless a concrete external reader requires
them.

## Writer Migration Plan

1. Add v2 schema constants and table dtype helpers.
2. Add the resolver/adapter with tests against small fake v1 and v2 stores.
3. Add `detect_bouts_multi_level --layout compact_v2` or
   `--schema-version 7` behind an explicit flag.
4. Write a v2 canary run with a new run ID; do not overwrite v1 runs.
5. Compare v1 default-level and v2 default-candidate outputs:
   bout count, start/end frames, peak frames, durations, path lengths, and
   inter-bout intervals must match within existing numeric tolerances.
6. Update Marimo, exporters, and `bout_kinematics.py` to use resolver-level
   semantics.
7. After readers are migrated, make compact v2 the default writer for promoted
   accepted runs. Keep broad sweeps in scratch outputs or Parquet sidecars.

## Validation Plan

- Unit-test strict JSON serialization for candidate and signal parameter
  records.
- Unit-test v1 adapter output from a minimal hierarchical fixture.
- Unit-test v2 native reader output from a minimal compact fixture.
- Unit-test that `speed_exponential` maps to `role="detector_response"` and an
  estimator signal ID rather than being mislabeled as physical speed.
- On the feeding canary, run v1-v2 equivalence checks for the current default
  candidate.
- Re-run `fisheye.utils.audit_zarr_group_counts` on the canary and record the
  object-count delta.

## Open Decisions

- Should v2 store histograms by default, or should histograms be UI/export-only
  derived products?
- Should `bout_points` be persisted by default, or generated on demand by
  joining `bouts` to track kinematics?
- Should a compact run contain multiple promoted candidates, or exactly one
  candidate with multiple signal variants? Multiple candidates are more compact;
  one candidate per run is easier to reason about.
- Should stable IDs be small integers only, or should tables also carry string
  UUID-style IDs for cross-run joins?
- How much direct Crimson compatibility should be provided before Crimson moves
  to a resolver/contract that understands v2?

## Recommendation

Implement the resolver first, then implement a v2 writer behind an explicit
flag. Do not migrate existing archives yet. Use the feeding canary as the first
equivalence and object-count comparison, because it is the current worst-case
`swim_bout_runs` outlier and already has downstream Marimo/Crimson coverage.
