# Stimulus Response Compact-V2 Design

<!-- design-meta
status: accepted
last_updated: 2026-05-11
-->

## Purpose

`analysis/stimulus_response_runs` is the next high-value compact-layout target
after swim bouts and bout kinematics. The current layout is clear for manual
inspection, but it fans out by protocol step, stimulus family, metric scope, and
window/trial family. That fanout is now one of the largest sources of Zarr
metadata objects.

This document defines the resolver-first migration plan. Compact-tabular-v2 is
the default writer layout as of 2026-05-11; hierarchical-v1 remains available
as an explicit compatibility/debug layout.

Implementation status:

- `fisheye.analysis.stimulus_response_io.resolve_stimulus_response_tables(...)`
  reads hierarchical-v1 and compact-tabular-v2 layouts and exposes logical
  tables.
- The cross-recording exporter, OMR plotter, and Marimo stimulus-response panels
  now use the resolver for the paths covered by moving-grating OMR and
  concentric radial OMR.
- `fisheye.analysis.stimulus_response` defaults to
  `--layout compact_tabular_v2`. Use `--layout hierarchical_v1` only when a
  legacy physical tree is explicitly required.

## Dense Time-Series Decision

Decision, 2026-05-11: compact-v2 does not persist dense per-frame or
time-series stimulus-response tables by default. This is intentional, not a
missing implementation slice.

The compact stimulus-response run is the query-grade derived-response surface:
per-fish, per-step, per-bout, per-window, early-window, and trial summaries.
Dense traces for plotting and debugging should be reconstructed from upstream
time-series sources through resolvers:

- `analysis/track_kinematics_runs` for aligned position, heading, speed, and
  movement validity traces.
- `analysis/swim_bout_runs` for bout detector intervals and detector-response
  signals.
- `analysis/stimulus_runs` for frame alignment, step spans, and stimulus
  metadata.
- `analysis/eye_angle_runs` or other modality-specific runs when response
  plots need dense eye/tail/pose traces.

This avoids duplicating upstream physical signals into every
`stimulus_response_runs/<run>` and prevents compact-v2 from recreating the old
metadata-fanout problem with large per-frame tables. A future optional dense
cache is allowed only if a named consumer needs it for performance. If added,
it should be clearly marked as a cache or separate compact table family, carry
source references/fingerprints, and not become the authoritative source for
physical track, bout, eye-angle, or stimulus time-series.

## Current Hierarchical-V1 Shape

Current writer: `src/fisheye/analysis/stimulus_response.py`.

Physical layout:

```text
analysis/stimulus_response_runs/<run>/
  attrs
  global/
    <recording-wide per-fish arrays>
    omr/per_fish/
  frames/
    <recording-wide stimulus annotations>
  steps/
    step_<index>/
      attrs
      per_fish/
      per_bout/
      grating/
        per_frame/
        per_fish/
        time_series/
        omr/
          per_fish/
          per_bout/
          windows/
          early_windows/
      concentric_grating/
        per_frame/
        per_fish/
        time_series/
        radial_omr/
          per_frame/
          per_fish/
          per_bout/
          windows/
          early_windows/
      looming/
        trials/
        per_frame/
        per_trial_per_fish/
        per_fish/
        time_series/
  visualizations/
```

The scientific model is sound: step-level outputs are tied to canonical
stimulus steps, and each stimulus family only exists when it applies. The
physical tree is the problem. Every extra step and family creates a new group
subtree with many small column arrays.

## Current Readers

Known readers now use the shared resolver for the paths needed by current
moving-grating OMR and concentric radial OMR workflows:

- `src/fisheye/utils/export_cross_recording_analytics.py`
  reads logical per-step and per-fish tables from
  `resolve_stimulus_response_tables(...)`, then joins optional grating,
  moving-grating OMR, concentric, and concentric radial-OMR summaries.
- `apps/marimo/track_kinematics_explorer.py`
  reads moving-grating OMR through `load_omr_step_summaries(...)` and has a
  local helper for concentric radial OMR tables.
- `src/fisheye/analysis/plot_stimulus_response_omr.py`
  reads moving-grating OMR through the shared resolver for PNG and interactive
  artifacts. Persisted artifact specs record layout-aware `source_paths`: for
  compact-v2 these point at `moving_grating_omr_per_fish`,
  `moving_grating_omr_per_bout`, `moving_grating_omr_windows`, and
  `moving_grating_omr_early_windows`, with `source_filters` carrying the
  selected `step_index` / `metric_family` constraints.

Additional future readers should use
`resolve_stimulus_response_tables(...)` rather than branching on physical Zarr
paths directly.

## Resolver Contract

Shared logical loader:

```python
resolve_stimulus_response_tables(run_group) -> StimulusResponseTables
```

The resolver reads `layout == "hierarchical_v1"` and
`layout == "compact_tabular_v2"` runs and returns the same logical tables.

Recommended logical outputs:

```text
run_attrs
global_per_fish
frame_annotations
step_index
step_per_fish
step_per_bout
stimulus_per_frame
stimulus_time_series
omr_per_fish
omr_per_bout
omr_windows
omr_early_windows
looming_trials
looming_per_trial_per_fish
```

Required join keys:

```text
step_index
stimulus_family          # moving_grating, concentric_grating, looming, none
metric_family            # base, grating, moving_grating_omr, radial_omr, looming
track_id                 # compatibility alias: fish_id
subject_id               # optional future identity key
bout_id                  # when bout-scoped
window_id                # when window-scoped
trial_id                 # when trial-scoped
frame_index              # when frame-scoped
```

For v1 reads, `fish_id` should be exposed as `track_id` while preserving a
compatibility `fish_id` column for exported tables. The underlying current
single-fish records are track-indexed in practice; future multi-subject work
should introduce `subject_id` without changing old run semantics.

## Compact-Tabular-V2 Layout

Compact runs write fewer groups and put step/family identity into columns or
index arrays:

```text
analysis/stimulus_response_runs/<run>/
  attrs:
    layout = "compact_tabular_v2"
    schema_id
    schema_version
    method
    method_version
    source_refs
    source_fingerprints
  step_index/
  global_per_fish/
  frame_annotations/
  step_per_fish/
  step_per_bout/
  grating_per_fish/
  moving_grating_omr_per_fish/
  moving_grating_omr_per_bout/
  moving_grating_omr_windows/
  moving_grating_omr_early_windows/
  concentric_per_fish/
  concentric_radial_omr_per_fish/
  concentric_radial_omr_per_bout/
  concentric_radial_omr_windows/
  concentric_radial_omr_early_windows/
  looming_trials/
  looming_per_trial_per_fish/
  looming_per_fish/
  visualizations/
```

Notes:

- `step_index` is the canonical in-run table for step identity and should carry
  step metadata needed by local readers. It should not replace the upstream
  `analysis/stimulus_runs/<run>/steps`, which remains the canonical protocol
  import.
- `step_per_fish` stores base movement and coverage metrics for all steps.
- This first compact slice keeps family-specific table names rather than one
  metric-long OMR table. That preserves current reader/export ergonomics while
  still removing the per-step subtree fanout.
- The high-volume per-frame and time-series tables are intentionally omitted
  from compact-v2 by default: `grating_per_frame`, `grating_time_series`,
  `concentric_per_frame`, `concentric_time_series`,
  `concentric_radial_omr_per_frame`, `looming_per_frame`, and
  `looming_time_series`. The run records these in `compact_omitted_tables`.
  Dense visual/debug traces are sourced from upstream time-series runs through
  resolvers rather than duplicated here.
- Looming remains a stimulus-response family, not a separate top-level run
  family. Its trial-scoped data belongs in `looming_trials` and
  `looming_per_trial_per_fish`.

## Migration Phases

1. Add the resolver and keep the writer unchanged. Done 2026-05-10.
2. Move exporter, Marimo, and OMR plotting through the resolver. Done for
   current moving-grating OMR and concentric radial OMR readers on 2026-05-10.
3. Add focused tests proving resolver parity on a hierarchical-v1 fixture. Done
   for the resolver, exporter, and OMR plotter on 2026-05-10.
4. Add an opt-in `--layout compact_tabular_v2` writer mode. Done 2026-05-10.
5. Write and validate one real canary run. Done 2026-05-10.
6. Add Crimson/Marimo/export smoke checks for v2. Done 2026-05-10/11 for
   current Palette consumers; Crimson stimulus-response read support remains a
   separate consumer contract when Crimson starts rendering these metrics.
7. Change the writer default after compact visualization artifact specs became
   layout-aware. Done 2026-05-11.

## Non-Goals

- Do not split OMR into a separate top-level `omr_runs` family. OMR is a
  stimulus-response metric family.
- Do not duplicate upstream track, bout, eye-angle, or stimulus arrays. Store
  source references, source revisions, and derived metric outputs.
- Do not remove hierarchical-v1 readers until historical runs and external
  readers have resolver-backed compatibility.
- Do not fill non-applicable families with NaN rows. Missing families should
  usually be represented by no rows for that family.

## Open Decisions

- Whether a named UI or batch consumer needs an optional dense cache table for
  performance. The default decision is no persisted dense time-series in
  compact-v2; any future cache must be explicitly marked as cache/derived view
  and sourced from upstream traces.
- Whether family-specific per-fish metrics should be wide tables
  (`omr_per_fish`) or a metric-long table. Wide tables are easier for current
  Marimo/export code; metric-long tables can be better for schema evolution.
- Whether visual artifact specs should point to logical resolver table names or
  physical v1/v2 paths. Logical names are preferred for compact migration.
- Whether `track_id` should replace `fish_id` in v2 persisted arrays, with
  `fish_id` retained only in resolver/export compatibility views.
