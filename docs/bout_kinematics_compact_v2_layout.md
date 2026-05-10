# Bout Kinematics Compact-V2 Layout

Date anchored: 2026-05-10

Status: initial opt-in writer and resolver contract.

## Purpose

`analysis/bout_kinematics_runs` historically used one Zarr subgroup per
measurement family:

```text
analysis/bout_kinematics_runs/<run>/
  movement/per_bout_metrics/
  heading_smoothed/per_bout_metrics/
  heading_raw/per_bout_metrics/
  eye_gaze/per_bout_metrics/
```

That layout is clear, but it creates many metadata objects as measurement
families and variants grow. Compact-v2 keeps the same logical tables while
storing them as fewer columnar tables under one run group.

## Run-Level Contract

A compact run is identified by:

```text
analysis/bout_kinematics_runs/<run>.attrs["layout"] = "compact_tabular_v2"
```

The writer remains opt-in. The default layout is still `hierarchical_v1`.

Run attrs must retain the same source and provenance fields as hierarchical
runs, including:

- `source_track_kinematics_run`
- `source_swim_bout_run`
- `source_swim_bout_speed_level`
- `source_track_id`
- `heading_levels`
- `default_heading_level`
- `analysis_levels`
- `parameters`
- `provenance`

The run `parameters` and provenance artifact metadata must include the chosen
`layout`.

## Physical Tables

Compact-v2 writes these tables directly under the run group:

```text
level_index
movement_metrics
heading_metrics
eye_gaze_metrics        # optional, only when eye gaze is requested
```

`level_index` is the semantic table catalog. It records each logical analysis
level, measurement family, heading-level id, default-heading marker, and row
count.

`movement_metrics` stores the logical `movement` rows. It includes compact
index columns:

- `analysis_level_id`
- `analysis_level_bytes`
- `heading_level_id`
- `heading_level_bytes`

`heading_metrics` concatenates all heading-level rows. The
`heading_level_bytes` column distinguishes `heading_smoothed`, `heading_raw`,
or future heading variants.

`eye_gaze_metrics` stores optional eye-gaze rows with the same compact index
columns.

Reader-facing metric schemas are unchanged after resolving: compact index
columns are storage metadata and should not appear in exported analytics rows.

## Resolver Contract

Callers should not branch on physical paths. Use:

```python
resolve_bout_kinematics_tables(run_group, heading_level=None)
```

The resolver:

- reads both `hierarchical_v1` and `compact_tabular_v2`
- returns records keyed by logical analysis level
- strips compact index columns from returned records
- returns level attrs and table attrs for labels/provenance
- supports filtering to one logical level, such as `raw`, `smoothed`,
  `movement`, or `eye_gaze`

Interactive notebooks, Parquet exporters, and Crimson should consume this
logical API or implement equivalent layout-aware reads.

## Visualization Artifacts

Compact-v2 currently rejects `--write-zarr-artifacts`. Existing visualization
artifact writers still reference hierarchical `*/per_bout_metrics` source
paths. This is intentional until visualization specs are updated to reference
logical resolver paths or compact table paths.

## Migration Policy

Compact-v2 is additive and opt-in:

- Existing hierarchical runs remain valid.
- Existing readers should migrate through the resolver before compact becomes
  default.
- No derived values are recomputed solely because the layout changes.
- A compact run may be generated from the same upstream track, swim-bout, and
  eye-angle sources as a hierarchical run; provenance distinguishes the layout.
