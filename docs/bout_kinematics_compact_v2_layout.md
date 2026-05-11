# Bout Kinematics Compact-V2 Layout

Date anchored: 2026-05-10

Status: default writer, resolver, and visualization artifact contract.

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

The writer defaults to compact-v2 as of 2026-05-11. `hierarchical_v1` remains
available as an explicit compatibility/debug layout.

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

Compact-v2 supports `--write-zarr-artifacts` as of 2026-05-11. Persisted PNG
snapshots and interactive plot specs use the same artifact names as
hierarchical-v1 runs, but their provenance is layout-aware:

- `source_paths` point at compact physical tables such as `movement_metrics`,
  `heading_metrics`, and `eye_gaze_metrics`.
- `source_filters` identify the logical rows inside each compact table, such as
  `heading_level_bytes = "heading_smoothed"` or
  `analysis_level_bytes = "movement"`.
- Artifact `parameters` include `layout = "compact_tabular_v2"` so signatures
  differ from hierarchical-v1 artifacts even when plotted values are identical.

Readers should treat `source_paths` plus `source_filters` as the compact
equivalent of hierarchical `*/per_bout_metrics` paths. The rendered PNG remains
a snapshot; the interactive spec remains the canonical machine-readable
description of how to re-read the plotted source data.

## Migration Policy

Compact-v2 is additive and default for new promoted bout-kinematics runs:

- Existing hierarchical runs remain valid.
- Existing readers should continue using the resolver instead of physical-path
  branching.
- No derived values are recomputed solely because the layout changes.
- A compact run may be generated from the same upstream track, swim-bout, and
  eye-angle sources as a hierarchical run; provenance distinguishes the layout.

Crimson validation update, 2026-05-11: Crimson loaded compact-v2
bout-kinematics metrics from the feeding canary, including `movement`,
`heading_smoothed`, `heading_raw`, and `eye_gaze` tables with 519 rows each, and
loaded the fresh compact bout-kinematics candidate with 519 bouts. The Crimson
consumer gate is passed for reading current compact-v2 bout-kinematics runs.

Default update, 2026-05-11: after Palette resolver/export validation, Crimson
consumer validation, and compact visualization artifact support were in place,
`compute_and_save_bout_kinematics(...)` and the CLI `--layout` default changed
from `hierarchical_v1` to `compact_tabular_v2`. Use
`--layout hierarchical_v1` only when a legacy physical tree is specifically
needed.

Smoke-run policy, 2026-05-11: real-Zarr validation runs may be written with
explicit `smoke` names, but should not remain the parent `latest` selection
unless intentionally promoted. On the feeding canary, the compact eye-angle
smoke run proved `--eye-angle-run latest` could resolve a compact-dense-v2
source, then `analysis/bout_kinematics_runs.attrs["latest"]` was restored to
`bk_tk_hyst4_low2_latch_s005_peak_event_prom4_w098_compact_v2_canary_20260510`
and metadata was reconsolidated. The smoke run remains available for explicit
regression checks.
