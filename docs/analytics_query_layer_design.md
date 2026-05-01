# Analytics Query Layer Design
<!-- design-meta
status: draft
last_updated: 2026-05-01
-->

Purpose: clarify how Palette should support biological cross-recording queries
without weakening the Zarr-based archive and real-time visualization model.

This note is intentionally architectural. The more detailed export schema draft
lives in
[cross_recording_analytics_export_design.md](cross_recording_analytics_export_design.md).

## Problem

Palette analysis data is currently centered on per-recording Zarr archives.
That is the right representation for long videos, masks, tracks, keypoints,
stimulus-aligned traces, and real-time visualization. It is not the right
primary interface for questions such as:

```text
Give me all swim bouts from fish of age 6 dpf, genotype X, within trial type Y.
```

That query crosses several concepts:

- registry metadata: recording, subject, DPF, genotype, protocol, dataset path
- selected analysis runs: track kinematics, stimulus, swim bouts, bout
  kinematics, stimulus response
- row-level analytic facts: bout boundaries, duration, speed, path length,
  validity, trial/stimulus context

Answering this directly from Zarr requires opening many archives, resolving
current runs, reading per-run arrays, and joining them back to registry
metadata. That traversal should be implemented once by an export/index layer,
not re-created by each downstream analysis.

## Decision

Keep the current authority split:

```text
Zarr     = authoritative per-recording artifact store
SQLite   = registry, lineage, metadata, status, and run discovery
Parquet  = derived cross-recording analytic query/export tables
```

Parquet/Arrow exports should be deterministic, disposable, and reproducible
from Zarr plus SQLite registry metadata. They should not become the source of
truth for corrections or real-time viewing.

## Why Zarr Should Stay Primary

Zarr remains the correct backing store for Palette's pipeline because it
supports:

- chunked reads over long recordings
- real-time and interactive visualization in Crimson or other viewers
- frame-window access without loading full recordings
- array-native data: images, masks, keypoints, tracks, probabilities, contours,
  and dense time series
- run-local provenance and visualization artifacts near their source data

The real-time visualization requirement is a hard constraint. Viewers should be
able to open a recording archive and read small time/frame windows from arrays
such as tracks, masks, stimulus annotations, and derived traces. Parquet should
not sit in that path.

## Where Zarr Becomes Awkward

Zarr is less ergonomic for cross-recording table questions:

- filter all bouts by genotype, DPF, protocol, or trial type
- join bout metrics to subject metadata and stimulus context
- compute cohort summaries over many recordings
- feed scalar/event features into DuckDB, Polars, R, or tabular ML workflows
- share compact analytic datasets without copying heavy archive arrays

Those are table operations. They should use a table-shaped export.

## Proposed Query Layer

Add a deferred CLI along these lines:

```bash
scripts/py -m fisheye.utils.export_cross_recording_analytics \
  --registry /nvme1/palette_registry.sqlite \
  --output-root /nvme1/analytics_exports \
  --tables recording_summary,swim_bout_metrics,bout_kinematics_metrics \
  --selection-query <name-or-sql-or-manifest>
```

The command should:

1. Select source recordings from the registry.
2. Resolve exact analysis Zarr paths and selected/current run IDs.
3. Read table-shaped metrics from Zarr.
4. Join registry metadata onto each analytic row.
5. Write partitioned Parquet tables plus a manifest.

The export manifest should record:

- export ID and creation time
- Palette git commit and dirty state
- registry path
- selection query or manifest
- source Zarr paths and source run IDs
- table schema versions
- row counts
- export parameters

## First Tables To Implement

Start with scalar/event tables. Do not begin with masks, full contours, images,
or dense probability arrays.

Recommended first datasets:

- `recording_summary`: one row per selected recording/archive
- `swim_bout_metrics`: one row per swim-bout candidate
- `bout_kinematics_metrics`: one row per bout-kinematics measurement
- `stimulus_response_per_step`: one row per recording, step, and fish
- `stimulus_response_per_bout`: one row per bout assigned to a stimulus step

Add `track_kinematics_timeseries` later if there is a concrete need for
cross-recording frame-level queries. It can be large, and many questions can be
answered from event/per-step summaries first.

## Required Common Columns

Every exported row should carry enough identity to map back to the authoritative
Zarr source:

```text
export_id
recording_id
dataset_id
subject_id
zarr_path
zarr_mtime_ns
stage_family
run_id
schema_id
schema_version
source_refs_json
```

Rows should also denormalize biological and protocol metadata needed for common
filters:

```text
dpf_at_acquisition
genotype
line_strain
species
recording_type
recording_subtype
behavior_mode
protocol_name
trial_type or stimulus_mode
arena_id
camera_id
started_utc
```

For swim bouts, include event identity and metrics:

```text
track_id
bout_id
speed_level
start_frame
end_frame
start_time_s
end_time_s
duration_s
path_length_mm
net_displacement_mm
mean_speed_mm_s
peak_physical_speed_mm_s
valid
failure_reason
```

With that export, the motivating query becomes a normal SQL/DataFrame query:

```sql
SELECT *
FROM read_parquet('/nvme1/analytics_exports/*/swim_bout_metrics/**/*.parquet')
WHERE dpf_at_acquisition = 6
  AND genotype = 'X'
  AND trial_type = 'Y'
  AND valid;
```

## Registry Role

SQLite should remain the operational registry and control plane. It should own:

- recordings and datasets
- subject metadata such as DPF and genotype
- protocol/session metadata
- Zarr paths
- pipeline step status
- run discovery and selected/current run policy
- lineage and staleness signals

SQLite does not need to store every dense frame-level metric. It may optionally
track generated Parquet exports in a small registry table later:

```text
analytics_exports
  export_id
  output_root
  manifest_path
  created_at_utc
  source_recording_count
  table_names_json
  schema_versions_json
  status
```

## Zarr Writer Guidance

Continue writing per-recording analysis outputs to Zarr. For table-shaped
outputs inside a run, prefer the existing columnar Zarr group pattern over
opaque structured arrays. That keeps row fields independently readable while
preserving Zarr as the local archive.

For real-time visualization, keep arrays chunked around expected access
patterns:

- frame/time windows for traces and stimulus annotations
- row chunks for per-detection/per-bout tables
- ROI/channel chunks for masks and image-like outputs

Do not move large array payloads to Parquet by default.

## Dense Track Kinematics Follow-Up

Current track kinematics stores sparse per-track arrays with `frame_indices`.
That is compact and auditable, but it pushes sparse-to-dense expansion into
downstream consumers. The existing
[analysis_dense_array_migration_todo.md](analysis_dense_array_migration_todo.md)
proposes adding dense frame-aligned arrays with explicit validity masks.

That direction is compatible with this design:

- keep sparse arrays for provenance and compact reads
- add dense arrays when repeated frame-window consumers justify it
- preserve gap-aware distance semantics from `track_kinematics`
- let real-time visualization read dense windows directly when available

## Non-Goals

- Do not replace Zarr as the primary archive.
- Do not require Parquet for Crimson or real-time visualization.
- Do not edit Parquet exports as authoritative corrections.
- Do not export dense masks, probability volumes, raw video, or full geometry by
  default.
- Do not create one giant wide table for all analytics.

## Deferred Implementation Plan

1. Inventory current scalar/event outputs in `swim_bout_runs`,
   `bout_kinematics_runs`, and `stimulus_response_runs`.
2. Implement a small library for loading columnar Zarr groups into row
   dictionaries or Arrow/Polars frames with source identity columns.
3. Add `export_cross_recording_analytics` with `recording_summary` and
   `swim_bout_metrics` first.
4. Join registry metadata from `recordings`, `datasets`, and
   `recording_subjects`.
5. Write a manifest and partitioned Parquet output.
6. Add DuckDB and Polars smoke queries for the motivating genotype/DPF/trial
   query.
7. Register export manifests in SQLite only after the export format stabilizes.

## Bottom Line

The current Zarr-first design is appropriate. The repository should not pivot
away from Zarr. The next layer should be a deterministic analytic export/query
surface that turns selected Zarr runs plus registry metadata into Parquet
tables for cross-recording biological questions.
