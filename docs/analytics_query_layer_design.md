# Analytics Query Layer Design
<!-- design-meta
status: draft
last_updated: 2026-05-08
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
DuckDB   = query engine over the Parquet analytics lake
```

Parquet/Arrow exports should be deterministic, disposable, and reproducible
from Zarr plus SQLite registry metadata. They should not become the source of
truth for corrections or real-time viewing.

## Incremental Analytics Lake

The export layer should be an appendable analytics lake, not a set of one-off
CSV files and not one file per protocol hash.

```text
/nvme1/analytics_exports/palette_analytics/v1/
  manifests/
  sessions/
  stimulus_steps/
  swim_bouts/
  bout_kinematics/
  bout_classifications/
  stimulus_response_per_fish_step/
  stimulus_response_windows/
```

Each table is a directory of Parquet parts. New recordings add new parts. If a
recording is reprocessed after masks or keypoints are fixed, the exporter writes
a new part with a new `export_run_id` and a new `source_lineage_hash`; old rows
remain available for before/after comparison.

Protocol hashes should be columns, not the primary file organization. The
current exporter writes `protocol_signature_hash`, a deterministic SHA256 over
the ordered canonical stimulus-step definition, plus `derived_protocol_hash` as
a temporary alias for existing analysis notebooks. When a Citrus/registry
authored `protocol_hash` is available, it should be exported as a separate
exact-content protocol snapshot hash rather than replacing the step-signature
hash.

Query engines such as DuckDB can filter by protocol hash directly:

```sql
SELECT *
FROM read_parquet('/nvme1/analytics_exports/palette_analytics/v1/stimulus_response_per_fish_step/**/*.parquet')
WHERE protocol_signature_hash = 'd4e71b4fcd5272227de23db51b441eedcf36fca9ed1f350948a62909796d7287';
```

Partition by low-cardinality fields only after measuring query patterns and
file counts. Reasonable early partitions are `recording_date`,
`protocol_name`, or `stimulus_mode`; exact `recording_id`, `run_id`,
`subject_id`, and `protocol_hash` are usually too high-cardinality for default
partitioning.

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

## Export Workflow

The normal cross-recording workflow should be:

```text
1. Query registry.
   Select recordings/fish by DPF, protocol, cross, clutch, date, genotype,
   path, or other cohort fields.

2. Resolve source Zarrs.
   Use the registry result rows to locate each analysis Zarr.

3. Resolve selected analysis runs inside each Zarr.
   Examples: latest or explicitly selected stimulus run, track-kinematics run,
   swim-bout run, bout-kinematics run, stimulus-response run, eye-angle run,
   tail-posture run, or classifier run.

4. Extract table-shaped metrics.
   Read only the Zarr arrays and attrs needed for the requested export tables.
   Keep dense traces, masks, images, and full geometry in Zarr unless a table
   export has a concrete use case.

5. Join registry context onto rows.
   Add DPF, cross, clutch, fish/subject identity, protocol name/hash, recording
   date, genotype, strain, arena, camera, and selected run IDs.

6. Write Parquet table parts.
   Append or write immutable part files into stable table directories such as
   `stimulus_response_per_fish_step/`, `swim_bout_metrics/`, or
   `bout_classifications/`.

7. Write an export manifest.
   Record source Zarrs, source run IDs, Palette commit, schema versions,
   row counts, export parameters, and lineage hashes.
```

In short:

```text
SQLite registry = cohort selector and control plane
Zarr archives   = authoritative source arrays and run provenance
Parquet tables  = queryable cross-session cache
DuckDB/Polars   = analysis/query engines over Parquet
```

For example, exporting "latency to follow the grating" should read OMR
per-fish/per-step arrays from each selected Zarr's `stimulus_response_runs`,
attach registry context, and write rows to
`stimulus_response_per_fish_step`. Subsequent cohort comparisons should query
that Parquet table rather than reopening every Zarr.

The export manifest should record:

- export run ID and creation time
- Palette git commit and dirty state
- registry path, selection query, or virtual collection manifest path
- collection ID and collection manifest SHA-256 when a virtual collection is
  used
- source Zarr paths and source run IDs
- table schema versions
- row counts
- export parameters

## First Tables To Implement

Start with scalar/event tables. Do not begin with masks, full contours, images,
or dense probability arrays.

Recommended first datasets:

- `recording_summary`: one row per selected recording/archive
- `stimulus_steps`: one row per canonical stimulus step
- `stimulus_step_summary`: one row per recording, fish, selected response/run
  lineage, and stimulus step
- `swim_bout_metrics`: one row per swim-bout candidate
- `bout_kinematics_metrics`: one row per bout-kinematics measurement
- `bout_classifications`: one row per classified bout
- `stimulus_response_per_fish_step`: one row per recording, step, and fish
- `stimulus_response_windows`: one row per recording, step, fish, and window
- `stimulus_response_per_bout`: one row per bout assigned to a stimulus step

Add `track_kinematics_timeseries` later if there is a concrete need for
cross-recording frame-level queries. It can be large, and many questions can be
answered from event/per-step summaries first.

## First Metric Families

The first export should favor raw-ish scalar facts over frozen summary plots.
That means exporting per-bout and per-step values from which medians,
histograms, and cohort summaries can be recomputed later.

### Recording/session level

`recording_summary` should include:

- total path length / cumulative distance;
- total moving time and fraction moving;
- mean and median speed while moving, plus whole-recording mean speed;
- total bout count and bout rate per minute;
- mean and median inter-bout interval;
- mean and median bout duration;
- mean and median bout path length and net displacement;
- mean and median peak speed and mean bout speed;
- mean and median vergence, including eye-frame/Bianco-style vergence when
  available;
- optional percent time above a configurable convergence threshold;
- QC coverage fields: tracking coverage, valid movement fraction,
  valid eye-angle fraction, valid bout-kinematics fraction, valid tail/posture
  fraction.

### Stimulus-step level

`stimulus_step_summary` should include:

- step duration, stimulus mode, direction/polarity, protocol name/hash;
- path length / cumulative distance during the step;
- bout count, bout rate, moving fraction;
- mean and median bout duration;
- mean and median bout path length and net displacement;
- mean and median peak speed and mean bout speed;
- mean and median inter-bout interval within or assigned to the step;
- mean and median net heading change per bout;
- mean and median within-bout heading range or peak-to-peak heading change;
- mean and median vergence during the step and around bouts;
- for translational OMR: `omr_path_index`, `bout_fraction_correct`,
  first aligned/classified/opposing bout latencies, and aligned/opposing counts;
- for concentric OMR: radial OMR index, radial/tangential displacement, radial
  polarity, and centering-success fields when target annulus metadata exists.

### Bout level

`swim_bout_metrics` / `bout_kinematics_metrics` should include enough columns
to rebuild distributions:

- bout identity, selected source run IDs, start/end/core frames, and duration;
- stimulus step assignment and stimulus context;
- path length, net displacement, and displacement/path ratio;
- peak speed and mean speed;
- pre/post heading, net heading change, and absolute heading change;
- within-bout heading range / peak-to-peak heading change;
- inter-bout interval before and after;
- OMR score and aligned/opposing/ambiguous label when applicable;
- bout classification label/probabilities when available;
- tail posture summaries when available, such as max absolute tail angle and
  tail-angle energy;
- eye summaries around the bout, such as pre/post/mean vergence,
  eye-frame vergence, and left/right eye angles.

### Histogram-ready exports

Do not make pre-binned histograms the only export surface. Store the per-bout
facts first:

```text
duration_s
path_length_mm
net_displacement_mm
peak_speed_mm_s
mean_speed_mm_s
inter_bout_interval_s
heading_change_deg
within_bout_heading_range_deg
vergence_eye_angle_deg
```

Merged histograms can then be rebuilt with whatever bins the downstream
analysis chooses. A compact `metric_histogram_counts` table may be added later
for dashboards, but it should reference the source table and bin policy:

```text
dataset_id
recording_id
protocol_hash
step_index
metric_name
bin_policy
bin_left
bin_right
count
source_table
source_lineage_hash
```

## Required Common Columns

Every exported row should carry enough identity to map back to the authoritative
Zarr source:

```text
export_run_id
export_created_at_utc
recording_id
dataset_id
session_uuid
subject_id
fish_id
dish_id
cross_id
clutch_id
dpf_at_acquisition
line_strain
genotype
protocol_name
protocol_hash
protocol_signature_hash
derived_protocol_hash
protocol_semantic_hash
zarr_path
zarr_mtime_ns
stage_family
run_id
schema_id
schema_version
source_refs_json
source_lineage_hash
is_latest
supersedes_export_run_id
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

For a grating OMR cohort question:

```sql
SELECT
  cross_id,
  count(*) AS n_fish_steps,
  median(first_aligned_bout_latency_s) AS median_latency_s,
  avg(omr_path_index) AS mean_omr_path_index
FROM read_parquet('/nvme1/analytics_exports/palette_analytics/v1/stimulus_response_per_fish_step/**/*.parquet')
WHERE dpf_at_acquisition = 6
  AND protocol_name = 'DefaultScreen'
  AND stimulus_mode = 'MOVING_GRATING'
  AND first_aligned_bout_latency_s IS NOT NULL
GROUP BY cross_id
ORDER BY median_latency_s;
```

For comparing the same fish across protocols, use `subject_id` only if it is a
true cross-session biological identity. If the identity is only known at the
dish, cross, or clutch level, group at that level instead.

## Registry Role

SQLite should remain the operational registry and control plane. It should own:

- recordings and datasets
- subject metadata such as DPF and genotype
- protocol/session metadata
- Zarr paths
- pipeline step status
- run discovery and selected/current run policy
- lineage and staleness signals

SQLite does not need to store every dense frame-level metric. It should index
immutable collection and export manifests so users can find exported analytics
products quickly without scanning the Parquet lake:

```text
analytics_collections
  collection_id
  manifest_sha256
  collection_name
  manifest_path
  record_count
  included_record_count
  status

analytics_exports
  export_run_id
  collection_id
  collection_manifest_sha256
  output_root
  export_manifest_path
  created_at_utc
  source_recording_count
  table_count
  row_counts_json
  tables_json
  status

analytics_export_tables
  export_run_id
  table_name
  table_path
  row_count
  part_count
```

The registry should also remain the place that tells the exporter which Zarrs
belong to a cohort, for example "6 dpf DefaultScreen fish from cross A and the
newest clutch." The Parquet table should denormalize those registry fields so
downstream DuckDB queries do not need to repeatedly join back to SQLite for
common biological filters.

Operational commands:

```bash
scripts/py -m fisheye.utils.index_analytics_manifests \
  --registry /nvme1/palette_registry.sqlite \
  --export-manifest /nvme1/exports/palette_analytics/v1/manifests/export_run_id=<id>.json

scripts/py -m fisheye.utils.query_analytics_exports \
  --registry /nvme1/palette_registry.sqlite \
  --collection-id movement_bouts_20260128_all_analysis_v002 \
  --table swim_bout_metrics
```

## Data Versioning And DVC

The first line of data versioning should be Palette-native lineage:

```text
export_run_id
source_lineage_hash
source run IDs
manifest JSON
supersedes_export_run_id
```

This is what tells a query whether a row came from old masks/keypoints or from a
newly recomputed downstream run.

DVC can be added later around selected export snapshots. It should not replace
Parquet, DuckDB, or Palette lineage metadata:

```text
Parquet + manifests = queryable exported facts and source lineage
DVC                 = versioned large-file/directory snapshots outside Git
DuckDB              = SQL query engine over Parquet
```

Good DVC candidates are frozen Parquet export snapshots, export manifests,
training datasets, and model checkpoints. Poor DVC candidates are hot mutable
analysis Zarrs and every transient canary export.

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

## Implemented First Slice

The first exporter slice is implemented as:

```bash
scripts/py -m fisheye.utils.export_cross_recording_analytics \
  --collection-manifest /nvme1/exports/palette_analytics/manifests/collections/movement_bouts_20260128_all_analysis_v002.manifest.json \
  --output-root /nvme1/exports/palette_analytics \
  --jobs 4
```

Implemented tables:

- `recording_summary`
- `stimulus_steps`
- `stimulus_step_summary`
- `stimulus_response_per_fish_step`
- `swim_bout_metrics`
- `bout_kinematics_metrics`

The exporter parallelizes Zarr extraction by recording, writes immutable
per-recording Parquet part files under
`<output-root>/v1/<table>/export_run_id=<run_id>/`, and writes a manifest under
`<output-root>/v1/manifests/`. Generated export IDs are prefixed with `run_`
so hive-partition readers such as Polars treat them as strings instead of
trying to parse compact UTC timestamps as dates.

When `--collection-manifest` is provided, included manifest records provide the
source Zarr list. The export manifest records the collection manifest path,
`collection_id`, and `manifest_sha256`; every exported row also carries
`collection_id`, `collection_manifest_sha256`, and
`collection_manifest_path`. This makes Parquet rows traceable back to the exact
immutable selection document used for the export.

`bout_kinematics_metrics` is additive and reads existing Zarr
`analysis/bout_kinematics_runs/<run>/<measurement_level>/per_bout_metrics`
tables without changing the Zarr writer. It exports one row per bout per
measurement level, including `movement`, `heading_raw`, `heading_smoothed`,
and optional `eye_gaze` levels when present. The table carries source run
identity, stimulus-step assignment, and heading-change fields such as
`net_delta_heading_deg`, `abs_net_delta_heading_deg`,
`within_heading_path_deg`, and angular-speed summaries.

Example Polars query:

```python
import polars as pl

run_id = "run_20260505T080239Z"
root = "/nvme1/exports/palette_analytics"

bouts = pl.scan_parquet(
    f"{root}/v1/swim_bout_metrics/export_run_id={run_id}/*.parquet",
    hive_partitioning=True,
)

summary = (
    bouts.group_by("stimulus_mode")
    .agg(
        pl.len().alias("bouts"),
        pl.col("duration_s").mean().alias("mean_duration_s"),
        pl.col("path_length_mm").mean().alias("mean_path_length_mm"),
        pl.col("peak_physical_speed_mm_s").mean().alias("mean_peak_speed_mm_s"),
    )
    .collect()
)
```

Reproducible cross-recording bout-kinematics plots can be generated from the
Parquet export, without rereading source Zarr archives:

```bash
scripts/py -m fisheye.utils.plot_cross_recording_bout_kinematics \
  --export-root /nvme1/exports/palette_analytics \
  --export-run-id latest \
  --output-dir /tmp/palette_lab_plots/latest_parquet \
  --measurement-level heading_smoothed
```

This writes overall heading histograms, net heading-change by stimulus mode,
angular-speed histograms, and JSON/TSV summaries from
`bout_kinematics_metrics`.

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
