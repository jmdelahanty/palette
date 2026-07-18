<!-- ARCHIVED 2026-07-17: rejected in favor of recording-local Zarr authority and external immutable cross-recording exports. -->

# Zarr-Local Parquet Sidecar Exports

## Status

Design note for deferred implementation.

## Context

Palette currently persists many derived analysis products directly as Zarr
groups and arrays. This is appropriate for canonical long-lived arrays,
video-like data, chunked traces, and data used by realtime visualization.

The original motivation for Zarr remains valid: many Palette artifacts are too
large or too spatially/temporally structured to load all at once. Long
frame-indexed time series, masks, crops, video frames, and geometry arrays often
need chunked partial reads. Crimson and other realtime/interactive tools benefit
from this layout because they can request only the frame window, ROI, mask
channel, or trace slice they need.

The problem is not "using Zarr for analysis." The problem is using deep Zarr
trees as the persistence format for high-churn tabular parameter sweeps. Once an
output is naturally a table of bout rows, candidate summaries, or cross-recording
metrics, the benefits of chunked N-dimensional array reads are much smaller, and
the Zarr metadata fanout becomes more visible.

It is less appropriate for large exploratory parameter sweeps. A canary archive
showed this clearly:

```text
2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr
  total zarr.json files: 17,433
  analysis/swim_bout_runs zarr.json files: 10,358
  analysis/swim_bout_runs candidate groups: 28
```

The 10k count is not 10k swim-bout analyses. It comes from Zarr v3 metadata
fanout:

```text
analysis/swim_bout_runs/
  groups: 884
  arrays: 9,474
  JSON metadata size: ~7.2 MB
  directory size: ~156 MB
```

One large swim-bout candidate run contains roughly:

```text
493 zarr.json files
36 groups
457 arrays
```

The immediate cause is repeated storage of multiple signal levels such as
`speed_raw`, `speed_filtered`, `speed_smoothed`, `speed_averaged`, and
`speed_exponential`, with each table column persisted as a separate Zarr array.

This is not data loss or corruption, but it is a storage-layout smell for
exploratory candidate sweeps. It makes validation, listing, metadata
consolidation, and human inspection noisier than necessary.

## Decision

Use Parquet sidecar files inside the `.zarr` directory for exploratory tabular
analysis exports.

The `.zarr` directory remains the archive container. Parquet files are colocated
with the archive so normal filesystem copy operations keep the data together.
They are not Zarr arrays and should not be represented as Zarr nodes.

Recommended initial location:

```text
<recording>_analysis.zarr/
  analysis/
    calibration/
    swim_bout_runs/                  # canonical accepted Zarr-native outputs
    exports/
      zarr.json                      # Zarr group metadata for discoverability attrs
      swim_bout_candidate_bouts.parquet
      swim_bout_candidate_index.parquet
```

Training archives may use the same convention if a colocated tabular export is
useful:

```text
<recording>_training.zarr/
  analysis/
    exports/
      swim_bout_candidate_bouts.parquet
      swim_bout_candidate_index.parquet
```

## Design Principles

- Keep Zarr as the canonical archive for raw video-like arrays, chunked arrays,
  calibrated traces, accepted analysis runs, and realtime visualization.
- Keep Zarr for data that should support partial reads by frame, ROI, channel,
  track, or time window without loading the whole recording.
- Use Parquet for dense tabular candidate outputs, summary metrics, and sweep
  comparisons.
- Use Parquet for exports that should be filtered, grouped, joined, queried
  across recordings, or handed to table-oriented analytics tools.
- Treat Parquet sidecars as derived and regenerable.
- Keep exploratory parameter sweeps out of long-lived deep Zarr run trees unless
  a candidate has been promoted to a canonical accepted run.
- Prefer one combined table per row axis over one file per candidate.
- Store discovery/provenance in `analysis/exports.attrs`, not in separate JSON
  files.

In short:

```text
Zarr    = canonical archive for chunked arrays, masks, traces, and accepted runs
Parquet = derived/export surface for tabular sweeps, summaries, and queries
Registry = metadata, discovery, provenance, and cross-recording selection
```

This design preserves the reason Palette chose Zarr while adding a cleaner
export path for table-shaped analytics.

## What Should Stay Zarr-Native?

Use Zarr groups/arrays for:

- raw or downsampled video frames;
- masks, crops, detections, keypoints, refined masks;
- long frame-indexed traces that need chunked reads;
- eye-angle, track-kinematics, tail, and other per-frame time series that users
  may inspect by time window;
- accepted/canonical derived arrays that realtime or interactive viewers need;
- accepted swim-bout run outputs used by downstream visualization;
- calibration and provenance metadata that other Palette/Crimson readers need
  through standard Zarr paths.

## What Should Become Parquet Sidecars?

Use archive-local Parquet files for:

- exploratory swim-bout parameter sweeps;
- one-row-per-bout candidate event tables;
- one-row-per-candidate summary/index tables;
- compact per-bout metrics used for comparison plots;
- analytic exports such as "all swim bouts matching age/genotype/trial filters";
- cross-candidate analytics that do not need chunked frame-by-frame reads.

Avoid writing every candidate sweep as:

```text
analysis/swim_bout_runs/<candidate>/<signal_level>/<table>/<column_array>
```

unless that candidate is intended to become a durable Zarr-native run.

## Recommended Swim-Bout Sidecar Files

### `swim_bout_candidate_index.parquet`

One row per candidate parameterization.

Suggested columns:

```text
recording_id
zarr_path
candidate_id
candidate_name
method
parameter_hash
parameter_json
source_track_kinematics_run
source_signal_path
source_signal_name
created_at_utc
palette_version
git_commit
n_bouts
bout_rate_per_min
mean_duration_s
median_duration_s
mean_peak_speed_mm_s
mean_path_length_mm
accepted_status              # candidate|accepted|rejected|superseded
accepted_reason
```

### `swim_bout_candidate_bouts.parquet`

One row per detected bout per candidate.

Suggested columns:

```text
recording_id
candidate_id
candidate_name
method
parameter_hash
track_id
subject_id
arena_id
bout_id
start_frame
end_frame
start_time_s
end_time_s
duration_s
core_start_frame
core_end_frame
core_duration_s
peak_frame
peak_time_s
peak_detection_signal_mm_s
peak_physical_speed_mm_s
mean_speed_mm_s
path_length_mm
net_displacement_mm
gap_censored
valid_transition_fraction
source_track_kinematics_run
source_signal_name
```

For a strict first implementation, this single bout table can repeat enough
candidate metadata to be useful without reading the index table. The index table
is still recommended because it avoids repeating large parameter JSON for every
bout row.

## Optional Tables

Only add these when needed:

```text
swim_bout_candidate_points.parquet
```

One row per sampled bout point or one row per bout-frame sample. This can grow
quickly, so it should not be part of the first implementation unless comparison
plots require detailed trajectories.

```text
swim_bout_candidate_signals.parquet
```

One row per frame per candidate signal. This can become very large and is often
better left as Zarr if frame-indexed chunked reads are needed.

## Discoverability Metadata

Create or reuse `analysis/exports/` as a Zarr group and store attrs such as:

```text
analysis/exports.attrs["schema_version"] = 1
analysis/exports.attrs["swim_bout_candidate_bouts_table"] =
  "analysis/exports/swim_bout_candidate_bouts.parquet"
analysis/exports.attrs["swim_bout_candidate_index_table"] =
  "analysis/exports/swim_bout_candidate_index.parquet"
analysis/exports.attrs["swim_bout_candidate_export_status"] = "derived_regenerable"
```

Do not add separate `manifest.json` files for this use case unless there is a
clear need. The goal is to reduce JSON side files, not add more of them.

If more structured manifest data is required later, prefer one of:

- a small Parquet manifest table;
- Zarr group attrs;
- a registry row that points to the archive-local sidecars.

## Reader Contract

Zarr readers should ignore Parquet sidecars unless they explicitly know about
the export.

Parquet readers should:

1. Open the archive root as a filesystem path.
2. Read `analysis/exports.attrs` or use the conventional relative path.
3. Resolve the sidecar path relative to the archive root.
4. Load with `pyarrow`, `polars`, `pandas`, or DuckDB.

Example:

```python
from pathlib import Path

import polars as pl
import zarr

zarr_path = Path("/nvme1/recordings/.../<recording>_analysis.zarr")
root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
exports = root["analysis/exports"]

rel = exports.attrs["swim_bout_candidate_bouts_table"]
bouts = pl.scan_parquet(zarr_path / rel)
```

## Copy And Packaging Rules

Filesystem-level copies should preserve Parquet sidecars:

```bash
rsync -a <recording>_analysis.zarr/ <destination>/<recording>_analysis.zarr/
```

Do not assume every Zarr-specific copy implementation will preserve non-Zarr
files. Any Palette archive-packing or transfer helper should explicitly include
sidecar files under `analysis/exports/`.

## Validation Rules

Validation scripts should distinguish:

- Zarr metadata files: `zarr.json`
- archive-local Parquet sidecars: `*.parquet`
- unexpected JSON files: `*.json` where basename is not `zarr.json`

Strict JSON validation should only parse `zarr.json` files unless it is
explicitly validating a known JSON sidecar. This design intentionally avoids new
JSON sidecars.

Recommended validation checks:

- every path declared in `analysis/exports.attrs` exists;
- each Parquet file has the expected required columns;
- `candidate_id` values in `swim_bout_candidate_bouts.parquet` exist in
  `swim_bout_candidate_index.parquet`;
- source run paths referenced by the Parquet rows exist in the archive;
- row counts and summary values in the index table match the bout table.

## Promotion Workflow

Exploratory candidates should begin in Parquet sidecars.

When a candidate is selected as canonical:

1. mark it as `accepted` in `swim_bout_candidate_index.parquet` or write a new
   accepted index file;
2. optionally materialize that accepted candidate as a normal Zarr run under
   `analysis/swim_bout_runs/<accepted_run>/`;
3. update downstream analysis to consume the accepted Zarr run or the accepted
   Parquet rows, depending on the workflow.

This separates high-churn parameter testing from stable archive structure.

## Open Questions

- Should accepted swim-bout outputs always be materialized back into Zarr, or is
  an accepted Parquet table sufficient for some downstream tools?
- Should the first implementation write one combined bout table per recording,
  or one table per analysis family such as `swim_bout_candidates`?
- Should the registry track archive-local sidecar freshness?
- Should sidecar tables be rebuilt automatically when source track-kinematics
  runs change?

## Initial Implementation TODO

- [ ] Add a writer that exports swim-bout candidate results to
      `analysis/exports/swim_bout_candidate_bouts.parquet`.
- [ ] Add `swim_bout_candidate_index.parquet` with one row per parameter set.
- [ ] Add `analysis/exports` attrs that point to the sidecar paths.
- [ ] Add validation for declared sidecar paths and required columns.
- [ ] Add a cleanup or promotion tool that can remove exploratory Zarr
      swim-bout candidate runs after their Parquet sidecars are written.
- [ ] Keep accepted/canonical swim-bout runs Zarr-native until downstream
      consumers are explicitly updated.
