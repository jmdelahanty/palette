# Cross-Recording Analytics Export Design
<!-- design-meta
status: draft
last_updated: 2026-04-29
-->

Purpose: define a future export strategy for querying Palette metrics across
many recordings with columnar tools such as Polars, DuckDB, Arrow, and Parquet.

This is not a replacement for Palette Zarr archives. It is a design for
regenerable analytics views built from those archives.

## Decision Summary

- Zarr remains the authoritative per-recording archive format.
- Parquet/Arrow exports should be derived, disposable, and reproducible from
  Zarr plus registry metadata.
- Cross-recording exports should prioritize scalar and tabular metrics first.
- Dense masks, raw probabilities, raw video frames, and large geometry arrays
  should stay in Zarr by default.
- Variable-length geometry such as contours can be exported later as optional
  nested/list columns or separate child tables, but should not be the first
  analytics export target.
- Every exported row must carry enough source identity to map back to the exact
  Zarr archive, run, row, component, track, bout, or frame.

## Why Use Parquet/Arrow Later?

Palette Zarrs are good for:

- self-contained recording archives
- chunked numeric arrays
- row-local visualization
- masks, probabilities, images, contours, and analysis arrays
- keeping derived products close to their source data

Parquet/Arrow becomes useful when the question is cross-recording and
table-oriented:

- "Compare bout duration distributions across all feeding recordings."
- "Query all frames where body-mask QC failed."
- "Join eye vergence metrics with stimulus condition and bout class."
- "Aggregate subject-shape body length by experiment, date, arena, and animal."
- "Build a modeling table from selected metrics across thousands of runs."

Polars and DuckDB are especially useful for this kind of work because they can
scan partitioned Parquet datasets without loading every column or every row
into memory.

## Non-Goals

- Do not replace Zarr as the primary archive.
- Do not edit Parquet outputs and treat them as source-of-truth corrections.
- Do not export dense masks or probability volumes by default.
- Do not require every analysis run to immediately support Parquet export.
- Do not make Parquet a dependency of realtime Crimson viewing.

## Authority Model

The authority model should be:

```text
Palette Zarr + registry metadata -> deterministic export command -> Parquet dataset
```

Parquet should be safe to delete and regenerate.

If a refined mask, analysis run, or registry record changes, dependent Parquet
exports are stale. They should be rebuilt rather than patched manually.

## Export Dataset Types

Use separate Parquet datasets for different row axes instead of one giant
wide table.

Recommended first dataset families:

### `recording_summary`

One row per recording/Zarr archive.

Possible columns:

- `recording_id`
- `zarr_path`
- `experiment_type`
- `arena_id`
- `recorded_at_utc`
- `zarr_purpose`
- `registry_recording_id`
- `duration_s`
- `fps`
- latest/selected run IDs for major analysis stages

### `refined_subject_mask_component_metrics`

One row per recording, refined run, source row, and component.

Possible columns:

- identity columns: `recording_id`, `zarr_path`, `refined_run`,
  `component`, `row_index`, `frame_index`, `detection_index`
- source labels: `label_schema_id`, `mask_labels_hash`
- metric columns: `mask_present`, `area_px`, `component_count`,
  `largest_component_fraction`, `hole_count`, `hole_area_fraction`,
  `solidity`, `bbox_width_px`, `bbox_height_px`
- QC columns: `requires_review`, `severe_qc_failure`, `reason`
- review/provenance columns: `component_review_state`,
  `refined_subject_mask_review_state`, `created_at_utc`,
  `source_subject_mask_run`

### `subject_shape_metrics`

One row per subject-shape row/component or one row per subject-shape row with
wide columns, depending on the metric.

Possible columns:

- identity columns: `recording_id`, `zarr_path`, `subject_shape_run`,
  `source_refined_subject_masks_run`, `row_index`, `frame_index`
- body metrics: `centerline_valid`, `body_arclength_px`,
  `tail_segment_arclength_px`, `tail_base_valid`
- swim metrics: `caudal_contour_valid`,
  `caudal_contour_projection_px`
- body-frame metrics: `body_frame_valid`, `heading_deg`
- failure reason columns for each major output

### `eye_angle_timeseries`

One row per eye-angle row.

Possible columns:

- identity columns: `recording_id`, `zarr_path`, `eye_angle_run`,
  `source_refined_subject_masks_run`, `source_subject_shape_run`,
  `row_index`, `frame_index`, `time_s`
- eye angle columns: `left_gaze_deg`, `right_gaze_deg`,
  `mean_eye_vergence_gaze_deg`, `vergence_gaze_deg`
- validity columns: `valid`, `left_valid`, `right_valid`, `failure_reason`
- body-frame source metadata: `body_frame_estimator`,
  `body_frame_schema_version`

### `track_kinematics_timeseries`

One row per track sample.

Possible columns:

- identity columns: `recording_id`, `zarr_path`, `track_kinematics_run`,
  `track_id`, `sample_index`, `frame_index`, `time_s`
- position columns: `x_px`, `y_px`, `x_mm`, `y_mm`
- movement columns: `speed_raw_mm_s`, `speed_filtered_mm_s`,
  `speed_smoothed_mm_s`, `acceleration_mm_s2`, `angular_velocity_deg_s`
- validity columns: `valid`, `gap`, `failure_reason`

### `swim_bout_metrics`

One row per swim bout candidate.

Possible columns:

- identity columns: `recording_id`, `zarr_path`, `swim_bout_run`,
  `track_kinematics_run`, `track_id`, `bout_index`
- boundary columns: `start_frame`, `end_frame`, `start_time_s`,
  `end_time_s`, `duration_s`
- segmentation columns: `speed_level`, `method`, `threshold`, `min_gap_s`,
  `boundary_mode`, `peak_prominence`, `peak_distance_frames`
- movement columns: `path_length_mm`, `net_displacement_mm`,
  `mean_speed_mm_s`, `peak_speed_mm_s`
- validity columns: `valid`, `failure_reason`

### `bout_kinematics_metrics`

One row per bout-kinematics measurement.

Possible columns:

- identity columns: `recording_id`, `zarr_path`, `bout_kinematics_run`,
  `source_swim_bout_run`, `source_track_kinematics_run`, `track_id`,
  `bout_index`
- heading columns: `pre_heading_deg`, `post_heading_deg`,
  `net_heading_change_deg`, `within_bout_heading_range_deg`,
  `within_bout_heading_peak_to_peak_deg`
- position columns: `pre_position_x_mm`, `pre_position_y_mm`,
  `post_position_x_mm`, `post_position_y_mm`,
  `net_displacement_mm`
- optional eye-gaze columns: `pre_vergence_deg`, `post_vergence_deg`,
  `within_bout_vergence_range_deg`
- validity columns: `valid`, `coverage_fraction`, `failure_reason`

## Long vs Wide Tables

Use long tables when the semantic axis is naturally repeated:

- component metrics: one row per `(recording, run, source_row, component)`
- track samples: one row per `(recording, run, track_id, sample)`
- swim bouts: one row per `(recording, run, track_id, bout)`

Use wide columns when the values are naturally measured together:

- `left_gaze_deg`, `right_gaze_deg`, and `vergence_deg`
- `x_mm`, `y_mm`, `speed_mm_s`, and `acceleration_mm_s2`
- `start_time_s`, `end_time_s`, and `duration_s`

Avoid tables that are both extremely wide and extremely sparse. Prefer
multiple dataset families with clear row axes.

## Partitioning Strategy

A practical filesystem layout could be:

```text
analytics_exports/
  export_id=<export_id>/
    manifest.json
    recording_summary/
      part-*.parquet
    refined_subject_mask_component_metrics/
      experiment_type=feeding/
        recording_date=2026-01-28/
          part-*.parquet
    subject_shape_metrics/
      experiment_type=feeding/
        recording_date=2026-01-28/
          part-*.parquet
    swim_bout_metrics/
      experiment_type=feeding/
        recording_date=2026-01-28/
          part-*.parquet
```

Recommended partition columns:

- `experiment_type`
- `recording_date`
- optionally `zarr_purpose`

Avoid over-partitioning by high-cardinality fields such as `recording_id`,
`run_id`, or `track_id` unless a specific query pattern needs it.

## Required Source Identity Columns

Every exported table should include enough identity to trace each row back to
Zarr.

Recommended common columns:

- `export_id`
- `export_created_at_utc`
- `recording_id`
- `zarr_path`
- `zarr_mtime_ns` or source archive version marker when available
- `stage_family`
- `run_id`
- `schema_id`
- `schema_version`
- `method`
- `method_version`
- `source_refs_json`

Row-axis specific columns:

- row-aligned arrays: `row_index`, `frame_index`, `detection_index`,
  `source_refined_row_id`
- track arrays: `track_id`, `sample_index`, `frame_index`, `time_s`
- bout arrays: `track_id`, `bout_index`, `start_frame`, `end_frame`
- component arrays: `component`

## Variable-Length Geometry Exports

Contours, centerline samples, splines, and tail-width profiles are not ideal
first-pass Parquet metrics because they are variable-length arrays.

Keep these in Zarr by default:

- dense masks
- raw probabilities
- full contours
- raw centerline sample arrays
- spline control points
- full per-frame visualization images

If exporting variable-length geometry becomes useful, prefer separate child
tables rather than embedding everything in the main metrics table.

Example contour child table:

```text
component_contour_points/
  recording_id
  zarr_path
  refined_run
  component
  row_index
  point_index
  x
  y
```

This is simple and queryable but can be large.

Alternative Arrow-native nested list column:

```text
component_contours/
  recording_id
  refined_run
  component
  row_index
  contour_xy: list<struct<x: float32, y: float32>>
```

This preserves row-local grouping but is less convenient for some SQL-style
queries. Use it only after testing with Polars and DuckDB.

## Example Query Goals

Polars example shape:

```python
import polars as pl

bouts = pl.scan_parquet("analytics_exports/export_id=.../swim_bout_metrics/**/*.parquet")
summary = (
    bouts
    .filter(pl.col("experiment_type") == "feeding")
    .group_by(["recording_id", "speed_level"])
    .agg([
        pl.len().alias("bout_count"),
        pl.col("duration_s").median().alias("median_duration_s"),
        pl.col("path_length_mm").median().alias("median_path_length_mm"),
    ])
)
```

DuckDB example shape:

```sql
SELECT
  experiment_type,
  speed_level,
  count(*) AS bout_count,
  median(duration_s) AS median_duration_s
FROM read_parquet('analytics_exports/export_id=*/swim_bout_metrics/**/*.parquet')
WHERE valid
GROUP BY experiment_type, speed_level;
```

## Export Manifest

Each export should write a manifest:

```text
analytics_exports/export_id=<export_id>/manifest.json
```

Recommended fields:

- `export_id`
- `created_at_utc`
- `palette_git_commit`
- `palette_git_dirty`
- `registry_path`
- `source_recording_count`
- `source_zarrs`
- `tables_written`
- `table_schema_versions`
- `selection_query`
- `source_run_selection_policy`
- `export_parameters`

The manifest is what makes a Parquet dataset reproducible and auditable.

## Staleness Policy

Parquet exports are stale when:

- a source Zarr is edited
- a source analysis run is regenerated
- a refined mask row is edited
- registry run-selection policy changes
- export code/schema changes

Do not patch stale Parquet rows by hand. Re-run the export with a new
`export_id`.

## Implementation Phases

### Phase 1. Scalar Metrics Inventory

- Inventory current analysis run families and scalar arrays.
- Define one table schema per row axis.
- Start with `recording_summary`, `refined_subject_mask_component_metrics`,
  `subject_shape_metrics`, and `swim_bout_metrics`.

### Phase 2. Export Manifest And Table Writers

- Add a CLI such as `fisheye.utils.export_cross_recording_analytics`.
- Accept explicit Zarr paths, recording roots, registry queries, or manifest
  inputs.
- Write partitioned Parquet datasets and a manifest.
- Keep dependencies optional until the workflow is stable.

### Phase 3. Polars/DuckDB Smoke Queries

- Add example Polars and DuckDB scripts/notebooks.
- Validate query performance on a small recording set.
- Verify schema evolution behavior when columns are missing from older runs.

### Phase 4. Derived Geometry Child Tables

- Add optional child tables for contours, centerlines, or splines only when
  there is a concrete cross-recording query that needs them.
- Prefer scalar summaries over full geometry export where possible.

### Phase 5. Registry Integration

- Register analytics export artifacts in the registry.
- Track source run selections and export manifests.
- Add stale/export superseded states when source archives or run selections
  change.

## Open Questions

- Should the first implementation use Polars directly, PyArrow directly, or
  write through pandas/pyarrow for compatibility?
- Should exports be recording-root driven or registry-query driven first?
- Which table schemas should be versioned independently?
- How should source Zarr modification state be represented robustly:
  filesystem mtime, registry run metadata, or explicit source revision attrs?
- Should large exports live under `/nvme1/analytics_exports`, the registry
  artifact root, or next to training datasets?
