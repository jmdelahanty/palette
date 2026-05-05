# Cross-Recording Analytics Export Design
<!-- design-meta
status: draft
last_updated: 2026-05-05
-->

Purpose: define a future export strategy for querying Palette metrics across
many recordings with columnar tools such as Polars, DuckDB, Arrow, and Parquet.

This is not a replacement for Palette Zarr archives. It is a design for
regenerable analytics views built from those archives.

## Decision Summary

- Zarr remains the authoritative per-recording archive format.
- Parquet/Arrow exports should be derived, disposable, and reproducible from
  Zarr plus registry metadata.
- Parquet should be organized as an incremental analytics lake: stable table
  directories with many append-only part files, not one monolithic file and not
  one file per question.
- DuckDB should be treated as the query layer over those Parquet datasets, not
  as the primary durable storage format.
- Cross-recording exports should prioritize scalar and tabular metrics first.
- Dense masks, raw probabilities, raw video frames, and large geometry arrays
  should stay in Zarr by default.
- Variable-length geometry such as contours can be exported later as optional
  nested/list columns or separate child tables, but should not be the first
  analytics export target.
- Every exported row must carry enough source identity to map back to the exact
  Zarr archive, run, row, component, track, bout, or frame.
- Every exported row should carry dependency lineage so stale exports can be
  detected when masks, keypoints, or downstream analysis runs are regenerated.

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

## Analytics Lake Model

The cross-recording export area should behave like a small analytics lake. Each
logical table is a directory of Parquet part files. New recordings or new export
runs add parts to the table.

Recommended layout:

```text
analytics_exports/
  palette_analytics/
    v1/
      manifests/
        export_run_id=20260505T120000Z.json
        export_run_id=20260506T093000Z.json
      sessions/
        export_run_id=20260505T120000Z/
          part-000.parquet
      stimulus_steps/
        export_run_id=20260505T120000Z/
          part-000.parquet
      swim_bout_metrics/
        export_run_id=20260505T120000Z/
          part-000.parquet
      bout_classifications/
        export_run_id=20260505T120000Z/
          part-000.parquet
      stimulus_response_per_fish_step/
        export_run_id=20260505T120000Z/
          part-000.parquet
      stimulus_response_windows/
        export_run_id=20260505T120000Z/
          part-000.parquet
```

Do not create one Parquet file per protocol hash by default. Store
`protocol_hash` and, in the future, `protocol_semantic_hash` as columns. Query
engines can filter by those columns efficiently, and keeping the hash as a
column avoids over-partitioning into many tiny files.

If exact protocol filtering becomes a dominant query pattern and file sizes are
large enough, a future implementation may partition by a low-cardinality
protocol family or protocol name. Exact hash partitioning should be introduced
only after measuring query and file-count behavior.

DuckDB can query the lake directly:

```sql
SELECT protocol_hash, median(first_aligned_bout_latency_s)
FROM read_parquet('analytics_exports/palette_analytics/v1/stimulus_response_per_fish_step/**/*.parquet')
WHERE dpf_at_acquisition = 6
  AND protocol_name = 'DefaultScreen'
GROUP BY protocol_hash;
```

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
- `total_path_length_mm`
- `moving_time_s`
- `moving_fraction`
- `mean_speed_mm_s`
- `mean_moving_speed_mm_s`
- `median_moving_speed_mm_s`
- `bout_count`
- `bout_rate_per_min`
- `mean_inter_bout_interval_s`
- `median_inter_bout_interval_s`
- `mean_bout_duration_s`
- `median_bout_duration_s`
- `mean_bout_path_length_mm`
- `median_bout_path_length_mm`
- `mean_bout_peak_speed_mm_s`
- `median_bout_peak_speed_mm_s`
- `mean_vergence_eye_angle_deg`
- `median_vergence_eye_angle_deg`
- coverage/QC columns: `tracking_coverage_fraction`,
  `valid_movement_fraction`, `valid_eye_angle_fraction`,
  `valid_bout_kinematics_fraction`, `valid_tail_posture_fraction`

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
  `displacement_path_ratio`, `mean_speed_mm_s`, `peak_speed_mm_s`
- inter-bout columns: `inter_bout_interval_before_s`,
  `inter_bout_interval_after_s`
- stimulus assignment columns: `stimulus_run`, `step_index`, `stimulus_mode`,
  `step_time_s`
- OMR columns when available: `omr_score`, `omr_label`
- classification columns when available: `bout_classification_run`,
  `predicted_label`, `predicted_label_name`, `classification_confidence`
- eye columns when available: `pre_vergence_eye_angle_deg`,
  `post_vergence_eye_angle_deg`, `mean_vergence_eye_angle_deg`
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
- physical movement columns: `detector_duration_s`,
  `physical_active_duration_s`, `physical_active_path_length_mm`,
  `physical_active_mean_speed_mm_s`, `physical_active_peak_speed_mm_s`,
  `physical_active_boundary_policy`, `physical_active_boundary_constraint`
- optional eye-gaze columns: `pre_vergence_deg`, `post_vergence_deg`,
  `mean_vergence_deg`, `within_bout_vergence_range_deg`,
  `pre_vergence_eye_angle_deg`, `post_vergence_eye_angle_deg`,
  `mean_vergence_eye_angle_deg`
- optional tail columns: `max_abs_tail_angle_rad`, `tail_angle_energy`,
  `tail_posture_valid_fraction`
- validity columns: `valid`, `coverage_fraction`, `failure_reason`

### `bout_classifications`

One row per bout classification output.

Possible columns:

- identity columns: `recording_id`, `zarr_path`, `bout_classification_run`,
  `source_swim_bout_run`, `source_bout_kinematics_run`,
  `source_tail_posture_run`, `track_id`, `bout_index`
- classifier columns: `classifier_name`, `classifier_version`,
  `classifier_source`, `classifier_input_mode`, `predicted_label`,
  `predicted_label_name`, `confidence`, `probabilities_json`
- interop columns: `tool_name`, `tool_version`, `tool_git_commit`,
  `tool_package_path`, `preprocessing_mode`
- validity columns: `valid_window`, `classification_valid`, `failure_reason`

### `stimulus_steps`

One row per canonical stimulus step per recording.

Possible columns:

- identity columns: `recording_id`, `zarr_path`, `stimulus_run`,
  `step_index`, `step_name`
- protocol columns: `protocol_name`, `protocol_hash`,
  `protocol_semantic_hash`, `stimulus_mode`, `stimulus_mode_id`
- timing columns: `start_frame`, `end_frame`, `start_time_s`, `end_time_s`,
  `duration_s`
- moving-grating columns: `direction_deg`, `direction_vector_x`,
  `direction_vector_y`, `spatial_frequency_cycles_per_mm`,
  `speed_mm_s`, `direction_mapping_status`
- concentric-grating columns: `radial_polarity`, `radial_sign`,
  `center_x_mm`, `center_y_mm`, `target_radius_min_mm`,
  `target_radius_max_mm`

### `stimulus_step_summary`

One row per fish, selected source lineage, and stimulus step. This table holds
step-local summaries that are useful even when a protocol-specific OMR response
run is not present.

Possible columns:

- identity columns: `recording_id`, `dataset_id`, `subject_id`, `fish_id`,
  `zarr_path`, `stimulus_run`, `step_index`
- registry/protocol columns: `dish_id`, `cross_id`, `clutch_id`,
  `dpf_at_acquisition`, `line_strain`, `genotype`, `protocol_name`,
  `protocol_hash`, `protocol_semantic_hash`, `stimulus_mode`, `step_name`
- source run columns: `source_track_kinematics_run`,
  `source_swim_bout_run`, `source_bout_kinematics_run`,
  `source_eye_angle_run`, `source_bout_classification_run`
- movement columns: `path_length_mm`, `moving_time_s`, `moving_fraction`,
  `mean_speed_mm_s`, `mean_moving_speed_mm_s`, `median_moving_speed_mm_s`
- bout columns: `bout_count`, `bout_rate_per_min`,
  `mean_bout_duration_s`, `median_bout_duration_s`,
  `mean_bout_path_length_mm`, `median_bout_path_length_mm`,
  `mean_bout_net_displacement_mm`, `median_bout_net_displacement_mm`,
  `mean_bout_peak_speed_mm_s`, `median_bout_peak_speed_mm_s`,
  `mean_inter_bout_interval_s`, `median_inter_bout_interval_s`
- heading columns: `mean_heading_change_deg`,
  `median_heading_change_deg`, `mean_abs_heading_change_deg`,
  `median_abs_heading_change_deg`, `mean_within_bout_heading_range_deg`,
  `median_within_bout_heading_range_deg`
- eye columns: `mean_vergence_eye_angle_deg`,
  `median_vergence_eye_angle_deg`, `mean_vergence_gaze_deg`,
  `median_vergence_gaze_deg`
- OMR columns when available: `omr_path_index`, `bout_fraction_correct`,
  `first_aligned_bout_latency_s`, `first_classified_bout_latency_s`,
  `first_opposing_bout_latency_s`, `n_classified_bouts`,
  `n_aligned_bouts`, `n_opposing_bouts`

### `stimulus_response_per_fish_step`

One row per fish, response run, and stimulus step. This is the first table to
use for cohort-level OMR questions such as "latency to follow the grating."

Possible columns:

- identity columns: `recording_id`, `dataset_id`, `subject_id`, `fish_id`,
  `zarr_path`, `stimulus_response_run`, `stimulus_run`, `step_index`
- registry columns: `dish_id`, `cross_id`, `clutch_id`,
  `dpf_at_acquisition`, `line_strain`, `genotype`, `recording_date`
- protocol columns: `protocol_name`, `protocol_hash`,
  `protocol_semantic_hash`, `stimulus_mode`, `step_name`
- source run columns: `source_track_kinematics_run`,
  `source_swim_bout_run`, `source_bout_kinematics_run`,
  `source_eye_angle_run`, `source_bout_classification_run`
- OMR summary columns: `omr_path_index`, `omr_displacement_index`,
  `bout_fraction_correct`, `bout_fraction_correct_weighted_by_path`,
  `time_fraction_correct`, `n_classified_bouts`, `n_aligned_bouts`,
  `n_opposing_bouts`
- latency columns: `first_aligned_bout_latency_s`,
  `first_classified_bout_latency_s`, `first_opposing_bout_latency_s`,
  `first_aligned_bout_id`, `first_aligned_bout_score`

Missing latency values should be stored as null in Parquet. NaN may be
acceptable inside Arrow arrays, but null is easier to query consistently across
DuckDB, Polars, pandas, and R.

### `stimulus_response_windows`

One row per fish, stimulus step, response run, and time window.

Possible columns:

- identity columns: `recording_id`, `dataset_id`, `subject_id`, `zarr_path`,
  `stimulus_response_run`, `stimulus_run`, `step_index`, `window_index`
- window columns: `window_start_s`, `window_end_s`, `window_start_frame`,
  `window_end_frame`, `window_duration_s`
- response columns: `omr_path_index`, `bout_fraction_correct`,
  `time_fraction_correct`, `n_classified_bouts`, `n_aligned_bouts`,
  `n_opposing_bouts`, `mean_speed_mm_s`, `path_length_mm`

### `metric_histogram_counts`

Optional dashboard-oriented table. This table should not replace per-bout facts;
it should only cache a known binning policy for faster dashboards.

Possible columns:

- identity columns: `dataset_id`, `recording_id`, `subject_id`, `zarr_path`,
  `protocol_hash`, `stimulus_run`, `step_index`
- histogram columns: `metric_name`, `bin_policy`, `bin_left`, `bin_right`,
  `count`
- provenance columns: `source_table`, `source_lineage_hash`,
  `export_run_id`

Prefer rebuilding merged histograms from `swim_bout_metrics` when changing bins
or comparing new cohorts.

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
  palette_analytics/
    v1/
      manifests/
      recording_summary/
        protocol_name=DefaultScreen/
          recording_date=2026-01-28/
            part-*.parquet
      refined_subject_mask_component_metrics/
        recording_date=2026-01-28/
          part-*.parquet
      subject_shape_metrics/
        recording_date=2026-01-28/
          part-*.parquet
      swim_bout_metrics/
        protocol_name=Feeding/
          recording_date=2026-01-28/
            part-*.parquet
```

Recommended partition columns:

- `recording_date`
- optionally `protocol_name`
- optionally `zarr_purpose`
- optionally `stimulus_mode` for stimulus-response tables

Avoid over-partitioning by high-cardinality fields such as `recording_id`,
`run_id`, `track_id`, exact `protocol_hash`, or exact `subject_id` unless a
specific query pattern and measured table size justify it.

## Required Source Identity Columns

Every exported table should include enough identity to trace each row back to
Zarr.

Recommended common columns:

- `export_run_id`
- `export_created_at_utc`
- `export_schema_version`
- `recording_id`
- `dataset_id`
- `session_uuid`
- `subject_id`
- `fish_id`
- `dish_id`
- `cross_id`
- `clutch_id`
- `dpf_at_acquisition`
- `line_strain`
- `genotype`
- `protocol_name`
- `protocol_hash`
- `protocol_semantic_hash`
- `zarr_path`
- `zarr_mtime_ns` or source archive version marker when available
- `stage_family`
- `run_id`
- `schema_id`
- `schema_version`
- `method`
- `method_version`
- `source_refs_json`
- `source_lineage_hash`
- `is_latest`
- `supersedes_export_run_id`

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

bouts = pl.scan_parquet("analytics_exports/palette_analytics/v1/swim_bout_metrics/**/*.parquet")
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
FROM read_parquet('analytics_exports/palette_analytics/v1/swim_bout_metrics/**/*.parquet')
WHERE valid
GROUP BY experiment_type, speed_level;
```

Stimulus-response cohort query shape:

```sql
SELECT
  cross_id,
  count(*) AS n_fish_steps,
  median(first_aligned_bout_latency_s) AS median_latency_s,
  avg(omr_path_index) AS mean_omr_path_index
FROM read_parquet('analytics_exports/palette_analytics/v1/stimulus_response_per_fish_step/**/*.parquet')
WHERE dpf_at_acquisition = 6
  AND protocol_name = 'DefaultScreen'
  AND stimulus_mode = 'MOVING_GRATING'
  AND first_aligned_bout_latency_s IS NOT NULL
GROUP BY cross_id
ORDER BY median_latency_s;
```

Same-fish-across-protocol query shape:

```sql
WITH multi_protocol_subjects AS (
  SELECT subject_id
  FROM read_parquet('analytics_exports/palette_analytics/v1/stimulus_response_per_fish_step/**/*.parquet')
  WHERE subject_id IS NOT NULL
  GROUP BY subject_id
  HAVING count(DISTINCT protocol_hash) >= 2
)
SELECT
  r.subject_id,
  r.protocol_hash,
  r.protocol_name,
  median(first_aligned_bout_latency_s) AS median_latency_s
FROM read_parquet('analytics_exports/palette_analytics/v1/stimulus_response_per_fish_step/**/*.parquet') r
JOIN multi_protocol_subjects m USING (subject_id)
WHERE r.dpf_at_acquisition = 6
  AND r.stimulus_mode = 'MOVING_GRATING'
GROUP BY r.subject_id, r.protocol_hash, r.protocol_name;
```

The second query only has biological meaning if `subject_id` really identifies
the same fish across sessions. If identity is only known at dish, cross, or
clutch level, the query should use those columns instead.

## Export Manifest

Each export should write a manifest:

```text
analytics_exports/palette_analytics/v1/manifests/export_run_id=<export_run_id>.json
```

Recommended fields:

- `export_run_id`
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
- `source_lineage_hashes`
- `supersedes_export_run_ids`
- `export_parameters`
- `row_counts_by_table`

The manifest is what makes a Parquet dataset reproducible and auditable.

## Data Versioning And Staleness Policy

Parquet exports are versioned derived views. The source of truth remains the
Zarr archive plus registry. This matters because upstream corrections can change
downstream biology:

- refined masks can be fixed;
- keypoints can be corrected;
- subject shape, tail posture, eye angles, movement, swim bouts, and stimulus
  response runs can be regenerated;
- registry metadata such as DPF, cross, clutch, or protocol identity can be
  corrected.

Every exported row should therefore include a `source_lineage_hash`. The hash
should be computed from the exact upstream run IDs and schema/method versions
used to produce that row. For a stimulus-response row, this could include:

```text
stimulus_run_id
track_kinematics_run_id
swim_bout_run_id
bout_kinematics_run_id
eye_angle_run_id
bout_classification_run_id
stimulus_response_run_id
registry_context_revision
```

For a bout-classification row, it could include:

```text
subject_shape_run_id
tail_posture_run_id
swim_bout_run_id
bout_kinematics_run_id
classifier_run_id
classifier_version
classifier_input_mode
```

Parquet exports are stale when:

- a source Zarr is edited
- a source analysis run is regenerated
- a refined mask row is edited
- registry run-selection policy changes
- export code/schema changes

Do not patch stale Parquet rows by hand. Re-run the export with a new
`export_run_id`.

The default update policy should be append-only:

1. Export new rows with a new `export_run_id`.
2. Preserve old rows for historical comparison.
3. Mark or infer old rows as superseded by comparing `dataset_id`, row identity,
   and `source_lineage_hash`.
4. Query "latest" rows through a DuckDB view or manifest-selected table set.

This allows explicit before/after checks such as "did fixing keypoints change
latency to follow the grating?" without losing the old result.

Example latest-only DuckDB shape:

```sql
SELECT *
FROM read_parquet('analytics_exports/palette_analytics/v1/stimulus_response_per_fish_step/**/*.parquet')
QUALIFY row_number() OVER (
  PARTITION BY dataset_id, subject_id, stimulus_response_run, step_index
  ORDER BY export_created_at_utc DESC, export_run_id DESC
) = 1;
```

The partition key in a real view should be table-specific. For bout tables it
should include bout identity; for per-step tables it should include fish and
step identity; for windowed tables it should include fish, step, and window.

## DVC And External Data Versioning

DVC can be useful later, but it solves a different layer of the problem.

```text
Parquet + manifests = queryable exported analytic facts and their row lineage
DVC                 = versioned large-file/directory snapshots outside Git
DuckDB              = query engine over the exported Parquet files
```

DVC should not replace Parquet, DuckDB, Zarr, or Palette lineage metadata. It
can wrap selected export snapshots so a manuscript, notebook, or model training
run can pin the exact exported files used for an analysis.

Good DVC candidates:

- `analytics_exports/palette_analytics/v1/manifests/`
- selected Parquet export directories used for a publication or milestone
- frozen training datasets
- model checkpoints

Poor DVC candidates:

- hot mutable analysis Zarrs during active review/refinement
- every transient canary export
- row-level "latest" semantics inside Parquet tables

If DVC is added, keep the responsibilities clean: Git tracks code and DVC
metadata, DVC tracks large exported files, Palette manifests track source
lineage, and DuckDB queries Parquet.

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
