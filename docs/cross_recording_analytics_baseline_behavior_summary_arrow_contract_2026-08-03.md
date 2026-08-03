# Baseline-behavior-summary exact Arrow contract — 2026-08-03

## Decision

`baseline_behavior_summary` uses exact physical Arrow schema v1 for new
immutable cross-recording analytics exports. Its 95 fields have a closed
order, Arrow type, and nullability contract. The logical table remains
`palette.analytics.table.baseline_behavior_summary` v1 inside analytics export
v2; the independently versioned physical schema is
`palette.analytics_export.arrow_table.baseline_behavior_summary` v1.

This freezes only the Parquet representation. It does not change or promote
the selected chaser-distance authority, producer defaults, recording-local
Zarr contracts, selectors, registries, or storage profiles.

## Producer boundary

The row vocabulary is the closed union of four maintained surfaces:

1. five shared export identity fields;
2. fixed chaser-distance and source-lineage fields;
3. the fixed output of `build_summary_metrics()`; and
4. three optional virtual-collection fields.

The source epoch row is not merged into the export. Exactly eight named source
summary metrics are projected:

- median and mean bout duration;
- median and mean bout path length;
- median and mean absolute bout net-heading change; and
- median and mean inter-bout interval.

Additional fields in the source structured table cannot silently become new
Parquet columns. This is important because the source component may evolve
independently of the portable analytics table.

The `fps` export field is nullable in physical schema v1. The current metric
computation falls back from a missing chaser-run FPS to verified track FPS,
while the exported lineage field is populated only from the chaser-run
attribute. Exact v1 records that present behavior rather than silently
repairing or reinterpreting source authority.

## Ordered physical schema

The order below is authoritative. `int32`, `int64`, and `float64` are Arrow
types, not inferred Python or NumPy representations.

| # | Field | Arrow type | Nullable |
|---:|---|---|:---:|
| 1 | `export_schema_version` | `int32` | no |
| 2 | `table_name` | `string` | no |
| 3 | `recording_id` | `string` | no |
| 4 | `zarr_path` | `string` | no |
| 5 | `source_lineage_hash` | `string` | no |
| 6 | `chaser_distance_run` | `string` | no |
| 7 | `chaser_distance_path` | `string` | no |
| 8 | `chaser_distance_schema_id` | `string` | yes |
| 9 | `chaser_distance_schema_version` | `int64` | yes |
| 10 | `chaser_distance_method` | `string` | yes |
| 11 | `chaser_distance_method_version` | `string` | yes |
| 12 | `source_detection_path` | `string` | yes |
| 13 | `source_detection_kind` | `string` | yes |
| 14 | `source_stimulus_run` | `string` | yes |
| 15 | `source_stimulus_path` | `string` | yes |
| 16 | `source_stimulus_epoch_run` | `string` | yes |
| 17 | `source_stimulus_epoch_path` | `string` | yes |
| 18 | `source_refs_json` | `string` | no |
| 19 | `coordinate_frame` | `string` | no |
| 20 | `coordinate_origin` | `string` | no |
| 21 | `fps` | `float64` | yes |
| 22 | `total_frames` | `int64` | yes |
| 23 | `pixels_per_mm_projector` | `float64` | no |
| 24 | `source_chaser_distance_run` | `string` | no |
| 25 | `source_chaser_distance_path` | `string` | no |
| 26 | `source_epoch_behavior_component` | `string` | no |
| 27 | `source_epoch_behavior_path` | `string` | no |
| 28 | `source_track_kinematics_run` | `string` | no |
| 29 | `source_track_kinematics_scope` | `string` | no |
| 30 | `source_track_kinematics_path` | `string` | no |
| 31 | `source_track_kinematics_track_path` | `string` | no |
| 32 | `source_speed_level` | `string` | no |
| 33 | `source_swim_bout_run` | `string` | yes |
| 34 | `source_swim_bout_path` | `string` | yes |
| 35 | `track_id` | `int64` | no |
| 36 | `arena_center_x_px` | `float64` | no |
| 37 | `arena_center_y_px` | `float64` | no |
| 38 | `arena_radius_px` | `float64` | no |
| 39 | `baseline_method` | `string` | no |
| 40 | `baseline_method_version` | `string` | no |
| 41 | `baseline_window_id` | `int64` | no |
| 42 | `baseline_window_label` | `string` | no |
| 43 | `start_frame` | `int64` | no |
| 44 | `end_frame` | `int64` | no |
| 45 | `start_time_s` | `float64` | no |
| 46 | `end_time_s` | `float64` | no |
| 47 | `duration_s` | `float64` | no |
| 48 | `total_frame_count` | `int64` | no |
| 49 | `valid_frame_count` | `int64` | no |
| 50 | `missing_frame_count` | `int64` | no |
| 51 | `tracking_dropout_fraction` | `float64` | yes |
| 52 | `speed_sample_count` | `int64` | no |
| 53 | `mean_speed_mm_s` | `float64` | yes |
| 54 | `median_speed_mm_s` | `float64` | yes |
| 55 | `p95_speed_mm_s` | `float64` | yes |
| 56 | `max_speed_mm_s` | `float64` | yes |
| 57 | `total_path_mm` | `float64` | yes |
| 58 | `bout_count` | `int64` | no |
| 59 | `bout_rate_per_min` | `float64` | yes |
| 60 | `arena_radius_mm` | `float64` | no |
| 61 | `wall_band_mm` | `float64` | no |
| 62 | `expected_uniform_wall_fraction` | `float64` | no |
| 63 | `experimental_area_geometry_type` | `string` | no |
| 64 | `boundary_distance_method` | `string` | no |
| 65 | `wall_fraction_denominator` | `string` | no |
| 66 | `wall_frame_count` | `int64` | no |
| 67 | `wall_fraction` | `float64` | yes |
| 68 | `mean_distance_from_arena_center_mm` | `float64` | yes |
| 69 | `median_distance_from_arena_center_mm` | `float64` | yes |
| 70 | `p95_distance_from_arena_center_mm` | `float64` | yes |
| 71 | `mean_distance_to_arena_boundary_mm` | `float64` | yes |
| 72 | `median_distance_to_arena_boundary_mm` | `float64` | yes |
| 73 | `p95_distance_to_arena_boundary_mm` | `float64` | yes |
| 74 | `mean_center_distance_norm` | `float64` | yes |
| 75 | `median_center_distance_norm` | `float64` | yes |
| 76 | `x_axis_direction` | `string` | no |
| 77 | `y_axis_direction` | `string` | no |
| 78 | `spatial_grid_size` | `int64` | no |
| 79 | `spatial_valid_sample_count` | `int64` | no |
| 80 | `spatial_visited_cell_count` | `int64` | no |
| 81 | `spatial_entropy_normalized` | `float64` | yes |
| 82 | `spatial_max_cell_fraction` | `float64` | yes |
| 83 | `quadrant_entropy_normalized` | `float64` | yes |
| 84 | `quadrant_max_fraction` | `float64` | yes |
| 85 | `median_bout_duration_s` | `float64` | yes |
| 86 | `mean_bout_duration_s` | `float64` | yes |
| 87 | `median_bout_path_length_mm` | `float64` | yes |
| 88 | `mean_bout_path_length_mm` | `float64` | yes |
| 89 | `median_abs_bout_net_heading_change_deg` | `float64` | yes |
| 90 | `mean_abs_bout_net_heading_change_deg` | `float64` | yes |
| 91 | `median_inter_bout_interval_s` | `float64` | yes |
| 92 | `mean_inter_bout_interval_s` | `float64` | yes |
| 93 | `collection_id` | `string` | yes |
| 94 | `collection_manifest_sha256` | `string` | yes |
| 95 | `collection_manifest_path` | `string` | yes |

## Publication and consumer behavior

- The writer normalizes rows into the declared order and rejects unexpected
  fields and null/missing non-nullable fields.
- Zero-row exports publish no Parquet placeholder part, but retain the exact
  table declaration and zero count in the immutable manifest.
- Every part footer binds exact mode, schema ID, schema version, and the
  canonical declaration digest.
- Staged and selected-publication validation reconstruct the installed
  contract. Reordered, missing, additional, wrong-type, wrong-nullability, or
  changed-footer schemas fail even if file and manifest digests are recomputed.
- Baseline-strategy analytics consumes this selected exact table through the
  same immutable manifest validation boundary. A duplicated `export_run_id`
  column or another undeclared field is rejected instead of becoming an
  inferred compatibility column.

## Source-authority quarantine

The current general export path deliberately preflights canonical
chaser-distance data and reports its derived analytics surfaces as unsealed.
This checkpoint does not bypass that policy in production. The representation
test replaces `_latest_run` with a raw fixture-group test double and
`load_track_kinematics_track` with a `SimpleNamespace` test double. Those seams
emulate the inputs that would exist after future verification; the test does
not itself verify either source. It exercises the real baseline loader, metrics
builder, exact writer, staged publication, and selected reader as
representation-path evidence only. It provides zero source-authority
validation, selection, or promotion evidence.

## Implementation checklist

- [x] Freeze all 95 fields in producer order.
- [x] Freeze Arrow types and nullability independently of observed values.
- [x] Bind the exact eight-key source-summary projection.
- [x] Keep `fps` nullable and record the existing producer discrepancy.
- [x] Add `baseline_behavior_summary` to the digest-bound exact table envelope.
- [x] Cover exact writer order, type, nullability, and footer digest.
- [x] Cover zero-row publication without placeholder parts.
- [x] Reject unexpected and missing required writer fields.
- [x] Reject rehashed envelope field order/type/nullability/membership tampering.
- [x] Reject reordered, missing, additional, wrong-type, wrong-nullability, and
  footer-metadata tampering after recomputing the part inventory.
- [x] Exercise the real representation path with selection/typed-loader test
  doubles, without claiming source-authority validation or promotion evidence.
- [x] Make the baseline-strategy fixture consume the exact physical schema.
- [x] Run the complete focused test set outside the sandbox: 59 tests passed.
- [x] Complete independent read-only review before commit: ACCEPT after the two
  required documentation/diff-hygiene cleanups were applied.

No storage-performance benchmark is required for this representation-only
change: row values, part partitioning, compression, and publication shape are
unchanged. Future source-authority promotion and cross-language consumption
remain separate checkpoints.

An additional atomic-publication/exporter regression run passed 58 of 62
tests. The four failures are the existing strict bout-kinematics legacy-layout
gates (`hierarchical` or unmanifested compact fixtures require explicit legacy
compatibility); the same failures are recorded at the clean coordination line
and occur before this baseline Arrow schema is exercised.
