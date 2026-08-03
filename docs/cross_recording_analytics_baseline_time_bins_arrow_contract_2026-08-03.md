# Baseline-time-bins exact Arrow contract — 2026-08-03

## Decision

`baseline_behavior_time_bins` uses exact physical Arrow schema v1 for new
immutable cross-recording analytics exports. Its 77 fields have a closed
order, Arrow type, and nullability contract. The logical table remains
`palette.analytics.table.baseline_behavior_time_bins` v1 inside analytics
export v2; the independently versioned physical schema is
`palette.analytics_export.arrow_table.baseline_behavior_time_bins` v1.

This freezes only the Parquet representation. It does not change or promote
chaser-distance authority, source selection, producer defaults, recording-local
Zarr contracts, registries, or storage profiles.

## Closed producer vocabulary

The executable producer assembles each row from four closed surfaces:

1. five shared export-identity fields;
2. the fixed chaser/source/track/arena lineage block;
3. the fixed dictionary returned by `build_time_bin_metrics()`; and
4. three optional virtual-collection fields.

The source `per_epoch_fish` row supplies only named window bounds before the
builder runs. It is never merged into the output row. Tests place a
`future_source_metric` column in that source table and prove that it does not
appear in the Parquet schema or values. The builder test also freezes its exact
38-key output order.

`fps` is nullable in physical v1 for the same reason as the companion summary:
computation may fall back to verified track FPS, while the portable lineage
field currently reads only the chaser-run attribute. This checkpoint records
that discrepancy rather than silently repairing source authority.

## Ordered physical schema

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
| 43 | `time_bin_index` | `int64` | no |
| 44 | `relative_start_s` | `float64` | no |
| 45 | `relative_end_s` | `float64` | no |
| 46 | `time_bin_duration_s` | `float64` | no |
| 47 | `source_start_frame` | `int64` | no |
| 48 | `source_end_frame` | `int64` | no |
| 49 | `expected_frame_count` | `int64` | no |
| 50 | `valid_position_count` | `int64` | no |
| 51 | `valid_position_fraction` | `float64` | yes |
| 52 | `speed_sample_count` | `int64` | no |
| 53 | `mean_speed_mm_s` | `float64` | yes |
| 54 | `median_speed_mm_s` | `float64` | yes |
| 55 | `p95_speed_mm_s` | `float64` | yes |
| 56 | `distance_travelled_mm` | `float64` | yes |
| 57 | `mean_center_distance_mm` | `float64` | yes |
| 58 | `median_center_distance_mm` | `float64` | yes |
| 59 | `mean_distance_to_arena_boundary_mm` | `float64` | yes |
| 60 | `median_distance_to_arena_boundary_mm` | `float64` | yes |
| 61 | `experimental_area_geometry_type` | `string` | no |
| 62 | `boundary_distance_method` | `string` | no |
| 63 | `wall_fraction_denominator` | `string` | no |
| 64 | `wall_frame_count` | `int64` | no |
| 65 | `wall_fraction` | `float64` | yes |
| 66 | `representative_position_method` | `string` | no |
| 67 | `representative_x_mm` | `float64` | yes |
| 68 | `representative_y_mm` | `float64` | yes |
| 69 | `mean_heading_deg` | `float64` | yes |
| 70 | `heading_resultant` | `float64` | yes |
| 71 | `bout_count` | `int64` | no |
| 72 | `x_axis_direction` | `string` | no |
| 73 | `y_axis_direction` | `string` | no |
| 74 | `time_bin_policy` | `string` | no |
| 75 | `collection_id` | `string` | yes |
| 76 | `collection_manifest_sha256` | `string` | yes |
| 77 | `collection_manifest_path` | `string` | yes |

## Null, fill, and empty behavior

- Missing or non-finite measurements use Arrow nulls in declared nullable
  fields. No numeric sentinel or NaN is part of the portable contract.
- Counts, indexes, time bounds, method/coordinate declarations, and fixed
  identities are non-null.
- A valid bin with no usable samples keeps its identity/count declarations and
  uses nulls for unavailable fractions and measurements.
- An export containing zero time-bin rows publishes no placeholder Parquet
  part, but retains the exact table declaration and a zero row count in the
  immutable manifest.

## Validation and consumer boundary

The generic exact-schema writer normalizes rows into the declared order and
rejects unexpected fields or null/missing required fields. Part footers bind
exact mode, schema ID, version, and declaration digest. Staged and
manifest-selected validation reconstruct the installed schema and reject
reordered, missing, additional, wrong-type, wrong-nullability, or changed
footer metadata even after part inventory hashes are recomputed.

Baseline-strategy analytics genuinely consumes this table. Its selected-reader
fixture now publishes exact summary and time-bin parts, proves the time-bin
rows drive temporal strategy features, and rejects a rehashed extra-column
part before analytics execution.

## Source-authority quarantine

The positive representation test replaces `_latest_run` with a raw
fixture-group test double and `load_track_kinematics_track` with a
`SimpleNamespace` test double. These seams emulate inputs that would exist
after future verification; the test does not itself verify either source. It
provides representation-path evidence through the real loader, metric builder,
exact writer, staged publication, and selected reader. It provides zero
source-authority validation, selection, or promotion evidence.

## Implementation checklist

- [x] Prove the producer vocabulary is closed and contains no dynamic merge.
- [x] Freeze all 77 fields in executable producer order.
- [x] Freeze exact Arrow types and nullability.
- [x] Keep `fps` nullable and document the current discrepancy.
- [x] Add the table to the digest-bound exact/inferred envelope.
- [x] Freeze the exact 38-key metric-builder order.
- [x] Cover exact writer/footer behavior.
- [x] Cover unexpected and missing required writer fields.
- [x] Cover zero rows without placeholder parts.
- [x] Reject fully rehashed declaration and physical-schema tampering.
- [x] Prove a dynamic source column cannot leak into the output.
- [x] Exercise the real representation path using explicitly labeled test doubles.
- [x] Exercise and adversarially test baseline-strategy selected consumption.
- [x] Run focused tests outside the sandbox: 77 tests passed.
- [x] Complete independent read-only review before commit: ACCEPT.

No storage-performance benchmark is required for this representation-only
change: row values, part partitioning, compression, and publication shape are
unchanged. Source-authority promotion and cross-language consumption remain
separate checkpoints.
