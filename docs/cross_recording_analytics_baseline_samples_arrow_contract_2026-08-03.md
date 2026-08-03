# Baseline-kinematic-samples exact Arrow contract — 2026-08-03

## Decision

`baseline_kinematic_samples` uses exact physical Arrow schema v1 for new
immutable cross-recording analytics exports. Its 71 fields have a closed
order, Arrow type, and nullability contract. The logical table remains
`palette.analytics.table.baseline_kinematic_samples` v1 inside analytics export
v2; the independently versioned physical schema is
`palette.analytics_export.arrow_table.baseline_kinematic_samples` v1.

This freezes the Parquet representation only. It does not change or promote
source authority, default table selection, sampling policy, recording-local
Zarr contracts, selectors, registries, or physical part partitioning.

## Closed producer vocabulary

The executable producer assembles each row from four closed surfaces:

1. five shared export identity fields;
2. the fixed chaser/source/track/arena lineage block;
3. the fixed 32-key dictionary returned by `build_sample_metrics()`; and
4. three optional virtual-collection fields.

The two metric coordinate keys replace fields already present in the lineage
prefix without changing their insertion positions, so the 32-key builder adds
30 new physical positions. The complete schema is therefore 38 prefix fields,
30 newly inserted metric fields, and three collection fields.

The source epoch row only supplies named window bounds before sampling. It is
never merged into a sample row. The representation fixture includes a
`future_source_metric` source column and proves that it cannot leak into the
Parquet output. A builder test separately freezes all 32 keys in executable
order.

`fps` remains nullable because computation may use verified track FPS while
the exported lineage field currently reads only the chaser-run attribute. The
contract records this behavior instead of changing source authority.

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
| 43 | `source_sample_index` | `int64` | no |
| 44 | `source_frame` | `int64` | no |
| 45 | `source_time_s` | `float64` | no |
| 46 | `relative_time_s` | `float64` | no |
| 47 | `x_arena_mm` | `float64` | yes |
| 48 | `y_arena_mm` | `float64` | yes |
| 49 | `x_arena_fraction` | `float64` | yes |
| 50 | `y_arena_fraction` | `float64` | yes |
| 51 | `speed_mm_s` | `float64` | yes |
| 52 | `heading_deg` | `float64` | yes |
| 53 | `frame_path_distance_mm` | `float64` | yes |
| 54 | `center_distance_mm` | `float64` | yes |
| 55 | `distance_to_arena_boundary_mm` | `float64` | yes |
| 56 | `wall` | `bool` | yes |
| 57 | `experimental_area_geometry_type` | `string` | no |
| 58 | `boundary_distance_method` | `string` | no |
| 59 | `position_valid` | `bool` | no |
| 60 | `sample_valid` | `bool` | no |
| 61 | `sampling_policy` | `string` | no |
| 62 | `sampling_stride_frames` | `int64` | no |
| 63 | `requested_sample_rate_hz` | `float64` | yes |
| 64 | `source_sample_rate_hz` | `float64` | no |
| 65 | `nominal_sample_rate_hz` | `float64` | no |
| 66 | `effective_sample_rate_hz` | `float64` | no |
| 67 | `x_axis_direction` | `string` | no |
| 68 | `y_axis_direction` | `string` | no |
| 69 | `collection_id` | `string` | yes |
| 70 | `collection_manifest_sha256` | `string` | yes |
| 71 | `collection_manifest_path` | `string` | yes |

## Null, fill, and sampling semantics

- Missing or non-finite scientific measurements use Arrow nulls. Numeric
  sentinels and NaN are not part of the portable contract.
- Invalid positions have null coordinates, normalized fractions, center and
  boundary distances, and `wall`; `position_valid` remains the non-null false
  validity declaration.
- Missing speed, heading, or frame-path values are null independently of
  `sample_valid`.
- `requested_sample_rate_hz` is null under the all-source-samples policy and
  non-null for target-rate sampling.
- Sample/source indexes, times, policies, stride, effective rates, coordinate
  declarations, and both validity booleans are non-null.
- Zero selected samples publish no placeholder Parquet part, while the exact
  table declaration and zero row count remain in the immutable manifest.

## Publication and consumer boundary

The exact writer normalizes every sample into declared order and rejects
unexpected fields or null/missing non-nullable fields. Footer metadata binds
exact mode, schema ID, schema version, and declaration digest. Staged and
manifest-selected validation reject reordered, missing, additional,
wrong-type, wrong-nullability, and changed-footer schemas even when part
inventory hashes are recomputed.

Baseline-strategy analytics consumes manifest-selected sample parts. The
consumer fixture publishes six identities with full sample sequences, proves
sample-dependent features and exploration episodes are produced, and rejects
a rehashed extra-column sample part before analytics execution.

## Source-authority quarantine

The positive representation test replaces `_latest_run` with a raw
fixture-group test double and `load_track_kinematics_track` with a
`SimpleNamespace` test double. Those seams emulate post-verification inputs;
the test does not verify the sources themselves. It provides representation
evidence through the real baseline loader, full-resolution builder, exact
writer, staged publication, and selected reader. It provides zero
source-authority validation, selection, or promotion evidence.

## Implementation checklist

- [x] Prove the producer vocabulary is closed and has no dynamic source merge.
- [x] Freeze all 71 fields in executable insertion order.
- [x] Freeze exact Arrow types and nullability.
- [x] Freeze the exact 32-key sample-builder order.
- [x] Document nullable `fps` and full-resolution requested-rate behavior.
- [x] Prove invalid scientific values become nulls, not sentinels.
- [x] Add the table to the digest-bound exact/inferred envelope.
- [x] Cover writer order, footer metadata, and required/unexpected fields.
- [x] Cover zero rows without placeholder parts.
- [x] Reject fully rehashed declaration and physical-schema tampering.
- [x] Prove a dynamic source column cannot leak into the output.
- [x] Exercise the real full-resolution representation path with test doubles.
- [x] Exercise and adversarially test baseline-strategy selected consumption.
- [x] Run focused tests outside the sandbox: 96 tests passed.
- [x] Complete independent read-only review before commit: ACCEPT.

No performance result is claimed here. Exact schema publication does not
change sample count, Parquet part partitioning, compression, or the opt-in
default. Because this table can be much larger than summary/time-bin tables,
part sizing and scan benchmarks remain a separate future checkpoint before any
physical-profile change.
