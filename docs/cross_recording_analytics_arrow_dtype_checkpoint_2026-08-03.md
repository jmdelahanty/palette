# Cross-recording analytics Arrow dtype checkpoint — 2026-08-03

## Outcome

Palette now has a closed, digest-bound Arrow-contract envelope for canonical
immutable analytics exports. Twelve of the 30 canonical tables have exact
ordered schemas; the remaining 18 are the chaser tables and remain explicitly
classified as `inferred_v2_compatibility`.

The exact set now includes the six default recording-level tables, the four
baseline/occupancy tables, and both group-statistics tables. This checkpoint
closes the final three default tables:

- `stimulus_response_per_fish_step`: 129 fields;
- `swim_bout_metrics`: 70 fields; and
- `bout_kinematics_metrics`: 150 fields.

Their installed declaration digests are, respectively,
`a8550bb949bdba74b98b81adaa787ab776f92344dbd16059cbeb815b5a1d60ba`,
`de9862bfbe6d14049af4d17526ccd85660d92e50f74eda87fe664338f1f76637`,
and `9f3376755e77782da65f844667f71f80e9429378f1bf0f181a83d92f446f1234`.

Their writers reject unexpected fields, null/missing or duplicate primary
keys, and data-dependent schema expansion. Staged publication and
manifest-selected reads compare complete physical schemas and footer digests
against installed declarations. Historical swim-bout and bout-kinematics
inputs require the explicit `legacy_compatibility=True` source policy; current
inputs fail closed by default.

This remains a bounded checkpoint, not a claim that every analytics export is
frozen. Historical canonical exports that predate the envelope require the
existing explicit `allow_legacy_layout=True` publication-reader compatibility
path. A strict current reader does not silently infer their contract.

No recording-local Zarr schema, selector, registry authority, storage planner,
production archive, or physical Zarr profile changed.

## Read-only census

Three maintained immutable, manifest-selected Parquet families exist today.

| Family | Manifest schema | Tables | Current Arrow state before this checkpoint |
|---|---|---:|---|
| Canonical cross-recording export and group statistics | `palette.analytics_export` v2 plus publication envelope `palette.analytics_export.publication` v1 | 30 | Required column subsets and same-columns-across-parts checks; field order was sorted observed keys and type/nullability were inferred |
| Baseline strategy analytics | `palette.baseline_strategy_analytics` v1 | 4 | Required field subset; type, nullability, and complete field inventory inferred |
| Whole-training response analytics | `palette.training_response_analytics` v1 | 3 | Required field subset; type, nullability, and complete field inventory inferred |

The canonical family consists of 28 recording/cohort tables plus
`group_statistical_summary` and `group_descriptive_summary`. The 28 recording
and cohort tables are:

- `recording_summary`, `stimulus_steps`, `stimulus_step_summary`,
  `stimulus_response_per_fish_step`, `swim_bout_metrics`, and
  `bout_kinematics_metrics`;
- `position_occupancy_histogram_2d`, `baseline_behavior_summary`,
  `baseline_behavior_time_bins`, and `baseline_kinematic_samples`;
- `chaser_epoch_spatial_occupancy_zones`, `chaser_epoch_distance_summary`,
  `chaser_epoch_behavior_summary`, `chaser_epoch_bout_events`,
  `chaser_epoch_bout_histogram`,
  `chaser_epoch_inter_bout_interval_histogram`,
  `chaser_epoch_center_distance_histogram`, `chaser_speed_distance_bins`, and
  `chaser_epoch_distance_histogram`;
- `chaser_quadrant_occupancy_summary`,
  `chaser_quadrant_occupancy_chaser_phase`,
  `chaser_quadrant_occupancy_density`,
  `chaser_near_field_occupancy_summary`,
  `chaser_near_field_occupancy_chaser_phase`,
  `chaser_near_field_occupancy_radial_density`,
  `chaser_near_field_occupancy_distance_cdf`,
  `chaser_egocentric_epoch_summary`, and
  `chaser_egocentric_distance_bearing_histogram`.

The baseline strategy family contains `baseline_strategy_features`,
`baseline_exploration_episodes`, `baseline_strategy_classification`, and
`baseline_strategy_clusters`. The training-response family contains
`training_response_features`, `training_response_classification`, and
`training_response_clusters`.

## Contract boundary

The manifest field `arrow_schema_contracts` is a closed envelope with:

- an exact schema ID and version;
- complete, ordered declarations for exact tables;
- an explicit sorted inventory of inferred-V2 compatibility tables; and
- SHA-256 digests over canonical strict JSON for the envelope and each exact
  table declaration.

Digest validation alone is insufficient. Readers rebuild the expected
envelope from installed declarations, so changing field order, type,
nullability, membership, or nested fields and then recomputing every digest
still fails closed.

An exact Parquet footer binds the Arrow table schema ID, version, and digest.
Compatibility tables carry `palette.arrow_schema_mode =
inferred_v2_compatibility`; this makes the remaining migration surface visible
instead of presenting inferred schemas as exact contracts.

## Core-table decisions

`position_occupancy_histogram_2d` was the first exact table because it had a
closed writer row shape and frozen coordinate/units semantics. The subsequent
default-table closure adds these rules:

- stimulus-response rows are a nullable closed union of maintained stimulus
  modes; arbitrary child attributes are not flattened, and only the two named
  OMR method-version attributes are projected;
- swim-bout rows preserve explicit candidate and signal identity and the
  complete maintained compact payload; and
- bout kinematics is keyed by recording, bout, and measurement level. Its
  movement, heading, and optional gaze fields form a nullable union, while
  physical fixed-width text names are exported without the storage-only
  `_bytes` suffix.

These are representation contracts for immutable query products. Recording-
local Zarr authorities and their selectors are unchanged.

## Implementation checklist

- [x] Census immutable manifest-selected Parquet families.
- [x] Separate the Arrow physical schema version from logical analytics V2.
- [x] Add a closed exact/inferred table partition to canonical manifests.
- [x] Bind exact declarations by schema ID, version, canonical digest, field
  order, type, and nullability.
- [x] Make the canonical exporter use the exact position-occupancy schema.
- [x] Make the group-statistics publisher declare its inferred compatibility
  status instead of emitting an unclassified inferred schema.
- [x] Validate staged batches and published manifest-selected parts.
- [x] Reject reordered, wrong-type, wrong-nullability, unexpected, missing,
  and fully rehashed declaration tampering.
- [x] Keep pre-envelope reads behind explicit legacy compatibility.
- [x] Freeze the six default canonical tables, beginning with stable identity
  and provenance columns shared by every row.
- [x] Freeze baseline behavior and kinematic tables.
- [ ] Freeze the remaining chaser table schemas family by family.
- [x] Freeze `group_statistical_summary` and `group_descriptive_summary` after
  deciding whether method-specific result columns remain one wide table or
  become versioned table variants.
- [ ] Add a closed Arrow envelope and exact schemas to baseline-strategy v2.
- [ ] Add a closed Arrow envelope and exact schemas to training-response v2.
- [ ] Add cross-language fixture reads after each family is exact.

Validation for this checkpoint passed 241 focused contract, Arrow,
atomic-publication, cross-recording exporter, and swim-bout tests. The matrix
also covers rehashed declaration tampering, exact empty tables, duplicate
primary keys, undeclared stimulus-response fields, explicit legacy input
policy, direct Parquet schema equality, and manifest-exclusive selection.

## Promotion rule

Do not remove a table from `inferred_v2_compatibility_tables` until its
producer emits one closed row inventory across all maintained source
capabilities, exact empty/all-null batches are covered, and both staged and
manifest-selected validation reject adversarial physical schemas. Version a
schema change; never reinterpret an existing schema ID/version in place.
