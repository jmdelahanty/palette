# Keypoint Data Profile Schema Contract
<!-- contract-meta
version: 1
status: draft
implementation: implemented
last_verified: 2026-04-10
-->

## Purpose

Define a canonical schema for keypoint-data distribution profiling so we can:

- compute once per dataset/run,
- store structured profile metrics in Zarr,
- project query-critical fields into registry tables/views,
- aggregate keypoint training data cards without rescanning every source store.

This is a schema contract document (design + field definitions). It does not
by itself implement writers/readers.

## Scope

In scope:
- per-dataset keypoint profile artifact schema (`v1`)
- registry projection schema (`keypoint_data_profile` + latest views)
- linkage fields consumed by keypoint training data-card aggregation

Out of scope:
- model-quality pass/fail policy thresholds
- plot styling or dashboard implementation details

## Layered Model

1. Dataset profile artifact (source of truth for profile metrics)
2. Registry projection (query-critical subset + references)
3. Training data card (aggregate across selected dataset profiles)

## Data Ownership and Denormalization Policy

Some fields intentionally exist in multiple locations for performance and
auditability.

- Canonical capture provenance originates from recording/dataset metadata
  and normalized registry lineage entities.
- `analysis/keypoint_profile_runs/<run>/attrs["profile_summary"]` stores a
  point-in-time profile snapshot for reproducibility.
- `keypoint_data_profile` and latest views store a query projection optimized
  for selection/filtering and fail-closed freshness checks.

This is a derived-cache model, not a many-writer model:

- source metadata is authoritative,
- profile payloads and registry profile rows are derived and refreshable.

Operational refresh policy:

- re-derive one dataset from Zarr truth (runs every extractor plus the
  detection/keypoint profile extractors idempotently):
  - `scripts/py -m fisheye.registry.maintenance --registry <registry.sqlite> --reconcile-dataset <recording_analysis.zarr>`
  - (the standalone `sync_keypoint_profile_registry` CLI was retired in favor of
    `Registry.reconcile_dataset_from_root`).
- bulk refresh across datasets via maintenance:
  - `scripts/py -m fisheye.registry.maintenance --registry <registry.sqlite> --refresh-keypoint-profiles`

## Source Association Policy

Keypoint profile runs are derived analytics artifacts that reference, but do
not mutate, source keypoint outputs.

- Canonical linkage field: `source.keypoint_path`
- Canonical path patterns:
  - `keypoints_runs/<run>`
  - `refined_keypoints_runs/<run>`
  - `keypoints_refined_runs/<run>` (legacy parent alias)

Associated source metadata:

- `source.keypoint_type`: `keypoints|refined`
- `source.keypoint_method`: e.g. `traditional_pose`, `yolo_pose`
- `source.keypoint_run`
- `source.refined_run`
- review fields (when refined source is used):
  - `review_state`
  - `review_method`
  - `review_intended_use`
  - `review_timestamp_utc`

## Canonical Dataset Profile Artifact (`v1`)

### Storage Target

Inside profiled Zarr:

- `analysis/keypoint_profile_runs/<run_name>/`
- parent attrs:
  - `latest`: latest profile run name

### Required Run Attributes

At `analysis/keypoint_profile_runs/<run_name>.attrs`:

- `schema_name`: `"keypoint_dataset_profile"`
- `schema_version`: `"v1"`
- `created_at_utc`
- `source_dataset_id`
- `source_recording_id`
- `source_zarr_use`
- `source_keypoint_path`
- `source_keypoint_method`
- `source_keypoint_run`
- `source_refined_run`
- `source_skeleton_id`
- `source_kpt_shape`
- `source_pose_schema_name`
- `source_pose_schema`
- `source_heading_computation_source`
- `source_heading_computation`
- `source_row_count`
- `profile_summary` (canonical full payload)

### Required Payload (`profile_summary`)

Canonical top-level shape:

```json
{
  "schema_name": "keypoint_dataset_profile",
  "schema_version": "v1",
  "created_at_utc": "2026-02-24T18:40:40+00:00",
  "dataset": {
    "dataset_id": "2026-01-28T19-22-28Z_arena_1:zc66de17bea1b",
    "recording_id": "2026-01-28T19-22-28Z_arena_1",
    "zarr_use": "training",
    "zarr_path": "/nvme1/recordings/..._training.zarr"
  },
  "source": {
    "keypoint_path": "refined_keypoints_runs/refined_keypoints_2026-02-04_12-45-00",
    "keypoint_type": "refined",
    "keypoint_method": "traditional_pose",
    "keypoint_run": "keypoints_2026-02-04_17-33-09",
    "refined_run": "refined_keypoints_2026-02-04_12-45-00",
    "review_state": "approved",
    "review_method": "manual",
    "review_intended_use": "training",
    "review_timestamp_utc": "2026-02-04T18:14:00+00:00",
    "skeleton_id": "traditional_v1",
    "kpt_shape": [3, 2],
    "pose_schema_name": "traditional_v1",
    "pose_schema": {
      "name": "traditional_v1",
      "skeleton_id": "traditional_v1",
      "kpt_shape": [3, 2],
      "edges": [[0, 1], [0, 2], [1, 2]],
      "metadata": {
        "heading_computation": {
          "version": 1,
          "enabled": true,
          "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"]
        }
      }
    },
    "heading_computation_source": "pose_schema.metadata.heading_computation",
    "heading_computation": {
      "version": 1,
      "enabled": true,
      "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"]
    }
  },
  "quality": {
    "rows_total": 231,
    "rows_usable": 231,
    "usable_keypoints_total": 231,
    "usable_rate": 1.0,
    "confidence_valid_rate": 1.0,
    "geometry_valid_rate": 1.0
  },
  "geometry": {
    "triangle_area": {"stats": {}},
    "min_angle": {"stats": {}},
    "heading": {"stats": {}},
    "derived_metrics_schema": {
      "schema_version": 1,
      "entity_kind": "keypoint_roi"
    },
    "derived_metrics": {
      "schema_id": "traditional_v2_derived_metrics",
      "metrics": []
    }
  },
  "composition": {
    "rig_id": "omnifin0",
    "camera_id": "2010094",
    "arena_id": "arena_1",
    "dish_design": "cedar",
    "canvas_name": "shadow",
    "protocol_name": "DefaultScreen",
    "genotype": "Tg(elavl3:gcamp7f)",
    "dpf_at_acquisition": 12
  }
}
```

### Geometry Metric Object Contract

For `geometry.<metric>.stats` (`triangle_area`, `min_angle`, `heading`):

- `count`
- `min`
- `max`
- `mean`
- `std`
- `p10`
- `p50`
- `p90`

### Optional Derived Metric Profile Payload

When a refined run exposes schema-driven derived metrics, `profile_summary` may
also include:

- `geometry.derived_metrics_schema`
- `geometry.derived_metrics.schema_id`
- `geometry.derived_metrics.schema_version`
- `geometry.derived_metrics.labels`
- `geometry.derived_metrics.normalization`
- `geometry.derived_metrics.metrics[*].name`
- `geometry.derived_metrics.metrics[*].valid_count`
- `geometry.derived_metrics.metrics[*].valid_rate`
- `geometry.derived_metrics.metrics[*].stats`
- `geometry.derived_metrics.metrics[*].stats_norm` (when normalized values exist)

These fields are profile-payload only in `v1`; they are not projected into
registry SQL columns yet.

## Registry Projection Schema (Query Layer)

### Table: `keypoint_data_profile`

Primary key:

- `PRIMARY KEY (dataset_id, profile_run)`

Identity/context:

- `dataset_id`
- `profile_run`
- `recording_id`
- `zarr_use`
- `keypoint_method`
- `source_keypoint_path`
- `source_keypoint_run`
- `skeleton_id`
- `kpt_shape`
- `profile_created_utc`
- `zarr_mtime_ns`
- `updated_utc`

Quality summary:

- `rows_total`
- `rows_usable`
- `usable_keypoints_total`
- `usable_rate`
- `confidence_valid_rate`
- `geometry_valid_rate`

Geometry summary:

- `triangle_area_p10`, `triangle_area_p50`, `triangle_area_p90`
- `min_angle_p10`, `min_angle_p50`, `min_angle_p90`
- `heading_p10`, `heading_p50`, `heading_p90`

Skeleton/heading metadata:

- `pose_schema_name`
- `pose_schema_json`
- `heading_computation_source`
- `heading_computation_json`

Deferred in `v1`:

- no SQL columns for skeleton-specific derived metrics
- no latest-view query fields for `total_length`, `tail_length`, `head_length`,
  `eye_span`, or future skeleton-specific metric names

Composition/lineage:

- `rig_id`, `camera_id`, `arena_id`
- `dish_design`, `canvas_name`, `protocol_name`
- `genotype`, `dpf_at_acquisition`

Opaque payload:

- `profile_json` (serialized `profile_summary`)

### Views

- `keypoint_data_profile_latest`
  - latest row per `dataset_id + keypoint_method`
- `recording_keypoint_data_profile_latest`
  - latest row per `recording_id + keypoint_method`
  - includes joined dataset metadata (`zarr_path`, `artifact_kind`, `dataset_status`)

### Query Surface

Registry query CLI modes:

- `--keypoint-data-profile-latest`
- `--recording-keypoint-data-profile-latest`

Shared profile filters:

- `--profile-dataset-id`
- `--profile-recording-id`
- `--profile-zarr-use`
- `--profile-detection-type` (maps to `keypoint_method` in keypoint profile modes)
- `--profile-coverage-min` (maps to `usable_rate` in keypoint profile modes)

Derived-metric filtering/query is intentionally deferred until we define a
cross-skeleton policy for metric availability and comparability.

## Freshness and Fail-Closed Semantics

For profile-driven aggregation/pipeline checks:

- `zarr_mtime_ns` in registry profile rows must match filesystem mtime,
- missing latest profile rows fail closed by default.

Explicit overrides (for operational recovery only):

- `--allow-profile-mtime-mismatch`
- `--allow-profile-fallback-scan`

Recommended remediation:

1. refresh/sync profile rows,
2. verify with `check_training_registry --view keypoint-profile`,
3. rerun aggregation/pipeline.

## Related Contracts

- Detect profile contract:
  - `docs/detection_data_profile_schema_contract.md`
- Keypoint training card contract:
  - `docs/keypoint_training_data_card_contract.md`
