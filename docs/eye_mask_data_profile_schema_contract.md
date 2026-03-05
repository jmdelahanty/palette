# Eye-Mask Data Profile Schema Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
-->

## Purpose

Define a canonical schema for eye-mask dataset profiling so we can:

- compute profile metrics once per dataset/run,
- persist structured profile payloads in Zarr,
- project query-critical fields into registry tables/views,
- enforce fail-closed freshness checks for downstream aggregation/pipeline
  consumers.

This is a schema contract document (design + field definitions). It does not
by itself implement writers/readers.

## Scope

In scope:

- per-dataset eye-mask profile artifact schema (`v1`)
- registry projection schema (`eye_mask_data_profile` + latest views)
- freshness/staleness metadata required by downstream consumers

Out of scope:

- model-quality thresholds/policy gates
- plotting/dashboard style details

## Layered Model

1. Dataset profile artifact (source of truth for profile metrics)
2. Registry projection (query-critical subset + freshness fields)
3. Training data card / pipeline consumers (read registry projection rows)

## Data Ownership and Denormalization Policy

Some fields intentionally exist in multiple locations for performance and
auditability.

- Canonical capture provenance originates from recording/dataset metadata and
  normalized registry lineage entities.
- `analysis/eye_mask_profile_runs/<run>/attrs["profile_summary"]` stores a
  point-in-time profile snapshot for reproducibility.
- `eye_mask_data_profile` and latest views store a query projection optimized
  for filtering and fail-closed freshness checks.

Derived-cache model (not many-writer):

- source metadata is authoritative,
- profile payloads and registry profile rows are derived and refreshable.

Operational refresh policy:

- refresh/sync profile rows:
  - `scripts/py -m fisheye.utils.sync_eye_mask_profile_registry --registry <registry.sqlite> --zarr-use any --apply`
- rerun downstream aggregation/pipeline after sync succeeds.

## Source Association Policy

Eye-mask profile runs are derived analytics artifacts that reference, but do
not mutate, source eye-mask outputs.

- Canonical linkage field: `source.eye_mask_path`
- Canonical path patterns:
  - `eye_masks_runs/<run>`
  - `refined_eye_masks_runs/<run>`

Associated source metadata:

- `source.stage_group`: `eye_masks_runs|refined_eye_masks_runs`
- `source.eye_mask_method`
- `source.eye_mask_run`
- `source.source_keypoint_path`
- `source.source_keypoint_run`
- `source.source_crop_run`
- review fields (when available):
  - `review_state`
  - `review_method`
  - `review_intended_use`
  - `review_timestamp_utc`

## Canonical Dataset Profile Artifact (`v1`)

### Storage Target

Inside profiled Zarr:

- `analysis/eye_mask_profile_runs/<run_name>/`
- parent attrs:
  - `latest`: latest profile run name

### Required Run Attributes

At `analysis/eye_mask_profile_runs/<run_name>.attrs`:

- `schema_name`: `"eye_mask_dataset_profile"`
- `schema_version`: `"v1"`
- `created_at_utc`
- `source_dataset_id`
- `source_recording_id`
- `source_zarr_use`
- `source_eye_mask_path`
- `source_eye_mask_run`
- `source_stage_group`
- `profile_summary` (canonical full payload)

### Required Payload (`profile_summary`)

Canonical top-level shape:

```json
{
  "schema_name": "eye_mask_dataset_profile",
  "schema_version": "v1",
  "created_at_utc": "2026-02-25T13:00:00+00:00",
  "dataset": {
    "dataset_id": "2026-01-28T19-22-28Z_arena_2:zc66de17bea1b",
    "recording_id": "2026-01-28T19-22-28Z_arena_2",
    "zarr_use": "training",
    "zarr_path": "/nvme1/recordings/..._training.zarr"
  },
  "source": {
    "stage_group": "refined_eye_masks_runs",
    "eye_mask_path": "refined_eye_masks_runs/refined_eye_masks_2026-02-25_13-00-00",
    "eye_mask_run": "refined_eye_masks_2026-02-25_13-00-00",
    "eye_mask_method": "traditional",
    "source_keypoint_path": "refined_keypoints_runs/refined_keypoints_2026-02-25_12-40-00",
    "source_keypoint_run": "refined_keypoints_2026-02-25_12-40-00",
    "source_crop_run": "crop_2026-02-25_12-35-00",
    "review_state": "approved",
    "review_method": "manual",
    "review_intended_use": "training",
    "review_timestamp_utc": "2026-02-25T12:59:10+00:00"
  },
  "quality": {
    "rows_total": 231,
    "rows_usable": 228,
    "usable_rate": 0.987,
    "reviewed_rate": 1.0,
    "excluded_rate": 0.013,
    "exclusion_reasons": {
      "missing_keypoint_support": 2,
      "ellipse_fit_failed": 1
    },
    "ellipse_success_rate": 0.995,
    "pair_success_rate": 0.987
  },
  "geometry": {
    "area": {
      "stats": {"p10": 300.0, "p50": 420.0, "p90": 590.0}
    },
    "left_area": {
      "stats": {"p10": 140.0, "p50": 200.0, "p90": 260.0}
    },
    "right_area": {
      "stats": {"p10": 145.0, "p50": 205.0, "p90": 265.0}
    },
    "union_area": {
      "stats": {"p10": 285.0, "p50": 395.0, "p90": 520.0}
    },
    "area_lr_ratio": {
      "stats": {"p10": 0.90, "p50": 0.98, "p90": 1.08}
    },
    "major_axis": {
      "stats": {"p10": 16.0, "p50": 20.0, "p90": 25.0}
    },
    "minor_axis": {
      "stats": {"p10": 8.0, "p50": 11.0, "p90": 14.0}
    },
    "aspect_ratio": {
      "stats": {"p10": 1.4, "p50": 1.8, "p90": 2.2}
    },
    "eye_separation": {
      "stats": {"p10": 24.0, "p50": 30.0, "p90": 36.0}
    }
  },
  "spatial": {
    "edge_proximity_rate": 0.04
  },
  "composition": {
    "rig_id": "omnifin0",
    "camera_id": "2010094",
    "arena_id": "arena_2",
    "dish_design": "cedar",
    "canvas_name": "shadow",
    "protocol_name": "DefaultScreen",
    "genotype": "Tg(elavl3:gcamp7f)",
    "dpf_at_acquisition": 12
  },
  "freshness": {
    "source_keypoint_stale": {
      "state": "fresh",
      "reason": null,
      "timestamp_utc": "2026-02-25T12:58:59+00:00"
    }
  }
}
```

## Registry Projection Schema (Query Layer)

### Table: `eye_mask_data_profile`

Primary key:

- `PRIMARY KEY (dataset_id, profile_run)`

Identity/context:

- `dataset_id`
- `profile_run`
- `recording_id`
- `zarr_use`
- `stage_group`
- `eye_mask_method`
- `source_eye_mask_path`
- `source_eye_mask_run`
- `source_keypoint_path`
- `source_keypoint_run`
- `source_crop_run`
- `profile_created_utc`
- `zarr_mtime_ns`
- `updated_utc`

Quality summary:

- `rows_total`
- `rows_usable`
- `usable_rate`
- `reviewed_rate`
- `excluded_rate`
- `exclusion_reasons_json`
- `ellipse_success_rate`
- `pair_success_rate`

Geometry/spatial summary:

- `area_p10`, `area_p50`, `area_p90`
- `left_area_p10`, `left_area_p50`, `left_area_p90` (optional, recommended)
- `right_area_p10`, `right_area_p50`, `right_area_p90` (optional, recommended)
- `union_area_p10`, `union_area_p50`, `union_area_p90` (optional, recommended)
- `area_lr_ratio_p10`, `area_lr_ratio_p50`, `area_lr_ratio_p90` (optional, recommended)
- `major_axis_p10`, `major_axis_p50`, `major_axis_p90`
- `minor_axis_p10`, `minor_axis_p50`, `minor_axis_p90`
- `aspect_ratio_p10`, `aspect_ratio_p50`, `aspect_ratio_p90`
- `eye_separation_p10`, `eye_separation_p50`, `eye_separation_p90`
- `edge_proximity_rate`

Review/freshness/staleness:

- `review_state`, `review_method`, `review_intended_use`, `review_timestamp_utc`
- `source_keypoint_stale_state`
- `source_keypoint_stale_reason`
- `source_keypoint_stale_timestamp_utc`
- `source_keypoint_stale_json`

Composition/lineage:

- `rig_id`, `camera_id`, `arena_id`
- `dish_design`, `canvas_name`, `protocol_name`
- `genotype`, `dpf_at_acquisition`

Opaque payload:

- `profile_json` (serialized `profile_summary`)

### Views

- `eye_mask_data_profile_latest`
  - latest row per `dataset_id + stage_group + eye_mask_method`
- `recording_eye_mask_data_profile_latest`
  - latest row per `recording_id + stage_group + eye_mask_method`
  - includes joined dataset metadata (`zarr_path`, `artifact_kind`, `dataset_status`)

## Freshness and Fail-Closed Semantics

For profile-driven aggregation/pipeline checks:

- `zarr_mtime_ns` in registry profile rows must match filesystem mtime.
- missing latest profile rows fail closed by default.
- source staleness metadata from `source_keypoint_stale_*` fields is propagated
  for downstream stale-run diagnostics.

Recommended remediation:

1. refresh/sync profile rows,
2. verify with profile registry/check surfaces,
3. rerun aggregation/pipeline.

## Related Contracts

- Detection profile contract:
  - `docs/detection_data_profile_schema_contract.md`
- Keypoint profile contract:
  - `docs/keypoint_data_profile_schema_contract.md`
