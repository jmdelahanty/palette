# Detection Data Profile Schema Contract
<!-- contract-meta
version: 2
status: active
last_verified: 2026-04-15
-->

## Purpose

Define a canonical schema for detection-data distribution profiling so we can:

- compute once per dataset/run,
- store structured metrics (not only plots),
- aggregate into training data cards without rescanning all Zarrs.

This is a schema contract document (design + field definitions). It does not
by itself implement writers/readers.

## Scope

In scope:
- per-dataset detection profile artifact schema (`v1`)
- registry projection schema for queryable summaries
- training data card aggregate schema (`v1`)

Out of scope:
- CI policy thresholds / pass-fail gates
- exact plotting styles or UI

## Layered Model

1. Dataset profile artifact (source of truth for profile metrics)
2. Registry projection (query-critical subset + references)
3. Training data card (aggregate across selected dataset profiles)

## Data Ownership and Denormalization Policy

Some fields intentionally exist in multiple locations. This is expected and
required for performance + auditability.

- Canonical capture provenance (including subject lineage such as genotype/DPF)
  originates from recording/dataset metadata in Zarr and registry provenance
  entities.
- `analysis/detection_profile_runs/<run>/attrs["profile_summary"]` stores a
  point-in-time snapshot used for reproducibility of profile-derived metrics.
- `detection_data_profile` and latest views store a query projection optimized
  for filtering/build selection without reopening Zarr stores.

This is not a many-writer model:
- source metadata is authoritative,
- profile + registry rows are derived caches/snapshots,
- derived state is refreshable and may be rebuilt.

Operational refresh policy:
- If lineage fields are missing/stale in registry projection rows:
  `scripts/py -m fisheye.utils.sync_detection_profile_registry --registry <registry.sqlite> --zarr-use any --apply`
- If lineage fields are also desired in historical on-disk profile payloads:
  rerun `backfill_detection_profiles --apply`, then rerun sync.

## Stage Association Policy

Detection profile runs are derived analytics artifacts that reference, but do
not mutate, production stage outputs.

- Canonical linkage to stage output is `source_detection_path`.
- Canonical linkage examples:
  - `detect_runs/<run>`
  - `refined_detect_runs/<run>/instances`
- Historical compatibility example:
  - `refined_detect_runs/<run>/manual`
- Selection policy for "best available" source can evolve, but links remain
  explicit per profile run.

Decision (current):
- Do **not** add stage-side reverse pointers (`profile_latest_ref`) in `v1`.
- If a reverse pointer is needed later for UX, add an explicit scoped pointer
  (for example `profile_latest_ref_default`) to avoid config ambiguity.

## Canonical Dataset Profile Artifact (`v1`)

### Storage Target

Recommended canonical location inside each profiled Zarr:

- `analysis/detection_profile_runs/<run_name>/`
- parent attrs:
  - `latest`: latest profile run name

This follows existing `analysis/<analysis_type>_runs/<run_name>/` conventions.

### Required Run Attributes

- `schema_name`: `"detection_dataset_profile"`
- `schema_version`: `"v1"`
- `created_at_utc`: ISO-8601 UTC timestamp
- `source_dataset_id`: registry `dataset_id` when known
- `source_recording_id`: recording ID when known (nullable)
- `source_zarr_use`: e.g. `analysis`, `training` (nullable)
- `source_detection_path`: canonical detection source path used for profile
  - examples:
    - `refined_detect_runs/<run>/instances`
    - `detect_runs/<run>`
- `source_detection_type`: `refined|manual|interpolated|filtered|detect`
  - Current curated refined profiles should use `refined` with
    `source_detection_path = refined_detect_runs/<run>/instances`.
  - `manual|interpolated|filtered` are historical compatibility values for
    legacy sparse archives.
- `source_resolution`: `full|sampled`
- `source_frame_count`: frame universe used for coverage denominator
- `source_frame_count_full`: full frame count when sampled universe is used (nullable)
- `profile_config`: dict (grid size, edge margin, histogram bins, etc.)

### Required Data Payload (JSON)

Writers should emit a single JSON payload in:

- `attrs["profile_summary"]` (canonical)
- optional mirrored array/blob payloads for large histograms

Canonical JSON shape:

```json
{
  "schema_name": "detection_dataset_profile",
  "schema_version": "v1",
  "created_at_utc": "2026-02-23T12:00:00+00:00",
  "dataset": {
    "dataset_id": "2026-01-28T19-22-28Z_arena_2:z...",
    "recording_id": "2026-01-28T19-22-28Z_arena_2",
    "zarr_use": "training",
    "zarr_path": "/nvme1/recordings/..._training.zarr"
  },
  "source": {
    "detection_path": "refined_detect_runs/refined_detect_.../instances",
    "detection_type": "refined",
    "detect_run": "detect_...",
    "refined_run": "refined_detect_...",
    "manual_group": null,
    "review_state": "approved",
    "review_method": "manual",
    "review_intended_use": "training",
    "review_timestamp_utc": "2026-02-23T11:58:00+00:00"
  },
  "coverage": {
    "frames_total": 231,
    "frames_with_detections": 231,
    "coverage_percent": 100.0,
    "frame_source": "full",
    "frames_full": 231
  },
  "counts": {
    "detections_total": 231,
    "detections_per_frame": {
      "mean": 1.0,
      "std": 0.0,
      "min": 1,
      "max": 1,
      "p10": 1.0,
      "p50": 1.0,
      "p90": 1.0
    }
  },
  "geometry_norm": {
    "cx": {},
    "cy": {},
    "w": {},
    "h": {},
    "area": {},
    "aspect_ratio": {}
  },
  "spatial": {
    "edge_margin_norm": 0.05,
    "edge_proximity_rate": 0.02,
    "center_heatmap": {
      "grid_h": 32,
      "grid_w": 32,
      "density": "row-major flattened float array"
    }
  },
  "histograms": {
    "w_norm": {
      "bin_edges": [0.0, 0.02, 0.04],
      "counts": [10, 42]
    },
    "h_norm": {
      "bin_edges": [0.0, 0.02, 0.04],
      "counts": [8, 44]
    },
    "area_norm": {
      "bin_edges": [0.0, 0.001, 0.002],
      "counts": [15, 37]
    },
    "aspect_ratio": {
      "bin_edges": [0.2, 0.4, 0.6],
      "counts": [3, 49]
    }
  },
  "composition": {
    "rig_id": "omnifin0",
    "camera_id": "2010094",
    "arena_id": "arena_2",
    "dish_design": "cedar",
    "canvas_name": "shadow",
    "protocol_name": "DefaultScreen"
  }
}
```

`source.manual_group` is a legacy compatibility field:
- use `null` for current `instances/`-based profiles
- populate it only when the profiled source was a historical sparse subgroup

### Metric Object Contract (`geometry_norm.*`)

Each geometry field (`cx`, `cy`, `w`, `h`, `area`, `aspect_ratio`) should use:

```json
{
  "count": 231,
  "min": 0.01,
  "max": 0.95,
  "mean": 0.42,
  "std": 0.08,
  "p01": 0.03,
  "p05": 0.08,
  "p10": 0.12,
  "p25": 0.35,
  "p50": 0.41,
  "p75": 0.49,
  "p90": 0.54,
  "p95": 0.58,
  "p99": 0.63
}
```

Notes:
- normalized fields use `[0, 1]` coordinate/size space.
- `aspect_ratio = w / h`.
- `area = w * h` in normalized image area units.

## Registry Projection Schema (Query Layer)

Registry should store a compact row per dataset profile run, plus latest views.

### Proposed Table: `detection_data_profile`

Key:
- `dataset_id TEXT NOT NULL`
- `profile_run TEXT NOT NULL`
- `PRIMARY KEY (dataset_id, profile_run)`

Identity/context:
- `recording_id TEXT`
- `zarr_use TEXT`
- `detection_type TEXT`
- `detection_path TEXT`
- `profile_created_utc TEXT`
- `zarr_mtime_ns INTEGER`
- `updated_utc TEXT`

Coverage/counts:
- `frames_total INTEGER`
- `frames_with_detections INTEGER`
- `coverage_percent REAL`
- `detections_total INTEGER`
- `detections_per_frame_p50 REAL`
- `detections_per_frame_p90 REAL`

Geometry summaries:
- `w_p10 REAL`, `w_p50 REAL`, `w_p90 REAL`
- `h_p10 REAL`, `h_p50 REAL`, `h_p90 REAL`
- `area_p10 REAL`, `area_p50 REAL`, `area_p90 REAL`
- `aspect_ratio_p10 REAL`, `aspect_ratio_p50 REAL`, `aspect_ratio_p90 REAL`
- `edge_proximity_rate REAL`

Composition facets:
- `rig_id TEXT`, `camera_id TEXT`, `arena_id TEXT`
- `dish_design TEXT`, `canvas_name TEXT`, `protocol_name TEXT`

Subject-lineage semantics:
- `dish_design` is capture context (dish hardware/design), not subject biology.
- `genotype` and `dpf_at_acquisition` are subject-lineage fields.
- Keep these dimensions separate in query filters and aggregate reporting.

Opaque payload:
- `profile_json TEXT` (full JSON payload for downstream analysis)

### Views

- `detection_data_profile_latest`: latest row per `dataset_id`
- `recording_detection_data_profile_latest`: latest row per `recording_id`

## Training Data Card Aggregate Schema (`v1`)

A training data card is an aggregate over selected dataset profiles.

Canonical payload:

```json
{
  "schema_name": "detection_training_data_card",
  "schema_version": "v1",
  "created_at_utc": "2026-02-23T12:15:00+00:00",
  "set_id": "detect_cedar_shadow_v008",
  "set_version": "v008",
  "selection": {
    "dataset_count": 52,
    "split": "train",
    "filters": {
      "review_state": "approved",
      "intended_use": "training"
    }
  },
  "coverage": {},
  "counts": {},
  "geometry_norm_aggregate": {},
  "spatial_aggregate": {},
  "composition_counts": {
    "rig_id": {"omnifin0": 52},
    "camera_id": {"2010093": 13, "2010094": 13, "2010095": 13, "2010096": 13}
  },
  "train_val_parity": {
    "w_p50_delta": 0.01,
    "h_p50_delta": 0.00,
    "area_p50_delta": 0.002
  },
  "profile_run_refs": [
    {"dataset_id": "...", "profile_run": "detection_profile_2026-02-23_12-00-00"}
  ]
}
```

### Planned Additive Extension (Subject Lineage)

These fields are planned as additive `v1` extensions for training-card lineage
coverage and biology-aware composition metrics:

- `subject_coverage`:
  - manifest dataset count
  - lineage-covered dataset count
  - missing-lineage dataset IDs
- `genotype_counts`:
  - map of genotype string -> count
- `dpf_stats`:
  - numeric summary over `dpf_at_acquisition` (count/min/max/mean/quantiles)
- `dpf_histogram`:
  - bucketed DPF distribution used by plotting helpers

Registry projection extension target (planned):
- `detection_data_profile` / latest views include optional lineage columns:
  - `genotype`
  - `dpf_at_acquisition`

## Invariants and Validation Rules

- `coverage_percent == 100 * frames_with_detections / frames_total` (when `frames_total > 0`)
- histogram counts sum to `detections_total` for bbox histograms
- percentiles are monotonic (`p01 <= p05 <= ... <= p99`)
- `detections_total >= frames_with_detections`
- `frames_total >= frames_with_detections`

## Versioning Policy

- New fields may be added additively within `v1` if optional.
- Breaking changes require `schema_version` bump (`v2`).
- Readers must ignore unknown fields.

## Implementation Notes

- Compute profile from the *resolved* detection source used for training parity:
  - current refined-first path:
    `refined_detect_runs/<run>/instances`
  - historical sparse compatibility path when current canonical `instances/`
    is absent:
    a selected legacy subgroup such as `manual`, `interpolated`, or `filtered`
  - else raw detect run
- Prefer writing numeric profile payload first; plots are derived artifacts.
- Keep profile writer deterministic (stable bin edges + quantile method).
