# Detect Performance Registry TODO

Purpose: make detect-stage runtime/performance queryable in SQLite so operators can quickly identify recordings with weak coverage or degraded runtime behavior.

Design constraints:
- migration-safe (additive schema, no destructive rewrites)
- operator-first (dry-run + backfill + simple SQL/CLI reporting)

## Scope Guardrails (Hybrid Model)

Do not mirror every `check_recording_steps` field into SQLite.

- Keep Zarr as source of truth for detailed stage state.
- Store only high-value, query-critical summaries in registry:
  - detect coverage/confidence/runtime,
  - latest detect run identity,
  - minimal method/config context needed for triage.
- Keep freshness markers on registry rows:
  - `zarr_mtime_ns`
  - `updated_utc`
- Treat `check_recording_steps` as the live verifier and fallback when registry summaries are stale or ambiguous.

Rationale:
- full mirroring increases drift risk between pipeline metadata and DB schema,
- creates frequent migration pressure as stage metadata evolves,
- duplicates logic and raises reconciliation complexity.

## Why This Exists

Today, detect metrics are written to `detect_runs/<run>.attrs` in Zarr, but cross-recording analysis requires opening many archives.

We want fast answers to questions like:
- which recordings had low detect coverage?
- which runs were decode/read bound?
- how does performance vary by rig, camera, arena, dish, model, or task context?

## Current Detect Metrics Already Emitted

Detect runs already include enough metadata to seed a registry table:
- coverage:
  - `summary_statistics.percent_frames_with_detections`
  - `summary_statistics.frames_with_detections`
  - `summary_statistics.frames_with_zero_detections`
- confidence:
  - `summary_statistics.mean_confidence`
  - `summary_statistics.min_confidence`
  - `summary_statistics.max_confidence`
- runtime:
  - `inference_duration_seconds`
  - `inference_average_fps`
  - `inference_avg_batch_ms`
  - `inference_avg_read_ms`
- method/config:
  - `detection_method`
  - `parameters.conf_threshold`
  - `parameters.iou_threshold`
  - `parameters.batch_size`
  - `inference_width`, `inference_height`
- provenance:
  - `model_path`, `model_name`
  - provenance `method`, inputs, git/environment metadata

## Phase Status

- [x] Phase 1 complete (2026-02-09):
  - additive schema migration v10 (`detect_performance` + indexes + latest views),
  - registry write-path extraction/upsert in `register_from_root`,
  - maintenance backfill/refresh command support with dry-run.
- [x] Phase 2A complete (2026-02-09):
  - additive schema migration v11 (`detect_model_performance_latest`, `recording_detect_model_performance_latest`),
  - default detect-performance backfill scope narrowed to source-analysis datasets,
  - opt-in broad scope flag added (`--detect-performance-all-datasets`).
- [x] Phase 2B complete (2026-02-09):
  - `registry_query` detect filters added (`--detect-coverage-min`, `--detect-fps-min`, `--detect-read-ms-max`, `--detect-method`, `--detect-model-like`),
  - model-focused shortcuts added (`--detect-model-only`, `--group-by-model`),
  - model summary output includes dataset/recording counts and coverage/fps/read-ms aggregates.
- [x] Phase 2C complete (2026-02-09):
  - `registry_query` summary mode extended with grouped distributions (`--group-by model|rig|camera|arena|dish`),
  - percentile outputs added for coverage/fps/read_ms (`p10`, `p50`, `p90`),
  - `--group-by-model` retained as compatibility alias for `--group-by model`.

## Immediate Next Priorities

- [x] Narrow default detect-performance backfill scope to source-analysis datasets:
  - `artifact_kind='source_recording'`
  - `zarr_use='analysis'`
  - keep an explicit opt-in path for broader/all-dataset backfills.
- [x] Keep run-level table broad, but make operator model analysis use model-backed views by default.
- [x] Align bbox contract docs to center-based detect boxes (`[cx, cy, w, h]`).

## Proposed Schema (Additive)

- [x] Add table: `detect_performance`
  - key:
    - `dataset_id TEXT NOT NULL`
    - `detect_run TEXT NOT NULL`
    - `PRIMARY KEY (dataset_id, detect_run)`
  - identity/context:
    - `recording_id TEXT`
    - `zarr_use TEXT`
    - `detect_created_utc TEXT`
    - `detection_method TEXT`
    - `model_path TEXT`
    - `model_name TEXT`
  - coverage:
    - `coverage_percent REAL`
    - `frames_with_detections INTEGER`
    - `frames_zero_detections INTEGER`
    - `total_frames INTEGER`
  - confidence:
    - `mean_confidence REAL`
    - `min_confidence REAL`
    - `max_confidence REAL`
  - runtime:
    - `inference_duration_seconds REAL`
    - `inference_average_fps REAL`
    - `inference_avg_batch_ms REAL`
    - `inference_avg_read_ms REAL`
  - parameters:
    - `conf_threshold REAL`
    - `iou_threshold REAL`
    - `batch_size INTEGER`
    - `inference_width INTEGER`
    - `inference_height INTEGER`
  - freshness:
    - `zarr_mtime_ns INTEGER`
    - `updated_utc TEXT`

- [x] Add indexes:
  - `idx_detect_perf_recording` on `(recording_id, detect_created_utc DESC)`
  - `idx_detect_perf_coverage` on `(coverage_percent)`
  - `idx_detect_perf_runtime` on `(inference_average_fps, inference_avg_read_ms)`
  - `idx_detect_perf_method` on `(detection_method, model_name)`
  - `idx_detect_perf_model_path` on `(model_path, model_name, detect_created_utc)`

- [x] Add latest-row view:
  - `detect_performance_latest` (one row per `dataset_id`, latest `detect_created_utc`)

- [x] Add recording-level convenience view:
  - `recording_detect_performance_latest`
  - one row per recording (latest source-recording detect row with joined recording/provenance fields)

## Write Path Plan

- [x] Extract detect performance in registry registration path (`register_from_root`).
  - parse latest detect run first; optionally parse all runs in a follow-up phase.
- [x] Add `replace_detect_performance(dataset_id, rows)` in `Registry`.
  - idempotent upsert, safe on rescans.
- [x] Keep table write optional if detect metadata is absent.
  - no hard failure for archives that have no detect runs.

## Backfill / Repair Plan

- [x] Add maintenance commands:
  - `--backfill-detect-performance`
  - `--refresh-detect-performance`
  - `--detect-performance-all-datasets` (override default scoped mode)
  - both support `--dry-run`.
- [x] Backfill strategy:
  - iterate active source datasets (Phase 2 default: source-recording analysis rows)
  - parse detect metrics from Zarr
  - upsert rows
  - report counts: scanned / with-detect / inserted / updated / skipped-missing.

## Query Surface (Operator-First)

- [x] Add `registry_query` filters:
  - `--detect-coverage-min`
  - `--detect-fps-min`
  - `--detect-read-ms-max`
  - `--detect-method`
  - `--detect-model-like`
  - `--detect-model-only`
- [x] Add summary mode:
  - grouped by model (`--group-by-model`)
  - aggregates: coverage/fps/read_ms (`avg`, coverage `min/max`)
- [x] Extend summary mode:
  - percentile distribution for coverage/fps/read_ms (`p10`, `p50`, `p90`)
  - grouped by rig/camera/arena/dish/model.

## Model-Centric TODO (Next)

- [x] Add model-only latest views for operator workflows:
  - `detect_model_performance_latest` (dataset-level latest run where model-backed detect was used),
  - `recording_detect_model_performance_latest` (recording-level latest model-backed run).
- [x] Define model-backed filter contract:
  - require `model_path` (or `model_name`) to be populated,
  - optionally constrain `detection_method` to known model-based methods (e.g. YOLO).
- [x] Add derived model-performance summaries (view first, materialized table only if needed):
  - `detect_model_performance_summary` (dataset-latest model-backed rows grouped by model identity),
  - `recording_detect_model_performance_summary` (recording-latest model-backed rows grouped by model identity),
  - metrics: coverage/fps/read_ms (`avg`, `min`, `max`, `p10`, `p50`, `p90`) plus dataset/recording counts.
- [x] Add `registry_query` shortcuts for model analysis:
  - `--detect-model-only`
  - `--group-by-model`.

## Example SQL (Target)

```sql
-- Recordings with weak detect coverage
SELECT recording_id, coverage_percent, inference_average_fps, model_name
FROM recording_detect_performance_latest
WHERE coverage_percent < 90
ORDER BY coverage_percent ASC;
```

```sql
-- Decode/read-bound outliers
SELECT recording_id, inference_avg_read_ms, inference_avg_batch_ms, inference_average_fps
FROM recording_detect_performance_latest
WHERE inference_avg_read_ms > 120
ORDER BY inference_avg_read_ms DESC;
```

```sql
-- Coverage distribution by camera
SELECT camera_id,
       COUNT(*) AS n,
       AVG(coverage_percent) AS coverage_avg,
       MIN(coverage_percent) AS coverage_min
FROM recording_detect_performance_latest
GROUP BY camera_id
ORDER BY coverage_avg ASC;
```

## Rollout Sequence

1. [x] Add migration (new table/index/view only).
2. [x] Add extraction/upsert methods in `registry/db.py`.
3. [x] Add maintenance backfill command with dry-run.
4. [x] Backfill on target DB.
5. [x] Run integrity + spot-check SQL.
6. [x] Add query CLI filters and operator docs.

## Validation Checklist

- [x] Migration is additive and reversible via DB backup restore.
- [x] Rescan/register remains successful for datasets with and without detect runs.
- [x] Backfill counts are deterministic on repeated runs.
- [x] Sample recording-level SQL returns expected row counts on target DB.
- [x] `registry_query` can expose low-coverage/model-specific outliers quickly.

Execution snapshot (2026-02-09, target DB `/nvme1/palette_registry.sqlite`):
- schema version: `12`
- detect backfill applied in scoped mode: `inserted=49`
- `detect_performance` rows: `49`
- `recording_detect_model_performance_latest` rows: `48`

## Open Decisions

- [x] Should we store all detect runs or only the latest per dataset on first pass?
  - Recommendation: store all runs in table, expose latest via view.
  - Status: implemented — `_extract_detect_performance_rows()` stores all runs; latest exposed via `detect_performance_latest` view.
- [ ] Should thresholds be policy-enforced now (alerts/gates) or observability-only?
  - Recommendation: observability-only first.
- [x] Do we include non-source datasets (`derived_training_merge`) in this table?
  - Recommendation: source-recording datasets first; extend later if needed.
  - Status: implemented — default scope is source-analysis only; `--detect-performance-all-datasets` opt-in for broader scope.
- [x] Should backfill default scope be narrowed to source-recording analysis datasets only?
  - Recommendation: yes (`artifact_kind='source_recording'` and `zarr_use='analysis'`) for operator-facing defaults.
  - Status: implemented (`source-analysis-only` default; `--detect-performance-all-datasets` opt-in).
