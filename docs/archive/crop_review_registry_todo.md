# Crop Review Registry TODO

Purpose: make crop review state queryable in SQLite so operators can find
recordings needing crop review without scanning/opening every Zarr.

Design constraints:
- migration-safe (additive schema, no destructive rewrites)
- operator-first (dry-run + backfill + simple query/report surfaces)
- keep Zarr as source of truth for detailed run metadata

## Current State

- Detect review status is registry-backed (`detect_quality`, `detect_quality_current`).
- Crop review status is currently Zarr-only:
  - `crop_runs/<run>.attrs["crop_review_status"]`
- Tools that need crop review state still open Zarrs:
  - `fisheye.utils.check_recording_steps`
  - `fisheye.utils.generate_review_list` (`--stage crop`)

This is why crop review workflows still require filesystem-scale polling.

## Scope Guardrails (Hybrid Model)

Do not mirror all crop arrays/metadata into SQLite.

- Keep registry rows to high-value, query-critical fields:
  - latest crop run identity,
  - crop coverage summary,
  - crop review state/method/intended_use/reviewer/timestamp,
  - source linkage needed for audit.
- Keep freshness markers:
  - `zarr_mtime_ns`
  - `updated_utc`
- Treat Zarr as detailed fallback when registry rows are stale or ambiguous.

## Proposed Schema (Additive)

- [x] Add table: `crop_quality`
  - key:
    - `dataset_id TEXT NOT NULL`
    - `crop_run TEXT NOT NULL`
    - `PRIMARY KEY (dataset_id, crop_run)`
  - identity/context:
    - `recording_id TEXT`
    - `zarr_use TEXT`
    - `crop_created_utc TEXT`
    - `source_detect_run TEXT`
    - `source_refined_run TEXT`
    - `detection_source_type TEXT`
    - `detection_source_path TEXT`
  - crop summary:
    - `total_rois INTEGER`
    - `frames_with_crops INTEGER`
    - `total_frames INTEGER`
    - `percent_frames_with_crops REAL`
    - `includes_interpolated INTEGER`
    - `n_real_detections INTEGER`
    - `n_interpolated_detections INTEGER`
  - review fields:
    - `review_state TEXT`
    - `review_method TEXT`
    - `review_intended_use TEXT`
    - `review_reviewer TEXT`
    - `review_timestamp_utc TEXT`
    - `review_notes TEXT`
  - freshness:
    - `zarr_mtime_ns INTEGER`
    - `updated_utc TEXT`

- [x] Add indexes:
  - `idx_crop_quality_dataset_id` on `(dataset_id)`
  - `idx_crop_quality_review_gate` on `(review_state, review_intended_use)`
  - `idx_crop_quality_source` on `(detection_source_type, source_refined_run)`
  - `idx_crop_quality_recording` on `(recording_id, crop_created_utc DESC)`

- [x] Add views:
  - `crop_quality_current` (latest crop row per dataset)
  - `recording_crop_quality_current` (recording-level latest row with joined dataset/provenance context)

## Write Path Plan

- [x] Add crop extraction helper in registry scan path:
  - parse crop runs and latest run attrs,
  - parse `crop_review_status`,
  - extract summary stats and source linkage.
- [x] Add `replace_crop_quality(dataset_id, rows)` in `Registry`.
- [x] Write crop rows from `register_from_root`/`scan_zarr`.
- [x] Keep write optional when crop runs are absent.

## Maintenance / Backfill Plan

- [x] Add maintenance actions:
  - `--backfill-crop-quality`
  - `--refresh-crop-quality`
  - both support `--dry-run`.
- [x] Default scope should be source-analysis datasets
  (`artifact_kind='source_recording' AND zarr_use='analysis'`), with explicit
  opt-in for broader scope if needed.

## Query / Operator Surface

- [x] Extend `registry_query` with crop review filters:
  - `--crop-review-state`
  - `--crop-review-intended-use`
  - `--crop-source-type`
  - optional coverage thresholds.
- [x] Add/extend grouped summary mode for crop-review backlog reporting.
- [x] Add registry-first mode in list tooling:
  - `generate_review_list --stage crop --registry ...` should query registry
    rows first instead of opening all Zarrs.
- [x] Optionally add registry-prefer mode in `check_recording_steps` so review
  columns can come from `crop_quality_current`.

## Freshness + Fail-Closed Behavior

- [x] Compare `crop_quality_current.zarr_mtime_ns` to current dataset mtime.
- [x] If stale/missing, mark row as stale and avoid false "approved" reports.
- [x] Keep optional Zarr fallback for diagnostics and recovery.

## Example Queries (Target)

```sql
-- Analysis datasets with crop present but not approved for training
SELECT recording_id, zarr_path, review_state, review_intended_use, crop_run
FROM recording_crop_quality_current
WHERE zarr_use = 'analysis'
  AND (review_state IS NULL OR review_state <> 'approved' OR review_intended_use <> 'training')
ORDER BY recording_id;
```

```sql
-- Crop-review backlog by reviewer/method
SELECT review_method, review_reviewer, COUNT(*) AS n
FROM crop_quality_current
WHERE review_state IS NULL OR review_state <> 'approved'
GROUP BY review_method, review_reviewer
ORDER BY n DESC;
```

## Rollout Phases

- [x] Phase 1: schema + extraction + scan write path.
- [x] Phase 2: maintenance backfill/refresh commands.
- [x] Phase 3: query/list/status tooling integration.
- [x] Phase 4: integrity/freshness checks + docs updates.

## Validation Checklist

- [x] Backfill/refresh commands are deterministic on repeat runs.
- [x] `crop_quality_current` row count matches expected analysis dataset coverage.
- [x] Registry queries can return "not approved" crop backlog without full
  filesystem scan.
- [x] Stale rows are detectable and do not silently pass as approved.
