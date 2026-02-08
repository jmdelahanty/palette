# Keypoint Quality Registry TODO

Purpose: move keypoint training-data quality gating from per-Zarr runtime checks to SQL-level registry filtering, while keeping build-time fail-closed validation.

## Policy Decisions

- [x] Use refined/reviewed quality for gating.
  - Gate criteria:
    - `review_state`
    - `review_intended_use`
    - `usable_keypoints_rate`
  - Do not gate on raw first-pass keypoint success.
  - Keep raw success metrics for diagnostics only.

- [x] Keep fail-closed behavior during build.
  - If selected rows are stale/missing/inconsistent at build time, fail loudly.
  - SQL filtering is convenience + speed, not the only safety layer.

## Priority 0 (Schema + Read Path)

- [x] Add a keypoint quality table in registry (`keypoint_quality`).
  - Proposed columns:
    - `dataset_id TEXT NOT NULL`
    - `refined_run TEXT NOT NULL`
    - `source_keypoint_run TEXT NOT NULL`
    - `keypoint_method TEXT` (`traditional_pose`, `yolo_pose`, etc.)
    - `review_state TEXT`
    - `review_intended_use TEXT`
    - `review_reviewer TEXT`
    - `review_timestamp_utc TEXT`
    - `usable_keypoints INTEGER`
    - `total_keypoints INTEGER`
    - `usable_keypoints_rate REAL`
    - `raw_keypoints_success_rate REAL` (diagnostic-only)
    - `raw_keypoints_successful INTEGER` (diagnostic-only)
    - `quality_updated_utc TEXT`
    - freshness fields: `zarr_mtime_ns INTEGER` (or equivalent)
  - Constraints/indexes:
    - uniqueness on `(dataset_id, refined_run)`
    - index on `(review_state, review_intended_use, keypoint_method, usable_keypoints_rate)`
    - index on `dataset_id`

- [x] Add a SQL view for “current” quality rows (`keypoint_quality_current`).
  - One canonical row per dataset + method.
  - Picks latest refined run (by timestamp) that has quality metadata.
  - Exposes fields needed by query/selection CLI.

- [x] Integrate query path in keypoint prep.
  - Apply review/use/usable-rate filters from registry first.
  - Preserve current selector semantics (`latest_traditional`, `latest_yolo`).

## Priority 1 (Write Path + Backfill)

- [x] Write/refresh quality rows during scan/maintenance.
  - Parse refined run + review attrs from Zarr.
  - Compute usable totals/rates.
  - Upsert into `keypoint_quality`.

- [x] Add explicit maintenance command for refresh/backfill.
  - Example flags:
    - `--backfill-keypoint-quality`
    - `--refresh-keypoint-quality`
    - optional path scoping
  - Print inserted/updated/skipped counts.

- [x] Backfill existing registry database.
  - Run one-time backfill over current datasets.
  - Verify expected row counts and method coverage.

## Priority 2 (Fail-Closed Build Semantics)

- [x] Enforce build-time consistency checks even after SQL filtering.
  - For each selected dataset:
    - ensure referenced refined run still exists
    - ensure review attrs in Zarr match selected registry row
    - ensure usable totals/rate still match expected values (or within strict tolerance)
  - Fail with actionable message on divergence.

- [x] Add clear UX around exclusion vs failure.
  - SQL query excludes below-threshold datasets.
  - Build step still fails on stale/invalid selected records.
  - Exclusions are non-fatal by default; selected rows still run fail-closed validation.

## Priority 3 (Observability + UX)

- [x] Extend registry status/report tooling for keypoint quality.
  - Show:
    - dataset count passing current quality gates
    - excluded count + top reasons
    - stale/divergent quality rows

- [x] Add an audit view/report for method alignment.
  - Catch cross-method review fallbacks explicitly.
  - Show whether fallback was required or avoided.

- [x] Document operator workflow.
  - “Review keypoints -> refresh registry quality -> build/train pipeline”
  - Include recovery steps for stale consolidated metadata.
  - Workflow doc: `docs/keypoint_quality_registry_workflow.md`.

## Validation Checklist

- [x] SQL query can filter by:
  - selector method
  - `review_state`
  - `review_intended_use`
  - minimum `usable_keypoints_rate`
- [x] Keypoint preflight no longer depends on raw-success thresholds for gating.
- [x] Build fails on stale/missing/divergent quality records.
- [x] Backfill populates expected rows for existing archives.
- [x] Audit output can explain every exclusion.
