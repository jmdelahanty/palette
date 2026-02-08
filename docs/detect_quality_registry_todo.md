# Detect Quality Registry TODO

Purpose: add SQL-level detect quality gating and observability with the same fail-closed posture used for keypoint quality.

## Policy Decisions

- [ ] Use refined/reviewed detect runs for gating.
  - Gate fields:
    - `review_state`
    - `review_intended_use`
    - detect quality stats (for example interpolated fraction)
  - Keep raw detect details for diagnostics.

- [ ] Keep fail-closed build behavior.
  - SQL filter is preselection only.
  - Build/preflight must still validate selected rows against current Zarr metadata.

## Priority 0 (Schema + Read Path)

- [ ] Add `detect_quality` table.
  - Proposed columns:
    - `dataset_id TEXT NOT NULL`
    - `refined_run TEXT NOT NULL`
    - `refined_created_utc TEXT`
    - `source_detect_run TEXT NOT NULL`
    - `detect_method TEXT`
    - `review_state TEXT`
    - `review_intended_use TEXT`
    - `review_reviewer TEXT`
    - `review_timestamp_utc TEXT`
    - `review_resolved_group TEXT`
    - `total_detections INTEGER`
    - `real_detections INTEGER`
    - `interpolated_detections INTEGER`
    - `interpolated_detections_rate REAL`
    - `quality_updated_utc TEXT`
    - `zarr_mtime_ns INTEGER`
  - Constraints/indexes:
    - uniqueness on `(dataset_id, refined_run)`
    - index on `(review_state, review_intended_use, detect_method, interpolated_detections_rate)`
    - index on `dataset_id`

- [ ] Add `detect_quality_current` view.
  - One canonical row per dataset + detect method.
  - Latest refined run wins.

## Priority 1 (Write Path + Backfill)

- [ ] Parse refined detect run quality/review attrs from Zarr.
  - read `detect_review_status`
  - read resolved group and detection-source counts
  - compute interpolated fraction

- [ ] Add maintenance actions:
  - `--backfill-detect-quality`
  - `--refresh-detect-quality`
  - include `--dry-run` and detailed counts

- [ ] Run one-time backfill on existing registry DB and verify expected row counts.

## Priority 2 (Fail-Closed Validation)

- [ ] Validate selected detect quality rows at build/preflight time.
  - referenced refined run exists
  - review attrs match row
  - detect counts/rates match row
  - freshness (`zarr_mtime_ns`) matches

## Priority 3 (Observability)

- [ ] Extend registry reporting to summarize detect quality pass/exclusion counts.
- [ ] Add exclusion reason breakdown (missing review, wrong state/use, high interpolation, stale row, etc.).

## Validation Checklist

- [ ] Backfill/refresh commands produce deterministic counts.
- [ ] `detect_quality_current` row count aligns with expected dataset-method coverage.
- [ ] Preflight rejects stale/divergent selected rows.
- [ ] Reports can explain exclusions by reason.
