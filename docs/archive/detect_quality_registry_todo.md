# Detect Quality Registry TODO

Purpose: add SQL-level detect quality gating and observability with the same fail-closed posture used for keypoint quality.

## Status Snapshot (2026-02-24)

Implemented already:
- `detect_quality` table + gate indexes in `src/fisheye/registry/db.py`.
- `detect_quality_current` latest-row view (dataset + method partition).
- refined-detect quality extraction (`_extract_detect_quality_rows`) with review + interpolation metrics.
- registry write helpers (`upsert_detect_quality`, `replace_detect_quality`, `refresh_detect_quality_for_dataset`).
- maintenance CLI actions:
  - `--backfill-detect-quality`
  - `--refresh-detect-quality`
  - shared `--dry-run` count reporting.
- finalized refinement visualization artifacts for approved refined runs:
  - `detect_quality_overview_png`
  - `refinement_pipeline_overview_png`
- artifact export/view helper:
  - `scripts/py -m fisheye.utils.export_detect_quality_overview`
  - supports `--artifact`, `--recursive`, `--zarr-use`, and direct `--view`.
- detect-quality registry reporting in `check_training_registry`:
  - `--view detect-quality`
  - pass/exclusion summary + exclusion reason breakdown
  - stale-row reason reporting (`missing zarr_mtime_ns`, missing path/file, mtime mismatch)
- detect preflight wrapper + pipeline exclusion summaries:
  - `Detect quality filter summary: passed=<n> excluded=<n> reasons=<...>`

All planned detect-quality TODO items are complete; execution evidence is recorded below.

Archived parallel execution contract:
`docs/archive/detect_quality_parallel_agents_contract.md`.

## Policy Decisions

- [x] Use refined/reviewed detect runs for gating.
  - Gate fields:
    - `review_state`
    - `review_intended_use`
    - detect quality stats (for example interpolated fraction)
  - Keep raw detect details for diagnostics.

- [x] Keep fail-closed build behavior.
  - SQL filter is preselection only.
  - Build/preflight must still validate selected rows against current Zarr metadata.

## Priority 0 (Schema + Read Path)

- [x] Add `detect_quality` table.
  - Proposed columns:
    - `dataset_id TEXT NOT NULL`
    - `refined_run TEXT NOT NULL`
    - `refined_created_utc TEXT`
    - `source_detect_run TEXT NOT NULL`
    - `detect_method TEXT`
    - `review_state TEXT`
    - `review_method TEXT`
    - `review_intended_use TEXT`
    - `review_reviewer TEXT`
    - `review_notes TEXT`
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

- [x] Add `detect_quality_current` view.
  - One canonical row per dataset + detect method.
  - Latest refined run wins.

## Priority 1 (Write Path + Backfill)

- [x] Parse refined detect run quality/review attrs from Zarr.
  - read `detect_review_status`
  - read resolved group and detection-source counts
  - compute interpolated fraction

- [x] Add maintenance actions:
  - `--backfill-detect-quality`
  - `--refresh-detect-quality`
  - include `--dry-run` and detailed counts

- [x] Run one-time backfill on existing registry DB and verify expected row counts.
  - Observed operator run (2026-02-24):
    - backfill dry-run: `scanned=53`, `no_quality=1`, `skipped_existing=52`, `inserted=0`, `updated=0`, `deleted=0`, `unchanged=54`
    - backfill apply: `scanned=53`, `no_quality=1`, `skipped_existing=52`, `inserted=0`, `updated=0`, `deleted=0`, `unchanged=54`
    - refresh dry-run: `scanned=53`, `no_quality=1`, `skipped_existing=0`, `inserted=0`, `updated=0`, `deleted=0`, `unchanged=54`
    - refresh apply: `scanned=53`, `no_quality=1`, `skipped_existing=0`, `inserted=0`, `updated=0`, `deleted=0`, `unchanged=54`
    - detect-quality view summary (initial snapshot):
      - `total rows=52`
      - `passing rows=51`
      - `excluded rows=1`
      - `top exclusion reasons: missing review=1`
    - follow-up targeted review/status repair (same day) + scoped refresh:
      - `total rows=52`
      - `passing rows=52`
      - `excluded rows=0`
      - `top exclusion reasons: none`

## Priority 2 (Fail-Closed Validation)

- [x] Validate selected detect quality rows at build/preflight time.
  - referenced refined run exists
  - review attrs match row
  - detect counts/rates match row
  - freshness (`zarr_mtime_ns`) matches

## Priority 3 (Observability)

- [x] Extend registry reporting to summarize detect quality pass/exclusion counts.
- [x] Add exclusion reason breakdown (missing review, wrong state/use, high interpolation, stale row, etc.).

## Priority 4 (Artifact Finalization + Inspection)

- [x] Finalize approved refined-detect runs into canonical visualization artifacts.
  - Artifact names:
    - `detect_quality_overview_png`
    - `refinement_pipeline_overview_png`
  - Command:
    - `scripts/py -m fisheye.utils.finalize_refinement_artifacts /nvme1/recordings --recursive --zarr-use training --required-intended-use training --apply`
- [x] Provide recursive inspection helper for finalized artifacts directly from Zarr.
  - Command examples:
    - `scripts/py -m fisheye.utils.export_detect_quality_overview /nvme1/recordings --recursive --zarr-use training --artifact detect_quality_overview_png --view`
    - `scripts/py -m fisheye.utils.export_detect_quality_overview /nvme1/recordings --recursive --zarr-use training --artifact refinement_pipeline_overview_png --view`
- [x] Capture operator execution evidence.
  - Latest observed run on real training data:
    - finalize dry-run: `would_finalize=52`
    - finalize apply: `rendered=52`, `errors=0`

## Validation Evidence (Code + Tests)

- `detect_quality_current` latest-row semantics + tie-breaks:
  - `tests/unit/fisheye/test_registry_detect_performance.py`
- detect-quality maintenance determinism:
  - `tests/unit/fisheye/test_registry_maintenance.py`
  - idempotent refresh apply, stale-row deletion, dry-run/apply parity
- fail-closed preflight rejection of stale/divergent rows:
  - `tests/unit/fisheye/test_prepare_detect_training_from_registry.py`
  - `tests/unit/fisheye/test_run_detect_training_pipeline.py`
- detect-quality reporting/exclusion breakdown:
  - `tests/unit/fisheye/test_check_training_registry.py`

## Validation Checklist

- [x] Backfill/refresh commands produce deterministic counts.
- [x] `detect_quality_current` row count aligns with expected dataset-method coverage.
- [x] Preflight rejects stale/divergent selected rows.
- [x] Reports can explain exclusions by reason.
- [x] Approved refined-detect runs can be finalized into both canonical PNG artifacts.
- [x] Finalized artifacts can be viewed recursively from Zarr without an intermediate export file.
