# Detection + Registry Curation TODO

Purpose: track the concrete follow-up work from the detection dataset and SQLite registry review.

## Priority 0 (Training Correctness)

- [x] Enforce requested detection source type during training dataset load.
  - Problem: loader currently warns and silently uses available source type.
  - File: `src/fisheye/training/zarr_yolo_dataset_loader.py`
  - Goal: fail fast on mismatch unless an explicit override flag is set.
  - Status: implemented with config/CLI override (`allow_source_mismatch`, `--allow-source-mismatch`) and report logging.
  - Acceptance:
    - Training fails when config requests `manual` but crop source is `filtered`.
    - Optional override allows fallback and records fallback in training report.

- [x] Implement actual sampling strategies (`balanced`, `proportional`, `weighted`).
  - Problem: strategies are parsed but not applied.
  - File: `src/fisheye/training/zarr_yolo_dataset_loader.py`
  - Goal: dataset composition reflects selected strategy.
  - Status: implemented in `GlobalIndexManager._build_global_index` with deterministic sampling and strict weighted key validation.
  - Acceptance:
    - `balanced`: equalized contribution across datasets.
    - `proportional`: contribution follows available sample counts.
    - `weighted`: contribution follows `dataset_weights`.

- [x] Implement per-dataset split handling.
  - Problem: per-dataset `split` is parsed but ignored; one global split is used.
  - File: `src/fisheye/training/zarr_yolo_dataset_loader.py`
  - Goal: apply split ratios per dataset before final merge.
  - Status: implemented via per-dataset splitting in `GlobalIndexManager.get_split_indices` with global fallback and per-dataset train/val logging.
  - Acceptance:
    - Reported train/val counts match per-dataset split config.
    - Global split is only fallback when per-dataset split is absent.

## Priority 1 (Registry Completeness)

- [x] Populate `detection_sources` table during registration/curation.
  - Problem: table exists but rows are not written.
  - File: `src/fisheye/registry/db.py`
  - Goal: track source type and counts for each dataset/refined run.
  - Status: implemented via `register_from_root` calling `replace_detection_sources(...)` with source/count extraction from latest crop (or detect fallback).
  - Acceptance:
    - Non-zero rows in `detection_sources`.
    - Rows update when refined/crop source changes.

- [x] Populate `training_sets` table from manifest generation.
  - Problem: table exists but rows are not written.
  - Files: `src/fisheye/diagnostics/prepare_detect_training.py`, `src/fisheye/registry/db.py`
  - Goal: store set name/version, query/filter metadata, dataset IDs.
  - Status: implemented via `Registry.upsert_training_set(...)` in manifest generation and manifest `set_id` propagation into training run logging.
  - Acceptance:
    - `prepare_detect_training --set-name ...` writes/updates `training_sets`.
    - `training_runs.set_id` can be joined to an existing set row.

- [x] Add missing-dataset lifecycle handling (`active` vs `missing`).
  - Problem: scanner upserts to `active` but does not mark disappeared paths as `missing`.
  - File: `src/fisheye/registry/db.py`
  - Goal: support periodic reconciliation of missing moved/deleted Zarrs.
  - Status: implemented via scoped reconciliation (`Registry.reconcile_missing_datasets`) integrated into scan/rescan flows.
  - Acceptance:
    - Rescan marks previously-known absent datasets as `missing`.
    - Reappearance flips status back to `active`.

## Priority 2 (Registry Usability)

- [x] Replace stub query CLI with feature-complete query interface.
  - Problem: `query.py` is labeled stub and does not expose full query/report capabilities.
  - File: `src/fisheye/registry/query.py`
  - Goal: support curated filtering and export (`table`, `json`, `csv`) for dataset selection.
  - Status: implemented with richer provenance/context filters, `table/json/csv` outputs, dataset ID/path exports, and optional training set/run joins (`--include-training`, `--trained-only`, `--set-id`).
  - Acceptance:
    - Filters for provenance completeness and context fields.
    - Optional training run joins for model/run lookup.

- [x] Reduce hard dependency on `zarr` for SQL-only registry commands.
  - Problem: registry CLIs can fail to start if `zarr` is unavailable.
  - File: `src/fisheye/registry/db.py`
  - Goal: lazy-import `zarr` only in scan/register paths.
  - Status: implemented with lazy `_import_zarr()` used in `scan_zarr`; SQL-only commands now import without `zarr`.
  - Acceptance:
    - `registry.status` and SQL query commands run without importing `zarr`.

- [x] Persist invocation metadata (CLI flags + environment fingerprint) for training sets/runs.
  - Problem: query filter captures normalized selection intent, but not full invocation context.
  - Files: `src/fisheye/utils/system.py`, `src/fisheye/diagnostics/prepare_detect_training.py`, `src/fisheye/registry/db.py`, `src/fisheye/training/train_detection.py`
  - Goal: make curation/training invocations auditable (`argv`, resolved args, git, environment summary, host/user).
  - Status: implemented via shared `build_invocation_record(...)`, persisted to manifest and registry JSON columns (`training_sets.invocation_json`, `training_runs.invocation_json`).
  - Acceptance:
    - `prepare_detect_training` manifest includes invocation block with raw args + runtime fingerprint.
    - `training_sets`/`training_runs` rows store invocation JSON when written.

## Priority 3 (Polish / Diagnostics)

- [x] Fix metadata display loop in detection training console output.
  - Problem: table print occurs outside loop, effectively showing only the last dataset.
  - File: `src/fisheye/training/train_detection.py`
  - Goal: print one table per dataset or a single consolidated table.
  - Status: implemented by printing each dataset table inside the loop and handling no-table cases explicitly.
  - Acceptance:
    - Multiple datasets produce complete, non-overwritten console summaries.

## Data Backfill Tasks

- [x] Backfill subject/protocol provenance where currently missing.
  - Tracked in dedicated `docs/provenance_backfill_todo.md` with full investigation plan.
  - Moved out of this TODO because it is a cross-cutting concern, not detection-specific.

- [x] Normalize/retire legacy source dataset IDs and prevent reintroduction.
  - Source recording rows now use canonical IDs: `{session_uuid}:z<path-hash>`.
  - Live cleanup completed; duplicate legacy source IDs were remapped and removed.
  - Guard added in registry identity resolution so rescans do not recreate `dataset_id=session_uuid`.

## Validation Checklist (after implementation)

- [x] `prepare_detect_training` + `train_detection` produce deterministic, source-faithful dataset composition.
  - Status: implemented via `GlobalIndexManager._build_global_index` with `balanced`/`proportional`/`weighted` strategies and source-type mismatch enforcement.
- [x] Registry tables in active DB have non-zero rows for:
  - `datasets`
  - `provenance`
  - `detection_sources`
  - `training_sets`
  - `training_runs`
  - `model_exports`
- [x] Query CLI can return filtered outputs for curation without ad-hoc SQL.
  - Status: `registry_query` supports genotype, DPF, cross-id, detect-coverage, provenance, and table/json/csv output formats.
- [x] Status report accurately reflects missing provenance and missing datasets.
  - Status: `check_training_registry` shows status with missing-provenance indicators and `--missing-provenance` filter in `registry_query`.
