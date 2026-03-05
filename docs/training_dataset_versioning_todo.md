# Training Dataset Versioning TODO

Purpose: track the work needed to make training dataset versioning strict, auditable, and reproducible.

## Policy Baseline

- Dataset versions are immutable.
- Any data edit creates a new dataset version (new `set_id`/`set_version`, manifest, and merged zarr path).
- Persist summary metadata at export time, but keep read-time fallback for compatibility with legacy or partial archives.

## Priority 0 (Versioning Enforcement)

- [ ] Enforce immutable version behavior in build/export commands.
  - Files: `src/fisheye/diagnostics/prepare_detect_training.py`, `src/fisheye/utils/export_detect_training_zarr.py`
  - Goal: prevent accidental in-place mutation of previously registered dataset versions.
  - Acceptance: rerunning export for an existing version requires explicit new versioning flow (or explicit override with strong warning).

- [ ] Add training set lineage fields.
  - Files: `src/fisheye/registry/db.py`, `src/fisheye/registry/query.py`
  - Goal: encode ancestry between versions (e.g., `supersedes_set_id` / `parent_set_id`).
  - Acceptance: each new version can be traced to its predecessor via SQL query and CLI output.

- [x] Record run lifecycle state in registry (`in_progress` -> `success`/`failed`).
  - Files: `src/fisheye/training/train_detection.py`, `src/fisheye/registry/db.py`
  - Status: implemented with stable `run_id` updates and failure-state writes.
  - Acceptance: active runs are visible without tailing trainer logs.

## Priority 1 (Metadata Correctness)

- [ ] Persist explicit crop summary stats for merged exports.
  - Files: `src/fisheye/utils/export_detect_training_zarr.py`
  - Goal: write stable `total_rois`/frame counts where downstream tools already look for them.
  - Acceptance: training metadata table shows non-zero counts without requiring fallback logic.

- [x] Add robust fallback metadata reads in trainer display path.
  - Files: `src/fisheye/training/train_detection.py`
  - Goal: if summary attrs are absent, derive counts from arrays (`bbox_norm_coords.shape[0]`, `images_ds.shape[0]`).
  - Status: implemented — `train_detection.py` falls back to `bbox_norm_coords.shape[0]` when summary attrs are absent.

- [ ] Add drift warning between persisted metadata and derived values.
  - Files: `src/fisheye/training/train_detection.py` (read path), `src/fisheye/utils/validate_detect_training_zarr.py` (validator)
  - Goal: surface silent edits when metadata no longer matches data arrays.
  - Acceptance: warning emitted with both values when mismatch is detected.

## Priority 2 (Auditability / Ops)

- [ ] Add dataset fingerprint to merged export metadata.
  - Files: `src/fisheye/utils/export_detect_training_zarr.py`
  - Goal: persist a deterministic hash over source dataset IDs, split indices, and critical config knobs.
  - Acceptance: same inputs produce same fingerprint; changes produce a different fingerprint.

- [ ] Add registry helper query for “active training status + latest versions”.
  - Files: `src/fisheye/registry/query.py` and/or `src/fisheye/utils/check_training_registry.py`
  - Goal: one command to inspect current run statuses and recent dataset versions.
  - Acceptance: output includes `run_id`, `set_id`, `status`, timestamps, and lineage reference.

- [ ] Document the immutable-version workflow in user-facing docs.
  - Files: `docs/training_data_workflow.md`, `docs/detection_merged_export_contract.md`
  - Goal: make operational behavior explicit for new users.
  - Acceptance: docs include “edit -> new version” examples and expected registry records.

## Validation Checklist

- [ ] Creating a new version never mutates prior version artifacts in place.
- [x] Registry can show current run state (`in_progress`/`success`/`failed`) for active training.
  - Status: implemented in `check_training_registry` with run lifecycle state display.
- [ ] Trainer metadata display is correct for merged datasets (non-zero frame/ROI counts).
- [ ] Lineage query can explain how a version was derived from previous versions.
- [ ] Fingerprint and manifest hashes are present for each registered training set/run.
