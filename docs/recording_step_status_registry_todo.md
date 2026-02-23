# Recording Step Status Registry TODO

Purpose: move recording step/status reporting from filesystem traversal
(`check_recording_steps`) into first-class registry tables/views so operators can
query status directly from SQLite.

## Goal

- Make status checks for recording pipelines queryable from registry:
  - raw/background presence
  - detect/refined-detect/crop/keypoints/refined-keypoints/eye-masks/refined-eye-masks
  - id assignment + track presence
  - stimulus/calibration presence
  - tuning key coverage (`dish_mask`, `detection_tuning`, `keypoint_tuning`,
    `eye_mask_tuning`, `subdish_mask_tuning`)
- Keep parity with existing `check_recording_steps` output while enabling
  fast registry views and filters.

## Current Gap Summary

- `check_recording_steps` derives status directly from Zarr attrs/groups and
  reports rich per-recording state.
- Registry already stores partial quality/performance state for detect/keypoint/eye-mask,
  but does not currently store the full recording step-status surface.
- Result: operators must traverse directories/Zarrs to answer status questions
  not covered by current registry views.

## Parallel Work Model

Use canonical task IDs with strict file ownership to minimize conflicts.
Parallel execution contract for RS3/RS4/RS7:
`docs/recording_step_status_parallel_agents_contract.md`.

### RS1: Schema + Views (Foundational)

- [x] Add schema migration for step-status ledger.
- [x] Add `recording_step_status` table (latest snapshot per dataset/step).
- [x] Add `recording_step_status_history` table (append-only event log, optional but preferred).
- [x] Add convenience views:
  - `recording_step_status_latest`
  - `recording_step_overview` (one row per recording with step aggregates)
- [x] Add indexes for `(recording_id, step_name)`, `(dataset_id, step_name)`, `(status)`.

Acceptance:

- New schema applies cleanly on empty DB and upgrades existing DB.
- `recording_step_overview` answers "which recordings are missing step X?" without Zarr reads.

Suggested owner files:

- `src/fisheye/registry/db.py`
- `tests/unit/fisheye/test_registry_maintenance.py` (or new targeted registry view test)

### RS2: Shared Status Writer API

- [x] Add shared helper for status upserts/event writes.
- [x] Resolve dataset/recording keys from zarr path using registry tables.
- [x] Standardize status enum:
  - `ok`, `missing`, `absent`, `na`, `error`
- [x] Standardize payload fields:
  - `step_name`, `status`, `run_name`, `method`, `coverage_pct`,
    `review_status_json`, `details_json`, `updated_utc`, `source`

Acceptance:

- Single helper used by multiple pipeline stages.
- Idempotent upsert behavior for repeated writes.

Suggested owner files:

- `src/fisheye/registry/status_ledger.py` (new)
- `tests/unit/fisheye/test_registry_status_ledger.py` (new)

### RS3: Detect/Crop Branch Write Hooks

- [x] Hook status writes into detect/crop-related producers:
  - detect inference completion
  - refined-detect completion
  - crop completion
- [x] Capture review/coverage fields where available.

Acceptance:

- Running detect/refine/crop updates registry step status without maintenance backfill.

Suggested owner files:

- `src/fisheye/inference/predict_detections.py`
- `src/fisheye/refinement/refine_detect.py`
- `src/fisheye/tracking/crop.py`

### RS4: Pose/Eye/Tracking Branch Write Hooks

- [x] Hook status writes into:
  - keypoint inference
  - refined keypoints
  - eye-mask inference
  - refined eye masks
  - id assignment / track (where applicable)
- [x] Capture review/quality pointers in `details_json` for query joins.

Acceptance:

- Running pose/eye/tracking steps updates registry status immediately.

Suggested owner files:

- `src/fisheye/inference/predict_pose.py`
- `src/fisheye/refinement/refine_keypoints.py`
- `src/fisheye/inference/predict_eye_masks.py`
- `src/fisheye/refinement/refine_eye_masks.py`
- `src/fisheye/tracking/assign_ids.py`

### RS5: Backfill + Reconcile Command

- [x] Add maintenance command to backfill step status from existing Zarrs.
- [x] Support `--dry-run`, `--apply`, and scoped filters (`--recording-id`,
  `--zarr-use`, path prefix).
- [x] Emit machine-readable summary counts.

Acceptance:

- One command populates status rows for historical archives.
- Re-running backfill is safe and convergent.

Suggested owner files:

- `src/fisheye/registry/maintenance.py`
- `tests/unit/fisheye/test_registry_maintenance.py`

### RS6: Read Path + Operator UX

- [x] Add registry viewer output for step status overview.
- [x] Add registry query filters for step status predicates.
- [x] Add dual-source mode to `check_recording_steps`:
  - `--status-source filesystem|registry|compare`
- [x] Add parity report mode to compare filesystem-derived vs registry-derived status.

Acceptance:

- Operators can inspect step status with `check_training_registry` and/or
  registry SQL views without traversing recording directories.
- Compare mode reports mismatches and supports rollout validation.

Suggested owner files:

- `src/fisheye/utils/check_training_registry.py`
- `src/fisheye/utils/registry_query.py`
- `src/fisheye/utils/check_recording_steps.py`
- `tests/unit/fisheye/test_check_training_registry.py`
- `tests/unit/fisheye/test_registry_query.py`
- `tests/unit/fisheye/test_check_recording_steps.py`

### RS7: Validation Harness + Smoke Script

- [x] Add scripted validation for one recording end-to-end:
  - run step(s)
  - verify registry status update
  - verify compare parity
- [x] Add deterministic acceptance checks for expected rows/views.

Acceptance:

- Script exits non-zero on missing status rows or parity mismatches.
- Script is documented and runnable by operators.

Suggested owner files:

- `scripts/validate_recording_step_status_registry.sh` (new)
- `docs/recording_step_status_registry_todo.md` (validation section updates)

Operator run example:

```bash
scripts/validate_recording_step_status_registry.sh \
  --recording-dir /nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen \
  --registry /nvme1/palette_registry.sqlite \
  --zarr-use training
```

## Task Dependencies

- RS1 must land before RS2/RS5/RS6.
- RS2 must land before RS3/RS4.
- RS5 can run in parallel with RS3/RS4 once RS1 exists.
- RS6 can start after RS1 (read views first), then finalize after RS3/RS4/RS5.
- RS7 runs throughout but final pass requires RS3-RS6.

## Parallel Agent Ownership Rules

- Agent A owns RS1 files only.
- Agent B owns RS2 files only.
- Agent C owns RS3 files only.
- Agent D owns RS4 files only.
- Agent E owns RS5 files only.
- Agent F owns RS6 files only.
- Agent G owns RS7 script/docs only.

No cross-task file edits without explicit handoff.

## Rollout Strategy

- Phase 0: dual-write disabled, backfill only (safe dry-run + apply).
- Phase 1: enable write hooks (RS3/RS4) while keeping filesystem checks as source of truth.
- Phase 2: use `--status-source compare` to prove parity on active recordings.
- Phase 3: switch default operator workflows to registry views/queries.
- Phase 4: keep filesystem mode as diagnostic fallback.

## Definition Of Done

- [x] Backfill completes on target recordings with expected row counts.
- [x] New pipeline runs update step status in registry automatically.
- [x] `check_training_registry` can show recording step status directly.
- [x] `check_recording_steps --status-source compare` reports no mismatches for validation set.
- [x] Unit test suite passes for touched modules.
