# Recording Analysis Pipeline Contract

Purpose: define the canonical, operator-first contract for analysis processing per recording.

Date anchored: 2026-02-09.

## Goals

- Keep stage responsibilities explicit and composable.
- Keep execution migration-safe and fail-closed on ambiguous inputs.
- Support single-recording and batch orchestration with consistent behavior.

## Canonical Stage Tools

- Stage 1 import tool:
  - Module: `fisheye.utils.import_recording_analysis`
  - Responsibility: ensure `*_analysis.zarr`, import video metadata, import stimulus metadata.
  - Explicit non-goal: no detect or refine orchestration.
- Stage 2 detect tool:
  - Modules:
    - `fisheye.detection.detect_yolo` (explicit model path/config)
    - `fisheye.utils.run_detect_with_registry_model` (registry model resolution + detect run provenance)
  - Responsibility: append one detect run to analysis Zarr.
- Stage 3 refine tool:
  - Module: `fisheye.refinement.refine_detect`
  - Responsibility: append refined detect outputs; keep raw detect immutable.
- Stage 4 registry tool:
  - Module: `fisheye.registry.db.Registry.scan_zarr`
  - Responsibility: rescan/update registry metadata for the resulting analysis Zarr.

## Canonical Orchestrators

- Single recording orchestrator:
  - Module: `fisheye.utils.run_recording_analysis_pipeline`
  - Required execution order:
    1. import (`process_recording_import`)
    2. detect (`run_detect_yolo` or `run_detect_registry_model`)
    3. refine (`run_refine_detect`, optional)
    4. register (optional)
- Batch orchestrator:
  - Module: `fisheye.utils.import_recordings_analysis`
  - Behavior: resolve many recording plans, then run single-recording pipeline per plan.

## Stage Order Invariant

The required order is:

1. `import_recording_analysis`
2. detect
3. refine (optional)
4. register (optional)

Rationale: detect/refine should run against an archive that already has analysis purpose and imported metadata context.

## Failure Semantics

- Input resolution is fail-closed for ambiguous single-recording inputs:
  - multiple `cams/*.mp4` without explicit `--video`
  - multiple `raw/*.h5` without explicit `--h5`
- Single-recording orchestrator:
  - stop immediately on first failed stage
  - return non-zero
  - report `failed_step` and `returncode` where available
- Batch orchestrator:
  - continue to next recording when one recording fails
  - summarize `ok/failed/skipped/missing`
  - return non-zero if any recording failed

## Idempotency and Data Safety

- Import stage:
  - archive creation is idempotent (`mode="a"` when archive exists)
  - stimulus import defaults to skip when runs already exist unless `--stimulus-always`
- Detect stage:
  - append-only detect runs; existing runs remain immutable
- Refine stage:
  - append-only refined runs/manual groups; source detect runs remain immutable
- Registry stage:
  - rescan updates registry view of the archive path/metadata; does not require destructive rewrites

## Model Resolution Contract

- `--model-source explicit`:
  - detect uses explicit `--model` and/or detect config behavior.
- `--model-source registry`:
  - detect uses `run_detect_with_registry_model`
  - resolver currently targets `task=detect`
  - run provenance for selected model is written on the detect run attrs

## Logging Contract

- Batch pipeline writes JSONL logs unless `--no-log`.
- Expected high-level events:
  - `run_start`, `recording_plan`, `recording_start`
  - stage events (`video_metadata_imported`, `stimulus_result`, `detect_result`, `refine_result`)
  - terminal events (`recording_ok`, `recording_failed`, `recording_skipped`, `run_end`)

## Operator Runbook (Current)

- Single recording dry-run:
  - `scripts/py -m fisheye.utils.run_recording_analysis_pipeline --recording-dir "$REC" --dry-run`
- Single recording apply with registry model + register:
  - `scripts/py -m fisheye.utils.run_recording_analysis_pipeline --recording-dir "$REC" --model-source registry --registry /nvme1/palette_registry.sqlite --register --apply`
- Batch apply:
  - `scripts/py -m fisheye.utils.import_recordings_analysis /nvme1/recordings --recursive --model-source registry --registry /nvme1/palette_registry.sqlite --apply`

## Out of Scope (Current Contract)

- Multi-camera recordings in one recording directory are fail-closed in this workflow.
- 3D/multi-view analysis layout is tracked separately in `docs/multicamera_3d_analysis_todo.md`.
