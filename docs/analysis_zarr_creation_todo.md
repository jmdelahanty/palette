# Analysis Zarr Creation Split TODO

Purpose: track the implementation steps to split analysis archive creation/provenance from `detect_yolo`, while keeping migration-safe, operator-first behavior.

## Current state (2026-06-04)

- `detect_yolo` can create a minimal output archive when missing (legacy compatibility path).
- New module `fisheye.analysis.create_analysis_zarr` now exists with:
  - dry-run/apply behavior
  - explicit plan output
  - JSONL logging
  - fail-closed input resolution for ambiguous `cams/*.mp4` and `raw/*.h5`
  - archive create/ensure with `zarr_purpose=analysis`
  - automatic source-video metadata import in apply mode (with opt-out/overwrite flags)
  - optional stimulus import and optional registry scan
- Import-only helpers now exist for organized recordings:
  - `fisheye.utils.import_recording_analysis` creates/imports one recording
    analysis archive.
  - `fisheye.utils.import_organized_recordings_analysis` consumes an
    `organize_recordings` JSONL log and imports the selected organized
    recording directories without running detect/refine. When `--registry` is
    supplied, it scans successful imports and skipped existing analysis zarrs
    before reporting a recording complete.
- The older batch flow `import_recordings_analysis` still orchestrates the
  broader analysis pipeline:
  - YOLO detect
  - stimulus import
  - optional refine
  - optional registry rescan
- Its name is therefore broader than its current behavior; use
  `import_organized_recordings_analysis` for import-only work.
- Registry helper `fisheye.utils.resolve_detect_model` exists to rank/select candidate detect models by recording metadata similarity.
- Multi-camera recordings are currently fail-closed in this flow (intentional for now).

## Target state

- Dedicated module owns analysis archive creation and metadata policy.
- `detect_yolo` focuses on inference run writes only.
- Batch orchestration composes clear, testable steps.

## Phase A: Contract + scaffolding

- [x] Write creation contract doc.
  - File: `docs/analysis_zarr_creation_contract.md`
- [x] Add module skeleton: `src/fisheye/analysis/create_analysis_zarr.py`
  - Dry-run/apply
  - explicit plan output
  - JSONL logging

## Phase B: Creation module behavior

- [x] Implement input resolution rules.
  - `recording_dir` -> resolve `cams/*.mp4`, `raw/*.h5`
  - fail on ambiguous candidates
- [x] Implement archive ensure/create.
  - output naming convention for `*_analysis.zarr`
  - enforce `zarr_purpose=analysis`
- [x] Implement optional stimulus import integration.
  - call `fisheye.analysis.import_stimulus_to_zarr`
  - skip/always/overwrite controls
- [x] Implement optional registry registration.
  - rescan created/updated archive
- [x] Minor cleanup: use one shared `run_id` value for JSONL filename and JSON payload run metadata.
  - Status: implemented — `create_analysis_zarr.py` uses `run_id = _run_id()` for both log filename and `JsonLogger` payload.

## Phase C: Integrate orchestrator

- [x] Add an import-only organized-recording wrapper.
  - File: `src/fisheye/utils/import_organized_recordings_analysis.py`
  - Behavior: consumes an organize log, resolves recording directories, and runs
    the import step only.
- [x] Add immediate registry sync to the import-only wrapper.
  - `--registry` scans newly imported and skipped existing analysis zarrs.
  - Registry sync failures are counted as recording failures.
- [ ] Update or rename `import_recordings_analysis` so its pipeline behavior is
      explicit rather than implied to be import-only.
- [ ] Update `import_recordings_analysis` to call the shared creation/import
      step first when it remains the full-pipeline wrapper.
- [x] Keep detect/refine optional toggles.
- [x] Ensure per-step failure semantics are explicit and non-destructive.

## Phase D: Adjust `detect_yolo` boundary

- [ ] Add explicit behavior switch for archive creation.
  - transitional compatibility flag (temporary)
- [ ] Add preferred mode docs: run against pre-created analysis archive.
- [ ] Emit warning when compatibility mode creates archive directly.

## Phase E: Registry-driven model selection (separate but adjacent)

- [x] Add standalone registry model resolver helper.
  - File: `src/fisheye/utils/resolve_detect_model.py`
  - Current behavior: ranks models by recording metadata similarity and defaults to `--task detect`.
- [x] Add orchestrator option for model selection source.
  - `--model-source explicit|registry`
- [ ] For `registry` mode, add deterministic selection policy.
  - recommend required selector: `--set-id`
  - fail-closed if no model or multiple unresolved candidates
- [ ] Log resolved `run_id`/`model_path` in JSONL and detect provenance attrs.

## Phase F: Documentation and runbooks

- [x] Update workflow docs to reflect split architecture.
  - `docs/training_data_workflow.md`
  - `docs/detection_refinement_workflow.md`
- [x] Add explicit stage-orchestration workflow contract.
  - `docs/recording_analysis_pipeline_contract.md`
- [x] Document the import-only organized-recording path.
  - `docs/operator_guide/organize_recordings.md`
  - `docs/recording_analysis_pipeline_contract.md`
- [ ] Track and resolve spec/runtime drift items listed in:
  - `docs/zarr_spec_runtime_drift_todo.md`
- [x] Add operator runbook snippet:
  - create -> detect -> refine -> register -> integrity

## Testing checklist

- [x] Unit: creation planner (single camera, missing inputs, ambiguity cases).
- [x] Unit: dry-run does not mutate filesystem/registry.
  - Filesystem no-mutation tests:
    - `fisheye.utils.import_recording_analysis` (default dry-run path)
    - `fisheye.utils.run_recording_analysis_pipeline` (default dry-run path)
  - Registry no-mutation guards:
    - `fisheye.utils.run_recording_analysis_pipeline --register` (dry-run)
    - `fisheye.utils.import_recordings_analysis --register` (dry-run)
- [x] Unit: purpose attr enforcement.
- [x] Unit: orchestrator propagation of step failures.
  - single-recording pipeline returns `failed_step`/`returncode` on detect failure
  - batch orchestrator logs `recording_failed` with step + return code and returns non-zero
- [ ] Integration: end-to-end single recording happy path.
- [ ] Regression: existing `detect_yolo` CLI behavior remains supported during migration.
- [x] Unit: multi-camera/multi-H5 fail-closed behavior for analysis import planner.
- [x] Local smoke run (operator): `import_recordings_analysis --recursive` completed on `/nvme1/recordings` for current batch.
- [x] Local GoodCopBadCop import-only smoke exposed and fixed
  `timestamp_ns_epoch` metadata handling in stimulus interpolation.
- [ ] Integration: import-only organized-recording wrapper creates the expected
      analysis archives and stops before detect/refine.

## Rollout and risk controls

- [ ] Keep compatibility behavior for one release window.
- [ ] Add clear warning banners when legacy behavior path is used.
- [ ] Gate any removal of implicit creation behind passing integration coverage.
- [ ] Require backup + integrity checks for first operator rollout.

## Exit criteria

- [ ] Operators can create/validate analysis archives independently of inference.
- [ ] `detect_yolo` reliably appends inference runs to pre-created archives.
- [x] Batch analysis flow is deterministic and fail-closed (single-camera scope).
- [x] Import-only organized-recording flow exists for staging-to-analysis-Zarr
      bootstrap.
- [ ] Docs and tests reflect the split clearly.
