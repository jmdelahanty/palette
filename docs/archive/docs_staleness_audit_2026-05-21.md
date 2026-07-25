# Documentation Staleness Audit: 2026-05-21

## Scope

This targeted pass reviewed docs affected by the runtime contract-hardening
slice:

- Zarr run-completion markers
- `emit_stage_completion` root requirements
- stage-array validation shadow-mode
- registry `details_json` telemetry
- `report_stage_array_validation_shadow`

## Corrections Applied

- `docs/archive/stage_array_validation_audit_2026-05-21.md`
  - Recorded that rootless `ok` completion sync has been remediated for
    keypoint/pose and eye-mask batch wrappers.
  - Added the shadow report command.
  - Updated `keypoints` and `refined_keypoints` verdicts so rootless sync is no
    longer listed as an outstanding blocker.

- `docs/recording_step_status_parallel_agents_contract.md`
  - Updated the contract date and version.
  - Clarified that Zarr stage success paths should use
    `emit_stage_completion`, while `upsert_recording_step_status` remains the
    lower-level ledger writer.
  - Documented the hard `ok` run-completion requirement and the shadow-mode
    array-validation fields written into `details_json`.
  - Added focused completion tests and the shadow report command to validation
    expectations.

- `src/fisheye/docs/zarr_structure.md`
  - Added `latest_complete` and per-run completion-marker attrs to the common
    `*_runs` description.
  - Documented the difference between hard completion-marker validation and
    shadow-mode array validation.

- `docs/zarr_run_completion_contract.md`
  - Updated the verification date.
  - Added the registry completion rule: `emit_stage_completion` must receive a
    readable root for `ok` run writes, resolves the named run group, and refuses
    incomplete completion markers.
  - Added the relationship between completion-marker validation and shadow-mode
    stage-array validation.

- `docs/keypoints_pipeline_inline_registry_report.md`
  - Marked the previous shared-helper section as historical.
  - Pointed readers at the implemented helper in
    `src/fisheye/registry/stage_complete.py`.

- `docs/stage_catalog_design.md`
  - Added the boundary between registry/status stage vocabulary and
    `StageSpec` Zarr array contracts.
  - Added the shadow report command as the pre-enforcement check.

## Remaining Deferred Work

- `recording_step_status_parallel_agents_contract.md` is still a candidate for
  archive or rename into a slimmer runtime ledger contract, as noted by the
  2026-05-20 master docs audit. This pass kept it active and current because it
  remains the most discoverable status-write contract.
- Hard stage-array enforcement is still intentionally deferred until candidate
  stages have real-run smoke evidence.
- Method-family contracts for `keypoints` and `eye_masks` remain unresolved;
  docs now call out that these are not safe for hard array enforcement yet.

## Validation

Commands run:

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m py_compile \
  src/fisheye/utils/report_stage_array_validation_shadow.py

scripts/py -m fisheye.utils.report_stage_array_validation_shadow --help

git diff --check
```

Result: all passed.
