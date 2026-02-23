# Recording Step Status Parallel Agent Contract

Purpose: define a conflict-free parallel execution plan for RS3/RS4 hook-write
work so multiple agents can implement status writes simultaneously and produce
compatible outputs.

## Scope

In scope:
- RS3 hook writes for detect/crop branch producers.
- RS4 hook writes for pose/eye/tracking branch producers.
- Shared runtime contract for writes via `upsert_recording_step_status(...)`.
- Targeted tests and a deterministic handoff format per agent.

Out of scope:
- Schema migrations and registry view design (already completed).
- Maintenance backfill logic as primary data path (already completed).

## Source Of Truth

- API: `src/fisheye/registry/status_ledger.py`
- TODO tracker: `docs/recording_step_status_registry_todo.md`
- Current parity oracle: `scripts/py -m fisheye.utils.check_recording_steps --status-source compare`

## Shared Runtime Write Contract

All producers must write through:
- `fisheye.registry.status_ledger.upsert_recording_step_status`

Required fields on every write:
- `dataset_id`
- `step_name`
- `status`
- `source`

Standardized status enum:
- `ok`
- `missing`
- `absent`
- `na`
- `error`

Standardized step names:
- `detect`
- `refined_detect`
- `crop`
- `keypoints`
- `refined_keypoints`
- `eye_masks`
- `refined_eye_masks`
- `id_assignment`
- `tracks`

Write behavior requirements:
- Idempotent: repeated writes for the same `(dataset_id, step_name)` must update
  one latest row and append one history row.
- Fail-closed on invalid status values (propagate `ValueError` from writer API).
- Do not silently skip writes when required run context is available.

Payload conventions:
- `run_name`: concrete run/group identifier when present.
- `method`: pipeline method identifier when present.
- `coverage_pct`: normalized float percentage when present.
- `review_status_json`: JSON payload for review state when available.
- `details_json`: JSON payload for step-specific provenance/quality pointers.
- `zarr_mtime_ns`: write when cheaply available from the current artifact path.
- `source`: stable runtime source token in the form
  `runtime_<module_or_cli_name>`.

## Agent Ownership (Strict)

No cross-task edits outside owned files without explicit handoff.

### Agent A (RS3-detect)

Owns:
- `src/fisheye/inference/predict_detections.py`
- `src/fisheye/refinement/refine_detect.py`
- detect/refine-detect unit tests

Must emit:
- `step_name=detect` and `step_name=refined_detect`
- review/coverage/quality pointers when available

### Agent B (RS3-crop)

Owns:
- `src/fisheye/tracking/crop.py`
- crop-related unit tests

Must emit:
- `step_name=crop`
- review/coverage fields where available

### Agent C (RS4-pose)

Owns:
- `src/fisheye/inference/predict_pose.py`
- `src/fisheye/refinement/refine_keypoints.py`
- pose/refine-keypoints unit tests

Must emit:
- `step_name=keypoints` and `step_name=refined_keypoints`

### Agent D (RS4-eye)

Owns:
- `src/fisheye/inference/predict_eye_masks.py`
- `src/fisheye/refinement/refine_eye_masks.py`
- eye-mask/refine-eye unit tests

Must emit:
- `step_name=eye_masks` and `step_name=refined_eye_masks`

### Agent E (RS4-tracking)

Owns:
- `src/fisheye/tracking/assign_ids.py`
- tracking/id-assignment unit tests

Must emit:
- `step_name=id_assignment`
- `step_name=tracks` where track materialization state is known

### Agent F (Validation/Harness)

Owns:
- `scripts/validate_recording_step_status_registry.sh` (new)
- validation docs updates for RS7
- validation-focused tests (if added)

Must verify:
- runtime writes exist without running maintenance backfill
- compare mode reports no mismatches on validation targets

## Per-Agent Process Contract

Each agent follows this exact sequence:

1. Confirm owned files only.
2. Add runtime write calls at success and relevant non-success decision points.
3. Preserve existing behavior and CLI output semantics.
4. Add/adjust targeted unit tests for new write behavior.
5. Run targeted tests with `scripts/py -m pytest ...`.
6. Produce handoff note using the format below.

Handoff note format:
- files touched
- statuses/steps written
- tests run (exact command)
- result summary
- known limitations

## Integration Contract

Integration order:
1. Agent A, B, C, D, E can land in parallel if file ownership is respected.
2. Agent F lands after at least one producer hook set is merged.
3. Final pass updates `docs/recording_step_status_registry_todo.md` checkboxes.

Conflict policy:
- If an agent must touch a non-owned file, pause and request handoff.
- Do not reformat unrelated files.
- Do not modify schema migrations for RS3/RS4 work.

## Validation Gates (Definition Of Done For RS3/RS4/RS7)

Required:
- Targeted unit tests for touched modules pass.
- `check_recording_steps --status-source compare` passes on validation recording(s)
  without needing a fresh maintenance backfill for newly produced runs.
- `check_training_registry --view recording-steps` reflects runtime updates.

Recommended smoke commands:

```bash
scripts/py -m pytest tests/unit/fisheye/test_check_recording_steps.py
scripts/py -m pytest tests/unit/fisheye/test_check_training_registry.py
scripts/py -m pytest tests/unit/fisheye/test_registry_status_ledger.py
```

```bash
scripts/py -m fisheye.utils.check_recording_steps \
  /nvme1/recordings \
  --recursive \
  --zarr-use training \
  --registry /nvme1/palette_registry.sqlite \
  --status-source compare
```

