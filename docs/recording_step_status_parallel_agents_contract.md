# Recording Step Status Parallel Agent Contract
<!-- contract-meta
version: 2
status: superseded
implementation: implemented
last_verified: 2026-05-21
superseded_date: 2026-08-20
superseded_by: docs/current_pipeline_contract.md and repository AGENTS.md
-->

Purpose: define a conflict-free parallel execution plan for RS3/RS4 hook-write
work so multiple agents can implement status writes simultaneously and produce
compatible outputs.

This is a historical implementation-coordination record. Its agent ownership
map and eye-mask-first stage list are no longer normative; current production
direction lives in `docs/current_pipeline_contract.md` and `AGENTS.md`.

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

Zarr stage producers should write through:
- `fisheye.registry.stage_complete.emit_stage_completion`

Low-level registry callers that do not own a Zarr run group may write through:
- `fisheye.registry.status_ledger.upsert_recording_step_status`

`emit_stage_completion` is preferred for pipeline stage writers because it
combines dataset row upsert, run-completion verification, shadow array
validation telemetry, latest status write, history append, and downstream
invalidation.

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
- `arena_assignment`
- `tracks`

Write behavior requirements:
- Idempotent: repeated writes for the same `(dataset_id, step_name)` must update
  one latest row and append one history row.
- Fail-closed on invalid status values (propagate `ValueError` from writer API).
- Do not silently skip writes when required run context is available.
- For `status="ok"` with `run_name`, pass a mutable/readable Zarr root to
  `emit_stage_completion`. The helper refuses to mark the stage complete when
  it cannot resolve the run group or when the run-completion marker is not
  complete.
- Non-`ok` statuses may omit the Zarr root when prebuilt dataset metadata is
  supplied; they intentionally bypass run-group validation.
- Do not call `upsert_recording_step_status` directly from a Zarr stage success
  path unless there is a documented reason the completion marker cannot apply.

Payload conventions:
- `run_name`: concrete run/group identifier when present.
- `method`: pipeline method identifier when present.
- `coverage_pct`: normalized float percentage when present.
- `review_status_json`: JSON payload for review state when available.
- `details_json`: JSON payload for step-specific provenance/quality pointers.
- `zarr_mtime_ns`: write when cheaply available from the current artifact path.
- `source`: stable runtime source token in the form
  `runtime_<module_or_cli_name>`.

`emit_stage_completion` adds these validation telemetry fields to
`details_json` when a run is marked `ok`:

- `stage_array_validation_status`: `ok`, `invalid`, or `no_spec`
- `stage_array_validation_stage`: canonical array-contract stage name, when known
- `stage_array_validation_enforced`: whether array validation was hard-enforced
- `stage_array_validation_errors`: required-array failures, when present
- `stage_array_validation_warnings`: optional-array or unknown-spec warnings

Array validation is currently hard-enforced for `detect_quality` and
shadow-mode for every other stage: the validator records what would fail, but
only stages listed in
`src/fisheye/registry/stage_complete.py::_ENFORCE_STAGE_ARRAY_VALIDATION_FOR`
block registry completion on array-contract failures. Completion-marker
validation is always hard for `ok` run writes.

Shadow telemetry report:

```bash
scripts/py -m fisheye.utils.report_stage_array_validation_shadow \
  --registry /nvme1/palette_registry.sqlite \
  --include-no-spec
```

The report command exits zero by default. Use `--fail-on-match` only when the
selected validation statuses should fail an explicit gate.

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
- `src/fisheye/tracking/arena_assignment.py`
- tracking/id-assignment unit tests

Must emit:
- `step_name=arena_assignment`
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
- Focused completion-contract tests pass when touching stage status writers:
  `tests/unit/fisheye/test_zarr_run_completion.py` and
  `tests/unit/fisheye/test_stage_completion_rooted_wrappers.py`.
- `check_recording_steps --status-source compare` passes on validation recording(s)
  without needing a fresh maintenance backfill for newly produced runs.
- `check_training_registry --view recording-steps` reflects runtime updates.
- Shadow report is reviewed after real pipeline runs when enabling new
  StageSpec enforcement candidates.

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
