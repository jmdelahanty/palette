<!-- ARCHIVED 2026-07-04: dated point-in-time snapshot / spent work ticket, retained for history only. -->

# Provenance Multi-Agent Handoff

Purpose: provide a shared, up-to-date execution plan for parallel agents
working on provenance contract migration and validation.

## Current Status

Completed provenance contract adoption:

- Refinement stages:
  - `refined_detect_runs`
  - `refined_keypoints_runs`
  - `refined_eye_masks_runs`
- Offline non-refinement stages:
  - `eye_masks_runs`
  - `keypoints_runs`
  - `detect_runs`
  - `crop_runs`
  - `id_assignment_runs`

Post-migration follow-up work:

- [x] Add `--zarr-use` filtering to diagnostics scanners where missing
- [x] Add targeted lineage repair tooling for stale refined-detect pointers
- [x] Resolve residual consistency mismatches in legacy training archives
- [x] Keep strict provenance checks enabled for migrated offline stages

## Canonical TODO IDs (for agent coordination)

T1: crop_runs contract migration (completed)  
T2: id_assignment_runs contract migration (completed)  
T3: legacy offline provenance backfill utility (completed)  
T4: offline diagnostics tightening (completed)  
T5: diagnostics `--zarr-use` filter parity (completed)  
T6: stale refined-detect lineage repair utility (completed)  
T7: residual consistency cleanup + revalidation (completed)

Deferred:

- `refine_online_detect` contract migration (explicitly deferred until offline
  provenance standardization is complete)

## Recent Milestones

- `fd4f8f8` Adopt stage provenance contract in refinement and capture checks
- `636c274` Add eye-mask stale resolution workflow for keypoint nudges
- `bec0feb` Adopt stage provenance contract for eye mask run writers
- `bb3f2d1` Adopt stage provenance contract for keypoint run writers
- `393971b` Adopt stage provenance contract for detect run writers

## Shared Contract

All migrated stages should write canonical:

- `attrs["provenance"]["contract"]["name"] == "palette_stage_provenance"`
- `attrs["provenance"]["contract"]["version"] >= 1`

Use shared helpers:

- `src/fisheye/shared/stage_provenance.py`
  - `build_stage_provenance(...)`
  - `write_stage_provenance(...)`
  - `get_stage_provenance(...)`
  - `get_stage_git(...)`
  - `get_stage_contract(...)`

Compatibility policy:

- Keep existing top-level convenience attrs (`git_commit`, `git_branch`) for
  shell visibility and legacy readers.

## Definition of Done (Contract Migration)

A stage migration is complete when:

1. `attrs["provenance"]["contract"]["name"] == "palette_stage_provenance"`
2. `attrs["provenance"]["contract"]["version"] >= 1`
3. Top-level `git_commit` and `git_branch` attrs remain present
4. Existing lineage fields (`source_*_run`) remain intact
5. `tests/unit/fisheye/test_check_provenance_capture.py` passes

## Canonical Stage Strings

All offline stage writers must use the following `stage` values in provenance:

- `detect_runs` -> `stage="detect"`
- `refined_detect_runs` -> `stage="refined_detect"`
- `crop_runs` -> `stage="crop"`
- `keypoints_runs` -> `stage="keypoints"`
- `refined_keypoints_runs` -> `stage="refined_keypoints"`
- `eye_masks_runs` -> `stage="eye_masks"`
- `refined_eye_masks_runs` -> `stage="refined_eye_masks"`
- `id_assignment_runs` -> `stage="id_assignment"`

These values must match what `check_provenance_capture` expects.

## Parallel Agent Tasks

### File Ownership Rules (Strict)

To prevent merge conflicts during parallel execution:

- T1 (crop migration) may modify:
  - `src/fisheye/tracking/crop.py` only
- T2 (id_assignment migration) may modify:
  - `src/fisheye/tracking/assign_ids.py` only
- T3 (backfill utility) may modify:
  - `src/fisheye/utils/backfill_stage_provenance.py`
  - its new unit tests only
- T4 (diagnostics tightening) may modify:
  - `src/fisheye/diagnostics/check_provenance_capture.py`
  - `tests/unit/fisheye/test_check_provenance_capture.py` only

No other files may be modified without parent-agent approval.

### Agent A: Crop Runs Migration

Goal: migrate `crop_runs` writer provenance to shared stage contract.

Primary files:

- `src/fisheye/tracking/crop.py`

Implementation notes:

- Replace ad-hoc `provenance` dict writes with `build_stage_provenance` +
  `write_stage_provenance`.
- Preserve existing source lineage attrs and crop signature attrs.
- Carry scheduler details into `provenance.scheduler`.
- Keep top-level attrs unchanged where possible.

Validation:

- `scripts/py -m pytest -q tests/unit/fisheye/test_check_provenance_capture.py`
- Existing crop-related tests impacted by provenance writes.
- If test updates are required, route them to T4/parent-agent approval per
  ownership rules.

### Agent B: ID Assignment Migration

Goal: migrate `id_assignment_runs` provenance to shared stage contract.

Primary files:

- `src/fisheye/tracking/assign_ids.py`

Implementation notes:

- Preserve source lineage (`source_detect_run`, `source_refined_run`) and any
  run summary fields.
- Use `stage="id_assignment"` (or current established stage value) consistently.

Validation:

- `scripts/py -m pytest -q tests/unit/fisheye/test_check_provenance_capture.py`
- Relevant ID-assignment unit tests.
- If test updates are required, route them to T4/parent-agent approval per
  ownership rules.

### Agent C: Legacy Backfill Utility

Goal: add a generic dry-run-first utility to normalize offline run provenance.

Suggested file:

- `src/fisheye/utils/backfill_stage_provenance.py`

Expected behavior (deterministic):

- `--dry-run` prints:
  - total runs scanned
  - count missing contract
  - count missing `git.commit`
  - first 5 paths that would be modified
- `--apply` performs minimal in-place updates only to:
  - `provenance.contract`
  - `provenance.git.commit`
- Never rewrite existing valid contract blocks.

Validation:

- New unit tests covering dry-run vs apply and fallback behavior.

### Agent D: Diagnostics Tightening

Goal: tighten provenance checks for migrated offline stages.

Primary files:

- `src/fisheye/diagnostics/check_provenance_capture.py`
- `tests/unit/fisheye/test_check_provenance_capture.py`

Implementation notes:

- Extend strict contract enforcement stage-by-stage after each migration/backfill.
- Preserve compatibility for non-migrated legacy runs until backfill is complete.
- Ensure clear missing-field reporting (`contract`, `contract.name`,
  `contract.version`).

## Dependency Order

Recommended order for parallel work merge:

1. Agent A (`crop_runs`) and Agent B (`id_assignment_runs`) can proceed in parallel.
2. Agent C backfill utility after both migrations are merged.
3. Agent D strict diagnostics tightening after backfill criteria are agreed.

## Important Documentation

Read first:

- `docs/provenance_todo.md`
- `docs/provenance_contract_draft.md`
- `docs/provenance_checks.md`
- `docs/pipeline_metadata_boundaries.md`

Workflow and lineage context:

- `src/fisheye/docs/provenance_workflow.md`
- `docs/recording_analysis_pipeline_contract.md`
- `docs/eye_mask_parity_todo.md`

## Working Conventions

- Use `scripts/py` for python/pytest commands.
- Keep changes non-destructive and append-only for run provenance semantics.
- Prefer targeted tests first; full zarr-heavy suites may hang in sandbox, so
  delegate those to local runs when needed.
- Keep stage name strings and contract fields consistent across writers and
  diagnostics.
