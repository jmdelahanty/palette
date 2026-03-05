# Pose/Detect Parity Parallel Agent Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
-->

Purpose: define a conflict-free parallel execution plan for the current
Pose/Detect parity P2 tasks so multiple agents can work simultaneously and
land compatible changes.

For the next review-status unification phase, use:
`docs/review_status_schema_unification_contract.md`.

## Scope

In scope:
- Method-aware keypoint prerequisite gating by default.
- Optional keypoint/refine-keypoint orchestration in analysis pipeline wrappers.
- Real recording smoke execution + evidence capture for
  `validate_pose_detect_parity_smoke.sh`.

Out of scope:
- Eye-mask parity tasks.
- Registry schema migration changes.
- Unrelated docs or workflow refactors.

## Source Of Truth

- TODO tracker: `docs/pose_detect_parity_todo.md`
- Smoke runner: `scripts/validate_pose_detect_parity_smoke.sh`
- Pipeline wrappers:
  - `src/fisheye/utils/run_recording_analysis_pipeline.py`
  - `src/fisheye/utils/import_recordings_analysis.py`
- Keypoint batch entrypoint:
  - `src/fisheye/utils/run_keypoints_batch.py`

## Canonical Task IDs

- `P2-A`: method-aware prerequisite gating.
- `P2-B`: optional pose/refine_keypoints orchestration in analysis wrappers.
- `VAL-E2E`: execute real-recording parity smoke and capture evidence.

## Agent Ownership (Strict)

No cross-task edits outside owned files without explicit handoff.

### Agent A (`P2-A`: Gating)

Owns:
- `src/fisheye/utils/run_keypoints_batch.py`
- Any directly related keypoint prerequisite helper in the same module.
- `tests/unit/fisheye/test_run_keypoints_batch.py`

Must deliver:
- Default prerequisite behavior is method-aware:
  - Traditional keypoints: keep existing background prerequisite behavior.
  - YOLO pose: background not required unless explicitly requested.
- No regression in existing batch lifecycle behavior/logging.

### Agent B (`P2-B`: Pipeline Orchestration)

Owns:
- `src/fisheye/utils/run_recording_analysis_pipeline.py`
- `src/fisheye/utils/import_recordings_analysis.py`
- `tests/unit/fisheye/test_run_recording_analysis_pipeline.py`
- `tests/unit/fisheye/test_import_recordings_analysis.py`

Must deliver:
- Optional pose stage toggle in analysis wrappers.
- Optional refine-keypoints stage toggle in analysis wrappers.
- Explicit stage logging + failure semantics consistent with existing
  detect/refine-detect behavior.

### Agent C (`VAL-E2E`: Real Execution Evidence)

Owns:
- `scripts/validate_pose_detect_parity_smoke.sh` (only if execution blockers
  require minimal script fix).
- `docs/pose_detect_parity_todo.md` validation checklist updates.
- Optional evidence note file under `docs/` if needed.

Must deliver:
- Run smoke on at least one real recording path.
- Capture command, outcome, and artifact directory.
- Check off execution item in TODO when successful.

## Per-Agent Process Contract

Each agent follows this sequence:

1. Confirm owned files only.
2. Implement required behavior with minimal blast radius.
3. Add or update targeted tests.
4. Run targeted tests using `scripts/py -m pytest ...`.
5. Produce handoff note with exact commands and results.

Handoff note format:
- task id (`P2-A`, `P2-B`, or `VAL-E2E`)
- files touched
- behavior changes made
- tests/commands run
- result summary
- remaining risks

## Integration Contract

Integration order:
1. Agent A and Agent B can land in parallel.
2. Agent C runs after A+B merge (or on top of their combined branch).
3. Final pass updates `docs/pose_detect_parity_todo.md` checkboxes.

Conflict policy:
- If non-owned file edits are required, pause and request handoff.
- Do not reformat unrelated files.
- Do not modify registry migrations for these tasks.

## Validation Gates

Required:
- Targeted unit tests for touched modules pass.
- Existing parity scripts remain runnable.
- Real-recording smoke run result is captured and reproducible.

Recommended commands:

```bash
scripts/py -m pytest tests/unit/fisheye/test_run_keypoints_batch.py
scripts/py -m pytest tests/unit/fisheye/test_run_recording_analysis_pipeline.py
scripts/py -m pytest tests/unit/fisheye/test_import_recordings_analysis.py
```

```bash
scripts/validate_pose_detect_parity_smoke.sh \
  --recording-dir /nvme1/recordings/<recording_dir> \
  --registry /nvme1/palette_registry.sqlite
```
