# Pose/Detect Workflow Parity TODO

## Goal
Bring pose/keypoint workflow parity with detect workflow for:
- provenance durability
- computational argument/runtime tracking
- registry status/query/selection ergonomics

Related eye-mask parity work is tracked separately in `docs/eye_masks_detect_pose_parity_todo.md`.
Parallel execution contract for current P2 + validation tasks:
`docs/pose_detect_parity_parallel_agents_contract.md`.

## P0 (Highest Priority)

- [x] Add registry-resolved pose inference wrapper (detect parity).
  - Add a pose equivalent to `run_detect_with_registry_model.py`.
  - Resolve pose model from registry using recording metadata similarity.
  - Write `model_resolution_*` attrs and `provenance.model_resolution` on keypoint runs.
  - Acceptance: pose runs show selected `run_id`, `set_id`, `model_path`, candidate payload in attrs/provenance.

- [x] Add strict keypoint review acceptance utility.
  - Add a keypoint equivalent to `accept_detect_review.py` with strict mode guardrails.
  - Enforce required fields in strict mode (at minimum reviewer + intended_use rules).
  - Acceptance: CLI supports dry-run/json and updates `keypoint_review_status_latest` safely.

- [x] Add keypoint unapproved-list utility.
  - Add a keypoint equivalent to `list_unapproved_analysis_zarrs.py`.
  - Report missing/non-approved keypoint review state for analysis zarrs.
  - Acceptance: produces path list + optional details output for batch workflows.

## P1 (High Priority)

- [x] Add keypoint performance registry table + latest views.
  - Detect has `detect_performance`; add analogous `keypoint_performance`.
  - Track runtime and throughput metrics from keypoint runs (both traditional and YOLO).
  - Acceptance: registry migrations create table/views; rescan populates rows.

- [x] Extend registry query CLI for keypoint quality/performance filters.
  - Add keypoint-focused filters to `registry_query.py` (review state/use, usable rate, method, runtime metrics).
  - Add group summaries for keypoint model/method similar to detect summaries.
  - Acceptance: one CLI can query both detect and keypoint status/performance slices.

- [x] Improve `run_keypoints_batch.py` result logging parity.
  - Detect batch logs a richer `results` payload; keypoints currently logs minimal success/fail.
  - Include run name, method, optional refine run, and core metrics in JSONL output.
  - Acceptance: keypoint JSONL supports downstream auditing/reconciliation like detect batch logs.

## P2 (Medium Priority)

- [x] Make keypoint prerequisite gating method-aware by default.
  - For YOLO pose, background should not be required unless explicitly requested.
  - Keep traditional behavior unchanged.
  - Acceptance: default gates match actual method dependencies.

- [x] Add optional pose stage orchestration in analysis pipeline wrappers.
  - Current recording analysis pipeline orchestrates import -> detect -> refine_detect -> register.
  - Add optional keypoint + refine_keypoints stage toggles with logging.
  - Acceptance: one pipeline command can run full analysis stack when requested.

- [x] Unify review-status schema shape across detect/keypoint where practical.
  - Canonical contract/spec: `docs/review_status_schema_unification_contract.md`.
  - Write canonical `timestamp_utc` in detect/keypoint review writers.
  - Keep backward-read compatibility for legacy timestamp keys (`timestamp`, `reviewed_at_utc`, `reviewed_at`).
  - Align shared fields across detect/keypoint quality surfaces (state/method/intended_use/reviewer/notes/timestamp_utc).
  - Keep keypoint signature support; decide whether detect gets a parity signature.
  - Acceptance:
    - New detect/keypoint review writes include canonical `timestamp_utc`.
    - Registry detect/keypoint quality views expose aligned shared review columns.
    - Query/consumer code can read aligned fields without ad hoc per-modality parsing.

  Subtasks:
  - [x] Writer normalization pass (`accept_detect_review.py`, `accept_keypoint_review.py`, `set_keypoint_review_status.py`).
  - [x] Registry schema/view alignment pass (`detect_quality`/`keypoint_quality` extraction + upsert + current views).
  - [x] Consumer/query alignment pass (`registry_query.py` and related reporting/selection paths).

## Validation/Testing TODO

- [x] Unit tests for new pose registry model resolution wrapper.
- [x] Unit tests for strict keypoint accept CLI.
- [x] Unit tests for unapproved keypoint lister.
- [x] Registry migration tests for keypoint performance schema/views.
- [x] Registry query tests for new keypoint filters/group summaries.
- [x] Batch logging tests to assert richer keypoint result payloads.
- [x] Unit tests for optional keypoints/refine_keypoints orchestration in analysis pipeline wrappers.
- [x] Unit tests for canonical review payload writing (`timestamp_utc`) in detect/keypoint review CLIs.
- [x] Registry tests for detect/keypoint shared review-column parity in quality views.
- [x] Query/consumer tests for aligned detect/keypoint review-field access.
- [x] End-to-end smoke test: analysis zarr -> keypoints -> refine_keypoints -> registry scan -> quality/perf query.
  - [x] Added operator runner script: `scripts/validate_pose_detect_parity_smoke.sh`.
  - [x] Execute on real registry recording and capture artifact output.
    - Artifact: `/tmp/pose_parity_smoke_20260222_233222` (`Smoke test passed.`).

## Post-Completion Follow-Ups (2026-02-23 Parity Audit)

- [x] Add keypoint-performance maintenance parity with detect/eye-mask.
  - Add maintenance flags for keypoint performance refresh/backfill (dry-run + apply).
  - Suggested flags:
    - `--backfill-keypoint-performance`
    - `--refresh-keypoint-performance`
    - `--keypoint-performance-all-datasets`.
  - Acceptance:
    - historical keypoint-performance rows can be reconciled without relying on full re-register scans,
    - maintenance output reports deterministic inserted/updated/deleted/unchanged counts.
