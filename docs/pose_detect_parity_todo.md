# Pose/Detect Workflow Parity TODO

## Goal
Bring pose/keypoint workflow parity with detect workflow for:
- provenance durability
- computational argument/runtime tracking
- registry status/query/selection ergonomics

Related eye-mask parity work is tracked separately in `docs/eye_masks_detect_pose_parity_todo.md`.

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

- [ ] Make keypoint prerequisite gating method-aware by default.
  - For YOLO pose, background should not be required unless explicitly requested.
  - Keep traditional behavior unchanged.
  - Acceptance: default gates match actual method dependencies.

- [ ] Add optional pose stage orchestration in analysis pipeline wrappers.
  - Current recording analysis pipeline orchestrates import -> detect -> refine_detect -> register.
  - Add optional keypoint + refine_keypoints stage toggles with logging.
  - Acceptance: one pipeline command can run full analysis stack when requested.

- [ ] Unify review-status schema shape across detect/keypoint where practical.
  - Align timestamp field naming and shared status fields.
  - Keep keypoint signature support; consider detect signature equivalent.
  - Acceptance: easier shared validation and fewer one-off parsers.

## Validation/Testing TODO

- [x] Unit tests for new pose registry model resolution wrapper.
- [x] Unit tests for strict keypoint accept CLI.
- [x] Unit tests for unapproved keypoint lister.
- [x] Registry migration tests for keypoint performance schema/views.
- [x] Registry query tests for new keypoint filters/group summaries.
- [x] Batch logging tests to assert richer keypoint result payloads.
- [ ] End-to-end smoke test: analysis zarr -> keypoints -> refine_keypoints -> registry scan -> quality/perf query.
  - [x] Added operator runner script: `scripts/validate_pose_detect_parity_smoke.sh`.
  - [ ] Execute on real registry recording and capture artifact output.
