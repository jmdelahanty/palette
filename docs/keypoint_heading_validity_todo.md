# Keypoint Heading Validity Drop Plan (Breaking)

Purpose: remove ambiguous `heading_valid` semantics and replace them with
explicit heading quality fields.

Date anchored: 2026-02-11.

Related follow-up:

- [keypoint_temporal_heading_heuristic_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_temporal_heading_heuristic_todo.md)

## Status (2026-02-25)

- [x] Phase 1 complete: runtime cutover (`heading_finite` + `heading_usable` writes in all detection/refinement/manual/patch paths; `heading_valid` writes removed from all files).
- [x] Phase 2 complete: reader cutover (review/provenance checks read new fields only; `heading_valid` no longer read by any runtime module). `src/fisheye/docs/zarr_structure.md` updated.
- [x] Backfill utility implemented: `fisheye.utils.backfill_keypoint_heading_fields` exists with dry-run/apply, covered by `test_backfill_keypoint_heading_fields.py`.
- [x] Crimson contracts updated: `crimson_keypoint_read_contract.md` references `heading_finite`/`heading_usable`.
- [ ] Phase 3 remaining: run one-time backfill on production archives (utility exists but has not been executed against live data).
- [ ] Phase 4 remaining: verify `docs/keypoint_review_policy.md` and `src/fisheye/docs/provenance_workflow.md` no longer reference `heading_valid`.
  - Note: `heading_valid` still appears as local Python variable names in `eye_angle_analysis.py` and `track_kinematics.py` (not Zarr array writes — non-blocking).

## Decision

We will take a **breaking cleanup now**:

- Drop `heading_valid` from active write paths and docs.
- Replace with:
  - `heading_finite` (`bool`): strict `isfinite(heading)`.
  - `heading_usable` (`bool`): downstream-safe heading gate.

No compatibility fallback in runtime code. This is an intentional cutover.

## New Canonical Semantics

For keypoint run rows:

- `heading_finite = isfinite(heading)`
- `heading_usable = detection_success && detection_source==0 && heading_finite`

For refined keypoint run rows:

- `heading_finite = isfinite(heading)`
- `heading_usable = refined_success && detection_source==0 && heading_finite`

Notes:

- `heading` may be `NaN` for degenerate geometry.
- `heading_usable` is the only field that should gate downstream heading-based
  analytics/visualization.

## Affected Paths

### Remove `heading_valid` writes

- `src/fisheye/detection/detect_keypoints_yolo.py`
- `src/fisheye/detection/detect_keypoints_traditional.py`
- `src/fisheye/refinement/refine_keypoints.py`
- `src/fisheye/tune/keypoint_failure_review.py`
- `src/fisheye/tune/keypoint_tuner.py`
- `src/fisheye/utils/patch_keypoints_from_crops.py`

### Add `heading_finite` and `heading_usable` writes

Same files as above.

### Update readers/diagnostics

- `src/fisheye/tune/keypoint_review.py`
  - stop reading `heading_valid`; read/report `heading_finite` and
    `heading_usable`.
- `src/fisheye/diagnostics/check_full_provenance.py`
  - replace checks that assume `heading_valid` with `heading_usable`.

### Documentation updates

- `src/fisheye/docs/zarr_structure.md`
- `docs/keypoint_review_policy.md`
- `src/fisheye/docs/provenance_workflow.md` (if keypoint validity fields are
  mentioned)
- Crimson contracts:
  - `~/gitrepos/crimson/docs/crimson_keypoint_read_contract.md`
  - `~/gitrepos/crimson/docs/crimson_keypoint_manual_write_contract.md`
  - `~/gitrepos/crimson/docs/crimson_keypoint_review_acceptance_contract.md`

## Existing Archive Cleanup (One-Time)

Because we are intentionally breaking now, run a one-time archive migration on
existing runs:

1. For each `keypoints_runs/<run>`:
   - compute/write `heading_finite`, `heading_usable`
   - delete `heading_valid` array if present
2. For each `refined_keypoints_runs/<run>`:
   - compute/write `heading_finite`, `heading_usable`
   - delete `heading_valid` array if present

Implement as a dedicated utility (dry-run + apply), for example:

- `fisheye.utils.backfill_keypoint_heading_fields`

## Execution Phases

### Phase 1: Runtime schema cutover

- Write new arrays in detection/refinement/manual/patch paths.
- Remove all writes to `heading_valid`.

### Phase 2: Reader cutover

- Update review and diagnostics to use only new fields.
- Remove runtime reads of `heading_valid`.

### Phase 3: Data cleanup

- Run one-time backfill + delete for existing archives.
- Verify with spot checks and status tooling.

### Phase 4: Doc/contract closure

- Remove `heading_valid` from Palette docs/contracts.
- Ensure Crimson docs reflect only `heading_finite` / `heading_usable`.

## Test Plan

### Unit tests

- Detection writes:
  - `heading_finite` equals `isfinite(heading)`
  - `heading_usable` follows raw formula above
- Refinement writes:
  - same assertions with refined formula
- Manual correction path:
  - recompute updates both fields correctly
  - degenerate edits yield `heading_finite=false`, `heading_usable=false`
- Patch/update paths:
  - maintain consistency of new fields

### Migration utility tests

- Dry-run counts are deterministic.
- Apply mode writes new fields and deletes `heading_valid`.
- Idempotent re-run yields no further changes.

### Regression tests

- Keypoint review summaries still work after field swap.
- Provenance diagnostics report expected interpolation/heading behavior.

## Acceptance Criteria

- No runtime module reads/writes `heading_valid`.
- All new keypoint/refined runs contain `heading_finite` and `heading_usable`.
- Existing archive set is migrated and `heading_valid` removed.
- Crimson keypoint docs and Palette docs agree on heading semantics.
