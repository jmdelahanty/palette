# Keypoint Training Refined-Run Tie Fix TODO

Purpose: capture the code fixes needed to stop `prepare_keypoint_training_from_registry`
from failing on stale `keypoint_quality_current.refined_run` mismatches after
`traditional_v2` migration runs were added.

## Problem Summary

Current failure mode:

- `prepare_keypoint_training_from_registry.py` resolves the refined keypoint run
  for a given `source_keypoints_run` by sorting only on `created_utc`.
- `keypoint_quality_current` in the registry picks the current refined run using
  a stronger ordering:
  1. `review_timestamp_utc`
  2. `refined_created_utc`
  3. `quality_updated_utc`
  4. `refined_run DESC`
- Many migrated `traditional_v2_seed` refined runs inherited the same
  `created_utc` as their older sibling refined runs, so preflight can resolve
  the old run while the registry points at the migrated run.
- Result: training preflight raises
  `stale keypoint_quality row: expected refined_run ..., observed ...`.

Operational mitigation now exists:

- [repair_keypoint_training_refined_run_ties.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/repair_keypoint_training_refined_run_ties.py)
- documented in [keypoint_training_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_training_workflow.md)

That repair is a workaround, not the real fix.

## Patch 1: Align Preflight Tie-Break Logic With Registry

Target:
- [prepare_keypoint_training_from_registry.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/prepare_keypoint_training_from_registry.py)

Current problematic function:
- `_resolve_refined_keypoint_quality(...)`

Current behavior:
- scans refined keypoint runs matching `source_keypoints_run`
- sorts candidates only by parsed `created_utc`

Required change:
- use a deterministic ordering that matches the registry current-view semantics
  closely enough that preflight and `keypoint_quality_current` choose the same
  refined run
- at minimum, do not rely on `created_utc` alone
- preferred ordering:
  1. effective review timestamp if present
  2. refined run creation timestamp
  3. run name descending as final tie-break

Practical implementation options:

1. Reuse `_resolve_review_status_sources(...)` and extract effective review
   timestamp for candidate ordering.
2. Fall back to `created_utc` / `timestamp_utc`.
3. Use `run_name` descending as the final tie-break, matching the registry
   pattern that prefers the lexically later refined run when timestamps tie.

Acceptance criteria:

- preflight chooses the same refined run that
  `Registry.query_keypoint_quality_current(...)` reports as current
- the stale refined-run mismatch no longer occurs on migrated
  `traditional_v2_seed` archives
- no manual attr repair is required for newly migrated datasets

## Patch 2: Stop Creating Tied `created_utc` Values During Migration

Target:
- [extend_keypoint_skeleton.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/extend_keypoint_skeleton.py)

Current issue:
- migrated refined runs such as `..._traditional_v2_seed` retain the original
  refined run `created_utc`
- migration time is stored separately in `migration_created_at_utc`
- this leaves old and migrated sibling refined runs with identical
  `created_utc`, which triggers the tie-break ambiguity above

Required change:
- when writing a migrated refined run, set `created_utc` to the migration
  creation time for the new run instead of copying the old refined run’s
  `created_utc`
- still preserve provenance of the original refined run via:
  - `migration_source_run`
  - `migration_source_group`
  - `migration_created_at_utc`

Acceptance criteria:

- new migrated refined runs are unambiguously newer than their source sibling
- `refined_keypoints_runs.latest` and downstream selectors naturally prefer the
  migrated run
- `repair_keypoint_training_refined_run_ties.py` is only needed for historical
  archives, not future ones

## Optional Cleanup

- Add a focused regression test for `_resolve_refined_keypoint_quality(...)`
  where two sibling refined runs share the same source run and timestamps, but
  only one is the migrated current run.
- Add a focused regression test for `extend_keypoint_skeleton.py` asserting that
  migrated refined runs get a distinct, newer `created_utc`.
- Consider whether the repair utility should remain as an operator tool for old
  archives or be retired after one-time cleanup.
