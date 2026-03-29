# Tracking Runs Contract Status

Date anchored: 2026-03-06

Purpose: document the implemented `tracking_runs` contract after the
`arena_assignment` rename and the `single_subject_per_arena` tracking work.

## Executive Summary

`tracking_runs` is now active architecture, not a legacy side path.

- [`src/fisheye/tracking/arena_assignment.py`](../src/fisheye/tracking/arena_assignment.py)
  writes `tracking_runs/<run>` automatically after a successful
  `arena_assignment_runs/<run>`.
- [`src/fisheye/tracking/single_subject_per_arena.py`](../src/fisheye/tracking/single_subject_per_arena.py)
  is the canonical producer and resolver for the current mode.
- [`src/fisheye/analysis/track_kinematics.py`](../src/fisheye/analysis/track_kinematics.py)
  now binds to one exact `tracking_runs/<run>` lineage and records
  `source_tracking_run`.
- Registry/status code treats `tracks` as a real downstream step backed by
  `tracking_runs`.

The remaining issue is not whether `tracking_runs` exists. It does. The open
questions are about how strict and how rich that contract should become.

## What Is Implemented

### Producer

Current producer:

- [`src/fisheye/tracking/single_subject_per_arena.py`](../src/fisheye/tracking/single_subject_per_arena.py)

Entry point used in practice:

- [`src/fisheye/tracking/arena_assignment.py`](../src/fisheye/tracking/arena_assignment.py)

Current method:

- `single_subject_per_arena`

Behavior:

- source rows are the exact detect/refined-detect rowset used by arena assignment
- occupied `arena_id >= 0` values are sorted ascending
- run-local `track_id` values are allocated deterministically as `0..n-1`
- rows with `arena_id == -1` get `track_id == -1`
- duplicate detections in the same `(frame_index, arena_id)` raise
  `TrackingConflictError`

### Persisted Arrays

Declared in:

- [`src/fisheye/shared/zarr/stage_arrays.py`](../src/fisheye/shared/zarr/stage_arrays.py)

Written by:

- [`src/fisheye/tracking/single_subject_per_arena.py`](../src/fisheye/tracking/single_subject_per_arena.py)

Current required arrays under `tracking_runs/<run>/`:

| Array | Meaning |
| --- | --- |
| `track_ids` | Track assignment per source row. |
| `arena_ids` | Arena assignment per source row. |
| `frame_indices` | Copied frame index per source row. |
| `source_row_indices` | `0..n_rows-1` index into the bound source rowset. |
| `track_ids_present` | Sorted list of emitted real track IDs. |
| `track_arena_ids` | Arena ID parallel to `track_ids_present`. |

## Persisted Attrs

Current attrs written by the producer:

- `method`
- `tracking_method`
- `created_at_utc`
- `source_detect_run`
- `source_refined_run` when applicable
- `source_arena_assignment_run`
- `source_rowset_path`
- `track_namespace`
- `unassigned_track_id`
- `conflict_policy`
- `num_tracks`
- `n_assigned_rows`
- `n_unassigned_rows`
- `unassigned_row_rate_percent`
- `tracking_qc_state`
- `tracking_warn_threshold_rows`
- `tracking_warn_threshold_percent`
- `tracking_block_threshold_rows`
- `tracking_block_threshold_percent`
- `summary_statistics`
- stage provenance payload via `write_stage_provenance(...)`

Important detail:

- the writer currently records `num_tracks` as a top-level attr
- `summary_statistics["n_tracks"]` also exists
- there is no top-level `duration_seconds` attr today

## Resolution Semantics

Resolver:

- [`resolve_tracking_run(...)`](../src/fisheye/tracking/single_subject_per_arena.py)

Current matching keys:

- `source_detect_run`
- `source_refined_run`
- optional `source_arena_assignment_run`

Selection rule:

- collect all exact lineage matches
- if parent `latest` is one of them, use it
- otherwise use the lexicographically last matching run name

That means `tracking_runs` is already source-bound, not latest-only.

## Main Consumers

### `track_kinematics`

Consumer:

- [`src/fisheye/analysis/track_kinematics.py`](../src/fisheye/analysis/track_kinematics.py)

Current behavior:

- offline analysis loads `track_ids` through `load_tracking_ids(...)`
- it persists `source_tracking_run`
- it persists `source_arena_assignment_run`
- it mirrors `track_arena_ids` into the output run when available

Important caveat:

- `track_kinematics` now drops `track_id < 0` rows by default
- `track_id == -1` is only retained when `--include-unassigned` is used for
  explicit diagnostic output

### Registry / Status

Main surfaces:

- [`src/fisheye/registry/maintenance.py`](../src/fisheye/registry/maintenance.py)
- [`src/fisheye/registry/db.py`](../src/fisheye/registry/db.py)
- [`src/fisheye/utils/check_recording_steps.py`](../src/fisheye/utils/check_recording_steps.py)

Current behavior:

- `tracks` is considered downstream of `arena_assignment`
- status is presence-based at the run-group level
- tracking QA is also exposed as a structured `tracking_qc_state`
- registry rows do not currently verify detailed lineage between
  `tracking_runs` and `arena_assignment_runs`

## Current Gaps

### 1. A few long-tail descriptions outside the tracking docs may still lag

The main tracking docs and core Zarr/provenance references are now aligned, but
some secondary descriptions or docstrings may still reflect older assumptions.

### 2. The single-subject tracking doc intentionally includes future optional fields

Example doc:

- [`docs/single_subject_per_arena_tracking_contract.md`](./single_subject_per_arena_tracking_contract.md)

Current nuance:

- the doc describes the active required arrays/attrs plus optional future-facing
  arrays such as `tracking_status`, `tracking_confidence`, `conflict_flags`, and
  `track_frame_counts`
- those optional arrays are not emitted by the current writer, which is expected
  rather than a contract mismatch

### 3. Blocking thresholds remain deferred policy metadata

Current runtime behavior:

- tracking writer preserves unassigned rows as `track_id == -1`
- tracking writer now records `n_assigned_rows`, `n_unassigned_rows`, and
  `unassigned_row_rate_percent`
- tracking writer now also records `tracking_qc_state` and default thresholds
- offline `track_kinematics` excludes them from public outputs by default
- the raw tracking artifact still keeps those rows for provenance/debugging
- registry/status surfaces now render `WARN` from structured QA state
- blocking thresholds are still recorded as future policy metadata rather than
  active runtime enforcement by default

Policy reference:

- [`docs/tracking_unassigned_row_policy.md`](./tracking_unassigned_row_policy.md)

### 4. The implemented tracking method is deterministic, but still trivial

Today `track_id` is separate from `arena_id`, but it is still derived entirely
from occupied arenas. There is not yet a multi-subject temporal identity model.

### 5. A few descriptions outside the tracking module are stale

Example:

- [`src/fisheye/training/train_pose.py`](../src/fisheye/training/train_pose.py)

Its top-level docstring still implies `tracking_runs` contains pose/keypoint
data, which is not the current contract.

## Recommended Next Steps

1. Keep runtime tracking QA non-blocking until a specific workflow explicitly
   opts into stricter gating.
2. Audit remaining long-tail descriptions outside the main tracking docs
   (including code docstrings) as they come up.
3. Decide whether the next tracking method should still write the exact same
   arrays/attrs so `track_kinematics` can stay unchanged.

## Bottom Line

The repo now has a real `tracking_runs` contract. For the current one-fish-per-
arena mode, it is deterministic, source-bound, and actively consumed. The next
architecture work should build on that contract, not treat it as dead legacy.
