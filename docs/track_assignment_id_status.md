# Track And Assignment ID Status

Date anchored: 2026-03-06

Purpose: document how `arena_assignment_runs`, `track_kinematics_runs`, and the
legacy `tracking_runs` concept currently relate to each other, so future
refactors can start from the implemented behavior rather than the intended one.

Historical note:

- This document captures the pre-`tracking_runs` integration state.
- The current implemented contract is documented in
  [`tracking_runs_contract_status.md`](./tracking_runs_contract_status.md).

## Executive Summary

The current analysis path does not maintain a clean separation between
assignment IDs and track IDs.

- `arena_assignment` writes ROI/subdish labels into `arena_assignment_runs/<run>/arena_ids`.
- Offline `track_kinematics` mostly reuses those same integer values as
  persisted `track_id`s.
- Online `track_kinematics` does not use `arena_assignment_runs`; it creates one
  synthetic track with ID `0`.
- Registry/status code still treats `tracks` / `tracking_runs` as a separate
  step, but the main analysis path does not consume that data.

So today `track_id` is not one stable identity concept across the stack.

## Terms As Implemented Today

| Term | Current meaning | Where it lives |
| --- | --- | --- |
| `assignment_id` | Spatial ROI label assigned by arena assignment; usually subdish or dish ID | `arena_assignment_runs/<run>/arena_ids` |
| `track_id` (offline) | Usually the same integer as `assignment_id` | `analysis/track_kinematics_runs/*/track_ids`, `tracks/id_<id>` |
| `track_id` (online) | Synthetic singleton ID `0` for the online target/chaser series | `analysis/track_kinematics_runs/online/*` |
| `tracks` step | Separate registry/status concept backed by `tracking_runs` | registry/status code, not the main analysis path |

## Producer Model

### `arena_assignment`

[`src/fisheye/tracking/arena_assignment.py`](../src/fisheye/tracking/arena_assignment.py)
currently implements spatial assignment, not general temporal identity
tracking.

Behavior:

- detections are assigned to the ROI whose bounding box contains the detection
  center
- assigned detections get `mask["id"]`
- unassigned detections get `-1`
- single-dish mode typically produces one ROI with ID `0`

Persisted outputs under `arena_assignment_runs/<run>`:

- `arena_ids`
- `n_detections_per_arena`
- attrs such as `source_detect_run`, `source_refined_run`,
  `assignment_source`, `assignment_method`, `arena_definitions`,
  `summary_statistics`

Important provenance note:

- the stage prefers the latest canonical curated refined detections at
  `refined_detect_runs/<run>/instances` when present
- legacy subgroup-era refined inputs may still appear only for historical
  archives
- otherwise it falls back to the latest raw detect run

## Consumer Model

### Offline `track_kinematics`

[`src/fisheye/analysis/track_kinematics.py`](../src/fisheye/analysis/track_kinematics.py)
loads assignment IDs through
[`src/fisheye/analysis/compute_speed.py`](../src/fisheye/analysis/compute_speed.py)
and groups rows by the unique values in that array.

Current behavior:

- `np.unique(track_ids)` defines the track groups
- those same integers become persisted `track_ids`
- subgroup names are written as `tracks/id_<track_id>`

Effectively, offline `track_id == assignment_id` for the current pipeline.

For refined detections:

- rows may be remapped back to source-detect assignment IDs
- interpolated rows inherit neighboring IDs by forward/back fill

### Online `track_kinematics`

Online `track_kinematics` does not read `arena_assignment_runs`.

Current behavior:

- all rows are assigned `track_ids_online = 0`
- one persisted track is written with ID `0`
- `chaser_index` is recorded separately in provenance/inputs

Effectively, online `track_id` is a synthetic label, not an assignment label.

## Where The Model Is Inconsistent

### 1. `track_id` does not mean one thing

Today `track_id` means:

- offline: assignment/ROI label
- online: synthetic singleton label
- registry/status `tracks`: something else again, tied to `tracking_runs`

That makes cross-stage reasoning harder than it should be.

### 2. `tracking_runs` still exists as a parallel concept

Registry and status code still model a separate `tracks` step backed by
`tracking_runs`.

Relevant files:

- [`src/fisheye/registry/maintenance.py`](../src/fisheye/registry/maintenance.py)
- [`src/fisheye/utils/check_recording_steps.py`](../src/fisheye/utils/check_recording_steps.py)

But the main analysis flow is:

`arena_assignment_runs -> track_kinematics_runs`

not:

`tracking_runs -> track_kinematics_runs`

So the repo currently has both:

- an active analysis path using `arena_assignment_runs`
- a separate status/legacy concept using `tracking_runs`

### 3. Assignment resolution is latest-based, not source-bound

[`load_arena_ids(...)`](../src/fisheye/analysis/compute_speed.py) resolves
the latest `arena_assignment_runs` entry and only warns if provenance does not
match the expected detect/refined-detect lineage.

That means a wrong-but-shape-compatible assignment run can be consumed if
multiple lineages exist in one archive.

### 4. Fallback behavior hides semantic differences

When no assignment run exists, `load_arena_ids(...)` falls back to all
zeros.

That makes `track_id == 0` ambiguous:

- it may be a real ROI/assignment ID
- or it may mean “no assignment data; collapse everything into one track”

### 5. Unassigned rows can become a persisted public track

If offline unassigned detections are not filtered out, `-1` can survive into
`track_kinematics` and become:

- `track_id == -1`
- subgroup `tracks/id_-1`

That may be useful diagnostically, but it is not clearly documented as a public
contract.

### 6. Schema/docs drift exists around assignment outputs

Current docs/schema still imply some contract pieces that the producer does not
write consistently.

Examples:

- `zarr_structure.md` used to describe a stale `confidence` array for arena assignment
- the active producer writes `arena_ids` and `n_detections_per_arena`

So there is at least one contract mismatch between docs/schema and runtime
behavior.

## What Seems True Today

If we describe the current pipeline in the most accurate plain language:

- `arena_assignment` is a spatial ROI-labeling stage
- offline `track_kinematics` is an analysis stage that aggregates rows by those
  ROI labels
- online `track_kinematics` is a singleton track summary over stimulus/chaser
  data
- `tracking_runs` is not the identity source for the current main analysis path

That is a coherent description of the code, even if it is not the desired final
architecture.

## Questions To Decide Before Refactoring

### 1. Should `assignment_id` and `track_id` be separate concepts?

Options:

- keep them separate and rename the current offline consumer-facing ID surface
  toward `assignment_id` / `roi_id`
- or formally define current offline `track_id` as “ROI-resolved identity” and
  keep the reuse explicit

### 2. Is `tracking_runs` active architecture or legacy drift?

We should decide whether:

- `tracking_runs` remains a future/parallel tracking product
- or it should be retired from registry/status expectations

### 3. Should `track_kinematics` bind to a specific assignment run?

Current answer is “latest, with warnings.”

A stronger model would resolve `arena_assignment_runs` by matching:

- `source_detect_run`
- `source_refined_run`

instead of relying on `latest`.

### 4. Should unassigned detections remain representable as a track?

If yes, that should be documented explicitly.

If no, offline track construction should strip or quarantine `-1` before
persisting public `track_ids`.

### 5. What should the public vocabulary be?

Candidate naming split:

- `assignment_id` or `roi_id`: value from `arena_assignment_runs/arena_ids`
- `track_id`: reserved for a true temporal identity/tracking namespace
- `chaser_index`: only for stimulus/online tracking semantics

## Recommended Next Review Topics

If we review this together before implementing changes, the highest-value topics
are:

1. whether offline `track_id` should remain equal to ROI/assignment ID
2. whether `tracking_runs` should stay in the active step model
3. whether strict source-resolved assignment lookup should be the first code
   change
4. whether docs/schema should be updated first or after behavior changes

## Bottom Line

The repo currently uses one set of integers for several adjacent concepts:
assignment labels, offline track labels, and in some places a broader notion of
tracking.

The code is understandable once you follow the lineage, but the semantics are
not clean enough yet to be a durable contract. This is a good point to clarify
the model before doing deeper tracking or analysis refactors.
