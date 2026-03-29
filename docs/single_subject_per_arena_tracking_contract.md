# Single-Subject-Per-Arena Tracking Contract

Date anchored: 2026-03-06

Purpose: define the concrete `tracking_runs` contract for the current operating
mode where each arena contains at most one subject.

This document is the first implementation target under
[`track_identity_target_architecture.md`](./track_identity_target_architecture.md).

## Scope

This contract is for recordings where:

- each detection belongs to at most one spatial arena
- each arena contains at most one subject for the duration of the run
- downstream analysis should operate on real `track_id`s, even though those
  tracks are derived from arena occupancy

Out of scope:

- multiple subjects within one arena
- identity swaps inside an arena
- online target/chaser identity

## Contract Summary

The `single_subject_per_arena` tracker consumes one source rowset plus one
source arena-assignment run and produces one `track_id` per occupied
`arena_id`.

Strong semantic rules:

- `arena_id` remains the spatial namespace
- `track_id` remains the temporal subject namespace
- `track_id` must not be interpreted as an arena label, even when the mapping
  is deterministic
- unassigned rows do not become tracks

## Inputs

### Required upstream lineage

The tracking run must bind to one exact source rowset:

- `detect_runs/<run>`, or
- `refined_detect_runs/<run>/<group>`

And one exact arena-assignment run:

- `arena_assignment_runs/<run>`

The tracker must resolve inputs by exact provenance, not by `latest`.

Required lineage attrs to match:

- `source_detect_run`
- `source_refined_run` when applicable

### Required source arrays

From the tracked rowset:

- `frame_indices`

From the arena-assignment run:

- `arena_ids`

### Arena assignment assumptions

Each row has one arena assignment state:

- `arena_id >= 0`: assigned to a valid arena
- `arena_id == -1`: unassigned

## Track Construction Rule

### Occupied arenas

An arena is occupied if it has at least one tracked row with `arena_id >= 0`.

### Track creation

The tracker creates exactly one `track_id` for each occupied arena.

Recommended deterministic mapping:

1. collect unique occupied `arena_id`s
2. sort them ascending
3. assign `track_id`s as `0..n_tracks-1` in that sorted order

Example:

- occupied `arena_id`s: `[5, 9, 12]`
- emitted `track_id`s: `[0, 1, 2]`
- mapping: `track 0 -> arena 5`, `track 1 -> arena 9`, `track 2 -> arena 12`

This keeps `track_id` separate from `arena_id` while remaining deterministic.

### Row-level assignment

For each source row:

- if `arena_id >= 0`, assign the corresponding run-local `track_id`
- if `arena_id == -1`, assign `track_id == -1`

## Persisted Layout

Target namespace:

`tracking_runs/<run_name>/`

### Required arrays

| Array | Shape | DType | Meaning |
| --- | --- | --- | --- |
| `track_ids` | `(n_rows,)` | `int32` | Track assignment per source row. `-1` means untracked. |
| `arena_ids` | `(n_rows,)` | `int32` | Arena assignment per source row. Mirrors the bound arena-assignment run. |
| `frame_indices` | `(n_rows,)` | `int32` | Copied from the tracked source rowset for direct auditing/debugging. |
| `source_row_indices` | `(n_rows,)` | `int32` | `0..n_rows-1` index into the exact tracked source rowset. |
| `track_ids_present` | `(n_tracks,)` | `int32` | Sorted list of real emitted track IDs. |
| `track_arena_ids` | `(n_tracks,)` | `int32` | Arena ID for each emitted track. Parallel to `track_ids_present`. |

### Suggested optional arrays

| Array | Shape | DType | Meaning |
| --- | --- | --- | --- |
| `tracking_status` | `(n_rows,)` | `int8` or UTF-8 | Per-row status such as `ok`, `unassigned_arena`, `conflict_dropped`. |
| `tracking_confidence` | `(n_rows,)` | `float32` | Confidence score when a relaxed conflict policy is used. |
| `conflict_flags` | `(n_rows,)` | `bool` | True for rows involved in arena/frame conflicts. |
| `track_frame_counts` | `(n_tracks,)` | `int32` | Number of rows assigned to each track. |

### Required attrs

| Attr | Meaning |
| --- | --- |
| `tracking_method` | Must be `single_subject_per_arena`. |
| `source_detect_run` | Bound detect lineage. |
| `source_refined_run` | Bound refined-detect lineage when applicable. |
| `source_arena_assignment_run` | Exact arena-assignment run consumed. |
| `source_rowset_path` | Full path of the tracked source group. |
| `track_namespace` | Recommended value: `local_per_run`. |
| `unassigned_track_id` | Recommended value: `-1`. |
| `conflict_policy` | See failure/strictness policy below. |
### Suggested attrs

| Attr | Meaning |
| --- | --- |
| `num_tracks` | Number of emitted real tracks. |
| `n_assigned_rows` | Count of source rows with a real `arena_id >= 0`. |
| `n_unassigned_rows` | Count of source rows with `arena_id == -1`. |
| `unassigned_row_rate_percent` | Percentage of source rows that remained unassigned. |
| `tracking_qc_state` | Current structured QA state: `ok` or `warn`. |
| `tracking_warn_threshold_rows` | Default warning threshold on row count. |
| `tracking_warn_threshold_percent` | Default warning threshold on unassigned percentage. |
| `tracking_block_threshold_rows` | Future blocking threshold on row count. |
| `tracking_block_threshold_percent` | Future blocking threshold on unassigned percentage. |
| `summary_statistics` | JSON-friendly counts and coverage summaries. |
| `arena_assignment_namespace` | `arena_assignment_runs`. |

## Track ID Semantics

### What `track_id` means here

In this mode, `track_id` means:

- the one subject trajectory associated with one occupied arena for this run

### What `track_id` does not mean

- not the arena label itself
- not the dish identifier
- not a registry subject identifier

### Why keep the namespaces separate now

Even though there is one subject per arena, keeping `track_id` separate buys us:

- a stable downstream contract for `track_kinematics`
- a clean migration path to multi-subject tracking later
- less confusion between visual ROI labels and subject identities

## Conflict And Failure Policy

### Recommended default: `fail`

For the current operating mode, the default conflict policy should be strict.

The tracker should fail if any frame contains more than one assigned detection
for the same `arena_id`.

Reason:

- the experiment contract says one subject per arena
- multiple detections in one arena/frame usually indicate duplicate detections,
  bad assignment, or an unexpected multi-subject condition
- silently collapsing these rows would hide real data quality problems

### Optional relaxed policies

If later needed, these can be supported explicitly:

- `warn_keep_first`
- `warn_keep_best`
- `warn_keep_largest`

But they should not be the default.

If a relaxed policy is used, the run should persist enough diagnostics to show:

- which rows conflicted
- which row won
- why it won

## Required Failure Modes

The tracker should fail hard for:

1. source lineage mismatch between the tracked rowset and the arena-assignment run
2. row-count mismatch between the tracked rowset and arena-assignment arrays
3. duplicate assigned detections within one `arena_id` and frame when
   `conflict_policy == fail`
4. missing required arrays or missing required provenance attrs

The tracker should not fail for:

1. zero detections overall
2. valid unassigned rows (`arena_id == -1`)
3. arenas that simply never become occupied

## No-Data And Unassigned Behavior

### No detections

If the source rowset has zero rows:

- create a valid empty `tracking_runs/<run>`
- `track_ids_present` and `track_arena_ids` are empty
- `n_tracks == 0`

### Unassigned rows

If rows are unassigned to any arena:

- keep `arena_id == -1`
- set `track_id == -1`
- do not include these rows in `track_ids_present`
- do not let them produce `tracks/id_-1` downstream
- monitor them as a first-class QA signal per
  [`tracking_unassigned_row_policy.md`](./tracking_unassigned_row_policy.md)

## Downstream Contract For `track_kinematics`

`track_kinematics` should bind to one exact `tracking_runs/<run>` and use:

- `track_ids`
- `track_ids_present`
- `track_arena_ids`

Expected behavior:

- group rows by real `track_id`
- expose `arena_id` as track metadata
- current runtime skips `track_id == -1` by default and only keeps it with
  `--include-unassigned`

Required lineage attrs for track-kinematics outputs:

- `source_tracking_run`
- `source_arena_assignment_run`

## Recommended CLI Surface

Suggested mode selection:

```bash
scripts/py -m fisheye.tracking.track \
  data.zarr \
  --method single_subject_per_arena
```

Suggested strictness knobs:

- `--conflict-policy fail|warn_keep_first|warn_keep_best|warn_keep_largest`
- `--arena-assignment-run <run>`
- `--source-detect-run <run>`
- `--source-refined-run <run>`

## Migration Notes

Near-term migration can keep the implementation simple:

1. read `arena_assignment_runs/<run>/arena_ids`
2. build `tracking_runs` from those arena assignments
3. make `track_kinematics` consume `tracking_runs`

That gives the current one-fish-per-dish workflow the right semantics before
any multi-subject tracker exists.

## Bottom Line

For the current operating mode, the correct model is:

- arena assignment tells us which dish/chamber/subdish a row belongs to
- tracking converts occupied arenas into one real track each
- track kinematics operates on those tracks

That is slightly more explicit than the current shortcut, but it is the right
foundation even for the simple one-subject-per-arena case.
