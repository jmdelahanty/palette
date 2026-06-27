# Mutable Review Runs Contract

Date anchored: 2026-06-27

Purpose: define how Palette should handle small human or automated review edits
without creating a new full Zarr run for every changed row.

Status: target contract. This is not yet the implemented write path.

Current code has pieces of this model, but not the full machinery:

- `crop_runs`, `arena_assignment_runs`, and `tracking_runs` already separate
  observation rows from assignment/identity interpretation.
- `refined_detect_runs/<run>/instances` is already the canonical curated detect
  surface.
- the browser labeling workflow currently uses a SQLite task/control plane and
  does not yet perform server-owned Zarr row patches on every save.
- no general `edit_revision`, `mutable`, `locked`, or append-only Zarr edit-log
  contract is implemented across these run families yet.

This document records the target policy for that implementation.

## Core Rule

Keep observation rowsets stable. Make review/assignment surfaces mutable.

Observation rowsets describe what was observed or materialized:

- `detect_runs/<run>`
- `crop_runs/<run>`
- model-output runs such as raw keypoints or raw subject-mask probabilities

Review and assignment surfaces describe interpretation of those observations:

- `refined_detect_runs/<run>/instances`
- `arena_assignment_runs/<run>`
- `tracking_runs/<run>`
- refined keypoint and refined subject-mask review surfaces

Small edits should patch the mutable review surface in place, increment a
review-scoped `edit_revision` counter, and append an audit event. They should
not rewrite immutable source rowsets and should not create a new run per row
edit.

## Why Not New Run Per Edit

Creating a new Zarr group for every bbox adjustment, arena reassignment, track
fix, keypoint correction, or mask paint stroke creates avoidable problems:

- many near-identical run groups
- expensive full-array rewrites for tiny changes
- ambiguous "latest" behavior during active review
- harder browser-review UX, because every save would change the target run
- noisy registry history where every click looks like a new analysis product

Versioned run groups are still useful, but they should represent meaningful
milestones or branches, not individual UI saves.

## Recommended Lifecycle

### 1. Automatic Initialization

An automatic stage creates an initial mutable review run.

Example:

```text
crop_runs/crop_001/                         # stable observation rowset

arena_assignment_runs/arena_auto_001/       # initial editable assignment
tracking_runs/tracks_auto_001/              # initial editable identity layer
```

### 2. Session-Staged Review Edits

Preferred v1 behavior: frequent user saves should persist a lightweight review
session, not immediately rewrite the canonical Zarr arrays.

The session layer may be SQLite, JSONL, or another small sidecar store owned by
the server/UI workflow:

```text
review_sessions/<session_id>/
  edits
  checkpoints
  state = active | closed | applied | abandoned
  target_run_path = "refined_detect_runs/review_001/instances"
  target_edit_revision = 17
```

During review:

- the UI renders the canonical Zarr run plus the session edit overlay
- "save" means persist the session edits/checkpoint
- canonical arrays remain unchanged
- work is recoverable if the UI or browser closes unexpectedly

On session close, submit, or explicit finalize:

1. validate that the target run still has the expected `edit_revision`
2. coalesce edits by target array and physical chunk
3. patch each touched canonical chunk once
4. increment `edit_revision`
5. append one or more durable edit events
6. update review/QC summaries and downstream stale markers

This avoids rewriting a large Zarr chunk on every drag/paint/save. For example,
if a refined-detect array has large row chunks, 200 bbox edits can be applied
as one batched chunk-aware write instead of 200 separate read-modify-write
cycles.

Other consumers should treat unapplied session edits as invisible unless they
explicitly opt into the session overlay. Durable analysis should consume only
applied review runs, locked runs, or finalized snapshots.

### 3. Direct In-Place Row Edits

Direct in-place row edits are still allowed for simple CLI tools, low-volume
repair scripts, or run families whose physical chunking is already
review-friendly. In that mode, edits patch touched rows directly.

Expected metadata:

```text
attrs:
  mutable = true
  locked = false
  edit_revision = 17
  review_status = "in_progress"
  source_rowset_path = "crop_runs/crop_001"
```

Each successful save:

1. validates the target run and source rowset
2. verifies the client edited the expected `edit_revision` or handles a conflict
3. patches only touched rows
4. increments `edit_revision`
5. appends an audit event
6. updates summary/QC metadata that depends on the edited rows

For a single human reviewer, optimistic `edit_revision` checks may be sufficient.
For parallel agents or multiple browser sessions editing the same run, writers
must use either optimistic conflict detection or an explicit server-side write
lock.

### 4. Lock Or Snapshot

When the run becomes a stable analysis input, either:

- lock the reviewed run in place with `locked = true`, or
- create a compact finalized snapshot run.

Snapshots are appropriate when a workflow needs compaction, branch isolation, or
publication-grade provenance. Ordinary review saves should not snapshot.

## Edit Event Contract

Mutable review runs should carry an append-only edit log. The exact storage can
evolve.

Minimal v1 required fields:

| Field | Meaning |
| --- | --- |
| `event_id` | Stable unique edit event ID. |
| `created_at_utc` | Edit timestamp. |
| `actor` | User, service, or agent that made the edit. |
| `run_path` | Mutated Zarr path. |
| `edit_revision_before` | Edit revision expected by the writer. |
| `edit_revision_after` | Edit revision after successful save. |
| `row_indices` | Source rows touched by the edit. |
| `operation` | `update`, `append`, `tombstone`, or `bulk_patch`. |

Recommended future fields:

| Field | Meaning |
| --- | --- |
| `tool` | UI/CLI/service name. |
| `source_rowset_path` | Bound observation rowset. |
| `fields` | Names of arrays/attrs changed. |
| `reason` | Optional human or machine-readable reason. |
| `restore_source_event_id` | Event being restored when `operation == "restore"`. |

The event log is for provenance and recovery. Current arrays remain the fast
read surface for normal consumers.

## Detection Review

`detect_runs/<run>` is immutable raw detector output.

`refined_detect_runs/<run>/instances` is the mutable curated surface.

Recommended row-edit behavior:

- bbox corrections update the touched sparse instance rows
- manual additions append new sparse instance rows with stable row IDs
- deletions should use tombstone/status fields rather than immediate physical
  compaction during active review
- source provenance such as `source_detect_row_index`, `source_kind_codes`, and
  `manual_edit_flags` must remain row-addressable

Compaction may happen during finalization, but consumers should not require
compaction for correctness.

### Refined-Detect Chunking Migration

The current refined-detect instance arrays are not yet optimized for frequent
small in-place edits. In current code, refined-detect row chunks can be large
enough that one bbox correction rewrites a large physical chunk. That is
acceptable for initialization and read-mostly workloads, but it conflicts with
the target interactive-edit policy.

Before making `refined_detect_runs/<run>/instances` a high-frequency mutable UI
surface, choose one of these explicitly:

- create review-oriented refined-detect runs with smaller row chunks
- keep large chunks and accept chunk-level read-modify-write per save for low
  volume review
- store row-level edit overlays and periodically compact into the main
  `instances` arrays

This decision should be made before implementing browser saves that patch
refined-detect Zarr arrays directly.

## Arena And Tracking Review

`crop_runs/<run>` should remain stable. It records frame indices, ROI geometry,
and source detection lineage for each crop row.

If a row is assigned to the wrong arena or track, patch the assignment layer:

```text
arena_assignment_runs/<run>/arena_ids[row] = corrected_arena_id
tracking_runs/<run>/track_ids[row] = corrected_track_id
```

Do not rewrite the crop row to "fix" identity. The crop is the observation; the
assignment/tracking run is the interpretation.

Required invariants for editable `tracking_runs`:

- `source_rowset_path` remains fixed for the run
- `source_row_indices` remain stable
- `frame_indices` remain copied from the bound source rowset
- `track_id == -1` remains the unassigned identity sentinel
- edits that create duplicate `(frame_index, arena_id)` conflicts under
  `single_subject_per_arena` must fail or update conflict/QC status explicitly

## Downstream Provenance

Downstream analysis should record the exact mutable surface it consumed.

Minimum recommended lineage:

- source run path
- `source_rowset_path`
- `edit_revision` consumed
- review status consumed
- locked/finalized state consumed

For durable/public analysis, prefer consuming a locked run or a finalized
snapshot. For interactive exploratory analysis, consuming an in-progress review
run is acceptable if the `edit_revision` is recorded.

## When To Create A New Run

Create a new run when the semantic product changes, not for every row edit.

Good reasons:

- initialize automatic output from a different model or parameter set
- fork review from a previous locked/final run
- publish a finalized compact snapshot
- rematerialize after upstream rowset changes
- switch tracking method, for example from `single_subject_per_arena` to a real
  multi-object tracker

Poor reasons:

- one bbox moved
- one crop row reassigned to a different arena
- one keypoint corrected
- one mask component painted

## Write Safety

Mutable review runs should be written by server-owned tools, not directly by
untrusted browsers or ad hoc clients.

Writers should use:

- row-level validation against `source_rowset_path`
- optimistic `edit_revision` checks or an explicit write lock
- append-only edit events
- row-granular chunks for actively edited arrays
- explicit stale markers for downstream derived surfaces

Large read-mostly arrays can use larger chunks, but actively edited arrays
should not be chunked so large that one row edit rewrites an excessive chunk.

## Relationship To Versioned Datasets

This contract does not remove versioning. It moves versioning to the right
level:

- row edits are `edit_revision`s inside one mutable review run
- milestones are locked runs or finalized snapshot runs
- branches are new runs with explicit source provenance

That keeps provenance strong without making every UI save a new dataset.

## Related Contracts

- [`single_subject_per_arena_tracking_contract.md`](./single_subject_per_arena_tracking_contract.md)
- [`track_identity_target_architecture.md`](./track_identity_target_architecture.md)
- [`refined_detect_collapse_v2.md`](./refined_detect_collapse_v2.md)
- [`refined_subject_masks_runs_contract.md`](./refined_subject_masks_runs_contract.md)
