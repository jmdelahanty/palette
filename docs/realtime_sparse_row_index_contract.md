# Realtime Sparse Row Index Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-28
-->

Purpose: define how Palette should make sparse, row-aligned instance data fast
enough for realtime or interactive consumers such as Crimson without changing
the canonical storage model.

## Motivation

Palette's future-safe multi-subject model is sparse and row-aligned:

```text
one row = one detected/refined subject instance
channels = semantic components of that instance
```

For refined subject masks, this means a single row may contain:

```text
subject_body, eye_left, eye_right, swim_bladder
```

Multiple subjects in one frame should be represented as multiple rows with the
same `frame_index`, not as identity-bearing channels such as `subject_0_body`
or `subject_1_body`.

That model is correct for multi-subject tracking, but it is not sufficient for
interactive viewers by itself. A viewer must be able to answer common questions
without scanning every row:

- Which rows should be drawn for frame `t`?
- Which rows belong to track `k`?
- Which mask and shape rows line up with a selected detection row?

## Design Rule

Canonical storage should remain sparse and row-aligned. Realtime viewers should
consume additive indexes that make common lookups cheap.

Do not make the canonical refined-mask or subject-shape arrays dense
`frame x slot` tables just for viewer convenience. Dense frame/track views may
be derived caches later, but they should not become the source of truth.

## Near-Term Implementation Policy

Palette may continue implementing and validating the current single-fish-per-
dish workflow first. The repository does not need a full multi-subject tracker
before subject-body QC, subject-shape centerlines, tail anchors, splines, or
viewer overlays can move forward.

However, new contracts and writers should remain multi-subject-compatible:

- allow multiple rows with the same `frame_index`
- never assume one row per frame in schemas, tests, or viewers
- keep component channels semantic, not identity-bearing
- keep `track_id` separate from `arena_id`, even when the current tracker maps
  one occupied arena to one track
- make frame indexes work for one row per frame and many rows per frame
- add track-aware joins only when an exact `tracking_runs/<run>` source is
  consumed

This is the intended compromise: implement the data we have now, but avoid
contract choices that would require a destructive redesign for multi-subject
data later.

## Required Fast Path

For a Crimson-style frame viewer, the desired access path is:

```text
frame_index
  -> row indices for that frame
  -> per-row masks/geometry
  -> optional track_id labels
```

A viewer should not have to perform:

```text
np.where(row_index/frame_indices == frame_index)
```

on every rendered frame.

## Recommended Frame Index Layout

For any large sparse row-aligned run that expects interactive frame-based
viewing, add an optional CSR-style frame index:

```text
<run>/
  row_index/
    frame_indices              (N,)
    detection_indices          (N,) optional
    source_refined_row_ids      (N,) optional
    source_row_indices          (N,) optional
    track_ids                  (N,) optional

  frame_index/
    frame_numbers              (F,) sorted unique frame numbers covered
    row_start                  (F,) int64 offset into row_indices
    row_count                  (F,) int32 count for each frame
    row_indices                (M,) int64 row indices grouped by frame
```

Lookup rule:

```text
i = searchsorted(frame_numbers, requested_frame)
if frame_numbers[i] == requested_frame:
    rows = row_indices[row_start[i] : row_start[i] + row_count[i]]
else:
    rows = []
```

`M` is normally equal to `N`, but the layout allows future filtered indexes.

## Recommended Track Index Layout

When a run has track-aware row assignments, add one of these equivalent
indexes:

```text
<run>/
  track_index/
    track_ids                  (T,) sorted track IDs
    row_start                  (T,) int64 offset into row_indices
    row_count                  (T,) int32 count for each track
    row_indices                (M,) int64 row indices grouped by track
```

or:

```text
<run>/
  tracks/
    id_<track_id>/
      row_indices              (K,)
      frame_indices            (K,) optional mirror for fast plotting
```

The CSR-style `track_index/` is better for low-overhead realtime readers. The
`tracks/id_<track_id>/` layout is better for human browsing and aligns with
existing `track_kinematics_runs`.

Both may coexist if needed.

## Relationship To Tracking

`track_id` remains a temporal subject identity produced by
`tracking_runs/<run>`. It must not be inferred from row order, arena order, or
mask channel index.

A row-aligned mask or shape run may include:

```text
row_index/track_ids
attrs/source_tracking_run
```

only when it has joined against one exact `tracking_runs/<run>` source.

If no tracking run was consumed, the run should remain row-aligned and omit
track IDs rather than inventing them.

## Realtime Crimson Read Pattern

For a frame viewer:

1. Resolve the active refined mask or subject-shape run.
2. Load `frame_index/frame_numbers`, `row_start`, `row_count`, and
   `row_indices`.
3. For each displayed frame, resolve rows from the frame index.
4. Draw each row's masks and geometry in row order or z-order chosen by the
   viewer.
5. If `row_index/track_ids` exists, display track labels and allow track-based
   filtering.

For a track viewer:

1. Resolve `track_index` or `tracks/id_<track_id>/row_indices`.
2. Use row indices to load aligned masks, subject-shape geometry, and frame
   numbers.
3. Use `frame_index` only for frame-to-row navigation, not as the source of
   identity.

## Consumer Guarantees

Writers that create these indexes should guarantee:

- all row indices are zero-based indices into the primary row axis of the run
- `frame_numbers`, `track_ids`, and row groups are sorted unless attrs say
  otherwise
- missing frames have no row group rather than a fake empty row
- `track_id == -1` rows are allowed in `row_index/track_ids` but should be
  excluded from normal `track_index` groups unless explicitly requested
- index attrs record the source row count and build method

Recommended attrs:

```text
frame_index.attrs:
  index_schema_id = "palette.frame_row_index"
  index_schema_version = 1
  source_row_axis
  source_row_count
  sorted_by = "frame_index_then_row_index"

track_index.attrs:
  index_schema_id = "palette.track_row_index"
  index_schema_version = 1
  source_tracking_run
  source_row_count
  excludes_unassigned_track_id = true
```

## Where These Indexes Should Appear

Good candidates:

- `refined_subject_masks_runs/<run>` for mask overlays and review.
- `analysis/subject_shape_runs/<run>` for centerline, spline, body-frame, and
  tail overlays.
- `refined_detect_runs/<run>/instances` once sparse refined-detect instances
  become the canonical detect authoring surface.

Track indexes should only appear when a run has consumed or been joined to
`tracking_runs`.

## Non-Goals

- Do not make masks dense over `frame x subject`.
- Do not encode biological identity in mask channel names.
- Do not make Crimson responsible for expensive per-frame scans.
- Do not treat frame indexes as identity. They are lookup accelerators only.

## Implementation Checklist

- [ ] Add a shared helper that builds `frame_index/` from a row-aligned
      `frame_indices` array.
- [ ] Add a shared helper that builds `track_index/` from row-aligned
      `track_ids`, excluding `-1` by default.
- [ ] Add tests for multiple rows in one frame, empty frames, and unassigned
      track rows.
- [ ] Add optional frame indexes to new subject-shape runs.
- [ ] Add optional frame indexes to refined subject-mask runs or a conservative
      backfill command.
- [ ] Update Crimson-facing handoff docs to prefer `frame_index/` when present
      and fall back to row scans only for small or legacy archives.
