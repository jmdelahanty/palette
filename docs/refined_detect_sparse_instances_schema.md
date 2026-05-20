# Refined Detect Sparse Instances Schema

<!-- design-meta
status: active
last_updated: 2026-05-20
-->

Purpose: define the active sparse instance schema for `refined_detect_runs` and
the compatibility path away from older dense `(frame_index, entity_id)` roots.
This schema is the canonical authoring surface for new refined detect runs and
supports future multi-subject detect and tracking.

This document is the implementation-shaped follow-up to
`docs/refined_detect_multisubject_goal.md`.

## Design Summary

Target policy:

- `detect_runs/<run>` remains immutable raw detector output.
- `refined_detect_runs/<run>` becomes the canonical sparse curated detect
  surface.
- tracking owns temporal identity assignment across frames.
- dense `frame x track` or `frame x slot` layouts are derived projections, not
  the canonical refined detect storage model.

## V1 Boundary

The recommended v1 implementation should start with a deliberately narrow
contract:

- `source_detections/` is guaranteed only for the exact bound raw
  `detect_runs/<source_detect_run>` rowset
- `instances/` is the canonical curated sparse rowset
- manual additions with no one-to-one raw candidate are valid in `instances/`
  with `source_detect_row_index = -1`
- legacy `manual`, `manual_*`, `filtered`, and `interpolated` subgroup
  semantics are compatibility-only and do not need to be reconstructed as
  first-class sparse v1 authoring inputs

Reason:

- this keeps the initial sparse design clean
- it avoids dragging the old detect architecture into the new one
- it lets Palette learn from new runs before committing to more elaborate
  backfill semantics

## Canonical Layout

```text
refined_detect_runs/
  attrs:
    latest                                  "<run_name>"
  <run_name>/
    attrs:
      refined_storage_semantics             "sparse_instances_v1"
      source_detect_run                     "<detect_run>"
      source_detect_method                  "<method>"                  # optional
      source_quality_run                    "<quality_run>"             # optional
      created_at_utc                        ISO-8601
      method                                "<writer method>"
      row_sort_order                        ["frame_indices", "refined_row_ids"]
      source_kind_code_map                  {...}
      source_detection_decision_code_map    {...}                       # if source_detections exists
      detect_review_status                  {...}                       # run-level review payload
      summary_statistics                    {...}
      stage_provenance                      {...}                       # existing Palette pattern

    source_detections/                      # exact bound raw candidate rowset + curation fields
      source_detect_row_index               (M,) int32
      frame_indices                         (M,) int32
      bbox_norm_coords                      (M, 4) float64
      bbox_img_xyxy                         (M, 4) float64             # recommended when available
      confidence_scores                     (M,) float32               # optional
      class_ids                             (M,) int32                 # optional
      decision_codes                        (M,) int8
      resolved_refined_row_id               (M,) int64                 # -1 if no surviving refined row
      reason_bytes                          (M, width) uint8           # recommended
      reason                                (M,) string                # optional mirror
      review_notes                          (M,) string                # optional

    instances/
      refined_row_ids                       (N,) int64
      frame_indices                         (N,) int32
      frame_offsets                         (F + 1,) int64
      bbox_img_xyxy                         (N, 4) float64
      bbox_norm_coords                      (N, 4) float64
      source_kind_codes                     (N,) int8
      manual_edit_flags                     (N,) bool
      confidence_scores                     (N,) float32               # optional
      class_ids                             (N,) int32                 # optional
      source_detect_row_index               (N,) int32                 # optional; -1 if none
      frame_counts                          (F,) int32                 # recommended compatibility summary
      reason_bytes                          (N, width) uint8           # recommended
      reason                                (N,) string                # optional mirror
      review_notes                          (N,) string                # optional

    projections/                            # optional derived outputs; not canonical authoring
      <projection_name>/
        ...
```

## Canonical Authoring Surface

The canonical authoring surface is:

- `refined_detect_runs/<run>/instances`

Each row represents one curated detection instance that Palette wants downstream
consumers to treat as real.

The canonical row key is:

- `refined_row_id`

The canonical grouping key is:

- `frame_index`

There is no required canonical `entity_id` or per-frame slot identity at this
stage.

The bound candidate surface for that run is:

- `refined_detect_runs/<run>/source_detections`

It is not a second detector output. It is the refinement-local mirror of the
exact `detect_runs/<source_detect_run>` candidate rowset that this refined run
is curating.

For v1, this statement applies only to the bound raw detect run. It does not
mean that legacy sparse subgroup contents must be projected back into a
candidate-complete `source_detections/` table.

## Row Ordering Rule

`instances/` arrays must be stored sorted by:

1. `frame_indices`
2. `refined_row_ids`

This is a real contract requirement, not an implementation accident.

Why:

- deterministic diffs and reproducible read order
- efficient per-frame slicing
- no dependence on arbitrary append history
- avoids the ambiguity that came with the dense root ordering question

## Required `instances/` Arrays

### `refined_row_ids`

- shape: `(N,)`
- dtype: `int64`
- one stable identifier per curated instance row
- remains stable across rewrites when the same curated row survives

### `frame_indices`

- shape: `(N,)`
- dtype: `int32`
- frame index for each curated instance

### `frame_offsets`

- shape: `(F + 1,)`
- dtype: `int64`
- compressed sparse-row style index into the `instances/` row arrays

Contract:

- rows for frame `f` live in the half-open slice
  `[frame_offsets[f], frame_offsets[f + 1])`
- `frame_offsets[0] == 0`
- `frame_offsets[-1] == N`
- `frame_offsets` must be nondecreasing

This is the main indexing primitive for sparse-by-frame detect access.

### `bbox_img_xyxy`

- shape: `(N, 4)`
- dtype: `float64`
- authoritative image-space geometry

### `bbox_norm_coords`

- shape: `(N, 4)`
- dtype: `float64`
- normalized mirror of the same curated boxes

### `source_kind_codes`

- shape: `(N,)`
- dtype: `int8`
- machine-readable provenance of the current curated row

Recommended minimum labels:

- `raw_detect`
- `manual`
- `derived`

`derived` covers future non-raw, non-hand-drawn cases such as imported rows or
algorithmically transformed rows that are intentionally preserved in the
curated surface.

### `manual_edit_flags`

- shape: `(N,)`
- dtype: `bool`
- sticky human-touch marker for the curated row

Policy:

- `False`: this row has never been explicitly changed by a human in this run
- `True`: this row was manually added, manually corrected, manually cleared and
  re-added, or materially altered by a human-guided retune flow

## `source_detections/` Subgroup

`source_detections/` is part of the recommended v1 design.

It should contain one row for every raw candidate row in the exact bound
`detect_runs/<source_detect_run>` input.

This subgroup exists so Palette can preserve:

- the full raw candidate rowset considered by refinement
- machine-readable accept/filter/duplicate/manual-clear decisions
- links from raw candidates to surviving curated instances

without restoring peer refined datasets such as `filtered/` or
`interpolated/`.

### Row identity and ordering

Recommended contract:

- row count `M` equals the number of rows in the bound raw detect run
- row ordering matches the raw detect row ordering
- `source_detect_row_index` is the raw detect row index

This makes `source_detections/` a true mirror of the raw candidate table plus
curation fields, not a lossy review summary.

It is intentionally raw-detect-anchored. For v1, writers should not attempt to
merge in candidate rows from legacy `manual` or `interpolated` subgroup
surfaces.

### Required arrays

#### `source_detect_row_index`

- shape: `(M,)`
- dtype: `int32`
- exact row index into `detect_runs/<source_detect_run>`

#### `frame_indices`

- shape: `(M,)`
- dtype: `int32`
- copied from the source raw detect rowset

#### `bbox_norm_coords`

- shape: `(M, 4)`
- dtype: `float64`
- copied or normalized from the source raw detect rowset

#### `decision_codes`

- shape: `(M,)`
- dtype: `int8`
- machine-readable curation disposition of the raw detect candidate

Recommended minimum labels:

- `accepted`
- `filtered`
- `duplicate`
- `manual_clear`

Contract note:

- `duplicate` is a same-frame local-conflict decision
- it means this raw candidate was suppressed in favor of another candidate in
  the same frame that won the local refinement conflict
- it does not mean "same temporal identity as another row in a different frame"
- it should only be used when refinement has enough evidence that the losing
  candidate is not a separate valid subject instance
- unresolved same-frame conflicts should use `decision_code=filtered` with a
  more specific `reason`, for example `unresolved_conflict`

#### `resolved_refined_row_id`

- shape: `(M,)`
- dtype: `int64`
- link to the surviving curated row, if one exists
- use `-1` when no refined instance survived from that raw candidate

### Recommended common arrays

#### `bbox_img_xyxy`

- shape: `(M, 4)`
- dtype: `float64`
- recommended when image-space geometry is available or cheap to derive

#### `confidence_scores`

- shape: `(M,)`
- dtype: `float32`
- copied from the source raw detect rowset when available

#### `class_ids`

- shape: `(M,)`
- dtype: `int32`
- copied from the source raw detect rowset when available

#### `reason_bytes`, `reason`, `review_notes`

- explanatory review payload attached to the candidate row

### Why this subgroup exists

This preserves the useful part of the old detect workflow:

- explicit filtering and review auditability

without restoring the harmful part:

- multiple peer refined detect authoring datasets

## Recommended Common `instances/` Arrays

### `confidence_scores`

- shape: `(N,)`
- dtype: `float32`
- optional detection confidence carried forward when meaningful

### `class_ids`

- shape: `(N,)`
- dtype: `int32`
- optional class label carried forward when meaningful

### `source_detect_row_index`

- shape: `(N,)`
- dtype: `int32`
- backlink into the source raw detect rowset
- use `-1` when the curated row has no one-to-one raw detect source, for
  example a fully manual addition

V1 note:

- `-1` is the expected representation for clean manual additions that do not
  correspond to a specific raw detect candidate

### `frame_counts`

- shape: `(F,)`
- dtype: `int32`
- recommended compatibility summary array
- must equal `np.diff(frame_offsets)`

### `reason_bytes` and `reason`

- explanatory labels only
- same Palette convention as other refined stages
- consumers may display them, but should not need to parse them to know whether
  a row is canonical

### `review_notes`

- optional free-text row note surface
- for human audit and tooling support

## Root Attr Semantics

Recommended required attrs on `refined_detect_runs/<run>`:

- `refined_storage_semantics = "sparse_instances_v1"`
- `source_detect_run`
- `created_at_utc`
- `method`
- `row_sort_order = ["frame_indices", "refined_row_ids"]`
- `source_kind_code_map`
- `source_detection_decision_code_map`
- `summary_statistics`

Recommended common attrs:

- `source_detect_method`
- `source_quality_run`
- `detect_review_status`
- `stage_provenance`

Review should remain primarily a run-level concept here, consistent with the
existing keypoint and subject-mask patterns.

## Write Rules

### `refine_detect`

`refine_detect` should:

- read raw `detect_runs/<run>`
- mirror the exact raw candidate rowset into `source_detections/`
- decide which candidates become curated refined instances
- write sorted sparse `instances/`
- attach candidate-level decision state on `source_detections/`
- avoid emitting interpolation-era peer datasets

### `detect_review`

`detect_review` should edit the sparse curated surface in place:

- update `source_detections/decision_codes`, `reason`, and
  `resolved_refined_row_id` when review changes candidate disposition
- manual correction updates an existing instance row
- manual addition appends a new instance row with a new `refined_row_id`
- manual clear removes the instance row from `instances/` and records the
  decision in `source_detections/` when applicable
- row order and `frame_offsets` must be refreshed after structural changes

Current implementation status:

- fixed-ROI refined review is implemented for one slot per `(frame, arena_id)`
  when subdish masks or arena-assignment metadata provide the arena layout
- the legacy one-slot-per-frame review path still exists for single-subject
  runs
- unconstrained multiple curated detections inside the same arena/ROI are still
  outside the current review UI

V1 compatibility rule:

- if review is operating on legacy subgroup-backed data that does not preserve a
  reliable raw candidate backlink, the writer does not need to synthesize a
  perfect `source_detections/` history from that subgroup
- the sparse v1 contract should stay correct for new raw-detect-anchored runs
  rather than overfitting legacy reconstruction

### `accept_detect_review` and status tooling

Approval remains run-level metadata on `refined_detect_runs/<run>`.

The resolved detect source should conceptually be:

- `refined_instances`

rather than `manual`, `filtered`, or `interpolated`.

For `state=approved` and `intended_use=training`, `accept_detect_review` and
interactive `detect_review` approval also materialize a detection-data profile
from the resolved source. Current sparse runs should therefore produce:

- `analysis/detection_profile_runs/<profile_run>/attrs["source_detection_path"]`
  equal to `refined_detect_runs/<run>/instances`
- source-content fingerprint attrs for the profiled arrays
- run-lineage fingerprint attrs for the profile run itself
- a best-effort registry projection update when the Zarr is registered

This keeps save/edit mutable, keeps approval explicit, and makes the approved
training surface queryable without requiring a separate manual registry sync.

## Consumer Rules

### General detect readers

Readers that want the canonical curated detect surface should read:

- `refined_detect_runs/<run>/instances`

They should iterate sparse rows directly, grouped by `frame_offsets`.

Readers that want full candidate-level auditability should inspect:

- `refined_detect_runs/<run>/source_detections`

### Crop

Crop should treat the sparse refined instance rowset as the source detection
rowset.

Current single-subject workflows may still enforce:

- at most one curated instance per frame, or
- at most one curated instance per `(frame, arena)` after arena assignment

But those are workflow constraints, not storage constraints.

### Tracking

Tracking should bind to the exact sparse refined instance rowset.

`tracking_runs/<run>/source_row_indices` should index into:

- `refined_detect_runs/<run>/instances`

not into a dense slot table.

### Dense projections

Consumers that truly need dense state should read a derived projection, for
example:

- `frame x track`
- `frame x arena`
- single-subject per-frame convenience views

Those projections should be produced explicitly and should never replace the
canonical sparse instance surface.

## Migration Shape

Recommended migration path:

1. Introduce `source_detections/` plus sparse `instances/` as the canonical
   future detect layout.
2. Switch crop, visualization, training, and tracking readers to the sparse
   instance surface.
3. Keep legacy dense-root reads only as optional archive-compatibility fallback.

Migration priority:

- prioritize correctness for new raw-detect-anchored refined runs
- treat legacy subgroup-derived sparse views as best-effort compatibility, not
  as the main contract target

## Relationship To Current Implementation

The active implementation now uses this sparse-first contract directly:

- `instances/` is canonical
- `source_detections/` is the candidate-audit surface
- multi-subject support should grow from sparse instances plus tracking, not
  from denser slot semantics at the detect stage
