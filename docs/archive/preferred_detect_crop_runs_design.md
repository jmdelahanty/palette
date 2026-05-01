# Preferred Detect / Crop Runs Design

<!-- design-meta
status: draft
last_updated: 2026-04-06
-->

> Superseded on 2026-04-07 by the refined-detect collapse.
> `refined_detect_runs/<run>` is now the canonical curated detect surface.
> Keep this note only for historical context around the retired preferred-layer
> experiment.

## Purpose

Define a forward-looking canonical design for dense, ID-assigned detect/crop
artifacts that preserve stable downstream row identity without rewriting the
meaning of existing sparse provenance stages.

This note is meant to answer:

- how to represent one logical detection/crop row per frame even when raw
  detections are sparse
- how to handle multiple candidate boxes in a frame
- how to model stable row identity for downstream stale propagation
- why this should be a new stage family rather than a silent redefinition of
  `refined_detect_runs` or `crop_runs`

Related notes:

- [preferred_detect_crop_phase1_manual_promotion_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/preferred_detect_crop_phase1_manual_promotion_design.md)
- [preferred_detect_crop_phase1_schema_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/preferred_detect_crop_phase1_schema_checklist.md)
- [preferred_detect_crop_phase1_module_plan.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/preferred_detect_crop_phase1_module_plan.md)
- [repo_wide_staleness_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_policy.md)
- [repo_wide_staleness_workflow_edge_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_workflow_edge_checklist.md)
- [crop_live_view_vs_materialized_stream_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/crop_live_view_vs_materialized_stream_design.md)
- [detection_refinement_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/detection_refinement_workflow.md)

## Short Answer

Do not make raw or sparse refined detect/crop runs dense in place.

Instead, introduce new canonical curated stage families:

- `preferred_detect_runs/<run>`
- `preferred_crop_runs/<run>`

These runs should be:

- dense over `(frame_index, entity_id)`
- explicit about `present` versus `missing`
- sourced from sparse refined/raw provenance
- the stable row-identity surface used by downstream training, review, and
  stale propagation

Raw and sparse refined stages should continue to answer:

- what candidate detections existed
- where they came from
- which sparse row was manually corrected or interpolated

Preferred dense stages should answer:

- which one logical slot is chosen for each `(frame, entity_id)`
- which sparse source row backs that choice
- whether the slot is present, missing, ambiguous, or explicitly negative

## Why Not Redefine `refined_detect_runs` Or `crop_runs`

### Sparse provenance and dense curated selection are different artifacts

`refined_detect_runs` currently means sparse corrected provenance:

- `filtered`
- `interpolated`
- `manual`

That is valuable because it preserves:

- duplicates
- alternates
- manual corrections
- interpolation provenance
- the true sparse candidate set

A dense curated timeline answers a different question:

- what is the one chosen row for entity `E` at frame `F`?

Changing `refined_detect_runs` itself to be dense would collapse those two
meanings into one stage and make provenance less clear.

### A missing crop is not the same thing as a real sparse crop row

`crop_runs` currently represents actual ROI geometry and, optionally, actual
materialized ROI pixels. A frame with no accepted detection has no real crop
row in the sparse provenance sense.

Dense missing semantics are still useful, but they should live in a stage whose
meaning is:

- one logical slot per `(frame, entity_id)`

not:

- one row per actual sparse crop that existed

### Forward architecture should make the semantic split explicit

The clean target is:

- `detect_runs`: raw sparse provenance
- `refined_detect_runs`: sparse corrected provenance
- `preferred_detect_runs`: dense curated detect selection
- `crop_runs`: sparse crop geometry/materialization
- `preferred_crop_runs`: dense curated crop geometry keyed to preferred detect

That is easier to reason about than overloading one existing stage with two
different jobs.

## Core Design Rule

Dense row identity should be keyed by:

- `frame_index`
- `entity_id`

not by:

- sparse detect row index
- sparse crop row index
- final tracking ID

This keeps the dense stage stable even when sparse source rows are reselected or
manually corrected.

## Entity Identity

### Current single-subject-per-arena workflow

In the current common case, one archive or arena effectively has one logical
fish slot.

That means:

- `entity_ids = [0]` is often enough
- dense timelines still add value because every frame has an explicit state

### Future multi-entity workflow

For future multi-subject or multi-slot workflows, `entity_id` should mean:

- a curated logical subject slot within the archive or arena

It should not mean:

- "the third sparse detect row seen so far"
- "whatever tracking later decided"

Recommended policy:

- `entity_id` is stable within a preferred run
- entity assignment may be seeded from current arena/track context, but the
  dense preferred stage owns the final per-frame slot mapping
- identity-breaking changes should create a new preferred run or invalidate
  downstream rows, not silently remap history inside one run

## Proposed `preferred_detect_runs/<run>`

### Purpose

Represent one curated detection slot per `(frame, entity_id)`.

### Core arrays

Recommended minimum arrays:

- `entity_ids`: `(n_entities,) int32`
- `present`: `(n_frames, n_entities) bool`
- `status_code`: `(n_frames, n_entities) int8`
- `bbox_norm_coords`: `(n_frames, n_entities, 4) float32`
- `scores`: `(n_frames, n_entities) float32`
- `source_row_index`: `(n_frames, n_entities) int32`
- `source_group_code`: `(n_frames, n_entities) int8`
- `reason_bytes` or `reason`: `(n_frames, n_entities)`

Recommended status vocabulary:

- `present`
- `missing`
- `explicit_negative`
- `ambiguous`
- `suppressed_duplicate`

Missing rows should use:

- `present = false`
- `source_row_index = -1`
- `bbox_norm_coords = NaN`

Do not encode absence as zeroed coordinates.

### Suggested attrs

- `source_detect_run`
- `source_refined_detect_run`
- `source_sparse_groups`
- `entity_scope`
- `entity_assignment_policy`
- `preferred_selection_policy`
- `total_frames`
- `summary_statistics`
- review status payload, separate from row status
- provenance / environment metadata

### Source mapping

Each dense cell should point back to the sparse source it selected:

- `source_group_code` identifies whether the chosen row came from
  `filtered`, `interpolated`, `manual`, or another sparse source
- `source_row_index` identifies the exact sparse row

That keeps auditability without forcing the dense stage to store all candidates.

## Proposed `preferred_crop_runs/<run>`

### Purpose

Represent one curated crop slot per `(frame, entity_id)` while keeping crop
geometry canonical and ROI pixels optional.

### Core arrays

Recommended minimum arrays:

- `entity_ids`: `(n_entities,) int32`
- `present`: `(n_frames, n_entities) bool`
- `status_code`: `(n_frames, n_entities) int8`
- `roi_coordinates_full`: `(n_frames, n_entities, 2) int32`
- `roi_coordinates_ds`: `(n_frames, n_entities, 2) int32`
- `bbox_norm_coords`: `(n_frames, n_entities, 4) float32`
- `source_preferred_detect_row_index`: `(n_frames, n_entities) int32`
- `source_sparse_crop_row_index`: `(n_frames, n_entities) int32`

If crop size can vary, add per-row size arrays; otherwise keep ROI size as run
attrs as today.

### Pixel materialization policy

Preferred crop pixels should not default to a giant dense image tensor full of
blanks.

Recommended design:

- dense geometry lives in the main run
- optional sparse materialization lives in a dedicated subgroup such as
  `materialized/`
- the dense run stores `source_sparse_crop_row_index` or
  `dense_to_materialized_row_index`

Example:

- `materialized/roi_images`: `(n_present_rows, h, w)`
- `materialized/dense_row_index`: `(n_present_rows, 2)` storing `(frame, entity)`

This preserves dense logical identity without paying dense-image storage cost.

## Multiple Detections In One Frame

This is the main reason to make the dense representation ID-assigned rather than
simply padded sparse rows.

### Example

Frame 100 has two fish and three sparse candidate detections:

- sparse row 17: fish A
- sparse row 18: fish B
- sparse row 19: duplicate false positive near fish A

Preferred detect resolves this to:

- `(frame=100, entity=0)` -> source row 17
- `(frame=100, entity=1)` -> source row 18

The duplicate row 19 remains available in sparse provenance but is not chosen
as the preferred dense row.

If frame 101 has only one accepted fish:

- `(frame=101, entity=0)` -> present
- `(frame=101, entity=1)` -> missing

### Important rule

The dense stage should represent one chosen slot per entity, not one row for
every raw candidate.

Raw sparse stages remain the place where all candidates live.

## Relationship To Review

Preferred stages should be curated working surfaces, but review must remain a
separate concept.

That means:

- dense row `status_code` is not review state
- run-level review attrs remain separate
- `stale` remains separate from both row status and review state

## Staleness / Invalidation Rules

### Row-stable correction

If a sparse refined-detect or preferred-detect correction keeps the same
`(frame, entity_id)` identity:

- downstream preferred crop/keypoint/mask rows should be marked `stale`
- targeted row-local repair is allowed

Examples:

- bbox move/resize
- source sparse-row reselection that still maps to the same logical entity

### Identity-breaking correction

If the correction changes which entity the row belongs to, or changes the slot
structure itself:

- downstream rows should be invalidated to `missing`
- a new preferred run may be more appropriate than row-local repair

Examples:

- add/delete/split/merge
- entity re-assignment across historical frames
- changing `n_entities` or entity-slot meaning inside the run

## Migration Guidance

### Phase 1

Introduce `preferred_detect_runs` as a derived dense stage sourced from current
`refined_detect_runs` review resolution.

### Phase 2

Introduce `preferred_crop_runs` sourced from `preferred_detect_runs` plus the
canonical crop geometry policy.

### Phase 3

Allow downstream stages to explicitly choose sparse or preferred sources:

- keypoints
- eye masks
- subject masks
- swim bladder

### Phase 4

Once preferred stages are stable and downstream tooling is migrated:

- prefer preferred stages for row-stable stale handling
- keep sparse stages for provenance and audit

## Non-Goals

- Do not rewrite existing `detect_runs` or `crop_runs` semantics in place.
- Do not force dense ROI image tensors by default.
- Do not use zeroed geometry as the representation for absence.
- Do not collapse dense entity identity to sparse row number.
- Do not make tracking the only source of entity identity for preferred detect.

## Recommendation

If implementation starts soon, the best sequence is:

1. write a formal contract for `preferred_detect_runs`
2. define `entity_id` policy for current single-subject-per-arena archives
3. add a first writer that derives preferred detect from existing refined
   detect review resolution
4. add `preferred_crop_runs` as dense geometry with optional sparse
   materialization
5. then wire downstream stale handling to the preferred stages
