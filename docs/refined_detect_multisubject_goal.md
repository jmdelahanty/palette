# Refined Detect Multi-Subject Goal

<!-- design-meta
status: draft
last_updated: 2026-04-09
-->

Purpose: define the long-term target architecture for refined detect storage
once Palette supports true multi-subject detect and tracking, while preserving
the current single-subject workflow until a deliberate migration is chosen.

Current-state note (2026-07-09): the sparse `instances/` target described here
is now active. See `docs/refined_detect_sparse_instances_schema.md` and
`docs/instance_track_subject_identity_contract.md`. This file is retained as
the migration rationale.

## Scope

This design decides:

- the desired canonical authoring model for refined detect
- how refined detect should relate to raw detect, tracking, and dense exports
- how manual review, provenance, and filtered decisions should survive the
  transition
- why detect should not blindly mirror keypoint and mask storage semantics

This design does not:

- replace the active 2026-04-07 detect contract today
- finalize every array name or exact Zarr path
- define a destructive migration schedule
- redefine the current `tracking_runs` contract

Concrete schema follow-up:

- `docs/refined_detect_sparse_instances_schema.md`

## Current State

As of 2026-04-07, the active detect contract is documented in
`docs/refined_detect_collapse_v2.md`.

Current implementation summary:

- `detect_runs/<run>` is raw detector output.
- `refined_detect_runs/<run>` is the canonical curated detect surface.
- current detect review edits sparse refined surfaces in place.

This is a valid short-term simplification for the current workflow, and manual
review now supports fixed sub-arena slotting when ROI definitions exist. It is
still not the preferred long-term target for true multi-subject support within
one arena/ROI.

## Historical Context

Palette's earlier detect refinement model was not a dense slot table.

Historically:

- raw detect stayed sparse
- refined detect wrote sparse `filtered/` and `interpolated/` groups
- manual review wrote another sparse subgroup under the refined run

That old model had too many parallel surfaces, but it did preserve one useful
property: detect authoring was fundamentally sparse and instance-based rather
than slot-based.

## Comparison With Keypoints And Masks

Keypoints and subject masks are canonical editable refined stages today, but
they are not the right template for detect storage.

Why:

- `refined_keypoints_runs/<run>` is dense over already-selected ROI rows
- `refined_subject_masks_runs/<run>` is dense over already-selected ROI rows
- both stages sit downstream of crop, so the row identity problem has already
  been solved upstream

Detect is different:

- detect is where the instance rowset is first created
- detect should not have to invent a dense `(frame, slot)` identity before
  tracking or explicit arena/entity assignment exists

So the correct lesson from keypoints and masks is:

- keep one canonical refined/editable detect surface
- keep machine-readable review/provenance fields
- do not require detect itself to be dense if the underlying identity is not
  stable yet

## Goal Summary

The long-term goal is:

1. `detect_runs/<run>` remains immutable raw detector output.
2. `refined_detect_runs/<run>` carries:
   - `source_detections/` as the refinement-local mirror of the exact bound raw
     detect candidate rowset
   - `instances/` as the canonical curated sparse instance surface
3. true temporal identity belongs in tracking, not in the basic refined-detect
   row key.
4. dense `frame x track` or `frame x entity` layouts become derived
   materializations for consumers that explicitly need them.
5. interpolation should not return as a first-class detect authoring stage.
6. new detect work should not reintroduce parallel `filtered`,
   `interpolated`, and `manual_*` authoring datasets.

This is intentionally closer to the sparse-instance-plus-track separation used
by SLEAP than to the current dense refined detect root.

## Target Conceptual Model

The canonical detect authoring surface should be sparse and instance-oriented.

Conceptually:

```text
detect_runs/<run>/                     # immutable raw detector output

refined_detect_runs/<run>/             # canonical curated detect authoring surface
  source_detections/                   # exact bound raw candidate rowset + review decisions
    source_detect_row_index
    frame_indices
    bbox_norm_coords
    decision_codes
    resolved_refined_row_id
    reason_bytes
    reason
    review_notes                       # optional

  instances/                           # sparse refined rows
    refined_row_ids
    frame_indices
    bbox_img_xyxy
    bbox_norm_coords
    confidence_scores                  # optional
    class_ids                          # optional
    source_detect_row_index            # optional backlink into raw detect
    source_kind_codes                  # raw_detect | manual | derived/other
    manual_edit_flags                  # sticky human-touch marker
    reason_bytes
    reason
    review_notes                       # optional

tracking_runs/<run>/                   # temporal identity / assignment layer
  track_ids
  frame_indices
  source_row_indices
  ...

derived exports or caches              # optional dense materializations
  frame x track
  frame x arena
  frame x slot
```

Important consequence:

- the canonical refined detect row key is not `(frame_index, entity_id)`
- it is a stable sparse row identity for each curated instance
- any dense slot namespace should be introduced only by a workflow that truly
  owns that namespace

## Filtered Decisions And Missing Frames

The target design does not mean "bring back filtered/interpolated groups."

Instead:

- raw detect continues to hold all raw detector candidates
- `source_detections/` mirrors that exact candidate rowset inside the refined
  run and records curation decisions against it
- v1 should anchor `source_detections/` only to the bound raw detect run; it
  should not try to perfectly reconstruct legacy `manual` or `interpolated`
  subgroup semantics
- refined detect holds the curated instance rows that Palette wants downstream
  consumers to use
- filtered decisions should be represented as machine-readable metadata, not as
  separate parallel refined datasets

For example:

- `source_kind_codes` should continue to say where a refined row came from
- `manual_edit_flags` should continue to say whether a human touched it
- `reason` should continue to explain why a row was corrected, filtered, or
  otherwise notable
- manual additions with no one-to-one raw candidate should live only in
  `instances/`, with `source_detect_row_index = -1`

For true "absence" semantics:

- absence should usually mean there is no refined instance row for that frame
  and track/arena yet
- if a downstream dense export needs explicit `missing` states, that export
  should materialize them from sparse refined instances plus the chosen track or
  slot namespace

## Why This Is Better For Multi-Subject Work

A dense refined root works best when all of these are true:

- one stable slot exists per frame
- "no detection" must be represented directly at the detect authoring layer
- multiple detections in one frame are either impossible or already resolved

That is not the long-term multi-subject case.

For multi-subject detect:

- there may be multiple valid detections in one frame
- there may be no natural per-frame slot ordering
- track identity may be unavailable at authoring time
- forcing a dense slot table too early would make detect storage depend on an
  arbitrary slotting policy

The sparse refined instance model avoids that trap.

## Relationship To Tracking

This design does not collapse detect and tracking into one stage.

Policy:

- refined detect owns curated per-frame instances
- tracking owns temporal identity assignment across frames
- arena-specific or slot-specific dense views should be derived from refined
  detect plus tracking or arena assignment, not treated as the canonical detect
  authoring surface

That keeps detect review focused on instance correctness and leaves temporal
identity to the stage that actually solves temporal identity.

## Migration Principles

When Palette moves toward this target, the migration should follow these rules:

1. Keep `detect_runs/<run>` immutable.
2. Do not restore interpolation as a first-class detect stage.
3. Do not reintroduce multiple peer refined detect authoring datasets.
4. Preserve sticky manual provenance such as `manual_edit_flags`.
5. Keep old archives readable without requiring destructive rewrite.
6. Provide explicit derived dense materializations only where a consumer
   genuinely benefits from them.
7. For v1, prefer a clean raw-detect-anchored sparse model over trying to make
   legacy subgroup backfills behave like first-class sparse authoring inputs.

## Near-Term Implication

The active dense refined root is acceptable as a current single-subject bridge,
but it should be treated as an implementation waypoint rather than the final
architecture for multi-subject detect.

The desired steady state is:

- explicit `source_detections/` for candidate-level auditability
- sparse canonical refined detect instances
- separate temporal identity assignment
- dense views only as derived projections
