# Preferred Detect / Crop Phase 1: Manual Promotion and ROI Mapping

<!-- design-meta
status: draft
last_updated: 2026-04-06
-->

> Superseded on 2026-04-07 by the refined-detect collapse.
> `refined_detect_runs/<run>` is now the canonical curated detect surface.
> Keep this note only for historical context around the retired preferred-layer
> experiment.

## Purpose

Define the first implementation phase for `preferred_detect_runs` and
`preferred_crop_runs`.

This phase is intentionally narrow. It is not trying to solve all track-aware
review or all dense curated detect/crop behavior at once.

It is meant to answer one concrete need:

- how to make saved manual detections first-class rows with canonical
  full-image geometry and explicit ROI/global mapping
- while preserving full-frame-first editing as the primary consumer model
- without redefining existing sparse provenance stages in place

Related notes:

- [preferred_detect_crop_phase1_schema_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/preferred_detect_crop_phase1_schema_checklist.md)
- [preferred_detect_crop_phase1_module_plan.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/preferred_detect_crop_phase1_module_plan.md)
- [preferred_detect_crop_runs_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/preferred_detect_crop_runs_design.md)
- [track_identity_target_architecture.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/track_identity_target_architecture.md)
- [crimson_detect_bbox_read_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/crimson_detect_bbox_read_contract.md)
- [crimson_refined_detect_manual_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/crimson_refined_detect_manual_contract.md)
- [crop_live_view_vs_materialized_stream_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/crop_live_view_vs_materialized_stream_design.md)

## Short Answer

Phase 1 should introduce:

- `preferred_detect_runs/<run>`
- `preferred_crop_runs/<run>`

with one primary promise:

- a saved manual detection becomes a first-class curated row with stable
  identity, canonical full-image bbox geometry, and explicit ROI mapping

This phase should not yet require:

- full multi-entity dense curation across all frames
- track-aware review indexes
- complete downstream regeneration of keypoints or masks
- replacing existing `refined_detect_runs` or `crop_runs`

## Core Rule

For saved manual detections:

- full-image coordinates are canonical
- ROI-local views are derived
- ROI mapping must be explicit
- sparse provenance remains preserved

That means Palette should not force Crimson or other tools to treat a crop
patch as the canonical truth for a saved manual box.

## Why A New Preferred Layer

Current stage meanings are still useful and should remain intact:

- `detect_runs`: raw sparse provenance
- `refined_detect_runs`: sparse corrected provenance
- `crop_runs`: sparse crop geometry and optional materialized ROI pixels

These are not the best consumer-facing edit surface for full-frame tools.

The preferred layer should exist to answer:

- what is the canonical curated detection row for this frame/entity slot
- what is its full-image geometry
- what ROI mapping corresponds to that row
- what downstream artifacts are still missing or pending

This keeps provenance and curation separate.

Phase-1 Palette-side read resolution for this preferred-first behavior lives in:

- `src/fisheye/shared/preferred_detect_resolution.py`

## Phase 1 Scope

Phase 1 should support:

- promoting saved manual detections into preferred rows
- storing canonical full-image bbox geometry
- storing explicit ROI mapping derived from crop policy
- exposing a stable row identity for downstream consumers
- recording explicit downstream artifact state

Phase 1 should not yet try to provide:

- dense preferred rows for every frame in every archive
- final multi-entity `entity_id` assignment policy
- cross-frame review queues by track
- automatic downstream regeneration after every manual detect edit

## Proposed `preferred_detect_runs/<run>`

### Role

Canonical curated detect rows for editing and downstream consumption.

Phase 1 target:

- rows exist for promoted manual detections
- existing non-manual detections may remain sparse-backed until later phases

### Canonical row fields

Each preferred detect row should have at minimum:

- `frame_index`
- `preferred_row_id`
- `entity_id` or `entity_slot`
- `bbox_img_xyxy`
- `bbox_norm_coords`
- `status_code`
- `source_kind`
- `source_sparse_group`
- `source_sparse_row_index`
- `review_state`
- `review_notes` or equivalent operator annotation payload

Recommended semantics:

- `preferred_row_id` is stable within the run
- `bbox_img_xyxy` is the canonical edit/write geometry
- `bbox_norm_coords` is a derived normalized mirror
- `source_kind` distinguishes:
  - `raw_detect`
  - `refined_detect`
  - `manual_promoted`
- `status_code` distinguishes:
  - `present`
  - `missing`
  - `pending`
  - `rejected`
  - `not_generated`

Phase 1 may keep this row store sparse if necessary, as long as row identity is
stable and full-image geometry is canonical.

## Proposed `preferred_crop_runs/<run>`

### Role

Canonical ROI mapping layer corresponding to preferred detect rows.

The important point is that ROI mapping is explicit and does not need to be
inferred from consumer heuristics.

### Canonical row fields

Each preferred crop row should have at minimum:

- `preferred_row_id`
- `frame_index`
- `entity_id` or `entity_slot`
- `roi_offset_xy_full`
- `roi_size_wh`
- `bbox_img_xyxy`
- `crop_policy_name`
- `source_preferred_detect_row_id`
- optional `materialized_crop_row_index`

Optional but desirable:

- `roi_offset_xy_ds`
- `roi_transform_type`
- `roi_transform_matrix` when mapping is more complex than translation

### Canonical invariant

Every preferred crop row must be projectable to full-image space without
guesswork.

For phase 1, translation-based ROI placement is enough:

- top-left offset in full-image pixels
- width/height

If later crop policy adds rotation, scaling, or asymmetric padding semantics,
that must be recorded explicitly rather than inferred.

## Manual Detection Promotion

### Before save

A new manual box drawn in Crimson is just scene geometry.

It is not yet a first-class Palette row because it may not have:

- a stable row identity
- a backing ROI mapping
- downstream artifact status

### After save

Promotion should do the following atomically at the preferred-layer level:

1. Create a preferred detect row.
2. Assign `preferred_row_id`.
3. Record canonical full-image bbox geometry.
4. Apply crop policy to derive ROI mapping.
5. Create the corresponding preferred crop row.
6. Mark downstream artifacts as `pending`, `missing`, or `not_generated`.

What should not happen:

- leaving the saved manual row as only a bbox inside a sparse manual subgroup
- forcing consumers to derive ROI placement from bbox alone
- making manual rows structurally different from non-manual curated rows

## Crop Policy For Promoted Manual Detections

Phase 1 should make crop policy explicit.

Recommended rule:

- bbox is object-local tight geometry
- ROI is the context window used for patch-local consumers

Minimum crop policy behavior:

- grow bbox by configured margin
- optionally square ROI
- clamp to image bounds
- record the exact resulting ROI mapping

This should be recorded as data, not just implied by code:

- `crop_policy_name`
- `crop_policy_version` or parameter hash
- resulting ROI offset and size

## Downstream Artifact State

Promoted manual rows do not need immediate keypoints or masks to be valid.

But their absence must be explicit.

Recommended phase-1 per-row downstream status fields:

- `keypoints_state`
- `subject_mask_state`
- `eye_mask_state`
- `swim_bladder_state`

Recommended values:

- `not_generated`
- `pending`
- `missing`
- `present`
- `not_applicable`

This is better than treating manually promoted rows as incomplete special cases.

## Relationship To Existing Sparse Stages

Phase 1 should preserve these behaviors:

- `refined_detect_runs/<run>/<manual_group>` remains the sparse provenance write
  surface for manual detect review compatibility
- `preferred_detect_runs/<run>` becomes the canonical curated consumer surface
- `crop_runs/<run>` remains sparse crop provenance/materialization
- `preferred_crop_runs/<run>` becomes the canonical ROI/global mapping surface

This means one manual edit may write to both:

- sparse refined provenance
- preferred curated surface

That duplication is acceptable in phase 1 because the two stages answer
different questions.

## Crimson Expectations In Phase 1

After this phase, Crimson should be able to assume:

- full-frame editing is primary
- saved manual detections become first-class curated rows
- every curated row has explicit ROI/global mapping
- crop preview is a derived convenience view, not the source of truth

Crimson should not have to:

- treat unsaved manual boxes and saved manual boxes as the same class
- infer ROI mapping from bbox-only state
- special-case manual rows forever

## Suggested Storage Shape

Phase 1 does not need the final dense matrix shape yet.

A row-oriented run is acceptable initially if it provides:

- stable `preferred_row_id`
- explicit `frame_index`
- explicit `entity_id`
- canonical full-image bbox
- explicit ROI mapping

Dense `(frame, entity)` materialization can come in a later phase.

This keeps the first implementation tractable while preserving the long-term
target architecture.

## Non-Goals

Phase 1 should not try to:

- solve general multi-entity preferred assignment
- add full review-index materialization by track/domain
- redefine `refined_detect_runs` to be the preferred layer
- make `crop_runs` dense
- require immediate regeneration of keypoints or masks on promotion

## Implementation Order

1. Define the preferred detect row schema and required attrs.
2. Define the preferred crop row schema and ROI mapping attrs.
3. Implement manual detection promotion into those stages.
4. Record downstream artifact states for promoted rows.
5. Teach Crimson and other consumers to prefer preferred rows when present.
6. Only then extend review/navigation indexes on top of the preferred layer.

## Open Questions

- Should `preferred_row_id` be run-local integer, UUID-like string, or both?
- Is `entity_id=0` sufficient for the first single-subject phase, or should
  phase 1 already reserve multi-entity shape explicitly?
- Should preferred rows carry image-space polygon/contour placeholders for
  future masks, or is bbox + ROI mapping enough initially?
- Should sparse manual write and preferred promotion happen in one command or a
  sparse-write-then-promote two-step flow?

## Bottom Line

If Palette wants Crimson to be full-frame-first without fragile manual-detection
special cases, phase 1 should not start with review indexes.

It should start by making saved manual detections first-class curated rows with:

- canonical full-image bbox geometry
- explicit ROI/global mapping
- stable row identity
- explicit downstream artifact states

That is the minimum contract needed before the broader cross-frame review model
becomes worth implementing.
