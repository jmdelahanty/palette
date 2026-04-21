# Preferred Detect / Crop Phase 1 Module Plan

<!-- design-meta
status: draft
last_updated: 2026-04-06
-->

> Superseded on 2026-04-07 by the refined-detect collapse.
> `refined_detect_runs/<run>` is now the canonical curated detect surface.
> Keep this note only for historical context around the retired preferred-layer
> experiment.

## Purpose

Turn the phase-1 preferred detect/crop design into a concrete implementation
plan:

- exact existing Palette modules to change
- new helper modules to introduce
- safest migration order
- what should remain unchanged in phase 1

This note is deliberately implementation-oriented. It is the bridge between:

- [preferred_detect_crop_phase1_manual_promotion_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/preferred_detect_crop_phase1_manual_promotion_design.md)
- [preferred_detect_crop_phase1_schema_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/preferred_detect_crop_phase1_schema_checklist.md)

## Short Answer

Phase 1 should preserve the current Crimson manual-write contract and add a new
Palette-side promotion step:

1. Crimson continues writing sparse manual detections under
   `refined_detect_runs/<latest>/<manual_group>`.
2. Palette promotes those saved manual rows into:
   - `preferred_detect_runs/<run>`
   - `preferred_crop_runs/<run>`
3. Crimson and other consumers can then read preferred rows as the canonical
   full-frame-first curated surface.

This avoids redefining `refined_detect_runs` or `crop_runs` in place while still
giving Crimson a first-class saved-row model.

## Current Status

As of `2026-04-06`, the core Phase 1A Palette-side pieces are now implemented.

Landed:

1. schema/spec additions for:
   - `preferred_detect_runs/<run>`
   - `preferred_crop_runs/<run>`
2. shared preferred helper module:
   - `src/fisheye/shared/preferred_detect_crop.py`
3. shared crop-geometry helper:
   - `src/fisheye/shared/crop_geometry.py`
4. dedicated promotion CLI:
   - `src/fisheye/utils/promote_manual_detections_to_preferred.py`
5. detect-review integration after sparse manual save:
   - `src/fisheye/tune/detect_review.py`
6. shared preferred read resolution:
   - `src/fisheye/shared/preferred_detect_resolution.py`
7. low-risk Palette consumers for preferred-stage reads:
   - `src/fisheye/visualization/visualize_refined_detections.py`
   - `src/fisheye/visualization/detection_visualizer.py`
8. preferred-run inspection utility:
   - `src/fisheye/utils/inspect_preferred_detect_runs.py`

Still intentionally deferred:

- registry indexing/query support for `preferred_*`
- broad consumer migration across the rest of Palette
- dense `(frame, entity)` preferred timelines
- automatic downstream keypoint/mask regeneration
- track-aware review/navigation

## Assessment Against Original Goals

### Goal: keep sparse provenance intact

Status: met.

Implemented behavior still preserves:

- `detect_runs` as raw sparse provenance
- `refined_detect_runs/<run>/<manual_group>` as sparse manual/refined provenance
- `crop_runs` as sparse bulk crop geometry/materialization

The preferred layer was added alongside these stages instead of redefining them.

### Goal: make saved manual detections first-class rows

Status: met for Palette Phase 1A.

Palette now promotes saved manual detections into first-class preferred rows
with:

- stable `preferred_row_id`
- canonical `bbox_img_xyxy`
- explicit ROI mapping in `preferred_crop_runs`
- explicit downstream artifact state initialized to `not_generated`

### Goal: full-image coordinates are canonical

Status: met.

`preferred_detect_runs` now stores canonical `bbox_img_xyxy`, with
`bbox_norm_coords` kept as a derived mirror. Preferred crop rows store
`roi_offset_xy_full` and `roi_size_wh` explicitly.

### Goal: Palette-side preferred-first read path exists

Status: met for low-risk consumers.

Palette now has one shared preferred-read resolver and at least two consumer
surfaces that can use it. This is enough to prove the read contract without
forcing an immediate repo-wide migration.

### Goal: preserve full-frame-first editing as the consumer model

Status: partially met.

The archive-side contract now supports this cleanly, and the main visualizers
can inspect preferred rows. But the broader Crimson-side review/navigation
experience is still future work.

### Goal: avoid forcing dense timelines too early

Status: met.

Phase 1A remains row-oriented and sparse-backed. It does not yet claim to solve
the final dense `(frame, entity)` preferred model described in the long-term
design note.

## Practical Conclusion

The current implementation is doing the job it was supposed to do in Phase 1A:

- it creates a first-class curated preferred layer
- it keeps raw/refined sparse provenance semantics intact
- it makes ROI/global mapping explicit
- it gives Palette a write path, read path, visualization path, and inspection
  path

What is still missing is not the core design. It is the second wave around:

- broader consumer adoption
- registry visibility
- eventual dense preferred timelines
- Crimson-first track-aware workflows

## Why Detect Is Different From Keypoints And Masks

This plan is intentionally detect-specific.

It should not be read as a repo-wide claim that every `refined_*` stage needs a
separate `preferred_*` stage.

Current repo reality is:

- `refined_keypoints_runs` already behaves like the editable curated working
  copy
- `refined_subject_masks_runs` is now the canonical refined authoring surface
  for subject/eye/swim mask work
- historical `refined_eye_masks_runs` has also been edited in place, even
  though the target steady-state is to treat it as a compatibility artifact

Detection is the outlier because `refined_detect_runs` currently means sparse
corrected provenance, not the canonical mutable consumer surface:

- `filtered/` and `interpolated/` remain sparse candidate/provenance groups
- manual corrections are stored in a separate manual subgroup
- downstream consumers resolve the active detect source through subgroup
  selection and review pointers

That is why a separate `preferred_detect_runs` / `preferred_crop_runs` layer is
justified for detect first, without implying that keypoints or masks should be
split the same way.

## Phase 1 Guardrails

- Keep `detect_runs` untouched.
- Keep `refined_detect_runs/<run>/<manual_group>` as the upstream sparse manual
  provenance contract.
- Keep `crop_runs` semantics unchanged in phase 1.
- Do not require dense `(frame, entity)` coverage in phase 1.
- Do not require immediate keypoint or mask regeneration in phase 1.
- Treat `preferred_*` as the curated working surface, not as raw provenance.

## Current Modules That Already Matter

### Existing manual write / detect selection surface

- [detect_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/detect_review.py)
- [refined_detect_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/refined_detect_review.py)
- [set_detect_review_status.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/set_detect_review_status.py)
- [crimson_refined_detect_manual_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/crimson_refined_detect_manual_contract.md)

These already define how manual detections are written and selected today.

### Existing crop geometry / provenance surface

- [crop.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tracking/crop.py)
- [crop_signature.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/crop_signature.py)
- [crop_image_source.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/crop_image_source.py)

These already define ROI placement, crop provenance, and crop read helpers.

### Existing schema / registry surface

- [stage_arrays.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/zarr/stage_arrays.py)
- [schema.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/zarr/schema.py)
- [zarr_structure.md](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/docs/zarr_structure.md)
- [db.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/registry/db.py)
- [registry_stage_complete.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/registry_stage_complete.py)

## Implementation Sequence

## Step 1: Add Preferred Stage Contracts

### Files to change

- [stage_arrays.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/zarr/stage_arrays.py)
- [schema.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/zarr/schema.py)
- [zarr_structure.md](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/docs/zarr_structure.md)

### What to add

- `PREFERRED_DETECT_SPEC`
- `PREFERRED_CROP_SPEC`
- root/group metadata entries for:
  - `preferred_detect_runs`
  - `preferred_crop_runs`
- parent attrs:
  - `latest`
  - optional `latest_manual_promotion`

### Why first

This makes the new preferred stages first-class in the archive contract before
any writer starts producing them.

## Step 2: Extract Shared Preferred Detect / Crop Helpers

### New module

- `src/fisheye/shared/preferred_detect_crop.py`

### Responsibilities

- canonical phase-1 status maps
- parent pointer helpers
- preferred run selection helpers
- `preferred_row_id` allocation / preservation
- summary-statistics builders
- normalization of attrs and source metadata

### Recommended API shape

- `resolve_preferred_detect_run(...)`
- `resolve_preferred_crop_run(...)`
- `create_or_open_preferred_detect_run(...)`
- `create_or_open_preferred_crop_run(...)`
- `assign_preferred_row_ids(...)`
- `build_preferred_detect_summary(...)`
- `build_preferred_crop_summary(...)`

### Why a new module

Phase 1 should not overload [refined_detect_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/refined_detect_review.py),
which is currently about sparse refined-detect group resolution only.

## Step 3: Extract Reusable Crop Geometry Policy

### New module

- `src/fisheye/shared/crop_geometry.py`

### Source to factor from

- `_compute_roi_coordinates(...)` in [crop.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tracking/crop.py)

### Responsibilities

- derive canonical ROI origin and size from full-image bbox geometry
- record translation-based ROI mapping for phase 1
- centralize crop-policy naming / parameter hashing inputs used by both:
  - bulk `crop_runs`
  - preferred manual-promotion flow

### Why extract it

Phase 1 should not import the full crop runtime just to compute ROI placement for
promoted manual detections.

## Step 4: Add A Dedicated Promotion Writer

### New CLI

- `src/fisheye/utils/promote_manual_detections_to_preferred.py`

### New shared writer entrypoint

- `promote_manual_detections_to_preferred(...)` in
  `src/fisheye/shared/preferred_detect_crop.py`

### Responsibilities

- resolve the active refined detect run and manual subgroup
- read sparse manual rows from `refined_detect_runs/<run>/<manual_group>`
- create or update `preferred_detect_runs/<run>`
- create or update `preferred_crop_runs/<run>`
- assign or preserve `preferred_row_id`
- mark downstream artifact states as `not_generated`
- update `latest` pointers
- write summary stats and provenance

### Why a separate CLI first

This lets Palette land the preferred-layer writer without immediately changing
the current Crimson or `detect_review.py` save path.

## Step 5: Integrate Manual Review Save Path

### File to change

- [detect_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/detect_review.py)

### Phase-1 behavior

After the sparse manual subgroup is written successfully:

- optionally call `promote_manual_detections_to_preferred(...)`
- default behavior can be one of:
  - explicit `--promote-preferred`
  - enabled-by-default with `--no-promote-preferred`

### Recommended failure behavior

- sparse manual subgroup write remains the primary success condition
- preferred promotion failures should warn and fail closed
- do not roll back the sparse manual write if preferred promotion fails

### Why this order

It preserves the current manual-write contract while making preferred rows an
incremental addition rather than a hard dependency.

## Step 6: Add Preferred Read Resolution

### New module

- `src/fisheye/shared/preferred_detect_resolution.py`

### Responsibilities

- resolve active preferred detect run
- resolve active preferred crop run
- expose fallback behavior:
  - preferred first when requested
  - sparse refined/raw fallback when preferred is absent

### Docs to update

- [crimson_detect_bbox_read_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/crimson_detect_bbox_read_contract.md)
- [preferred_detect_crop_phase1_manual_promotion_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/preferred_detect_crop_phase1_manual_promotion_design.md)

### Phase-1 read rule

Crimson should be able to:

- continue reading sparse refined/manual detections today
- prefer `preferred_detect_runs` when available
- use `preferred_crop_runs` for explicit ROI/global mapping

Current status:

- shared read resolution is implemented in
  `src/fisheye/shared/preferred_detect_resolution.py`
- low-risk read adoption is now present in
  `src/fisheye/visualization/visualize_refined_detections.py` via
  `--show-preferred`
- broader consumer-by-consumer adoption is still pending

## Step 7: Leave `crop_runs` Bulk Semantics Alone

### Files intentionally not changed in phase 1

- [crop.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tracking/crop.py)
- [crop_image_source.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/crop_image_source.py)

### Explicit decision

Phase 1 should not make `crop_runs` consume `preferred_detect_runs` as its new
source of truth.

Instead:

- `preferred_crop_runs` is the canonical curated ROI mapping layer for promoted
  rows
- `crop_runs` remains the sparse bulk crop/materialization stage

This keeps phase 1 tractable and avoids a wider crop-pipeline migration.

## Step 8: Registry Support Is Phase 1B, Not Phase 1A

### File likely to change later

- [db.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/registry/db.py)

### Recommendation

Do not block phase 1 on registry indexing for `preferred_*`.

Phase 1A should make the archive contract and writer/read path work first.

Phase 1B can then decide whether to add:

- preferred detect quality rows
- preferred crop summary rows
- step-status emission for preferred stages

### Why defer it

Crimson is a direct archive consumer. Registry support is useful, but it is not
the critical path for making saved manual detections first-class rows.

## Recommended New Test Files

### Shared helper tests

- `tests/unit/fisheye/test_preferred_detect_crop.py`
- `tests/unit/fisheye/test_preferred_detect_resolution.py`
- `tests/unit/fisheye/test_crop_geometry.py`

### CLI / integration-style tests

- `tests/unit/fisheye/test_promote_manual_detections_to_preferred.py`

### Existing files likely to gain coverage

- `tests/unit/fisheye/test_stage_arrays.py`
- `tests/unit/fisheye/test_detect_review.py`

### Sandbox guidance

Prefer fake-group / in-memory zarr harnesses for phase-1 coverage. Real-zarr
integration tests can be added later as deferred local validation if needed.

## Migration Order

Recommended landing order:

1. schema/spec additions
2. shared preferred helpers
3. shared crop-geometry helper
4. promotion CLI
5. detect-review integration
6. preferred-read contract updates
7. optional registry support

This order keeps the current system usable at every intermediate step.

## Explicit Non-Goals For Phase 1

- dense `(frame, entity)` preferred timelines
- track-aware review queues
- automatic downstream keypoint/mask generation
- reworking `crop_runs.latest`
- replacing the sparse Crimson manual-write contract

## Open Decisions To Resolve Before Coding

### Preferred run mutability

Recommended phase-1 decision:

- allow the active preferred run to be updated in place
- preserve `preferred_row_id` for stable source keys

Reason:

- preferred runs are curated working-surface artifacts, not raw provenance
- this matches the repo-wide policy of keeping raw runs append-only while
  allowing curated/refined layers to remain editable

### Preferred run naming

Recommended phase-1 default:

- `preferred_detect_manual_<timestamp>`
- `preferred_crop_manual_<timestamp>`

with parent `latest` pointers carrying the active run selection.

### Initial entity policy

Recommended phase-1 default:

- `entity_id = 0` for single-subject archives

### Downstream state defaults

Recommended phase-1 default:

- `keypoints_state = not_generated`
- `subject_mask_state = not_generated`
- `eye_mask_state = not_generated`
- `swim_bladder_state = not_generated`

## Concrete First PR Slice

The smallest useful first PR should include only:

- stage-array and schema additions for `preferred_*`
- `src/fisheye/shared/preferred_detect_crop.py`
- `src/fisheye/shared/crop_geometry.py`
- `src/fisheye/utils/promote_manual_detections_to_preferred.py`
- focused fake-group tests

That is enough to prove:

- a sparse manual subgroup can be promoted
- preferred detect rows are canonicalized
- preferred crop rows record explicit ROI mapping
- no existing raw/refined/crop semantics had to be broken to get there
