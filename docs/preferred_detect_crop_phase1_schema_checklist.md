# Preferred Detect / Crop Phase 1 Schema Checklist

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
checklist:

- exact stage names
- minimum arrays
- minimum attrs
- status vocabularies
- save/promotion flow

This is intentionally a phase-1 checklist, not the final long-term contract for
all preferred detect/crop behavior.

Related notes:

- [preferred_detect_crop_phase1_manual_promotion_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/preferred_detect_crop_phase1_manual_promotion_design.md)
- [preferred_detect_crop_phase1_module_plan.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/preferred_detect_crop_phase1_module_plan.md)
- [preferred_detect_crop_runs_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/preferred_detect_crop_runs_design.md)
- [crimson_refined_detect_manual_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/crimson_refined_detect_manual_contract.md)
- [crimson_detect_bbox_read_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/crimson_detect_bbox_read_contract.md)

## Phase 1 Scope

Phase 1 should support:

- promotion of saved manual detections into preferred rows
- canonical full-image bbox storage
- explicit ROI/global mapping storage
- stable run-local preferred row identity
- explicit downstream artifact state on promoted rows

Phase 1 should not require:

- dense `(frame, entity)` arrays for every frame
- multi-entity assignment beyond a simple initial `entity_id`
- track-level review indexes
- immediate downstream artifact generation

## Stage Names

Phase 1 should introduce these new stage families:

- `preferred_detect_runs/<run>`
- `preferred_crop_runs/<run>`

Parent pointers:

- `preferred_detect_runs.attrs["latest"]`
- `preferred_crop_runs.attrs["latest"]`

Optional phase-1 convenience pointers:

- `preferred_detect_runs.attrs["latest_manual_promotion"]`
- `preferred_crop_runs.attrs["latest_manual_promotion"]`

## Canonical Rule

For preferred rows:

- full-image bbox geometry is canonical
- ROI placement is explicit
- ROI-local patch data is derived
- sparse refined/manual detect provenance remains preserved separately

## `preferred_detect_runs/<run>`

### Required row arrays

Row count for phase 1:

- `n_rows = number of promoted manual detections represented in this preferred run`

Required arrays:

- `preferred_row_ids` : `(n_rows,) int64`
- `frame_indices` : `(n_rows,) int32`
- `entity_ids` : `(n_rows,) int32`
- `bbox_img_xyxy` : `(n_rows, 4) float64`
- `bbox_norm_coords` : `(n_rows, 4) float64`
- `status_codes` : `(n_rows,) int8`
- `source_kind_codes` : `(n_rows,) int8`
- `source_sparse_row_index` : `(n_rows,) int32`
- `source_sparse_group_codes` : `(n_rows,) int8`
- `review_state_codes` : `(n_rows,) int8`
- `keypoints_state_codes` : `(n_rows,) int8`
- `subject_mask_state_codes` : `(n_rows,) int8`
- `eye_mask_state_codes` : `(n_rows,) int8`
- `swim_bladder_state_codes` : `(n_rows,) int8`

Recommended optional arrays:

- `source_detect_row_index` : `(n_rows,) int32`
- `confidence_scores` : `(n_rows,) float32`
- `class_ids` : `(n_rows,) int32`
- `reason_bytes` : `(n_rows, width) uint8`
- `reason` : `(n_rows,) string`
- `review_notes` : `(n_rows,) string`
- `created_utc` : `(n_rows,) string`
- `updated_utc` : `(n_rows,) string`

### Required attrs

- `source_detect_run`
- `source_refined_detect_run`
- `source_manual_group`
- `preferred_selection_policy = "manual_promotion_phase1"`
- `entity_assignment_policy`
- `coordinate_space = "full_image_xyxy"`
- `row_identity_policy`
- `summary_statistics`
- `status_code_map`
- `source_kind_code_map`
- `review_state_code_map`
- `artifact_state_code_map`
- provenance / environment metadata

Recommended phase-1 attr values:

- `entity_assignment_policy = "single_subject_default_entity0"` unless a more
  explicit arena-local assignment exists
- `row_identity_policy = "stable_run_local_preferred_row_id"`

### Required semantics

- `preferred_row_ids` must be stable within the run
- `bbox_img_xyxy` is the canonical edit/read geometry
- `bbox_norm_coords` is a derived normalized mirror of `bbox_img_xyxy`
- `source_sparse_group_codes` identifies whether the source row came from:
  - `manual`
  - `interpolated`
  - `filtered`
  - `raw`
- `source_sparse_row_index = -1` is invalid in phase 1 for promoted manual rows
- `entity_ids` may all be `0` in the initial single-subject implementation

## `preferred_crop_runs/<run>`

### Required row arrays

Row count:

- must match the promoted preferred-detect row count for the corresponding
  preferred run

Required arrays:

- `preferred_row_ids` : `(n_rows,) int64`
- `frame_indices` : `(n_rows,) int32`
- `entity_ids` : `(n_rows,) int32`
- `bbox_img_xyxy` : `(n_rows, 4) float64`
- `roi_offset_xy_full` : `(n_rows, 2) int32`
- `roi_size_wh` : `(n_rows, 2) int32`
- `status_codes` : `(n_rows,) int8`
- `source_preferred_detect_row_id` : `(n_rows,) int64`
- `materialized_crop_row_index` : `(n_rows,) int32`

Recommended optional arrays:

- `roi_offset_xy_ds` : `(n_rows, 2) int32`
- `roi_bbox_img_xyxy` : `(n_rows, 4) float64`
- `roi_transform_type_codes` : `(n_rows,) int8`
- `roi_transform_matrix` : `(n_rows, 3, 3) float64`

### Required attrs

- `source_preferred_detect_run`
- `crop_policy_name`
- `crop_policy_version` or `crop_policy_hash`
- `coordinate_space = "full_image_xyxy"`
- `roi_mapping_type = "translation_wh"`
- `status_code_map`
- `summary_statistics`
- provenance / environment metadata

### Required semantics

- `roi_offset_xy_full` and `roi_size_wh` must be sufficient to place the ROI in
  full-image space without inference
- `materialized_crop_row_index = -1` means there is no sparse/materialized crop
  row yet
- `source_preferred_detect_row_id` must match a row in the corresponding
  preferred detect run

## Status Vocabularies

### Preferred detect `status_codes`

Required phase-1 values:

- `0 = present`
- `1 = missing`
- `2 = pending`
- `3 = rejected`
- `4 = not_generated`

Phase-1 expectation for promoted manual rows:

- `present`

### `source_kind_codes`

Required phase-1 values:

- `0 = raw_detect`
- `1 = refined_detect`
- `2 = manual_promoted`

Phase-1 expectation for promoted manual rows:

- `manual_promoted`

### `review_state_codes`

Required phase-1 values:

- `0 = unknown`
- `1 = approved`
- `2 = pending`
- `3 = needs_review`
- `4 = rejected`

### Downstream artifact state codes

Applies to:

- `keypoints_state_codes`
- `subject_mask_state_codes`
- `eye_mask_state_codes`
- `swim_bladder_state_codes`

Required phase-1 values:

- `0 = not_generated`
- `1 = pending`
- `2 = missing`
- `3 = present`
- `4 = not_applicable`

Phase-1 default for newly promoted manual rows:

- `not_generated`

## Row Identity Rules

Phase-1 requirement:

- `preferred_row_id` must be stable for the same promoted row across updates to
  the same preferred run

Recommended matching key for preserving an existing row id:

- `source_refined_detect_run`
- `source_manual_group`
- `source_sparse_row_index`

If that source key is unchanged, reuse the prior `preferred_row_id`.

If the source key changes identity semantically, create a new preferred row id
or a new preferred run; do not silently recycle the old row id for a different
scene object.

## Coordinate Conventions

### Canonical detect geometry

`bbox_img_xyxy` should use:

- full-image pixel coordinates
- `[x0, y0, x1, y1]`
- float64 in phase 1

`bbox_norm_coords` should be:

- derived from the canonical image-space bbox
- stored as normalized `[cx, cy, w, h]`

### Canonical crop mapping

Phase-1 ROI mapping should assume translation-only placement:

- `roi_offset_xy_full = top-left ROI origin in full-image pixels`
- `roi_size_wh = width/height of ROI`

If the crop policy later needs a more complex transform, phase 1 should fail
closed rather than leave consumers guessing.

## Save / Promotion Flow

Phase-1 write flow should be:

1. Crimson writes the sparse manual subgroup under
   `refined_detect_runs/<latest>/<manual_group>` per the existing manual-write
   contract.
2. Palette resolves the active manual subgroup and source refined run.
3. Palette creates or updates `preferred_detect_runs/<run>`.
4. For each promoted manual detection row:
   - resolve full-image dimensions
   - compute canonical `bbox_img_xyxy`
   - compute derived `bbox_norm_coords`
   - assign or preserve `preferred_row_id`
   - assign `entity_id`
   - set `source_kind_codes = manual_promoted`
   - write downstream artifact states as `not_generated`
5. Palette creates or updates `preferred_crop_runs/<run>`.
6. For each promoted preferred row:
   - apply crop policy
   - write `roi_offset_xy_full`
   - write `roi_size_wh`
   - write `materialized_crop_row_index = -1` unless a sparse/materialized crop
     row is created immediately
7. Palette updates parent `latest` pointers.
8. Palette records provenance and summary statistics.

## Minimum Summary Statistics

Required run-level summary fields:

### On `preferred_detect_runs/<run>`

- `total_rows`
- `rows_manual_promoted`
- `rows_present`
- `rows_pending`
- `rows_rejected`
- `entity_count`
- `frame_count_covered`

### On `preferred_crop_runs/<run>`

- `total_rows`
- `rows_with_roi_mapping`
- `rows_with_materialized_crop`
- `rows_without_materialized_crop`
- `crop_policy_name`

## Consumer Expectations

Phase-1 consumers such as Crimson should be able to assume:

- preferred rows are the canonical saved-manual row model
- full-image bbox geometry is authoritative
- ROI placement is explicit
- downstream artifact absence is represented as state, not by structural
  omission alone

Consumers should not need to:

- infer ROI placement from bbox-only state
- treat saved manual rows as a different object class than other preferred rows
- fall back to sparse refined-detect rows for canonical scene geometry

## Validation Checklist

- [ ] `preferred_detect_runs/<run>` exists
- [ ] `preferred_crop_runs/<run>` exists
- [ ] row counts match across preferred detect and crop runs
- [ ] every preferred detect row has `preferred_row_id`
- [ ] every preferred crop row has `source_preferred_detect_row_id`
- [ ] every promoted manual row has canonical `bbox_img_xyxy`
- [ ] every promoted manual row has explicit `roi_offset_xy_full` and
      `roi_size_wh`
- [ ] every promoted manual row has downstream artifact states populated
- [ ] preferred parent `latest` pointers resolve to real runs
- [ ] sparse manual subgroup provenance remains intact

## Deferred To Later Phases

- dense `(frame, entity)` tensors
- non-manual preferred row coverage for all sparse detections
- track-aware review indexes
- frame-level review summaries
- automatic downstream artifact generation from promotion
- complex ROI transform contracts beyond translation + size
