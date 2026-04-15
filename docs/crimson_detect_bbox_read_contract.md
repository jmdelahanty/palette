# Crimson Detect BBox Read Contract
<!-- contract-meta
version: 4
status: active
last_verified: 2026-04-15
-->

Purpose: define the read contract Crimson should use to load detect bounding
boxes from Palette Zarr archives.

## Primary Read Surface

Current primary source:

- `refined_detect_runs/<run_name>/instances` when present

`source_detections/` is a candidate-audit surface and is not the primary bbox
render source.

Fallbacks:

1. legacy sparse refined groups:
   - `refined_detect_runs/<run_name>/<legacy_group>`
   - `refined_detect_runs/<run_name>/interpolated`
   - `refined_detect_runs/<run_name>/filtered`
2. raw detect:
   - `detect_runs/<run_name>`

Current archives should not be expected to contain separate preferred detect or
crop arrays.
Current Crimson readers should treat the subgroup-era fallbacks as
compatibility-only and prefer `instances/` whenever it exists.

## Run Selection Rules

Default refined selection:

1. use `refined_detect_runs.attrs["latest"]`
2. if that run has `instances/`, read `instances/`
3. otherwise fall back to legacy sparse resolution:
   - `manual_review_latest` (legacy pointer)
   - `manual`
   - `interpolated`
   - `filtered`
   - raw detect

## Required Arrays

### Curated sparse `instances/`

- `frame_indices`
- `bbox_img_xyxy`
- `bbox_norm_coords`
- `source_kind_codes`

Common optional arrays:

- `refined_row_ids`
- `confidence_scores`
- `class_ids`
- `source_detect_row_index`
- `reason_bytes`
- `reason`

Reader rule:

- Rows in `instances/` are already the curated accepted detections. Render only
  finite bbox geometry.

### Legacy sparse refined groups

These are compatibility-only for historical archives.

- `frame_indices`
- `bbox_norm_coords`

Optional:

- `detection_source`
- `reason_bytes`
- `reason`
- `scores`
- `class_ids`

### Raw detect

- `frame_indices`
- `bbox_norm_coords`

## Semantics

`source_kind_codes` is the machine-readable provenance state on current refined
surfaces:

- `none`
- `raw_detect`
- `interpolated`
- `manual`

`reason` is explanatory only. Crimson should display it, but should not depend
on parsing it to determine whether a row is present.

For current sparse refined runs, prefer `source_kind_codes` over any string
label when deciding whether a row is operator-corrected or inherited from raw
detect.

For `source_detections/`, `decision_codes` is the machine-readable raw
candidate disposition. It is useful for audit and UI summaries, but it should
not be treated as the bbox render surface.

For legacy sparse refined groups:

- if `reason_bytes` exists, decode and use it
- else if `reason` exists, use it
- else if `detection_source` exists, derive:
  - `0 -> clean`
  - `1 -> interpolated`
- else default to `clean`

## Coordinate Rules

Canonical refined detect geometry:

- `bbox_img_xyxy` is authoritative
- `bbox_norm_coords` is a normalized mirror

Legacy sparse/raw geometry:

- `bbox_norm_coords` is `[cx, cy, w, h]`
- normalize against the detector input frame dimensions

## Metadata Hints

Helpful attrs when present:

- `source_detect_run`
- `detect_review_status`
- `summary_statistics`
- `curated_row_storage`
- `refined_storage_semantics`
- `status_code_map`
- `source_kind_code_map`

`detect_review_status["resolved_group"]` should normally be `refined` for current
runs with `instances/`.

## Failure Modes

- Missing `refined_detect_runs` and `detect_runs`: return a structured
  "no detections available" status.
- Empty arrays: valid, render no boxes.
- Missing optional arrays: still valid.

## Related Documents

- `docs/refined_detect_collapse_v2.md`
- `docs/detection_refinement_workflow.md`
- `src/fisheye/docs/zarr_structure.md`
- `docs/crimson_refined_detect_manual_contract.md`
