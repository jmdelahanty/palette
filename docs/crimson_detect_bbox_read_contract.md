# Crimson Detect BBox Read Contract
<!-- contract-meta
version: 5
status: active
last_verified: 2026-05-16
-->

Purpose: define the read contract Crimson should use to load detect bounding
boxes from Palette Zarr archives.

Scope note for clipped recordings:

- This contract applies to traditional top-level analysis Zarrs and
  materialized training Zarrs.
- For clipped analysis archives, this contract defines the bbox leaf-group
  read rules after a resolver has selected a concrete clip-local refined run,
  such as
  `clips/<clip_id>/cameras/<camera_serial>/refined_detect_runs/<run>/instances`.
- It does not by itself define how to select a finalized collection, map parent
  frames to clip-local frames, or switch videos at clip boundaries.
  Crimson needs a finalized collection resolver for that.
  See `docs/clipped_recording_consumer_mapping_contract.md`.

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

## Zarr Chunking Rules

The logical bbox interface is `[N, 4]` for both traditional and clipped
archives. Crimson should read Zarr arrays by logical indices and must not assume
that a row is contiguous inside one physical chunk.

Palette now writes refined-detect bbox arrays with preferred chunk shape:

```text
(row_chunk, 4)
```

for:

- `instances/bbox_img_xyxy`
- `instances/bbox_norm_coords`
- `source_detections/bbox_img_xyxy`
- `source_detections/bbox_norm_coords`

Older archives may still have auto-chosen chunks such as `(26664, 2)`, which
split the fixed-width bbox columns. That is valid Zarr storage but can expose
reader bugs. Use `fisheye.utils.validate_refined_detect_run` to warn on this
layout and `fisheye.utils.rechunk_refined_detect_bbox_arrays` to opt-in repair
existing refined-detect runs.

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
- `docs/clipped_recording_consumer_mapping_contract.md`
