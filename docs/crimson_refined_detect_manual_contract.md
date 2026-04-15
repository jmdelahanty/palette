# Crimson Refined-Detect Manual Write Contract
<!-- contract-meta
version: 2
status: historical
last_verified: 2026-04-15
stage_arrays_spec: REFINED_DETECT_SPEC
-->

Purpose: preserve the older subgroup-era manual detect write contract that
predated Palette's current sparse-first refined-detect layout.

Current status note:
- This document describes the older subgroup-era manual detect write contract.
- Current Palette refined detect runs are sparse-first, with canonical curated
  data in `refined_detect_runs/<run>/instances` and candidate audit rows in
  `refined_detect_runs/<run>/source_detections`.
- `detect_review_status.resolved_group` should normally be `refined` for
  current runs.
- Do not use this note as the contract for new sparse refined-detect writer
  work. Use it only as historical background for legacy archives until a
  replacement sparse write contract is published.
- The `manual_group` terminology below is historical and refers to the legacy
  sparse subgroup naming pattern, not the current canonical refined surface.

Date anchored: 2026-04-11.

String/encoding policy references:
- `src/fisheye/docs/zarr_structure.md` (authoritative schema + encoding conventions)
- `docs/zarr_string_encoding_todo.md` (standardization rollout/status)

## Scope

This historical note is for the older behavior that updated:
- `refined_detect_runs/<latest>/<manual_group>`
- review pointers/status on the refined run

It is intentionally aligned with the legacy subgroup-era behavior in
`fisheye.tune.detect_review`.

For acceptance/status policy after inspection, see:
- `docs/crimson_detect_review_acceptance_contract.md`

## Historical Minimum Required Writes

Given `run_name = refined_detect_runs.attrs["latest"]` and `manual_group` (default `"manual"`):

1. Create or overwrite:
- `refined_detect_runs/<run_name>/<manual_group>`

2. Required arrays in manual subgroup:
- `frame_indices` (`int32`, shape `(n_detections,)`)
- `bbox_norm_coords` (`float64`, shape `(n_detections, 4)`, normalized `[cx, cy, w, h]`)
- `scores` (`float32`, shape `(n_detections,)`)
- `class_ids` (`int32`, shape `(n_detections,)`)
- `frame_counts` (`int32`, shape `(n_frames,)`)
- `n_detections` (`int32`, shape `(n_frames,)`)  # alias of `frame_counts`
- `frame_mapping` (`int32`, shape `(n_detections,)`)  # alias of `frame_indices`

3. Optional arrays (recommended when available):
- `detection_source` (`int8`, shape `(n_detections,)`)
- `retune_id` (`int32`, shape `(n_detections,)`, `-1` for non-retuned)
- `reason_bytes` (`uint8`, shape `(n_detections, width)`, null-terminated UTF-8; preferred cross-tool encoding)
- `reason` (UTF-8 variable-length string, shape `(n_detections,)`)

`reason` write guidance:
- If `detection_source` is written, write `reason` as well.
- Recommended mapping:
  - `detection_source == 0` -> `reason = "clean"` (or `"manual"` for explicit manual rows)
  - `detection_source == 1` -> `reason = "interpolated"`
- Custom labels are allowed (for example `retune`), but lengths must still match
  `n_detections` and labels must be UTF-8 strings.

`reason_bytes` guidance:
- Write `reason_bytes` for Crimson/TensorStore compatibility, even if `reason` is also written.
- Keep row alignment identical to `frame_indices`.
- Encode each row as UTF-8 bytes with `0x00` terminator and `0x00` padding.
- Keep fallback metadata aligned with Palette readers:
  - `reason_encoding="utf8-null-terminated"`
  - `reason_bytes_width=<int>`
  - `reason_bytes_null_terminated=true`
  - `reason_fallback_order=["reason_bytes","reason","detection_source"]`

4. Manual subgroup attrs:
- `storage_layout = "columnar"`
- `column_fields` (list matching present arrays)
- `field_names` (same as `column_fields`)
- `detection_source_type = "manual"` (or `"retune"` for retune outputs)
- `detection_source_path = "refined_detect_runs/<run_name>/<manual_group>"`
- `source_refined_run = <run_name>`
- `source_variant = "interpolated"` or `"filtered"` (the base used)
- `manual_review_timestamp` (ISO UTC)

5. Historical refined run pointer update:
- `refined_detect_runs/<run_name>.attrs["manual_review_latest"] = <manual_group>`

6. Review status update (strongly recommended so historical crop/registry
   compatibility resolves correctly):
- `refined_detect_runs/<run_name>.attrs["detect_review_status"] = { ... }`
- `refined_detect_runs.attrs["detect_review_status_latest"] = <run_name>`

## `detect_review_status` Payload

Recommended payload fields:
- `state` (e.g., `approved`, `needs_review`)
- `method` (`manual`, `retune`, `algorithmic`, etc.)
- `intended_use` (`training`, `analysis`, etc.)
- `timestamp` (ISO UTC)
- `resolved_group`
  - current runs should normally use `"refined"`
  - legacy subgroup-era writes may still record a sparse subgroup such as
    `"manual"`
- `preference_chain` (default: `["refined", "manual", "interpolated", "filtered", "raw"]`)
- optional: `reviewer`, `notes`

## Behavioral Expectations in Palette

1. Resolution preference:
- Palette prefers `refined_detect_runs/<run>/instances` for current runs.
- Legacy sparse fallback is `manual -> interpolated -> filtered -> raw`.

2. Crop stage:
- `crop --source-type auto` resolves to the canonical refined surface when available.
- When the canonical sparse refined surface is missing, crop falls back through the
  review-guided sparse detect chain.

3. Registry extraction:
- Registry maintenance reads review status and resolved group from refined attrs.
- Missing/inconsistent fields reduce traceability and can mislabel review state.

## Write Safety Rules

1. Never mutate `detect_runs/<run>` in-place.
2. For this historical subgroup-era flow, keep manual edits in the refined
   manual subgroup only.
3. If overwriting an existing manual subgroup, rewrite full subgroup atomically.
4. Keep array lengths consistent:
- `len(frame_indices) == len(bbox_norm_coords) == len(scores) == len(class_ids)`
- `len(frame_counts) == n_frames`
- if present, `reason_bytes.shape[0] == n_detections`
- if `reason` exists: `len(reason) == n_detections`

## Validation Queries / Checks

Run after Crimson writes:

1. Manual subgroup exists and is selected:
- `manual_review_latest` points to an existing subgroup.

2. Review payload exists:
- for this historical subgroup-era flow,
  `detect_review_status.resolved_group` equals the selected manual group
- current sparse-first runs should normally resolve to `refined`

3. Shape consistency:
- all detection-level arrays share `n_detections`.
- `sum(frame_counts) == n_detections`.
- if present, `reason_bytes.shape[0] == n_detections`.
- if present, `reason` and `detection_source` lengths match `n_detections`.

4. Resolution check:
- Palette helper resolution returns manual group as active source.

## Implementation Note For Crimson Agent

When recovering historical subgroup-era behavior, mirror Palette's semantics from:
- `src/fisheye/tune/detect_review.py`
- `src/fisheye/shared/refined_detect_review.py`

Do not invent alternate field names for status/pointers; Palette tooling already
consumes the names listed above.
