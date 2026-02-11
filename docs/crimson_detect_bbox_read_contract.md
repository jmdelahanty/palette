# Crimson Detect BBox Read Contract

Purpose: define the read-only contract Crimson should use to load detection
bounding boxes from Palette Zarr archives.

Date anchored: 2026-02-11.

## Scope

- Read detection boxes for overlay/inspection.
- Support both raw detect runs and refined/manual detect sources.
- Do not define write behavior (manual write contract is separate).

## Primary Source Paths

1. Raw detect source:
- `detect_runs/<run_name>`

2. Refined detect source (optional mode):
- `refined_detect_runs/<run_name>/<group_name>`
- where `<group_name>` is usually one of:
  - `manual` (or `manual_review_latest` target)
  - `interpolated`
  - `filtered`

## Quick Reference

| Source | Can show boxes | Can show clean/interpolated labels | Can show missing/artifact frame status |
|---|---|---|---|
| `detect_runs/<run>` | Yes | No (raw detect does not carry interpolation provenance) | Yes, via `quality_reports/<run>/quality_flags` when present |
| `refined_detect_runs/<run>/<group>` | Yes | Yes, from `reason_bytes` (`reason`/`detection_source` fallback) | Not directly; use detect-quality frame labels from raw detect run |
| `detect_runs/<run>/quality_reports/<quality_run>` | No (labels only) | Detection-level labels available via `detection_quality_labels` | Yes, frame-level via `quality_flags` (`-1/0/2/3/4`) |

## Run Selection Rules

### Raw detect default

1. Use `detect_runs.attrs["latest"]` when present.
2. If missing, choose newest run name lexicographically.
3. If `detect_runs` missing, report "no detect runs" (do not crash).

### Refined detect preferred source

If refined source mode is requested, use this order:
1. `manual_review_latest` subgroup when it exists
2. `manual` subgroup when it exists
3. `interpolated`
4. `filtered`
5. fallback to raw `detect_runs/<source_detect_run>`

This matches Palette runtime resolution in:
- `src/fisheye/shared/refined_detect_review.py`

## Required Arrays (for any selected detect group)

- `frame_indices` (`int32`, shape `(N,)`)
- `bbox_norm_coords` (`float64` in current runtime, shape `(N, 4)`)

`bbox_norm_coords` ordering is:
- `[cx, cy, w, h]`
- normalized to detector input frame dimensions

## Optional Arrays

- `scores` (`float32`, shape `(N,)`)
- `class_ids` (`int32`, shape `(N,)`)
- `detection_source` (`int8`, refined groups only: `0=real`, `1=interpolated`)
- `reason_bytes` (`uint8`, shape `(N, width)`, null-terminated UTF-8 strings; preferred for C++ TensorStore compatibility)
- `reason` (`string`, refined groups; detect semantics commonly `clean` / `interpolated`, manual/retune may use other labels)
- `frame_counts` (`int32`, shape `(num_frames,)`)
- `n_detections` (`int32`, alias of `frame_counts`)

## Reason Array Contract (Refined Detect)

When reading `refined_detect_runs/<run>/<group>`, Crimson should treat reason
labels as the human-readable source tag for each detection row.

Minimum expected semantics:
- `clean`: detection comes from observed/kept detection rows
- `interpolated`: detection was added by interpolation

Possible additional values in manual/retune flows:
- `manual`
- `retune`
- other operator-defined labels

Reader behavior:
1. If `reason_bytes` exists, decode null-terminated UTF-8 rows and use for UI/status labeling.
2. If `reason_bytes` is missing but `reason` exists, use `reason`.
3. If neither reason array exists but `detection_source` exists, derive:
   - `0 -> clean`
   - `1 -> interpolated`
4. If neither exists, default label to `clean` and warn once.

Do not hard-fail on unknown reason strings; display them as-is.

## Missing / Artifact Reason Codes

Crimson can optionally read detect-quality labels to explain missing detections
or likely artifacts during overlay inspection.

Primary source:
- `detect_runs/<run_name>/quality_reports/<quality_run>/quality_flags`

Recommended `quality_run` selection:
1. `quality_reports.attrs["latest"]` when present
2. otherwise newest run name lexicographically

`quality_flags` semantics:
- `-1`: no detection on this frame (empty frame)
- `0`: clean frame
- `2`: blip
- `3`: jump
- `4`: multi-detection

Important:
- `quality_flags` is frame-level (`shape=(total_frames,)`).
- Missing detections are represented by `quality_flags == -1` and/or by no
  detection row in `frame_indices` for that frame.
- `detection_quality_labels` (when present) is detection-level and has no
  entries for empty frames.

This is complementary to refined `detection_source`:
- `detection_source` answers "real vs interpolated" for a detection row.
- `quality_flags` answers frame-level quality / missing status.

Compared to keypoints:
- keypoints commonly use per-ROI `reason` string tags on refined runs.
- detect currently uses numeric quality labels (`quality_flags`,
  `detection_quality_labels`) plus optional `reason` strings only in refined
  manual/retune groups.

## Type Tolerance

Crimson should accept:
- `bbox_norm_coords` as `float32` or `float64`
- `frame_indices` as any integer dtype coercible to `int64`

Do not hard-fail on dtype width alone; coerce at load.

## Coordinate Conversion

Given frame width `W` and height `H`:

- `x1 = (cx - w/2) * W`
- `y1 = (cy - h/2) * H`
- `x2 = (cx + w/2) * W`
- `y2 = (cy + h/2) * H`

Recommended safety:
- clip normalized values to `[0, 1]` before conversion
- then clip pixel bounds to `[0, W-1]` / `[0, H-1]`

## Consistency Checks

For selected group:
- `len(frame_indices) == len(bbox_norm_coords)`
- if `scores` present: `len(scores) == len(frame_indices)`
- if `class_ids` present: `len(class_ids) == len(frame_indices)`
- if `frame_counts` present: `sum(frame_counts) == len(frame_indices)` (advisory; allow mismatch with warning)

## Metadata Hints (Optional)

Helpful attrs to read when present:
- run attrs: `detection_method`, `source_detect_run`, `detect_review_status`
- parent attrs: `latest`, `detect_review_status_latest`
- root attrs: `source_video_path`, `source_video`, `inference_width`, `inference_height`
- quality attrs: `quality_reports.attrs["latest"]`,
  `artifact_detection_params`, `detection_quality_summary`
- analysis stimulus-run attrs (analysis archives only):
  - `analysis/stimulus_runs/<latest>.attrs["source_h5"]`
  - `analysis/stimulus_runs/<latest>.attrs["source_stimulus_video_path"]` (optional; rendered stimulus video path when available)

Training archive note:
- `source_stimulus_video_path` is not expected on training Zarrs.
- Absence on training archives should not be treated as an error.

## Expected Failure Modes

- Missing `detect_runs`: return a structured "no detections available" status.
- Empty arrays (`N=0`): valid, render no boxes.
- Missing optional arrays (`scores`, `class_ids`): still valid.

## Related Documents

- `docs/crimson_palette_zarr_alignment_todo.md`
- `docs/crimson_refined_detect_manual_contract.md`
- `src/fisheye/docs/detection.md`
- `src/fisheye/docs/detection_structure.md`
- `src/fisheye/docs/zarr_structure.md`
