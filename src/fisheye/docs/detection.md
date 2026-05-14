# Detection Quality Labels Documentation

## Overview

`fisheye.refinement.detect_quality` analyzes a raw detect run and writes
artifact labels under `detect_runs/<run>/quality_reports/<quality_run>`.

Those labels are consumed by `fisheye.refinement.refine_detect` to filter raw
detections before writing the sparse curated refined surface at
`refined_detect_runs/<run>/instances`.

Interpolation is no longer part of the normal sparse-first detect workflow.
Legacy interpolation remains a compatibility concern for older archives, not
the primary outcome of `detect_quality`.

Naming split:

- `detect_quality` reports describe raw detect artifacts.
- Refined detect approval/review metadata lives on refined runs and in the
  registry gating surfaces, not in `quality_reports`.
- Dish-mask spatial gating is also a refined-run policy. It does not mutate
  `detect_quality` labels or raw model predictions; refinement preserves those
  raw candidates in `source_detections` and marks outside-dish rows with
  reason `outside_dish_mask`.

## Two Label Systems

### 1. Per-Frame Quality Flags (`quality_flags`)

Array properties:

- Length: `total_frames` (matches video frame count)
- Index space: frame indices (`0..total_frames-1`)
- Data type: `int8`
- Location: `detect_runs/<run>/quality_reports/<quality_run>/quality_flags`

Label values:

| Value | Meaning | Description |
| --- | --- | --- |
| `-1` | No detection | Frame has no detections |
| `0` | Clean | Frame has a clean, valid detection |
| `2` | Blip | Detection appears after a long gap |
| `3` | Jump | Detection position jumps too far from the last valid position |
| `4` | Multi-detection | Frame has multiple detections |

Example:

```python
quality_flags = quality_group["quality_flags"][:]

empty_frames = np.where(quality_flags == -1)[0]
clean_frames = np.where(quality_flags == 0)[0]
artifact_frames = np.where(quality_flags >= 2)[0]
```

### 2. Per-Detection Quality Labels (`detection_quality_labels`)

Array properties:

- Length: `total_detections`
- Index space: detection indices aligned with `bbox_norm_coords`, `scores`,
  `frame_indices`, and `class_ids`
- Data type: `int8`
- Location: `detect_runs/<run>/quality_reports/<quality_run>/detection_quality_labels`

Label values:

| Value | Meaning | Description |
| --- | --- | --- |
| `0` | Clean | Valid detection |
| `2` | Blip | Detection after a long gap |
| `3` | Jump | Detection with unrealistic position change |
| `4` | Multi-detection | Detection from a multi-detection frame |

There is no per-detection `-1`; empty frames have no detection rows.

Example:

```python
detection_quality_labels = quality_group["detection_quality_labels"][:]

bbox_coords = detect_group["bbox_norm_coords"][:]
scores = detect_group["scores"][:]

clean_mask = detection_quality_labels == 0
clean_bboxes = bbox_coords[clean_mask]
clean_scores = scores[clean_mask]
```

## Detection Parameters

Quality analysis parameters are saved for provenance:

```python
params = quality_group.attrs["artifact_detection_params"]
jump_threshold = params["jump_threshold"]
blip_gap_threshold = params["blip_gap_threshold"]
```

Default values:

- `jump_threshold`: `100.0` pixels
- `blip_gap_threshold`: `10` frames

## Quality Summary Statistics

Quick summary stats are saved in:

```python
summary = quality_group.attrs["detection_quality_summary"]
```

Common fields:

- `total_frames`
- `empty_frames`
- `clean_frames`
- `total_detections`
- `clean_detections`
- `blip_detections`
- `jump_detections`
- `multi_detections`
- `clean_percentage`

## Common Use Cases

### 1. Filter for Clean Detections Only

```python
clean_mask = detection_quality_labels == 0
clean_bboxes = bbox_coords[clean_mask]
clean_scores = scores[clean_mask]
```

### 2. Inspect Artifact Frames

```python
artifact_frames = np.where(quality_flags >= 2)[0]
print(f"Artifact frames: {artifact_frames[:20]}")
```

### 3. Remove Jump Artifacts

```python
no_jumps_mask = detection_quality_labels != 3
filtered_bboxes = bbox_coords[no_jumps_mask]
filtered_scores = scores[no_jumps_mask]
```

### 4. Compare Clean vs Raw Counts

```python
summary = quality_group.attrs["detection_quality_summary"]
print(f"raw detections:   {summary['total_detections']}")
print(f"clean detections: {summary['clean_detections']}")
print(f"jump detections:  {summary['jump_detections']}")
print(f"blip detections:  {summary['blip_detections']}")
```

## Integration with Pipeline

Typical workflow:

1. Run detection.

   ```bash
   scripts/py -m fisheye.detection.detect_traditional /path/to/zarr
   ```

2. Run detect quality.

   ```bash
   scripts/py -m fisheye.refinement.detect_quality /path/to/zarr
   ```

   This writes `quality_flags` and `detection_quality_labels` for the selected
   raw detect run.

3. Build the sparse curated refined surface.

   ```bash
   scripts/py -m fisheye.refinement.refine_detect /path/to/zarr
   ```

   `refine_detect` consumes the quality labels and writes
   `refined_detect_runs/<run>/instances`. Interpolation is disabled in the
   current default workflow.

4. Review and approve the curated refined surface when needed.

   ```bash
   scripts/py -m fisheye.tune.detect_review /path/to/zarr
   scripts/py -m fisheye.utils.accept_detect_review /path/to/zarr --state approved --intended-use training --reviewer <name>
   ```

## Visualization

`detection_visualizer.py` displays quality labels with color coding:

- Gray: no detection
- Green: clean
- Orange: blip
- Magenta: jump
- Yellow: multi-detection

## Notes

- Empty frames are marked as `-1` in `quality_flags` but have no entry in
  `detection_quality_labels`.
- For single-fish tracking (`max_fish=1`), `4` (multi-detection) should be
  uncommon or absent.
- Quality labels are preserved through refine for traceability.
- `detect_quality` is a raw detect artifact-labeling stage, not the refined
  detect approval stage.
