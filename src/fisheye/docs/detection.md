# Detection Quality Labels Documentation

## Overview

The `detect_quality.py` module assigns quality labels to both frames and individual detections, providing complete data provenance for downstream processing steps like filtering and interpolation.

## Two Label Systems

### 1. Per-Frame Quality Flags (`quality_flags`)

**Array Properties:**
- **Length**: `total_frames` (matches video frame count)
- **Index space**: Frame indices (0 to total_frames-1)
- **Data type**: `int8`
- **Location**: `detect_runs/{run_name}/quality_reports/{quality_run}/quality_flags`

**Label Values:**

| Value | Meaning | Description |
|-------|---------|-------------|
| `-1` | No Detection | Frame has no detections (empty frame) |
| `0` | Clean | Frame has a clean, valid detection |
| `2` | Blip | Detection appears after a long gap (≥10 frames by default) |
| `3` | Jump | Detection position jumps too far from last valid position |
| `4` | Multi-Detection | Frame has multiple detections (when `max_fish > 1`) |

**Usage:**
```python
# Load quality flags
quality_flags = quality_group['quality_flags'][:]

# Find empty frames
empty_frames = np.where(quality_flags == -1)[0]

# Find clean frames with detections
clean_frames = np.where(quality_flags == 0)[0]

# Find all artifact frames
artifact_frames = np.where(quality_flags >= 2)[0]
```

---

### 2. Per-Detection Quality Labels (`detection_quality_labels`)

**Array Properties:**
- **Length**: `total_detections` (sum of all detections across all frames)
- **Index space**: Detection indices (matches `bbox_coords`, `scores`, etc.)
- **Data type**: `int8`
- **Location**: `detect_runs/{run_name}/quality_reports/{quality_run}/detection_quality_labels`

**Label Values:**

| Value | Meaning | Description |
|-------|---------|-------------|
| `0` | Clean | Valid, high-quality detection |
| `2` | Blip | Detection after a long gap |
| `3` | Jump | Detection with unrealistic position change |
| `4` | Multi-Detection | One of multiple detections in a frame |

**Note:** This array only contains entries for actual detections. Empty frames (with no detections) have no corresponding entries.

**Usage:**
```python
# Load detection quality labels
detection_quality_labels = quality_group['detection_quality_labels'][:]

# Load detection data
bbox_coords = detect_group['bbox_norm_coords'][:]
scores = detect_group['scores'][:]

# Filter for only clean detections
clean_mask = detection_quality_labels == 0
clean_bboxes = bbox_coords[clean_mask]
clean_scores = scores[clean_mask]

# Exclude jumps
no_jumps_mask = detection_quality_labels != 3
filtered_bboxes = bbox_coords[no_jumps_mask]
```

---

## Detection Parameters

Quality analysis parameters are saved with each quality report for full provenance:

```python
params = quality_group.attrs['artifact_detection_params']
jump_threshold = params['jump_threshold']  # Distance threshold in pixels
blip_gap_threshold = params['blip_gap_threshold']  # Frame gap threshold
```

**Default Values:**
- `jump_threshold`: 100.0 pixels
- `blip_gap_threshold`: 10 frames

---

## Quality Summary Statistics

Summary statistics are saved as attributes for quick reference:

```python
summary = quality_group.attrs['detection_quality_summary']

# Frame-level stats
total_frames = summary['total_frames']
empty_frames = summary['empty_frames']
clean_frames = summary['clean_frames']

# Detection-level stats
total_detections = summary['total_detections']
clean_detections = summary['clean_detections']
blip_detections = summary['blip_detections']
jump_detections = summary['jump_detections']
multi_detections = summary['multi_detections']
clean_percentage = summary['clean_percentage']
```

---

## Common Use Cases

### 1. Filter for Clean Detections Only

```python
# Get only clean detections
clean_mask = detection_quality_labels == 0
clean_bboxes = bbox_coords[clean_mask]
clean_scores = scores[clean_mask]

print(f"Kept {np.sum(clean_mask)}/{len(detection_quality_labels)} clean detections")
```

### 2. Find Gaps to Interpolate

```python
# Find gaps between clean detections
clean_detection_frames = np.where((quality_flags == 0) & (n_detections > 0))[0]

gaps = []
for i in range(len(clean_detection_frames) - 1):
    gap_size = clean_detection_frames[i+1] - clean_detection_frames[i] - 1
    if 0 < gap_size <= 20:  # Only gaps ≤20 frames
        gaps.append({
            'start': clean_detection_frames[i],
            'end': clean_detection_frames[i+1],
            'size': gap_size
        })
```

### 3. Remove Jump Artifacts

```python
# Create filtered dataset without jumps
no_jumps_mask = detection_quality_labels != 3
filtered_bboxes = bbox_coords[no_jumps_mask]
filtered_scores = scores[no_jumps_mask]

# Also need to update n_detections per frame
# This is more complex - see interpolation pipeline
```

### 4. Analyze Quality by Frame

```python
# Check quality of a specific frame
frame_idx = 100
frame_quality = quality_flags[frame_idx]

if frame_quality == -1:
    print(f"Frame {frame_idx}: No detection")
elif frame_quality == 0:
    print(f"Frame {frame_idx}: Clean detection")
elif frame_quality == 3:
    print(f"Frame {frame_idx}: Jump detected!")
```

---

## Integration with Pipeline

### Typical Processing Workflow:

1. **Run Detection**
   ```bash
   python -m fisheye.detect data.zarr
   ```

2. **Analyze Quality**
   ```bash
   python -m fisheye.refinement.detect_quality data.zarr --threshold 100
   ```

3. **Filter Clean Detections**
   ```python
   clean_mask = detection_quality_labels == 0
   clean_data = bbox_coords[clean_mask]
   ```

4. **Interpolate Gaps**
   ```python
   # Only interpolate between clean detections
   gaps = find_gaps_between_clean_detections(quality_flags, max_gap=20)
   interpolated_data = interpolate_gaps(clean_data, gaps)
   ```

---

## Visualization

The `detection_visualizer.py` tool displays quality labels with color coding:

- **Gray**: No detection
- **Green**: Clean detection
- **Orange**: Blip
- **Magenta**: Jump
- **Yellow**: Multi-detection

Navigate through artifacts using:
- `n` - Jump to next artifact
- `p` - Jump to previous artifact

---

## Notes

- Empty frames are marked as `-1` in `quality_flags` but have no entry in `detection_quality_labels`
- For single-fish tracking (`max_fish=1`), flag `4` (multi-detection) will never appear
- Quality labels are preserved through the pipeline for full traceability
- All parameters used for quality assessment are saved for reproducibility