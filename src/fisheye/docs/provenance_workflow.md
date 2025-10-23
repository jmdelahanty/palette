# Provenance-Safe Refinement Workflow

This note captures the sequencing we used to align refined detections, crops, keypoints, and ID assignment so downstream analysis (movement, metrics, plots) stays consistent.

## Stage Overview

```
detect_runs → refined_detect_runs → crop_runs → keypoints_runs → refined_keypoints_runs → id_assignment_runs → movement_runs
```

Relevant provenance attributes:

| Stage | Primary Array | Key Attrs |
| --- | --- | --- |
| `detect_runs/<run>` | `bbox_norm_coords` | `detect_timestamp_utc`, `total_detections` |
| `refined_detect_runs/<run>/interpolated` | `bbox_norm_coords`, `detection_source` | `source_detect_run`, `interpolated_roi_path` |
| `crop_runs/<run>` | `roi_images` | `detection_source_path`, `refined_roi_path`, `refined_roi_count` |
| `keypoints_runs/<run>` | `heading`, `frame_indices` | `source_crop_run` |
| `refined_keypoints_runs/<run>` | `heading`, `detection_success` | `source_keypoints_run`, `source_crop_run` |
| `id_assignment_runs/<run>` | `detection_ids` | `source_detect_run`, `source_refined_run` |

## Regenerating Interpolated Crops

When refined detections introduce interpolated boxes, we need matching ROI imagery.

1. **Regenerate crops** (GPU decode will be used when available):
   ```bash
   python -m fisheye.refinement.regenerate_interpolated_crops \
     <archive>.zarr \
     --video-path /path/to/video.mp4 \
     --source-crop-run <existing_crop_run> \
     --overwrite
   ```
   * `--source-crop-run` identifies which crop run to augment. If omitted, the script now attempts to infer the crop run based on `detection_source_path`.

2. Inspect linkage:
   ```bash
   python -m fisheye.diagnostics.check_refined_roi_links <archive>.zarr
   ```
   Confirms `interpolated_roi_path`, counts, decoder, and crop metadata stay in sync.

## Re-running Dependent Stages

After new ROIs exist, rerun stages that depend on them:

1. YOLO keypoints:
   ```bash
   python -m fisheye.detection.detect_keypoints_yolo <archive>.zarr --model <weights>.pt
   python -m fisheye.refinement.refine_keypoints <archive>.zarr
   ```
   The keypoint script now auto-applies refined ROI overrides when available.

2. ID assignment (if IDs align with refined detections):
   ```bash
   python -m fisheye.tracking.assign_ids <archive>.zarr
   ```

3. Movement analysis / downstream pipelines can now run without length mismatches.

## Provenance Validation

Two diagnostics help verify alignment:

* `python -m fisheye.diagnostics.check_provenance_consistency <archive>.zarr`
  * Compares row counts across detect, refined detect, crop, keypoint, and ID runs.
* `python -m fisheye.diagnostics.check_refined_roi_links <archive>.zarr`
  * Audits refined detection runs and ensures regenerated ROIs are discoverable from the crop side.

Expected healthy output shows consistent row counts (e.g., refined detections = crop ROIs = keypoint headings = ID rows) and no reported issues.

## Summary Checklist

1. `regenerate_interpolated_crops` with correct crop run.
2. Rerun YOLO keypoints and keypoint refinement.
3. Rerun ID assignment (and analysis).
4. Confirm with diagnostics.

Following these steps keeps provenance clean and prevents downstream stage failures when interpolated detections are added. 
