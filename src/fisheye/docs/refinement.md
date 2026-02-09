Refined Runs Implementation Specification
Overview
The refined_detect_runs/ group stores filtered and interpolated detection data, providing clean datasets for downstream analysis while maintaining full traceability to the original detections.

### Status semantics for refine-detect

Refined runs always include `filtered/` and `interpolated/` groups to keep the
schema stable for downstream tools. When refinement is a no-op, we rely on
metadata + status labels rather than changing the on-disk layout:

- **passthrough**: Training/sample imports where refinement is intentionally
  disabled (`refine_mode=passthrough`); refined data mirrors the source.
- **unchanged**: Standard refinement ran but removed 0 detections and added 0
  interpolations; refined data is identical to the source.
- **filtered/interpolated**: Refinement actually changed the data.

The status reporter (`check_recording_steps.py`) uses these labels so “interpolated”
does not imply synthetic data when refinement was a no-op.

### Manual detection review

If you need to correct missing detections by hand, use:

```bash
python -m fisheye.tune.detect_review path/to/session.zarr --variant interpolated
```

This writes a new group under `refined_detect_runs/<run>/manual` and records
`manual_review_latest` on the refined run. The crop stage automatically
prefers the manual subgroup when present (unless you explicitly request
`--crop-source filtered`).

When you approve a refined run in the review UI, it records
`detect_review_status` on the refined run (and updates the parent-level
`detect_review_status_latest`). Crop runs created with
`--crop-source preferred` or `auto` consult this status to choose the
resolved group, and store both a snapshot (`detect_review_status`) and a
reference (`detect_review_status_ref`) on the crop run for provenance.

Retunes are incremental when a manual subgroup already exists: the retune UI
uses that subgroup as its base and records `retune_base_group` plus
`retune_params` for auditability. The retuned subgroup also stores per-detection
`retune_id` and `reason` labels, along with `detection_source_type` and
`detection_source_path` metadata.

Preferred crop resolution uses the refinement review status plus a policy
chain (`manual → interpolated → filtered → raw`). The policy label is recorded
in crop runs as `detection_preferred_policy`.

For sampled training imports, coverage percentages are computed against the
sampled frame universe and stored on the refined run as
`coverage_frames_total`, `coverage_frame_source`, and (when applicable)
`coverage_frames_full`.

### Verifying that crops reference refined detections

After creating a refined detect run, confirm that subsequent crop runs pulled from the interpolated coordinates with:

```bash
python -m fisheye.diagnostics.check_crop_sources path/to/session.zarr
# limit to a specific run
python -m fisheye.diagnostics.check_crop_sources path/to/session.zarr --crop-run crop_2025-10-25_19-25-05
```

The diagnostic reports the recorded `detection_source_path`, whether that path exists (and matches the latest `refined_detect_runs/<latest>/interpolated` group), whether the copied `frame_indices` match the source detections, and if the `detection_source` array is present (with a real vs. interpolated breakdown). This makes it easy to catch situations where the crop stage accidentally pointed back at the original detect run.

Before launching a new crop run you can also preview what the stage would consume via:

```bash
python -m fisheye.diagnostics.crop_dry_run path/to/session.zarr --config configs/fisheye/default.yaml
# override the source the same way as the pipeline CLI
python -m fisheye.diagnostics.crop_dry_run path/to/session.zarr --crop-source interpolated
```

`crop_dry_run` resolves the detection source using the same precedence rules as the pipeline (CLI overrides → config values), then prints total detections, frame counts, coverage, and whether interpolated metadata would be available.

## Eye Mask Refinement (new)

Eye segmentation runs can now be post-processed with `python -m fisheye.refinement.refine_eye_masks`. The tool:

- Reads an existing `eye_masks_runs/<source>` entry plus its paired keypoint run
- New eye-mask runs default to the latest refined keypoints when present and record `source_keypoint_group` for provenance
- Reassigns pixels to anatomical left/right using keypoint geometry (with heading-based fallback)
- Emits a new `refined_eye_masks_runs/<run_name>` containing the same arrays as the traditional segmenter (`masks_roi`, ellipse metrics, contour tables, etc.)
- Records provenance in the run attributes (`method="refine_eye_masks"`, `source_*_run`, `source_eye_masks_method`, `eye_labels`, summary stats)

This keeps the original masks untouched for provenance while providing a refined alternative that matches keypoint labels.

By default, traditional segmentations use a fast path that preserves the source masks and ellipse fits and skips smoothing/component enforcement. Use `--force-refine-traditional` to opt into the full refinement pass.

After refinement, use `python -m fisheye.tune.eye_mask_review --retune/--manual` to correct failures and `--audit` to refresh the postprocess summary stored in `summary_statistics`.

> ℹ️ **Optional datasets.**  
> - YOLO-based segmentation writes an additional `mask_probs_roi` dataset with float16 probabilities.  
> - Soft-ellipse moments from YOLO runs are stored in `ellipse_params_soft` (float32) when available; check the run attribute `ellipse_soft_available`.  
> Downstream consumers should treat these datasets as optional and guard on their presence (e.g., `if "mask_probs_roi" in run_group`).

Directory Structure
/refined_detect_runs/
  @latest = "refined_detect_2025-10-03_21-00-00"
  
  /refined_detect_2025-10-03_21-00-00/
    # Root-level metadata
    @source_detect_run = "detect_2025-10-03_20-28-11"
    @source_quality_run = "detect_quality_2025-10-03_20-30-45"
    @refinement_timestamp = "2025-10-03T21:00:00Z"
    @operations = ["filter", "interpolate"]
    @parameters = {
        "max_gap": 20,
        "interpolation_method": "linear",
        "filters_applied": ["remove_jumps"],
        "parameter_source": "config",
        "refine_mode": "standard",
        "sampled_import": false,
        "sampled_import_meta": {}
    }
    
    /filtered/
      bbox_norm_coords     # (N_filtered, 4) float64
      scores              # (N_filtered,) float32
      n_detections        # (total_frames,) int32
      frame_mapping       # (N_filtered,) int32
      reason              # (N_filtered,) utf8 (currently "clean")
      
      @total_detections
      @dropped_detections
      @drop_reasons = {"jumps": 2, "blips": 0}
      
    /interpolated/
      bbox_norm_coords     # (N_interpolated, 4) float64
      scores              # (N_interpolated,) float32
      n_detections        # (total_frames,) int32
      frame_mapping       # (N_interpolated,) int32
      detection_source    # (N_interpolated,) int8
      reason              # (N_interpolated,) utf8 ("clean"/"interpolated")
      
      @total_detections
      @original_detections
      @interpolated_detections
      @gaps_filled
      @interpolation_stats = {
          "gaps_filled": 12,
          "mean_gap_size": 5.2,
          "max_gap_size": 15
      }
Array Specifications
filtered/ Arrays
bbox_norm_coords

Shape: (N_filtered, 4)
Type: float64
Description: Normalized bounding boxes [center_x, center_y, width, height] with jumps removed
Values: All values in [0, 1]

scores

Shape: (N_filtered,)
Type: float32
Description: Detection confidence scores (from original detections)
Values: [0, 1]

n_detections

Shape: (total_frames,)
Type: int32
Description: Number of detections per frame after filtering
Note: Most values will be 0 or 1 for single-fish tracking

frame_mapping

Shape: (N_filtered,)
Type: int32
Description: Maps each detection index back to its original frame number
Example: frame_mapping[5] = 107 means detection 5 came from frame 107

interpolated/ Arrays
bbox_norm_coords

Shape: (N_interpolated, 4)
Type: float64
Description: Bounding boxes including interpolated detections
Format: [center_x, center_y, width, height] normalized to frame dimensions
Note: N_interpolated = N_filtered + N_synthetic

scores

Shape: (N_interpolated,)
Type: float32
Description: Confidence scores (NaN or decayed for interpolated)
Values: [0, 1] for real detections, NaN or <original for interpolated

n_detections

Shape: (total_frames,)
Type: int32
Description: Number of detections per frame after interpolation
Note: Frames in filled gaps will have 1 detection

frame_mapping

Shape: (N_interpolated,)
Type: int32
Description: Frame number for each detection (real or interpolated)

detection_source

Shape: (N_interpolated,)
Type: int8
Description: Source of each detection
Values:

0 = Original clean detection
1 = Interpolated (synthetic)

reason

Shape: (N_interpolated,)
Type: UTF-8 string
Description: Per-detection label that mirrors `detection_source`
Values:

"clean" = original clean detection
"interpolated" = synthetic detection filled during gap interpolation



Metadata Attributes
Root-level Attributes
python{
    "source_detect_run": "detect_2025-10-03_20-28-11",
    "source_quality_run": "detect_quality_2025-10-03_20-30-45",
    "refinement_timestamp": "2025-10-03T21:00:00Z",
    "operations": ["filter_jumps", "interpolate_gaps"],
    "parameters": {
        "max_gap": 20,
        "interpolation_method": "linear",
        "filters_applied": ["remove_jumps"]
    }
}
filtered/ Attributes
python{
    "total_detections": 45973,
    "dropped_detections": 2,
    "drop_reasons": {
        "jumps": 2,
        "blips": 0
    }
}
interpolated/ Attributes
python{
    "total_detections": 46123,
    "original_detections": 45973,
    "interpolated_detections": 150,
    "gaps_filled": 12,
    "interpolation_stats": {
        "gaps_filled": 12,
        "mean_gap_size": 5.2,
        "max_gap_size": 15,
        "min_gap_size": 1
    }
}
Usage Examples
Loading Filtered Data (Real Detections Only)
pythonimport zarr
import numpy as np

root = zarr.open('data.zarr', mode='r')
refined_run = root['refined_detect_runs'].attrs['latest']
filtered_group = root[f'refined_detect_runs/{refined_run}/filtered']

# Load clean detections
bboxes = filtered_group['bbox_norm_coords'][:]
scores = filtered_group['scores'][:]
frame_mapping = filtered_group['frame_mapping'][:]

print(f"Loaded {len(bboxes)} clean detections")
print(f"Coverage: {np.sum(filtered_group['n_detections'][:] > 0)} frames")
Loading Interpolated Data
pythoninterp_group = root[f'refined_detect_runs/{refined_run}/interpolated']

# Load all detections (real + interpolated)
bboxes = interp_group['bbox_norm_coords'][:]
detection_source = interp_group['detection_source'][:]

# Separate real from synthetic
real_mask = detection_source == 0
interpolated_mask = detection_source == 1

real_bboxes = bboxes[real_mask]
synthetic_bboxes = bboxes[interpolated_mask]

print(f"Real: {np.sum(real_mask)}, Synthetic: {np.sum(interpolated_mask)}")
Filtering by Detection Source
python# For behavioral analysis - use only real detections
real_only = bboxes[detection_source == 0]

# For visualization - use complete trajectory
complete_trajectory = bboxes  # includes interpolated
Mapping Back to Original Frames
python# Get detections from a specific frame range
frame_mapping = interp_group['frame_mapping'][:]

# Find all detections from frames 100-200
frame_mask = (frame_mapping >= 100) & (frame_mapping <= 200)
frame_bboxes = bboxes[frame_mask]
frame_sources = detection_source[frame_mask]

print(f"Frames 100-200: {np.sum(frame_mask)} detections")
print(f"  Real: {np.sum(frame_sources[frame_mask] == 0)}")
print(f"  Interpolated: {np.sum(frame_sources[frame_mask] == 1)}")
Pipeline Integration
Creating a Refined Run
pythonfrom fisheye.refinement import create_refined_run

# Create new refined run
refined_run = create_refined_run(
    zarr_path='data.zarr',
    detect_run='detect_2025-10-03_20-28-11',  # optional, uses latest
    quality_run=None,  # optional, uses latest
    max_gap=20,
    interpolation_method='linear',
    filters=['remove_jumps']  # Could also include 'remove_blips'
)

print(f"Created refined run: {refined_run}")
Typical Workflow
bash# 1. Run detection
python -m fisheye.detect data.zarr

# 2. Analyze quality
python -m fisheye.refinement.detect_quality data.zarr --threshold 100

# 3. Create refined data
python -m fisheye.refinement.refine data.zarr --max-gap 20 --filter jumps

# 4. Analyze refined data
python -m fisheye.analysis.kinematics data.zarr --use-refined
Design Rationale
Why Two Stages (filtered + interpolated)?

Filtered is purely real data - important for:

Behavioral analysis requiring ground truth
Statistics on actual detections
Validation against other measurements


Interpolated includes synthetic data - useful for:

Smooth trajectory visualization
Continuous tracking applications
Gap-free time series analysis



Why Store Both?
Some analyses should use real data only (behavioral metrics), others benefit from continuous data (kinematics). By storing both, users can choose the appropriate dataset for their analysis without re-running refinement.
Why detection_source Instead of Separate Arrays?
Keeping all detections in one array (with a source label) maintains temporal ordering and makes it easy to:

Plot complete trajectories
Calculate frame-to-frame distances (with awareness of synthetic data)
Filter on-the-fly based on needs

Notes

Empty frames remain empty in both filtered and interpolated (unless filled by interpolation)
frame_mapping is critical for maintaining temporal information after filtering
Interpolated confidence scores could use decay: score * (1 - gap_fraction)
All refined runs are immutable once created
Multiple refined runs can exist with different parameters for comparison
