Refined Runs Implementation Specification
Overview
The refined_runs/ group stores filtered and interpolated detection data, providing clean datasets for downstream analysis while maintaining full traceability to the original detections.

## Eye Mask Refinement (new)

Eye segmentation runs can now be post-processed with `python -m fisheye.refinement.refine_eye_masks`. The tool:

- Reads an existing `eye_masks_runs/<source>` entry plus its paired keypoint run
- Reassigns pixels to anatomical left/right using keypoint geometry (with heading-based fallback)
- Emits a new `eye_masks_runs/<run_name>` containing the same arrays as the traditional segmenter (`masks_roi`, ellipse metrics, Feret axes, contour tables, etc.)
- Records provenance in the run attributes (`method="refine_eye_masks"`, `source_*_run`, `source_eye_masks_method`, `eye_labels`, summary stats)

This keeps the original masks untouched for provenance while providing a refined alternative that matches keypoint labels.

Directory Structure
/refined_runs/
  @latest = "refined_2025-10-03_21-00-00"
  
  /refined_2025-10-03_21-00-00/
    # Root-level metadata
    @source_detect_run = "detect_2025-10-03_20-28-11"
    @source_quality_run = "detect_quality_2025-10-03_20-30-45"
    @refinement_timestamp = "2025-10-03T21:00:00Z"
    @operations = ["filter_jumps", "interpolate_gaps"]
    @parameters = {
        "max_gap": 20,
        "interpolation_method": "linear",
        "filters_applied": ["remove_jumps"]
    }
    
    /filtered/
      bbox_norm_coords     # (N_filtered, 4) float64
      scores              # (N_filtered,) float32
      n_detections        # (total_frames,) int32
      frame_mapping       # (N_filtered,) int32
      
      @total_detections
      @dropped_detections
      @drop_reasons = {"jumps": 2, "blips": 0}
      
    /interpolated/
      bbox_norm_coords     # (N_interpolated, 4) float64
      scores              # (N_interpolated,) float32
      n_detections        # (total_frames,) int32
      frame_mapping       # (N_interpolated,) int32
      detection_source    # (N_interpolated,) int8
      
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
refined_run = root['refined_runs'].attrs['latest']
filtered_group = root[f'refined_runs/{refined_run}/filtered']

# Load clean detections
bboxes = filtered_group['bbox_norm_coords'][:]
scores = filtered_group['scores'][:]
frame_mapping = filtered_group['frame_mapping'][:]

print(f"Loaded {len(bboxes)} clean detections")
print(f"Coverage: {np.sum(filtered_group['n_detections'][:] > 0)} frames")
Loading Interpolated Data
pythoninterp_group = root[f'refined_runs/{refined_run}/interpolated']

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
