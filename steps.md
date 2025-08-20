# Fish Tracking Pipeline: From Raw Video to Heatmaps

## Overview
This document outlines the complete pipeline for processing fish tracking data from raw video and H5 stimulus files to final spatial heatmaps and analysis.

## Prerequisites

### Required Files
- **Video file**: `Cam{camera_id}.mp4` (e.g., `Cam2010096.mp4`)
- **Original stimulus H5 file**: Raw H5 file from the stimulus program (may need processing)
- **YOLO model**: Trained model weights (`.pt` file)

### Required Software
- Python environment with necessary packages
- YOLO (Ultralytics) for detection
- Zarr for data storage
- H5py for H5 file handling

### Important Note on H5 Files
The stimulus program may output a raw H5 file that needs to be processed into an `out_analysis.h5` file with the proper structure. Step 3 handles this conversion.

## Pipeline Steps

### Step 1: Process Video with YOLO → Create Detections Zarr
Run YOLO detection on the video to create initial detection data.

```bash
python yolo_video_zarr.py \
    --video Cam2010096.mp4 \
    --model path/to/your/model.pt \
    --output 2025-01-20_arena_4_detections.zarr \
    --conf 0.25 \
    --device cuda:0
```

**Output**: `detections.zarr` containing:
- `/bboxes` - Shape: [n_frames, max_detections, 4] with [x1, y1, x2, y2] format
- `/scores` - Detection confidence scores
- `/class_ids` - Class IDs for each detection
- `/n_detections` - Number of detections per frame

### Step 2: Frame Interpolation
Fill gaps in detection data where YOLO missed frames.

```bash
python zarr_frame_interpolator.py \
    2025-01-20_arena_4_detections.zarr \
    --method linear \
    --max-gap 10 \
    --min-confidence 0.5
```

**Output**: Enhanced zarr with:
- `/interpolation_runs/{run_name}/bboxes` - Interpolated bounding boxes
- `/interpolation_runs/{run_name}/interpolation_mask` - Boolean mask of interpolated frames
- `/interpolation_runs/{run_name}/metadata` - Interpolation parameters and statistics

### Step 3: Create Analysis H5 File with Frame Interpolation
Create a complete analysis H5 file from the raw stimulus H5, including interpolation of missing camera frames.

```bash
python create_analysis_h5.py \
    2025-08-12T20-25-51Z_arena_4_chaser.h5 \
    2025-08-12T20-25-51Z_arena_4_chaser_analysis.h5
```

**What this does**:
- Analyzes gaps in camera frame sequences (e.g., 698 missing frames out of 14,973)
- Interpolates missing frame metadata to maintain 60 FPS → 120 FPS alignment
- Copies all essential data groups with proper structure
- Creates interpolation masks and statistics

**Example output**:
```
📊 Analyzing frame gaps...
  ✅ Original frames: 14275
  ⚠️ Missing frames: 698
  🔍 Number of gaps: 320
  📏 Largest gap: 45 frames

🔧 Interpolating missing frames...
  ✅ Created 1396 interpolated records for 698 missing camera frames

📈 Summary:
  - Original frames: 14275
  - Interpolated frames: 1396
  - Total frames: 15671
  - Gaps filled: 320
```

**Output**: Complete `analysis.h5` containing:
- `/video_metadata/frame_metadata` - Interpolated frame timing (15,671 records)
- `/tracking_data/bounding_boxes` - Fish detection data from stimulus program
- `/tracking_data/chaser_states` - Chaser position data (29,763 records)
- `/calibration_snapshot/` - Homography matrix and calibration data
- `/events` - Experimental events (64 events)
- `/protocol_snapshot/` - Protocol definition
- `/analysis/` - Interpolation statistics and mask

### Step 4: Verify Coordinate Transformations
Ensure coordinate systems are properly understood before distance calculations.

```bash
# First, verify the coordinate transformation setup
python coordinate_transform_module.py \
    2025-08-12T20-25-51Z_arena_4_chaser_analysis.h5 \
    --verify
```

**Expected output**:
```
COORDINATE TRANSFORMATION VERIFICATION
Texture space: 358x358
Camera space: 4512x4512
Scale: 12.604 camera_px/texture_unit
✅ Texture center (179, 179) → Camera (2256, 2256)
```

### Step 5: Calculate Fish-Chaser Distances
Analyze spatial relationships with CORRECTED coordinate transformations.

```bash
python chaser_fish_analyzer.py \
    --zarr 2025-01-20_arena_4_detections.zarr \
    --h5 2025-08-12T20-25-51Z_arena_4_chaser_analysis.h5 \
    --interpolation-run interp_linear_20250120 \
    --use-texture-scaling  # Use direct scaling, not homography
```

**Critical Process Changes**:
1. **DO NOT use homography for chaser transformation** (homography is for projector↔camera)
2. **Use direct scaling** from texture (358×358) to camera (4512×4512) space
3. **Scale factors**: 12.604 for both X and Y (camera_pixels/texture_unit)
4. Calculate frame-by-frame distances in camera space
5. Generate behavioral metrics

**Coordinate Transformation Chain**:
```
Chaser (Texture: 358×358) → [Scale ×12.604] → Camera (4512×4512)
Fish (Camera: 4512×4512) → [No transform needed] → Camera (4512×4512)
Distance calculation in common camera space
```

**Output**: Updates zarr with:
- `/analysis/chaser_fish_distances` - Distance metrics in pixels and world units
- `/analysis/relative_velocities` - Velocity calculations
- `/analysis/pursuit_angles` - Angular relationships

### Step 5: Copy Events Data to Analysis H5 (if missing)
If events weren't copied during H5 creation, add them now.

```bash
python h5_add_events_copier.py \
    original_stimulus.h5 \
    out_analysis.h5
```

**Output**: H5 file updated with `/events` group containing:
- Stimulus onset/offset times
- Training period markers
- Experimental conditions

### Step 6: Generate Training Heatmaps
Create spatial heatmaps comparing pre/post training periods.

```bash
python training_heatmap_analyzer.py \
    2025-01-20_arena_4_detections.zarr \
    out_analysis.h5 \
    --interpolation-run interp_linear_20250120 \
    --bin-size 50 \
    --save-plot heatmaps_arena_4.png
```

**Process**:
1. Load events to identify training periods
2. Extract fish positions for pre/post training
3. Apply coordinate transformations if needed
4. Generate 2D histograms (heatmaps)
5. Calculate spatial statistics

**Output**:
- Heatmap visualizations (PNG/PDF)
- Statistical report of spatial occupancy changes
- Behavioral metrics (distance traveled, velocity distributions)

### Step 7: (Optional) Visualize Results
Interactive visualization to verify tracking quality.

```bash
# For high-performance visualization
python vispy_pyqt_visualizer.py \
    --video Cam2010096.mp4 \
    --h5 out_analysis.h5 \
    --zarr 2025-01-20_arena_4_detections.zarr

# For frame-by-frame inspection
python interactive_prediction_viewer.py \
    --zarr 2025-01-20_arena_4_detections.zarr \
    --model path/to/your/model.pt \
    --output visualization_output/
```

## Coordinate System Handling

### Understanding the Coordinate Spaces

**THREE distinct coordinate systems exist**:

1. **TEXTURE SPACE** (358×358)
   - Where chaser/target positions are defined
   - Origin: Top-left of texture
   - Center: (179, 179)
   - Used by: Stimulus program for chaser positions

2. **CAMERA SPACE** (4512×4512)  
   - Raw pixel coordinates from tracking camera
   - Origin: Top-left of camera image
   - Center: (2256, 2256)
   - Used by: YOLO detections, fish positions

3. **PROJECTOR SPACE** (1920×1080)
   - Global projector coordinates
   - Sub-arena exists within this space at specific offset
   - Used by: Display system (not directly used in analysis)

### Critical Transformation Rules

**✅ CORRECT Transformations**:
```python
# Texture → Camera (Simple scaling)
scale = 4512 / 358  # = 12.604
camera_x = texture_x * scale
camera_y = texture_y * scale

# Example: Chaser at texture center
texture_pos = (179, 179)
camera_pos = (179 * 12.604, 179 * 12.604) = (2256, 2256)
```

**❌ INCORRECT (but commonly mistaken)**:
- Do NOT use homography for texture→camera transformation
- Homography is ONLY for projector↔camera transformation
- Texture space requires simple linear scaling, not perspective transform

### Transformation Chain for Analysis
```
Chaser Position (Texture: 358×358)
    ↓ [Linear scale ×12.604]
Chaser Position (Camera: 4512×4512)
    ↓ [Direct comparison]
Fish Position (Camera: 4512×4512) [Already in camera space]
    ↓ [Calculate Euclidean distance]
Distance in Camera Pixels
```

### Key Transformations
1. **Texture→Camera**: Linear scaling (×12.604 for 4512/358)
2. **Projector↔Camera**: Homography matrix (rarely needed for analysis)
3. **Frame Alignment**: 60 FPS video ↔ 120 FPS stimulus
4. **Bbox Format**: Zarr [x1,y1,x2,y2] ↔ H5 [x,y,w,h]

## Quality Checks

### After Each Step
1. **Post-YOLO**: Check detection rate, confidence scores
2. **Post-Interpolation**: Verify gap filling, check interpolation mask coverage
3. **Post-Distance Calc**: Validate coordinate transformations, check for outliers
4. **Post-Heatmap**: Ensure sufficient data in both periods, check for artifacts

### Validation Commands
```bash
# Inspect zarr structure
python zarr_inspector.py 2025-01-20_arena_4_detections.zarr

# Validate homography
python load_homography_test.py out_analysis.h5

# Check frame alignment
python debug_bbox_transform.py 2025-01-20_arena_4_detections.zarr
```

## Common Issues & Solutions

### Issue: Low detection rate
**Solution**: Adjust YOLO confidence threshold or retrain model

### Issue: Interpolation creating artifacts
**Solution**: Reduce max-gap parameter or use different interpolation method

### Issue: Coordinate transformation errors
**Solution**: Validate homography matrix, check for proper inverse calculation

### Issue: Misaligned frames
**Solution**: Verify frame rate ratio, check camera_frame_ids in H5

### Issue: Empty heatmaps
**Solution**: Check event timing, ensure sufficient detections in analysis periods

## Expected Outputs

### Final Data Products
1. **Enhanced Zarr File** with:
   - Original detections
   - Interpolated positions
   - Distance metrics
   - Analysis results

2. **Updated H5 File** with:
   - Original stimulus data
   - Events and metadata
   - Cross-references to zarr analysis

3. **Visualization Outputs**:
   - Spatial heatmaps (PNG/PDF)
   - Statistical reports (TXT/CSV)
   - Interactive visualizations (if generated)

## Performance Considerations

- **GPU Usage**: YOLO detection benefits significantly from GPU
- **Memory**: Large videos may require chunked processing
- **Storage**: Zarr files can be large; ensure sufficient disk space
- **Processing Time**: Full pipeline ~10-30 minutes per video depending on length

## Next Steps

After completing the pipeline:
1. Aggregate results across multiple fish/sessions
2. Statistical analysis of behavioral changes
3. Generate publication-ready figures
4. Archive processed data with metadata

## References

- [YOLO Documentation](https://docs.ultralytics.com)
- [Zarr Format Specification](https://zarr.readthedocs.io)
- [HDF5 Documentation](https://www.hdfgroup.org/solutions/hdf5/)