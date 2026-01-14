# Palette Zarr Layout (v3)

This reference summarizes the structure produced by the modern Palette
pipeline.  It should match what `fisheye.shared.zarr.schema` and the
stage implementations currently write.

---

## Root Group

**Attributes**

- `schema_version`, `zarr_format`
- `created_at`, `pipeline_version`, `command_line_args`
- `git_info`, `platform_info`
- `source_video_metadata` (width, height, fps, frames, codec, path)
- `processing_history` *(optional ordered list)*

**Immediate children**

- `raw_video/`
- `background_runs/`
- `detect_runs/`
- `crop_runs/`
- `keypoints_runs/`
- `eye_masks_runs/`
- `refined_detect_runs/`
- `refined_keypoints_runs/`
- `refined_eye_masks_runs/`
- `refined_online_runs/`
- `id_assignment_runs/`
- `calibration/`
- `analysis/`
- `analysis_metadata/`

Every `*_runs` group carries:

- `attrs["latest"]` → most recent run name.
- Child run groups named `<stage>_<YYYY-MM-DD_hh-mm-ss>` (possibly with a
  `_NNN` suffix if repeated inside the same second).
- Run attributes capturing provenance (`provenance` dict with command,
  timestamps, git/environment snapshots), upstream run references
  (`source_detect_run`, `source_crop_run`, etc.), configuration snapshots,
  and `duration_seconds`.

---

## `raw_video/`

Arrays written during import (kvikIO or standard path):

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `images_full` | `(n_frames, H, W)` | `uint8` | Full-resolution frames (optional) |
| `images_ds` | `(n_frames, H_ds, W_ds)` | `uint8` | Downsampled frames (optional) |
| `timestamps` | `(n_frames,)` | `float64` | Seconds since start (optional) |

Attributes include import method, device, chunk/shard sizes, duration,
throughput, and source video metadata.

---

## `background_runs/`

Run arrays vary by method; common layout:

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `background_full` | `(H, W)` | `uint8` | Full-resolution background |
| `background_ds` | `(H_ds, W_ds)` | `uint8` | Downsampled background |
| `frame_indices` | `(n_samples,)` | `int32` | Frames sampled for background |

Attributes: `method`, `parameters`, `num_samples`, `duration_seconds`,
provenance, GPU/system info.

---

## `detect_runs/`

Outputs from blob/YOLO detection stages.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` | `(n_detections,)` | `int32` | Corresponding frame per detection |
| `frame_counts` | `(n_frames,)` | `int32` | Number of detections per frame |
| `n_detections` | `(n_frames,)` | `int32` | Alias of `frame_counts` (kept for legacy consumers) |
| `bbox_norm_coords` | `(n_detections, 4)` | `float32` | Normalized `[x, y, w, h]` |
| `scores` | `(n_detections,)` | `float32` | Confidence scores |
| `class_ids` *(optional)* | `(n_detections,)` | `int32` | Detector class labels |
| `centers_px` *(optional)* | `(n_detections, 2)` | `float32` | Pixel centers (blob) |

Attributes store detector `method`, model identifiers, thresholds, duration,
and upstream background information. Standard values:

| Method | When used |
| ------ | --------- |
| `blob` | Traditional background-subtraction detector |
| `yolo_detect` | YOLO object detector |
| `yolo_pose` | YOLO pose detector |

---

## `crop_runs/`

Each run stores the cropped ROI tensors and the bookkeeping needed to map
ROIs back to frames.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `roi_images` | `(n_rois, h, w)` | `uint8` | Cropped grayscale patches |
| `roi_coordinates_full` | `(n_rois, 2)` | `int32` | Top-left (x, y) in full-res pixels |
| `roi_coordinates_ds` | `(n_rois, 2)` | `int32` | Same offsets in downsampled space |
| `bbox_norm_coords` | `(n_rois, 4)` | `float32` | Normalized ROI bounding boxes |
| `frame_indices` | `(n_rois,)` | `int32` | Frame index per ROI |
| `frame_counts` | `(n_frames,)` | `int32` | Count of ROIs per frame |
| `detection_source` *(optional)* | `(n_rois,)` | `int8` | 0 = real detection, 1 = interpolated (copied from refined runs) |
| `detection_indices` *(optional)* | `(n_rois,)` | `int32` | Index into source detect run |

Attributes:

- `source_detect_run`, `source_background_run`, `detection_source_type`,
  `detection_source_path`, `includes_interpolated`, `n_real_detections`,
  `n_interpolated_detections`, ROI size, scaling factors.
- `summary_statistics` (frames with crops, total ROIs, percentage coverage).
- GPU/environment provenance.

Cropping resolves the ROI source via `crop.source_type` (`detect`, `filtered`,
`interpolated`) or an explicit `crop.source_path` override such as
`detect_runs/<run>` or `refined_detect_runs/<run>/interpolated`, and the chosen
path is recorded in `detection_source_path`.

---

## `keypoints_runs/`

Produced by the keypoint detection stage (traditional or YOLO-based).

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` | `(n_rois,)` | `int32` | Inherit from corresponding crops |
| `frame_counts` | `(n_frames,)` | `int32` | ROIs per frame (mirrors `n_rois`) |
| `n_rois` | `(n_frames,)` | `int32` | Alias maintained for legacy callers |
| `keypoints_roi` | `(n_rois, n_keypoints, 2)` | `float64` | Coordinates in ROI pixels |
| `keypoints_img` | `(n_rois, n_keypoints, 2)` | `float64` | Full-image pixels |
| `keypoints_norm` | `(n_rois, n_keypoints, 2)` | `float64` | Normalized [0,1] |
| `heading` | `(n_rois,)` | `float64` | Degrees, NaN when unavailable |
| `confidence` | `(n_rois,)` | `float64` | Overall score |
| `effective_threshold` | `(n_rois,)` | `float64` | Per-ROI threshold used |
| `effective_se2_radius` | `(n_rois,)` | `float64` | Search radius actually applied |
| `detection_success` | `(n_rois,)` | `bool` | True if keypoints converged |

Attributes: `source_crop_run`, `source_detect_run`, `method`, `parameter_source`,
`keypoint_labels`, scheduler configuration, timing, QA summaries.

---

## `eye_masks_runs/`

Generated by segmentation inference (`infer_unet_eye_masks.py`,
`eye_segmentation_yolo.py`, etc.).

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` | `(n_rois,)` | `int32` | ROI frame mapping (copied from crop run) |
| `frame_counts` | `(n_frames,)` | `int32` | Count of ROIs per frame (matches source crop run) |
| `detection_indices` | `(n_rois,)` | `int32` | Index into `crop_runs/<run>/roi_images` |
| `masks_roi` | `(n_rois, 2, H, W)` | `uint8` | Binary masks (left/right) |
| `mask_probs_roi` | `(n_rois, 2, H, W)` | `float16/float32` | Raw probabilities |
| `mask_scores` *(optional)* | `(n_rois,)` | `float32` | Confidence per ROI |

Attributes: `source_crop_run`, `source_keypoint_run`, `method`, model info,
thresholds, smoothing parameters, aggregate metrics.

---

## `refined_detect_runs/`

Created by `fisheye.refinement.refine_detect`.  Inherits the detection arrays
from its source run and adds QA channels.

| Array | Shape | Notes |
| ----- | ----- | ----- |
| `bbox_norm_coords` | `(n_detections, 4)` | Updated boxes |
| `scores` | `(n_detections,)` | Updated confidences |
| `reason` | `(n_detections,)` | UTF-8 labels (e.g. `refined`, `copied`, `filtered`) |
| `qa_metrics/*` | Various | Stage-specific floats/ints for diagnostics |

Attributes summarize counts by `reason`, thresholds, and references to the
source detect and crop runs. Optional visual artifacts may be stored under
`visualizations/` (e.g., `visualizations/detect_quality_overview_png` containing a
PNG summary of the detection-quality analysis with attrs `mime='image/png'`,
`source_detect_run`, and `source_quality_run`).

---

## `refined_keypoints_runs/`

Outputs from `fisheye.refinement.refine_keypoints`.

- Copies `keypoints_roi`, `keypoints_img`, `heading`, etc. from the source
  run (after refinement).
- Adds `reason` (string tag per ROI), boolean masks for validity, and QA
  metrics like `heading_std`, `drift_px`, `blend_weight`.
- Attributes include source keypoint + crop runs, smoothing parameters,
  per-reason statistics, derivative filters applied.

---

## `refined_eye_masks_runs/`

See `fisheye.refinement.refine_eye_masks`.  Key arrays:

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `masks_roi` | `(n_rois, 2, H, W)` | `uint8` | Refined masks |
| `mask_probs_roi_refined` | `(n_rois, 2, H, W)` | `float16` | Probabilities after refinement |
| `ellipse_params` | `(n_rois, 2, 5)` | `float32` | `[cx, cy, major, minor, angle]` |
| `ellipse_success` | `(n_rois, 2)` | `bool` | Fit success per eye |
| `eye_separation` | `(n_rois,)` | `float32` | Centroid distance |
| `reason` | `(n_rois,)` | `string` | Classification tags (`refined`, `copied`, `filtered_*`) |

`metrics/` subgroup:

- Scalar QA arrays such as `area_refined`, `area_delta_vs_source`,
  `centroid_error`, `symmetry_offsets`, `separation_delta`, `axis_ratio`,
  `circularity`, `probability_*`, `filter_flags`, `connectivity_flags`,
  `smoothing_flags`, `pixels_reassigned`.

Attributes expose `metrics_summary`, configuration snapshots,
per-eye filter thresholds, and links to source runs.

---

## `id_assignment_runs/`

Generated by `fisheye.tracking.assign_ids` and friends.

Common arrays:

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `identities` | `(n_detections,)` | `int32` | Assigned fish/ROI IDs |
| `confidence` | `(n_detections,)` | `float32` | Assignment score |
| `frame_indices` | `(n_detections,)` | `int32` | Optional copy of frame map |

Attributes describe the assignment strategy, ROI definitions used,
expected counts, and QA tallies (`assigned`, `unassigned`).

---

## `calibration/`

Calibration metadata imported from stimulus H5 files. This group stores calibration
parameters for both camera and projector coordinate spaces.

**Attributes**:

| Attribute | Description | Units | Notes |
|-----------|-------------|-------|-------|
| `pixel_to_mm` | Camera-space calibration | pixels/mm | For camera coordinates (4512×4512) |
| `pixels_per_mm_camera` | Alias for pixel_to_mm | pixels/mm | Camera space |
| `pixels_per_mm_projector` | Projector-space calibration | pixels/mm | **For texture/stimulus coordinates (358×358)** |
| `z_eff_mm` | Effective viewing distance | mm | Accounts for refraction through media |
| `measured_fps` | Measured frame rate | fps | Actual measured FPS |
| `arena_shape` | Arena shape | - | "CIRCLE" or "RECTANGLE" |
| `arena_center_x_px` | Arena center X | pixels | Camera space |
| `arena_center_y_px` | Arena center Y | pixels | Camera space |
| `arena_radius_px` | Arena radius (if circle) | pixels | Camera space |
| `arena_width_px` | Arena width (if rectangle) | pixels | Camera space |
| `arena_height_px` | Arena height (if rectangle) | pixels | Camera space |

**CRITICAL: Dual Calibration System**

There are **two separate calibrations** because the camera and projector operate in
different coordinate spaces:

1. **Camera Space** (4512×4512 pixels):
   - Used for: Offline detection/tracking, bounding boxes, keypoint positions
   - Calibration: `pixels_per_mm_camera` (or `pixel_to_mm`)
   - Typical value: ~5.8 pixels/mm

2. **Texture/Projector Space** (358×358 pixels):
   - Used for: Online stimulus positions (chaser, target), stimulus rendering
   - Calibration: `pixels_per_mm_projector`
   - Typical value: ~0.44 pixels/mm

**Why Two Calibrations?**
- Camera and projector have vastly different resolutions
- Camera may view the arena at an angle (requires homography)
- Scaling between spaces (factor ~12.6) does NOT preserve physical distances
- 1 pixel in texture space ≠ 12.6 pixels in camera space (in real-world distance)

**Usage Guidelines**:
- **Online/stimulus data**: Use `pixels_per_mm_projector`
- **Offline/detection data**: Use `pixels_per_mm_camera`
- **Never mix calibrations**: Apply the calibration matching the coordinate space

---

## `refined_online_runs/`

Refined online target positions from stimulus runs. These are the smoothed, outlier-removed,
and gap-interpolated positions used for accurate movement analysis.

**Structure**: `refined_online_runs/<run_name>/`

See `analysis/refined_online_runs/` section below for detailed structure.

---

## `analysis/`

Organized by analyzer:

### `analysis/stimulus_runs/`

Stimulus run data imported from Citrus H5 files. Each run contains:

**Structure**: `analysis/stimulus_runs/<run_name>/`

**Run Attributes**:
- `created_at_utc`: Import timestamp
- `source_h5`: Path to source H5 file
- `import_version`: Import script version
- `protocol_json`: Protocol definition (JSON string)
- `arena_config_json`: Arena/calibration configuration (JSON string)
- `coordinate_transform`: Coordinate system info (JSON string)
  - `texture_dimensions`: Stimulus texture size (typically [358, 358])
  - `camera_dimensions`: Camera resolution (typically [4512, 4512])
  - `texture_to_camera_scale`: Scale factor (~12.6)
  - `coordinate_note`: Description of coordinate spaces
- Gap analysis stats: `total_frames`, `missing_frames`, `max_gap_size`, etc.

**Arrays**:

| Array/Group | Description |
|-------------|-------------|
| `video_metadata/frame_metadata` | Columnar table with frame timing and IDs (see below) |
| `interpolation_mask` | Boolean mask indicating interpolated frames |
| `frame_alignment/camera_to_metadata_index` | Absolute camera frame → metadata index map. Length = max camera frame + 1. Slots with no stimulus data are `-1`. |
| `frame_alignment/camera_interpolation_mask` | Boolean mask for the above indices (`True` only when every metadata row for that camera frame was recorded, not interpolated). Frames with `-1` entries are `False`. |
| `frame_alignment` attrs | `camera_frame_offset` *(legacy)*: original minimum triggering frame retained for backward compatibility. |
| `tracking_data/chaser_states` | Per-frame chaser position/state data (columnar) |
| `tracking_data/bounding_boxes` | Detection bounding boxes from tracking system (columnar) |
| `events/` | Experimental events (columnar storage, see below) |

**Frame Metadata Fields** (`video_metadata/frame_metadata`):
- Stored as separate datasets (columnar layout) for Zarr v3 compatibility.
- `stimulus_frame_num`: Stimulus frame counter (uint64)
- `triggering_camera_frame_id`: Corresponding camera frame ID (uint64)
- `timestamp_relative_ns`: Time since session start (int64)

**Bounding Box Fields** (`tracking_data/bounding_boxes`):
- Stored column-wise. Geometry columns (`x_min`, `y_min`, `width`, `height`) come directly from the tracker.
- `centroid_x`, `centroid_y`: Computed during import as the midpoint of each bounding box in camera pixels.
- `payload_timestamp_ns_epoch`, `received_timestamp_ns_epoch`: Tracker timing fields (int64)
- `payload_frame_id`, `payload_camera_id`: Tracker identifiers for frame/camera (uint64/uint16)
- `box_index_in_payload`: Index of the detection in the tracker payload (uint8)
- `class_id`, `confidence`: Detector metadata (uint16/float32)

**Chaser States Fields** (`tracking_data/chaser_states`):
All fields stored as separate columnar arrays. Key fields include:
- `frame_number`: Stimulus frame counter (uint64)
- `camera_frame_id`: Camera frame ID (uint64)
- `relative_timestamp_ns`: Time since session start (int64)
- `chaser_index`: Chaser agent index (int32)
- `pos_x_px`, `pos_y_px`: Chaser position in **texture space** (float32)
- `target_x_px`, `target_y_px`: Target position in **texture space** (float32)
- `target_visible`: Whether target is tracked (bool)
- `current_radius_px`: Chaser radius (float32)
- `distance_to_target_px`, `distance_to_target_mm`: Distance to target (float32)
- `chase_speed_px_per_s`, `chase_speed_mm_per_s`: Speed (float32)
- `visual_angle_deg`: Visual angle at target (float32)
- `angular_velocity_deg_s`: Angular expansion rate (float32)
- `tau_ms`: Time to collision (float32)
- `loom_mode`, `loom_phase`: Loom behavior state (uint8)
- `trial_state`: PRE/TRAINING/POST period (uint8)
- `chase_sequence_active`: Whether chase is active (bool)

**IMPORTANT**: Chaser positions (`pos_x_px`, `pos_y_px`, `target_x_px`, `target_y_px`)
are in **texture space** (358×358 pixels), NOT camera space (4512×4512). Use
`texture_to_camera_scale` from `coordinate_transform` if conversion is needed.

**Events Structure** (`events/`):
Events are stored in columnar format with separate arrays for each field:
- `relative_timestamp_ns`: Event timestamp (int64)
- `event_type`: Event type ID (int32)
- `step_index`: Protocol step index (int32)
- `event_name`: Event name (variable-length UTF-8)
- `stimulus_mode`: Active stimulus mode (int32)
- `details`: Event details JSON (variable-length UTF-8)
- `stimulus_frame_num`: Stimulus frame (uint64)
- `camera_frame_id`: Camera frame (uint64)

**Enum Definitions** (`analysis/enums/`):
Enumeration tables for event types, stimulus modes, etc. Each enum is stored as:
- `enums/<enum_name>/id`: Enum ID values (int32)
- `enums/<enum_name>/name`: Enum names (variable-length UTF-8)

Common enums:
- `event_types`: All experimental event types
- `stimulus_modes`: All stimulus mode types
- `chaser_loom_modes`: Chaser looming behaviors
- `chaser_trial_states`: PRE/TRAINING/POST periods

### `analysis/calibration/`

Calibration metadata extracted from H5 files:

**Attributes**:
- `pixel_to_mm`: Camera-space calibration (pixels/mm) **[for camera space coordinates]**
- `pixels_per_mm_camera`: Alias for `pixel_to_mm` (pixels/mm)
- `pixels_per_mm_projector`: Projector/texture-space calibration (pixels/mm) **[for texture space coordinates]**
- `z_eff_mm`: Effective viewing distance through media
- `measured_fps`: Measured frame rate
- `arena_shape`: CIRCLE or RECTANGLE
- `arena_center_x_px`, `arena_center_y_px`: Arena center
- `arena_radius_px` or `arena_width_px`, `arena_height_px`: Arena dimensions

**CRITICAL CALIBRATION NOTE**:
There are **two separate calibrations** for the two coordinate spaces:
- **`pixels_per_mm_camera`**: For camera-space coordinates (4512×4512 pixels)
  - Used for offline detection/tracking data
  - Used for bounding boxes from detection system
- **`pixels_per_mm_projector`**: For projector/texture-space coordinates (358×358 pixels)
  - Used for online stimulus positions (chaser, target)
  - Used for stimulus-related measurements
  - **This is the authoritative calibration for distance calculations on online/stimulus data**

These are different because:
1. Camera and projector have different resolutions
2. Camera may view the arena at an angle
3. Scaling positions between spaces does NOT preserve physical distances
4. A 1-pixel movement in texture space ≠ 12.6× the distance of 1 pixel in camera space

### `analysis/movement_runs/`

Movement analysis results organized by type:

**Structure**: `analysis/movement_runs/<online|offline>/<run_name>/`

**Run Attributes**:
- `method`: Analysis method used
  - `movement_analysis_online`: Raw online data (transformed to camera space)
  - `movement_analysis_online_refined`: Refined online data (texture space)
  - `movement_analysis_offline`: Offline detection data
- `created_at_utc`: Analysis timestamp
- `fps`: Frame rate used
- `smoothing_seconds`: Temporal smoothing window
- `pixel_to_mm`: Calibration used for this run
  - For `online_refined`: Uses `pixels_per_mm_projector` (texture space)
  - For `online` and `offline`: Uses `pixels_per_mm_camera` (camera space)
- `coordinate_space`: "texture" or "camera"
- `inputs`: Source data references
  - Online refined: `refined_online_run`, `stimulus_run`, `chaser_index`
  - Online raw: `stimulus_run`, `chaser_index`
  - Offline: `detection_run`, `keypoint_run`, optional `chaser_metrics` dict (metrics run, stimulus run, chaser index)
- `summary`: Per-track summary statistics
- `total_distance_px`, `total_distance_mm`: Aggregate distances

**Shared Root Arrays (offline runs only)**:
- `camera_frame_ids` (`int64`): Master frame index aligned to all chaser metrics.
- `stimulus_frame_nums` (`int64`), `timestamp_ns` (`int64`), `trial_state` (`int16`): Optional context per frame.
- `metadata_mask` (`bool`, optional): Propagated interpolation/original mask when available.
- `has_offline` (`bool`): Indicates frames with valid chaser metrics.
- `distance_to_target_px`, `distance_to_target_mm` (`float32`): Chaser→target separation per frame.
- `distance_to_target_interpolated_px`, `distance_to_target_interpolated_mm` (`float32`, optional): Raw distances with short NaN gaps (duration ≤ `distance_interpolation_seconds` × FPS) filled via linear interpolation.
- `distance_to_target_smoothed_px`, `distance_to_target_smoothed_mm` (`float32`, optional): Moving-average smoothing applied to the interpolated series using the movement run's `fps` and `smoothing_seconds`.
- `chaser_position_px`, `chaser_positions_px` (`float32`, `[N, 2]`): Chaser centroid in camera pixels (duplicate naming retained for compatibility).
- `fish_centroid_px`, `fish_centroids_px` (`float32`, `[N, 2]`): Target centroid in camera pixels.
- `angle_signed_deg`, `angle_unsigned_deg`, `heading_deg` (`float32`): Per-frame angular metrics from `compute_chaser_fish_metrics`.

Consumers map from track-level `frame_indices` into these arrays using `camera_frame_ids` and the `has_offline` mask.

**Per-Track Data** (`tracks/id_<track>/`):
Each track stores the ordered samples for that ID:
- `frame_indices` (`int64`), `time_seconds` (`float32`), `detection_indices` (`int64`)
- `positions_px`, `positions_mm` (`float32`, `[N, 2]`)
- `instantaneous_speed_px`, `instantaneous_speed_mm`, `smoothed_speed_px`, `smoothed_speed_mm`
- `instantaneous_speed_filtered_px`, `instantaneous_speed_filtered_mm` *(optional)*: Speeds using displacement pre-smoothing (saved in speed/movement runs)
- `heading_degrees`, `heading_radians`, `smoothed_heading_degrees`, `smoothed_heading_radians`
- `acceleration_px`, `acceleration_mm`, `smoothed_acceleration_px`, `smoothed_acceleration_mm`
- `distance_per_frame_px`, `distance_per_frame_mm`, `cumulative_distance_px`, `cumulative_distance_mm`
- `distance_per_frame_raw_px`, `distance_per_frame_raw_mm`: Pre-smoothing frame-to-frame displacement (pixels / converted millimeters)
- `second_indices`, `speed_per_second_px`, `speed_per_second_mm`, `heading_per_second_degrees`, `heading_per_second_resultant`
- `keypoint_success`, `detection_source`, plus per-track manifest metadata in subgroup attributes
- `swim_bouts/`: columnar arrays mirroring `analysis/swim_bout_runs/<run>/bouts` (e.g., `bout_id`, `start_time_s`, `end_time_s`, `start_frame`, `end_frame`, `duration_s`, `distance_px`, `distance_mm`, `mean_speed_mm_s`, `peak_speed_mm_s`, `start_x_px`, `end_x_px`, …) with subgroup attrs recording the source swim-bout run.

Track-level arrays remain unchanged between online and offline runs; only the root-level chaser metrics are added for offline runs.

### `analysis/refined_online_runs/`

Refined online target positions from stimulus runs (smoothed, outlier-removed, gap-filled).

**Structure**: `analysis/refined_online_runs/<run_name>/`

**Run Attributes**:
- `source_stimulus_run`: Source stimulus run name
- `chaser_index`: Which chaser was tracked
- `texture_to_camera_scale`: Scale factor between coordinate spaces
- `coordinate_space`: "texture" (positions in texture space)
- `pixels_per_mm_projector`: Texture-space calibration (pixels/mm)
- `refinement_timestamp`: Processing timestamp
- `processing_time_seconds`: Time to process
- `operations`: List of operations performed (["smooth", "outlier_removal", "interpolate"])
- `parameters`: Refinement parameters (window_length, polyorder, displacement_threshold, max_gap)
- `coverage_stats`: Coverage before/after refinement
- `outlier_stats`: Outliers detected and removed
- `interpolation_stats`: Gaps filled and frames interpolated

**Arrays**:
- `camera_frame_ids`: Camera frame IDs (N,) int32
- `original_valid_mask`: Original validity mask (N,) bool
- `smoothed_mask`: Valid after smoothing (N,) bool
- `outlier_mask`: Detected outliers (N,) bool

**Subgroups**:

`filtered/`: After smoothing and outlier removal
- `camera_frame_ids`: Frame IDs
- `positions_px`: Positions in **texture space** (N, 2) float64
- `valid_mask`: Valid positions mask

`interpolated/`: Final refined positions (after gap filling)
- `camera_frame_ids`: Frame IDs
- `positions_px`: Positions in **texture space** (N, 2) float64
- `valid_mask`: Valid positions mask
- `interpolation_mask`: Identifies interpolated positions (bool)

**IMPORTANT**: All positions are in **texture space** (358×358 pixels).
Use `pixels_per_mm_projector` for distance calculations.

### `analysis/eye_angle_runs/`

Eye angle analysis results:
- Per-ROI and per-frame eye-angle metrics
- QA masks and quality indicators
- `reason_codes` for data quality classification

### Additional Analysis Groups

Other analyzers follow the same `analysis/<analysis_type>_runs/<run_name>/` pattern with
analyzer-specific arrays and provenance attributes.

---

## `analysis_metadata/`

Lightweight store for metadata generated during tuning or diagnostics.
Examples:

- `attrs["dish_mask"]` – saved dish mask parameters from the tuner
  (circle center/radius with Hough params or rectangle ROI).
- `attrs["subdish_mask_tuning"]` – multi-dish ROI definitions.
- Other agents may add read-only metadata blocks here.

---

## Coordinate Spaces & Calibration Quick Reference

**Question: Which calibration should I use?**

| Data Source | Coordinate Space | Calibration to Use | Location |
|-------------|-----------------|-------------------|----------|
| Offline detections (YOLO, blob) | Camera (4512×4512) | `pixels_per_mm_camera` | `calibration` attrs |
| Keypoint positions | Camera (4512×4512) | `pixels_per_mm_camera` | `calibration` attrs |
| Bounding boxes | Camera (4512×4512) | `pixels_per_mm_camera` | `calibration` attrs |
| Online stimulus positions (chaser, target) | Texture (358×358) | `pixels_per_mm_projector` | `calibration` attrs |
| Refined online positions | Texture (358×358) | `pixels_per_mm_projector` | `refined_online_runs/<run>` attrs |
| Movement runs (online_refined) | Texture (358×358) | `pixels_per_mm_projector` | Already in `pixel_to_mm` attr |
| Movement runs (online, offline) | Camera (4512×4512) | `pixels_per_mm_camera` | Already in `pixel_to_mm` attr |

**Common Pitfalls**:
1. ❌ Don't scale texture positions to camera space and use camera calibration
2. ❌ Don't assume all pixel distances are equal (they're not!)
3. ✅ Always use the calibration matching your coordinate space
4. ✅ Check the `coordinate_space` attribute to know which space you're in

**Typical Values** (for reference):
- `pixels_per_mm_camera`: ~5.8 pixels/mm (camera space)
- `pixels_per_mm_projector`: ~0.44 pixels/mm (texture space)
- `texture_to_camera_scale`: ~12.6 (spatial scaling factor, NOT distance scaling!)

---

## Provenance & Access Tips

- Always inspect run attributes first — they encode the upstream run names,
  configuration, quality summaries, and time spent.
- **Check `coordinate_space` attribute** to determine which calibration to use.
- Array chunking follows the detection/ROI axis for efficient sequential
  reads.  Use `zarr.open_group(path, mode="r")` and slice natively.
- `fisheye.shared.zarr.schema.get_run_group(root, stage)` resolves the run
  path respecting `attrs["latest"]`.
- QA-sensitive tooling should filter using the stage-specific `reason` or
  metrics arrays instead of assuming all records are valid.
- **For distance calculations**: Always verify you're using the correct
  `pixels_per_mm` value for your coordinate space.

This document should remain in sync with the schema module and the stage
implementations.  When a new run group or array is added, update both places.
