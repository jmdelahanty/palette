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
- `id_assignment_runs/`
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
| `feret_axes_major/minor` | `(n_rois, 2, 4)` | `float32` | Endpoints for Feret diameters |
| `feret_roundness` | `(n_rois, 2)` | `float32` | Major/minor ratio |
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

## `analysis/`

Organized by analyzer:

- `movement_runs/`: per-fish tracks (`tracks/id_<id>/positions_px`,
  `speed_mm_s`, `heading_deg`, etc.), summary attributes (`track_manifest`,
  distance totals, smoothing config).
- `stimulus_runs/`: imports stimulus metadata (see docstring in
  `fisheye.analysis.import_stimulus_to_zarr`).  Stores source H5 metadata,
  frame alignment arrays, variable-length UTF-8 fields, interpolation masks.
- `eye_angle_runs/`: per-ROI and per-frame eye-angle metrics, QA masks,
  `reason_codes`.
- Additional analyzers (e.g., heatmaps, swim-bout stats) follow the same
  `<analysis>_runs/<run_name>/` pattern with analyzer-specific arrays and
  provenance attributes.

---

## `analysis_metadata/`

Lightweight store for metadata generated during tuning or diagnostics.
Examples:

- `attrs["dish_mask"]` – saved circle parameters from the mask tuner
  (center, radius, Hough params).
- `attrs["subdish_mask_tuning"]` – multi-dish ROI definitions.
- Other agents may add read-only metadata blocks here.

---

## Provenance & Access Tips

- Always inspect run attributes first — they encode the upstream run names,
  configuration, quality summaries, and time spent.
- Array chunking follows the detection/ROI axis for efficient sequential
  reads.  Use `zarr.open_group(path, mode="r")` and slice natively.
- `fisheye.shared.zarr.schema.get_run_group(root, stage)` resolves the run
  path respecting `attrs["latest"]`.
- QA-sensitive tooling should filter using the stage-specific `reason` or
  metrics arrays instead of assuming all records are valid.

This document should remain in sync with the schema module and the stage
implementations.  When a new run group or array is added, update both places.
