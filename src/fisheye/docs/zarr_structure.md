# Palette Zarr Layout (v3)

This reference summarizes the structure produced by the modern Palette
pipeline. It is the **authoritative** human-readable spec.

Machine-readable per-stage array contracts (required/optional arrays, dtypes,
and shape templates) live in `fisheye.shared.zarr.stage_arrays` and should be
treated as the runtime-validation counterpart to this document.

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
- `subject_mask_runs/`
- `refined_detect_runs/`
- `refined_keypoints_runs/`
- `refined_eye_masks_runs/`
- `refined_subject_masks_runs/`
- `refined_online_runs/`
- `tracking_runs/`
- `arena_assignment_runs/`
- `calibration/`
- `analysis/`
- `analysis_metadata/`

Most `*_runs` groups carry:

- `attrs["latest"]` → most recent run name.
- Child run groups named `<stage>_<YYYY-MM-DD_hh-mm-ss>` (possibly with a
  `_NNN` suffix if repeated inside the same second).
- Run attributes capturing provenance (`provenance` dict with command,
  timestamps, git/environment snapshots), upstream run references
  (`source_detect_run`, `source_crop_run`, etc.), configuration snapshots,
  and often `duration_seconds`.

Lightweight stages such as `tracking_runs/` may omit some of the broader
timing-oriented attrs while still recording canonical lineage and summary
statistics.

---

## String/Text Encoding Conventions

String-like data should follow these conventions across runtime writers:

1. Reason/status labels that must be TensorStore/C++ compatible:
- Primary encoding: `reason_bytes` as `uint8[N,width]`, null-terminated UTF-8.
- Optional mirror: `reason` as variable-length UTF-8 text for Python ergonomics.
- Required attrs when reason bytes are present:
  - `reason_encoding="utf8-null-terminated"`
  - `reason_bytes_width=<int>`
  - `reason_bytes_null_terminated=true`
  - `reason_fallback_order=["reason_bytes","reason","detection_source"]`

2. General text columns/arrays:
- Canonical runtime encoding is variable-length UTF-8.
- Avoid new fixed-width Unicode writes (`<U...`) in runtime code paths.

3. Read compatibility:
- Readers should tolerate legacy fixed-width string arrays where they exist.
- Preferred read order for reason labels remains:
  `reason_bytes` -> `reason` -> labels derived from `detection_source`.

See also: `docs/zarr_string_encoding_todo.md`.

---

## `raw_video/`

Arrays written during import (kvikIO or standard path):

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `images_full` | `(n_frames, H, W)` | `uint8` | Full-resolution frames (optional) |
| `images_ds` | `(n_frames, H_ds, W_ds)` | `uint8` | Downsampled frames (optional) |
| `images_ds_rgb` | `(n_frames, H_ds, W_ds, 3)` | `uint8` | Downsampled RGB frames (optional) |
| `original_frame_indices` | `(n_import_frames,)` | `int32` | Present for sampled imports (`frame_step > 1`) |
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
| `bbox_norm_coords` | `(n_detections, 4)` | `float32` | Normalized `[cx, cy, w, h]` |
| `scores` | `(n_detections,)` | `float32` | Confidence scores |
| `class_ids` | `(n_detections,)` | `int32` | Detector class labels |
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

Each run stores crop-stage geometry/provenance and, in materialized mode, the
cropped ROI tensors needed by downstream consumers.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `roi_images` *(conditional)* | `(n_rois, h, w)` | `uint8` | Cropped grayscale patches. Present for `crop_storage_mode=materialized`; may be omitted for `geometry_only` runs. |
| `roi_coordinates_full` | `(n_rois, 2)` | `int32` | Top-left (x, y) in full-res pixels |
| `roi_coordinates_ds` | `(n_rois, 2)` | `int32` | Same offsets in downsampled space |
| `bbox_norm_coords` | `(n_rois, 4)` | `float32` | Normalized ROI bounding boxes (`[cx, cy, w, h]`) |
| `frame_indices` | `(n_rois,)` | `int32` | Frame index per ROI |
| `frame_counts` | `(n_frames,)` | `int32` | Count of ROIs per frame |
| `detection_source` | `(n_rois,)` | `int8` | 0 = real detection, 1 = interpolated (copied from refined runs) |
| `detection_indices` | `(n_rois,)` | `int32` | Index into source detect run |

Attributes:

- `source_detect_run`, `source_background_run`, `detection_source_type`,
  `detection_source_path`, `includes_interpolated`, `n_real_detections`,
  `n_interpolated_detections`, ROI size, scaling factors.
- `crop_storage_mode` declares whether the run is `materialized` or
  `geometry_only`.
- `detect_review_status` (snapshot of refined review status when crop ran)
- `detect_review_status_ref` (refined run path where review status lives)
- `detection_selection_policy` (policy label used for auto source selection)
- `crop_signature` (signature of crop inputs: source path/type, ROI size, parameters hash)
- `crop_review_status` (review status payload for this crop run, optional)
- `crop_review_signature` (signature snapshot stored when crop review was set)
- `summary_statistics` (frames with crops, total ROIs, percentage coverage).
- GPU/environment provenance.

Parent-group pointer semantics during mixed-mode migration:

- `crop_runs.attrs["latest"]` remains materialized-compatible for backward
  compatibility.
- `crop_runs.attrs["latest_materialized"]` tracks the latest materialized run.
- `crop_runs.attrs["latest_any"]` tracks the latest run regardless of storage
  mode.

Current policy note:

- Crop writers still default to materialized mode for both training and
  analysis archives.
- Training archives should reject geometry-only crop writes; canonical training
  crop runs are expected to persist `roi_images`.
- `geometry_only` runs are currently opt-in rather than the default.
- Many traditional/training/export consumers still require materialized
  `roi_images` even though mixed-mode readers now exist for some ROI-model
  workflows.

Cropping resolves the ROI source via `crop.source_type` (`detect`, `refined`,
`filtered`, `interpolated`, `manual`, `auto`) or an explicit
`crop.source_path` override such as `detect_runs/<run>` or the canonical
refined path `refined_detect_runs/<run>/instances`. Legacy sparse
subgroup overrides such as `refined_detect_runs/<run>/manual` remain
compatibility-only for historical archives. The chosen path is recorded in
`detection_source_path`.
`auto` resolves to the canonical curated refined surface when it exists, then
falls back to the legacy sparse chain only for historical archives.

---

## `keypoints_runs/`

Produced by the keypoint detection stage (traditional or YOLO-based).

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` | `(n_rois,)` | `int32` | Inherit from corresponding crops |
| `frame_counts` | `(n_frames,)` | `int32` | ROIs per frame (mirrors `n_rois`) |
| `n_rois` | `(n_frames,)` | `int32` | Alias maintained for legacy callers |
| `detection_indices` | `(n_rois,)` | `int32` | Index into `crop_runs/<run>/roi_images` |
| `keypoints_roi` | `(n_rois, n_keypoints, 2)` | `float64` | Coordinates in ROI pixels |
| `keypoints_img` | `(n_rois, n_keypoints, 2)` | `float64` | Full-image pixels |
| `keypoints_norm` | `(n_rois, n_keypoints, 2)` | `float64` | Normalized [0,1] |
| `heading` | `(n_rois,)` | `float64` | Degrees, NaN when unavailable |
| `confidence` | `(n_rois,)` | `float64` | Overall score |
| `keypoint_confidences` | `(n_rois, n_keypoints)` | `float64` | Per-keypoint confidences (swim_bladder, left, right) |
| `effective_threshold` | `(n_rois,)` | `float64` | Per-ROI threshold used |
| `effective_se2_radius` | `(n_rois,)` | `float64` | Search radius actually applied |
| `detection_success` | `(n_rois,)` | `bool` | True if keypoints converged |
| `detection_source` | `(n_rois,)` | `int8` | 0=real, 1=interpolated (from crop source) |
| `heading_finite` | `(n_rois,)` | `bool` | True when `heading` is finite |
| `heading_usable` | `(n_rois,)` | `bool` | True when source is real, detection succeeded, and heading is finite |
| `n_keypoints` | `(n_frames,)` | `int32` | Successful keypoints per frame |
| `triangle_angles` | `(n_rois, 3)` | `float64` | Triangle angles in canonical order (swim_bladder, left, right) |
| `triangle_angles_raw` | `(n_rois, 3)` | `float64` | Triangle angles in candidate order (largest -> smallest blob) |
| `triangle_area` | `(n_rois,)` | `float64` | Triangle area (pixels^2) |

Attributes: `source_crop_run`, `source_crop_storage_mode`,
`source_crop_signature`, `source_crop_revision`,
`source_detect_review_status_ref`, `source_background_run`,
`source_detect_run`, `source_refined_run` (if available), `method`,
`parameter_source`, `parameters`, `skeleton_id`, `kpt_shape`, `pose_schema`, `keypoint_labels`,
`keypoint_confidence_labels`, `triangle_angle_order`,
`triangle_angle_raw_order`, `heading_computation_override`,
scheduler configuration, timing, QA summaries.

Heading metadata note:

- `pose_schema.metadata.heading_computation` is the canonical heading
  definition for the skeleton.
- `heading_computation_override` is an optional run-level override/disable
  payload.
- Readers should resolve in this order:
  1. run attr `heading_computation_override`
  2. `pose_schema.metadata.heading_computation`
  3. deprecated run attr `heading_computation`
  4. heading semantics unavailable
- See `docs/keypoint_heading_computation_contract.md`.

---

## `detect_runs/<run>/quality_reports/<qrun>/`

Produced by `fisheye.refinement.detect_quality`.
This is a raw detect artifact-label surface used by `refine_detect`; it is not
the refined detect review/approval surface.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `quality_flags` | `(n_frames,)` | `int8` | Frame-level quality labels |
| `detection_quality_labels` | `(n_detections,)` | `int8` | Detection-level quality labels |

Attributes include thresholds/modes used for artifact detection, summary counts
of jumps/blips/gaps, and provenance references for the analyzed detect run.

---

## `eye_masks_runs/`

Generated by segmentation inference (`infer_unet_eye_masks.py`,
`eye_segmentation_yolo.py`, etc.).
Row lineage (`frame_indices`, `detection_indices`, `frame_counts`) follows:
`docs/eye_mask_row_mapping_contract.md`.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` *(legacy-optional)* | `(n_rois,)` | `int32` | ROI frame mapping (new runs should copy from source crop run; missing in some legacy runs) |
| `frame_counts` *(legacy-optional)* | `(n_frames,)` | `int32` | Count of ROIs per frame (new runs should match source crop run) |
| `detection_indices` *(legacy-optional)* | `(n_rois,)` | `int32` | Index into source detect rows via `crop_runs/<run>/detection_indices` |
| `masks_roi` *(optional)* | `(n_rois, C, H, W)` | `uint8` | Binary ROI-local masks. `C=1` for union, `C=2` for left/right. |
| `mask_probs_roi` *(optional)* | `(n_rois, C, H, W)` | `float16/float32/uint8` | Semantic ROI-local probabilities in `[0,1]`; `C=1` for union, `C=2` for left/right. Analysis U-Net runs default to quantized `uint8`; if stored as `uint8`, decode with `p = stored / 255`. |
| `mask_scores` *(optional)* | `(n_rois,)` | `float32` | Confidence per ROI |
| `ellipse_params` | `(n_rois, 2, 5)` | `float32` | `[cx, cy, major, minor, angle]` in ROI pixels |
| `ellipse_success` | `(n_rois, 2)` | `bool` | Ellipse fit success per eye |
| `eye_separation` | `(n_rois,)` | `float32` | Centroid distance (ROI pixels) |
| `detection_source` | `(n_rois,)` | `int8` | 0=real, 1=interpolated (copied from crop run) |
| `contour_left_ptr` | `(n_rois,)` | `int32` | Pointer into `contours_left` |
| `contour_left_len` | `(n_rois,)` | `int32` | Number of points for left eye contour |
| `contour_right_ptr` | `(n_rois,)` | `int32` | Pointer into `contours_right` |
| `contour_right_len` | `(n_rois,)` | `int32` | Number of points for right eye contour |
| `contours_left` | `(n_points, 2)` | `float32` | Concatenated left eye contours (x, y) |
| `contours_right` | `(n_points, 2)` | `float32` | Concatenated right eye contours (x, y) |
| `reason` | `(n_rois,)` | `string` | Per-ROI labels (`clean`, `keypoint_fail`, `no_region`, `overlap`, `too_close`, `too_far`, `incomplete`) |

Attributes: `source_crop_run`, `source_crop_storage_mode`,
`source_crop_signature`, `source_crop_revision`,
`source_detect_review_status_ref`, canonical `source_keypoints_run`
*(legacy alias: `source_keypoint_run` may be present for migration
compatibility)*, `source_keypoint_group` (defaults to
`refined_keypoints_runs` when present), `method`,
model info, thresholds, separation limits, `successful_eyes`,
`successful_roi_pairs`, `reason_counts`, `ellipse_angle_units`,
`ellipse_fit_backend` (currently `opencv` for refined runs),
`ellipse_fit_method` (currently `cv2.fitEllipse` for refined runs),
`ellipse_contour_mode` (currently `external_pixel_contour` for refined runs), `eye_labels`, `segmenter_label_mode`,
`mask_probs_chunk_rois`, `duration_seconds`,
optional threshold-calibration metadata:
- `recommended_probability_threshold`
- `recommended_probability_threshold_review`

Probability-storage attrs when `mask_probs_roi` exists:

- `probabilities_dtype`: physical storage dtype for `mask_probs_roi`
- `probabilities_encoding`:
  - `unit_float` for direct float probabilities
  - `linear_uint8_0_255` for quantized `uint8` probabilities

Reader contract:

- `mask_probs_roi` is ROI-local, not full-frame.
- `masks_roi` is ROI-local, not full-frame.
- Readers should interpret the dataset semantically as probabilities in
  `[0,1]` regardless of physical dtype.
- Full-frame placement should be derived from `source_crop_run` geometry.
- Analysis-oriented runs may use `segmenter_label_mode="union"` with a single
  channel and let refinement assign left/right identity later.

Lineage compatibility policy for eye-mask runs:

- New runs should treat `source_keypoints_run` as required canonical lineage.
- Legacy `source_keypoint_run` is read-compatible fallback only.
- Diagnostics/readers resolve canonical first, then legacy fallback.

---

## `refined_detect_runs/`

Created by `fisheye.refinement.refine_detect`.

Each refined run contains the canonical curated detect surface. Primary writes
land on the sparse refined subgroups. Shared readers should prefer the
`instances/` sparse surface when it exists.

### Run-level attributes

Common attrs on `refined_detect_runs/<run>`:
- `source_detect_run`, `source_quality_run`
- `refinement_timestamp`, `processing_time_seconds`
- `operations`
- `parameters`
- `coverage_frames_total` (frame universe used for coverage percent)
- `coverage_frame_source` (`full` or `sampled`)
- `coverage_frames_full` (full frame count when sampled coverage is used)
- `detect_review_status` (review metadata dict; see below)
- `summary_statistics`
- `curated_row_storage`
- `entity_assignment_policy`
- `coordinate_space`
- `row_identity_policy`
- status/source/review/artifact code maps
- provenance/environment metadata

Parent attrs on `refined_detect_runs/`:
- `latest`
- `detect_review_status_latest` (run name containing the most recent review status)

`detect_review_status` payload fields (may be extended over time):
- `state` (e.g., approved/needs_review)
- `method` (manual/retune/auto)
- `intended_use` (training/analysis/etc.)
- `timestamp`
- `resolved_group` (`refined` for current runs; legacy runs may still use manual/interpolated/filtered/raw)
- `preference_chain` (ordered list used for resolution)
- optional `reviewer`, `notes`

### Metadata-Only Run Root

The run root is now metadata-only for current sparse refined detect runs.
Canonical bbox data lives in the `instances/` and `source_detections/`
subgroups below.

Explicit downstream overrides should target `instances/` when they need a
stable current-run curated bbox path.

Current review/runtime note:

- `fisheye.tune.detect_review` edits the sparse refined surfaces directly
- when fixed sub-arena ROI definitions are present, review operates on one slot
  per `(frame, arena_id)`
- unconstrained multiple curated detections inside the same arena/ROI are not
  yet supported by the manual detect-review UI

### Sparse Curated Read Surfaces

When present, the active curated refined-detect surfaces are:

- `refined_detect_runs/<run>/instances`
- `refined_detect_runs/<run>/source_detections`

#### `instances/`

Primary curated bbox read surface for current runs.

Required arrays:

- `refined_row_ids`
- `frame_indices`
- `frame_offsets`
- `bbox_img_xyxy`
- `bbox_norm_coords`
- `source_kind_codes`
- `manual_edit_flags`

Common optional arrays:

- `confidence_scores`
- `class_ids`
- `source_detect_row_index`
- `reason_bytes`
- `reason`
- `review_notes`

Reader rule:

- rows in `instances/` are already the curated accepted detections; render only
  rows with finite bbox geometry

#### `source_detections/`

Candidate-audit surface mirroring the exact bound raw detect rowset for current
refined runs.

Required arrays:

- `source_detect_row_index`
- `frame_indices`
- `bbox_norm_coords`
- `decision_codes`
- `resolved_refined_row_id`

Common optional arrays:

- `bbox_img_xyxy`
- `confidence_scores`
- `class_ids`
- `reason_bytes`
- `reason`
- `review_notes`

Reader rule:

- treat `source_detections/` as an audit/provenance surface, not the primary
  bbox render surface

### Legacy Sparse Subgroups

Older archives may still contain sparse subgroups such as:

- `filtered`
- `interpolated`
- `manual`
- `manual_*`

These remain compatibility/provenance artifacts for legacy runs. They are no
longer the primary detect contract for new runs.

---

## `refined_keypoints_runs/`

Outputs from `fisheye.refinement.refine_keypoints`.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` | `(n_rois,)` | `int32` | Copied from source keypoint run |
| `frame_counts` | `(n_frames,)` | `int32` | Copied from source keypoint run |
| `n_rois` | `(n_frames,)` | `int32` | Alias maintained for legacy callers |
| `detection_indices` *(optional)* | `(n_rois,)` | `int32` | Copied from source when present |
| `detection_source` | `(n_rois,)` | `int8` | 0=real, 1=interpolated |
| `retune_id` | `(n_rois,)` | `int32` | Batch retune parameter set label (`-1` = none) |
| `keypoints_roi` | `(n_rois, 3, 2)` | `float64` | Refined keypoints (ROI pixels) |
| `keypoints_img` | `(n_rois, 3, 2)` | `float64` | Refined keypoints (full image) |
| `keypoints_norm` | `(n_rois, 3, 2)` | `float64` | Refined keypoints (normalized) |
| `heading` | `(n_rois,)` | `float64` | Heading after refinement |
| `confidence` | `(n_rois,)` | `float64` | Overall score (copied from source) |
| `keypoint_confidences` *(optional)* | `(n_rois, 3)` | `float64` | Per-keypoint confidences (copied, eyes swapped if flipped) |
| `effective_threshold` *(optional)* | `(n_rois,)` | `float64` | Copied from source if present |
| `effective_se2_radius` *(optional)* | `(n_rois,)` | `float64` | Copied from source if present |
| `triangle_area` | `(n_rois,)` | `float64` | Triangle area |
| `min_angle` | `(n_rois,)` | `float64` | Minimum triangle angle (deg) |
| `triangle_angles` | `(n_rois, 3)` | `float64` | Triangle angles (deg) in canonical order |
| `quality_labels` | `(n_rois,)` | `int8` | 0=clean, 4=source_failed, 6=flip_corrected |
| `refined_success` | `(n_rois,)` | `bool` | Refinement executed successfully |
| `source_success` | `(n_rois,)` | `bool` | Source keypoint success mask |
| `flip_corrected` | `(n_rois,)` | `bool` | True if left/right eyes were swapped |
| `heading_finite` | `(n_rois,)` | `bool` | True when `heading` is finite |
| `heading_usable` | `(n_rois,)` | `bool` | True when refined succeeded, source is real, and heading is finite |
| `heading_delta_prev_deg` *(optional)* | `(n_rois,)` | `float32` | Circular absolute heading delta to previous usable temporal neighbor; omitted when temporal-heading review is disabled |
| `heading_delta_next_deg` *(optional)* | `(n_rois,)` | `float32` | Circular absolute heading delta to next usable temporal neighbor; omitted when temporal-heading review is disabled |
| `heading_temporal_outlier` *(optional)* | `(n_rois,)` | `bool` | True when both temporal deltas exceed the configured outlier threshold; omitted when temporal-heading review is disabled |
| `confidence_valid` | `(n_rois,)` | `bool` | All per-keypoint confidences >= threshold |
| `geometry_valid` | `(n_rois,)` | `bool` | Triangle angle/area pass thresholds |
| `usable_keypoints` | `(n_rois,)` | `bool` | Confidence + geometry valid |
| `reason_bytes` | `(n_rois, width)` | `uint8` | Null-terminated UTF-8 reason labels (TensorStore-safe primary encoding) |
| `reason` | `(n_rois,)` | `string` | Pipe-delimited tags (e.g., `flip_corrected|geometry_issue`) |
| `failure_indices` | `(n_failures,)` | `int32` | ROI indices where source keypoints failed |

Attributes: `source_keypoints_run`, `source_crop_run`,
`source_crop_storage_mode`, `source_crop_signature`,
`source_crop_revision`, `source_detect_review_status_ref`,
`source_detect_run`, `skeleton_id`, `kpt_shape`, `pose_schema`,
`heading_computation_override`, `derived_metrics_schema`,
refinement parameters (thresholds),
`summary_statistics`, `retune_params`, `keypoint_signature`,
`keypoint_review_status`, `keypoint_review_signature`, scheduler config,
environment/provenance metadata.

Heading metadata note:

- `pose_schema.metadata.heading_computation` is the canonical heading
  definition for the skeleton.
- `heading_computation_override` is an optional run-level override/disable
  payload for exceptional cases.
- Readers should resolve in this order:
  1. run attr `heading_computation_override`
  2. `pose_schema.metadata.heading_computation`
  3. deprecated run attr `heading_computation`
  4. unavailable
- If the override payload exists and `enabled=false`, that disables heading
  semantics for the run even if `pose_schema.metadata.heading_computation`
  exists.
- See `docs/keypoint_heading_computation_contract.md`.

Derived-metrics metadata note:

- `derived_metrics_schema` is a run-level semantic contract for derived arrays
  and boolean/status gates.
- For current refined keypoint runs, it declares the triangle geometry metric
  semantics behind `triangle_area`, `triangle_angles`, `min_angle`, and
  `geometry_valid`.
- It is separate from the entity schema (`pose_schema`, `keypoint_labels`) and
  separate from heading semantics.
- See `docs/derived_metrics_schema_contract.md`.

Temporal-heading policy notes:

- Temporal-heading review is intended for temporally contiguous refined runs.
- Sampled imports (for example `raw_video/import_mode="sampled"` or archives
  with `original_frame_indices`) disable temporal-heading review.
- When disabled, the optional temporal arrays above are omitted and
  `summary_statistics.postprocess` records:
  - `temporal_heading_status`
  - `temporal_heading_disabled_reason`

Post-refinement coordinate-space diagnostics attrs (when audit is enabled):

- `post_refinement_audit_json`: absolute path to `<dataset>_audit.json`
- `post_refinement_audit_generated_utc`: UTC timestamp of audit generation
- `post_refinement_audit_status_counts`: status-count snapshot from the audit report

Optional overlap-analysis attrs (when overlap analysis is enabled):

- `post_refinement_overlap_json`: absolute path to `<dataset>_overlap.json`
- `post_refinement_overlap_generated_utc`: UTC timestamp of overlap report generation
- `post_refinement_overlap_bad_row_count`: bad-row count from overlap analysis

Reason-label attrs on refined keypoint runs:

- `reason_encoding="utf8-null-terminated"`
- `reason_bytes_width=<int>`
- `reason_bytes_null_terminated=true`
- `reason_fallback_order=["reason_bytes","reason","detection_source"]`

Consumers should read reason labels in this order:
`reason_bytes` -> `reason` -> labels derived from `detection_source`
(`0=clean`, `1=interpolated`).

`summary_statistics` is a dict with a refine-time snapshot plus optional
postprocess counts, e.g.:

- `summary_statistics.refine`: counts written by `refine_keypoints`
- `summary_statistics.postprocess`: counts recomputed after retune/manual review
- `summary_statistics.postprocess_updated_utc`: timestamp of the last audit

`retune_params` maps `retune_id` values to the keypoint-tuning parameter sets
applied during batch retuning.

---

## `refined_eye_masks_runs/`

See `fisheye.refinement.refine_eye_masks`.  Key arrays:
Row lineage (`frame_indices`, `detection_indices`, `frame_counts`) is copied
from `source_eye_masks_run`; see `docs/eye_mask_row_mapping_contract.md`.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `masks_roi` | `(n_rois, 2, H, W)` | `uint8` | Refined masks |
| `ellipse_params` | `(n_rois, 2, 5)` | `float32` | `[cx, cy, major, minor, angle]` |
| `ellipse_success` | `(n_rois, 2)` | `bool` | Fit success per eye |
| `eye_separation` | `(n_rois,)` | `float32` | Centroid distance |
| `retune_id` *(optional)* | `(n_rois,)` | `int32` | Batch retune parameter set label (`-1` = none) |
| `frame_indices` *(legacy-optional)* | `(n_rois,)` | `int32` | Copied from source eye-mask run (new runs should include) |
| `frame_counts` *(legacy-optional)* | `(n_frames,)` | `int32` | Copied from source eye-mask run (new runs should include) |
| `detection_indices` *(legacy-optional)* | `(n_rois,)` | `int32` | Copied from source eye-mask run (new runs should include when upstream exists) |
| `mask_probs_roi_refined` *(optional)* | `(n_rois, 2, H, W)` | `float16` | Optional refined ROI-local left/right probabilities in `[0,1]`; new refined runs are binary-first and only write this dataset when explicitly requested. |
| `contour_left_ptr` | `(n_rois,)` | `int32` | Pointer into `contours_left` |
| `contour_left_len` | `(n_rois,)` | `int32` | Number of points for left eye contour |
| `contour_right_ptr` | `(n_rois,)` | `int32` | Pointer into `contours_right` |
| `contour_right_len` | `(n_rois,)` | `int32` | Number of points for right eye contour |
| `contours_left` | `(n_points, 2)` | `float32` | Concatenated left eye contours (x, y) |
| `contours_right` | `(n_points, 2)` | `float32` | Concatenated right eye contours (x, y) |

`metrics/` subgroup:

- Scalar QA arrays such as `area_refined`, `area_delta_vs_source`,
  `centroid_error`, `symmetry_offsets`, `separation_delta`, `axis_ratio`,
  `circularity`, `probability_*`, `filter_flags`, `connectivity_flags`,
  `smoothing_flags`, `pixels_reassigned`, `reason` (tags include
  `refined`, `copied_original`, `filtered_*`, `retuned`, `manual_correction`).

Attributes expose `metrics_summary`, configuration snapshots, per-eye filter
thresholds, `summary_statistics`, `retune_params`, and links to source runs
(`source_eye_masks_run`, `source_eye_masks_method`, `source_keypoint_group`,
canonical `source_keypoints_run` with optional legacy alias
`source_keypoint_run`, `source_crop_run`).
`traditional_fast_path=true` indicates masks/ellipses were copied from the
source (used for traditional segmentation unless
`force_refine_traditional=true`).

Important refined-output policy:

- `masks_roi` is the canonical refined artifact.
- `mask_probs_roi_refined` is optional debug/high-fidelity output.
- Refined runs record the threshold used to derive binary masks from
  probability inputs via `mask_probability_threshold`.
- Refined runs also record `mask_probability_threshold_source` describing
  whether that threshold came from:
  - explicit CLI input
  - source-run recommendation metadata
  - the default fallback

Semantic note:

- Refined eye-mask outputs remain ROI-local.
- A refined run may promote a raw union-mask source into explicit left/right
  channels using keypoint-informed refinement.

Refinement keypoint-source binding is strict by default:

- Resolution order is `--keypoint-run` -> source lineage attrs on source eye-mask run
  (`source_keypoint_group` + canonical/legacy keypoint-run attr) -> error.
- `--allow-latest-keypoint-fallback` enables legacy compatibility fallback and
  should be treated as temporary recovery behavior.

`summary_statistics` mirrors refined keypoints: the `refine` snapshot is written
by `refine_eye_masks`, and `postprocess` is updated by the review tooling
(`eye_mask_review --retune/--manual/--audit`). The postprocess stats include
manual correction counts, retune totals, and reason tag counts. `retune_params`
maps `retune_id` values to the parameter sets applied during batch retuning.

---

## `subject_mask_runs/`

Unified subject-mask runs produced by the traditional body segmenter, U-Net,
SAM/SAM2/SAM3 write-back, and swim-bladder refresh flows.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` *(recommended)* | `(n_rois,)` | `int32` | Copied from the source crop run when available |
| `frame_counts` *(recommended)* | `(n_frames,)` | `int32` | Copied from the source crop run when available |
| `detection_indices` *(recommended)* | `(n_rois,)` | `int32` | Copied from the source crop run when available |
| `detection_source` | `(n_rois,)` | `int8` | Expected to align with the source crop run |
| `masks_roi` | `(n_rois, C, H, W)` | `uint8` | Canonical binary multilabel masks |
| `mask_probs_roi` | `(n_rois, C, H, W)` | `float16/float32/uint8` | Decoded or quantized semantic probabilities in `[0,1]` |
| `available_channels` | `(C,)` | `bool` | Run-level declaration of which channels are semantically valid |

`metrics/` subgroup:

- required: `prob_max`, `mask_present`
- recommended: `area_px`, `centroid_xy`, `centroid_valid`, `bbox_xyxy`,
  `bbox_valid`

Important attrs:

- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `source_detect_review_status_ref` *(when the crop source exposes detect-review linkage)*
- `label_schema_id`
- `mask_labels`
- `output_semantics="multilabel"`
- `overlap_policy="independent_sigmoid"`
- `method`
- `run_semantics`
- `probabilities_dtype`
- `probabilities_encoding`
- `summary_statistics`

Component-local lineage lives under `components/<component>/provenance`.
Those provenance payloads mirror the canonical crop snapshot fields so merged
and refined subject-mask stages can preserve the exact crop surface they
consumed.

---

## `refined_subject_masks_runs/`

Canonical refined/editable subject-mask runs produced by
`fisheye.tune.refined_subject_mask_review` and
`fisheye.refinement.assemble_refined_subject_masks`.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` *(recommended)* | `(n_rois,)` | `int32` | Copied from the upstream subject-mask/crop lineage when available |
| `frame_counts` *(recommended)* | `(n_frames,)` | `int32` | Copied from the upstream subject-mask/crop lineage when available |
| `detection_indices` *(recommended)* | `(n_rois,)` | `int32` | Copied from the upstream subject-mask/crop lineage when available |
| `detection_source` | `(n_rois,)` | `int8` | Expected to align with the source crop run |
| `masks_roi` | `(n_rois, C, H, W)` | `uint8` | Canonical refined binary masks |
| `available_channels` | `(C,)` | `bool` | Declares which refined components are intentionally present |
| `edit_applied` | `(n_rois, C)` | `bool` | True when the refined channel differs from the source subject-mask run |
| `reason_bytes` *(optional)* | `(n_rois, width)` | `uint8` | Null-terminated UTF-8 reason labels |
| `reason` *(optional)* | `(n_rois,)` | `string` | Human-readable reason tags |

`metrics/` subgroup:

- required: `mask_present`, `area_px`
- recommended: `centroid_xy`, `centroid_valid`, `bbox_xyxy`, `bbox_valid`

Important attrs:

- `source_subject_mask_run`
- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `source_detect_review_status_ref` *(when the crop source exposes detect-review linkage)*
- `label_schema_id`
- `mask_labels`
- `output_semantics="multilabel"`
- `refinement_semantics="canonical_component_masks"`
- `method`
- `refined_subject_mask_review_status`
- `component_review_statuses`
- `summary_statistics`
- `component_summary_statistics`

Refined subject-mask runs preserve the same portable crop snapshot contract as
their upstream `subject_mask_runs/<run>` source rather than re-deriving lineage
from the latest crop run later. Component-local lineage continues to live under
`components/<component>/provenance`.

---

## `arena_assignment_runs/`

Generated by `fisheye.tracking.arena_assignment` and friends.

Common arrays:

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `arena_ids` | `(n_detections,)` | `int32` | Assigned arena/ROI IDs |
| `n_detections_per_arena` | `(n_frames, n_arenas)` | `int32` | Per-frame counts for each arena |

Run attrs include the assignment source, arena definitions, and summary counts.

Attributes describe the assignment strategy, arena definitions used,
expected counts, and QA tallies (`assigned`, `unassigned`).

---

## `tracking_runs/`

Generated by `fisheye.tracking.single_subject_per_arena` through
`fisheye.tracking.arena_assignment` for the current one-subject-per-arena
workflow.

Common arrays:

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `track_ids` | `(n_detections,)` | `int32` | Run-local track ID per source row. `-1` means unassigned. |
| `arena_ids` | `(n_detections,)` | `int32` | Arena assignment mirrored from the bound arena-assignment run. |
| `frame_indices` | `(n_detections,)` | `int32` | Source frame index per row for auditing/debugging. |
| `source_row_indices` | `(n_detections,)` | `int32` | `0..n_rows-1` index into the exact tracked rowset. |
| `track_ids_present` | `(n_tracks,)` | `int32` | Sorted list of emitted real track IDs. |
| `track_arena_ids` | `(n_tracks,)` | `int32` | Arena ID for each emitted track. Parallel to `track_ids_present`. |

Important attrs:

- `tracking_method`: currently `single_subject_per_arena`
- `source_detect_run`
- `source_refined_run` when applicable
- `source_arena_assignment_run`
- `source_rowset_path`
- `track_namespace`: currently `local_per_run`
- `unassigned_track_id`: currently `-1`
- `conflict_policy`: currently `fail`
- `num_tracks`
- `n_assigned_rows`
- `n_unassigned_rows`
- `unassigned_row_rate_percent`
- `tracking_qc_state`: current runtime values are `ok` or `warn`
- `tracking_warn_threshold_rows`, `tracking_warn_threshold_percent`
- `tracking_block_threshold_rows`, `tracking_block_threshold_percent`
- `summary_statistics`

Unassigned rows remain explicit in `tracking_runs/` as `track_id == -1` for QA
and provenance. They are excluded from public `analysis/track_kinematics_runs`
outputs by default unless the diagnostic `--include-unassigned` path is used.

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
| `measured_stimulus_fps` | Measured stimulus frame rate | fps | Computed from `/video_metadata/frame_metadata` timestamps |
| `measured_fps` | Legacy alias for measured_stimulus_fps | fps | Maintained for backward compatibility |
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
- `source_stimulus_video_path`: Path to rendered stimulus video next to source H5 (when present; analysis stimulus runs only)
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
- `measured_stimulus_fps`: Measured stimulus frame rate (from H5 frame metadata timestamps)
- `measured_fps`: Legacy alias for `measured_stimulus_fps`
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

### `analysis/track_kinematics_runs/`

Track kinematics results organized by type:

**Structure**: `analysis/track_kinematics_runs/<online|offline>/<run_name>/`

**Run Attributes**:
- `method`: Analysis method used
  - `track_kinematics_online`: Raw online data (transformed to camera space)
  - `track_kinematics_online_refined`: Refined online data (texture space)
  - `track_kinematics_offline`: Offline detection data
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
  - Offline: `detection_run`, `keypoint_run`, `source_tracking_run`, `source_arena_assignment_run`, optional `chaser_metrics` dict (metrics run, stimulus run, chaser index)
- Offline runs also persist `source_tracking_run` and
  `source_arena_assignment_run` as top-level attrs for direct lineage lookup.
- `summary`: Per-track summary statistics
- `total_distance_px`, `total_distance_mm`: Aggregate distances

**Shared Root Arrays (offline runs only)**:
- `camera_frame_ids` (`int64`): Master frame index aligned to all chaser metrics.
- `stimulus_frame_nums` (`int64`), `timestamp_ns` (`int64`), `trial_state` (`int16`): Optional context per frame.
- `metadata_mask` (`bool`, optional): Propagated interpolation/original mask when available.
- `has_offline` (`bool`): Indicates frames with valid chaser metrics.
- `distance_to_target_px`, `distance_to_target_mm` (`float32`): Chaser→target separation per frame.
- `distance_to_target_interpolated_px`, `distance_to_target_interpolated_mm` (`float32`, optional): Raw distances with short NaN gaps (duration ≤ `distance_interpolation_seconds` × FPS) filled via linear interpolation.
- `distance_to_target_smoothed_px`, `distance_to_target_smoothed_mm` (`float32`, optional): Moving-average smoothing applied to the interpolated series using the track-kinematics run's `fps` and `smoothing_seconds`.
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
- `heading_degrees`, `heading_radians`, `delta_heading_degrees`, `angular_velocity_deg_s`, `smoothed_heading_degrees`, `smoothed_heading_radians`
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

### `analysis/stimulus_response_runs/`

Per-step behavioral metrics across stimulus types. Consumes identity-resolved
track data from `track_kinematics_runs` and stimulus metadata from
`stimulus_runs`. See `docs/stimulus_response_run_design.md` for full metric
definitions and `docs/stimulus_response_implementation_plan.md` for design
decisions.

**Structure**: `analysis/stimulus_response_runs/<run_name>/`

**Run Attributes**:
- `provenance`: Stage provenance contract (`palette_stage_provenance`)
- `source_track_kinematics_run`: Source kinematics run name
- `source_track_kinematics_type`: `"online"` or `"offline"`
- `source_stimulus_run`: Source stimulus run name
- `source_bout_run`: Source bout run name (optional, present when bout data used)
- `n_steps`, `n_fish`, `fish_ids`: Recording summary

**`global/`** — recording-wide per-fish movement summary:

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `fish_id` | `(n_fish,)` | `int32` | Track IDs |
| `total_distance_mm` | `(n_fish,)` | `float32` | |
| `mean_speed_mm_s` | `(n_fish,)` | `float32` | |
| `total_active_s` | `(n_fish,)` | `float32` | Time above moving threshold |
| `fraction_moving` | `(n_fish,)` | `float32` | |

**`steps/step_{i}/`** — one group per protocol step:

Step attributes: `step_index`, `step_name`, `stimulus_mode`, `stimulus_mode_id`,
`start_frame`, `end_frame`, `duration_s`, `stimulus_params`.

**`steps/step_{i}/per_fish/`** — base movement metrics (all stimulus types):

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `fish_id` | `(n_fish,)` | `int32` | |
| `total_distance_mm` | `(n_fish,)` | `float32` | |
| `mean_speed_mm_s` | `(n_fish,)` | `float32` | |
| `median_speed_mm_s` | `(n_fish,)` | `float32` | |
| `max_speed_mm_s` | `(n_fish,)` | `float32` | |
| `fraction_moving` | `(n_fish,)` | `float32` | |
| `coverage` | `(n_fish,)` | `float32` | Fraction of step frames with valid detection |
| `num_bouts` | `(n_fish,)` | `int32` | Optional, present when bout data available |
| `mean_bout_duration_s` | `(n_fish,)` | `float32` | Optional |
| `mean_interbout_interval_s` | `(n_fish,)` | `float32` | Optional |

**`steps/step_{i}/per_bout/`** — bout-level metrics (optional):

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `fish_id` | `(n_bouts,)` | `int32` | |
| `bout_id` | `(n_bouts,)` | `int32` | |
| `start_frame` | `(n_bouts,)` | `int64` | |
| `end_frame` | `(n_bouts,)` | `int64` | |
| `duration_s` | `(n_bouts,)` | `float32` | |
| `mean_speed_mm_s` | `(n_bouts,)` | `float32` | |
| `peak_speed_mm_s` | `(n_bouts,)` | `float32` | |

**`steps/step_{i}/grating/`** — MOVING_GRATING steps only:

This group only exists when `stimulus_mode == "MOVING_GRATING"`.

`grating/per_frame/`:

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `frame_indices` | `(n_step_frames,)` | `int64` | |
| `alignment_angle_deg` | `(n_fish, n_step_frames)` | `float32` | 0 = following, ±180 = opposing |
| `alignment_cos` | `(n_fish, n_step_frames)` | `float32` | +1 = following, -1 = opposing |
| `speed_along_grating_mm_s` | `(n_fish, n_step_frames)` | `float32` | Speed projected onto grating direction |
| `angular_velocity_deg_s` | `(n_fish, n_step_frames)` | `float32` | |

`grating/per_fish/`:

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `mean_alignment_cos` | `(n_fish,)` | `float32` | |
| `resultant_vector_length` | `(n_fish,)` | `float32` | Circular mean resultant (0=random, 1=consistent) |
| `fraction_following` | `(n_fish,)` | `float32` | |
| `fraction_opposing` | `(n_fish,)` | `float32` | |
| `fraction_perpendicular` | `(n_fish,)` | `float32` | |
| `speed_weighted_alignment` | `(n_fish,)` | `float32` | |
| `optomotor_gain` | `(n_fish,)` | `float32` | mean(speed_along) / grating_speed |
| `drift_along_grating_mm` | `(n_fish,)` | `float32` | Net displacement in grating direction |
| `drift_perp_grating_mm` | `(n_fish,)` | `float32` | Net displacement perpendicular to grating |
| `latency_to_follow_s` | `(n_fish,)` | `float32` | Time to first sustained following (NaN if never) |

`grating/time_series/`:

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `bin_center_s` | `(n_bins,)` | `float32` | |
| `alignment_cos` | `(n_fish, n_bins)` | `float32` | |
| `speed_mm_s` | `(n_fish, n_bins)` | `float32` | |
| `fraction_following` | `(n_fish, n_bins)` | `float32` | |
| `optomotor_gain` | `(n_fish, n_bins)` | `float32` | |

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
- `attrs["keypoint_tuning"]` – parameters saved from the keypoint tuner
  (`roi_thresh`, `se1_radius`, `se2_radius`, `min_area`, `min_valid_angle`,
  `max_valid_angle`, `min_triangle_area`, plus `tuned_on_frame` and
  `tuned_on_detection`).
- `attrs["eye_mask_tuning"]` – parameters saved from the eye mask tuner:
  - `method`: `"global_threshold_otsu"`
  - `version`: Schema version (e.g., `"1.0"`)
  - `tuned_timestamp`: ISO 8601 UTC timestamp
  - `tuned_parameters`:
    - `roi_padding`: Padding around keypoint (int)
    - `pre_threshold`: Pre-threshold cutoff (int or null)
    - `sobel_strength`: Edge subtraction strength 0–1 (float)
    - `min_area`, `max_area`: Region area bounds (int, max may be null)
    - `closing_radius`, `opening_radius`: Morphological radii (int)
    - `min_eye_separation`, `max_eye_separation`: Eye gap bounds (float or null)
  - `context`: ROI index, crop/keypoint runs used, success flag
  - Note: this metadata is expected for traditional eye-mask parameter tuning; YOLO/U-Net model-based runs (including `_analysis.zarr`) may not include it.
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
- QA-sensitive tooling should filter using stage-specific reason/metrics arrays.
  For refined detect/keypoint groups, prefer `reason_bytes`, then `reason`,
  then `detection_source` as fallback.
- **For distance calculations**: Always verify you're using the correct
  `pixels_per_mm` value for your coordinate space.

This document should remain in sync with the schema module and the stage
implementations.  When a new run group or array is added, update both places.
