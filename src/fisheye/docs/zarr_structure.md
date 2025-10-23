# Palette Zarr Layout

This note gives agents a fast reference for what lives inside a Palette `.zarr`
archive and how we name the major run artifacts.  The goal is to make it easy to
read data without grepping through the pipeline code.

## Root Group

* **Attributes**
  * `schema_version`, `zarr_format` – versioning from `fisheye.shared.zarr.schema`
  * `created_at`, `command_line_args`, `git_info`, `platform_info`
  * `source_video_metadata` – width, height, fps, frame count, etc.
  * `processing_history` – ordered list of stage names (optional)
* **Children (top level)**
  * `raw_video/` – original frames or downsampled versions (optional)
  * `crop_runs/`
  * `detection_runs/`
  * `keypoints_runs/`
  * `eye_masks_runs/`
  * `refined_eye_masks_runs/`
  * `refined_keypoints_runs/` (created by the keypoint refinement stage)
  * `refined_detect_runs/` (if the detection refinement stage is used)
  * `analysis/` – optional downstream exports
  * `metadata/` – experiment configuration, ROI definitions, stimulus events

All `*_runs` groups follow the same pattern:

* An attribute `latest` pointing at the most recent run name.
* Each child run (`<stage>_<YYYY-MM-DD_hh-mm-ss>`) stores stage-specific arrays.
* Run attributes capture provenance: `method`, `command`, upstream run names,
  `duration_seconds`, and a rich `provenance` dict (git hash, environment info,
  parameter snapshot).

## Key Stage Groups

### `crop_runs/`

Per run arrays:

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `crops` | `(n_frames, h, w)` | `uint8` | Optional raw crops |
| `bbox` | `(n_frames, 4)` | `float32` | XYWH in source coordinates |

Attributes record the detector that generated the crop boxes.

### `detection_runs/`

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `n_detections` | `(n_frames,)` | `int32` | Count per frame |
| `bboxes` | `(n_total, 4)` | `float32` | Normalized `[x, y, w, h]` |
| `scores` | `(n_total,)` | `float32` | Confidence |
| `class_ids` | `(n_total,)` | `int32` | Optional class labels |

Latest YOLO runs will set `attrs["method"] = "yolo_detect"` and record model
names, checkpoints, thresholds, etc.

### `keypoints_runs/`

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `keypoints_roi` | `(n_rois, n_keypoints, 2)` | `float32` | ROI (crop) coordinates |
| `keypoints_world` | `(n_rois, n_keypoints, 2)` | `float32` | Optional global coords |
| `heading` | `(n_rois,)` | `float32` | Heading in degrees |
| `detection_success` | `(n_rois,)` | `bool` | Keypoint pipeline success flag |
| `keypoint_scores` | `(n_rois, n_keypoints)` | `float32` | Confidence per landmark |

Attributes include `skeleton`, `source_detection_run`, training checkpoint info,
and thresholds.

### `eye_masks_runs/`

Runs produced by the segmentation stage (`infer_unet_eye_masks.py` or YOLO mask
variants).

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `masks_roi` | `(n_rois, 2, H, W)` | `uint8` | Left/right channels (0/1 mask) |
| `mask_probs_roi` | `(n_rois, 2, H, W)` | `float16/float32` | Probabilities |
| `mask_probs_roi_refined` | optional | | Post-processed probabilities |

Attributes specify `method`, model checkpoint, crop/keypoint sources, and per-run
configuration (thresholds, smoothing).

### `refined_eye_masks_runs/`

Created by `fisheye.refinement.refine_eye_masks`.  In addition to the refined
masks the run stores extensive QA metrics so downstream tooling can triage bad
eyes.

**Primary arrays**

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `masks_roi` | `(n_rois, 2, H, W)` | `uint8` | Refined left/right masks |
| `mask_probs_roi_refined` | `(n_rois, 2, H, W)` | `float16` | Optional union probs |
| `ellipse_params` | `(n_rois, 2, 5)` | `float32` | `[cx, cy, major, minor, angle]` |
| `ellipse_success` | `(n_rois, 2)` | `bool` | Fit success flags |
| `feret_axes_major` / `_minor` | `(n_rois, 2, 4)` | `float32` | Flattened endpoints |
| `feret_roundness` | `(n_rois, 2)` | `float32` | Major/minor ratio measure |
| `eye_separation` | `(n_rois,)` | `float32` | Distance between fitted centroids |
| `reason` | `(n_rois,)` | `string` | Pipe-delimited tags (`refined`, `keypoint_fail`, `filtered_left`, …) |

**Metrics subgroup**

`metrics/` contains per-ROI QA arrays (all `float32` unless noted):

* `area_refined`, `area_source`, `area_delta_vs_source`, `area_zscore`
* `centroid_error`, `symmetry_offsets`, `symmetry_sum`, `symmetry_abs_diff`
* `separation_refined`, `separation_keypoint`, `separation_delta`
* `axis_ratio`, `circularity`
* `connectivity_flags` (`uint8` bitmask: smoothing used, components reassigned, probabilities used)
* `smoothing_flags` (`uint8`, per eye)
* `pixels_reassigned` (`int32`)
* `probabilities_used` (`bool`)
* `probability_mean`, `_max`, `_var`, `_high_fraction`
* `filter_flags` (`bool`, shape `(n_rois, 2)`): per-eye area outlier status

Run attributes expose a summary in `metrics_summary` (means, std-dev, filter
counts) and the configuration used (`mask_parameters["area_filter"]` contains the
z-score threshold and mode).

### `refined_keypoints_runs/` & `refined_detect_runs/`

Follow the same pattern: copies of the source arrays plus stage-specific QA
metrics and `reason` labels. Always check the run attributes (`refine_stats`,
`metrics_summary`, upstream run references) before consuming arrays.

## Analysis Runs

### `analysis/movement_runs/`

Each movement run stores tracks grouped by ID:

* `tracks/id_<track_id>/` arrays contain frame indices, timestamps, positions
  (px/mm), headings, instantaneous & smoothed speeds, accelerations, per-frame
  distances, and per-second aggregates.
* `track_manifest` (run attribute) summarises speed/distance/heading metrics.
* Run attributes document inputs, smoothing parameters, calibration info, and
  global totals.

### `analysis/stimulus_runs/`

Created by `fisheye.analysis.import_stimulus_to_zarr`. This replaces the legacy
`analysis.h5` workflow by importing the stimulus metadata directly into the
archive.

```
analysis/
  stimulus_runs/
    attrs:
      latest -> <run_name>
    <run_name>/
      attrs:
        created_at_utc
        source_h5 (path to original stimulus file)
        import_version
        arena_config_json (raw calibration snapshot)
        coordinate_transform (JSON with texture/camera dims, scale)
        protocol_json (optional)
        interpolation statistics (original/interpolated frames, gap info)
      video_metadata/
        frame_metadata/<field>  # one array per structured-field (numbers or UTF-8 strings)
      interpolation_mask        # bool array marking original vs interpolated rows
      frame_alignment/
        camera_frame_offset (attribute)
        camera_to_metadata_index
        camera_interpolation_mask
      tracking_data/
        chaser_states/<field>
        bounding_boxes/<field>
      events/<field>
```

Notes:

* Structured stimulus datasets (metadata, chaser states, bounding boxes,
  events) are expanded into subgroups with one array per field to avoid
  unsupported structured dtypes in Zarr v3. String fields use
  `VariableLengthUTF8()` so they round-trip as Python strings.
* `interpolation_mask` flags which metadata rows came from the original H5
  (`True`) vs. synthesized during interpolation (`False`).
* The frame-alignment datasets let tooling map between camera frames (detection
  timeline) and stimulus frames without recomputing the lookup.

Downstream analyzers (movement, training heatmaps, chaser plots) can now depend
entirely on the Zarr archive—no separate analysis H5 is required.

## Metadata and Provenance Conventions

* Every stage writes a `provenance` dict with:
  * `stage`, `command`, `created_at_utc`
  * `git` (`commit`, `branch`, `is_dirty`)
  * `environment` (host, python, GPU availability)
  * `scheduler` (for Dask jobs)
  * `inputs` (names of upstream runs)
* Long-running stages also store:
  * `metrics_summary` – high-level QA aggregates
  * `refine_stats` – raw counters (refined vs. copied, filtered counts, etc.)

When reading a run, always inspect its attributes first for context and to find
linked datasets.  The `reason` array plus the metrics group flag the cases that
were copied, filtered, or otherwise atypical.

## Access Tips for Agents

* Use `zarr.open(<path>, mode="r")` and index with native slices—the archive is
  chunked along ROIs or time for efficient sequential access.
* `fisheye.shared.zarr.schema.get_run_group(root, stage_name)` returns the run
  group and the selected run name (respecting `attrs["latest"]`).
* Inspect run attributes first to understand provenance, configuration, and QA
  summaries before consuming arrays.
* For QA-sensitive work: filter using stage-specific flags (`reason`, metrics
  masks) before aggregating results.
* When stimulus data is needed, access `analysis/stimulus_runs/<run>/...` rather
  than maintaining a separate `analysis.h5`.
    `empty_union`.
* New analysis scripts should read from the refined groups whenever possible;
  the raw `_runs` groups remain immutable inputs from earlier pipeline stages.

This document should be kept in sync with the pipeline when new stages or
metrics are added.  When introducing a new dataset, add a short entry here so
future agents know exactly where to look.
