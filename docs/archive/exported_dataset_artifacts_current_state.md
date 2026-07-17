<!-- ARCHIVED 2026-07-17: dated /nvme1 inventory superseded by the shared /groups export contract. -->

# Exported Dataset Artifacts Current State
<!-- doc-meta
status: current_state
last_verified: 2026-06-18
-->

Purpose: give collaborators a practical map of what exists under
`/nvme1/recordings`, what is inside one recording, and what is inside the
current cross-recording exported analytics artifacts.

This is a README-style current-state document. It does not replace the stage
contracts in [`src/fisheye/docs/zarr_structure.md`](../../src/fisheye/docs/zarr_structure.md)
or the export design in
[`cross_recording_analytics_export_design.md`](../cross_recording_analytics_export_design.md).

## Authority Model

Use these roles when deciding what to hand to collaborators:

| Surface | Role | Authority |
| --- | --- | --- |
| Recording directory under `/nvme1/recordings/<recording>/` | Acquisition bundle plus derived sidecars for one recording | Physical source package |
| `<recording>/recording_manifest.json` | Recording context, file inventory, acquisition schema, preflight, repairs | Authoritative recording-context manifest for organizer/registry ingestion |
| `<recording>/zarr/*_analysis.zarr` | Per-recording analysis archive | Authoritative per-recording scientific artifact |
| `<recording>/zarr/*_training.zarr` | Sampled training/review archive | Training/export product derived from recording data |
| `<recording>/zarr/*_clipped_training.zarr` | Sampled training archive from rolling clips | Training product; row provenance comes from `source_frame_index.parquet` |
| `/nvme1/palette_registry.sqlite` | Discovery, status, current locators, quality summaries | Index/cache over source artifacts, not the scientific value store |
| `/nvme1/exports/palette_analytics/v1/...` | Cross-recording Parquet tables | Rebuildable derived analytics product |
| Virtual collection manifest | Frozen cross-recording source selection | Reproducibility boundary for exports |

Practical rule: Zarr archives remain the per-recording source of truth.
Registry rows and Parquet exports are the query/distribution surfaces.

## Hot Filesystem Inventory

Observed under `/nvme1/recordings` on 2026-06-18:

| Item | Count | Notes |
| --- | ---: | --- |
| Top-level directories | 67 | Includes recording directories plus `figures`, `logs`, and `smoke` |
| Directories with `recording_manifest.json` | 64 | Current hot recording packages |
| Directories with `recording_clip_index.json` or `.csv` | 4 | Rolling-clip `sleepyfish` recordings |
| Directories with `recording_frame_index.parquet` | 4 | Full parent-frame maps for rolling clips |
| Directories with `raw/` | 60 | Some video-only packages keep only `cams/` plus Zarr |
| Directories with `cams/` | 64 | Canonical camera-video bundle for recordings |
| Directories with `derived/` | 57 | Acquisition diagnostics, snapshots, external-recorder products, or review artifacts |
| Analysis Zarr stores | 61 | `*/zarr/*_analysis.zarr` |
| Regular training Zarr stores | 72 | `*/zarr/*_training.zarr`, excluding clipped-training suffix |
| Clipped-training Zarr stores | 4 | `*/zarr/*_clipped_training.zarr` |

Protocol split from current hot `recording_manifest.json` files:

| Protocol | Count |
| --- | ---: |
| `DefaultScreen` | 26 |
| `Feeding` | 26 |
| `GoodCopBadCop` | 4 |
| `sickyfish` | 4 |
| `sleepyfish` | 4 |

Recording manifest schema split:

| `artifact_schema_id` | Count | Meaning |
| --- | ---: | --- |
| `behavior_v1` | 52 | Older Citrus/H5 behavior recordings with raw H5/video/timing plus camera video |
| `video_only_v1` | 8 | Video-only recordings, currently `sickyfish` and `sleepyfish` |
| `orange_external_ipc_single_clip_v1` | 4 | Orange external-IPC full-frame plus crop-stream recordings, currently `GoodCopBadCop` |

The registry is broader than this hot filesystem count. It currently has 76
`behavior/free` recording rows and many derived dataset rows. That difference is
expected: registry history can include missing, derived, duplicated, or
normalized rows that do not map one-to-one to current top-level directories.

## Individual Recording Anatomy

Most current recordings have this shape:

```text
/nvme1/recordings/<recording_name>/
  recording_manifest.json
  raw/
  cams/
  derived/
  zarr/
    <recording_name>_analysis.zarr
    <recording_name>_training.zarr
```

Optional or schema-specific children:

| Path | Meaning |
| --- | --- |
| `raw/*.h5` | Citrus/Orange H5 context, protocol, metadata, and timing source when available |
| `raw/*.mp4` | Compatibility or raw acquisition video, often lower-level than canonical `cams/` video |
| `raw/*_update_timing.csv` | Timing profile for `behavior_v1` recordings |
| `raw/recording_session.json`, `raw/ptp_sync_summary.json`, recorder contracts | Orange/runtime acquisition context |
| `cams/Cam*.mp4` | Canonical full-frame camera video |
| `cams/Cam*_meta.csv` | Camera metadata/frame timing table |
| `cams/Cam*_keyframe.json` | Keyframe index for HEVC seeking when present |
| `derived/recording_snapshot.json` | Organized snapshot copied from acquisition context |
| `derived/external_recorder/` | Orange external full-frame recorder diagnostics |
| `derived/external_crop_recorder/` | Orange crop video, crop metadata, YOLO runtime logs, and recorder diagnostics |
| `derived/original_sidecars/` | Repair backups, for example trimmed-video sidecar backups |
| `clips/clip_000000/...` | Rolling-clip source videos for long recordings |
| `recording_clip_index.json`, `recording_clip_index.csv` | Clip inventory for rolling-clip recordings |
| `recording_frame_index.parquet` | Full parent-frame map from recording frames to clip/video/metadata paths |
| `recording_frame_index_manifest.json` | Metadata for the frame-index sidecar |
| `labeled_data/` | Historical/manual labeling sidecar when present |

### `recording_manifest.json`

This file is the first thing to inspect for a recording. It usually contains:

- stable recording names and IDs: `recording_name`, `recording_id`,
  `session_uuid`, `session_start_iso8601_utc`;
- acquisition context: `rig_id`, `arena_id`, `camera_id`, `canvas_name`,
  protocol name fields, source directory, host;
- schema and behavior fields: `recording_type`, `recording_subtype`,
  `behavior_mode`, `artifact_schema_id`;
- file inventory grouped into `files.raw`, `files.cams`, and `files.derived`;
- HEVC keyframe checks for videos;
- optional `preflight` status;
- optional repair/migration notes, especially for video-only recordings.

See [`recording_manifest_contract.md`](../recording_manifest_contract.md) for the
contracted fields and schema-specific expectations.

## Zarr Archives Inside A Recording

Palette uses Zarr v3 archives. Many current stores have consolidated metadata
inline in root `zarr.json`, so consumers should not assume every array has a
separate `zarr.json` file.

Common top-level run families are documented in
[`src/fisheye/docs/zarr_structure.md`](../../src/fisheye/docs/zarr_structure.md).
The main ones collaborators will encounter are:

| Group | Typical content |
| --- | --- |
| `raw_video/` | Imported or sampled frames: `images_full`, `images_ds`, optional `images_ds_rgb`, `original_frame_indices`, `timestamps` |
| `detect_runs/<run>/` | Raw detections: `frame_indices`, `frame_counts`, `bbox_norm_coords`, scores/classes when available |
| `crop_runs/<run>/` | ROI crop geometry or materialized crop tensors, aligned to detect/refined rows |
| `keypoints_runs/<run>/` | Raw keypoint inference output |
| `eye_masks_runs/<run>/` | Legacy/raw eye-mask inference output during migration |
| `subject_mask_runs/<run>/` | Raw multi-component subject-mask probabilities/masks and metrics |
| `refined_detect_runs/<run>/instances/` | Current curated detection authority for training/export consumers |
| `refined_keypoints_runs/<run>/` | Current curated keypoint authority |
| `refined_subject_masks_runs/<run>/` | Current curated component mask authority for body, eyes, and swim bladder |
| `tracking_runs/<run>/` | Arena assignment and track identity outputs |
| `calibration/` | Arena, camera, protocol, rig, and subject metadata snapshots |
| `analysis/stimulus_runs/<run>/` | Source stimulus step timing and geometry |
| `analysis/track_kinematics_runs/` | Track-level position/speed time series |
| `analysis/swim_bout_runs/` | Swim-bout event detection outputs |
| `analysis/bout_kinematics_runs/` | Per-bout physical/heading/eye-gaze measurements |
| `analysis/eye_angle_runs/` | Eye-angle and gaze analysis outputs |
| `analysis/subject_shape_runs/<run>/` | Derived subject-shape geometry: body frame, snout, centerline, tail, B-spline, and component relations |
| `analysis/stimulus_response_runs/` | OMR/stimulus-response summaries |
| `analysis/*/visualizations/` | Persisted PNG/spec artifacts for reviewable analysis runs |

### Analysis Zarrs

`*_analysis.zarr` archives are the per-recording analysis surface. Older
January archives may store `zarr_purpose="production"` in root attrs even when
the registry normalizes them as analysis datasets.

Representative current examples:

| Store | Root attrs / size | Notable contents |
| --- | --- | --- |
| `2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr` | 23,287 frames, 60 fps, 4512 x 4512 source, 557 groups, 4,759 arrays | detection, crops, keypoints, refined detect/keypoints/masks, tracking, stimulus, track kinematics, swim bouts, bout kinematics, eye angles, stimulus response |
| `sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr` | 1,188,000 frames, 30 fps, 4512 x 4512 source, 5 groups, 6 arrays | full-recording detection rows only in current inspected store; latest detect example has 1,187,087 bbox rows |
| `2026-05-29T18-11-16Z_arena_3_GoodCopBadCop_analysis.zarr` | 143,447 frames, 100 fps, 4512 x 4512 source, 51 groups, 172 arrays | detection, refined detection, crops, keypoints, stimulus runs, detection comparison analysis |

### Training Zarrs

`*_training.zarr` archives are sampled training/review products. They usually
contain downsampled frame tensors and row-aligned labels/metadata:

```text
<recording>_training.zarr/
  raw_video/
    images_ds
    images_full              # optional
    original_frame_indices
    timestamps
  detect_runs/
  crop_runs/
  refined_detect_runs/
  keypoints_runs/
  refined_keypoints_runs/
  subject_mask_runs/
  refined_subject_masks_runs/
  analysis/
  calibration/
```

Representative January training example:

- `raw_video/images_ds`: `[231, 640, 640] uint8`
- `raw_video/original_frame_indices`: `[231] int32`
- `refined_detect_runs/<run>/instances/bbox_norm_coords`: `[205, 4]` or
  `[231, 4]`, depending on the run
- `eye_masks_runs/<run>/masks_roi`: `[231, 2, 512, 512] uint8`
- `subject_mask_runs/<run>/masks_roi`: often `[231, 3 or 4, 512, 512] uint8`

### Subject Masks, Refined Masks, And Shape Runs

For collaborators interested in anatomy-level masks, midlines, and spline
fits, use the three-stage model below. These stages are related but not
interchangeable.

| Stage | Typical path | Owns | Do not use it for |
| --- | --- | --- | --- |
| Raw subject masks | `subject_mask_runs/<run>/` | Model probability evidence and cached mask metrics for body, eye union, and swim bladder | Final curated left/right component masks or body-shape interpretation |
| Refined subject masks | `refined_subject_masks_runs/<run>/` | Canonical binary component masks, mask-local geometry, contours, per-component provenance, and review state | Body centerlines, body-frame axes, tail curvature, or B-spline fits |
| Subject shape | `analysis/subject_shape_runs/<run>/` | Derived biological shape geometry from refined masks: body frame, snout, midline/centerline, tail samples, B-spline fits, eye/swim-bladder relations | Raw probability evidence or editable component masks |

Current registry and filesystem state on 2026-06-18:

| Surface | Current observed state |
| --- | --- |
| `subject_mask_runs` | 104 Zarr stores on disk with 339 run directories and 91 distinct run names. Registry latest step status has 105 `ok` and 13 `missing` rows. The current analysis U-Net batch run, `subject_masks_unet_registry_batch_20260504`, is indexed for 45 analysis recordings. |
| `refined_subject_masks_runs` | 104 Zarr stores on disk with 295 run directories and 129 distinct run names. Registry latest step status has 104 `ok` and 66 `missing` rows. The current smart-finalizer analysis batch, `refined_subject_masks_smart_finalizer_batch_20260504`, is indexed for 47 analysis recordings, with a 4-recording pilot run also present. |
| Refined-mask component overview | The registry component-quality view spans 52 analysis recording IDs for `eye_left`, `eye_right`, and `swim_bladder`, and 46 analysis recording IDs for `subject_body`. Current analysis review state is `pending` and lifecycle state is `in_progress`. |
| `analysis/subject_shape_runs` | 48 analysis stores on disk with 56 run directories and 10 distinct run names. The main batch run, `subject_shape_v3_snout_medialjoin_batch_20260505`, is present in 47 stores. Registry latest step status currently reports only `subject_shape|missing|24`, so these shape runs should be treated as batch/canary derived products that are present on disk but not yet registry-promoted as an `ok` latest step. |

#### Raw Subject Masks

Raw subject-mask runs are the segmentation model output layer. The current
analysis batch uses:

- `method="unet_subject_mask_segmenter"`;
- `label_schema_id="subject_v1_union"`;
- `mask_labels=["subject_body", "eyes_union", "swim_bladder"]`;
- `mask_probs_roi`: encoded per-component probability images in ROI pixels;
- `metrics/*`: cached threshold/geometry summaries for quick review.

For the representative archive
`2026-01-28T21-47-47Z_arena_1_DefaultScreen_analysis.zarr`, the current raw
batch run is:

```text
subject_mask_runs/subject_masks_unet_registry_batch_20260504
```

Representative arrays from that run:

| Array | Shape | Type | Meaning |
| --- | ---: | --- | --- |
| `mask_probs_roi` | `[23218, 3, 512, 512]` | `uint8` | Encoded probability images. Channel order comes from `mask_labels`, not hard-coded indices. |
| `available_channels` | `[3]` | `bool` | Which component channels are available in this run. |
| `frame_indices` | `[23218]` | `int32` | Source frame for each detection/ROI row. |
| `detection_indices` | `[23218]` | `int32` | Detection index within the frame. |
| `source_detect_row_index` | `[23218]` | `int32` | Row link back to the source detection table. |
| `metrics/prob_max` | `[23218, 3]` | `float32` | Maximum probability per row/component. |
| `metrics/mask_present` | `[23218, 3]` | `bool` | Cached component-presence flag. |
| `metrics/area_px` | `[23218, 3]` | `float32` | Cached component area in ROI pixels. |
| `metrics/centroid_xy` | `[23218, 3, 2]` | `float32` | Cached centroid in ROI pixel coordinates. |
| `metrics/bbox_xyxy` | `[23218, 3, 4]` | `float32` | Cached ROI-local bounding box. |

Use raw subject-mask runs when collaborators need probability evidence or want
to audit model outputs. Do not treat `eyes_union` as left/right eye identity,
and do not treat this stage as the final mask authority.

#### Refined Subject Masks

Refined subject-mask runs are the current per-recording authority for component
masks. The current smart-finalizer runs use:

- `method="smart_finalize_subject_masks_v1"`;
- `label_schema_id="subject_v1_lr"`;
- `mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"]`;
- `refinement_semantics="canonical_component_masks"`;
- `source_subject_mask_run` pointing back to the raw subject-mask run.

For the representative archive above, the current refined batch run is:

```text
refined_subject_masks_runs/refined_subject_masks_smart_finalizer_batch_20260504
```

Representative arrays from that run:

| Array | Shape | Type | Meaning |
| --- | ---: | --- | --- |
| `masks_roi` | `[23218, 4, 512, 512]` | `uint8` | Canonical binary masks in ROI-local pixels. Channel order comes from `mask_labels`. |
| `available_channels` | `[4]` | `bool` | Component availability. |
| `edit_applied` | `[23218, 4]` | `bool` | Whether a refinement/edit changed a row/component mask. |
| `frame_indices` | `[23218]` | `int32` | Source frame for each row. |
| `detection_indices` | `[23218]` | `int32` | Detection index within the source frame. |
| `metrics/mask_present` | `[23218, 4]` | `bool` | Per-row/component mask presence. |
| `metrics/area_px` | `[23218, 4]` | `float32` | Component mask area in pixels. |
| `metrics/centroid_xy` | `[23218, 4, 2]` | `float32` | Component centroid in ROI-local pixels. |
| `metrics/bbox_xyxy` | `[23218, 4, 4]` | `float32` | ROI-local component bounding box. |
| `components/subject_body/contours/points_xy` | `[9632157, 2]` | `float32` | Ragged body-contour point pool, paired with contour pointer/length arrays. |
| `components/eye_left/geometry/ellipse_params` | `[23218, 5]` | `float32` | Per-row left-eye ellipse fit parameters. |
| `relations/eye_pair/metrics/separation_px` | `[23218]` | `float32` | Left/right eye separation in ROI pixels. |

Component groups contain mask-local details such as:

- `reason` and `reason_bytes` for refinement decisions;
- `manual_override` and `edit_applied`;
- `metrics/*` such as component count, solidity, hole count/fraction,
  largest-component fraction, curvature variation, and noise estimates;
- `geometry/*` such as ellipse parameters for eyes and swim bladder;
- `contours/{points_xy, ptr, len}` for ragged contour storage;
- `provenance/*` for source row/component lineage.

Those arrays are intended to explain or reproduce a component mask. They are
not the stage that owns the body midline, tail axis, B-spline, or curvature.

#### Subject Shape, Midline, And Spline Fits

Subject-shape runs are a deterministic analysis layer derived from refined
subject masks. The current main implementation is:

- `schema_id="analysis.subject_shape_runs"`;
- `schema_version=3`;
- `method="subject_shape_from_refined_masks_v8"`;
- `method_version=8`;
- `source_refined_subject_masks_run="refined_subject_masks_smart_finalizer_batch_20260504"`;
- `centerline_method="snout_anchored_skeleton_longest_endpoint_path_v1"`;
- `centerline_snout_extension_method="prepend_mask_path_to_body_frame_guided_join_v1"`;
- `head_endpoint_semantics="validated_snout_tip"`.

For the representative archive above, the current shape batch run is:

```text
analysis/subject_shape_runs/subject_shape_v3_snout_medialjoin_batch_20260505
```

The run is row-aligned to refined-mask rows through:

- `row_index/frame_indices`;
- `row_index/detection_indices`;
- `row_index/source_refined_row_ids`;
- `source_refined_subject_masks/row_revision` when available.

Coordinate convention: shape outputs are ROI-local pixel coordinates unless a
field name says otherwise. The body-frame `origin_xy` is estimator-defined,
usually the eye-pair midpoint in current runs, and should not be interpreted as
the snout. The `forward_axis_xy` points from posterior/tail toward
anterior/head in image coordinates; `left_axis_xy` is the corresponding lateral
axis; `heading_deg` is derived from the forward axis.

Representative shape arrays:

| Array | Shape | Type | Meaning |
| --- | ---: | --- | --- |
| `row_index/frame_indices` | `[23218]` | `int32` | Source frame for each shape row. |
| `row_index/source_refined_row_ids` | `[23218]` | `int64` | Row link to the refined-mask run. |
| `body_frame/origin_xy` | `[23218, 2]` | `float32` | Body-frame origin, commonly eye-pair midpoint. |
| `body_frame/forward_axis_xy` | `[23218, 2]` | `float32` | Unit body axis pointing tail-to-head. |
| `body_frame/left_axis_xy` | `[23218, 2]` | `float32` | Unit lateral axis. |
| `body_frame/heading_deg` | `[23218]` | `float32` | Heading angle derived from the forward axis. |
| `body_frame/valid` | `[23218]` | `bool` | Whether body-frame estimation succeeded. |
| `components/subject_body/snout_tip_xy` | `[23218, 2]` | `float32` | Validated anterior contour anchor. |
| `components/subject_body/head_endpoint_xy` | `[23218, 2]` | `float32` | In current v3/v8 runs, equals the validated snout tip when centerline is valid. |
| `components/subject_body/centerline_xy` | `[23218, 64, 2]` | `float32` | Ordered body centerline/midline samples from snout/head toward tail. |
| `components/subject_body/body_arclength_px` | `[23218]` | `float32` | Centerline arclength in pixels. |
| `components/subject_body/bspline_control_points_xy` | `[23218, 64, 2]` | `float32` | B-spline control-point representation of the body curve. |
| `components/subject_body/bspline_knots` | `[23218, 68]` | `float32` | B-spline knot vector. |
| `components/subject_body/bspline_sample_xy` | `[23218, 64, 2]` | `float32` | Resampled points along the fitted B-spline. |
| `components/subject_body/tail_sample_xy` | `[23218, 32, 2]` | `float32` | Tail segment samples. |
| `components/subject_body/tail_curvature_px_inv` | `[23218, 32]` | `float32` | Tail curvature in inverse pixels. |
| `components/swim_bladder/caudal_contour_point_xy` | `[23218, 2]` | `float32` | Caudal swim-bladder contour anchor. |
| `components/swim_bladder/caudal_contour_projection_px` | `[23218]` | `float32` | Projection of the caudal anchor along the body frame. |
| `relations/eye_pair/separation_px` | `[23218]` | `float32` | Eye-pair separation in pixels. |
| `relations/eyes_to_body/left_eye_axis_angle_to_body_rad` | `[23218]` | `float32` | Left-eye axis angle relative to the body frame. |

The current centerline/spline workflow is:

1. Read refined masks for `subject_body`, `eye_left`, `eye_right`, and
   `swim_bladder`.
2. Build the body frame from the eye pair and swim bladder.
3. Estimate the snout tip from the body contour using maximum forward
   projection in the body frame.
4. Estimate the caudal swim-bladder contour anchor.
5. Skeletonize the body mask and find the longest endpoint path.
6. Orient the skeleton path using the body frame.
7. Select a medial head-region join point and prepend a mask-bounded path from
   the snout to the skeleton path.
8. Resample the body centerline, project the caudal swim-bladder anchor to the
   tail base, fit the B-spline, and sample tail tangents, normals, and
   curvature.

Validity fields matter. Consumers should check the relevant `*_valid` array
before using a geometry array, and should inspect `*_failure_reason_bytes` when
a fit is invalid. Important validity fields include:

- `body_frame/valid`;
- `components/subject_body/snout_tip_valid`;
- `components/subject_body/centerline_valid`;
- `components/subject_body/centerline_reaches_snout`;
- `components/subject_body/bspline_valid`;
- `components/subject_body/tail_base_valid`;
- `components/subject_body/tail_sample_valid`;
- `components/swim_bladder/caudal_contour_valid`;
- source mask QC fields such as
  `components/subject_body/source_mask_qc_requires_review` and
  `components/subject_body/source_mask_qc_reason_bytes`.

The documented v8 canary on
`2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr` processed 19,235 ROIs:
17,874 rows had a valid snout tip, 17,496 had valid centerlines that reached
the snout, 17,496 had valid B-splines, and 17,495 had valid tail samples.
There were 1,360 severe source-mask QC failures, and the remaining non-QC
centerline failures were recorded as `snout_extension_too_long`. Treat these as
run-specific quality indicators, not a universal guarantee.

Related contracts and implementation notes:

- [`subject_mask_runs_contract.md`](../subject_mask_runs_contract.md)
- [`refined_subject_masks_runs_contract.md`](../refined_subject_masks_runs_contract.md)
- [`subject_shape_runs_contract.md`](../subject_shape_runs_contract.md)
- [`subject_shape_snout_centerline_workflow.md`](../subject_shape_snout_centerline_workflow.md)
- [`subject_shape_landmark_conventions.md`](../subject_shape_landmark_conventions.md)
- [`body_frame_contract.md`](../body_frame_contract.md)

Minimal reader example for the transferred example recording:

```bash
scripts/py docs/examples/read_subject_masks_from_example_recording.py --row 0
```

Minimal conda environment for collaborators who do not have Palette installed:

```bash
conda env create -f docs/examples/subject_mask_reader_environment.yml
conda activate palette-subject-mask-reader
python docs/examples/read_subject_masks_from_example_recording.py --row 0
```

That example opens
`/groups/anibody/anibody/fish/example_recording/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_analysis.zarr`,
prints raw/refined subject-mask labels and array shapes, and reads one ROI row
through `fisheye.shared.mask_store.open_mask_store(...)`, so it works whether
the refined source stores dense `masks_roi`, compact editable `mask_bitpacked`,
compact final/read-mostly `mask_rle`, or a combination.

### Clipped Training Zarrs

`*_clipped_training.zarr` archives are sampled from rolling clips. They keep
sample-local arrays in Zarr and exact source-video provenance in a Parquet
sidecar.

Representative `sleepyfish` clipped-training example:

- root `zarr_purpose="training"`;
- root `source_frame_index_path="source_frame_index.parquet"`;
- root `source_recording_frame_index_path` points to the parent
  `recording_frame_index.parquet`;
- `raw_video/images_ds`: `[238, 640, 640] uint8`;
- `raw_video/original_frame_indices`: `[238] int64`, holding parent-frame
  indices;
- `source_frame_index.parquet`: 238 rows with `sample_index`, parent frame,
  clip ID, clip-local frame, video path, metadata path, keyframe path, and
  `sample_plan_id`.

For full rolling-clip parent provenance, read
`recording_frame_index.parquet`; the inspected `sleepyfish` example has
1,188,000 rows and maps every parent frame to a clip/video/metadata/keyframe
location.

## Cross-Recording Analytics Exports

The current cross-recording export area is:

```text
/nvme1/exports/palette_analytics/
  manifests/collections/
    movement_bouts_20260128_all_analysis_v002.manifest.json
  v1/
    manifests/
      export_run_id=run_20260511T_compact_batch_v001.json
    recording_summary/
    stimulus_steps/
    stimulus_step_summary/
    stimulus_response_per_fish_step/
    swim_bout_metrics/
    bout_kinematics_metrics/
```

The active collection is:

| Field | Value |
| --- | --- |
| `collection_id` | `movement_bouts_20260128_all_analysis_v002` |
| Name | `2026-01-28 movement-bouts analysis v002` |
| Manifest | `/nvme1/exports/palette_analytics/manifests/collections/movement_bouts_20260128_all_analysis_v002.manifest.json` |
| Records | 52 total, 52 included |
| Purpose | Frozen source selection for cross-recording movement-bout analytics |

The latest active export is:

| Field | Value |
| --- | --- |
| `export_run_id` | `run_20260511T_compact_batch_v001` |
| Manifest | `/nvme1/exports/palette_analytics/v1/manifests/export_run_id=run_20260511T_compact_batch_v001.json` |
| Created | `2026-05-11T22:31:36.809276+00:00` |
| Source recordings | 52 |
| Tables | 6 |
| Diagnostics | 0 |
| Palette commit | `2197b51ae7a16d17c4283a4710992af25fc3baa8` |
| Git dirty at export | `false` |

The registry also has an older active export,
`run_20260507T_manifest_v002`. Both active exports pass file validation, but
collaborators should normally resolve the latest export through the registry
instead of hard-coding paths.

### Latest Active Table Inventory

Validated for `run_20260511T_compact_batch_v001`:

| Table | Row axis | Rows | Parts | Columns | Bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| `recording_summary` | one row per source recording | 52 | 52 | 29 | 675,247 |
| `stimulus_steps` | one row per stimulus step | 156 | 52 | 64 | 1,382,099 |
| `stimulus_step_summary` | one row per fish/step summary | 156 | 52 | 38 | 823,011 |
| `stimulus_response_per_fish_step` | one row per fish/response run/step | 156 | 52 | 172 | 3,239,769 |
| `swim_bout_metrics` | one row per swim bout | 19,662 | 52 | 70 | 5,517,892 |
| `bout_kinematics_metrics` | one row per bout-kinematics measurement family/level | 78,648 | 52 | 150 | 21,890,523 |

All six tables are partitioned by export run directory and have one Parquet
part per source recording.

Common identity/provenance columns include:

- `collection_id`
- `collection_manifest_path`
- `collection_manifest_sha256`
- `export_schema_version`
- `recording_id`
- `zarr_path`
- `table_name`
- `source_lineage_hash`
- source run IDs such as `stimulus_run`, `source_track_kinematics_run`,
  `swim_bout_run`, `source_swim_bout_run`, and `bout_kinematics_run`
- protocol signature columns such as `protocol_signature_hash`,
  `derived_protocol_hash`, `protocol_mode_sequence`, and
  `protocol_step_count`

### Table Semantics

| Table | Use for |
| --- | --- |
| `recording_summary` | Fast per-recording cohort overview: protocol signature, source run IDs, movement totals, fish counts |
| `stimulus_steps` | Exact stimulus-step timing and protocol/stimulus geometry |
| `stimulus_step_summary` | Movement and bout summaries per fish per stimulus step |
| `stimulus_response_per_fish_step` | OMR/stimulus-response metrics, latency fields, grating/radial response summaries |
| `swim_bout_metrics` | Bout event boundaries, duration, peak speed, path length, step assignment |
| `bout_kinematics_metrics` | Per-bout heading, movement, physical-active, and eye-gaze measurements |

### Recommended Access Commands

Resolve latest active exports through the registry:

```bash
scripts/py -m fisheye.utils.query_analytics_exports \
  --registry /nvme1/palette_registry.sqlite \
  --latest
```

Resolve one table path:

```bash
scripts/py -m fisheye.utils.resolve_analytics_export \
  --registry /nvme1/palette_registry.sqlite \
  --table swim_bout_metrics \
  --format path
```

Validate indexed table files:

```bash
scripts/py -m fisheye.utils.check_analytics_exports \
  --registry /nvme1/palette_registry.sqlite \
  --check-files
```

Example resolved latest `swim_bout_metrics` path:

```text
/nvme1/exports/palette_analytics/v1/swim_bout_metrics/export_run_id=run_20260511T_compact_batch_v001
```

Python/Polars example:

```python
from pathlib import Path
import polars as pl

table_dir = Path(
    "/nvme1/exports/palette_analytics/v1/swim_bout_metrics/"
    "export_run_id=run_20260511T_compact_batch_v001"
)
df = pl.scan_parquet(str(table_dir / "*.parquet"))
summary = (
    df.group_by("recording_id")
    .agg(
        pl.len().alias("bout_count"),
        pl.col("duration_s").mean().alias("mean_duration_s"),
        pl.col("peak_physical_speed_mm_s").median().alias("median_peak_speed_mm_s"),
    )
    .collect()
)
```

DuckDB example:

```sql
SELECT recording_id, count(*) AS bout_count, median(duration_s) AS median_duration_s
FROM read_parquet('/nvme1/exports/palette_analytics/v1/swim_bout_metrics/export_run_id=run_20260511T_compact_batch_v001/*.parquet')
GROUP BY recording_id
ORDER BY recording_id;
```

### Historical Export Directories

There are nine export-run directories on disk under
`/nvme1/exports/palette_analytics/v1`, but only two are active in the registry.

Active registry-indexed exports:

- `run_20260511T_compact_batch_v001`
- `run_20260507T_manifest_v002`

Older filesystem exports with manifests but no active registry row:

- `20260505T075638Z`
- `run_20260505T075854Z`
- `run_20260505T080239Z`
- `run_20260505T081936Z`
- `run_20260505T083307Z`
- `run_20260505T103324Z`
- `run_20260505T202135Z`

Treat those as historical/debug artifacts unless a collaborator explicitly
needs to reproduce an older result.

## What To Share

For collaborators who want one recording:

1. Share the whole recording directory when practical.
2. Point them first to `recording_manifest.json`.
3. Use `zarr/*_analysis.zarr` for authoritative per-recording analysis.
4. For subject masks, midlines, and spline fits, point them to
   `subject_mask_runs`, `refined_subject_masks_runs`, and
   `analysis/subject_shape_runs` inside the analysis Zarr, in that order.
5. Use `zarr/*_training.zarr` only when they need sampled training/review rows.
6. For rolling-clip training archives, include both
   `source_frame_index.parquet` inside the Zarr and the parent
   `recording_frame_index.parquet`.

For collaborators who want cross-recording movement-bout analytics:

1. Use the latest active registry-resolved export.
2. Start with `recording_summary`, `swim_bout_metrics`, and
   `bout_kinematics_metrics`.
3. Use `stimulus_steps`, `stimulus_step_summary`, and
   `stimulus_response_per_fish_step` for protocol/OMR questions.
4. Keep the collection manifest with any delivered Parquet tables; it freezes
   the 52 source recordings and source selections.

## Copying A Full Recording Folder

For a full recording-directory transfer, keep the final recording layout intact:

```text
<destination>/<recording_name>/
  recording_manifest.json
  raw/
  cams/
  derived/
  zarr/
    <recording_name>_analysis.zarr
    <recording_name>_training.zarr
```

Recommended policy:

1. Copy only from a stable source. No process should be writing to the Zarr
   stores or sidecars while the copy is running.
2. Use plain `rsync` for ordinary files and large contiguous files such as MP4,
   H5, CSV, JSON, and Parquet. Do not pass `-z`; these recordings are already
   dominated by binary/video data and compression usually wastes CPU.
3. Treat each `.zarr` directory as a transport bundle when crossing filesystems
   or hosts. An uncompressed tar stream or tar file converts many small Zarr
   files into one sequential transfer.
4. Unpack the tarred Zarr at the destination if the recipient needs a normal
   readable Palette recording. The tarball is a transport artifact, not the
   canonical storage format.
5. Validate after transfer. Use a cheap structure check first, and use a full
   checksum pass only when the copy needs byte-level assurance.

The recommended command is:

```bash
scripts/py -m fisheye.utils.copy_recording \
  /nvme1/recordings/<recording_name> \
  /new/location
```

That is a dry run. It plans a destination at
`/new/location/<recording_name>`, copies ordinary files with `rsync` excluding
`zarr/*.zarr/`, and transfers each top-level Zarr store with an uncompressed tar
stream. Add `--apply` after reviewing the plan:

```bash
scripts/py -m fisheye.utils.copy_recording \
  /nvme1/recordings/<recording_name> \
  /new/location \
  --apply
```

Useful options:

- `--destination-is-recording-dir`: treat the second argument as the exact
  recording directory instead of a parent directory.
- `--resume`: allow copying into a non-empty destination recording directory.
- `--validate checksum`: after copying, run an expensive `rsync --checksum`
  dry-run to compare source and destination byte content.
- `--zarr-mode tarball --archive-dir <dir>`: create `.zarr.tar` files for
  delivery instead of unpacking readable Zarr stores at the destination.
- `--json`: emit a machine-readable plan/result.

Example local or mounted-filesystem copy when the destination should be
directly readable:

```bash
src=/nvme1/recordings/<recording_name>
dst=/new/location/<recording_name>

mkdir -p "$dst"

rsync -a --info=progress2 --partial --partial-dir=.rsync-partial \
  --exclude='zarr/*.zarr/' \
  "$src"/ "$dst"/

mkdir -p "$dst/zarr"
for store in "$src"/zarr/*.zarr; do
  name=$(basename "$store")
  tar -C "$src/zarr" -cf - "$name" | tar -C "$dst/zarr" -xf -
done
```

Example tarball staging for delivery or later unpacking:

```bash
src=/nvme1/recordings/<recording_name>
archive_dir=/scratch/<recording_name>_zarr_tar

mkdir -p "$archive_dir"
for store in "$src"/zarr/*.zarr; do
  name=$(basename "$store")
  tar -C "$src/zarr" -cf "$archive_dir/$name.tar" "$name"
done
```

Example remote transfer without creating an intermediate tar file:

```bash
src=/nvme1/recordings/<recording_name>
remote=<user>@<host>
dst=/new/location/<recording_name>

ssh "$remote" "mkdir -p '$dst'"
rsync -a --info=progress2 --partial --partial-dir=.rsync-partial \
  --exclude='zarr/*.zarr/' \
  "$src"/ "$remote:$dst"/

for store in "$src"/zarr/*.zarr; do
  name=$(basename "$store")
  ssh "$remote" "mkdir -p '$dst/zarr'"
  tar -C "$src/zarr" -cf - "$name" | ssh "$remote" "tar -C '$dst/zarr' -xf -"
done
```

For tarball delivery instead of direct unpacking, create hashes next to the
archives and verify them after the rsync:

```bash
sha256sum /path/to/*.zarr.tar > /path/to/SHA256SUMS
rsync -a --info=progress2 --partial /path/to/*.zarr.tar /path/to/SHA256SUMS <destination>/
cd <destination>
sha256sum -c SHA256SUMS
```

If the recipient needs a normal readable recording after tarball delivery,
unpack the Zarr archives under the destination `zarr/` directory:

```bash
mkdir -p "$dst/zarr"
for archive in "$archive_dir"/*.zarr.tar; do
  tar -C "$dst/zarr" -xf "$archive"
done
```

For a definitive source-vs-destination check after unpacking, run an expensive
dry-run checksum comparison:

```bash
rsync -a --dry-run --checksum --delete "$src"/ "$dst"/
```

Do not run the final checksum pass casually on large recordings; it reads the
full source and destination. It is useful for one-time handoff validation or
when a storage migration needs a strong audit trail.

## Current Gaps And Caveats

- The current cross-recording Parquet export covers the 52-recording
  `movement_bouts_20260128_all_analysis_v002` collection. It does not cover
  `GoodCopBadCop`, `sickyfish`, `sleepyfish`, or newer recording-only exports.
- Dense masks, raw probabilities, raw/downsampled frames, contours, and other
  high-dimensional data remain in per-recording Zarrs by design. They are not
  exported to the analytics Parquet lake by default.
- Subject-shape centerlines, B-splines, and tail samples also remain in
  per-recording analysis Zarrs. The current cross-recording Parquet export does
  not include these high-dimensional shape arrays.
- The on-disk `subject_shape_v3_snout_medialjoin_batch_20260505` runs are
  useful batch products, but registry latest-step status has not promoted
  `subject_shape` to `ok`. Share that status with collaborators if they need a
  production-authority signal.
- Absolute paths currently point at `/nvme1`. If recordings are moved, use the
  registry and relocation policy rather than editing historical manifests in
  place. See
  [`recording_store_relocation_components.md`](../recording_store_relocation_components.md).
- Some older analysis Zarrs use root `zarr_purpose="production"` while the
  registry and current terminology call them analysis datasets.
- Video-only and GoodCopBadCop manifests may have null biological metadata such
  as genotype or DPF. Do not assume these fields are complete unless registry
  context confirms them.
- Clipped analysis shells are still a distinct/prototype layout. Current
  clipped-training Zarrs are usable sampled training products, but full
  rolling-clip analysis consumption should go through the documented finalized
  collection resolver once present.

## Validation Performed

This document was built from:

- filesystem inventory under `/nvme1/recordings`;
- `recording_manifest.json` files;
- Zarr metadata from `zarr.json` and consolidated metadata, without opening
  real Zarr groups through the Python zarr API;
- Parquet footers for export schemas and row counts;
- registry queries against `/nvme1/palette_registry.sqlite`;
- subject-mask, refined-mask, body-frame, and subject-shape contracts listed
  above;
- `scripts/py -m fisheye.utils.check_analytics_exports --registry /nvme1/palette_registry.sqlite --check-files`,
  which reported `ok` for every active indexed table and zero missing parts.
