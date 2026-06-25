# Acquisition Crop-Video Pose Training Workflow
<!-- contract-meta
version: 2
status: draft
last_verified: 2026-06-25
-->

## Purpose

Some Orange recordings include acquisition-time crop videos under
`derived/external_crop_recorder/`. These videos are the crop stream Orange used
or can use for online pose inference: a high-resolution crop cut from a much
larger source frame.

This workflow persists training rows directly from those crop videos. It is
intended for pose-model training where the model input should match the crop
stream Orange will feed at runtime.

This is separate from full-frame detector training. Full-frame detector training
still uses sampled camera frames and full-frame boxes. Acquisition crop-video
pose training starts with crop-video pixels and crop/source lineage; crop-local
keypoint labels are added by review or by projecting existing refined labels
when available.

Source-selection policy: `docs/acquisition_video_stream_source_policy.md`.

## Preferred Combined Training Zarr

For new recordings such as `RedScare`, the preferred shape is one
`<recording>_training.zarr` per recording:

```text
raw_video/
  images_full                 sampled full-frame luma frames
  images_ds                   sampled downsampled full-frame luma frames
  original_frame_indices      source frame ids for detector training rows

crop_runs/<acquisition_crop_video_run>/
  roi_images                  sampled decoded crop-video frames
  frame_indices               same source frame clock as raw_video/original_frame_indices
  source_training_row_indices row indices into raw_video sampled frames
  source_crop_xywh            full-frame crop geometry from crop_meta
  source_crop_video_frame_indices
                              zero-based frame ids inside the crop video
  source_crop_local_frame_ids Orange/acquisition-local ids from crop_meta
```

Build it in two phases:

```bash
scripts/py -m fisheye.utils.import_recordings_training /groups/johnson/johnsonlab/jeremy/recordings \
  --path-contains RedScare \
  --target-sampled-frames 200 \
  --include-acquisition-crop-video \
  --acquisition-crop-run-prefix crop_red_scare_acquisition_crop_video_training \
  --apply
```

The wrapper first imports sampled full-frame frames for detector training. It
then appends sampled crop-video frames into `crop_runs/<run>` using
`raw_video/original_frame_indices` as the sampling plan. This keeps full-frame
detector rows and crop-video pose-labeling rows on the same source frame clock.

The lower-level append command is:

```bash
scripts/py -m fisheye.utils.append_acquisition_crop_video_training \
  /path/to/recording/zarr/<recording>_training.zarr \
  --recording-dir /path/to/recording \
  --run-name crop_red_scare_acquisition_crop_video_training_<recording> \
  --apply
```

Dry run omits `--apply` and prints selected crop-video row counts and reject
counts.

The append step resolves:

- `recording_manifest.json.video_streams.streams.crop.video`, or the single
  `derived/external_crop_recorder/*_crop_external.mp4` fallback.
- `derived/external_crop_recorder/*_crop_meta.csv`.
- source frame rows from `raw_video/original_frame_indices`.

## Row Selection

For each sampled source frame in `raw_video/original_frame_indices`:

- `frame_indices` maps to crop metadata by `recording_frame_id - 1`.
- `source_crop_video_frame_indices` maps to crop MP4 frame order.
- crop metadata must exist for the source frame.
- `blank_frame` must be false.
- `has_detection` must be true.
- `crop_x,crop_y,crop_w,crop_h` must be finite and positive.

Reject counts are reported so bad crop sizing or bad realtime tracking is
visible before writing a training zarr.

If the realtime detector failed and Orange encoded a blank crop frame, that row
is not written to `crop_runs/<run>/roi_images`. The crop run may therefore have
fewer rows than `raw_video/original_frame_indices`. Use
`source_training_row_indices` to map crop-video rows back to the sampled
full-frame training rows.

## Crop-Video Output Structure

The appended crop run uses familiar Palette crop-run surfaces:

```text
crop_runs/<crop_run>/
  roi_images                         uint8, (N,H,W)
  frame_indices                      int64, source Palette frame indices
  source_frame_indices               int64, alias of frame_indices
  source_recording_frame_ids         int64, original 1-based acquisition ids
  source_training_row_indices        int64, row ids into raw_video sampled frames
  source_crop_meta_row_indices       int64
  source_crop_video_frame_indices    int64, zero-based frame indices inside crop video
  source_crop_local_frame_ids        int64, Orange/acquisition-local ids from crop_meta
  source_crop_xywh                   float32, full-frame crop geometry
  bbox_roi_xyxy                      float32, realtime detection bbox in crop pixels
  bbox_norm_coords                   float32, realtime detection bbox as xywhn
  realtime_detection_bbox_roi_xyxy   float32, realtime bbox from crop_meta in crop pixels
  detection_indices                  int32
  frame_counts                       int32
  detection_source                   int8
```

`source_crop_xywh` is intentionally retained. It is the reversible mapping back
to the original source frame and is useful for QC, debugging, and comparing
runtime crop sufficiency. `bbox_roi_xyxy` and `bbox_norm_coords` come from the
realtime acquisition bbox projected into crop-video coordinates and are included
for QC/visual checks.

The crop run attrs also persist the detection gate:

```text
crop_detection_required = true
blank_crop_frames_excluded = true
source_sample_count
selected_sample_count
rejected_missing_crop_meta_frame
rejected_blank_crop_frame
rejected_crop_has_no_detection
rejected_nonfinite_crop_geometry
```

When crop metadata carries an explicit `crop_video_frame_index` column, Palette
uses it for `source_crop_video_frame_indices`. Current RedScare crop metadata
does not carry that column, so Palette falls back to `source_crop_meta_row_indices`
because the encoded crop MP4 has one frame per metadata row. `local_frame_id` is
retained as `source_crop_local_frame_ids` for provenance only; it is not used to
decode the crop MP4.

When reviewed/refined keypoints already exist, a separate labeled export can
also write `keypoints_runs/<run>` with crop-video-local keypoints. That is not
required for initial manual labeling zarrs.

## Pixel Contract

Current exported `roi_images` are decoded with PyNvVideoCodec luma:

```text
roi_pixel_contract_name = orange_mono_pynvvc_luma_uint8_v1
decode_backend = pynvvc_luma
frame_format_confirmation_status = pending_orange_confirmation
```

This is the best current assumption for Orange-compatible monochrome crop
training, but it is not yet a confirmed statement about the exact in-memory
frames Orange feeds into its crop-video encoder or future pose model.

## Orange Confirmation Needed

Before treating this as the final canonical Orange pose-training pixel contract,
ask Orange to confirm:

- What exact array is passed into the crop-video encoder: raw luma, RGB, BGR,
  NV12 Y plane, or another representation?
- Is the crop encoded from an in-memory crop before compression, or should
  training match frames decoded back from the recorded crop video?
- What bit depth, dtype, and numeric range are used before encode?
- Are frames resized, padded, or letterboxed before encode?
- What codec, pixel format, color range, and color matrix are requested during
  encode?
- For future online pose, will the model consume the same pre-encoded crop, a
  decoded crop-video frame, or a separately normalized tensor?

Until Orange answers these, the exporter records
`frame_format_confirmation_status=pending_orange_confirmation`.

## Boundary

The crop-video training run inside `_training.zarr` is a training asset. It does
not create a normal analysis-stage `crop_run` inside the source analysis zarr.
If acquisition boxes should drive normal analysis stages, import them as a
standard `detect_runs/<run>` first and route through detect quality/refinement.
