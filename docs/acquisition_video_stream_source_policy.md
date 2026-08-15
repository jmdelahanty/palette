# Acquisition Video Stream Source Policy
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-06-29
-->

## Principle

Acquisition video streams are recording-level available media. They are not
implicitly downstream model inputs.

For Orange external-IPC recordings, Palette may know about both:

- the full-frame camera stream under `cams/`
- an acquisition-time crop stream under `derived/external_crop_recorder/`

The presence of the crop stream means "this artifact exists and can be used by
eligible consumers." It does not mean keypoints, subject masks, crops, training
exports, or review tools used that stream.

The productionization follow-on for complete per-frame Zarr import and current
GoodBatBadBat hybrid pixel routing is documented in
[`goodbatbadbat_acquisition_crop_stream_production_checklist_20260815.md`](goodbatbadbat_acquisition_crop_stream_production_checklist_20260815.md).
Future recordings that intentionally retain crop pixels without continuous
full-frame video are a separate storage profile governed by
[`crop_only_recording_storage_profile.md`](crop_only_recording_storage_profile.md).

## Availability Surface

Recording import/backfill mirrors acquisition media inventory into:

```text
analysis/acquisition_video_streams/
  streams/full/
  streams/crop/
```

That surface describes file paths, stream-level sidecars, row counts,
availability status, crop metadata location, frame clock, dimensions, codec, and
other stream facts when available.

Palette mirrors every crop-meta CSV row into a digest-bound immutable ledger at
`analysis/acquisition_video_streams/streams/crop/ledger_runs/<run>`. The crop
stream's `canonical_ledger_*` attrs are written pointer-last after validation.
These arrays are acquisition media metadata, not proof that any model stage
consumed crop-video pixels.

## Orange External Crop Metadata Contract

As clarified by the Orange writer/code review on 2026-06-29, the external crop
recorder metadata has two different geometry concepts that must not be
collapsed:

```text
crop_x, crop_y, crop_w, crop_h
detection_x, detection_y, detection_w, detection_h
```

`crop_x`, `crop_y`, `crop_w`, and `crop_h` are the canonical crop-window
geometry for the encoded crop video frame. They are `xywh` in full-frame camera
pixels, top-left origin, x right, y down, zero-based, and represent the actual
clamped source ROI copied into the crop video. These fields are not fish
bounding boxes.

`detection_x`, `detection_y`, `detection_w`, and `detection_h` are the selected
live model detection that drove the crop controller when `has_detection=1`.
They are also `xywh` in full-frame camera pixels, but they are only the selected
postprocessed detection, not the full live detection history, not ground truth,
and not proof that the full fish is inside the crop.

If `Cam<serial>_yolo_events.jsonl` is available, its `detections[]` rows are
the preferred source for canonical Orange live model detections because they
can contain all recorded live detections and status/provenance fields. If only
`Cam<serial>_crop_meta.csv` is available, `detection_*` may be imported as the
selected live bbox provenance stream, but crop-video sufficiency should still be
evaluated against offline refined detections.

For no-detection crop frames, Orange writes an explicit blank/black crop video
frame. These rows are identified by:

```text
has_detection = 0
blank_frame = 1
```

On those rows, `crop_*` default zero values must not be treated as a meaningful
top-left source crop. Consumers should classify them as blank no-detection crop
frames and either skip them or recover pixels from the full-frame video using
offline detection geometry.

Frame alignment rules:

- crop-video frame index is the zero-based row order in `crop_meta.csv`
- `recording_frame_id` is one-based during active recording
- when rows are continuous, `crop_video_frame_index = recording_frame_id - 1`
- `local_frame_id` is Orange's acquisition-thread local frame id and is
  provenance, not a crop-video frame index unless a source contract explicitly
  says they are identical
- `camera_frame_id` is source-native and its index base is not guaranteed

Current Orange crop videos are luma source crops: Mono8 crop buffers are encoded
as NV12/HEVC MP4 by filling the Y plane with crop pixels and chroma with 128.
They are acquisition ROI pixels, not RGB visualization exports and not the raw
YOLO tensor preprocessing surface.

## Downstream Run Source Selection

Every downstream run that consumes pixels must declare the pixel source it
actually used. Consumers must not infer source selection from stream
availability alone.

Recommended run-level source values:

```text
source_pixels = palette_crop_run
source_pixels = acquisition_crop_video
source_pixels = raw_camera_video
source_pixels = analysis_raw_video
```

Examples:

- A normal Palette keypoint run from `crop_runs/<run>/roi_images` should record
  `source_pixels=palette_crop_run` and `source_crop_run=<run>`, even if an
  acquisition crop video exists.
- A crop-video pose training export should record
  `source_pixels=acquisition_crop_video`, the crop-video path, the crop-meta
  path or zarr array path, `source_crop_xywh`, crop-video frame indices, and
  acquisition-local frame IDs when available.
- A full-frame detector run should record `source_pixels=raw_camera_video` or
  the equivalent source-video path and should not claim crop-video lineage.

## Required Lineage For Crop-Video Consumers

If a run uses acquisition crop-video pixels, it must carry enough lineage to map
each output row back to the source recording frame and the source crop geometry:

```text
source_crop_video_path
source_crop_meta_path or source_crop_meta_array_path
source_crop_meta_row_indices
source_crop_video_frame_indices
source_crop_local_frame_ids
source_recording_frame_ids
frame_indices
source_crop_xywh
```

`source_crop_video_frame_indices` are zero-based frame indices into the encoded
crop video. `source_crop_local_frame_ids` preserve Orange/acquisition-local frame
IDs when present, but they must not be treated as crop-video frame indices unless
the source contract explicitly states that they are identical.

`source_crop_xywh` is in full-frame source-image coordinates. It is the
reversible mapping between crop-video pixel coordinates and full-frame
coordinates. It is also useful for crop sufficiency checks, visual QC, and
runtime/offline comparison.

### Coordinate Contract

Model execution may operate entirely in crop-video or ROI-local coordinates.
For example, a pose or SAM subject-body model consuming
`384x384` acquisition crop-video frames can use ROI-local keypoint prompts and
ROI-local mask pixels without needing the full-frame image dimensions at
inference time.

Persisted Palette outputs must still remain linkable back to the parent
recording frame and full-frame coordinate system. The durable contract is:

- row identity comes from `frame_indices`, `source_recording_frame_ids`,
  `source_crop_video_frame_indices`, and crop-meta row lineage
- crop placement comes from full-frame `source_crop_xywh` or equivalent
  `roi_coordinates_full`
- ROI-local points, boxes, and masks are interpreted relative to the decoded
  crop-video frame
- full-frame pixel coordinates are derived as
  `full_x = source_crop_x + roi_x` and `full_y = source_crop_y + roi_y` when
  crop-video pixels are 1:1 with `source_crop_xywh`
- full-frame normalized boxes are derived only after projection into the
  parent full-frame pixel coordinate system

Do not overload `bbox_norm_coords` with two meanings. Unqualified
`bbox_norm_coords` must mean canonical full-frame-normalized `[cx, cy, w, h]`
for every crop source. If a crop-video or ROI-local normalized box is useful for
QC, write it with an explicitly local name such as `bbox_crop_norm_coords`.
Use attrs and lineage arrays to describe where the crop pixels came from; do not
change the meaning of shared geometry arrays based on `source_pixels`.

Useful optional QC arrays:

```text
bbox_roi_xyxy
bbox_norm_coords
bbox_crop_norm_coords
realtime_detection_bbox_roi_xyxy
selected_live_detection_bbox_img_xyxy
selected_live_detection_bbox_norm_coords
offline_center_inside_crop
offline_bbox_inside_crop
margin_to_crop_edge
crop_state
crop_sufficiency_status
```

For acquisition crop videos, `crop_state` should use explicit values such as:

```text
detected_crop
blank_no_detection
missing_or_dropped
```

Normal detection-quality metrics should run on canonical detection streams
(`detect_runs`, `refined_detect_runs`, or an explicitly imported Orange live
detection run). They should not run on `crop_*` geometry. Crop videos need a
separate crop-sufficiency comparison against offline detections, including
whether the offline center/bbox lies inside the crop window and how close it is
to the crop edge.

## Allowed Mixed State

It is valid for a recording to have acquisition crop-video media and still have
downstream keypoints or subject masks generated from normal Palette crop runs.

This mixed state is expected because:

- some crop videos are too small for high-quality keypoint or mask inference
- some model paths need 512x512 Palette ROI caches for now
- crop-video parity may not yet be validated for every recording
- crop-video pixels and Palette crop-run pixels can intentionally be compared
  as separate sources

The invariant is not "use crop video when it exists." The invariant is "declare
the pixel source used by each run, and provide row-level lineage when that source
is acquisition crop video."

## Consumer Rule

Readers should resolve pixel source in this order:

1. Read the run-level source declaration.
2. Follow the declared source-specific lineage arrays/attrs.
3. Treat missing source declaration as legacy and infer only from the run type,
   with a warning when ambiguity matters.
4. Never switch a run to crop-video semantics solely because
   `analysis/acquisition_video_streams/streams/crop` exists.

This lets Palette keep acquisition media inventory complete while allowing each
stage to choose the safest input representation for its current model and
validation state.
