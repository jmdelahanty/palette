# Acquisition Video Stream Source Policy
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-06-25
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

Future work may also mirror per-frame crop metadata arrays from the crop-meta
CSV into the analysis zarr. Those arrays should still be treated as acquisition
media metadata, not as proof that any model stage consumed crop-video pixels.

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

Useful optional QC arrays:

```text
bbox_roi_xyxy
bbox_norm_coords
realtime_detection_bbox_roi_xyxy
offline_center_inside_crop
offline_bbox_inside_crop
margin_to_crop_edge
```

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
