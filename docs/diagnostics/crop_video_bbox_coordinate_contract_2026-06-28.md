# Crop-Video BBox Coordinate Contract Repair
<!-- contract-meta
version: 1
status: design-note
last_verified: 2026-06-28
-->

## Summary

Acquisition crop videos should not create a second downstream coordinate
contract. They are a different pixel source for a normal crop run, not a new
geometry model.

The repair direction is:

- `bbox_norm_coords` always means canonical full-frame-normalized
  `[cx, cy, w, h]`.
- Crop-local boxes use explicit local names, for example `bbox_roi_xyxy` and,
  only if needed, `bbox_crop_norm_coords`.
- Crop-video provenance lives in attrs and lineage arrays such as
  `source_pixels=acquisition_crop_video`, `source_crop_xywh`,
  `source_crop_video_frame_indices`, and `source_crop_meta_row_indices`.
- Consumers should not need separate coordinate interpretation rules depending
  on whether crop pixels came from Palette crop generation or Orange crop-video
  media.

## Why This Came Up

Orange external-IPC recordings can contain both:

- a full-frame camera video under `cams/`
- an acquisition-time crop video under `derived/external_crop_recorder/`

The crop video contains actual pixels produced at acquisition time. It is not
only an instruction stream. It can differ from a later Palette-reconstructed
crop because of detection latency, crop-selection policy, blank-frame policy,
codec/pixel format, and crop size.

Geometrically, though, it is still the same object Palette already knows how to
represent: a crop row with local pixels plus a placement transform back into the
parent full-frame recording.

## Current Divergence

The Orange crop metadata contract says geometry columns in `crop_meta.csv` are
full-frame pixels:

```text
crop_x, crop_y, crop_w, crop_h
detection_x, detection_y, detection_w, detection_h
```

Palette currently projects `detection_x/y/w/h` into crop-video coordinates by
subtracting the crop origin and normalizing by crop width/height. The resulting
local-normalized box is written to:

```text
crop_runs/<run>/bbox_norm_coords
```

and marked with:

```text
bbox_norm_coords_semantics =
"realtime_detection_bbox_xywh_normalized_to_crop_video_frame"
```

This was intended as a QC convenience for acquisition crop-video training rows,
but it is a poor long-term contract because `bbox_norm_coords` is already widely
understood as the canonical normalized detection box surface.

Concrete current writers:

- `src/fisheye/utils/append_acquisition_crop_video_training.py`
- `src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py`

The key risk is consumer confusion. For example, code that projects
`bbox_norm_coords` by multiplying by parent full-frame width/height is correct
for canonical detection boxes and wrong for crop-frame-normalized boxes.

## Design Decision

Unqualified geometry names must have one meaning across crop-run sources.
Provenance should describe where the pixels came from, not change the meaning of
the core coordinate arrays.

### Canonical crop-run geometry

```text
frame_indices
roi_images                         optional materialized local crop pixels
source_crop_xywh                   full-frame crop window, [x, y, w, h]
roi_coordinates_full               full-frame crop top-left, [x, y]
bbox_img_xyxy                      detection box in full-frame pixels
bbox_norm_coords                   detection box normalized to full-frame dims
bbox_roi_xyxy                      detection box in local crop/ROI pixels
```

### Acquisition crop-video lineage

```text
source_pixels = "acquisition_crop_video"
source_crop_video_path
source_crop_meta_path or source_crop_meta_array_path
source_crop_meta_row_indices
source_crop_video_frame_indices
source_crop_local_frame_ids
source_recording_frame_ids
```

### Optional local-normalized helper

If a local-normalized box is worth retaining for QC or crop-local model prompts,
it should use a qualified name:

```text
bbox_crop_norm_coords
```

That array must be documented as normalized to the decoded crop/ROI frame, not
to the parent full-frame camera image.

## Are These Arrays All Necessary?

Some redundancy is useful because different consumers need different access
patterns.

- `source_crop_xywh` is the full crop window. It preserves the crop extent and
  is the reversible transform between crop pixels and parent full-frame pixels.
- `roi_coordinates_full` overlaps with `source_crop_xywh[:, :2]`, but it keeps
  compatibility with existing crop/keypoint/mask placement code.
- `bbox_img_xyxy` is the least ambiguous detection geometry for debugging,
  painting, QC, and crop-local projection.
- `bbox_norm_coords` is the canonical training/export mirror of the same
  full-frame detection box.
- `bbox_roi_xyxy` is useful for crop-local prompts and review overlays without
  requiring parent frame dimensions.

The array we should avoid is an unqualified local-normalized
`bbox_norm_coords`. If retained, local-normalized values need a local-qualified
name.

## Derivation Rules

When crop-video pixels are one-to-one with `source_crop_xywh` dimensions:

```text
full_x = source_crop_x + roi_x
full_y = source_crop_y + roi_y
```

When the decoded crop-video frame has dimensions `(roi_w, roi_h)` that differ
from `source_crop_xywh` dimensions `(crop_w, crop_h)`:

```text
full_x = source_crop_x + roi_x * (crop_w / roi_w)
full_y = source_crop_y + roi_y * (crop_h / roi_h)
```

Then:

```text
bbox_img_xyxy = [full_x0, full_y0, full_x1, full_y1]
bbox_norm_coords = xyxy_to_cxcywh(bbox_img_xyxy) / [frame_w, frame_h, frame_w, frame_h]
bbox_roi_xyxy = project full-frame box into local crop pixels
```

The parent full-frame dimensions must come from a trusted source such as
`raw_video` attrs, video metadata, or recording metadata. If they cannot be
resolved, a writer should not emit canonical `bbox_norm_coords`.

## Migration Plan

1. Detect affected crop runs by attr:

   ```text
   bbox_norm_coords_semantics =
   "realtime_detection_bbox_xywh_normalized_to_crop_video_frame"
   ```

   or:

   ```text
   bbox_norm_coords_semantics =
   "pose_bbox_from_keypoint_extents_xywh_normalized_to_crop_video_frame"
   ```

2. Preserve existing local-normalized values as `bbox_crop_norm_coords`.

3. Derive `bbox_img_xyxy` from `bbox_roi_xyxy` plus `source_crop_xywh` and the
   crop frame dimensions.

4. Derive canonical full-frame-normalized `bbox_norm_coords` from
   `bbox_img_xyxy` and parent frame dimensions.

5. Update run attrs:

   ```text
   bbox_norm_coords_semantics = "bbox_xywh_normalized_to_full_frame"
   bbox_img_xyxy_semantics = "bbox_xyxy_full_frame_pixels"
   bbox_roi_xyxy_semantics = "bbox_xyxy_crop_roi_pixels"
   bbox_crop_norm_coords_semantics = "bbox_xywh_normalized_to_crop_roi_frame"
   ```

6. Update writers so new acquisition crop-video crop runs use the repaired
   contract from the start.

7. Update consumers so any code that requires canonical boxes rejects
   crop-frame-normalized `bbox_norm_coords` instead of silently projecting them
   as full-frame boxes.

## Implemented Repair Utility

The repair/inventory utility is:

```bash
scripts/py -m fisheye.utils.repair_acquisition_crop_bbox_contract
```

Single-Zarr dry-run:

```bash
scripts/py -m fisheye.utils.repair_acquisition_crop_bbox_contract \
  /path/to/recording_training.zarr \
  --output-json /tmp/crop_bbox_repair_dryrun.json
```

Registry-scoped RedScare training dry-run:

```bash
scripts/py -m fisheye.utils.repair_acquisition_crop_bbox_contract \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --zarr-use training \
  --path-contains RedScare \
  --output-json /tmp/redscare_training_bbox_repair_registry_dryrun.json
```

Apply uses the same command with `--apply`.

Observed RedScare registry dry-run before repair on 2026-06-28:

```json
{
  "status": "ok",
  "zarr_count": 28,
  "affected_crop_run_count": 28,
  "changed_crop_run_count": 28,
  "blocked_crop_run_count": 0,
  "failed_zarr_count": 0
}
```

Approved RedScare registry apply on 2026-06-28:

```json
{
  "status": "ok",
  "zarr_count": 28,
  "affected_crop_run_count": 28,
  "changed_crop_run_count": 28,
  "blocked_crop_run_count": 0,
  "failed_zarr_count": 0
}
```

Post-apply RedScare registry dry-run on 2026-06-28:

```json
{
  "status": "ok",
  "zarr_count": 28,
  "affected_crop_run_count": 0,
  "changed_crop_run_count": 0,
  "blocked_crop_run_count": 0,
  "failed_zarr_count": 0
}
```

Full registry read-only inventory after the RedScare repair on 2026-06-28:

```json
{
  "status": "ok",
  "zarr_count": 214,
  "affected_crop_run_count": 0,
  "changed_crop_run_count": 0,
  "blocked_crop_run_count": 0,
  "failed_zarr_count": 0
}
```

This means the active registry-visible crop-video surfaces no longer expose
crop-frame-normalized values through canonical `bbox_norm_coords`.

## Consumer Rule

Consumers should be able to treat every crop run as:

```text
local crop pixels + full-frame placement geometry + row lineage
```

They may inspect `source_pixels` when exact pixel provenance matters. They
should not need to inspect `source_pixels` to interpret `bbox_norm_coords`.

## Acceptance Criteria

- New acquisition crop-video crop runs write full-frame-normalized
  `bbox_norm_coords`.
- Existing affected crop-video runs can be backfilled without changing row
  identity.
- SAM and related crop-local prompt paths use `bbox_roi_xyxy` or
  `bbox_crop_norm_coords` for local boxes, not ambiguous `bbox_norm_coords`.
- Full-frame detection/review/training consumers can continue using
  `bbox_norm_coords` without source-specific branching.
- Docs and tests state that unqualified `bbox_norm_coords` is canonical
  full-frame-normalized geometry.
