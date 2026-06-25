# Orange Crop-Video Frame Contract Request

## Summary

Palette found a frame-domain ambiguity in RedScare acquisition crop metadata.
The crop video itself is frame-indexed by metadata row order, but the CSV field
named `local_frame_id` is not the crop MP4 frame index.

This caused Palette's first RedScare training import smoke to decode crop video
frames `140112, 140811, ...` from a crop MP4 that only has `139908` frames.

## Concrete Example

Recording:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare
```

Crop metadata:

```text
derived/external_crop_recorder/Cam2010093_2026-06-23T16-01-09Z_arena_1_crop_meta.csv
```

Observed columns:

```text
recording_frame_id: 1..139908
local_frame_id:    5205..145112
camera_frame_id:   wraps/uses a camera clock
CSV row index:     0..139907
crop MP4 frames:   0..139907
```

The crop MP4 appears to align with CSV row order:

```text
crop_video_frame_index = csv_row_index
parent_frame_index = recording_frame_id - 1
```

`local_frame_id` should therefore be treated as acquisition/camera provenance,
not as a zero-based frame index into the crop MP4.

## Requested Orange Contract Change

Please add an explicit crop-video frame column to crop metadata:

```text
crop_video_frame_index
```

Required semantics:

```text
zero-based frame index in the encoded crop MP4;
row i should normally have crop_video_frame_index == i
```

Please keep or clarify these existing fields:

```text
recording_frame_id
```

The global recording frame clock. Current RedScare data uses 1-based values, so
Palette maps to zero-based parent frame as `recording_frame_id - 1`.

```text
camera_frame_id
```

Camera/acquisition frame clock, useful for provenance.

```text
local_frame_id
```

If retained, document it as an Orange/acquisition-local frame identifier, not as
the crop MP4 frame index.

## Palette Compatibility Behavior

Palette will support current data by falling back to:

```text
source_crop_video_frame_indices = source_crop_meta_row_indices
```

Palette will continue preserving:

```text
source_crop_local_frame_ids = local_frame_id
```

as provenance.

Once Orange emits `crop_video_frame_index`, Palette should prefer that column and
only use row-index fallback for older recordings.

## Why This Matters

Palette training zarrs need frame-perfect lineage among:

```text
raw_video/original_frame_indices
crop_meta recording_frame_id
crop MP4 frame index
crop geometry
runtime detection bbox
future pose/mask labels
```

Overloading `local_frame_id` makes this ambiguous and can decode the wrong crop
pixels or fail when the acquisition-local frame clock exceeds crop-video length.
