# Zarr Split Policy (Training vs Production)

This document defines how we store training imports vs production inference
outputs to keep metadata consistent and provenance checks meaningful.

## Why split?

Training imports often:
- downsample frames
- sample every Nth frame
- omit frames near the tail

Production inference typically runs on the *full* video stream.

If we store both in a single Zarr, `total_frames`, `fps`, and provenance
become ambiguous. Consistency checks will also fail because counts no longer
refer to a single frame universe. Splitting avoids these contradictions.

## Policy

Use **separate Zarrs** for training vs production:

### Training Zarr
- Created by import pipeline (frames stored in Zarr).
- Contains: `raw_video/images_*`, background models, tuning, crops, keypoints.
- May be sampled or downsampled.
- Runs: detect/refine/crop/keypoints are tied to the imported frame universe.

### Production Zarr
- Created by `detect_yolo` (no frame import).
- Contains: detect/refine outputs and metadata-only `raw_video` attrs.
- No `raw_video/images_*` arrays.
- `raw_video` group is metadata-only and marked:
  - `import_method=metadata_only`
  - `import_mode=metadata_only`
  - `has_raw_video=False` at the root

## Linking the two

We link training and production Zarrs via:

- `session_uuid` (if present)
- `source_video_path` / `source_video`
- optional `source_zarr_path` on production Zarrs (points to the training Zarr)

The registry can use these fields to connect datasets without mixing frame
universes.

## What this enables

- Clean provenance checks (counts align within each Zarr)
- Stable metadata in the registry
- Independent training and production workflows

## Future extension (optional)

If we later need to embed multiple frame universes in one archive, we should
introduce a dedicated schema (e.g., `video_sources/<id>` with per-source
metadata and explicit links from each run).
