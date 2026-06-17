# Orange Runtime Video Artifact Contract

Last verified: 2026-06-16

This document defines how Palette organizes Orange `external_ipc` video artifacts.
It covers sessions that produce both a merged full-frame camera video and an
acquisition-time cropped video stream.

## Scope

Orange may write many runtime artifacts while recording. Palette keeps only the
artifacts needed to reproduce ingestion context, decode the authoritative video
streams, and audit acquisition/runtime decisions. GPU shard outputs used by
Orange during full-frame recording are not canonical Palette inputs.

## Canonical Streams

Each organized external-IPC recording may expose two video streams in
`recording_manifest.json` under `video_streams`.

### Full-Frame Stream

The full-frame stream is the ingest-authoritative camera video.

Required organized paths:

- `cams/Cam<camera>_<session>.mp4`
- `cams/Cam<camera>_<session>_meta.csv`
- `cams/Cam<camera>_<session>_keyframe.json`
- `cams/Cam<camera>_<session>_external_summary.json`

Contract:

- `role`: `ingest_authoritative_full_frame`
- `frame_clock`: `recording_frame_id`
- `coordinate_space`: full-frame camera pixels when Orange provides it
- The compatibility `Cam*_meta.csv` may be copied from the crop metadata table
  when it shares the same `recording_frame_id` and timestamp clock.

### Runtime Crop Stream

The crop stream is a first-class acquisition-time derived input. It is not a
debug preview. It may be useful for future workflows that intentionally consume
runtime-selected crop pixels instead of re-cropping offline.

Required organized paths:

- `derived/external_crop_recorder/Cam<camera>_<session>_crop_external.mp4`
- `derived/external_crop_recorder/Cam<camera>_<session>_crop_meta.csv`
- `derived/external_crop_recorder/Cam<camera>_<session>_crop_external_keyframe.json`
- `derived/external_crop_recorder/Cam<camera>_<session>_crop_external_summary.json`

Contract:

- `role`: `runtime_derived_acquisition_input`
- `frame_clock`: `recording_frame_id`
- `video_pixel_coordinate_space`: `crop_frame_pixels`
- `source_geometry_coordinate_space`: `full_frame_pixels`
- Geometry columns in `crop_meta.csv` are full-frame coordinates:
  `crop_x`, `crop_y`, `crop_w`, `crop_h`, `detection_x`, `detection_y`,
  `detection_w`, `detection_h`
- `has_detection` identifies frames with a selected detection.
- `blank_frame` identifies crop-video frames encoded as a blank placeholder.
- `blank_frame_policy` and `selection_policy` should come from
  `recording_session.json` when Orange writes them.

Consumers must not treat crop-video pixel coordinates as full-frame coordinates.
Use `crop_meta.csv` geometry to map crop-local measurements back to the
full-frame camera image.

## Canonical Session Context

The organizer preserves session-level acquisition context in `raw/`:

- `recording_session.json`
- `recording_snapshot_runtime.json`
- `ptp_sync_summary.json`
- `transfer_complete.json`
- `orange_local_control.events.jsonl`
- `external_recorder_contract.json`
- `external_crop_recorder_contract.json`
- `external_recorder_supervisor_plan.json`
- `external_crop_recorder_supervisor_plan.json`

The organizer preserves Citrus runtime startup context under `derived/citrus/`:

- `*threading_startup*.json`

The organizer preserves recorder runtime diagnostics under:

- `derived/external_recorder/`
- `derived/external_crop_recorder/`

These diagnostics are retained for auditability, but stream identity and file
placement should be read from `recording_manifest.json`.

## Noncanonical Debug Artifacts

Palette intentionally does not copy full-frame recorder shard artifacts into
organized recording folders:

- `*_shard*_gpu*.mp4`
- `*_encode_shard*.csv`
- `*_keyframes_shard*.json`

Those files are Orange implementation/debug outputs. Orange may keep, rotate, or
delete them independently after the merged full-frame stream and its summaries
have been produced.

## Manifest Shape

`recording_manifest.json` should include:

```json
{
  "artifact_schema_id": "orange_external_ipc_single_clip_v1",
  "recording_backend": "external_ipc",
  "video_streams": {
    "schema_id": "orange_runtime_video_streams_v1",
    "frame_clock": "recording_frame_id",
    "streams": {
      "full": {
        "role": "ingest_authoritative_full_frame",
        "output_kind": "full",
        "video": "cams/Cam2010093_session.mp4"
      },
      "crop": {
        "role": "runtime_derived_acquisition_input",
        "output_kind": "crop",
        "video": "derived/external_crop_recorder/Cam2010093_session_crop_external.mp4",
        "metadata": "derived/external_crop_recorder/Cam2010093_session_crop_meta.csv",
        "video_pixel_coordinate_space": "crop_frame_pixels",
        "source_geometry_coordinate_space": "full_frame_pixels"
      }
    }
  }
}
```

The manifest paths are relative to the organized recording root.

## Rolling Clip Future

Rolling crop/full-frame clips should remain children of one
recording/session. They should not become separate recording rows unless the
acquisition itself created separate biological recordings. A future clip index
should reference both full-frame and crop-stream artifacts by clip id, camera id,
stream role, and clip-local frame range.
