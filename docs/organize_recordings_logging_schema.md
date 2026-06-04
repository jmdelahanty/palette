# Organize Recordings: Logging Schema

This document describes the JSONL logging produced by
`src/fisheye/utils/organize_recordings.py`.

## Overview
- **Format**: JSON Lines (one JSON object per line).
- **Default log directory**: `<dest-root>/logs/organize_recordings/`
- **File name**: `organize_recordings_<YYYYMMDDThhmmssZ>_<pid>.jsonl`
- **Log controls**:
  - Override directory: `--log-dir /path/to/logs`
  - Environment variable: `PALETTE_LOG_ROOT=/path/to/logs`

## Environment Variables

These are the recommended variables to set in your shell or job environment:

- `PALETTE_STAGING_ROOT`  
  Default source root when no positional `source` is provided.
- `PALETTE_RECORDINGS_ROOT`  
  Used by `scripts/organize_staging.sh` to set `--dest-root`.
- `PALETTE_LOG_ROOT`  
  Overrides the log directory for JSONL logs.

Example (bash):
```bash
export PALETTE_STAGING_ROOT=/nvme1/staging
export PALETTE_RECORDINGS_ROOT=/nvme1/recordings
export PALETTE_LOG_ROOT=/nvme1/recordings/logs/organize_recordings
```

Each log line includes:
- `event`: event type string.
- `ts_utc`: ISO‑8601 UTC timestamp.
- `run_id`: identifier for this run (timestamp + PID).

## Event Types

### run_start
Emitted once at the beginning of the run.
```json
{
  "event": "run_start",
  "ts_utc": "2026-02-01T12:00:00+00:00",
  "run_id": "20260201T120000Z_12345",
  "source_root": "/nvme1/staging",
  "process_all": true,
  "recursive": true,
  "dest_root": "/nvme1/recordings",
  "rename_cams": true,
  "write_manifest": true,
  "snapshot_mode": "split",
  "video_only": false,
  "external_ipc": false,
  "metadata_csv": null,
  "run_video_diagnostics": false,
  "run_h5_diagnostics": false
}
```

### batch_start
Emitted at the start of each staging batch (when `--process-all` is set).
```json
{
  "event": "batch_start",
  "ts_utc": "2026-02-01T12:00:01+00:00",
  "run_id": "20260201T120000Z_12345",
  "batch_source": "/nvme1/staging/2026_01_28_14_36_16"
}
```

### recording_plan
Emitted for each recording in the plan.
```json
{
  "event": "recording_plan",
  "ts_utc": "2026-02-01T12:00:02+00:00",
  "run_id": "20260201T120000Z_12345",
  "recording_name": "2026-01-28T19-36-18Z_arena_1_Feeding",
  "session_uuid": "2026-01-28T19-36-18Z_arena_1",
  "camera_id": "2010093",
  "missing": [],
  "raw_files": [
    "2026-01-28T19-36-18Z_arena_1_Feeding.h5",
    "2026-01-28T19-36-18Z_arena_1_Feeding.mp4",
    "2026-01-28T19-36-18Z_arena_1_Feeding_update_timing.csv"
  ],
  "cam_files": [
    "Cam2010093_2026-01-28T19-36-18Z_arena_1.mp4",
    "Cam2010093_2026-01-28T19-36-18Z_arena_1_meta.csv"
  ],
  "derived_files": []
}
```

### file_moved
Emitted once for each file moved during `--apply`.
```json
{
  "event": "file_moved",
  "ts_utc": "2026-02-01T12:00:03+00:00",
  "run_id": "20260201T120000Z_12345",
  "recording_name": "2026-01-28T19-36-18Z_arena_1_Feeding",
  "session_uuid": "2026-01-28T19-36-18Z_arena_1",
  "source": "/nvme1/staging/2026_01_28_14_36_16/citrus/2026-01-28T19-36-18Z_arena_1_Feeding.h5",
  "dest": "/nvme1/recordings/2026-01-28T19-36-18Z_arena_1_Feeding/raw/2026-01-28T19-36-18Z_arena_1_Feeding.h5"
}
```

### snapshot_written
Emitted when `recording_snapshot.json` is attached.
```json
{
  "event": "snapshot_written",
  "ts_utc": "2026-02-01T12:00:04+00:00",
  "run_id": "20260201T120000Z_12345",
  "recording_name": "2026-01-28T19-36-18Z_arena_1_Feeding",
  "session_uuid": "2026-01-28T19-36-18Z_arena_1",
  "dest": "/nvme1/recordings/2026-01-28T19-36-18Z_arena_1_Feeding/derived/recording_snapshot.json"
}
```

### manifest_written
Emitted when `recording_manifest.json` is created.
```json
{
  "event": "manifest_written",
  "ts_utc": "2026-02-01T12:00:05+00:00",
  "run_id": "20260201T120000Z_12345",
  "recording_name": "2026-01-28T19-36-18Z_arena_1_Feeding",
  "session_uuid": "2026-01-28T19-36-18Z_arena_1",
  "organized_utc": "2026-02-01T12:00:05+00:00",
  "dest": "/nvme1/recordings/2026-01-28T19-36-18Z_arena_1_Feeding/recording_manifest.json"
}
```

### recording_applied
Emitted after a recording finishes processing (even if some files are missing).
```json
{
  "event": "recording_applied",
  "ts_utc": "2026-02-01T12:00:06+00:00",
  "run_id": "20260201T120000Z_12345",
  "recording_name": "2026-01-28T19-36-18Z_arena_1_Feeding",
  "session_uuid": "2026-01-28T19-36-18Z_arena_1",
  "camera_id": "2010093",
  "dest_dir": "/nvme1/recordings/2026-01-28T19-36-18Z_arena_1_Feeding",
  "moved_files": 3
}
```

### warning
Emitted for any warning (missing files, destination exists, snapshot issues, cleanup failures).
```json
{
  "event": "warning",
  "ts_utc": "2026-02-01T12:00:07+00:00",
  "run_id": "20260201T120000Z_12345",
  "recording_name": "2026-01-28T19-36-18Z_arena_1_Feeding",
  "session_uuid": "2026-01-28T19-36-18Z_arena_1",
  "message": "Missing source: /nvme1/staging/..."
}
```

### run_end
Emitted once at the end of the run.
```json
{
  "event": "run_end",
  "ts_utc": "2026-02-01T12:00:10+00:00",
  "run_id": "20260201T120000Z_12345"
}
```

## Query Examples

Find when a UUID was organized:
```bash
rg '\"session_uuid\": \"2026-01-28T19-36-18Z_arena_1\"' /nvme1/recordings/logs/organize_recordings
```

Find all warnings in a run:
```bash
rg '\"event\": \"warning\"' /nvme1/recordings/logs/organize_recordings/organize_recordings_20260201T120000Z_12345.jsonl
```
