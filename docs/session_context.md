# Session Context Metadata (Acquisition)

This document defines the **session context** fields captured at acquisition time
and how they should be mirrored into Zarr for downstream audit and training.

The intent is to make model training and deployment traceable to **rig/arena/canvas/camera**
configuration, which can change over time and affect model performance.

## Scope
- H5 **root attributes** written by acquisition software.
- Zarr `analysis_metadata` mirror.

## Required fields (root attributes)
- `session_uuid` (string)
- `session_start_iso8601_utc` (string)
- `rig_id` (string)
- `arena_id` (string)
- `camera_id` (string)  
  - Use the numeric camera serial id (e.g., `2010096`).
- `protocol_name_from_definition` (string)
- `loaded_protocol_filepath` (string)

## Recommended fields (root attributes)
- `canvas_name` (string)
  - The stimulus canvas or projection layout name used by the rig.
  - Suggested key: `canvas_name`.
- `stimulus_output_width` (int)
- `stimulus_output_height` (int)
- `ipc_source_name` (string)
- `active_ipc_source` (string)
- `hostname` (string)
- `software_version` (string)

## Zarr mirror
Mirror session context into `analysis_metadata.session_context` as a JSON blob.

Example:
```json
{
  "session_uuid": "2026-01-20T19-35-22Z_arena_1",
  "session_start_iso8601_utc": "2026-01-20T19:35:22Z",
  "rig_id": "omnifin0",
  "arena_id": "arena_1",
  "camera_id": "2010093",
  "canvas_name": "DefaultScreen",
  "camera_id_source": "ipc_source_name",
  "canvas_name_source": "protocol_name_from_definition",
  "protocol_name_from_definition": "DefaultScreen",
  "loaded_protocol_filepath": "group_screen.json",
  "stimulus_output_width": 1920,
  "stimulus_output_height": 1080
}
```

## Fallbacks / compatibility
- If `camera_id` is not explicitly provided, derive from `ipc_source_name`
  when possible (e.g., `/shm_cam_2010093` → `2010093`).
- If `canvas_name` is not provided, you may infer from protocol or use
  `protocol_name_from_definition` as a temporary placeholder.
- When inferred, record the source in `camera_id_source` or `canvas_name_source`.

## Notes
- Keep this metadata **PII‑free**.
- Session context is separate from **subject metadata** (`/subject_metadata`)
  and **zebrobot snapshot** data.
- Training pipelines should fail fast when datasets mix different contexts.

## Workflow
1) Acquisition writes H5 root attributes with session context fields.
2) Import or update the Zarr mirror:
   - `python -m fisheye.analysis.import_stimulus_to_zarr /path/to/session.zarr --h5 /path/to/stimulus.h5`
3) Verify the mirror:
   - `python src/zarr_inspector.py /path/to/session.zarr`
   - `python -m fisheye.diagnostics.inspect_session_context /path/to/session.zarr`
4) Register the dataset (optional but recommended):
   - `python -m fisheye.registry.scan /path/to/session.zarr`
   - `python -m fisheye.registry.status --list-issues`
