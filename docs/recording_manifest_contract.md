# Recording Manifest Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
-->

This document defines the minimum metadata contract for `recording_manifest.json`
files that are later ingested into the palette registry.

## Purpose

- Acquisition only transfers recording folders.
- Registry ingestion/backfill happens later.
- This manifest contract ensures recordings can be normalized and validated
  consistently during `--backfill-recording-entities` and `--check-integrity`.

## Required Fields

These keys should always be present in `recording_manifest.json`:

- `recording_type`
- `recording_subtype`
- `behavior_mode`
- `artifact_schema_id`

Recommended additional keys (already used by backfill when present):

- `session_uuid`
- `recording_name`
- `session_start_iso8601_utc`
- `rig_id`
- `arena_id`
- `camera_id`
- `canvas_name`
- `protocol_name_from_definition`

## Controlled Vocabulary

### `recording_type`

- `behavior`
- `microscopy`
- `histology`

### `recording_subtype` by `recording_type`

- `behavior`: `free`, `embedded`
- `microscopy`: `lightsheet`, `confocal`, `2p`
- `histology`: `section`, `wholemount`

### `behavior_mode`

- `free`
- `embedded`
- `none`

Constraint:
- If `recording_type="behavior"`, then `recording_subtype` must equal `behavior_mode`.

## Artifact Schema

Example known schema:

- `artifact_schema_id="behavior_v1"`

For `behavior_v1`, integrity currently checks required artifact types:

- `h5_log`
- `camera_video`
- `camera_metadata_csv`
- `timing_profile_csv`

## Zarr Artifact Naming Convention

To avoid mixing training and exploratory analysis runs in the same artifact:

- Training source Zarr: `<recording_base>_training.zarr`
- Analysis working Zarr: `<recording_base>_analysis.zarr`

Notes:
- `*_training.zarr` should remain stable and curated for training eligibility.
- `*_analysis.zarr` may accumulate multiple derived runs (detect/keypoint/metrics).
- `zarr_purpose` should match suffix intent:
  - `*_training.zarr` -> `training`
  - `*_analysis.zarr` -> `analysis`
- `session_uuid` may be shared across both artifacts; `dataset_id` must be unique per artifact path.
- Stage lineage/provenance attrs are expected in both archive types when stage outputs are present.
  - This is required for reproducible, versioned training datasets.
  - Example (eye masks): runs should carry keypoint lineage (`source_keypoints_run`, `source_keypoint_group`) and crop lineage (`source_crop_run`).
  - Example (refined eye masks): runs should also carry `source_eye_masks_run` and `source_eye_masks_method` when available.
  - Legacy alias `source_keypoint_run` may still exist in older runs; backfill to canonical `source_keypoints_run` is expected.

## Example Manifests

### Behavior, freely swimming

```json
{
  "session_uuid": "2026-01-28T22-42-59Z_arena_1",
  "recording_name": "2026-01-28T22-42-59Z_arena_1_DefaultScreen",
  "session_start_iso8601_utc": "2026-01-28T22:42:59+00:00",
  "recording_type": "behavior",
  "recording_subtype": "free",
  "behavior_mode": "free",
  "artifact_schema_id": "behavior_v1",
  "rig_id": "omnifin0",
  "arena_id": "arena_1",
  "camera_id": "2010093",
  "canvas_name": "DefaultScreen",
  "protocol_name_from_definition": "DefaultScreen"
}
```

### Microscopy with embedded behavior

```json
{
  "session_uuid": "2026-03-15T10-22-11Z_microscopy_01",
  "recording_name": "2026-03-15T10-22-11Z_microscopy_01",
  "session_start_iso8601_utc": "2026-03-15T10:22:11+00:00",
  "recording_type": "microscopy",
  "recording_subtype": "lightsheet",
  "behavior_mode": "embedded",
  "artifact_schema_id": "behavior_v1",
  "rig_id": "scope_rig_01",
  "arena_id": "scope_stage_a",
  "camera_id": "scope_cam_01",
  "canvas_name": "MicroscopyEmbedded",
  "protocol_name_from_definition": "MicroscopyEmbeddedBehavior"
}
```

## Post-Transfer Validation Workflow

After recording folders are copied to storage:

1. Validate manifests before registry writes:

```bash
scripts/py -m fisheye.utils.validate_recording_manifest \
  /nvme1/recordings \
  --recursive \
  --registry /nvme1/palette_registry.sqlite
```

If legacy manifests are missing required fields, patch defaults then revalidate:

```bash
scripts/py -m fisheye.utils.validate_recording_manifest \
  /nvme1/recordings \
  --recursive \
  --registry /nvme1/palette_registry.sqlite \
  --apply-defaults
```

2. Backfill recording entities:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry /nvme1/palette_registry.sqlite \
  --backfill-recording-entities
```

3. Run integrity checks:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry /nvme1/palette_registry.sqlite \
  --check-integrity --list-limit 100
```

4. Inspect distribution and allowed vocab:

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry /nvme1/palette_registry.sqlite \
  --recording-summary
```
