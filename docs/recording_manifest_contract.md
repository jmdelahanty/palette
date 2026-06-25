# Recording Manifest Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-06-16
-->

This document defines the minimum metadata contract for `recording_manifest.json`
files that are later ingested into the palette registry.

## Purpose

- Acquisition only transfers recording folders.
- Registry ingestion/backfill happens later.
- This manifest contract ensures recordings can be normalized and validated
  consistently during `--backfill-recording-entities` and `--check-integrity`.

## Biological Identity Naming

`recording_manifest.json` is primarily a recording-context contract. It does not
define the canonical biological subject namespace for the registry.

Rules:

- canonical registry biological identity is `subject_id`
- legacy acquisition/snapshot metadata may still use `fish_id`
- if source Zarr or provenance snapshot metadata provides `fish_id`, registry
  normalization should map that value into `subject_id`
- `fish_id` remains a compatibility/source name during migration, not a second
  canonical registry identity namespace

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

## Optional Preflight Section

`recording_manifest.json` may include a machine-written `preflight` object. This
is written by `fisheye.utils.organize_recordings` when post-organize
diagnostics hooks run.

Example shape:

```json
{
  "preflight": {
    "status": "warn",
    "checked_at_utc": "2026-04-17T22:49:40.614931+00:00",
    "video": {
      "status": "warn",
      "media_status": "pass",
      "tooling_status": "error",
      "videos_scanned": 2,
      "finding_codes": ["video.opencv_unavailable"]
    },
    "h5": {
      "status": "pass",
      "core_status": "pass",
      "optional_status": "pass",
      "tooling_status": "pass",
      "finding_codes": []
    }
  }
}
```

Semantics:

- `preflight.status` is the combined stored verdict.
- New manifests start with `preflight.status="not_run"`, `checked_at_utc=null`,
  and empty video/H5 sections. That means organize wrote the manifest but did
  not run diagnostics hooks yet.
- `organize_recordings` updates this object only when run with
  `--run-video-diagnostics`, `--run-h5-diagnostics`, or both.
- `video.status` summarizes the unified raw-video preflight.
- `h5.status` summarizes the unified H5 preflight.
- `video.media_status` and `h5.core_status` are the strongest import-relevant
  fields.
- tooling-only problems may produce `warn` without indicating bad media or an
  unimportable H5.
- Automatic video preflight uses the OpenCV decode smoke by default. Decord
  remains available as an explicit manual diagnostic backend, but it is not the
  default gate for current imports.
- Current Citrus/Orange frame metadata uses `timestamp_ns_epoch` in
  `/video_metadata/frame_metadata`; legacy `timestamp_ns`,
  `timestamp_ns_session`, and `relative_timestamp_ns` are compatibility aliases.
- Current camera metadata CSVs use `recording_frame_id`; legacy `frame_id` and
  `local_frame_id` are compatibility aliases when validating older sidecars.

Downstream import commands currently block only when `preflight.status=fail`,
and allow `warn` by default. Commands that enforce this gate expose an explicit
`--allow-preflight-failures` override.

## Optional Analysis Import Status

`recording_manifest.json` may include machine-written fields that describe the
folder-local analysis import state. These fields are intentionally duplicated
from central registry state so a copied recording folder remains self-describing.

Recommended shape after a successful analysis import:

```json
{
  "import_status": "ok",
  "import_log": "/groups/.../import_organized_recordings_analysis_20260621T231809Z_746229.jsonl",
  "imported_at_utc": "2026-06-21T23:18:53.512193+00:00",
  "analysis_zarr_path": "/groups/.../zarr/<recording>_analysis.zarr",
  "analysis_import": {
    "status": "ok",
    "zarr_path": "/groups/.../zarr/<recording>_analysis.zarr",
    "import_log": "/groups/.../import_organized_recordings_analysis_20260621T231809Z_746229.jsonl",
    "imported_at_utc": "2026-06-21T23:18:53.512193+00:00",
    "run_id": "20260621T231809Z_746229",
    "registry_path": "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite",
    "registry_dataset_id": "dataset_..."
  }
}
```

Semantics:

- `import_status="ok"` means the analysis zarr exists and the import wrapper
  reached `recording_ok`, or an existing analysis zarr was verified during a
  registry-sync repair pass.
- `imported_at_utc` should be the `recording_ok.ts_utc` timestamp when
  backfilled from an original import JSONL. For a new import run, it is written
  by `fisheye.utils.import_organized_recordings_analysis`.
- `analysis_zarr_path` is folder-local provenance only. The registry remains the
  query authority for cross-recording workflows.
- `registry_dataset_id` and `registry_path` are optional and only written when a
  registry sync ran successfully.

Backfill from existing import logs:

```bash
scripts/py -m fisheye.utils.recording_manifest_import_status \
  --file-list /path/to/import_logs.txt \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --apply
```

Refresh preflight diagnostics for already-organized recordings:

```bash
scripts/py -m fisheye.utils.refresh_recording_preflight \
  /groups/johnson/johnsonlab/jeremy/recordings \
  --recursive \
  --apply
```

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
- `artifact_schema_id="video_only_v1"`
- `artifact_schema_id="orange_external_ipc_single_clip_v1"`
- future: `artifact_schema_id="orange_rolling_clips_v1"`

For `behavior_v1`, integrity currently checks required artifact types:

- `h5_log`
- `camera_video`
- `camera_metadata_csv`
- `timing_profile_csv`

For `video_only_v1`, the recording may be imported from MP4 without an H5.
Recommended manifest content:

- include the same recording identity fields when known (`session_uuid`,
  `recording_name`, `rig_id`, `arena_id`, `camera_id`, `canvas_name`,
  `protocol_name_from_definition`)
- include `dish_design` when manually known
- include at least one `camera_video` entry under `files.cams`
- when available, keep the camera frame table and keyframe summary beside the
  video under `files.cams` as the camera stream bundle: `Cam*.mp4`,
  `Cam*_meta.csv`, `Cam*_keyframe.json`
- keep session-level raw context in `files.raw` (`ptp_sync_summary.json`,
  `recording_snapshot_runtime.json`) and secondary diagnostics/repair backups
  in `files.derived`

`video_only_v1` does not currently trigger the `behavior_v1` required-artifact
integrity check, so missing H5/CSV sidecars are tolerated.

For `orange_external_ipc_single_clip_v1`, the recording comes from an Orange
`external_ipc` batch with Citrus H5 context plus an ingest-authoritative
full-frame external recorder video. Runtime crop videos are first-class
acquisition-time derived inputs and are described separately from the full-frame
stream. Recommended manifest content:

- include the normal recording identity fields from the Citrus H5
- include `recording_backend="external_ipc"`, `orange_session_id`,
  `orange_producer`, and `orange_recording_mode` when known
- include `video_streams.schema_id="orange_runtime_video_streams_v1"` with
  separate `full` and `crop` stream entries when Orange wrote both streams
- analysis-zarr import mirrors this `video_streams` payload into
  `analysis/acquisition_video_streams` as a recording-level acquisition media
  inventory; this does not make crop videos Palette `crop_runs`
- include one full-frame camera video entry under `files.cams`
- include compatibility camera sidecars under `files.cams`:
  `Cam*.mp4`, `Cam*_meta.csv`, and `Cam*_keyframe.json`
- the compatibility `Cam*_meta.csv` may be copied from the crop metadata table
  when that table shares the same `recording_frame_id` / timestamp clock as the
  full-frame video
- preserve cropped video and crop-native sidecars under
  `files.derived` / `derived/external_crop_recorder/`
- declare crop stream pixels as `video_pixel_coordinate_space="crop_frame_pixels"`
  and crop metadata geometry as
  `source_geometry_coordinate_space="full_frame_pixels"`
- preserve acquisition/session context in `files.raw`: `recording_session.json`,
  `recording_snapshot_runtime.json`, `ptp_sync_summary.json`,
  `transfer_complete.json`, `orange_local_control.events.jsonl`, recorder
  contracts, and recorder supervisor plans
- preserve Citrus startup diagnostics under
  `files.derived` / `derived/citrus/`
- preserve external recorder diagnostics under
  `files.derived` / `derived/external_recorder/`
- do not include shard intermediates such as `*_shard*_gpu*.mp4`,
  `*_keyframes_shard*.json`, or `*_encode_shard*.csv`

See
[`docs/orange_runtime_video_artifact_contract.md`](orange_runtime_video_artifact_contract.md)
for the full stream and artifact retention contract.

Palette organizer metadata sources for this schema:

- H5 root attrs provide recording identity and acquisition context:
  `session_uuid`, `session_start_iso8601_utc`, `rig_id`, `arena_id`,
  `camera_id` when present, `canvas_name`, `protocol_name_from_definition`,
  `loaded_protocol_filepath`, IPC source fields, `hostname`, and optional
  root-level biology/dish fields when Orange writes them.
- If H5 root `camera_id` is absent, the organizer derives `camera_id` from
  `ipc_source_name` patterns such as `/shm_cam_2010093`.
- `protocol_name` is lifted from `protocol_snapshot.attrs["protocol_name"]`
  or `protocol_snapshot/protocol_definition_json.protocol_name` when the root
  attrs do not provide it.
- `dish_design` is lifted from
  `calibration_snapshot/arena_config_json.selected_dish_type_name`, falling
  back to `dish_config.dish_name` when available.
- `genotype` is lifted from `subject_metadata.attrs["genotype"]`.
- `dpf_at_acquisition` is lifted from
  `subject_metadata.attrs["days_post_fertilization"]` or
  `subject_metadata.attrs["dpf_at_acquisition"]`.
- `software_version` uses the H5 root attr when populated; otherwise, for
  Orange external-IPC batches, it falls back to
  `recording_snapshot.json.source_version.describe`, then
  `source_version.commit_short`, then `producer_version`.
- `num_dishes` and `fish_per_dish` are not inferred from
  `subject_metadata.fish_count` or `subject_count`. Those fields have different
  possible meanings across recordings, so Orange or an operator metadata layer
  must write them explicitly before Palette treats them as manifest facts.

Existing manifests can be refreshed without re-organizing the recording:

```bash
scripts/py -m fisheye.utils.refresh_recording_manifest_metadata \
  /path/to/recordings/root \
  --recursive \
  --refresh-external-ipc-artifacts
```

The refresh tool is dry-run by default. Add `--apply` after reviewing the field
diffs. By default it only fills empty fields from H5/runtime metadata and
preserves non-empty manifest values; `--overwrite-existing` is available for
deliberate repairs. With `--refresh-external-ipc-artifacts`, it also backfills
external-IPC `video_streams`, copies small retained session/control artifacts
from the manifest `source_dir`, and adds their relative paths to `files`. It
does not copy, move, or delete shard debug outputs.

`orange_external_ipc_single_clip_v1` is an organizer-side compatibility layout
for single-clip external IPC sessions. It is not the rolling-clip collection
contract.

For future Orange rolling-clip recordings, keep clips as children of one
recording/session rather than separate recording rows. The authoritative clip
inputs are `recording_session.json`, `recording_clip_index.{json,csv}`, each
`clips/clip_%06d/clip_manifest.json`, and each per-clip camera bundle
(`Cam*.mp4`, `Cam*_meta.csv`, `Cam*_keyframe.json`). Per-clip metadata row
order is the clip-local frame index; native Orange `Cam*_meta.csv` `frame_id`
is the session-continuous `recording_frame_id`. See
[`docs/orange_rolling_clip_recording_contract.md`](orange_rolling_clip_recording_contract.md).

### Video-only Organizer CSV

`fisheye.utils.organize_recordings --video-only --metadata-csv` consumes an
operator-authored CSV before it writes `recording_manifest.json`. This CSV is an
intake aid, not a persisted contract artifact. Its schema is documented in
[`docs/operator_guide/organize_recordings.md`](operator_guide/organize_recordings.md#video-only-batches-without-h5).

Use `fisheye.utils.draft_video_only_organizer_manifest` to draft this CSV from a
staging directory. The helper can fill discoverable values from `Cam*.mp4`,
`Cam*_meta.csv`, and `recording_snapshot.json`; operator-known values such as
`dish_design`, `genotype`, `dpf_at_acquisition`, `num_dishes`, and
`fish_per_dish` should be supplied by flags or edited in the CSV before apply.

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

### Video-only manual intake

`cams/` is the first-class single-file camera layout: one MP4 plus frame
metadata/keyframe sidecars per camera. This remains the preferred representation
for short or moderate recordings where a single video per camera is easy to
manage. Rolling `clips/` are a separate first-class layout for long recordings
or cluster-parallel workflows, not a replacement that deprecates `cams/`.

```json
{
  "session_uuid": "2026-03-09_colleague_set_001",
  "recording_name": "colleague_set_001",
  "recording_type": "behavior",
  "recording_subtype": "free",
  "behavior_mode": "free",
  "artifact_schema_id": "video_only_v1",
  "dish_design": "cedar",
  "rig_id": "omnifin0",
  "arena_id": "arena_1",
  "camera_id": "2010093",
  "protocol_name_from_definition": "ManualProtocol",
  "files": {
    "cams": [
      "cams/Cam2010093.mp4",
      "cams/Cam2010093_meta.csv",
      "cams/Cam2010093_keyframe.json"
    ],
    "raw": [
      "raw/ptp_sync_summary.json",
      "raw/recording_snapshot_runtime.json"
    ],
    "derived": [
      "derived/Cam2010093_pipeline_perf.csv",
      "derived/Cam2010093_acquisition_cadence_probe.csv"
    ]
  }
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
