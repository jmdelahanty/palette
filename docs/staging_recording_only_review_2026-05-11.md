# Staging Recording-Only Review - 2026-05-11
<!-- contract-meta
status: inventory
last_verified: 2026-05-11
purpose: Read-only inventory of /nvme1/staging and review of the current recording-only organize/import/registry workflow.
-->

## Scope

This is a read-only review of the current staging directory. No files were
organized, imported, moved, copied, registered, or processed.

Reviewed staging root:

- `/nvme1/staging`

Current batch found:

- `/nvme1/staging/2026_05_05_17_45_30`
- total size: approximately `1.2T`
- no `.h5` files found
- no `TRANSFER_DONE` marker found in the top-level inventory

This is therefore a recording-only/video-only intake candidate, not a standard
Citrus H5 experiment import.

## Staging Inventory

The batch contains four camera videos plus per-camera metadata and diagnostics:

| Artifact family | Count | Notes |
| --- | ---: | --- |
| `Cam*.mp4` | 4 | HEVC camera videos, approximately 308 GB each |
| `Cam*_meta.csv` | 4 | per-frame camera metadata, header `frame_id,timestamp,timestamp_sys` |
| `Cam*_pipeline_perf.csv` | 4 | acquisition/pipeline performance telemetry |
| `Cam*_acquisition_cadence_probe.csv` | 4 | cadence/probe telemetry |
| `Cam*_keyframe.json` | 4 | keyframe summary, codec/fps/frame count |
| `recording_snapshot.json` | 1 | session/camera runtime snapshot |
| `ptp_sync_summary.json` | 1 | PTP/camera sync summary |
| `.h5` | 0 | no stimulus/protocol H5 source present |

Per-camera summary:

| Camera | MP4 bytes | Frame metadata rows | Codec | FPS | MP4/keyframe frames | Keyframes | Snapshot resolution | Exposure | Gain |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `2010093` | `307956344649` | `2163445` | `hevc` | 30 | `2163445` | `86538` | `4512x4512` | 100 | 256 |
| `2010094` | `307847859847` | `2163445` | `hevc` | 30 | `2163445` | `86538` | `4512x4512` | 50 | 256 |
| `2010095` | `308075797326` | `2163445` | `hevc` | 30 | `2163445` | `86538` | `4512x4512` | 100 | 256 |
| `2010096` | `307951720448` | `2163445` | `hevc` | 30 | `2163445` | `86538` | `4512x4512` | 50 | 256 |

Snapshot/session fields observed:

- `recording_id`: `2026_05_05_17_45_30`
- `timestamp_utc`: `2026-05-05T21:45:30Z`
- `producer_version`: `unknown`
- `session.full_frame_video_enabled`: `true`
- `session.recording_sink_mode`: `real`
- `sync.mode`: `ptp_local`
- `sync.camera_sync_enabled`: `true`
- `sync.num_cameras_expected`: `4`
- `sync.captured_at_utc`: `2026-05-05T21:45:30Z`

PTP summary reports `frame_count = 2166481` for each camera, while the MP4
keyframe summaries and per-frame metadata CSVs report `2163445` frames. Orange
source review clarified that `ptp_sync_summary.json` `frame_count` is
`camera_state.frame_count`: an acquisition-loop count of successfully received
camera frames since the camera acquisition stream started, not a recording-local
encoded-frame count. It can include frames before recording start or during
stop/drain/finalization.

Palette intake rule for this batch:

- Treat `Cam*_meta.csv` row count and `Cam*_keyframe.json.total_frames` as
  ingest-authoritative.
- If practical, validate MP4 probe/decode frame count against those encoded
  metadata counts.
- Treat `ptp_sync_summary.json.frame_count` as acquisition/sync telemetry only.
- Do not require PTP `frame_count` to match encoded-video frame count.
- Escalate real acquisition concerns on encoded metadata mismatch, noncontiguous
  recording frame IDs, decode/probe mismatch, dropped frames, get-frame errors,
  or large unexplained cross-camera cadence/timing problems.

## Current Organizer Behavior

Primary organizer:

- `src/fisheye/utils/organize_recordings.py`
- wrapper: `scripts/organize_staging.sh`
- operator doc: `docs/operator_guide/organize_recordings.md`

The standard path is H5-centered:

1. discover `.h5` files under a staging batch
2. read H5 root attrs to derive recording/session/camera context
3. pair each H5 with `Cam<camera_id>.mp4` and `Cam<camera_id>_meta.csv`
4. move files into `<recording>/raw`, `<recording>/cams`, `<recording>/derived`
5. write `recording_manifest.json`
6. optionally run video/H5 diagnostics and persist preflight status

This path will not process the current staging batch as-is because there are no
`.h5` files.

The organizer also has a video-only path:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  /nvme1/staging/2026_05_05_17_45_30 \
  --video-only \
  --metadata-csv <video_only_manifest.csv> \
  --dest-root /nvme1/recordings \
  --dry-run
```

Important video-only constraints in current code:

- `--metadata-csv` is required.
- `--process-all` is not supported with `--video-only`.
- `--require-done` is not supported with `--video-only`.
- `--run-h5-diagnostics` is not supported with `--video-only`.
- `--run-video-diagnostics` is supported, but only with `--apply`.
- The existing `Cam*_meta.csv` files are per-frame camera metadata, not the
  organizer metadata CSV. They do not contain the required `source_video` /
  `video_path` / `camera_video` column.
- Optional Orange sidecars are now preserved when present:
  `Cam<id>_keyframe.json` is moved to `cams/` beside the MP4 and frame metadata
  CSV because it describes encoded frame count and seek/keyframe structure;
  `Cam<id>_pipeline_perf.csv` and `Cam<id>_acquisition_cadence_probe.csv` are
  moved to `derived/`; shared
  `ptp_sync_summary.json` and `recording_snapshot.json` are copied into each
  recording's `raw/` as session-level recovery/telemetry files.
- Missing optional sidecars do not fail video-only organization.

The video-only organizer metadata CSV should provide one row per intended
recording. Required or useful columns include:

- `source_video` or `video_path` or `camera_video`
- optional `source_camera_metadata_csv` or `camera_metadata_csv`
- `camera_id`
- `session_uuid`
- `recording_id`
- `recording_name`
- `session_start_iso8601_utc`
- `recording_type`
- `recording_subtype`
- `behavior_mode`
- `dish_design`
- optional `rig_id`, `arena_id`, `genotype`, `dpf_at_acquisition`,
  `num_dishes`, `fish_per_dish`

Current code marks missing `dish_design` in video-only plans, so this field
should be explicit before applying organization.

## Recording Layout Decision

The current single-recording analysis pipeline is still single-camera oriented:
if a recording directory contains multiple `cams/*.mp4` files, single-recording
plan resolution fails unless `--video` is passed explicitly.

This batch should be represented as one organized recording directory per
camera. These are independent camera recordings in Palette's current processing
model, not one multi-camera 3D recording container. Use names such as:

- `sleepyfish_2026_05_05_17_45_30_cam2010093`
- `sleepyfish_2026_05_05_17_45_30_cam2010094`
- `sleepyfish_2026_05_05_17_45_30_cam2010095`
- `sleepyfish_2026_05_05_17_45_30_cam2010096`

That keeps each analysis Zarr single-camera and compatible with existing
detection, pose, segmentation, kinematics, and registry tooling. A true
multi-camera recording container remains a separate design problem and is not
needed for this batch.

## Analysis Import And Registry Path

Recording-only import is supported by:

- `src/fisheye/utils/import_recording_analysis.py`
- `src/fisheye/utils/run_recording_analysis_pipeline.py`
- `src/fisheye/utils/import_recordings_analysis.py`
- contract: `docs/recording_analysis_pipeline_contract.md`

For import-only archive creation after organization, use the single-recording
import helper:

```bash
scripts/py -m fisheye.utils.import_recording_analysis \
  --recording-dir /nvme1/recordings/<recording> \
  --recording-only \
  --dry-run
```

Later apply would create `<recording>/zarr/<recording>_analysis.zarr`, import
camera video metadata, and write root attrs including:

- `zarr_purpose = "analysis"`
- `experiment_context_status = "absent"`
- `experiment_context_source = "none"`
- `stimulus_runs_available = false`
- `recording_type = "behavior"`
- `recording_subtype = "free"`
- `behavior_mode = "free"`
- `artifact_schema_id = "recording_analysis_v1"`

The batch pipeline entry point `import_recordings_analysis --recording-only`
exists, but it is a full import-plus-detect orchestrator. It should be used
when detection is also intended. For archive creation without detection, use
`import_recording_analysis` per organized recording, then rescan:

```bash
scripts/py -m fisheye.utils.registry_rescan \
  /nvme1/recordings/<recording>/zarr/<recording>_analysis.zarr \
  --registry /nvme1/palette_registry.sqlite
```

For full processing later, the single-recording pipeline supports
`--recording-only --register` and calls `Registry.scan_zarr(...)` after a
successful run:

```bash
scripts/py -m fisheye.utils.run_recording_analysis_pipeline \
  --recording-dir /nvme1/recordings/<recording> \
  --recording-only \
  --model-source registry \
  --registry /nvme1/palette_registry.sqlite \
  --register \
  --dry-run
```

## Training Zarr Creation Path

Training Zarr creation for video-only recordings is supported by:

- `src/fisheye/utils/intake_video_only_recording.py`

Use this for a sampled training archive from the camera MP4. For these
11-hour, 30 fps `sleepyfish` videos, do not blindly use the historical
`--frame-step 100`: that would import approximately `11,880` frames per camera
and create very large training Zarrs. Existing short-recording training Zarrs
with `frame_step=100` contain roughly `185-231` frames and are about `4-5 GB`.
For a comparable per-camera sample count on the 11-hour videos, use
`--frame-step 5000`, which yields approximately `238` frames per camera.

Dry-run example:

```bash
scripts/py -m fisheye.utils.intake_video_only_recording \
  /nvme1/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093/cams/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093.mp4 \
  --recording-dir /nvme1/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093 \
  --zarr-path /nvme1/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093/zarr/sleepyfish_2026_05_05_17_45_30_cam2010093_training.zarr \
  --frame-step 5000 \
  --skip-tail-frames 0 \
  --session-uuid sleepyfish_2026_05_05_17_45_30_cam2010093 \
  --recording-id sleepyfish_2026_05_05_17_45_30_cam2010093 \
  --recording-name sleepyfish_2026_05_05_17_45_30_cam2010093 \
  --protocol-name sleepyfish \
  --dish-design palm \
  --camera-id 2010093 \
  --num-dishes 1 \
  --fish-per-dish 1 \
  --dry-run
```

Remove `--dry-run` to write the training archive. Add `--register --registry
/nvme1/palette_registry.sqlite` when the archive should be indexed immediately.

## Completed Training-Zarr Creation

The four `sleepyfish_2026_05_05_17_45_30_cam201009*` training Zarrs were
created with `frame_step=5000` and registered in
`/nvme1/palette_registry.sqlite`. Each training Zarr has:

- `zarr_purpose = "training"`;
- `raw_video` sampled frame count `238`;
- `raw_video.frame_step = 5000`;
- `experiment_context_status = "absent"`;
- `stimulus_runs_available = false`;
- strict-JSON-readable Zarr metadata.

The stable operator guidance for this workflow now lives in
`docs/operator_guide/sampled_import.md`.

## Historical Dry-Run Sequence

The original pre-apply plan was to draft a video-only organizer manifest CSV in
a scratch location such as `/tmp`, with one row per camera:

```bash
scripts/py -m fisheye.utils.draft_video_only_organizer_manifest \
  /nvme1/staging/2026_05_05_17_45_30 \
  --output /tmp/2026_05_05_17_45_30_video_only_manifest.csv \
  --session-uuid-template sleepyfish_{recording_id}_cam{camera_id} \
  --recording-name-template sleepyfish_{recording_id}_cam{camera_id} \
  --dish-design palm \
  --protocol-name sleepyfish \
  --num-dishes 1 \
  --fish-per-dish 1
```

If some metadata is known only at intake time, either provide it as flags or
edit the generated CSV before applying organization. Then run:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  /nvme1/staging/2026_05_05_17_45_30 \
  --video-only \
  --metadata-csv /tmp/2026_05_05_17_45_30_video_only_manifest.csv \
  --dest-root /nvme1/recordings \
  --write-manifest \
  --rename-cams \
  --dry-run
```

Review that dry-run for:

- exactly four planned recordings
- one camera MP4 per recording
- one paired `Cam*_meta.csv` per recording
- `dish_design = palm`
- `num_dishes = 1`
- `fish_per_dish = 1`
- stable recording names
- no accidental multi-camera directory

Only after the dry-run looks correct should organization be applied, ideally
with video diagnostics:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  /nvme1/staging/2026_05_05_17_45_30 \
  --video-only \
  --metadata-csv /tmp/2026_05_05_17_45_30_video_only_manifest.csv \
  --dest-root /nvme1/recordings \
  --write-manifest \
  --rename-cams \
  --run-video-diagnostics \
  --apply
```

If the batch has already been partially organized with only `cams/` files
present in `/nvme1/recordings`, repair the sidecars instead of rerunning normal
organization:

```bash
scripts/py -m fisheye.utils.backfill_video_only_sidecars \
  /nvme1/staging/2026_05_05_17_45_30 \
  --metadata-csv /tmp/2026_05_05_17_45_30_video_only_manifest_sleepyfish.csv \
  --dest-root /nvme1/recordings \
  --dry-run
```

Review the dry-run, then run the same command with `--apply`. The repair copies
`ptp_sync_summary.json` and `recording_snapshot_runtime.json` into each
recording's `raw/`, moves the per-camera keyframe summary into `cams/`, moves
performance/acquisition CSV sidecars into `derived/`, and updates
`recording_manifest.json`.

## Post-Organization Trim Repair

The current organized `sleepyfish_2026_05_05_17_45_30_cam*` recordings use
MP4s that were shortened to the first 11 hours with:

```bash
ffmpeg -i "$f" -map 0 -t 11:00:00 -c copy "first_11h/${f%.mp4}_first11h.mp4"
```

The organized MP4s probe as exactly `1,188,000` frames at 30 fps, while the
Orange camera CSV/keyframe sidecars still describe the original `2,163,445`
frame acquisition. This mismatch is expected given the manual trim, but it must
be repaired before import because Palette treats camera metadata rows as
frame-aligned with the encoded video.

Use:

```bash
scripts/py -m fisheye.utils.repair_trimmed_video_sidecars \
  /nvme1/recordings \
  --name-prefix sleepyfish_2026_05_05_17_45_30 \
  --dry-run
```

After reviewing, replace `--dry-run` with `--apply`. The tool keeps original
CSV/keyframe sidecars under `derived/original_sidecars/`, writes corrected
frame-aligned sidecars in place (`cams/*_meta.csv` and `cams/*_keyframe.json`
under the current layout), records a `metadata_repairs` manifest entry, and
refreshes video preflight.

If a large HEVC decode smoke is too slow, use:

```bash
scripts/py -m fisheye.utils.repair_trimmed_video_sidecars \
  /nvme1/recordings \
  --name-prefix sleepyfish_2026_05_05_17_45_30 \
  --apply \
  --video-preflight-decode-backend none
```

This still validates encoded frame count, sampled timing/GOP metadata, and
camera-CSV row alignment.

## Remaining Question Before Apply

- No naming question remains for this batch. Use
  `sleepyfish_2026_05_05_17_45_30_cam<serial>`.
