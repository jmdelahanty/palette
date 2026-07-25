# Organize Recordings

This guide walks you through taking a freshly acquired dataset and organizing
it into the directory structure that the rest of the Palette pipeline expects.

## Prerequisites

- Access to a machine with the Palette conda environment activated.
- Your recording files (from Citrus or transferred from the rig machine).
- The environment variables below set in your shell profile (ask Jeremy if
  you're unsure what values to use for your machine):

```bash
export PALETTE_STAGING_ROOT=/nvme1/staging
export PALETTE_RECORDINGS_ROOT=/nvme1/recordings
# Optional — defaults to <PALETTE_RECORDINGS_ROOT>/logs/organize_recordings
export PALETTE_LOG_ROOT=/nvme1/recordings/logs/organize_recordings
```

## What Citrus produces

After an experiment finishes, Citrus writes a batch of files. A single
recording session typically produces:

| File | Description |
|------|-------------|
| `<timestamp>_arena_<N>_<Protocol>.h5` | Primary experimental data (events, metadata, protocol params) |
| `<timestamp>_arena_<N>_<Protocol>.mp4` | Stimulus replay video |
| `<timestamp>_arena_<N>_<Protocol>_update_timing.csv` | Frame timing diagnostics |
| `Cam<id>.mp4` | Camera video for the session |
| `Cam<id>_meta.csv` | Per-frame camera metadata (timestamps, exposure, etc.) |
| `extracted_<id>_*.png` | Derived calibration/scale images |
| `recording_snapshot.json` | Optional session snapshot with arena config |

The `<id>` in the camera files is the numeric camera serial (e.g. `2010093`).
A batch directory may contain files for multiple arenas if the rig ran several
arenas in one session.

## Step 1: Get your files into staging

The organize step expects your recording batch to live under the staging root
in a timestamped directory.

### If Citrus writes directly to staging (current setup for some rigs)

Nothing to do — your files are already in the right place. They'll look like:

```
$PALETTE_STAGING_ROOT/
  2026_01_28_14_36_16/
    TRANSFER_DONE
    citrus/
      2026-01-28T19-36-18Z_arena_1_Feeding.h5
      2026-01-28T19-36-18Z_arena_1_Feeding.mp4
      2026-01-28T19-36-18Z_arena_1_Feeding_update_timing.csv
      Cam2010093.mp4
      Cam2010093_meta.csv
      extracted_2010093_homography_image.png
      extracted_2010093_scale_image.png
      recording_snapshot.json
```

### If you need to copy files manually

Copy (or `rsync`) your recording batch into a new directory under the staging
root. The outer directory name doesn't matter as long as it's unique — the
convention is the rig-local timestamp (e.g. `2026_01_28_14_36_16`). The files
themselves should be in a `citrus/` subdirectory.

```bash
# Example: copy from a rig network share
rsync -avP /mnt/rig3/data/2026_01_28_14_36_16/ \
  "$PALETTE_STAGING_ROOT/2026_01_28_14_36_16/"
```

Once the transfer is complete, create the done marker so the organizer knows
the batch is ready. The marker lives at the batch root, not inside `citrus/`:

```bash
touch "$PALETTE_STAGING_ROOT/2026_01_28_14_36_16/TRANSFER_DONE"
```

## Step 2: Preview the organization plan

Always dry-run first to make sure the plan looks right:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  "$PALETTE_STAGING_ROOT/2026_01_28_14_36_16" \
  --recursive --dry-run
```

This prints an ASCII tree showing where each file would be moved. No files are
touched. Review the output and check that:

- Every recording has its H5, MP4, timing CSV, and camera files accounted for.
- No files are listed under `missing`.
- The destination folder names look correct.

### Processing all pending batches at once

If you have several batches waiting in staging:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  --process-all --require-done --recursive --dry-run
```

`--process-all` iterates over every subdirectory of the staging root.
`--require-done` skips any batch that doesn't have a `TRANSFER_DONE` marker,
so in-progress transfers won't be touched.

### Video-only batches without H5

Some recording batches contain camera videos and camera metadata CSVs but no
Citrus H5/protocol file. These are valid recording-only inputs, but the
organizer cannot infer the recording context from an H5. Use a separate
operator-reviewed CSV with one row per intended recording:

```bash
scripts/py -m fisheye.utils.draft_video_only_organizer_manifest \
  "$PALETTE_STAGING_ROOT/2026_05_05_17_45_30" \
  --output /tmp/2026_05_05_17_45_30_video_only_manifest.csv \
  --dish-design cedar \
  --num-dishes 1 \
  --fish-per-dish 1
```

The draft helper discovers `Cam*.mp4`, fills `camera_id` from the filename,
links `Cam<id>_meta.csv` when present, and reads `recording_id` /
`timestamp_utc` from `recording_snapshot.json` when available. Fields that the
software cannot know, such as `dish_design`, genotype, dpf, or fish count, can
be provided as CLI flags or edited in the CSV before apply. For interactive
manual entry, add `--prompt-metadata`; for unattended/reproducible work, prefer
explicit flags.

For Orange video-only batches, use encoded-video metadata as the ingest frame
count source. `Cam*_meta.csv` rows and `Cam*_keyframe.json.total_frames` should
match and are authoritative for video import. `ptp_sync_summary.json`
`frame_count` is Orange acquisition/sync telemetry counted since the camera
stream started; it can include frames outside the recording-local encoded video
and is not expected to equal the MP4 frame count.

If the camera MP4 was shortened after acquisition, for example with:

```bash
ffmpeg -i "$f" -map 0 -t 11:00:00 -c copy "first_11h/${f%.mp4}_first11h.mp4"
```

the camera CSV and keyframe JSON must be repaired to match the encoded MP4
before import. Do not treat the mismatch as a harmless warning: downstream
frame-indexed datasets assume one camera metadata row per encoded frame.

After organizing and sidecar backfill, dry-run the repair:

```bash
scripts/py -m fisheye.utils.repair_trimmed_video_sidecars \
  "$PALETTE_RECORDINGS_ROOT" \
  --name-prefix sleepyfish_2026_05_05_17_45_30 \
  --dry-run
```

Then apply:

```bash
scripts/py -m fisheye.utils.repair_trimmed_video_sidecars \
  "$PALETTE_RECORDINGS_ROOT" \
  --name-prefix sleepyfish_2026_05_05_17_45_30 \
  --apply
```

The repair tool probes each MP4 for its encoded frame count, trims
`cams/*_meta.csv` to that many rows, updates `cams/*_keyframe.json`
`total_frames` and `keyframe_frames`, stores original sidecars under
`derived/original_sidecars/`, appends a `metadata_repairs` manifest entry, and
reruns video preflight so stale row-count failures do not block import.
For very large HEVC files where the decode smoke is slow or hangs, add
`--video-preflight-decode-backend none`; this still checks probe metadata,
sampled timing/GOP metadata, and camera-CSV frame alignment.

When present, video-only organization preserves optional sidecars without
requiring them:

- camera stream sidecars are moved into `cams/`: `Cam<id>_keyframe.json` sits
  beside the MP4 and `Cam*_meta.csv` because it describes encoded frame count
  and seek/keyframe structure
- per-camera diagnostics are moved into `derived/`: `Cam<id>_pipeline_perf.csv`
  and `Cam<id>_acquisition_cadence_probe.csv`
- shared session files are copied into each recording's `raw/`:
  `ptp_sync_summary.json` and `recording_snapshot_runtime.json`
- missing optional sidecars do not fail the plan
- the camera MP4, `Cam<id>_meta.csv`, and `Cam<id>_keyframe.json` form the
  primary `cams/` payload when the keyframe summary is available

The CSV consumed by `organize_recordings --video-only --metadata-csv` is not the
same as the generated `recording_manifest.json`. It is a staging/intake table.
The accepted schema is:

| Column | Required | Meaning |
| --- | --- | --- |
| `source_video` | yes* | Source camera video. Relative paths are resolved against the staging source or CSV directory. |
| `video_path` | yes* | Alias accepted instead of `source_video`. |
| `camera_video` | yes* | Alias accepted instead of `source_video`. |
| `source_camera_metadata_csv` | no | Per-frame camera metadata CSV sidecar. |
| `camera_metadata_csv` | no | Alias accepted instead of `source_camera_metadata_csv`. |
| `camera_id` | recommended | Camera serial/id; inferred from `Cam<id>.mp4` if omitted. |
| `session_uuid` | recommended | Stable session identity; defaults to `recording_id` or video stem if omitted. |
| `recording_id` | recommended | Stable recording identity; defaults to `session_uuid` if omitted. |
| `recording_name` | recommended | Destination folder name before filename sanitization; defaults to `session_uuid`. |
| `session_start_iso8601_utc` | recommended | Acquisition start time when known. |
| `recording_type` | no | Defaults to `behavior`. |
| `recording_subtype` | no | Defaults to `free`. |
| `behavior_mode` | no | Defaults to `free`. |
| `artifact_schema_id` | no | Defaults to `video_only_v1`. |
| `dish_design` | recommended | Dish/chamber design. Missing values are surfaced in the plan. |
| `rig_id` | no | Rig/system identifier. |
| `arena_id` | no | Arena/chamber identifier. |
| `canvas_name` | no | Display/canvas name, if relevant. |
| `protocol_name` | no | Manual protocol label, if known. |
| `protocol_name_from_definition` | no | Defaults to `protocol_name` when omitted. |
| `genotype` | no | Subject genotype/cross label when known. |
| `dpf_at_acquisition` | no | Integer days post fertilization. |
| `num_dishes` | recommended | Number of dishes in view. |
| `fish_per_dish` | recommended | Expected fish per dish. |

`source_video`, `video_path`, and `camera_video` are aliases; one of the three
must be present. The same is true for `source_camera_metadata_csv` and
`camera_metadata_csv`, but that sidecar is optional.

After reviewing the CSV:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  "$PALETTE_STAGING_ROOT/2026_05_05_17_45_30" \
  --video-only \
  --metadata-csv /tmp/2026_05_05_17_45_30_video_only_manifest.csv \
  --dest-root "$PALETTE_RECORDINGS_ROOT" \
  --write-manifest \
  --rename-cams \
  --dry-run
```

If a video-only batch was already organized before optional sidecar handling was
available, repair the existing recording folders instead of rerunning normal
organization:

```bash
scripts/py -m fisheye.utils.backfill_video_only_sidecars \
  "$PALETTE_STAGING_ROOT/2026_05_05_17_45_30" \
  --metadata-csv /tmp/2026_05_05_17_45_30_video_only_manifest.csv \
  --dest-root "$PALETTE_RECORDINGS_ROOT" \
  --dry-run
```

After reviewing the plan, replace `--dry-run` with `--apply`. This copies
shared sidecars into each recording, moves the keyframe summary to `cams/`,
moves per-camera diagnostic sidecars to `derived/`, and patches
`recording_manifest.json` `files.raw` / `files.cams` / `files.derived`.

### Orange `external_ipc` batches with full-frame and crop videos

Newer Orange sessions may write a batch root with `recording_session.json`,
`external_recorder/`, `external_crop_recorder/`, and Citrus H5 files under
`citrus/`. In this layout, the merged full-frame external recorder MP4 is the
ingest-authoritative camera video, while the cropped MP4 is a first-class
runtime-derived acquisition input. Shard intermediates are Orange
implementation/debug details and should not be copied into canonical recording
folders. See
[`docs/orange_runtime_video_artifact_contract.md`](../orange_runtime_video_artifact_contract.md).

`organize_recordings` auto-detects this layout when `recording_session.json`
contains `recording_outputs` from `external_ipc`; `--external-ipc` can be passed
to make the mode explicit.

Dry-run first:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  "$PALETTE_STAGING_ROOT/2026_05_29_14_11_07" \
  --external-ipc \
  --dest-root "$PALETTE_RECORDINGS_ROOT" \
  --dry-run
```

Expected planning behavior:

- each Citrus H5 becomes one recording folder, as in legacy multi-arena Citrus
  batches
- `recording_manifest.json` metadata is assembled from H5 root attrs plus
  nested H5 context where available: `protocol_snapshot` for `protocol_name`,
  `calibration_snapshot/arena_config_json` for `dish_design`, and
  `subject_metadata` for `genotype` / `dpf_at_acquisition`
- Orange `recording_snapshot.json` is copied as
  `raw/recording_snapshot_runtime.json` and is also used as a fallback source
  for `software_version` when the H5 root attr is empty
- transfer and runtime-control context is copied into `raw/`:
  `transfer_complete.json`, `orange_local_control.events.jsonl`,
  `ptp_sync_summary.json`, recorder contracts, and recorder supervisor plans
- Citrus threading startup JSONs are copied under `derived/citrus/`
- `num_dishes` and `fish_per_dish` are not inferred from subject counts; they
  must be written explicitly by acquisition or supplied by an operator metadata
  layer before Palette treats them as manifest facts
- `cams/` receives the merged full-frame `external_recorder/Cam*_external.mp4`
  renamed to the normal `Cam<id>_<session>.mp4` convention
- `cams/` also receives a compatibility `Cam<id>_<session>_meta.csv` copied
  from the crop metadata table, because it has the same recording-frame clock
  and lets existing single-video frame-index builders discover the recording
- `cams/` receives `Cam<id>_<session>_keyframe.json` from the full-frame
  external recorder keyframe summary
- crop outputs and crop-specific metadata are preserved under
  `derived/external_crop_recorder/`
- `recording_manifest.json` includes `video_streams` with separate `full` and
  `crop` entries. The crop entry declares `video_pixel_coordinate_space` as
  `crop_frame_pixels` and its source geometry columns as full-frame pixels.
- full external recorder diagnostics that are not part of the compatibility
  camera bundle are preserved under `derived/external_recorder/`
- files with shard names such as `*_shard*_gpu*.mp4`,
  `*_keyframes_shard*.json`, and `*_encode_shard*.csv` are intentionally ignored

Apply only after the dry-run has no unexpected `missing` entries:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  "$PALETTE_STAGING_ROOT/2026_05_29_14_11_07" \
  --external-ipc \
  --dest-root "$PALETTE_RECORDINGS_ROOT" \
  --write-manifest \
  --apply
```

For Orange `external_ipc` batches that have `recording_session.json` and
external recorder videos but no Citrus H5 files, use the explicit no-H5 mode:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  "$PALETTE_STAGING_ROOT/2026_07_07_15_59_35" \
  --external-ipc-recording-only \
  --dest-root "$PALETTE_RECORDINGS_ROOT" \
  --dry-run
```

This creates one recording folder per camera output in `recording_outputs`.
The recording folder includes the camera ID to keep multi-camera sessions
distinct, while camera artifacts use the normal `Cam<id>_<session>` file naming
without repeating the camera suffix. Full-frame-only sessions are valid; when no
crop metadata table exists, the manifest omits the compatibility
`frame_clock_metadata` entry instead of inventing one. If crop outputs exist,
the crop video and crop metadata are preserved under
`derived/external_crop_recorder/` and the crop metadata is also copied into
`cams/` for compatibility. H5 diagnostics are not supported in this mode.

The organizer preserves Orange acquisition-host absolute paths by remapping
paths below the transferred batch directory name back under the local staging
batch root. This allows `recording_session.json` written on the rig to remain
the source of truth after transfer.

If external-IPC recordings were organized before Palette learned to lift the
nested protocol/dish/subject metadata, refresh their manifests in place:

```bash
scripts/py -m fisheye.utils.refresh_recording_manifest_metadata \
  "$PALETTE_RECORDINGS_ROOT" \
  --recursive \
  --refresh-external-ipc-artifacts
```

This is a dry-run. Re-run with `--apply` after reviewing the field list. The
tool fills empty `protocol_name`, `dish_design`, `genotype`,
`dpf_at_acquisition`, and `software_version` fields from the organized H5 and
runtime snapshot, leaves `num_dishes` / `fish_per_dish` untouched unless they
were explicitly present already, and for external-IPC recordings can copy the
small retained session/control artifacts plus add `video_streams` to existing
manifests. It does not copy, move, or delete full-frame shard debug outputs.

## Step 3: Apply the organization

Once the dry-run looks correct, apply it:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  "$PALETTE_STAGING_ROOT/2026_01_28_14_36_16" \
  --recursive \
  --apply \
  --write-manifest \
  --rename-cams \
  --cleanup-empty
```

Or use the convenience wrapper, which processes all done batches with sensible
defaults:

```bash
scripts/organize_staging.sh
```

The wrapper runs with `--process-all --require-done --recursive --apply
--write-manifest --rename-cams --cleanup-staging --cleanup-empty`.

### What `--apply` does

For each recording discovered via its H5 file, the organizer:

1. Creates a destination folder named after the recording
   (e.g. `2026-01-28T19-36-18Z_arena_1_Feeding/`).
2. Moves the H5, stimulus MP4, and timing CSV into `raw/`.
3. Moves camera MP4 and metadata CSV into `cams/`, renaming them to include
   the session ID (e.g. `Cam2010093_2026-01-28T19-36-18Z_arena_1.mp4`).
4. Moves derived images and snapshots into `derived/`.
5. Creates an empty `zarr/` directory (used by later pipeline stages).
6. Writes `recording_manifest.json` with metadata extracted from the H5 and,
   for Orange external-IPC batches, selected runtime sidecars.
7. Validates HEVC keyframe flags on the camera video.
8. If diagnostics hooks are enabled, records a `preflight` summary in the manifest.

## After organization: what you should have

```
$PALETTE_RECORDINGS_ROOT/
  2026-01-28T19-36-18Z_arena_1_Feeding/
    raw/
      2026-01-28T19-36-18Z_arena_1_Feeding.h5
      2026-01-28T19-36-18Z_arena_1_Feeding.mp4
      2026-01-28T19-36-18Z_arena_1_Feeding_update_timing.csv
      recording_snapshot_runtime.json
    cams/
      Cam2010093_2026-01-28T19-36-18Z_arena_1.mp4
      Cam2010093_2026-01-28T19-36-18Z_arena_1_meta.csv
    zarr/
    derived/
      extracted_2010093_homography_image.png
      extracted_2010093_scale_image.png
      recording_snapshot.json
    recording_manifest.json
```

`raw/recording_snapshot_runtime.json` is the full, unfiltered snapshot from
Citrus — preserved as-is for recovery. `derived/recording_snapshot.json` is
the per-camera filtered version used by downstream tools.

This is the starting point for all downstream pipeline steps (detection,
keypoints, segmentation, analysis, etc.).

## Optional: run diagnostics before import

Once a recording has been organized, you can run both unified diagnostics
preflights before creating the analysis Zarr:

```bash
scripts/py -m fisheye.diagnostics.video batch \
  "$PALETTE_RECORDINGS_ROOT/2026-01-28T19-36-18Z_arena_1_Feeding"

scripts/py -m fisheye.diagnostics.h5 report \
  "$PALETTE_RECORDINGS_ROOT/2026-01-28T19-36-18Z_arena_1_Feeding"
```

The video preflight checks both `cams/` and `raw/` videos by default, groups
entries by recording, and validates the paired `Cam..._meta.csv` camera
metadata file for each `cams/*.mp4`.

The H5 preflight resolves the organized `raw/*.h5` file automatically and
checks whether the raw Citrus H5 meets Palette import requirements while also
reporting optional section health for tracking, snapshots, and enums.

If you want `organize_recordings` to run these preflights immediately after
`--apply`, use `--run-video-diagnostics`, `--run-h5-diagnostics`, or both:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  "$PALETTE_STAGING_ROOT/<batch>" \
  --apply \
  --write-manifest \
  --run-video-diagnostics \
  --run-h5-diagnostics
```

When these hooks run, the organizer persists a `preflight` block into
`recording_manifest.json`. That block stores the combined preflight verdict plus
separate video and H5 summaries. Downstream import entry points now refuse
`preflight.status=fail` by default:

If the hooks do not run, the manifest keeps the default
`preflight.status="not_run"` with `checked_at_utc=null`; that is an unchecked
state, not a media/H5 failure.

- `scripts/py -m fisheye.utils.import_recording_analysis ...`
- `scripts/py -m fisheye.utils.import_organized_recordings_analysis ...`
- `scripts/py -m fisheye.utils.run_recording_analysis_pipeline ...`
- `scripts/py -m fisheye.utils.import_recordings_analysis ...`

Use `--allow-preflight-failures` on those commands only when you intentionally
want to bypass a failed recorded preflight. A stored `warn` does not block
import.

Use the video report when you want media and camera-metadata confidence. Use
the H5 report when you want to know whether stimulus import should succeed.
See [video_diagnostics.md](video_diagnostics.md),
[h5_diagnostics.md](h5_diagnostics.md), and
[test_data.md](test_data.md) for the full CLI, output reference, and the shared
real-data fixture convention under `/nvme1/palette_test_data`.

## Create Analysis Archives After Organizing

After `organize_recordings --apply`, use the import-only batch command to create
analysis zarrs from the organized `raw/*.h5` and `cams/*.mp4` layout:

```bash
scripts/py -m fisheye.utils.import_organized_recordings_analysis \
  "$PALETTE_RECORDINGS_ROOT" \
  --dry-run

scripts/py -m fisheye.utils.import_organized_recordings_analysis \
  "$PALETTE_RECORDINGS_ROOT" \
  --registry /nvme1/palette_registry.sqlite \
  --apply
```

For a single organize run, pass the organizer JSONL log so only recordings from
that run are imported:

```bash
scripts/py -m fisheye.utils.import_organized_recordings_analysis \
  --organize-log "$PALETTE_RECORDINGS_ROOT/logs/organize_recordings/<run>.jsonl" \
  --registry /nvme1/palette_registry.sqlite \
  --dry-run

scripts/py -m fisheye.utils.import_organized_recordings_analysis \
  --organize-log "$PALETTE_RECORDINGS_ROOT/logs/organize_recordings/<run>.jsonl" \
  --registry /nvme1/palette_registry.sqlite \
  --apply
```

This command only creates/updates `zarr/<recording>_analysis.zarr`, imports
source-video metadata, and imports H5 stimulus metadata. When `--registry` is
provided, each successful import is scanned into that registry before the
recording is reported as `ok`; existing analysis zarrs skipped by the default
skip-existing behavior are also scanned so registry-backed review tools can
discover them. If registry sync fails, the import batch exits non-zero instead
of reporting a plain success. The command does not run detection, refinement,
or keypoints. The older
`fisheye.utils.import_recordings_analysis --apply` command is a full pipeline
wrapper despite its name, so use it only when you intentionally want inference
stages after import.

## Finalize Completed Staging Batches

After organize and import have succeeded, finalize the staging batch so old
transfer artifacts do not keep looking like active work. Finalization is a
separate safety gate, not part of import. It verifies that each organized
recording is self-contained before moving or deleting the source staging batch.

Dry-run first:

```bash
scripts/py -m fisheye.utils.finalize_organized_staging \
  "$PALETTE_STAGING_ROOT/<batch>" \
  --recordings-root "$PALETTE_RECORDINGS_ROOT" \
  --organize-log "$PALETTE_RECORDINGS_ROOT/logs/organize_recordings/<organize-run>.jsonl" \
  --import-log "$PALETTE_RECORDINGS_ROOT/logs/import_organized_recordings_analysis/<import-run>.jsonl"
```

If the report is `ready`, apply it:

```bash
scripts/py -m fisheye.utils.finalize_organized_staging \
  "$PALETTE_STAGING_ROOT/<batch>" \
  --recordings-root "$PALETTE_RECORDINGS_ROOT" \
  --organize-log "$PALETTE_RECORDINGS_ROOT/logs/organize_recordings/<organize-run>.jsonl" \
  --import-log "$PALETTE_RECORDINGS_ROOT/logs/import_organized_recordings_analysis/<import-run>.jsonl" \
  --apply
```

Default apply behavior moves the verified batch to
`$PALETTE_STAGING_ROOT/.processed/<batch>`. Use `--delete --apply` only when you
intentionally want immediate deletion.

The finalizer blocks when:

- no organized recordings can be matched to the staging batch
- any `recording_manifest.json` is missing or invalid
- any manifest-listed `raw/`, `cams/`, or `derived/` file is missing
- any expected `zarr/<recording>_analysis.zarr` is missing
- a supplied import log has `recording_failed` or `recording_skipped` for a
  matched recording
- a supplied organize log contains warnings for the batch, unless
  `--allow-organize-warnings` is passed

After import and before production detect/refine, create or verify the dish mask
on each analysis Zarr with `fisheye.tune.mask_tuner` unless a trusted
acquisition-time mask was already imported into
`analysis_metadata.attrs["dish_mask"]`. Refined detection uses this mask for
outside-dish gating.

## Logs

Every organize run writes a JSONL log file to the log directory. If something
goes wrong or you need to audit what happened:

```bash
# Find the log for a specific session
rg '"session_uuid": "2026-01-28T19-36-18Z_arena_1"' \
  "$PALETTE_RECORDINGS_ROOT/logs/organize_recordings/"
```

See [organize_recordings_logging_schema.md](../organize_recordings_logging_schema.md)
for the full log format reference.

## Troubleshooting

**"Source path does not exist"** — Check that `PALETTE_STAGING_ROOT` is set
and the batch directory exists.

**Missing camera files** — The organizer looks for `Cam<id>.mp4` next to the
H5 file and in parent directories. If your camera files are in a different
location, use `--cam-root /path/to/cam/dir`.

**Batch skipped with `--require-done`** — The batch directory is missing its
`TRANSFER_DONE` marker. Either the transfer isn't complete yet, or you need to
create the marker manually with `touch` at the batch root.

**Destination already exists** — The organizer won't overwrite an existing
recording directory. If you need to re-organize, remove or rename the existing
destination first.
