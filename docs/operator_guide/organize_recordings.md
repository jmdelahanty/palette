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
6. Writes `recording_manifest.json` with all metadata extracted from the H5.
7. Validates HEVC keyframe flags on the camera video.

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

Use the video report when you want media and camera-metadata confidence. Use
the H5 report when you want to know whether stimulus import should succeed.
See [video_diagnostics.md](video_diagnostics.md) and
[h5_diagnostics.md](h5_diagnostics.md) for the full CLI and output reference.

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
