# Video Diagnostics

This guide covers the unified raw-video diagnostics tool:

```bash
scripts/py -m fisheye.diagnostics.video ...
```

Use it to inspect organized recordings under `raw/` and `cams/`, validate
camera-side metadata CSVs, and distinguish actual media problems from
environment/tooling problems such as a broken decode backend.

## When to use it

Run diagnostics when you want to:

- sanity-check a newly organized recording before import or analysis
- compare `cams/` and `raw/` videos for the same recording
- confirm whether a warning is a real media problem or just a backend issue
- export machine-readable results for later review

## Where It Fits

The recommended operator workflow is:

1. organize the recording into `raw/`, `cams/`, and `zarr/`
2. run video diagnostics against the organized recording directory
3. if `Media` passes, create or import the analysis Zarr
4. continue with detection and downstream analysis

Diagnostics are a pre-import checkpoint. You can run them manually, or opt
into them during `fisheye.utils.organize_recordings --apply` with
`--run-video-diagnostics`. When the organizer runs them, it records the result
in `recording_manifest.json` under `preflight.video`.

Import commands still do not execute video diagnostics automatically, but they
now honor the recorded manifest gate: `preflight.status=fail` blocks import by
default, while `warn` does not. Use `--allow-preflight-failures` on the import
command only when you intentionally want to bypass a failed recorded preflight.

For repeatable real-data smoke checks, prefer the shared fixture and run
layout documented in [test_data.md](test_data.md).

## Main entry points

### Single-file report

```bash
scripts/py -m fisheye.diagnostics.video report \
  /nvme1/recordings/<recording>/cams/Cam2010093_<recording>.mp4
```

This runs the combined report and prints:

- `Overall`: the default verdict, based on media health
- `Media`: media-only status
- `Tooling`: tooling/backend status
- container checks
- stream metadata
- timing checks
- GOP/keyframe checks
- `Camera CSV` checks for `cams/*.mp4`
- decode backend results

### Batch report

```bash
scripts/py -m fisheye.diagnostics.video batch \
  /nvme1/recordings/<recording>
```

This scans one or more files/directories and prints:

- file-level media/tooling counts
- recording-level media/tooling counts
- grouped `cams/` and `raw/` entries under each recording root

Example:

```text
Overall: pass

Summary
  scanned: 2
  media_files: pass=2, warn=0, fail=0, error=0, skip=0
  tooling_files: pass=0, warn=0, fail=0, error=2, skip=0
  sources: cams=1, raw=1, other=0
  media_recordings: pass=1, warn=0, fail=0, error=0, skip=0
  tooling_recordings: pass=0, warn=0, fail=0, error=1, skip=0
```

This means the recording media looks healthy, but one tooling backend failed.

## Status model

The diagnostics intentionally separate media health from tooling health.

- `Overall`: default operator verdict. This follows `Media`.
- `Media`: whether the video or paired metadata looks bad.
- `Tooling`: whether the inspection environment is healthy.

Typical examples:

- `Overall: pass`, `Media: pass`, `Tooling: error`
  The video looks fine, but a backend such as Decord is unavailable.
- `Overall: fail`, `Media: fail`, `Tooling: pass`
  The environment is fine and the media itself failed checks.

Tooling errors do not downgrade the default media verdict.

## What gets checked

### Container

- codec normalization for HEVC/H.264 variants
- MP4 `moov` scan for `stss` sync-sample entries
- HEVC seek-risk warning when `stss` is missing

### Stream

- container and codec
- resolution
- fps
- duration
- pixel format
- frame count when available from `ffprobe`

### Timing

- sampled or full-scan PTS/DTS checks
- PTS monotonicity
- DTS monotonicity
- suspicious timestamp gaps

By default the tool runs a quick sampled inspection. Use `--full-scan` for a
full frame-level timing pass.

### GOP

- keyframe count
- maximum GOP size
- keyframe interval
- HEVC B-frame presence

### Camera CSV

For `cams/*.mp4`, the tool looks for a sibling `*_meta.csv` file, for example:

- `Cam2010093_<recording>.mp4`
- `Cam2010093_<recording>_meta.csv`

It validates:

- required columns: `frame_id`, `timestamp`, `timestamp_sys`
- row count
- `frame_id` monotonicity
- `frame_id` contiguity
- `timestamp` monotonicity
- `timestamp_sys` monotonicity
- row-count match against the video frame count, when available
- median timestamp step sizes
- drift between `timestamp` and `timestamp_sys`

For `raw/*.mp4`, this section is skipped.

### Decode

By default the tool attempts both:

- OpenCV
- Decord

This is useful because one backend can fail even when the media is fine.

The organizer and `backfill_hevc_keyframe_flags.py` now reuse the same shared container-check logic as the unified video diagnostics.

## Quick vs full scan

Quick mode is the default.

Current quick defaults:

- timing/GOP sample: `120` frames
- decode sample: `30` frames
- random-seek sample: `10` positions

Use:

```bash
scripts/py -m fisheye.diagnostics.video report <video> --full-scan
```

when you want a slower but stronger frame-level timing inspection.

## Source filtering in batch mode

Batch mode inspects both `cams/` and `raw/` by default.

You can limit it to one side:

```bash
scripts/py -m fisheye.diagnostics.video batch /nvme1/recordings --source cams
scripts/py -m fisheye.diagnostics.video batch /nvme1/recordings --source raw
```

Available values:

- `all`
- `cams`
- `raw`
- `other`

Batch discovery also skips repaired `*_fixed.mp4` files automatically.

## JSON and JSONL output

### Structured batch JSON

```bash
scripts/py -m fisheye.diagnostics.video batch /nvme1/recordings --json
```

This emits one JSON object for the full batch report, including grouped
recordings and summary counts.

### Per-video JSONL export

```bash
scripts/py -m fisheye.diagnostics.video batch \
  /nvme1/recordings \
  --jsonl /tmp/video_diagnostics_batch.jsonl
```

This writes one JSON object per inspected video. Each line includes the full
per-file report, including:

- `overall_status`
- `media_status`
- `tooling_status`
- `file_info`
- `stream_info`
- `timing`
- `gop`
- `camera_csv`
- `decode`
- `findings`

This is the easiest format to post-process with `jq`, Python, or pandas.

## Focused subcommands

You can run narrower checks when needed:

```bash
scripts/py -m fisheye.diagnostics.video probe <video>
scripts/py -m fisheye.diagnostics.video timing <video>
scripts/py -m fisheye.diagnostics.video gop <video>
scripts/py -m fisheye.diagnostics.video decode <video>
```

## Compatibility wrappers

These older scripts still exist, but now delegate to the unified tool:

- `src/video_integrity_checker.py`
- `src/video_diagnostic_tool.py`

Prefer the unified CLI for new usage.

## Suggested workflow

After organization and before import:

```bash
scripts/py -m fisheye.diagnostics.video batch \
  "$PALETTE_RECORDINGS_ROOT/<recording>"

scripts/py -m fisheye.diagnostics.h5 report \
  "$PALETTE_RECORDINGS_ROOT/<recording>"
```

If you want the video results archived:

```bash
scripts/py -m fisheye.diagnostics.video batch \
  "$PALETTE_RECORDINGS_ROOT/<recording>" \
  --jsonl "/tmp/<recording>_video_diagnostics.jsonl"
```

If `Media` passes but `Tooling` errors, the recording is probably okay and the
problem is with the inspection environment, not the video itself. Use
[h5_diagnostics.md](h5_diagnostics.md) alongside the video preflight when you
also want to confirm that the raw Citrus H5 is importable.
