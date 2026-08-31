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
- structural MP4 video-track sample-table inspection
- explicit MP4 sync-sample semantics:
  - `indexed_sync_samples`: the video track contains `stss`, whose entries name
    the sync samples
  - `all_samples_sync`: the video track omits `stss`, which ISO BMFF defines as
    every sample in that track being a sync sample
  - `unreadable`: `moov`, the video sample table, or the atom layout could not
    be inspected
- independent Orange crop proof, when the declared crop summary and keyframe
  sidecar are available:
  - `container_declared`
  - `orange_idr_sidecar_verified`
  - `orange_idr_sidecar_unavailable`
  - `orange_idr_sidecar_contradiction`

An absent `stss` box is not a missing-table defect and does not by itself
justify re-encoding. In particular, Orange lossless HEVC crop streams with
`resolved_gop_length=1` are expected to contain only IDR/keyframes, so FFmpeg
may omit the redundant table. Palette validates the producer proof by requiring
the summary frame count and sidecar `total_frames` to agree and by streaming the
sidecar indices to prove exact coverage of `0..N-1`. This validation does not
load a million-frame keyframe list into memory.

If an Orange inter-frame stream omits `stss`, or if its GOP=1 summary and
keyframe sidecar disagree, diagnostics report
`video.orange_sync_evidence_contradiction`. Operators should investigate the
producer evidence rather than rewrite the immutable recording. Missing or
malformed `moov`/sample-table structure remains
`video.container_inspection_error`.

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

- required columns: `recording_frame_id`, `timestamp`, `timestamp_sys`
- legacy aliases: `frame_id` and `local_frame_id` are accepted for
  `recording_frame_id`
- row count
- `recording_frame_id` monotonicity
- `recording_frame_id` contiguity
- `timestamp` monotonicity
- `timestamp_sys` monotonicity
- row-count match against the video frame count, when available
- median timestamp step sizes
- drift between `timestamp` and `timestamp_sys`

For `raw/*.mp4`, this section is skipped.

### Decode

By default the tool attempts:

- OpenCV

Decord remains available with `--backend decord` or `--backend all` for manual
compatibility checks, but it is no longer the default preflight backend. Current
pipeline decode work has moved toward PyNvVideoCodec, and a Decord failure
should not block current recording imports unless an operator explicitly asks
to validate that backend.

The organizer and `backfill_hevc_keyframe_flags.py` reuse the same shared
container-check logic as the unified video diagnostics. Organizer manifests
retain the compatibility observation `has_stss`, but consumers should use
`sync_sample_semantics`, `sync_sample_proof`, and
`container_inspection_status` for decisions. Historical inputs without Orange
summaries remain inspectable from their MP4 declaration; Palette does not
invent producer evidence for them.

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
