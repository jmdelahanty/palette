# H5 Diagnostics

This guide covers the unified raw H5 diagnostics tool:

```bash
scripts/py -m fisheye.diagnostics.h5 ...
```

Use it to inspect organized `raw/*.h5` Citrus outputs, validate Palette import
requirements, and distinguish import-critical problems from optional missing or
malformed sections such as tracking data.

## When to use it

Run diagnostics when you want to:

- sanity-check a newly organized recording before import
- confirm whether a raw Citrus H5 is importable into Palette
- inspect whether optional sections such as tracking, snapshots, and enums are present
- export machine-readable results for later review

## Where It Fits

The recommended operator workflow is:

1. organize the recording into `raw/`, `cams/`, and `zarr/`
2. run video diagnostics against the organized recording directory
3. run H5 diagnostics against the organized recording directory or `raw/*.h5`
4. if both preflights pass, create or import the analysis Zarr
5. continue with detection and downstream analysis

Diagnostics are pre-import checkpoints. You can run them manually, or opt
into them during `fisheye.utils.organize_recordings --apply` with
`--run-h5-diagnostics`. They are still not run automatically by
`fisheye.analysis.create_analysis_zarr` or
`fisheye.utils.import_recording_analysis`.

For repeatable real-data smoke checks, prefer the shared fixture and run
layout documented in [test_data.md](test_data.md).

## Main entry points

### Single-recording or single-file report

```bash
scripts/py -m fisheye.diagnostics.h5 report \
  /nvme1/recordings/<recording>
```

You can also point directly at a file:

```bash
scripts/py -m fisheye.diagnostics.h5 report \
  /nvme1/recordings/<recording>/raw/<recording>.h5
```

If you pass a recording directory, the tool resolves the `raw/*.h5` file
automatically.

This runs the combined report and prints:

- `Overall`: the default verdict, based on import-critical H5 health
- `Core`: whether Palette can ingest the H5 for stimulus import
- `Optional`: the health of optional Citrus sections if present
- `Tooling`: file-open or parser problems
- core dataset presence and row counts
- events checks
- frame metadata checks
- tracking dataset summaries
- snapshot/metadata presence
- enum dataset summaries

### Batch report

```bash
scripts/py -m fisheye.diagnostics.h5 batch \
  /nvme1/recordings/<recording>
```

This scans one or more files/directories and prints:

- file-level core/optional/tooling counts
- recording-level grouped entries under each recording root
- per-file finding codes when present

Example:

```text
Overall: pass

Summary
  scanned: 1
  core_files: pass=1, warn=0, fail=0, error=0, skip=0
  optional_files: pass=1, warn=0, fail=0, error=0, skip=0
  tooling_files: pass=1, warn=0, fail=0, error=0, skip=0
```

## Status model

The diagnostics intentionally separate import-critical validity from optional
sections.

- `Overall`: default operator verdict. This follows `Core`.
- `Core`: whether the H5 meets Palette ingest requirements.
- `Optional`: whether optional Citrus sections are present and well-formed.
- `Tooling`: whether the inspection environment was able to open and read the file.

Typical examples:

- `Overall: pass`, `Core: pass`, `Optional: pass`
  The H5 is importable and optional sections look healthy.
- `Overall: fail`, `Core: fail`, `Optional: pass`
  The H5 is missing an ingest-critical section such as `/events` or `/video_metadata/frame_metadata`.
- `Overall: pass`, `Core: pass`, `Optional: fail`
  The H5 is importable, but an optional section such as `tracking_data/chaser_states` is malformed.

Optional section failures do not downgrade the default import verdict.

## What gets checked

### Core

The default `palette-import` profile treats these as import-critical:

- H5 file opens successfully
- `/events` exists
- `/video_metadata/frame_metadata` exists
- required event fields are present
- required frame metadata fields are present
- required datasets are nonempty

### Events

The `/events` checker reports:

- row count
- `timestamp_ns_session` monotonicity
- event type distribution
- `details_json` parse-failure count

`details_json` parse issues are surfaced as metrics, not automatic import failures.

### Frame metadata

The `/video_metadata/frame_metadata` checker reports:

- row count
- `stimulus_frame_num` monotonicity
- `triggering_camera_frame_id` nondecreasing behavior
- unique camera frame count
- missing camera frame count across the observed range
- mean and median stimulus-rows-per-camera-frame
- sparse ratio irregularity metrics
- cumulative alignment drift

The current heuristic is intentionally tolerant of sparse compensated `1/3` or
`3/1` per-camera mappings when there is no missing-frame behavior and cumulative
drift stays low.

### Tracking

Tracking checks are optional. They do not fail the import verdict just because
tracking data is absent.

The tool inspects:

- `/tracking_data` presence
- `bounding_boxes`
- `chaser_states`
- `independent_motion_grid_states`
- `moving_grating_states`

If a tracking dataset exists but is malformed, `Optional` can warn or fail while
`Core` still passes.

### Snapshots and metadata

The tool checks for the presence and basic JSON parseability of:

- `/protocol_snapshot`
- `/calibration_snapshot`
- `/recording_snapshot`
- `/subject_metadata`
- `/session_metadata`
- `/stimulus_coordinates`

These are optional sections.

### Enums

The tool inspects `/enums/*` datasets and validates `id`/`name` structure when
present.

This is useful because enums improve downstream decode fidelity, but missing or
partial enum coverage is not treated as a core ingest failure.

## Profiles

The CLI currently supports:

- `palette-import`
- `citrus-contract`

The default is `palette-import`.

Use:

```bash
scripts/py -m fisheye.diagnostics.h5 report <recording-or-h5> --profile citrus-contract
```

when you want a broader Citrus-output view instead of the stricter Palette
importability view.

Batch mode also supports:

- `--no-recursive` to stop directory recursion
- `--limit N` to cap how many H5 files are inspected

## JSON and JSONL output

### Structured batch JSON

```bash
scripts/py -m fisheye.diagnostics.h5 batch /nvme1/recordings --json
```

This emits one JSON object for the full batch report, including grouped
recordings and summary counts.

### Per-H5 JSONL export

```bash
scripts/py -m fisheye.diagnostics.h5 batch \
  /nvme1/recordings \
  --jsonl /tmp/h5_diagnostics_batch.jsonl
```

This writes one JSON object per inspected H5. Each line includes the full
per-file report, including:

- `overall_status`
- `core_status`
- `optional_status`
- `tooling_status`
- `file_info`
- `core`
- `events`
- `frame_metadata`
- `tracking`
- `snapshots`
- `enums`
- `findings`

## Focused subcommands

You can run narrower checks when needed:

```bash
scripts/py -m fisheye.diagnostics.h5 events <recording-or-h5>
scripts/py -m fisheye.diagnostics.h5 frame-metadata <recording-or-h5>
scripts/py -m fisheye.diagnostics.h5 tracking <recording-or-h5>
scripts/py -m fisheye.diagnostics.h5 snapshots <recording-or-h5>
scripts/py -m fisheye.diagnostics.h5 enums <recording-or-h5>
```

## Current contract boundary

The current practical Palette import boundary is:

Required for default `Core: pass`:

- H5 exists and opens
- `/events` exists and is usable
- `/video_metadata/frame_metadata` exists and is usable

Optional for `Core: pass`:

- `/tracking_data/*`
- `/enums/*`
- `/protocol_snapshot`
- `/calibration_snapshot`
- `/recording_snapshot`
- `/subject_metadata`
- `/session_metadata`
- `/stimulus_coordinates/*`

This means missing `chaser_states` or `bounding_boxes` should not, by themselves,
be treated as broken recordings.

## Suggested workflow

After organization and before import:

```bash
scripts/py -m fisheye.diagnostics.video batch \
  "$PALETTE_RECORDINGS_ROOT/<recording>"

scripts/py -m fisheye.diagnostics.h5 report \
  "$PALETTE_RECORDINGS_ROOT/<recording>"
```
