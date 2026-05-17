# Analysis Zarr Creation Contract (Proposed)
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-02-27
-->

Purpose: define a migration-safe, operator-first contract for creating analysis archives independently from `detect_yolo` inference.

## Scope

- Create or validate an analysis archive before inference.
- Import/mirror core metadata and stimulus context.
- Register/refresh archive metadata in registry.
- Keep detector responsibilities limited to inference run writes.

## Non-goals

- Replacing `detect_yolo` inference logic.
- Introducing multi-camera 3D behavior in this first contract.
- Changing training-data import workflows.

## Motivation

Current behavior lets `detect_yolo` create a minimal output archive when missing.
That is useful but mixes two concerns:

- archive lifecycle/provenance/registry policy
- model inference

This contract separates these concerns so operators can:

- run `--dry-run` readiness checks before expensive inference
- fail fast on metadata/provenance gaps
- enforce consistent analysis archive shape and registry invariants

## New Module Contract

Module: `fisheye.analysis.create_analysis_zarr`

Primary intent:
- ensure `<recording>/zarr/<recording>_analysis.zarr` exists with canonical metadata
- optionally import stimulus metadata from H5
- optionally register/rescan in registry

### CLI (proposed)

```bash
scripts/py -m fisheye.analysis.create_analysis_zarr \
  --recording-dir /nvme1/recordings/<recording_dir> \
  [--video /path/to/cams/CamXXXX.mp4] \
  [--h5 /path/to/raw/session.h5] \
  [--output /path/to/<recording>_analysis.zarr] \
  [--import-stimulus] [--stimulus-run-name <name>] [--stimulus-overwrite] \
  [--allow-preflight-failures] \
  [--register --registry /nvme1/palette_registry.sqlite] \
  [--apply]
```

Defaults:
- mode is dry-run unless `--apply` is passed
- output defaults to `<recording_dir>/zarr/<recording_dir_name>_analysis.zarr`
- `zarr_purpose` is enforced as `analysis`
- stimulus import defaults to enabled when `--h5` is resolvable

### Inputs and resolution rules

- `--recording-dir` is required unless both `--video` and `--output` are explicit.
- `--video` resolution:
  - if omitted, resolve from `recording_dir/cams/*.mp4` for the first-class
    single-video camera layout
  - if multiple candidates exist, fail with explicit message
- `--h5` resolution:
  - if omitted, resolve from `recording_dir/raw/*.h5`
  - if multiple candidates exist, fail with explicit message

### Required write outcomes (apply mode)

1. Archive exists at output path.
2. Root attrs include canonical analysis-purpose metadata:
   - `zarr_purpose = "analysis"`
   - source video path/name and basic video fields when resolvable
3. If `--import-stimulus` succeeds:
   - `analysis/stimulus_runs/<run>` exists
   - `analysis/stimulus_runs.attrs["latest"]` updated
4. If `--register`:
   - archive is scanned into registry
   - dataset row is active and path is current

### Failure semantics

- Fail-closed on ambiguous camera/H5 resolution.
- Fail-closed if `--import-stimulus` requested but H5 unavailable.
- Fail-closed when `recording_manifest.json` records `preflight.status=fail`,
  unless `--allow-preflight-failures` is passed.
- Return non-zero if any requested step fails.
- Dry-run must never mutate filesystem or registry.

### Logging

- JSONL log output with per-step events:
  - plan
  - create/validate archive
  - stimulus import
  - registry scan
  - run summary

## `detect_yolo` Contract After Split

`detect_yolo` remains an inference module with these invariants:

- requires explicit model path (or config model path) as today
- writes one new `detect_runs/<run>` entry
- updates detect-run provenance/summary
- does not own higher-level archive lifecycle policy

### Transitional compatibility mode

During migration, keep existing behavior behind explicit flags:

- `--allow-create-zarr` (temporary compatibility)
- preferred path: run against pre-created analysis archive from `create_analysis_zarr`

## Orchestration Contract

Batch module (`import_recordings_analysis`) becomes orchestrator:

1. create/validate analysis archive (`create_analysis_zarr`)
2. run detector (`detect_yolo`)
3. refine detect (optional)
4. registry rescan/integrity (optional)

This preserves operator-first execution and keeps each stage explicit.

## Registry and Provenance Requirements

- Source-of-truth purpose is `datasets.zarr_use='analysis'` (normalized from `zarr_purpose`).
- Archive creation must avoid identity churn:
  - stable output path convention
  - rescan updates existing row when path hash unchanged
- Stimulus imports remain append-only runs under `analysis/stimulus_runs`.

## Testing Contract

Minimum tests required before enabling by default:

- unit tests for input resolution and fail-closed ambiguity
- unit tests for dry-run no-mutation
- unit tests for purpose attr enforcement (`analysis`)
- integration test:
  - create archive -> detect -> refine -> registry scan
- regression test proving current single-camera workflow behavior remains stable

## Open decisions

- whether `detect_yolo` should keep compatibility creation path long-term or deprecate fully
- whether registry model auto-selection is part of creation/orchestration contract or separate phase
- when to enable multi-camera archive creation under this command (tracked separately)
