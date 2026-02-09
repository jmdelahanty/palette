# Multi-Camera 3D Analysis TODO

Purpose: define a future-safe path for multi-camera recordings (for 3D pose and related analysis) without blocking current single-camera workflows.

Status: planning-only. Not required for current Phase 2/6 registry normalization progress.

## Why this doc exists

Today, analysis import/detect tooling is intentionally single-camera oriented.
For future 3D recordings, we need a model that can represent:

- one recording/session
- multiple camera streams
- per-camera 2D detections/keypoints
- synchronized multi-view associations
- optional 3D reconstruction outputs

This doc captures the work needed before that becomes production-ready.

## Current constraints (as of 2026-02-09)

- `import_recordings_analysis` is single-camera only.
- If a recording has multiple camera videos, command fails safe with an explicit message.
- Registry can represent recording-level biological metadata, but camera streams are not first-class entities yet.

## Design goals

- Migration-safe: additive schema changes first; avoid destructive rewrites.
- Operator-first: explicit dry-run/apply workflows and clear failure reasons.
- Backward compatible: single-camera commands continue to work unchanged.
- Auditable lineage: camera-level and 3D-derived outputs must be queryable in registry.

## Key product questions (decide before implementation)

- [ ] Archive layout choice:
  - Option A: one recording-level analysis Zarr containing per-camera groups.
  - Option B: one analysis Zarr per camera plus an optional recording-level aggregate.
- [ ] Time sync source of truth:
  - H5 frame mapping only, camera metadata clocks, or both.
- [ ] 3D output contract:
  - triangulated joints only vs. full reprojection/residual diagnostics.
- [ ] Minimum camera calibration contract:
  - required fields for intrinsics/extrinsics and versioning policy.

## Proposed data model (target)

### Recording-level entities

- [ ] Add camera stream entity (logical model): `camera_streams`
  - `camera_stream_id` (PK)
  - `recording_id` (FK)
  - `camera_id`
  - `video_path`
  - `stream_role` (e.g. `left`, `right`, `top`, `cam_1`)
  - `fps`, `resolution`, timestamps

- [ ] Add per-step run entities for multi-view workflows
  - `analysis_runs` extension or sibling typed tables for:
  - 2D detect per camera
  - 2D keypoints per camera
  - cross-view association
  - 3D triangulation/reconstruction

### Query convenience

- [ ] Add future view: `recording_multicam_overview`
  - recording/session identifiers
  - camera count and stream completeness
  - latest 2D/3D run status
  - calibration availability/version

## Proposed Zarr layout (Option A sketch)

- [ ] Keep one recording-level analysis archive with camera-scoped data:
  - `cameras/<camera_id>/detect_runs/...`
  - `cameras/<camera_id>/keypoints_runs/...`
  - `multiview/associations_runs/...`
  - `multiview/triangulation_runs/...`
  - shared timing/alignment groups

Notes:
- This keeps per-recording browseability high.
- Requires careful namespacing updates in existing single-camera modules.

## Implementation phases

### Phase MC1: read-only discovery and validation

- [ ] Add CLI to inspect recording camera topology (no writes).
- [ ] Validate required camera files and H5 camera mapping.
- [ ] Emit explicit readiness report for multi-camera processing.

### Phase MC2: camera-stream registration

- [ ] Add additive migration for camera stream tables.
- [ ] Backfill camera stream rows from existing recordings where possible.
- [ ] Add integrity checks for duplicate/missing stream assignments.

### Phase MC3: multi-camera analysis import

- [ ] Implement new command (separate from single-camera path):
  - create camera-scoped analysis structure
  - import stimulus/time-alignment metadata once per recording
  - record stream-level provenance
- [ ] Keep `--dry-run` and `--apply` with clear per-stream status.

### Phase MC4: per-camera 2D inference/refinement

- [ ] Batch run detect/keypoint per camera stream.
- [ ] Persist run linkage to camera stream entity in registry.
- [ ] Add review status at camera-stream granularity.

### Phase MC5: cross-view + 3D outputs

- [ ] Add association step across camera streams.
- [ ] Add triangulation step and quality metrics.
- [ ] Register 3D outputs as typed artifacts with lineage to camera-stream runs.

## Operator runbook requirements (before enabling)

- [ ] One-page runbook for multi-camera dry-run -> apply -> integrity checks.
- [ ] Explicit rollback steps and backup instructions.
- [ ] Defined behavior for partial failures (one camera failed, others succeeded).

## Testing requirements

- [ ] Unit tests for topology parsing and stream selection.
- [ ] Unit tests for schema migrations and integrity checks.
- [ ] Integration test with synthetic 2-camera recording fixture.
- [ ] Regression tests proving single-camera behavior is unchanged.

## Non-goals for first 3D milestone

- [ ] Real-time online multi-camera inference.
- [ ] Automatic calibration solving from scratch.
- [ ] Full UI support for 3D review (CLI/report first).

## Exit criteria for first usable 3D release

- [ ] Can ingest one recording with >=2 camera streams.
- [ ] Can run per-camera 2D inference and refinement.
- [ ] Can produce one triangulated 3D output run with quality summary.
- [ ] Registry can answer:
  - "Which recordings have complete camera coverage for 3D?"
  - "Which recordings have valid 3D runs and with what calibration version?"
