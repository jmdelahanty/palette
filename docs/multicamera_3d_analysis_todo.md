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

## Timing-authority and temporal-mapping addendum (2026-09-02)

### Terms and current boundary

A **timing authority** declares what one frame or timestamp means, including its
recording and camera identity, clock domain, origin, timescale, reference point,
validity, and exact source digest. A **temporal mapping** is a separately derived
and sealed relationship between rows governed by two timing authorities. Clock
**synchronization** is evidence that those authorities share a usable timebase
within a declared offset, drift, and uncertainty bound. Merely finding similar
timestamp values is none of these things.

| Relationship | Current Palette status |
| --- | --- |
| Frames within one camera recording | Implemented through the immutable `palette.acquisition_frame_clock.v1` authority when acquisition metadata is available |
| Motion, body, eye, and other rows within that camera | Exact acquisition-frame joins are supported only when their bound row and timing identities agree |
| Bouts to framewise kinematics | Exact contracted frame-interval joins are supported; timestamps are measurements rather than the primary identity |
| Source frame to encoded video frame | A `palette.recording_frame_index.v1` sidecar can retain exact video/clip location and source-local frame identity |
| Historical chaser state to acquisition input | Implemented only as a sealed controller-input-provenance proxy |
| Chaser raster physically displayed during camera exposure | Unavailable for the historical cohort |
| General cross-camera frame alignment | Per-camera clock ingredients exist, but no completed session-level mapping successor is authoritative yet |

`palette.acquisition_frame_clock.v1` retains `recording_frame_id`, the complete
zero-based `parent_frame_index`, exact integer `camera_timestamp_ns` and
`system_timestamp_ns` values, explicit validity arrays, and digests of the
camera-bound source arrays. The camera timestamp is classified as
IEEE-1588/PTP TAI only when the recording provides combined configuration,
offset, latch-agreement, and camera-versus-host evidence. Otherwise its epoch
remains explicitly device-defined and unknown. PTP enablement alone does not
authorize cross-camera comparison.

The current `palette.provider_recording_timing_authority` schema version 1
binds one single-video recording, its acquisition frame clock, canonical
source-video metadata, camera identity, frame count, and nominal FPS. Its
numerical motion semantics remain acquisition-frame difference divided by
nominal FPS; it does not copy or relabel camera timestamp arrays. Derived track,
body, eye, and tail surfaces must independently seal their exact
row-to-acquisition-frame lineage.

### Target session-level alignment successor

The provisional `palette.multicamera_temporal_alignment.v1` product should be
an immutable, selector-ineligible successor with three normalized surfaces:

1. `camera_clock_bindings`: one row per participating camera authority,
   including session, recording, camera ID/serial, clock semantics, frame-clock
   digest, source manifest, timestamp reference point, and availability state.
2. `clock_transforms`: one row per camera and validity interval, mapping its
   clock to one declared session reference clock with scale, offset, drift
   model, uncertainty, evidence method, and exact input/output digests.
3. `frame_associations`: one row per requested camera-pair/frame relationship,
   including both frame identities and timestamps, corrected time difference,
   mapping policy, tolerance, uncertainty, and a typed `matched`, `unmatched`,
   or `ambiguous` disposition.

The frame-association grain should contain at least:

```text
session_id
alignment_run_id
source_recording_id
source_camera_id
source_acquisition_frame_id
source_timestamp_ns
target_recording_id
target_camera_id
target_acquisition_frame_id
target_timestamp_ns
corrected_delta_ns
mapping_policy_id
maximum_allowed_delta_ns
estimated_uncertainty_ns
mapping_status
source_clock_binding_sha256
target_clock_binding_sha256
clock_transform_sha256
alignment_receipt_sha256
```

This may be stored as partitioned Parquet for indexed associations while the
compact numeric per-camera frame clocks remain Zarr authorities. One sealed
manifest must inventory every part and bind the session roster, clock records,
mapping policy, schemas, row counts, file digests, software commit, and
validation receipt.

### Join and failure rules

- Same-camera consumers join canonical recording/camera/instance/frame keys;
  they do not need a timestamp-nearest mapping when exact frame identity exists.
- Cross-camera consumers may use only an exact sealed association or a declared
  tolerance policy. Raw nearest-timestamp matching is not an implicit fallback.
- Timestamp equality is insufficient without compatible clock domains,
  origins, timescales, reference points, and bound source identities.
- Every dropped, out-of-tolerance, or multiply eligible frame remains explicit
  `unmatched` or `ambiguous` evidence rather than disappearing.
- Mapping tolerances and uncertainty are scientific parameters and belong in
  the receipt; a viewer cannot change them.
- Shared session membership does not prove synchronized frames. Four cameras
  may coexist in one cohort manifest while remaining separately keyed and not
  cross-camera-joinable.
- A clock transform does not prove stimulus presentation. Display alignment
  additionally requires a presentation sequence, raster-buffer identity,
  presentation/vblank or photodiode timestamp, a transform into camera time,
  and a declared exposure reference and duration.

For historical GoodBatBadBat chaser recordings,
`source_acquisition_frame_index = recording_frame_id - 1` proves which Orange
acquisition identity Citrus held while producing a logged controller state. It
does not prove that state was visible during the named camera exposure. Those
products must retain `controller_input_provenance_proxy`,
`physical_presentation_verified=false`, and their existing scientific-use
classification.

### Timing implementation checklist

- [ ] Census every intended camera clock, its semantic classification, PTP
      evidence, timestamp coverage, monotonicity, cadence, and digest.
- [ ] Bind the exact session/camera roster and reject duplicated, missing, or
      foreign recording and camera identities.
- [ ] Select and document the session reference clock and prove any shared PTP
      grandmaster or other clock relationship rather than inferring it by name.
- [ ] Freeze clock-binding, transform, association, disposition, and receipt
      schemas before implementation.
- [ ] Define mapping tolerances from measured hardware evidence, including
      exposure reference and uncertainty, rather than from nominal FPS alone.
- [ ] Implement bounded mapping with explicit unmatched and ambiguous rows and
      no unrestricted nearest-neighbor fallback.
- [ ] Test offset, drift, dropped frames, duplicate candidates, clock resets,
      unknown epochs, mixed sessions, and post-binding source mutation.
- [ ] Run a controlled four-camera canary and compare all pairwise residuals,
      coverage, and dispositions before permitting scientific consumption.
- [ ] Keep the successor selector-ineligible until focused validation and every
      required repository CI check pass; promotion remains a separate decision.

## Design goals

- Migration-safe: additive schema changes first; avoid destructive rewrites.
- Operator-first: explicit dry-run/apply workflows and clear failure reasons.
- Backward compatible: single-camera commands continue to work unchanged.
- Auditable lineage: camera-level and 3D-derived outputs must be queryable in registry.

## Key product questions (decide before implementation)

- [ ] Archive layout choice:
  - Option A: one recording-level analysis Zarr containing per-camera groups.
  - Option B: one analysis Zarr per camera plus an optional recording-level aggregate.
- [x] Time-sync source-of-truth architecture:
  - per-camera acquisition clocks remain authority;
  - H5/frame mappings remain separately classified relationship evidence; and
  - cross-camera use requires the sealed transform and association successor
    described above.
- [ ] Select and validate the concrete reference clock, evidence thresholds,
  and mapping tolerances for each supported acquisition profile.
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
