# Crimson Palette Integration Acceptance Checklist

Date anchored: 2026-04-10

Purpose: provide a concrete Palette-side validation checklist for Crimson
adoption of:

- explicit keypoint heading metadata
- sparse refined-detect storage
- current track motion, swim-bout, and subject-shape-derived analysis surfaces

This checklist is intentionally operator-facing. It uses known migrated archives
and exact commands so the integration can be checked quickly after Crimson
changes land.

## Scope

In scope:

- keypoint heading resolution from `pose_schema.metadata.heading_computation`
- legacy-label tolerance for older `bladder` runs
- refined detect read path from `refined_detect_runs/<run>/instances`
- detect audit access from `refined_detect_runs/<run>/source_detections`

Out of scope:

- unconstrained multi-subject tracking inside one arena
- detect retune parity in arena-aware mode
- keypoint or detect registry redesign
- first-class clipped analysis-Zarr review. Clipped analysis shells require a
  finalized collection resolver before Crimson should flatten clip-local runs
  onto a parent recording timeline. See
  `docs/clipped_recording_consumer_mapping_contract.md`.

## Contract References

Read these first if the expected behavior is unclear:

- `~/gitrepos/contracts/palette-crimson/keypoint_heading_computation.md`
- `~/gitrepos/contracts/palette-crimson/keypoint_read.md`
- `~/gitrepos/contracts/palette-crimson/detect_bbox_read.md`
- `~/gitrepos/contracts/palette-crimson/refined_detect_manual.md`
- `~/gitrepos/contracts/palette-crimson/track_motion_read.md`
- `docs/keypoint_heading_computation_contract.md`
- `docs/refined_detect_sparse_instances_schema.md`
- `docs/detection_refinement_workflow.md`
- `docs/crimson_track_motion_read_contract.md`
- `docs/clipped_recording_consumer_mapping_contract.md`

## Test Archives

### Keypoints

Analysis archive:

- `/nvme1/recordings/2026-01-28T21-47-47Z_arena_1_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_analysis.zarr`

Training archive:

- `/nvme1/recordings/2026-01-28T21-47-47Z_arena_1_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_training.zarr`

Known runs:

- analysis raw keypoints:
  `keypoints_runs/keypoints_2026-02-27_23-12-20`
- analysis refined keypoints:
  `refined_keypoints_runs/refined_keypoints_2026-03-02_13-43-33`
- training raw keypoints:
  `keypoints_runs/keypoints_2026-02-04_17-31-19`
- training refined keypoints:
  `refined_keypoints_runs/refined_keypoints_2026-02-04_12-41-55_traditional_v2_seed`

### Detect

Analysis archive:

- `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_analysis.zarr`

Training archive:

- `/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_training.zarr`

These archives were migrated to the sparse refined-detect structure and are
good integration canaries.

## Section A: Keypoint Metadata Presence

### A1. Canonical heading metadata on analysis raw run

Run:

```bash
jq '.attributes.pose_schema.metadata.heading_computation' \
  /nvme1/recordings/2026-01-28T21-47-47Z_arena_1_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_analysis.zarr/keypoints_runs/keypoints_2026-02-27_23-12-20/zarr.json
```

Pass criteria:

- payload exists
- `enabled == true`
- `direction_from.label == "swim_bladder"`
- `dependent_keypoints == ["swim_bladder", "eye_left", "eye_right"]`

### A2. Canonical heading metadata on analysis refined run

Run:

```bash
jq '.attributes.pose_schema.metadata.heading_computation' \
  /nvme1/recordings/2026-01-28T21-47-47Z_arena_1_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_analysis.zarr/refined_keypoints_runs/refined_keypoints_2026-03-02_13-43-33/zarr.json
```

Pass criteria:

- same payload as A1
- confirms refined runs preserve skeleton-owned heading metadata

### A3. Legacy label tolerance on older training raw run

Run:

```bash
jq '.attributes.pose_schema.metadata.heading_computation' \
  /nvme1/recordings/2026-01-28T21-47-47Z_arena_1_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_training.zarr/keypoints_runs/keypoints_2026-02-04_17-31-19/zarr.json
```

Pass criteria:

- payload exists
- `direction_from.label == "bladder"`
- `dependent_keypoints == ["bladder", "eye_left", "eye_right"]`

Interpretation:

- Crimson should trust the metadata as written
- Crimson should not rewrite `bladder` to `swim_bladder` on its own

### A4. Extended skeleton excludes non-heading points

Run:

```bash
jq '.attributes.pose_schema.metadata.heading_computation' \
  /nvme1/recordings/2026-01-28T21-47-47Z_arena_1_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_training.zarr/refined_keypoints_runs/refined_keypoints_2026-02-04_12-41-55_traditional_v2_seed/zarr.json
```

Pass criteria:

- `direction_from.label == "swim_bladder"`
- `dependent_keypoints == ["swim_bladder", "eye_left", "eye_right"]`
- `snout_tip` is not in `dependent_keypoints`
- `tail_tip` is not in `dependent_keypoints`

Interpretation:

- editing `snout_tip` or `tail_tip` alone must not trigger dashed
  candidate-heading recomputation in Crimson

## Section B: Expected Crimson Keypoint Behavior

Pass criteria:

1. Crimson resolves heading metadata in this order:
   - `heading_computation_override`
   - `pose_schema.metadata.heading_computation`
   - deprecated `heading_computation`
2. Crimson does not infer heading semantics from hardcoded label names when
   metadata is present.
3. Dashed candidate heading is recomputed only when edited labels intersect
   `dependent_keypoints`.
4. If heading metadata is absent or disabled, Crimson suppresses dashed
   candidate-heading preview unless it is intentionally running a temporary
   legacy fallback.

## Section C: Refined Detect Storage Verification

### C1. Analysis refined detect archive uses sparse surfaces

Run:

```bash
scripts/py -m fisheye.utils.inspect_refined_detect_run \
  /nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_analysis.zarr
```

And:

```bash
find /nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_analysis.zarr/refined_detect_runs/refined_detect_2026-02-09_16-40-49 -maxdepth 1 -mindepth 1 -printf '%f\n' | sort
```

Pass criteria:

- inspector shows `resolved_group == "refined"`
- inspector shows populated `Instances` summary
- inspector shows `Source detections: available: True`
- directory listing includes:
  - `instances`
  - `source_detections`
- directory listing does not show dense root array names such as:
  - `frame_indices`
  - `entity_ids`
  - `status_codes`
  - `source_kind_codes`

### C2. Detect metadata confirms sparse-first contract

Run:

```bash
jq '.attributes | {curated_primary_surface, refined_storage_semantics, summary_statistics}' \
  /nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_analysis.zarr/refined_detect_runs/refined_detect_2026-02-09_16-40-49/zarr.json
```

Pass criteria:

- `curated_primary_surface == "instances"`
- `refined_storage_semantics == "sparse_instances_v1"`
- `summary_statistics.source_detection_candidates` is present

### C3. Training refined detect archive is also sparse-first

Run:

```bash
scripts/py -m fisheye.utils.inspect_refined_detect_run \
  /nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_training.zarr
```

Pass criteria:

- `resolved_group == "refined"`
- `state == "pending"` after migration reset
- `Instances` preview is present
- `Source detections` preview is present

## Section D: Expected Crimson Detect Behavior

Pass criteria:

1. Primary refined detect reads come from:
   - `refined_detect_runs/<run>/instances`
2. Candidate/audit reads come from:
   - `refined_detect_runs/<run>/source_detections`
3. Crimson does not treat subgroup-era `manual -> interpolated -> filtered` as
   the normal current refined-detect model.
4. Legacy subgroup fallback may remain for historical archives, but it should
   not override current sparse refined runs.
5. If Crimson still exposes operator dataset labels, the active refined detect
   label should map to the sparse refined surface, not removed dense-root or
   subgroup-era storage.

## Section E: Final Integration Sign-Off

The integration is ready to sign off when all of the following are true:

- keypoint heading metadata is present on the canary archives above
- Crimson uses metadata-driven heading resolution rather than hardcoded fish
  assumptions when metadata is present
- `traditional_v2` non-heading points do not affect candidate heading
- Crimson loads migrated refined detect runs from `instances`
- Crimson can still open historical fallback archives if needed

## Notes

- The current Palette backfill intentionally preserves run-consistent legacy
  labels for older runs, such as `bladder`, instead of rewriting them to
  `swim_bladder`.
- This is acceptable because the heading contract is now metadata-driven.
- Palette-side schema and backfill work was completed on 2026-04-10; this
  checklist exists to validate the downstream Crimson adoption pass.
