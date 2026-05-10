# Stage Catalog Design
<!-- contract-meta
status: design
last_verified: 2026-05-10
purpose: Define the canonical recording-stage vocabulary that registry/status,
pipeline, launcher, and future readers should converge on.
-->

Palette currently has several independent stage vocabularies:

- `src/fisheye/core/pipeline.py` uses pipeline command names such as `import`,
  `refine`, `keypoints_refine`, and `track`.
- `src/fisheye/cli/interactive_launcher.py` has a smaller UI-oriented stage
  list and tuner aliases.
- `src/fisheye/registry/step_cascade.py` has a registry invalidation DAG.
- `src/fisheye/registry/maintenance.py` defines the status-page step names.
- The registry wide status view has its own pivot column list.

The first phase is not a behavior change. The goal is to make the intended
vocabulary explicit in code, then migrate call sites one at a time.

## Terms

**Stage ID** is the stable registry/status identifier. This is the canonical
name used for query surfaces, manifests, and staleness records.

**Artifact family** is where outputs live in Zarr. A stage may write zero, one,
or multiple artifact families over time. Artifact paths are advisory in the
phase-1 catalog, not a full schema contract.

**Command/action** is how a user asks a tool to do work. Commands may use
legacy or UI-friendly names, but those should translate into a canonical stage
ID before updating registry/status state.

This separation matters because names such as `refine` are good command verbs
but poor registry identifiers. The registry should say `refined_detect`.

## Canonical Stage IDs

The canonical source lives in `src/fisheye/registry/stage_catalog.py`.

| Stage ID | Category | Aliases | Direct dependencies | Direct invalidations |
| --- | --- | --- | --- | --- |
| `raw` | core pipeline | `import` | | `background` |
| `calibration` | recording metadata | | | |
| `stimulus` | recording metadata | | | |
| `background` | core pipeline | | `raw` | `detect` |
| `detect` | core pipeline | | `background` | `detect_quality` |
| `detect_quality` | core pipeline | | `detect` | `refined_detect` |
| `refined_detect` | core pipeline | `refine` | `detect_quality` | `crop` |
| `crop` | core pipeline | | `refined_detect` | `keypoints`, `subject_masks` |
| `keypoints` | core pipeline | | `crop` | `refined_keypoints` |
| `refined_keypoints` | core pipeline | `keypoints_refine` | `keypoints` | `eye_masks`, `arena_assignment` |
| `eye_masks` | core pipeline | | `refined_keypoints` | `refined_eye_masks` |
| `refined_eye_masks` | core pipeline | | `eye_masks` | |
| `subject_masks` | core pipeline | | `crop` | `refined_subject_masks` |
| `refined_subject_masks` | core pipeline | | `subject_masks` | |
| `arena_assignment` | core pipeline | `assign_ids` | `refined_keypoints` | `tracks` |
| `tracks` | core pipeline | `track` | `arena_assignment` | |
| `dish_mask` | tuning | `mask` | | |
| `detection_tuning` | tuning | `threshold` | | |
| `keypoint_tuning` | tuning | | | |
| `subject_mask_tuning` | tuning | `subject-mask`, `subject_mask` | | |
| `eye_mask_tuning` | tuning | `eye-mask`, `eye_mask` | | |
| `subdish_mask_tuning` | tuning | `subdish`, `subdish-mask`, `subdish_mask` | | |

`downsample` is intentionally not in the canonical phase-1 registry/status
stage list. It exists in older pipeline/launcher code as an implementation
stage, but it is not currently a registry/status artifact boundary.

Some launcher tuner commands are intentionally not aliases because they collide
with core stage IDs. For example, `detect` can mean the detection stage or the
detection tuner depending on UI context. Ambiguous command names should stay
local to the command parser and should not be global stage aliases.

## Dependency Policy

The catalog uses registry/status artifact boundaries, not necessarily the
minimum command-line prerequisite checks in legacy launchers.

Two examples:

- `refined_detect` depends on `detect_quality` in the catalog, even though some
  legacy launchers only require `detect`. The catalog is documenting the modern
  intended review/QC flow.
- `tracks` depends on `arena_assignment`, not directly on `keypoints`. This
  leaves room for future multi-subject identity assignment without reworking
  registry semantics.

## Phase-1 Scope

Initial scope:

- Introduce `StageSpec` and the canonical `STAGE_SPECS`.
- Provide alias translation helpers.
- Add tests that lock uniqueness, registry coverage, and known legacy gaps.
- Document the migration targets.

First follow-up slice completed on 2026-05-10:

- `src/fisheye/core/pipeline.py` exposes `STAGE_CANONICAL_IDS` so runtime
  command-stage names translate to registry stage IDs.
- `src/fisheye/cli/interactive_launcher.py` records canonical IDs for launcher
  stage rows while preserving UI command names.
- The catalog includes derived-analysis stages for track kinematics, swim
  bouts, bout kinematics, eye angles, subject shape, tail kinematics, tail
  posture views, bout classification, and stimulus response.
- `src/fisheye/registry/maintenance.py` backfills presence-level status rows
  for those derived-analysis run families.
- Backfill also checks semantic freshness for current source refs:
  - `tail_kinematics_runs` must point at the current `subject_shape` run.
  - `tail_posture_view_runs` must point at the current `subject_shape` run and
    must match the current `tail_kinematics` run when that optional source attr
    is declared.
  - `bout_classification_runs` must point at the current tail posture view,
    track kinematics, and swim-bout runs.
  - `bout_kinematics_runs` must point at the current swim-bout run.
  - `eye_angle_runs` must point at the current refined-keypoint and refined
    subject-mask runs.
  - `subject_shape_runs` must point at the current refined subject-mask run.
  - `stimulus_response_runs` must point at the current stimulus, track
    kinematics, and swim-bout runs.
- Source freshness states render distinctly in the wide status view:
  `STALE` means a stored source ref no longer matches the current upstream run;
  `UNVER` means the source ref cannot be verified because source attrs are
  missing or the expected upstream source is unavailable.
- `recording_step_status_wide` and the status-page query layer expose those
  derived-analysis stages.

Still out of scope:

- Rename runtime pipeline commands or launcher UI commands.
- Make individual derived-analysis writers upsert their own status rows.
- Compute semantic freshness for the remaining presence-only derived-analysis
  families by comparing source refs/revisions against current upstream
  selections where the writers expose enough source identity.
- Redesign Zarr layout or artifact schemas.

Derived analysis runs such as track kinematics, swim bouts, bout kinematics,
eye angles, subject shape, tail kinematics, tail posture views, bout
classification, and stimulus responses now use the same catalog shape with
`category="derived_analysis"`. Tail behavior, bout-kinematics, eye-angle,
subject-shape, and stimulus-response runs now get source-ref freshness checks
during registry backfill. Other derived families still mostly detect whether a
latest run is present and do not yet decide whether that run is fresh relative
to its stored source refs.

## Migration TODO

Completed in the first migration pass:

- `src/fisheye/registry/maintenance.py` derives
  `RECORDING_STEP_NAMES` and `RECORDING_TUNING_STEP_NAMES` from the catalog.
- `src/fisheye/registry/db.py` uses the catalog to generate status-pivot
  columns for `recording_step_status_wide`; subject-mask and subject-mask
  tuning stages are now included in the wide view.
- `src/fisheye/registry/step_cascade.py` derives `STEP_DEPENDENTS` from the
  catalog invalidation map.

Remaining:

1. Writer-side status emission

   Derived-analysis writers should eventually call the shared status ledger when
   they create or fail runs, rather than relying only on registry backfill.

2. Source-ref freshness

   Status rows should distinguish "run exists" from "run is fresh relative to
   current upstream selections/revisions".

3. Registry status overview

   The wide view now consumes the catalog. The older
   `recording_step_overview` aggregate still has hand-written count columns and
   should be migrated separately if we keep that view.

## Validation

`tests/unit/fisheye/test_stage_catalog.py` locks the phase-1 contract:

- Canonical IDs are unique.
- Aliases are unambiguous and resolve to canonical IDs.
- Existing maintenance status names are covered by the catalog.
- Current pipeline/launcher stage definitions either resolve through the
  catalog or are explicitly documented as legacy implementation stages.
- Declared artifact families are unique.
