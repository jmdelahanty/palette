# Stage Catalog Design
<!-- contract-meta
status: design
last_verified: 2026-05-09
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

In scope:

- Introduce `StageSpec` and the canonical `STAGE_SPECS`.
- Provide alias translation helpers.
- Add tests that lock uniqueness, registry coverage, and known legacy gaps.
- Document the migration targets.

Out of scope:

- Rewire runtime pipeline execution.
- Change registry update behavior.
- Add derived analysis stages to the registry.
- Redesign Zarr layout or artifact schemas.

Derived analysis runs such as track kinematics, swim bouts, bout kinematics,
eye angles, and stimulus responses need their own registry/staleness policy.
They should eventually use the same catalog shape with
`category="derived_analysis"`, but they are deliberately not in phase 1.

## Migration TODO

1. `src/fisheye/core/pipeline.py`

   Translate command-stage names through the catalog before writing registry or
   status state. Keep command names available for CLI compatibility.

2. `src/fisheye/cli/interactive_launcher.py`

   Split UI command names from registry stage IDs. Keep ambiguous tuner
   commands local to the launcher instead of making them global aliases.

3. `src/fisheye/registry/step_cascade.py`

   Generate `STEP_DEPENDENTS` from `stage_catalog.invalidation_map()` after
   confirming the `detect -> detect_quality -> refined_detect` transition does
   not break runtime cascade expectations.

4. `src/fisheye/registry/maintenance.py`

   Replace `RECORDING_STEP_NAMES` and `RECORDING_TUNING_STEP_NAMES` with catalog
   projections. This should happen after status backfill tests cover all
   current steps.

5. Registry status wide view

   Generate the wide-view pivot columns from the catalog so subject-mask,
   tuning, stimulus, and calibration status do not drift from maintenance.

## Validation

`tests/unit/fisheye/test_stage_catalog.py` locks the phase-1 contract:

- Canonical IDs are unique.
- Aliases are unambiguous and resolve to canonical IDs.
- Existing maintenance status names are covered by the catalog.
- Current pipeline/launcher stage definitions either resolve through the
  catalog or are explicitly documented as legacy implementation stages.
- Declared artifact families are unique.
