# Stage catalog vs. reality — known gaps

<!-- contract-meta
status: active
created: 2026-07-01
owner: jeremy
purpose: running input list for the "make stage_catalog load-bearing" task
-->

`registry/stage_catalog.py` is the intended single source of truth for the stage DAG,
and the `palette` narrow waist (see `docs/palette_cli_narrow_waist_design.md`) now
*enforces* it at run time. That enforcement surfaces every place the catalog disagrees
with how the pipeline actually works. This doc collects those gaps as they're found;
each entry is an input to the catalog-accuracy task, not a bug in the waist.

**Design rule adopted 2026-07-01:** the catalog encodes only *universal structural*
dependencies. Method-conditional prerequisites (e.g. a detector variant that needs a
background model) are runtime preconditions owned by the runner, surfaced through the
envelope's `blocked` + `next_hints` mechanism. Do not add conditional edges to the
catalog.

## Gaps

1. **DONE 2026-07-02: `detect depends_on background` is wrong for the current pipeline.**
   Historical artifact: detection originally ran by background subtraction over
   imported frames. Current detection (YOLO over video/crops) does not need the
   background stage — confirmed by the maintainer 2026-07-01, and demonstrated by
   recordings (e.g. RedScare arena-1) where detect completed without background.
   The catalog edge was removed. The frozen legacy `core.pipeline.Pipeline`
   still carries `detect: ["background"]`; that intentional catalog-vs-legacy
   difference is tracked in `tests/unit/fisheye/test_stage_catalog_drift.py`.
   Traditional background-subtraction detection continues to check for a
   background model at runtime per the design rule above.

2. **RECONCILED 2026-07-02: `keypoints depends_on background` was already fixed.**
   The live catalog now has `keypoints depends_on ("crop",)` only. The frozen
   legacy `core.pipeline.Pipeline` still carries `keypoints: ["crop", "background"]`
   because traditional keypoints historically required a background model. That
   mismatch is intentional and covered by
   `tests/unit/fisheye/test_stage_catalog_drift.py`.

3. **BRIDGED 2026-07-02: artifact-name mismatch: catalog `background` vs on-disk `background_runs`.**
   Live tooling writes `background_runs`; the catalog artifact name says `background`.
   This is bridged inside the oracle (`cli/palette.py`) and works today. Moving
   canonical artifact-family ownership into the catalog is deferred because it
   would ripple through every `artifact_families` consumer. Track catalog-side
   canonicalization as a later architecture cleanup, not part of this bug-fix slice.

4. **No recording-type stage profiles.** The catalog DAG is recording-type-agnostic,
   so `palette plan` recommends every structurally-possible stage (e.g. `dish_mask`,
   `arena_assignment` on an acquisition-crop-video training canary that will never run
   them). Fourteen candidate next-actions is not "trivial selection" for a small-model
   driver. Needs recording profiles (training / analysis / canary) that scope which
   stages apply.

5. **DONE 2026-07-02: deprecated stages appear in `plan` recommendations.**
   The oracle already excludes `StageSpec.deprecated` stages from `next` while
   keeping them visible in `status`. `eye_masks` and `refined_eye_masks` were
   already marked deprecated; `eye_mask_tuning` is now marked deprecated as the
   remaining dead-surface tuning stage. The broader question of whether live
   interactive tuning stages (`detection_tuning`, `keypoint_tuning`,
   `subject_mask_tuning`) belong in `plan next` is part of recording profiles
   (gap item 4), not this slice.

6. **DONE 2026-07-02: run-verbs need a general `--force` escape hatch.** Even with an accurate catalog,
   an operator/agent must be able to override a dependency gate with a loud warning in
   the envelope (`"forced": true`). The waist must never be *stricter than reality*
   with no exit. The `palette detect`, `palette crop`, and `palette keypoints`
   run verbs now accept `--force`; forced dependency overrides record the original
   `blocked_by` and `provenance.forced_dependency_overrides`.

7. **Crop has no registry-model runner idiom.** detect/keypoints select models via the
   registry; crop shims to `fisheye.utils.crop_batch` / `tracking.crop.crop_detections`
   with no registry-mediated selection. Consistency gap, not a catalog gap — parked here
   until it has a better home.

## Coordination note

`stage_catalog.py` edits are serialized: the eye-mask severance agent owns the file for
deprecation markers first; catalog-accuracy fixes (items 1–3) go in a subsequent slice.
