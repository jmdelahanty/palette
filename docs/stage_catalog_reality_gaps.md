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

1. **`detect depends_on background` is wrong for the current pipeline.**
   Historical artifact: detection originally ran by background subtraction over
   imported frames. Current detection (YOLO over video/crops) does not need the
   background stage — confirmed by the maintainer 2026-07-01, and demonstrated by
   recordings (e.g. RedScare arena-1) where detect completed without background.
   Consequence today: `palette detect` reports a false `BLOCKED_BY_BACKGROUND`.
   Fix: remove the edge; the traditional (background-subtraction) detector checks for
   a background model at runtime per the design rule above.

2. **`keypoints depends_on background` (legacy DAG) is questionable for the same
   reason.** Maintainer statement 2026-07-01: the background model is not currently
   required. Verify whether any live keypoint path reads the background model; if not,
   remove the edge. If nothing in the live workflow needs `background`, consider
   marking the stage itself legacy/optional in the catalog.

3. **Artifact-name mismatch: catalog `background` vs on-disk `background_runs`.**
   Live tooling writes `background_runs`; the catalog artifact name says `background`.
   Currently bridged inside the oracle (`cli/palette.py`); the bridge should move into
   the catalog (or a canonical stage-id → group-name map next to it) so every consumer
   gets it.

4. **No recording-type stage profiles.** The catalog DAG is recording-type-agnostic,
   so `palette plan` recommends every structurally-possible stage (e.g. `dish_mask`,
   `arena_assignment` on an acquisition-crop-video training canary that will never run
   them). Fourteen candidate next-actions is not "trivial selection" for a small-model
   driver. Needs recording profiles (training / analysis / canary) that scope which
   stages apply.

5. **Deprecated stages appear in `plan` recommendations.** `eye_masks` /
   `eye_mask_tuning` were offered as next actions on RedScare. Once the severance adds
   deprecated markers to the catalog, the oracle must exclude deprecated stages from
   `next` (keep them visible in `status` as historical). ~Ten-line follow-up in
   `cli/palette.py`.

6. **Run-verbs need a general `--force` escape hatch.** Even with an accurate catalog,
   an operator/agent must be able to override a dependency gate with a loud warning in
   the envelope (`"forced": true`). The waist must never be *stricter than reality* with
   no exit.

7. **Crop has no registry-model runner idiom.** detect/keypoints select models via the
   registry; crop shims to `fisheye.utils.crop_batch` / `tracking.crop.crop_detections`
   with no registry-mediated selection. Consistency gap, not a catalog gap — parked here
   until it has a better home.

## Coordination note

`stage_catalog.py` edits are serialized: the eye-mask severance agent owns the file for
deprecation markers first; catalog-accuracy fixes (items 1–3) go in a subsequent slice.
