# Subject Mask Refinement TODO

## Goal

Establish a first-class refinement and review model for subject-mask components
that covers:

- whole-subject/body masks
- swim-bladder masks
- eye masks

while preserving the current specialized eye-refinement workflow until the
unified subject-mask path is ready.

## Why This Is Needed

The current direction is:

- raw segmentation lives in `subject_mask_runs`
- future models may emit body, eye, and swim-bladder channels together
- operators will still need editable/refined artifacts for training-quality
  labels and QA

The eye-mask pipeline already demonstrates the right pattern:

- raw run
- refined canonical artifact
- review/edit tooling
- registry visibility for review state and quality

We want the same refinement affordances for body and swim bladder, without
breaking the current eye tools before the unified model is ready.

## Current State

- `subject_mask_runs` exists as the new raw component-mask stage.
- It can currently represent sparse eye-only compatibility runs and future dense
  multi-component runs.
- `subject_mask_training_artifact_contract.md` exists and the merged
  `subject_masks` exporter has been started.
- `refined_eye_masks_runs` remains the canonical edited/refined eye artifact.
- `refined_subject_masks_runs` now exists as a stage contract and runtime stage
  spec.
- A first review/editor entrypoint exists at
  [src/fisheye/tune/refined_subject_mask_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/refined_subject_mask_review.py).
- Subject-mask registry tables/views now exist for:
  - run-level quality/performance
  - component-level availability/review state
- The current operator workflow for tuning, batch propagation, materialization,
  and refinement is now documented in
  [subject_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_tuning_workflow.md).
- Traditional `subject_body` materialization now exists for canary-scale use,
  but execution scaling is still deferred to
  [traditional_subject_segmentation_scaling_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/traditional_subject_segmentation_scaling_todo.md).
- The main short-term blocker is no longer storage or registry design.
  The blocker is that we still do not have real body-mask and swim-bladder-mask
  data to curate, review, and export at scale.

## Immediate Remaining Work

This is the near-term rollout order that should happen before more schema work.

### 1. Get real body/swim-bladder mask data into a canary archive

- [ ] Decide the first source of body/swim-bladder masks:
  - manual painting in `refined_subject_masks_runs`
  - projected traditional masks
  - model-native `subject_mask_runs`
- [ ] Pick one canary training zarr and create the first non-eye refined masks.
- [ ] Confirm the component set for that canary:
  - `subject_body` only, or
  - `subject_body + swim_bladder`

### 2. Treat the first refined masks as the acceptance test

- [ ] Verify that the new review/editor is usable enough for body-mask work.
- [ ] Verify that saved refined masks write:
  - `masks_roi`
  - `edit_applied`
  - component `reason_bytes`
  - component review payloads
- [ ] Decide what “good enough to save” means for a first swim-bladder mask.

### 3. Delay downstream geometry until labels exist

- [ ] Do not start `subject_shape_runs` implementation until we have at least a
      small curated refined body-mask set.
- [ ] Do not design body/spline contour arrays beyond the current contract until
      we see what the first curated masks actually look like.

### 4. Keep eye migration deferred

- [ ] Continue using `refined_eye_masks_runs` for left/right eye editing.
- [ ] Do not move eye editing into `refined_subject_masks_runs` until body/swim
      workflows are proven.

## What Is Actually Missing Now

The missing pieces are now mostly workflow/data problems, not schema problems:

- real `subject_body` masks
- real `swim_bladder` masks
- review-time conventions for those components
- enough curated examples to validate export and future training paths

The storage and registry side is now far enough along that more schema churn is
less valuable than getting the first real body/swim-bladder labels into a
training zarr.

## Key Policy Decisions

### 1. Keep one raw subject-mask stage

Do not split raw runtime prediction immediately into separate stage families for
body, eye, and swim bladder.

The canonical raw dense-prediction stage should remain:

- `subject_mask_runs`

### 2. Add a refined subject-mask stage

We will likely want a future stage such as:

- `refined_subject_masks_runs`

This stage should hold edited/refined component masks for:

- `subject_body`
- `swim_bladder`
- possibly eye channels later

### 3. Defer eye migration

Do not migrate away from `refined_eye_masks_runs` yet.

For now:

- `refined_eye_masks_runs` stays first-class
- eye review/edit continues to operate there
- subject-mask unification should be designed so eye refinement can move under
  the subject-mask component model later without another schema reset

### 4. Keep eye geometry specialized for now

Even after raw masks are unified, eye refinement currently carries specialized
derived outputs that are not shared by other components, such as:

- left/right assignment
- contours
- ellipse parameters
- eye separation
- eye-specific reason/status handling

This is a good reason to defer the migration rather than forcing eye refinement
into a generic component mask stage too early.

## Proposed Stage Relationship

Near-term:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_eye_masks_runs/<run>      # eye-specialized path remains
```

Target medium-term:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_subject_masks_runs/<run>  # body/swim bladder editable masks
  -> refined_eye_masks_runs/<run>      # still specialized, may read from subject masks
  -> subject_shape_runs/<run>          # geometry derived from refined subject/body
```

Possible longer-term convergence:

```text
subject_mask_runs
  -> refined_subject_masks_runs
     -> refined_eye_masks_runs         # specialized derivative or sibling view
```

## Scope

This TODO covers:

- refined/editable runtime storage for body and swim bladder
- future unification path for eye refinement under the subject-mask component
  model
- registry/review implications
- downstream geometry implications for body masks

This TODO does not by itself define:

- final `subject_shape_runs` schema
- tail spline/curvature contracts
- migration timing for removing old eye-mask authoring flows

## Refined Subject-Mask Requirements

`refined_subject_masks_runs` should eventually support:

- edited binary masks per component
- component-specific review state
- component-specific reasons/status
- component-specific derived geometry where appropriate
- provenance back to raw `subject_mask_runs`

Minimum expected components:

- `subject_body`
- `swim_bladder`

Optional later:

- `eyes_union`
- `eye_left`
- `eye_right`

## Component-Specific Expectations

### Subject Body

Likely refined/derived outputs:

- refined binary mask
- contour(s)
- centroid
- major/minor axis or body orientation
- centerline/spline seeds or derived shape references
- reasons for rejection/edit state

This component is expected to feed later:

- `subject_shape_runs`
- tail segmentation / centerline workflows
- body-axis calculations

### Swim Bladder

Likely refined/derived outputs:

- refined binary mask
- centroid
- contour
- ellipse or blob summary if useful
- reasons for rejection/edit state

This component may later support:

- body-axis anchoring
- more stable anatomy-aware normalization

### Eyes

Keep current specialized refined path for now.

Future unification should make eyes a subject-mask component conceptually, but
the current refined eye artifact is richer than a plain component mask.

The unification target is therefore:

- common component model at the raw-mask level
- shared review/state semantics where possible
- eye-specific derived geometry retained until a general refinement framework
  can express it cleanly

## Review / Editing Model

We should eventually support a refinement/review workflow for:

- raw subject masks
- refined subject masks
- refined eye masks

Recommended distinction:

- raw subject-mask review:
  checks model output availability and coarse acceptability by component
- refined subject-mask review:
  checks edited canonical masks intended for training/downstream geometry
- refined eye-mask review:
  remains specialized because of left/right and ellipse geometry

## Registry Implications

We should not reduce this to one boolean on recordings.

The registry should represent:

1. coarse stage presence
   - `subject_masks`
   - `refined_eye_masks`
   - future `refined_subject_masks`
   - future `subject_shape`

2. component availability
   - `subject_body`
   - eye component(s)
   - `swim_bladder`

3. component review state
   - available / unavailable
   - pending / approved / rejected / not-applicable

4. component semantics
   - `eyes_union` vs `eye_left`/`eye_right`

This means the registry should eventually answer questions like:

- does this recording have any subject-mask run?
- does it have refined body masks?
- does it have refined swim-bladder masks?
- does it have refined eye geometry?
- which components are reviewed for training use?

## Phase 0: Contract Decisions

- [ ] Decide whether `refined_subject_masks_runs` is the canonical name.
- [ ] Define whether eye channels belong in the refined subject-mask contract in
      v1, or only body/swim bladder do.
- [ ] Define canonical source lineage attrs:
  - `source_subject_mask_run`
  - `source_crop_run`
  - optional `source_keypoints_run`

## Phase 1: Refined Subject-Mask Contract

- [x] Write a contract doc for `refined_subject_masks_runs`.
  See [refined_subject_masks_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_masks_runs_contract.md).
- [x] Define required arrays for edited masks and component validity.
- [x] Define attrs for:
  - `label_schema_id`
  - `mask_labels`
  - review payloads
  - component review payloads
  - reason/status summaries
- [ ] Define component-derived geometry payloads for:
  - `subject_body`
  - `swim_bladder`

## Phase 2: Review / Editor Surface

- [x] Add a refinement/review tool for `refined_subject_masks_runs`.
- [x] Support paint/edit workflows for:
  - body mask
  - swim-bladder mask
- [x] Add per-component review actions and status writing.
- [ ] Keep eye review tooling unchanged during this phase.
- [ ] Validate the new tool on real non-eye masks rather than only empty/copy
      initialized channels.

## Phase 3: Registry Integration

- [x] Implement subject-mask component registry tables/views from
      [subject_mask_registry_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_registry_contract.md).
- [x] Add coarse step projection for future `refined_subject_masks`.
- [ ] Add component latest views that can distinguish:
  - raw-only body/swim-bladder availability
  - refined body/swim-bladder availability
  - refined eye availability
- [ ] Wire those new registry surfaces into:
  - `check_recording_steps`
  - stale-step cascade / invalidation views
  - registry UI / TUI surfaces

## Phase 4: Geometry Integration

- [ ] Define `subject_shape_runs` to consume refined body masks, not raw body
      masks.
- [ ] Decide whether swim-bladder refined geometry should also feed
      `subject_shape_runs` or a sibling analysis stage.
- [ ] Define how body contour/spline outputs should reference refinement
      provenance.

## Phase 5: Eye Unification Path

- [ ] Define what “unify eye refinement under subject-mask component model”
      actually means:
  - shared component identity vocabulary
  - shared review payload schema
  - shared registry component rows
  - optional shared refined-mask storage
- [ ] Decide whether `refined_eye_masks_runs` becomes:
  - a specialized derivative of `refined_subject_masks_runs`, or
  - a long-lived sibling artifact with aligned component semantics
- [ ] Add a non-destructive migration/backfill plan when the target becomes
      clear.

## Acceptance Criteria

- [ ] Whole-subject and swim-bladder masks have a clear future refined/editable
      stage.
- [x] Registry can represent raw presence vs refined presence vs review state by
      component.
- [ ] `refined_eye_masks_runs` remains supported during transition.
- [ ] The future unification path for eyes is explicit enough to avoid another
      schema reset later.
- [ ] Downstream body-shape work is clearly anchored to refined body masks.

## Risks

- Trying to force eye refinement into a generic mask stage too early could
  discard important eye-specific geometry semantics.
- Splitting body/swim-bladder refinement into too many independent stage
  families could fragment reviewer workflows.
- If registry design is too coarse, future recording summaries will hide the
  important distinction between raw availability and refined training-ready
  masks.

## Related Docs

- [subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md)
- [refined_subject_masks_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_masks_runs_contract.md)
- [swim_bladder_patch_review_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_patch_review_design.md)
- [subject_mask_registry_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_registry_contract.md)
- [subject_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_training_artifact_contract.md)
- [subject_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_tuning_workflow.md)
- [pose_kinematics_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_kinematics_run_design.md)
- [review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md)
