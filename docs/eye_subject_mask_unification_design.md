# Eye Subject-Mask Unification Design

<!-- design-meta
status: active
last_verified: 2026-04-01
-->

Purpose: define the target runtime/refined model for moving eye refinement under
the subject-mask stage family without losing the eye-specific geometry, review,
and provenance semantics that currently live in `refined_eye_masks_runs`.

## Scope

This design decides:

- the canonical write target for new eye-capable work
- the canonical refined schema target for eye components
- where eye ellipses, contours, and review payloads should live
- how cross-eye `eye_separation` should be represented
- the steady-state role of `refined_eye_masks_runs`
- the non-destructive migration path

This design does not by itself update runtime code or finalize every contract.

## Current State

Today the repo has two different refined-mask families:

- `refined_subject_masks_runs/<run>`
  - canonical refined/editable stage for body and swim bladder
  - component-local review, reasons, QC, and provenance are already implemented
- `refined_eye_masks_runs/<run>`
  - eye-specific refined stage with left/right masks, ellipse fits, contours,
    `eye_separation`, and eye-specific QA

That split is acceptable during transition, but it is not the desired
steady-state model for new multi-component work.

## Design Summary

1. New canonical refined authoring target for eye-capable work should be
   `refined_subject_masks_runs/<run>`.
2. The canonical refined eye-capable schema should be `subject_v1_lr`, with
   `eye_left` and `eye_right` as explicit component identities.
3. Eye-local geometry should live under
   `components/eye_left|eye_right/{geometry,contours}`.
4. Cross-eye derived values such as `eye_separation` should live in a run-level
   relation subtree, not duplicated into both eye components.
5. `refined_eye_masks_runs` should become a compatibility/adapter artifact in
   the long-term design, not a parallel canonical authoring surface.
6. Direct multi-source assembly should target `refined_subject_masks_runs`
   rather than introducing a required assembled raw subject-mask intermediate.
7. An assembled unified run is only valid after the subject-mask
   refinement/finalization step materializes the canonical QA, metrics,
   reasons, and review scaffolding.
8. Historical eye-mask runs remain readable and backfillable; no destructive
   archive rewrite is required.

## Canonical Write Targets

### Raw runtime target

The canonical raw stage remains:

- `subject_mask_runs/<run>`

For new eye-capable raw writers:

- use `subject_v1_lr` when the producer can guarantee anatomical left/right
  identity
- use `subject_v1_union` as a compatibility/raw bridge when the producer cannot
  guarantee LR identity

That means `subject_v1_union` remains valid for:

- legacy eye-only backfill
- raw union-eye models
- unlabeled pair-eye producers that are not yet trustworthy LR writers

### Refined target

The canonical refined target for new eye-capable work should be:

- `refined_subject_masks_runs/<run>`

Refined eye-capable runs should target:

- `label_schema_id = "subject_v1_lr"`
- `mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]`

Reason:

- refined eye geometry is inherently left/right-specific
- `subject_v1_union` is acceptable for raw compatibility and some exports, but
  it is too lossy as the refined canonical authoring surface
- refined workflows can legitimately promote a union/raw source into LR
  semantics using keypoints or other eye-specific refinement logic

## Assembly And Finalization Policy

Unified subject-mask assembly should not stop at "collect component masks into a
shared tensor."

Required policy:

- do not require an assembled raw `subject_mask_runs/<run>` intermediate for
  sparse multi-source workflows
- assemble directly into `refined_subject_masks_runs/<run>`
- always run the assembled result through the subject-mask
  refinement/finalization layer before treating it as a valid refined run

Why:

- other Palette refined artifacts are materialized QA/validation stages, not
  just storage containers
- the subject-mask refined stage is where component-local and run-level
  metrics, reasons, provenance, and review scaffolding become canonical
- skipping finalization would make `refined_subject_masks_runs` the only
  "refined" family that does not actually guarantee a refinement-time QA
  surface

Initial source policy for unified assembly:

- normal sources should be raw component producers such as `subject_mask_runs`
- one transitional already-refined source is explicitly allowed:
  `refined_eye_masks_runs`
- importing components from an existing `refined_subject_masks_runs` run is
  deferred; it is not part of the initial unification plan

## Canonical Refined Storage Shape

Eye masks remain canonical in the shared run-level tensor surface:

```text
refined_subject_masks_runs/
  <run>/
    masks_roi
    available_channels
    edit_applied
    metrics/
      mask_present
      area_px
      centroid_xy
      centroid_valid
      bbox_xyxy
      bbox_valid
    components/
      eye_left/
        provenance/
        reason_bytes
        reason
        metrics/
        geometry/
          ellipse_params
          ellipse_success
        contours/
          ptr
          len
          points_xy
      eye_right/
        provenance/
        reason_bytes
        reason
        metrics/
        geometry/
          ellipse_params
          ellipse_success
        contours/
          ptr
          len
          points_xy
    relations/
      eye_pair/
        metrics/
          separation_px
          separation_valid
```

## Eye-Local Arrays That Move Under Components

The following refined-eye outputs should migrate into
`refined_subject_masks_runs` when eye unification is implemented:

- left/right refined masks
  - canonical home remains run-level `masks_roi`
- per-eye ellipse parameters
  - canonical home:
    - `components/eye_left/geometry/ellipse_params`
    - `components/eye_right/geometry/ellipse_params`
- per-eye ellipse validity
  - canonical home:
    - `components/eye_left/geometry/ellipse_success`
    - `components/eye_right/geometry/ellipse_success`
- per-eye contour stores
  - canonical home:
    - `components/eye_left/contours/{ptr,len,points_xy}`
    - `components/eye_right/contours/{ptr,len,points_xy}`
- per-eye reason/review/QC payloads
  - canonical home:
    - `components/eye_left/`
    - `components/eye_right/`
  - the exact physical review payload carrier may remain the existing
    refined-subject review schema rather than requiring a new per-component
    `review/` subgroup immediately

Eye-local QC that is currently emitted in `refined_eye_masks_runs/metrics/`
should become component-local under `components/<eye>/metrics/` where the value
is truly per-eye, for example:

- axis ratio
- circularity
- probability summaries
- filter/connectivity/smoothing flags
- per-eye area or centroid deltas

Run-level `metrics/` should continue to hold only cross-component fixed-shape
summary arrays that already apply to every subject-mask component.

## Review And Registry Alignment

Eye unification should also align semantics above the storage layer.

Required behavior:

- eye components use the same canonical component identity vocabulary as the
  rest of subject-mask refinement:
  - `eye_left`
  - `eye_right`
- eye components use the same component review payload schema already used for
  other refined subject-mask components
- registry component rows should eventually project eye availability and review
  state from `refined_subject_masks_runs` when unified eye refinement is in use

This does not require removing specialized eye-specific quality/profile tables
immediately. It only means the canonical component review surface should stop
being eye-stage-specific once unified eye refinement is implemented.

## Contour Storage Decision

Contour storage should remain per-eye component-local.

Canonical location:

- `components/eye_left/contours/`
- `components/eye_right/contours/`

Canonical arrays:

- `ptr`
- `len`
- `points_xy`

Why this is the preferred design:

- contours are owned by one eye component at a time
- component-local storage avoids left/right-prefixed top-level array names in a
  unified stage
- it keeps edit ownership disjoint and component-scoped
- it avoids creating one mixed contour table that has to be repartitioned on
  every read

There is no need for a shared run-level contour index in v1.

## Cross-Eye Relation Decision

`eye_separation` should not be duplicated into both eye components and should
not be represented by inventing a fake mask component.

Canonical location:

- `relations/eye_pair/metrics/separation_px`
- `relations/eye_pair/metrics/separation_valid`

Why:

- `eye_separation` is a pairwise derived value, not an intrinsic property of
  `eye_left` alone or `eye_right` alone
- duplicating it under both eyes creates synchronization risk and ambiguous
  ownership
- introducing a fake `eye_pair` mask component would confuse the component
  vocabulary and the channel layout

The `relations/` subtree should be treated as the standard extension point for
cross-component derived values that are not owned by one component.

## Provenance And Update Policy

Eye components inside `refined_subject_masks_runs` should follow the same
component provenance model already used for body and swim bladder:

- `components/eye_left/provenance/`
- `components/eye_right/provenance/`

Required behavior:

- if eye components are seeded from `refined_eye_masks_runs`, provenance must
  point to `refined_eye_masks_runs`
- if eye components are seeded from raw `subject_mask_runs`, provenance must
  preserve the true source channels such as `["eyes_union"]`,
  `["eye_left"]`, `["eye_right"]`, or positional channel names
- subject-mask finalization must then record the assembled unified run's own
  `last_update_*` state without erasing the original component `source_*`
  lineage
- later edits update `last_update_*` without erasing the original `source_*`
  origin

This is necessary for mixed-source refined runs such as:

- `subject_body` from a SAM subject-mask run
- `eye_left` and `eye_right` from refined eye masks
- `swim_bladder` from a separate raw subject-mask source

## Role Of `refined_eye_masks_runs`

### During transition

During transition, `refined_eye_masks_runs` remains supported and may remain
the live eye-specific refinement surface until the unified eye path reaches
feature parity.

That means:

- historical archives keep using it unchanged
- the legacy failure-local eye review/edit tooling can keep targeting it when
  explicitly requested
- canonical manual eye review authority should move to
  `refined_subject_masks_runs`
- migration/backfill into subject-mask stages remains non-destructive

### Steady-state target

For new fully eye-capable subject-mask work, `refined_eye_masks_runs` should be
treated as a compatibility or adapter artifact derived from the canonical
subject-mask refined state, rather than as a second independent source of
truth.

Target steady-state:

- canonical editable refined state:
  - `refined_subject_masks_runs/<run>`
- optional compatibility/export artifact:
  - `refined_eye_masks_runs/<run>`

Important rule:

- do not allow both refined stage families to become independent mutable
  authoring surfaces for the same semantic run

If both artifacts exist for the same modern recording, one must be clearly
canonical and the other must be derived.

## Migration Phases

### Phase A: current transition

- keep `refined_eye_masks_runs` available for specialized failure/ellipse
  editing, but stop treating it as the default manual review authority
- keep `refined_subject_masks_runs` authoritative for body/swim refinement
- move canonical manual eye review onto `refined_subject_masks_runs` via a
  compatibility `subject_mask_runs` projection when needed
- prefer unified subject-mask component registry/query/operator surfaces for
  eye visibility, projecting legacy eye-stage rows when needed
- preserve provenance across projection and backfill

### Phase B: aligned storage

- add `eye_left` and `eye_right` component support to
  `refined_subject_masks_runs`
- support seeding those eye components from `refined_eye_masks_runs`
- add the `relations/eye_pair/` subtree for cross-eye derived values
- require subject-mask finalization after the seed/assembly step so the unified
  run has canonical refined QA and review metadata

### Phase C: canonical unification

- new eye-capable refined authoring writes canonical state to
  `refined_subject_masks_runs`
- direct assembly into `refined_subject_masks_runs` remains the preferred path;
  a merged raw subject-mask intermediate is not required
- `refined_eye_masks_runs` becomes an optional derived compatibility artifact
  for consumers that still require the legacy eye-specific layout

### Phase D: eventual simplification

- once all important consumers can read the unified refined-subject layout,
  compatibility materialization of `refined_eye_masks_runs` can become opt-in
  rather than routine

## Concrete Mapping From Current Refined-Eye Layout

| Current refined-eye payload | Unified target |
| --- | --- |
| `masks_roi[:, 0]` | `masks_roi[:, eye_left_channel]` |
| `masks_roi[:, 1]` | `masks_roi[:, eye_right_channel]` |
| `ellipse_params[:, 0, :]` | `components/eye_left/geometry/ellipse_params` |
| `ellipse_params[:, 1, :]` | `components/eye_right/geometry/ellipse_params` |
| `ellipse_success[:, 0]` | `components/eye_left/geometry/ellipse_success` |
| `ellipse_success[:, 1]` | `components/eye_right/geometry/ellipse_success` |
| `contour_left_ptr`, `contour_left_len`, `contours_left` | `components/eye_left/contours/{ptr,len,points_xy}` |
| `contour_right_ptr`, `contour_right_len`, `contours_right` | `components/eye_right/contours/{ptr,len,points_xy}` |
| `eye_separation` | `relations/eye_pair/metrics/separation_px` |
| eye-local QA arrays in `metrics/` | `components/eye_left|eye_right/metrics/` |
| run/source lineage attrs | run-level lineage plus component-scoped provenance |

`mask_probs_roi_refined` should remain optional debug/high-fidelity output if a
future unified implementation still needs it. It is not part of the required
canonical refined-subject surface.

## Non-Goals For This Design

This design does not yet decide:

- the exact registry projection rows for `relations/eye_pair/`
- whether a compatibility `refined_eye_masks_runs` artifact is materialized by
  default or only on request during the later transition phases
- the exact implementation seam for converting union/raw eye sources into
  canonical refined LR eye components
- the exact CLI/API that seeds a new assembled unified run before
  finalization
- whether training/export defaults should stay `subject_v1_union` even after
  runtime/refined authoring becomes canonically `subject_v1_lr`

## Recommended Follow-On Changes

1. Update [refined_subject_masks_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_masks_runs_contract.md) to include:
   - `components/eye_left|eye_right/geometry/ellipse_*`
   - `components/eye_left|eye_right/contours/`
   - `relations/eye_pair/metrics/separation_*`
2. Update [subject_mask_stage_unification_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_stage_unification_todo.md) and [subject_mask_refinement_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_refinement_todo.md) to mark the eye-target design questions as settled.
3. When swim-bladder canary work is ready, validate the first multi-component
   refined run before starting implementation of unified eye writes.

## Related Docs

- [subject_mask_stage_unification_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_stage_unification_todo.md)
- [subject_mask_refinement_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_refinement_todo.md)
- [refined_subject_masks_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_masks_runs_contract.md)
- [subject_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_training_artifact_contract.md)
- [segmentation_stage_split_review.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/segmentation_stage_split_review.md)
- [src/fisheye/docs/zarr_structure.md](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/docs/zarr_structure.md)
