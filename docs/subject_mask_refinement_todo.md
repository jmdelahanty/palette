# Subject Mask Refinement TODO

## Goal

Establish a first-class refinement and review model for subject-mask components
that covers:

- whole-subject/body masks
- swim-bladder masks
- eye masks

while preserving the current specialized eye-refinement workflow until the
unified subject-mask path is ready, and treating legacy eye-specific stages as
transition compatibility inputs rather than a second canonical mask family.

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
- `refined_eye_masks_runs` remains the current eye-specific editing surface
  during the transition to subject-mask unification, but registry/query and
  operator-facing component views should treat it as a compatibility source for
  unified subject-mask eye components rather than a parallel canonical stage.
- `refined_subject_masks_runs` now exists as a stage contract and runtime stage
  spec.
- A first review/editor entrypoint exists at
  [src/fisheye/tune/refined_subject_mask_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/refined_subject_mask_review.py).
- A scheduler-aware non-UI apply entrypoint now exists at
  [src/fisheye/refinement/refine_subject_masks.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/refinement/refine_subject_masks.py),
  and the pipeline now exposes it as the `refined_subject_masks` stage.
- A direct multi-source assembler/finalizer now exists at
  [src/fisheye/refinement/assemble_refined_subject_masks.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/refinement/assemble_refined_subject_masks.py)
  for sparse body/eye/swim workflows that should seed
  `refined_subject_masks_runs/<run>` directly rather than creating an
  assembled raw intermediate first.
- Subject-mask registry tables/views now exist for:
  - run-level quality/performance
  - component-level availability/review state
- Component-scoped provenance is now defined and written for:
  - raw `subject_mask_runs`
  - refined `refined_subject_masks_runs`
  - merge/projection utilities
  - legacy backfill paths
- The current operator workflow for tuning, batch propagation, materialization,
  and refinement is now documented in
  [subject_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_tuning_workflow.md).
- Traditional `subject_body` materialization now exists for canary-scale use,
  but execution scaling is still deferred to
  [traditional_subject_segmentation_scaling_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/traditional_subject_segmentation_scaling_todo.md).
- A body-only canary refined run has now been validated on 2026-03-31:
  - archive:
    - `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`
  - source run:
    - `subject_masks_canary_sam_points_body_eyes_001`
  - refined run:
    - `refined_subject_masks_canary_sam_points_body_001`
  - batch apply result:
    - `changed_roi_count = 0`
    - `noop_roi_count = 227`
- A first assembled multi-source refined canary has now been created on
  2026-04-01:
  - archive:
    - `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`
  - body source run:
    - `subject_masks_canary_sam_points_body_eyes_001`
  - eye source run:
    - `subject_masks_from_refined_eye_masks_2026-02-12_19-51-24`
  - swim source run:
    - `traditional_swim_bladder_masks_canary_001`
  - refined run:
    - `refined_subject_masks_canary_body_eyes_swim_001`
  - canonical schema:
    - `label_schema_id = "subject_v1_lr"`
    - `mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]`
- The main short-term blocker is no longer storage or registry design.
  The blocker is that we still do not have real body-mask and swim-bladder-mask
  data to curate, review, and export at scale.

## Immediate Remaining Work

This is the near-term rollout order that should happen before more schema work.

### 1. Get real body/swim-bladder mask data into a canary archive

- [x] Decide the first source of body/swim-bladder masks:
  - first validated refined-body canary uses model-native raw
    `subject_mask_runs` input from the SAM canary source run
- [x] Pick one canary training zarr and create the first non-eye refined masks.
- [x] Confirm the component set for that canary:
  - first validated non-eye canary was `subject_body` only
  - first assembled multi-source refined canary now includes:
    - `subject_body`
    - `eye_left`
    - `eye_right`
    - `swim_bladder`
  - next acceptance target is no longer storage assembly
  - next acceptance target is component-local swim-bladder review/approval
    conventions on real data

### 2. Treat the first refined masks as the acceptance test

- [x] Verify that the new review/editor is usable enough for body-mask work.
- [x] Verify that saved refined masks write:
  - `masks_roi`
  - `edit_applied`
  - component `reason_bytes`
  - component review payloads
- [x] Decide what “good enough to save” and “good enough to approve” mean for
      the first swim-bladder masks.
  See
  [swim_bladder_review_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_review_policy.md).
- [x] Decide whether the current threshold/blob swim-bladder tuner is good
      enough for canary use, or whether we should switch to the boundary-based
      method family in
      [swim_bladder_polar_boundary_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_polar_boundary_design.md)
      before scaling up curation.
  - boundary-based `swim_bladder_polar_boundary_v1` is now implemented in the
    tuner and materializer
  - first raw canary source run
    `subject_mask_runs/traditional_swim_bladder_masks_canary_001` was
    materialized successfully on 2026-04-01

### 3. Delay downstream geometry until labels exist

- [ ] Do not start `subject_shape_runs` implementation until we have at least a
      small curated refined body-mask set.
- [ ] Do not design body/spline contour arrays beyond the current contract until
      we see what the first curated masks actually look like.

### 4. Keep eye migration deferred

- [ ] Continue using `refined_eye_masks_runs` for left/right eye editing as the
      current specialized editor surface.
- [ ] Do not move eye editing into `refined_subject_masks_runs` until body/swim
      workflows are proven.
- [ ] Prefer unified subject-mask component registry/query/operator surfaces for
      eye availability and review visibility, even while eye-specific editing
      remains transitional.

## What Is Actually Missing Now

The missing pieces are now mostly workflow/data problems, not schema problems:

- broader curated `subject_body` coverage beyond the first canary
- curated/refined `swim_bladder` masks and review conventions
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

The refined editable stage for this work is now:

- `refined_subject_masks_runs`

This stage should hold edited/refined component masks for:

- `subject_body`
- `swim_bladder`
- possibly eye channels later

Policy note:

- sparse multi-source workflows should not require an assembled raw
  `subject_mask_runs/<run>` intermediate
- instead, component sources should seed `refined_subject_masks_runs/<run>`
  directly and then pass through the subject-mask refinement/finalization step
- this keeps `refined_subject_masks_runs` aligned with other Palette refined
  artifacts, where the refined run is a QA/metrics materialization stage rather
  than just an assembled container

Current implementation note:

- the shipped subject-mask review/assembly helpers still load source inputs
  from `subject_mask_runs`
- new raw eye orchestration now writes a companion eye-only
  `subject_mask_runs/<run>` immediately after successful `eye_masks_runs/<run>`
  completion, using `subject_v1_union`
- when eye content starts in legacy eye stages, the implemented bridge is to
  project/backfill a compatibility `subject_mask_runs/<run>` first and then
  assemble or refine from that subject-mask source
- direct `refined_eye_masks_runs` -> `refined_subject_masks_runs` seeding is a
  future extension, not the current code path

### 3. Defer eye migration

Do not migrate away from `refined_eye_masks_runs` yet, but also do not treat it
as a second canonical refined family for new operator-facing status.

For now:

- `refined_eye_masks_runs` stays supported as the current specialized
  eye-editing surface
- eye review/edit continues to operate there for now
- subject-mask unification should be designed so eye refinement can move under
  the subject-mask component model later without another schema reset
- registry/query/operator surfaces should prefer unified subject-mask component
  rows for eye visibility, projecting legacy eye-stage data when necessary
- contract target: unified assembly may later allow eye components to seed
  directly from `refined_eye_masks_runs`
- current implementation: the first assembled refined canary still goes
  through a projected compatibility `subject_mask_runs` eye source:
  `subject_masks_from_refined_eye_masks_2026-02-12_19-51-24`

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
  -> subject_mask_runs/<run>            # dense/raw or component source snapshots
  -> refined_subject_masks_runs/<run>   # body-only refined path now exists
  -> refined_eye_masks_runs/<run>      # eye-specialized path remains
```

Current sparse-source assembly path:

```text
component/raw sources
  -> refined_subject_masks_runs/<run>  # direct assembly + subject-mask finalization
  -> refined_eye_masks_runs/<run>      # still specialized during transition
  -> subject_shape_runs/<run>          # future geometry derived from refined subject/body
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

## Recommended Refinement Model

The preferred refinement model is:

- mask-first
- component-local
- idempotent

Meaning:

- the canonical editable truth is the refined binary mask for each component
- save operations should only update the touched component(s)
- save operations should recompute only the derived metadata for the touched
  component(s)
- repeated saves of unchanged pixels should be a no-op
- unrelated components should remain untouched

This should hold whether refinement is driven from Palette-native review tools,
Paintera, or a future Crimson editing path.

## Settled Near-Term Design Decisions

These decisions are settled for the first implementation slice unless a later
contract review deliberately changes them.

### 1. Metrics location split

Use:

- run-level `metrics/` for common cross-component arrays
- `components/<component>/metrics/` for component-specific QC

Near-term run-level metrics remain:

- `mask_present`
- `area_px`
- `centroid_xy`
- `centroid_valid`
- `bbox_xyxy`
- `bbox_valid`

Component-aware QC should live under the component subtree so body, swim
bladder, and future eye metrics can evolve independently.

The intent is:

- common fixed-shape geometry like centroid and bbox stays at run-level
  `metrics/`
- component-specific QC and anatomy-dependent quality signals stay under
  `components/<component>/metrics/`

### 2. Body-mask scope metadata

`subject_body` should use component-local scope metadata with:

- a versioned component schema id
- explicit policy attrs

Recommended shape:

- `component_schema_id = "subject_body_vX"`
- `anatomical_scope = ...`
- `pectoral_fin_policy = ...`

This should live under the `subject_body` component subtree rather than as a
run-global convention.

### 3. First-wave refined body QC metrics

The first refined body implementation should recompute only the stable topology
metrics that do not depend on unresolved fin/skeleton policy:

- `mask_present`
- `area_px`
- `component_count`
- `largest_component_fraction`
- `hole_count`
- `hole_area_fraction`

Boundary-roughness and skeleton-branching metrics are intentionally deferred to
the next phase after the first saveback path is proven.

## Metrics vs Geometry Structure

Recommended structure split:

- keep small fixed-shape QC and summary arrays in `metrics/`
- keep richer component-specific structured outputs in `geometry/`

Use `metrics/` for things like:

- `mask_present`
- `area_px`
- `component_count`
- `largest_component_fraction`
- other fixed-shape values that are useful for filtering, QA, and registry
  projection

Use `geometry/` for things like:

- contours
- centroids if we want a richer geometry namespace
- bbox / axis summaries if they grow beyond simple scalar fields
- splines / centerlines
- ellipse parameters

Recommended rule of thumb:

- if the value is a small fixed-shape QC summary, prefer `metrics/`
- if the value is a richer derived shape/structure payload, prefer
  `components/<component>/geometry/`

This keeps the refined subject-mask stage consistent with the broader design:

- `metrics/` is embedded within the run as the stable QC-summary surface
- `geometry/` is the extension point for evolving component-specific structure

## Recommended QC Metrics For Refined Masks

For the current single-subject, low-occlusion recordings, body-mask QC should
prioritize topology and refinement fitness over generic shape heuristics.

Recommended priority order:

### 1. Topology / validity checks first

These are the most interpretable first-pass checks for current body masks:

- `component_count`
- `largest_component_fraction`
- `hole_count`
- `hole_area_fraction`
- `n_branch_points`
- `n_endpoints`

Current expected body-mask topology:

- one connected component
- no holes
- no skeleton branches
- two skeleton endpoints

If these fail, that is usually a segmentation/refinement quality problem, not a
normal biological outcome for the current recordings.

### 2. Boundary roughness next

Once topology is acceptable, the next useful QC class is boundary noise:

- `sigma_noise`
  - contour-to-smoothed-contour deviation
- `curvature_var`
  - curvature variance on a suitably smoothed/resampled contour
- `ipr`
  - isoperimetric ratio, as a coarse malformed-mask screen
- `solidity`
  - convexity/concavity screen

Recommended policy:

- treat `sigma_noise` as the primary boundary-roughness metric
- treat `curvature_var`, `ipr`, and `solidity` as supporting signals rather
  than the first gate

### 3. Temporal consistency later

Temporal consistency metrics are useful, but they belong to a later
track-aware/sequence-aware QC layer rather than the first core refined-mask
contract.

Examples:

- `temporal_iou`
- `centroid_displacement`

These should likely live in a later analysis/QC stage rather than in the
canonical per-mask refined stage.

## Recommended QC Storage Split

Run-level `metrics/` should hold common fixed-shape arrays that make sense
across components, such as:

- `mask_present`
- `area_px`
- `bbox`
- `centroid`

Component-aware QC should prefer `components/<component>/metrics/`, for
example:

- `component_count`
- `largest_component_fraction`
- `hole_count`
- `hole_area_fraction`
- `sigma_noise`
- `curvature_var`
- `ipr`
- `solidity`
- `n_branch_points`
- `n_endpoints`

Richer structured derived outputs should live under
`components/<component>/geometry/`, for example:

- contours
- skeleton graphs
- ellipse parameters
- splines / centerlines

Recommended policy for flags:

- store raw QC metrics as the canonical truth
- optionally store derived flags/reasons with a versioned QC policy
- do not make a single `flagged` boolean the only persisted truth

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

Body-mask scope note:

- some datasets may intentionally include visible pectoral fins inside
  `subject_body`
- other datasets may not have enough resolution to capture those fins
- this should be treated as a resolution/annotation-scope choice, not
  necessarily a mask defect

Implication:

- raw body-mask topology and raw skeleton branching cannot be treated with one
  universal rule across all datasets
- if fins are intentionally included, a raw skeleton may legitimately contain
  side branches
- downstream body-axis/centerline derivation should therefore distinguish:
  - raw shape/skeleton structure
  - pruned main-path body axis

Recommended metadata direction:

- add a versioned descriptor for `subject_body` rather than relying on implicit
  conventions
- do not treat fin inclusion as a new top-level component label by itself
- keep this descriptor component-local so downstream QC/geometry code can decide
  what body-mask topology is expected

Likely shape:

- a versioned body/component schema id, for example:
  - `component_schema_id = "subject_body_v1_core"`
  - `component_schema_id = "subject_body_v2_with_pectoral_fins"`
- plus explicit policy attrs when needed, for example:
  - `anatomical_scope = "core_body"` or `"body_with_pectoral_fins"`
  - `pectoral_fin_policy = "excluded" | "included_when_visible" | "required_when_visible"`

Reason:

- the same `subject_body` channel should not silently mean different anatomical
  scope in different datasets
- body-mask QC thresholds and centerline/skeleton expectations may depend on
  whether pectoral fins are intended to be part of the mask
- this metadata should become part of the future body-refinement/body-geometry
  contract before those workflows get more complex

Recommended default before fin-aware curation is formalized:

- `component_schema_id = "subject_body_v1"`
- `anatomical_scope = "body_core"`
- `pectoral_fin_policy = "excluded_or_unresolved"`

Why this default:

- current datasets do not yet have one stable, explicit fin-inclusion rule
- some masks may visually include fin structure while others may not, depending
  on resolution and segmentation behavior
- the default should therefore avoid overclaiming that pectoral fins were
  intentionally included or intentionally excluded with strong consistency

Recommended later upgrade when fin-aware curation becomes explicit:

- `component_schema_id = "subject_body_v2"`
- `anatomical_scope = "body_with_pectoral_fins"`
- `pectoral_fin_policy = "included_when_visible"`

Recommended near-term refinement behavior:

- treat the refined body mask as the canonical editable artifact
- recompute simple body-local derived outputs on save:
  - `mask_present`
  - `area_px`
  - contour
  - centroid
  - bbox
- defer centerline/spline-specific schema work until real curated masks exist

The first viable body-refinement workflow should not depend on downstream shape
contracts being complete.

Suggested body-specific QC metrics:

- topology:
  - `component_count`
  - `largest_component_fraction`
  - `hole_count`
  - `hole_area_fraction`
- boundary roughness:
  - `sigma_noise`
  - `curvature_var`
  - `ipr`
  - `solidity`
- skeleton quality:
  - `n_branch_points_raw`
  - `n_endpoints_raw`
  - `n_branch_points_pruned`
  - `n_endpoints_pruned`

Recommended interpretation:

- for low-resolution datasets where fins are not expected, raw branching should
  usually remain near zero
- for higher-resolution datasets where fins are intentionally included, raw
  branching may be anatomically expected
- the stronger long-term QC target is that the pruned main body-axis skeleton is
  simple and usable, even when the raw body silhouette has side branches

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

Recommended near-term refinement behavior:

- keep swim-bladder refinement lightweight and blob-oriented
- recompute simple component-local outputs on save:
  - `mask_present`
  - `area_px`
  - centroid
  - bbox
  - contour
- defer richer geometry unless a clear downstream need appears

The swim bladder should not be forced into the same derived-geometry shape as
body or eyes just for schema symmetry.

Suggested swim-bladder-specific QC metrics:

- `component_count`
- `largest_component_fraction`
- `hole_count`
- `sigma_noise`
- `ipr`
- `solidity`

Skeleton-branching metrics are likely less central here than for body masks.

### Eyes

Keep current specialized refined path for now.

Recommended near-term policy:

- continue to use `refined_eye_masks_runs` as the current specialized
  eye-editing surface
- do not force eye editing into `refined_subject_masks_runs` until body/swim
  workflows are stable
- prefer unified subject-mask component rows in registry/query/operator
  surfaces when asking whether a dataset has reviewed eye components
- keep eye-specific derived outputs specialized for now:
  - left/right assignment
  - contours
  - ellipse parameters
  - ellipse success
  - eye separation

Longer-term target:

- eye masks may eventually live under
  `refined_subject_masks_runs/components/eye_left|eye_right`
- but only once the unified component model can preserve the current eye
  geometry/review affordances without regression

Current-state note:

- `refined_eye_masks_runs` does not yet follow the exact refined-subject layout
  proposed here
- it is an older specialized stage that mixes top-level eye geometry arrays with
  a `metrics/` subgroup
- it should be treated as a strong precedent for the kinds of derived outputs we
  want, but not as the final structural template for generic component
  refinement

Eye-specific QC should remain specialized for now; body-mask topology or
skeleton metrics should not be copied over mechanically.

## Review And Saveback Model

Recommended review model:

- one run-level review payload
- one component-level review payload per component
- component-scoped `reason_bytes` / `reason`

Recommended saveback behavior:

- editing `subject_body` updates only body mask pixels and body-derived metadata
- editing `swim_bladder` updates only swim-bladder pixels and metadata
- future eye editing should update only the touched eye component plus
  eye-specific derived fields

This keeps refinement aligned with the broader subject-mask design:

- one canonical refined stage for generic component masks
- component-specific geometry where needed
- no assumption that every component shares the same derived formulas

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

### Palette Inspector vs Paintera

We should not make Paintera the primary conceptual UI for subject-mask
inspection just because it can edit pixels.

Recommended role split:

- Palette-native subject-mask inspector:
  - browse subject-mask and refined subject-mask runs
  - overlay masks on ROI crops
  - show lineage, method, component availability, and review state
  - show component QC and provenance summaries
  - filter or jump to suspicious ROIs
  - compare raw masks, refined masks, and prompt/context overlays
- Palette-native lightweight review/editor:
  - keep the current focused OpenCV-style paint/erase workflow for refined
    subject masks
  - remain optimized for small ROI-local cleanup rather than generic dense
    labeling
- Paintera:
  - remain the heavy-duty pixel editor when that interaction model is useful
  - not be treated as the canonical inspection surface for subject-mask
    provenance or QC

Rationale:

- Paintera is useful for pixel editing, but it is not naturally pipeline-aware.
- Palette is the layer that understands:
  - lineage such as `source_subject_mask_run`, `source_keypoints_run`, and
    crop provenance
  - component availability and review state
  - component-local QC metrics such as topology and roughness summaries
  - method/model-specific provenance such as SAM prompt settings
- A thin Palette-native inspector is consistent with the repo's current
  collection of stage-specific review and visualization tools.
- What we should avoid is trying to recreate a full connectomics-style generic
  labeling workstation inside Palette.

Design direction:

- build a separate read-mostly subject-mask inspector first
- keep Paintera as an optional editing backend/tool of opportunity
- let the existing refined-subject review tool stay narrow and editing-focused

## Registry Implications

We should not reduce this to one boolean on recordings.

The registry should represent:

1. coarse stage presence
   - `subject_masks`
   - legacy `refined_eye_masks` compatibility signals during transition
   - `refined_subject_masks`
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
- does it have refined eye components, whether native subject-mask rows or
  projected legacy eye-stage compatibility rows?
- which components are reviewed for training use?

## Phase 0: Contract Decisions

- [x] Decide whether `refined_subject_masks_runs` is the canonical name.
- [x] Define whether eye channels belong in the refined subject-mask contract in
      v1, or only body/swim bladder do.
- [x] Define canonical source lineage attrs:
  - `source_subject_mask_run`
  - `source_crop_run`
  - optional `source_keypoints_run`
  - component-scoped `source_subject_mask_runs` supplements the coarse
    run-level lineage when the refined run is assembled from multiple sources

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
- [x] Move canonical manual eye review authority onto
      `refined_subject_masks_runs`, while preserving the legacy refined-eye
      failure UI as an explicit compatibility fallback.
- [x] Validate the new tool on real non-eye masks rather than only empty/copy
      initialized channels.
- [x] Add a separate Palette-native subject-mask inspector for read-mostly
      browsing, QC triage, and provenance-aware mask inspection.
- [x] Keep that inspector distinct from Paintera's role as an optional heavier
      pixel-editing surface.
- [x] Route refinement execution through a scheduler-aware engine rather than a
      permanently serial editor-only save path.

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

- [x] Define what “unify eye refinement under subject-mask component model”
      actually means:
  - shared component identity vocabulary
  - shared review payload schema
  - shared registry component rows
  - optional shared refined-mask storage
  See [eye_subject_mask_unification_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_subject_mask_unification_design.md).
- [x] Decide whether `refined_eye_masks_runs` becomes:
  - a specialized derivative of `refined_subject_masks_runs`, or
  - a long-lived sibling artifact with aligned component semantics
  Decision: target steady-state is a compatibility/adapter artifact derived
  from canonical refined-subject state, while the current transition keeps it
  supported for eye-specific workflows.
- [x] Add a non-destructive migration/backfill plan when the target becomes
      clear.
  See the migration phases in
  [eye_subject_mask_unification_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_subject_mask_unification_design.md).

## Acceptance Criteria

- [x] Whole-subject and swim-bladder masks have a clear future refined/editable
      stage.
- [x] Registry can represent raw presence vs refined presence vs review state by
      component.
- [x] `refined_eye_masks_runs` remains supported during transition.
- [x] The future unification path for eyes is explicit enough to avoid another
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
- [swim_bladder_review_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_review_policy.md)
- [swim_bladder_patch_review_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_patch_review_design.md)
- [subject_mask_registry_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_registry_contract.md)
- [subject_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_training_artifact_contract.md)
- [refined_subject_mask_scheduler_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_mask_scheduler_todo.md)
- [subject_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_tuning_workflow.md)
- [mask_review_save_approval_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/mask_review_save_approval_policy.md)
- [pose_kinematics_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_kinematics_run_design.md)
- [review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md)
