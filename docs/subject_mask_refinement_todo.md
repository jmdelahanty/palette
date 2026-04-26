# Subject Mask Refinement TODO

For the active operator-facing source-of-truth contract, see
[current_pipeline_contract.md](current_pipeline_contract.md). This TODO tracks
subject-mask refinement rollout work against that contract.

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
- current U-Net subject-mask models can emit body, eye, and swim-bladder
  channels together
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
- It can currently represent sparse eye-only compatibility runs and dense
  multi-component U-Net runs.
- `subject_mask_training_artifact_contract.md` exists, and the merged
  `subject_masks` exporter, validator, loader, trainer, and registry preflight
  path are implemented.
- `refined_eye_masks_runs` remains a specialized and compatibility stage during
  the transition to subject-mask unification, but canonical manual eye review
  has now moved to `refined_subject_masks_runs` and derived compat refined-eye
  runs are read-only in legacy viewers.
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
  intermediate raw run.
- A smart probability-run finalizer now exists at
  [src/fisheye/refinement/finalize_subject_masks.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/refinement/finalize_subject_masks.py)
  for raw U-Net `subject_v1_union` outputs that should become canonical
  `subject_v1_lr` refined subject-mask candidates.
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
- U-Net subject-mask training can now emit validation preview PNGs, and a
  full-size short smoke run showed that direct `eye_left` / `eye_right`
  prediction can collapse into "both eyes in both channels" even when the
  exported target channels are distinct. This reinforces the design boundary in
  [eye_subject_mask_unification_design.md](eye_subject_mask_unification_design.md):
  raw models should favor visually identifiable eye masks (`eyes_union` or
  unordered instances), while biological LR identity should be assigned by a
  geometry-aware subject-mask refinement/finalization step when orientation
  evidence is available. Eye-capable refined subject-mask runs should expose
  canonical eye identity as `eye_left` / `eye_right`; `eyes_union` should remain
  raw/model input, seed/provenance context, or review/debug evidence rather than
  the refined eye authority.
- Subject-mask training now writes live epoch metrics to
  `training_history_live.jsonl` during the run, writes final
  `training_history.json` on completion, and supports optional TensorBoard
  scalar logging via `--tb-logdir`.

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

### 3. Delay downstream analysis geometry until labels exist

- [ ] Do not start `analysis/subject_shape_runs` implementation until we have at
      least a small curated refined body-mask set.
- [ ] Do not design body/spline contour arrays beyond the current contract until
      we see what the first curated masks actually look like.

### 4. Keep eye-specific geometry compatibility clean while review authority is unified

- [x] Move canonical manual eye review into `refined_subject_masks_runs`.
- [x] Guard derived `refined_eye_masks_runs` compatibility artifacts against
      drift by treating them as read-only in legacy eye viewers.
- [x] Route the first active eye geometry/export consumers through canonical
      `refined_subject_masks_runs` with `refined_eye_masks_runs` fallback
      compatibility.
- [x] Audit remaining legacy eye-specific viewers/diagnostics and decide which
      should become subject-mask-aware versus remain explicitly historical.
      Outcome: active eye geometry/export consumers now use the shared refined
      subject-mask resolver, while legacy eye-mask viewers, patch tools,
      profile utilities, and old eye-stage diagnostics remain explicitly
      historical/compatibility surfaces.
- [ ] Prefer unified subject-mask component registry/query/operator surfaces for
      eye availability and review visibility, using legacy eye stages only as
      compatibility inputs or diagnostics.
      Progress: registry component current/latest views now prefer available
      per-component refined-subject rows, including partial refined runs, while
      preserving stale lifecycle state instead of hiding stale refined rows
      behind raw or legacy compatibility stages. `check_recording_steps`
      registry-mode summaries now overlay component availability/review state
      from the unified subject-mask component registry views instead of trusting
      stale step-detail snapshots. Component registry/query/training/operator
      surfaces now also project refined-run `source_subject_mask_stale_*`
      metadata so stale source drift is visible separately from review state.

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

- the shipped subject-mask review/assembly helpers load source inputs from
  `subject_mask_runs`, direct eye components from `refined_eye_masks_runs`,
  and component-specific sources from existing `refined_subject_masks_runs`
- new raw eye orchestration now writes a companion eye-only
  `subject_mask_runs/<run>` immediately after successful `eye_masks_runs/<run>`
  completion, using `subject_v1_union`
- when eye content starts in legacy eye stages, the implemented bridge is to
  project/backfill a compatibility `subject_mask_runs/<run>` first and then
  assemble or refine from that subject-mask source
- direct `refined_eye_masks_runs` -> `refined_subject_masks_runs` seeding is
  supported for `eye_left` / `eye_right`
- split refined component consolidation is explicit: assemble a new
  `refined_subject_masks_runs/<run>` from safe component sources rather than
  having exporters silently stitch split refined runs
- remaining work is consumer cleanup and deciding when compatibility
  materialization becomes opt-in

### 3. Keep legacy eye-geometry compatibility during migration

Do not remove `refined_eye_masks_runs` yet, but also do not treat it as a
second canonical refined family for new operator-facing review status.

For now:

- canonical manual eye review/edit now operates through
  `refined_subject_masks_runs`
- `refined_eye_masks_runs` stays supported as the specialized legacy/compat
  eye layout for historical runs and eye-specific consumers
- derived compatibility `refined_eye_masks_runs/<run>` artifacts refreshed
  from canonical refined-subject eye state should be treated as read-only in
  legacy viewers
- subject-mask unification should keep eye-specific geometry/export consumers
  working without another schema reset
- current eye-angle analysis and eye-mask training export already route through
  `fisheye.shared.eye_geometry_source`, preferring canonical refined-subject
  eye geometry with refined-eye fallback for historical archives
- registry/query/operator surfaces should prefer unified subject-mask component
  rows for eye visibility, projecting legacy eye-stage data when necessary
- contract target: unified assembly and finalization produce canonical
  `eye_left` / `eye_right` components in `refined_subject_masks_runs`
- current implementation: legacy refined-eye sources can seed unified refined
  subject-mask components directly or through projected compatibility
  `subject_mask_runs`

### 4. Preserve eye-local QC while shape analysis becomes unified

Even after raw masks are unified, eye refinement currently carries specialized
mask-local outputs that are not shared by every component, such as:

- left/right assignment
- contours
- ellipse parameters
- eye separation
- eye-specific reason/status handling

These outputs are a good reason to keep eye-local QC primitives with
`refined_subject_masks_runs`, while analysis-facing eye geometry is written to
`analysis/subject_shape_runs` as part of coherent body/eyes/swim shape
analysis.

Current gap:

- the refined subject-mask storage and registry surfaces are ready to hold
  `eye_left` and `eye_right`
- existing assembly/merge paths mostly preserve upstream LR labels
- `assemble_refined_subject_masks --subject-run` now supports the preferred
  single-source raw subject-mask path by copying all available canonical
  components from one `subject_mask_runs/<run>` into a finalized refined run
- `eyes_union` assignment now has an initial keypoint-based path that generates
  canonical `eye_left` / `eye_right` refined seeds when source keypoint lineage
  is available
- existing eye geometry paths compute ellipses/separation from already-labeled
  LR components
- remaining work is to harden assignment confidence/status semantics for
  ambiguous rows and unordered eye instances

Recommended next design slice:

- extend LR assignment confidence beyond nearest-keypoint splitting by using
  heading/body axis, eye centroids, separation checks, and ellipse fit quality
  as explicit assignment evidence
- write assignment confidence/status and reason labels per row
- mark ambiguous rows for review rather than silently guessing
- keep component provenance explicit, including the source eye mask and the
  assignment method

## Proposed Stage Relationship

Near-term:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>            # raw model output; future multi-component source
  -> refined_subject_masks_runs/<run>   # body-only refined path now exists
  -> refined_eye_masks_runs/<run>      # eye-specialized path remains
```

Preferred future all-component assembly path:

```text
crop_runs/<run>
  -> subject_mask_runs/<unet run>       # raw probabilities for subject_body, eyes_union, swim_bladder
  -> refined_subject_masks_runs/<run>   # canonical subject_body, eye_left, eye_right, swim_bladder
```

Raw/refined storage policy:

- raw U-Net/model outputs should persist probability masks plus
  model/config/provenance in `subject_mask_runs/<run>`
- thresholded masks should not be the canonical raw payload for native model
  output
- thresholding, hole filling, gap closing, island removal, left/right eye
  assignment, QC metrics, review state, and approval belong to the refined
  candidate in `refined_subject_masks_runs/<run>`
- refined candidates do not need to duplicate the pre-refinement binary mask;
  the "before" state is recoverable from the source raw probability run and its
  explicit threshold/refinement policy
- topology cleanup is a refined-subject finalization responsibility:
  body masks may close small gaps, fill holes, remove tiny detached islands, and
  keep one best body component; swim-bladder masks should use stricter compact
  component selection; eye-union masks must allow two valid eye components
  rather than blindly keeping the largest component
- cleanup must write metrics/reasons for removed area, removed probability
  mass, hole filling, and large cleanup deltas; high-confidence removed islands
  should force `needs_review`, not silent approval

Current sparse-source assembly and repair path:

```text
component/raw sources
  -> refined_subject_masks_runs/<run>  # direct assembly + subject-mask finalization
  -> refined_eye_masks_runs/<run>      # still specialized during transition
  -> analysis/subject_shape_runs/<run> # downstream interpreted shape geometry
```

## Scope

This TODO covers:

- refined/editable runtime storage for body and swim bladder
- unification path for eye refinement under the subject-mask component
  model
- registry/review implications
- downstream geometry implications for body and swim-bladder masks

This TODO does not by itself define:

- final body centerline/spline implementation
- tail spline/curvature algorithms
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

Policy note:

- `eyes_union` is useful as raw/model output and as refinement input, but an
  eye-capable refined subject-mask run should canonicalize reviewed eye content
  into `eye_left` and `eye_right`
- if refinement cannot assign anatomical side safely, the run should record
  ambiguity/review state rather than claiming complete refined eye availability

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
bladder, and eye metrics can evolve independently.

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

- `analysis/subject_shape_runs`
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

Current canonical review policy:

- use `refined_subject_masks_runs/components/eye_left|eye_right` as the
  canonical reviewed eye-mask authority for modern unified workflows.
- keep `refined_eye_masks_runs` readable as a historical/compatibility layout,
  and materialize it only as a derived bridge when an old consumer requires it.
- prefer unified subject-mask component rows in registry/query/operator
  surfaces when asking whether a dataset has reviewed eye components.

Mask-local eye outputs may remain in `refined_subject_masks_runs`:

- left/right assignment provenance
- contours
- ellipse parameters
- ellipse success
- eye separation
- eye-specific reason/status handling

Analysis-facing eye geometry belongs in `analysis/subject_shape_runs` when it
is part of a coherent body/eyes/swim shape run. Specialized outputs such as
`analysis/eye_angle_runs` may remain separate during migration, but new
eye-angle writers should consume subject-shape eye geometry when that surface is
available.

Current-state note for `refined_eye_masks_runs`:

- it is an older specialized stage that mixes top-level eye geometry arrays with
  a `metrics/` subgroup
- it should be treated as a strong precedent for the kinds of derived outputs we
  want, but not as the final structural template for generic component
  refinement

Eye-local QC should remain component-specific; body-mask topology or skeleton
metrics should not be copied over mechanically.

## Review And Saveback Model

Recommended review model:

- one run-level review payload
- one component-level review payload per component
- component-scoped `reason_bytes` / `reason`

Recommended saveback behavior:

- editing `subject_body` updates only body mask pixels and body-derived metadata
- editing `swim_bladder` updates only swim-bladder pixels and metadata
- eye editing should update only the touched eye component plus eye-specific
  derived fields

This keeps refinement aligned with the broader subject-mask design:

- one canonical refined stage for generic component masks
- component-specific geometry where needed
- no assumption that every component shares the same derived formulas

Future unification should make eyes a subject-mask component conceptually, but
legacy refined-eye artifacts are richer than a plain component mask.

The unification target is therefore:

- common component model at the raw and refined subject-mask levels
- shared review/state semantics where possible
- eye-local QC primitives retained with refined subject masks
- analysis-facing body, swim-bladder, and eye geometry expressed together in
  `analysis/subject_shape_runs`

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
- [x] Define and implement refined eye geometry payloads for:
  - `components/eye_left/geometry/ellipse_params`
  - `components/eye_right/geometry/ellipse_params`
  - `components/eye_left|eye_right/contours`
  - `relations/eye_pair/metrics/separation_px`

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
- [x] Add component latest views that can distinguish:
  - raw-only body/swim-bladder availability
  - refined body/swim-bladder availability
  - refined eye availability
- [ ] Continue wiring those registry surfaces into every operator/query surface:
  - `check_recording_steps`
  - stale-step cascade / invalidation views
  - registry UI / TUI surfaces
  Current status: registry ranking and `check_recording_steps` registry mode
  now prefer unified refined-subject component rows, including partial refined
  runs, while stale-step cascade and UI/TUI parity remain open.

## Phase 4: Geometry Integration

- [x] Define the boundary between refined-mask-local geometry and downstream
      interpreted shape metrics.
  Decision: component contours, centroids, bboxes, areas, validity flags, and
  simple component shape descriptors belong with
  `refined_subject_masks_runs`; body centerlines/splines, anatomical axes,
  canonical body B-spline fits, canonical centerline/B-spline body length,
  swim-bladder-to-body relationships, and eye angles relative to heading belong
  in `analysis/subject_shape_runs` or a specialized downstream analysis run.
- [x] Define `analysis/subject_shape_runs` to consume refined body/swim/eye
      masks, not raw `subject_mask_runs`.
  See [subject_shape_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_shape_runs_contract.md).
- [x] Decide that subject shape lives under `analysis/subject_shape_runs`, not at
      the zarr root.
  See [derived_analysis_run_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/derived_analysis_run_contract.md).
- [x] Add a refinement-side refresh path for mask-local body/swim/eye metrics and
      generated `needs_review_metric_*` reason tags.
  See `scripts/py -m fisheye.utils.backfill_refined_subject_mask_metrics`.
- [ ] Implement the first `analysis/subject_shape_runs` writer.
- [ ] Include body B-spline fit support in the first body-shape writer or define
      it as the first follow-up slice.

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
- [x] Implement `eyes_union -> eye_left/eye_right` assignment inside
      refined-subject finalization using declared assignment keypoint lineage.
- [x] Materialize eye geometry and eye-pair relation metrics from refined
      subject-mask eye components.
- [x] Validate the full body/eyes/swim path on a real analysis-zarr canary.

## Current End-To-End Status

As of 2026-04-26, the subject-mask training/inference/refinement path is
implemented for the current U-Net design:

- registry preflight can select coherent raw or approved refined subject-mask
  training sources and write a manifest/config
- merged subject-mask training zarr export, validation, loader, and U-Net
  training are implemented
- subject-mask training runs can be recorded in the model registry and resolved
  by inference for component-coverage-aware model selection
- U-Net inference writes probability-first `subject_mask_runs/<run>` snapshots
  with `subject_v1_union` labels:
  `["subject_body", "eyes_union", "swim_bladder"]`
- `fisheye.refinement.finalize_subject_masks` thresholds and finalizes those
  raw probabilities into canonical `subject_v1_lr`
  `refined_subject_masks_runs/<run>` candidates with
  `["subject_body", "eye_left", "eye_right", "swim_bladder"]`
- the smart finalizer writes cleanup metrics, reason tags, source-seed masks,
  component provenance, review-triage counts, and Dask execution metadata
- refined-subject eye geometry can be written during finalization or backfilled
  afterward with `fisheye.utils.backfill_refined_subject_eye_geometry`

Real canary evidence:

- source archive:
  `/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr`
- raw source run:
  `subject_masks_unet_registry_gpu_metrics_profile_2026-04-26`
- latest refined candidate:
  `refined_subject_masks_smart_finalizer_dask_processes48_c64_canary_2026-04-26`
- refined candidate shape:
  `masks_roi = (19235, 4, 512, 512)`
- refined labels:
  `["subject_body", "eye_left", "eye_right", "swim_bladder"]`
- eye geometry status:
  `computed`, with valid eye-pair separation on `19233 / 19235` rows

Remaining work is no longer open-ended architecture. It is operational hardening:

- visual inspection and component approval of smart-finalized candidates
- temporal QC as a second pass that flags suspicious rows without changing masks
- faster/chunked eye-geometry backfill if this becomes a frequent full-run task
- body/eyes/swim subject-shape-stage implementation and downstream
  subject-shape consumers
- complete subject/refined-subject stale repair parity with the eye-mask
  precedent
- top-level segmentation orchestration in `core/pipeline.py` so operators do not
  need to call model-specific segmentation CLIs directly

## Acceptance Criteria

- [x] Whole-subject and swim-bladder masks have a clear future refined/editable
      stage.
- [x] Registry can represent raw presence vs refined presence vs review state by
      component.
- [x] `refined_eye_masks_runs` remains supported during transition.
- [x] The unification path for eyes is explicit enough to avoid another
      schema reset later.
- [x] Downstream body/eyes/swim shape work is clearly anchored to refined
      subject masks and separated from mask-local geometry.

## Risks

- Trying to force eye-local QC into a generic derived-analysis stage too early
  could discard important eye-specific review semantics.
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
