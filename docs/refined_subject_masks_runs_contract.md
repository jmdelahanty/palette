# Refined Subject Masks Runs Contract (Draft v1)
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-01
-->

Purpose: define the runtime/storage contract for editable, refined
subject-mask artifacts that can hold canonical component masks for body and swim
bladder now, while also defining the target canonical home for future
left/right eye refinement under the same component model.

## Scope

- Define `refined_subject_masks_runs/<run>` as the canonical refined/editable
  subject-mask stage.
- Support refined body masks and refined swim-bladder masks.
- Support component-scoped review and reasons.
- Reserve space for component-specific derived geometry such as contours,
  centroids, and centerline/spline-related outputs.
- Define the target eye-capable refined layout for:
  - `eye_left`
  - `eye_right`
  - per-eye ellipses
  - per-eye contours
  - cross-eye relation metrics such as `eye_separation`

## Non-goals

- Replacing `refined_eye_masks_runs` in v1.
- Defining the final exact geometry array schema for body contours or splines.
- Defining `subject_shape_runs`.
- Defining merged training artifact layout.

## Relationship To Existing Stages

Near-term canonical relationship:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_subject_masks_runs/<run>
  -> subject_shape_runs/<run>          # future
```

Legacy eye-specialized compatibility path during transition:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_eye_masks_runs/<run>
```

Policy:

- `refined_subject_masks_runs` is the refined stage for generic subject-mask
  components.
- `refined_eye_masks_runs` remains supported during the transition as the
  current eye-specific refined stage.
- registry/query/operator surfaces should prefer unified subject-mask component
  rows for eye availability, with legacy eye stages projected in only as
  compatibility inputs when native eye-capable subject-mask rows are absent.
- the target steady-state for new eye-capable refined authoring is still
  `refined_subject_masks_runs`
- future unification should align eye refinement under the subject-mask
  component model without forcing a destructive migration now
- sparse multi-source workflows should not require an assembled raw
  `subject_mask_runs/<run>` intermediate before refinement

## Canonical Label Scope

Recommended minimum currently implemented component scope:

- `subject_body`
- `swim_bladder`

Optional/compatibility labels:

- `eyes_union`
- `eye_left`
- `eye_right`

Canonical target for eye-capable refined authoring:

- `label_schema_id = "subject_v1_lr"`
- `mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]`

Compatibility/raw model-output schema:

- `subject_v1_union` remains valid for raw compatibility and export use
- `subject_v1_union` is not the preferred long-term refined eye authoring
  schema because it loses anatomical eye identity

Writers must always persist:

- `label_schema_id`
- `mask_labels`
- `available_channels`

Readers must never infer component meaning from channel index alone.

## Evolution Policy

This contract is intended to support:

1. refined body/swim-bladder masks while eyes remain specialized elsewhere,
2. fuller multi-component refined subject masks later, and
3. eventual eye alignment under the subject-mask component model.

V1 directly supports case 1 and is shaped to permit cases 2 and 3 later
without changing the stage family name.

## Assembly And Finalization Semantics

`refined_subject_masks_runs/<run>` is not merely a bag of assembled component
masks. It is the canonical refined/editable working artifact, and it should be
treated as valid only after subject-mask refinement/finalization has
materialized the canonical QA surface.

Required behavior:

- sparse multi-source assembly may seed a new
  `refined_subject_masks_runs/<run>` directly
- the seed/assembly step must be followed by subject-mask finalization before
  the run is treated as a valid refined artifact
- finalization is responsible for canonical run/component metrics, reasons,
  review scaffolding, provenance updates, and any refinement-time geometry
  derived by this stage family

Initial allowed seed sources for unified assembly:

- raw `subject_mask_runs`
- transitional `refined_eye_masks_runs` for eye components

Current implementation note as of 2026-04-02:

- the shipped assembler/review helpers currently resolve seed inputs through
  `subject_mask_runs`-backed sources
- for legacy eye-stage data, the implemented path is:
  `refined_eye_masks_runs` or `eye_masks_runs`
  -> projected/backfilled `subject_mask_runs/<run>`
  -> assembled/finalized `refined_subject_masks_runs/<run>`
- direct `refined_eye_masks_runs` -> `refined_subject_masks_runs` seeding
  remains a contract target and future extension, not current shipped behavior

Deferred source pattern:

- importing components from another `refined_subject_masks_runs/<run>` is not
  part of the initial unification plan and should be treated as a later
  extension if needed

## Output Layout

```text
refined_subject_masks_runs/
  attrs:
    latest                                  "<run_id>"
  <run_id>/
    frame_indices                           (N,) int32           # recommended
    frame_counts                            (F,) int32           # recommended
    detection_indices                       (N,) int32           # recommended
    detection_source                        (N,) int8
    masks_roi                               (N, C, H, W) uint8
    available_channels                      (C,) bool
    edit_applied                            (N, C) bool
    metrics/
      mask_present                          (N, C) bool
      area_px                               (N, C) float32
      centroid_xy                           (N, C, 2) float32   # recommended
      centroid_valid                        (N, C) bool         # recommended
      bbox_xyxy                             (N, C, 4) float32   # recommended
      bbox_valid                            (N, C) bool         # recommended
    components/
      <component_name>/
        provenance/                        # attrs-only subgroup for component lineage/update provenance
        reason_bytes                        (N, width) uint8     # recommended
        reason                              (N,) string          # optional mirror
        mask_present                        (N,) bool            # recommended
        area_px                             (N,) float32         # recommended
        geometry_valid                      (N,) bool            # optional
        edit_applied                        (N,) bool            # recommended
        metrics/                            # optional component-local QC summary arrays
          ...
        geometry/                           # optional extension point
          ...
        contours/                           # optional component-local contour storage
          ...
    relations/                              # optional cross-component derived values
      <relation_name>/
        metrics/
          ...
```

## `refined_subject_masks_runs/<latest>`

Required arrays:

- `detection_source`
  - shape: `(N,)`
  - expected to align with the source crop run
- `masks_roi`
  - shape: `(N, C, H, W)`
  - canonical refined binary masks
- `available_channels`
  - shape: `(C,)`
  - run-level declaration of which components are semantically available in the
    refined run
- `edit_applied`
  - shape: `(N, C)`
  - true when the refined mask row/channel was changed relative to the source
    subject-mask run

Required `metrics/` arrays:

- `mask_present`
  - shape: `(N, C)`
- `area_px`
  - shape: `(N, C)`

Recommended common geometry `metrics/` arrays:

- `centroid_xy`
  - shape: `(N, C, 2)`
- `centroid_valid`
  - shape: `(N, C)`
- `bbox_xyxy`
  - shape: `(N, C, 4)`
- `bbox_valid`
  - shape: `(N, C)`

Recommended lineage arrays:

- `frame_indices`
- `frame_counts`
- `detection_indices`

## Required attrs

- `source_subject_mask_run`
- `source_crop_run`
- `label_schema_id`
- `mask_labels`
- `output_semantics = "multilabel"`
- `refinement_semantics = "canonical_component_masks"`
- `method`
- `created_at_utc`
- `duration_seconds`

Required review attrs:

- `refined_subject_mask_review_status`
- `component_review_statuses`

Optional attrs:

- `source_keypoints_run`
- `source_keypoint_group`
- `source_refined_eye_masks_run`
- `source_subject_shape_run`
- `summary_statistics`
- `component_summary_statistics`

## `available_channels` semantics

`available_channels` means the refined run contains semantically valid refined
data for that component at all.

Meaning:

- `available_channels[c] == true` means component `c` is intentionally present
  in this refined run
- `available_channels[c] == false` means component `c` is a placeholder channel
  and must not be treated as a true negative

Required invariants:

- if `available_channels[c] == false`, then `masks_roi[:, c]` must be all-zero
- if `available_channels[c] == false`, then `edit_applied[:, c]` must be all-false
- if `available_channels[c] == false`, then `metrics/mask_present[:, c]` must be all-false

## `edit_applied` semantics

`edit_applied[n, c]` records whether the refined channel for row `n` differs
from the source subject-mask channel in a way that should be treated as a human
or deterministic refinement, rather than a plain copy-through.

This field is intended to support:

- QA summaries
- training provenance
- future review UI filtering

It does not by itself imply manual editing; the review payload should carry the
review method.

## Review Payloads

Run-level review payload:

- `refined_subject_mask_review_status`

Component-level review payload mapping:

- `component_review_statuses`

Canonical review keys:

- `state`
- `method`
- `intended_use`
- `reviewer`
- `notes`
- `timestamp_utc`

Example:

```json
{
  "refined_subject_mask_review_status": {
    "state": "approved",
    "method": "manual",
    "intended_use": "training",
    "reviewer": "alice",
    "timestamp_utc": "2026-03-10T20:15:00Z"
  },
  "component_review_statuses": {
    "subject_body": {
      "state": "approved",
      "method": "manual",
      "intended_use": "training",
      "timestamp_utc": "2026-03-10T20:14:00Z"
    },
    "swim_bladder": {
      "state": "needs_review",
      "method": "manual",
      "intended_use": "training",
      "timestamp_utc": "2026-03-10T20:14:30Z"
    }
  }
}
```

## Component-Scoped Provenance

Run-level `source_subject_mask_run` remains required as the coarse lineage
pointer for the refined run as a whole, but it is not sufficient once one
refined run may contain components seeded from different upstream artifacts.

Canonical home:

- `components/<component_name>/provenance/`

The provenance subgroup should be attrs-only in v1 unless a later contract
needs per-row lineage.

Required provenance attrs for an available component:

- `source_stage`
  - stage family that seeded the component, for example `subject_mask_runs`,
    or transitional `refined_eye_masks_runs`
- `source_run`
  - source run id within that stage family
- `source_method`
  - upstream run `method` used to seed or replace this component
- `source_channels`
  - list of source channel names used to seed this component

Recommended provenance attrs:

- `source_label_schema_id`
  - the source run's `label_schema_id`
- `last_update_stage`
  - stage/tool that most recently changed this component in the current refined
    run
- `last_update_mode`
  - recommended values include `create`, `interactive`, `batch`, `projection`
- `last_update_method`
  - method/tool label for the last change
- `updated_at_utc`
  - component-local last update timestamp

Semantics:

- `source_*` identifies the upstream artifact that seeded or replaced the
  component in this refined run
- `source_label_schema_id` is the source artifact's `label_schema_id`
- `source_*` does not imply the current component is byte-identical to that
  source after later edits
- subject-mask finalization is expected to run after seeding and may update
  `last_update_*` while preserving the original `source_*` origin
- `last_update_*` records the most recent operation that changed this component
  inside the current refined run
- if a component is copied through during refined-run creation and never edited,
  writers may record `last_update_mode = "create"`

This distinction is required for future mixed-source refined runs such as:

- `subject_body` seeded from a SAM subject-mask run
- `eye_left` and `eye_right` seeded from a UNet/refined-eye workflow
- `swim_bladder` seeded from a different raw subject-mask source

## Component Subgroups

`components/<component_name>/` is the standard extension point for
component-specific refinement metadata.

Recommended arrays per available component:

- `reason_bytes`
  - shape: `(N, width)`
  - null-terminated UTF-8 primary encoding
- `reason`
  - shape: `(N,)`
  - optional string mirror
- `mask_present`
  - shape: `(N,)`
- `area_px`
  - shape: `(N,)`
- `edit_applied`
  - shape: `(N,)`

Optional arrays:

- `geometry_valid`
  - shape: `(N,)`
- component-specific quality flags
- component-specific review artifacts

Optional subgroups:

- `provenance/`
  - component-scoped lineage and last-update attrs
- `metrics/`
  - component-local fixed-shape QC summary arrays
- `geometry/`
  - component-local derived geometry arrays
- `contours/`
  - component-local contour stores when contour ownership belongs to one
    component

Optional component attrs:

- `component_schema_id`
- `anatomical_scope`
- component-local policy attrs such as `pectoral_fin_policy`

Recommended current `subject_body` defaults:

- `component_schema_id = "subject_body_v1"`
- `anatomical_scope = "body_core"`
- `pectoral_fin_policy = "excluded_or_unresolved"`

Recommended examples for `components/<component>/metrics/`:

- `component_count`
- `largest_component_fraction`
- `hole_count`
- `hole_area_fraction`
- `sigma_noise`
- `curvature_var`
- `ipr`
- `solidity`

Common cross-component geometry such as centroid and bbox should stay at
run-level `metrics/`, while component-specific QC should live under
`components/<component>/metrics/`.

Why per-component subgroups:

- body and swim bladder will not share identical derived geometry
- eye refinement is even more specialized
- this avoids freezing the whole stage around one component’s geometry layout

## Cross-Component Relation Subgroups

`relations/<relation_name>/` is the standard extension point for derived values
that conceptually span multiple components and are not owned by one component.

Canonical example for future eye-capable refined runs:

- `relations/eye_pair/metrics/`
  - `separation_px`
  - `separation_valid`

Why this belongs under `relations/` rather than under one eye component:

- `eye_separation` is a pairwise derived value
- duplicating it under both eye components creates synchronization risk
- it should not require inventing a fake mask component such as `eye_pair`

## Geometry Extension Policy

Component-specific geometry should live under:

- `components/<component>/geometry/`

This contract keeps body/swim-bladder geometry intentionally open, but the
target eye-capable refined layout is now explicit.

Expected future examples:

### `subject_body`

- contour tables
- centroid and axis summaries
- centerline seeds
- spline control points or sampled centerline points
- body-orientation summaries

### `swim_bladder`

- contour tables
- centroid
- ellipse/blob summaries

### `eye_left` / `eye_right`

- eye-specific review, reasons, QC, and provenance remain component-local under
  `components/eye_left|eye_right/`
- eye-specific geometry should live under:
  - `components/eye_left/geometry/ellipse_params`
  - `components/eye_left/geometry/ellipse_success`
  - `components/eye_right/geometry/ellipse_params`
  - `components/eye_right/geometry/ellipse_success`
- eye-specific contour stores should live under:
  - `components/eye_left/contours/{ptr, len, points_xy}`
  - `components/eye_right/contours/{ptr, len, points_xy}`
- cross-eye relation metrics should live under:
  - `relations/eye_pair/metrics/separation_px`
  - `relations/eye_pair/metrics/separation_valid`

Geometry policy:

- refined component masks remain the canonical source artifact
- geometry derived from those masks should carry its own validity flags
- downstream `subject_shape_runs` should consume refined body masks or refined
  body geometry, not raw `subject_mask_runs`

## Reason Encoding Policy

If `reason_bytes` is present for a component subgroup, writers should also set:

- `reason_encoding = "utf8-null-terminated"`
- `reason_bytes_width = <int>`
- `reason_bytes_null_terminated = true`
- `reason_fallback_order = ["reason_bytes", "reason", "detection_source"]`

Recommended reason tags may include:

- `clean`
- `manual_correction`
- `manual_creation`
- `incomplete`
- `missing_component`
- `geometry_issue`
- `overlap`
- `ambiguous_boundary`

These are examples, not a frozen vocabulary yet.

## Body / Swim-Bladder Expectations In V1

Recommended minimum v1 support:

- `subject_body` refined masks may be available
- `swim_bladder` refined masks may be available
- either component may be unavailable without invalidating the whole run

That means the stage must support cases like:

- body-only refinement run
- swim-bladder-only refinement run
- body + swim-bladder refinement run

without inventing separate stage families.

## Relationship To `refined_eye_masks_runs`

During transition:

- `refined_eye_masks_runs` remains supported for historical archives and
  existing eye-specific tooling
- it may remain the live eye-specific refinement surface until unified
  eye-capable refined-subject writes reach parity

Target steady-state:

- `refined_subject_masks_runs` is the canonical refined authoring surface for
  new eye-capable subject-mask work
- `refined_eye_masks_runs` becomes a compatibility or adapter artifact rather
  than a second independent canonical authoring surface

Required provenance rule:

- if eye components in `refined_subject_masks_runs` are seeded from
  `refined_eye_masks_runs`, component provenance must point to that true source
  stage/run rather than collapsing everything to the run-level
  `source_subject_mask_run`

## Registry Implications

This stage should eventually project to the registry at two levels:

1. coarse step presence
   - `step_name = "refined_subject_masks"`
2. component-level refined availability and review state
   - `subject_body`
   - `swim_bladder`
   - later eye component(s) if added

The registry must be able to distinguish:

- raw subject-mask availability
- refined body/swim-bladder availability
- refined eye availability projected from unified refined-subject component rows
- specialized refined eye availability during the transition period

## Migration Policy

Recommended transition:

1. keep `refined_eye_masks_runs` unchanged
2. introduce `refined_subject_masks_runs` for body/swim bladder
3. extend `refined_subject_masks_runs` contracts to cover unified eye-local
   geometry and cross-eye relations
4. align registry and review payloads across refined stages
5. move new eye-capable refined authoring to `refined_subject_masks_runs`
6. treat `refined_eye_masks_runs` as a compatibility artifact once adapter
   readers/materializers exist

This contract is intentionally non-destructive.

## Validation Invariants

- all row-aligned arrays share the same first dimension `N`
- `masks_roi.shape[1] == available_channels.shape[0] == edit_applied.shape[1]`
- `metrics/mask_present.shape == metrics/area_px.shape == (N, C)`
- if a component subgroup exists, its per-row arrays must have first dimension `N`
- unavailable channels must remain zero/false across mask/edit/metrics arrays

## Open Questions

- Should `components/<component>/mask_present` and `area_px` remain duplicated
  from `metrics/`, or should one be derived-only?
- Should body contour/spline geometry live directly here, or only in a later
  `subject_shape_runs` stage with this stage remaining mask-centric?
- At what point should compatibility materialization of `refined_eye_masks_runs`
  become opt-in rather than routine once unified refined-subject eye writes are
  available?

## Related Docs

- [subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md)
- [subject_mask_registry_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_registry_contract.md)
- [eye_subject_mask_unification_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_subject_mask_unification_design.md)
- [subject_mask_refinement_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_refinement_todo.md)
- [review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md)
- [pose_kinematics_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_kinematics_run_design.md)
