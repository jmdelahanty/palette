# Refined Subject Masks Runs Contract (Draft v1)
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-28
-->

Purpose: define the runtime/storage contract for editable, refined
subject-mask artifacts that hold canonical component masks for body, swim
bladder, and modern left/right eye refinement under the same component model.

## Scope

- Define `refined_subject_masks_runs/<run>` as the canonical refined/editable
  subject-mask stage.
- Support refined body masks and refined swim-bladder masks.
- Support canonical refined `eye_left` and `eye_right` masks when raw/model
  sources provide assignable eye evidence.
- Support component-scoped review and reasons.
- Reserve space for component-specific derived geometry such as contours,
  centroids, and centerline/spline-related outputs.
- Define the implemented eye-capable refined layout for:
  - `eye_left`
  - `eye_right`
  - per-eye ellipses
  - per-eye contours
  - cross-eye relation metrics such as `eye_separation`

## Non-goals

- Removing read support for historical `refined_eye_masks_runs`.
- Defining the final exact geometry array schema for body contours or splines.
- Defining `analysis/subject_shape_runs`.
- Defining merged training artifact layout.

## Relationship To Existing Stages

Near-term canonical relationship:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_subject_masks_runs/<run>
  -> analysis/subject_shape_runs/<run> # derived analysis geometry
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
  eye-specific refined compatibility and historical stage.
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

Optional/compatibility seed labels:

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

Refined eye-authoring policy:

- `eyes_union` is allowed as raw/model output, import, provenance, or
  transitional seed input, but it is not the canonical refined eye authority.
- when eye content is promoted into `refined_subject_masks_runs`, subject-mask
  refinement/finalization should materialize `eye_left` and `eye_right`
  components when anatomical side can be assigned
- if a union or unordered eye source cannot be assigned safely, the refined run
  should record ambiguity/review state instead of claiming complete refined
  left/right eye availability
- operator, geometry, and training consumers that require refined eye identity
  should consume `eye_left` / `eye_right`, not `eyes_union`

Writers must always persist:

- `label_schema_id`
- `mask_labels`
- `available_channels`

Readers must never infer component meaning from channel index alone.

## Evolution Policy

This contract is intended to support:

1. refined body/swim-bladder masks while eyes remain specialized elsewhere,
2. fuller multi-component refined subject masks, and
3. eye alignment under the subject-mask component model.

V1 directly supports case 1 and now supports case 2/3 for the implemented
subject-mask U-Net and smart-finalizer path:

- raw `subject_v1_union` outputs are finalized into `subject_v1_lr`
- `eyes_union` is assigned into `eye_left` and `eye_right` using declared
  assignment keypoint lineage
- refined-subject eye geometry and eye-pair relation metrics are written from
  the generated LR masks

Manual review/editor workflows and compatibility materialization are still
transition-state surfaces, but the storage contract no longer treats LR eye
components as future-only.

## Assembly And Finalization Semantics

`refined_subject_masks_runs/<run>` is not merely a bag of assembled component
masks. It is the canonical refined/editable working artifact, and it should be
treated as valid only after subject-mask refinement/finalization has
materialized the canonical QA surface.

Required behavior:

- the preferred future seed path is a single raw
  `subject_mask_runs/<run>` containing all model-predicted subject-mask
  probability components plus model/config/provenance
- sparse multi-source assembly may seed a new
  `refined_subject_masks_runs/<run>` directly, but this is a compatibility and
  repair path rather than the steady-state model-output path
- thresholding raw probabilities into binary masks is part of
  refined-subject finalization, not a requirement of native raw model output
- the seed/assembly step must be followed by subject-mask finalization before
  the run is treated as a valid refined artifact
- finalization is responsible for canonical run/component metrics, reasons,
  review scaffolding, provenance updates, and any refinement-time geometry
  derived by this stage family
- refined candidates store the post-refinement binary mask and QC surface; they
  do not need to duplicate the pre-refinement thresholded mask because the
  source raw probability run and threshold/refinement policy are the recoverable
  "before" state
- for eye-capable runs, finalization is also responsible for promoting
  `eyes_union` or unordered eye seeds into canonical `eye_left` / `eye_right`
  components when the assignment is safe, or marking the affected rows/components
  ambiguous for review when it is not
- finalization is also responsible for component-specific topology cleanup:
  body masks may close small gaps, fill holes, remove detached islands, and keep
  one best body component; swim-bladder masks may fill small holes and choose
  one compact internal component; eye-union masks may preserve two valid eye
  components instead of keeping only the largest
- topology cleanup must write metrics/reasons that expose removed area and
  probability mass, and rows with large or ambiguous cleanup deltas must be
  marked for review instead of silently approved
- subject-body mask-level QC is owned by this refined-mask stage, not by
  downstream subject-shape extraction. See
  [subject_body_mask_qc_design.md](subject_body_mask_qc_design.md) for the
  additive QC group and review-gating policy for connected but implausible body
  masks such as attached dish scratches.

Initial allowed seed sources for unified assembly:

- raw `subject_mask_runs`
- transitional `refined_eye_masks_runs` for eye components
- canonical `refined_subject_masks_runs` component sources when assembling a
  new coherent refined run from previously split refined component runs

Current implementation note:

- the shipped assembler/finalizer accepts a single raw
  `subject_mask_runs/<run>` via `--subject-run`; all available canonical
  components in that source are copied as refined seeds and finalized into one
  coherent `refined_subject_masks_runs/<run>`
- it also accepts raw `subject_mask_runs` component sources for
  body/eyes/swim bladder when repairing or combining split sources
- it also accepts direct `refined_eye_masks_runs` sources for canonical
  `eye_left` / `eye_right` component seeding
- raw `eyes_union` is treated as refinement input/provenance, not as a
  canonical refined component; a `--subject-run` exposing available
  `eyes_union` can be assigned into `eye_left` / `eye_right` when explicit
  assignment keypoint attrs or source keypoint lineage resolve to usable
  anatomical eye keypoints
- `assignment_keypoints_run` / `assignment_keypoint_group` are preferred over
  `source_keypoints_run` / `source_keypoint_group` for `eyes_union` assignment,
  because raw subject-mask segmentation may be crop-only while the LR split is
  a later deterministic refinement step
- generated LR eye components record `eyes_union` as the source channel plus
  assignment method/keypoint provenance; the refined-subject finalizer then
  writes the standard eye geometry/QC surface from the generated LR masks
- if keypoint lineage is missing or the assignment produces no usable LR rows,
  assembly fails closed instead of creating a misleading refined eye surface
- it now accepts `refined_subject_masks_runs/<run>` as an explicit component
  source for split-run consolidation; the new component provenance points to
  the immediate refined source and carries the upstream component provenance
  under `upstream_component_provenance`
- refined component sources are approved-only by default: assembly from an
  existing `refined_subject_masks_runs/<run>` requires the requested component
  to have `component_review_statuses[component].state == "approved"`, with
  `--allow-unapproved-components` reserved for draft/QA assembly
- source review state is recorded as component provenance, but target component
  approval is not inherited by default; pass `--promote-source-review` only when
  the operator explicitly wants approved source review payloads copied onto the
  assembled/finalized target run
- `fisheye.refinement.finalize_subject_masks` is the smart finalizer for raw
  probability-first `subject_mask_runs`; it writes deterministic row chunks,
  cleanup metrics, source-seed masks, component provenance, reason tags,
  review-triage counts, Dask execution metadata, and optional eye geometry
- the measured local fast path for a full 19,235-row analysis-zarr canary used
  `--execution-backend dask_worker_chunks --scheduler processes --num-workers 48
  --chunk-size 64 --metric-level cheap`, then refreshed eye geometry with
  `fisheye.utils.backfill_refined_subject_eye_geometry`
- for legacy raw eye-stage data, the compatibility bridge remains:
  `refined_eye_masks_runs` or `eye_masks_runs`
  -> projected/backfilled `subject_mask_runs/<run>`
  -> assembled/finalized `refined_subject_masks_runs/<run>`

Safety rule:

- the assembler must reject split refined component sources unless crop
  lineage, row lineage, row count, detection source, and ROI shape match
- a historical refined source-view crop signature mismatch is allowed only
  when the mismatch is limited to
  `source_crop_signature.detection_source_path` and
  `source_crop_signature.detection_source_type`, and the sources otherwise
  share crop identity, row lineage, row count, detection source, and ROI shape
- production assembly from split refined component sources must also reject
  pending, missing, or non-approved component review states; unapproved sources
  are only allowed with an explicit draft/QA override

## Additive Unified Eye/Swim Migration Procedure

Use this procedure when historical `refined_eye_masks_runs/<run>` eye masks
and approved `refined_subject_masks_runs/<run>` swim-bladder components need a
single canonical refined-subject surface.

Principles:

- do not delete or rewrite historical refined-eye or refined-subject component
  source runs
- create a new additive `refined_subject_masks_runs/<run>` target
- seed `eye_left` and `eye_right` directly from `refined_eye_masks_runs`
- seed `swim_bladder` from the existing approved refined-subject component run
- keep immediate component provenance pointing to the true source stage/run
- use approved-only assembly by default
- do not promote source approval onto the assembled target by default; add
  `--promote-source-review` only after deciding the assembled/finalized target
  should inherit approved source review payloads

Recommended sequence:

1. Discover source pairs per archive.

   - choose the refined-eye source from `refined_eye_masks_runs`
   - choose the swim-bladder source from `refined_subject_masks_runs` where
     `component_review_statuses["swim_bladder"].state == "approved"`

2. Run assembly in dry-run mode.

   ```bash
   scripts/py -m fisheye.refinement.assemble_refined_subject_masks \
     <archive>.zarr \
     --refined-eye-run <refined_eye_run> \
     --swim-refined-run <refined_subject_swim_run> \
     --run-name refined_subject_masks_unified_eye_swim_<stamp> \
     --dry-run
   ```

3. If dry-run reports only historical refined source-view crop-signature
   differences, verify that the mismatch is metadata-only.

   The only acceptable crop-signature differences are:

   - `source_crop_signature.detection_source_path`
   - `source_crop_signature.detection_source_type`

   Row lineage, row count, detection source, ROI shape, and crop identity must
   still match. Do not use this exception for real crop-policy or source-run
   drift.

4. Apply only to approved-compatible archives.

   ```bash
   scripts/py -m fisheye.refinement.assemble_refined_subject_masks \
     <archive>.zarr \
     --refined-eye-run <refined_eye_run> \
     --swim-refined-run <refined_subject_swim_run> \
     --run-name refined_subject_masks_unified_eye_swim_<stamp>
   ```

5. Verify the new run.

   Expected surface:

   - `mask_labels = ["eye_left", "eye_right", "swim_bladder"]`
   - `available_channels = [true, true, true]`
   - component provenance:
     - `eye_left` / `eye_right`: `source_stage = "refined_eye_masks_runs"`
     - `swim_bladder`: `source_stage = "refined_subject_masks_runs"`

6. Verify review state.

   Expected default review behavior:

   - source review payloads are preserved under component provenance
   - target component review states remain `pending`
   - the assembled run remains `pending` until the operator reviews it or reruns
     assembly with explicit source-review promotion

   To opt into source-review promotion:

   ```bash
   scripts/py -m fisheye.refinement.assemble_refined_subject_masks \
     <archive>.zarr \
     --refined-eye-run <refined_eye_run> \
     --swim-refined-run <refined_subject_swim_run> \
     --run-name refined_subject_masks_unified_eye_swim_<stamp> \
     --promote-source-review
   ```

   Promotion only copies approved source payloads. Pending or missing source
   review still leaves the target component pending.

Example batch result from the 2026-04-25 recording migration:

- 52 recording training zarrs scanned
- 51 approved-compatible unified eye/swim runs written
- 50 runs became fully approved after explicit legacy refined-eye review
  promotion
- 1 run remained pending because the legacy refined-eye source review was
  pending

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

Required common `metrics/` arrays:

- `mask_present`
  - shape: `(N, C)`
- `area_px`
  - shape: `(N, C)`
- `centroid_xy`
  - shape: `(N, C, 2)`
- `centroid_valid`
  - shape: `(N, C)`
- `bbox_xyxy`
  - shape: `(N, C, 4)`
- `bbox_valid`
  - shape: `(N, C)`

These are the shared run-level mask geometry arrays. They apply uniformly to
every refined component channel and are represented in code by
`REFINED_SUBJECT_MASKS_SPEC`.

Recommended lineage arrays:

- `frame_indices`
- `frame_counts`
- `detection_indices`

## Required attrs

- `source_subject_mask_run`
- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `label_schema_id`
- `mask_labels`
- `output_semantics = "multilabel"`
- `refinement_semantics = "canonical_component_masks"`
- `method`
- `created_at_utc`
- `duration_seconds`

Required when the source crop run exposes detect-review linkage:

- `source_detect_review_status_ref`

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

Crop-snapshot semantics:

- `source_crop_run` + `source_crop_storage_mode` + `source_crop_signature` +
  `source_crop_revision` form the refined run's portable crop snapshot.
- `source_detect_review_status_ref` remains a separate stable lineage field and
  must not be folded into `source_crop_signature`.
- Current `refined_subject_masks_runs/<run>` writers preserve this crop
  snapshot from the upstream `subject_mask_runs/<run>` source rather than
  re-deriving it from whichever crop run happens to be latest later.

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

## Strict Contract Validation

Use the Crimson-facing validator before asking downstream readers to special-case
an archive:

```bash
scripts/py -m fisheye.utils.validate_refined_subject_mask_contract <archive>.zarr
```

Default behavior is validate-only. It resolves
`refined_subject_masks_runs.attrs["latest"]`, checks `mask_labels` /
`available_channels` channel semantics, verifies required run arrays and
run-level metrics, requires available component subgroups to expose
`reason_bytes`, `mask_present`, `area_px`, and `edit_applied`, and fails when
required review or provenance fields are missing.

Backfill is explicit:

```bash
scripts/py -m fisheye.utils.validate_refined_subject_mask_contract <archive>.zarr --backfill
```

The backfill path is intentionally conservative. It may recreate
`available_channels` from declared component availability, recreate `masks_roi`
from component-local mask arrays when channel order is proven by `mask_labels`,
and derive missing mask metrics or component-local mirrors from existing
`masks_roi`. It must not split `eyes_union` into left/right eyes, invent review
state, or fake missing component provenance.

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
- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`

Required when the crop source exposes detect-review linkage:

- `source_detect_review_status_ref`

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

- `quality_code`
  - shape: `(N,)`
  - compact machine-generated review-routing enum
- `quality_score`
  - shape: `(N,)`
  - numeric severity for "next problematic frame" navigation
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
- `area_ratio_prev`
- `area_delta_zscore`
- `centroid_jump_px`
- `bbox_area_ratio_prev`
- `mask_present_gap`
- `component_count_jump`
- `sigma_noise`
- `curvature_var`
- `ipr`
- `solidity`

Common cross-component geometry such as centroid and bbox should stay at
run-level `metrics/`, while component-specific QC should live under
`components/<component>/metrics/`. These component-local arrays are represented
in code by `REFINED_SUBJECT_COMPONENT_METRICS`.

Why per-component subgroups:

- body and swim bladder will not share identical derived geometry
- eye refinement is even more specialized
- this avoids freezing the whole stage around one component’s geometry layout

Review queue policy:

- `quality_code`, `quality_score`, and reason tags are machine-generated
  review-routing signals, not human approval state
- temporal QC should add reason tags and quality score contributions but should
  not overwrite masks by itself
- UI navigation should be able to filter by component and jump to the next row
  with highest unresolved `quality_score`

## Cross-Component Relation Subgroups

`relations/<relation_name>/` is the standard extension point for derived values
that conceptually span multiple components and are not owned by one component.

Canonical example for eye-capable refined runs:

- `relations/eye_pair/metrics/`
  - `separation_px`
  - `separation_valid`

Why this belongs under `relations/` rather than under one eye component:

- `eye_separation` is a pairwise derived value
- duplicating it under both eye components creates synchronization risk
- it should not require inventing a fake mask component such as `eye_pair`
- this relation surface is represented in code by
  `REFINED_SUBJECT_EYE_PAIR_METRICS`

## Geometry Ownership Policy

Refined subject-mask runs own mask-local geometry primitives: values that are
computed directly from one refined component mask and are useful for mask QC,
review navigation, visualization, or lossless downstream reuse. They do not own
interpreted biological coordinate-frame metrics.

Component-specific mask-local geometry should live under:

- `components/<component>/geometry/`

Component-specific contour stores should live under:

- `components/<component>/contours/`

Run-level common mask geometry can stay under `metrics/` when it is fixed-shape
and naturally shared across every component:

- `area_px`
- `centroid_xy`
- `centroid_valid`
- `bbox_xyxy`
- `bbox_valid`

Component-local mirrors are allowed when they make component-native consumers or
review tooling simpler, but the source of truth must remain documented by the
writer's schema attrs.

Recommended component-local primitives:

### `subject_body`

- contour tables under `components/subject_body/contours/`
- centroid, area, bbox, mask-present, and validity metrics
- simple shape descriptors directly derived from the mask, such as component
  count, hole fraction, solidity, or an unoriented ellipse/PCA summary when the
  convention is explicitly documented
- approximate long-axis QC descriptors directly derived from the mask, such as
  `major_axis_length_px` or `feret_diameter_px`, when the method and sensitivity
  to contour noise are documented
- optional debug seeds for later shape fitting, if they are clearly marked as
  non-canonical seeds rather than final biological body axes

### `swim_bladder`

- contour tables under `components/swim_bladder/contours/`
- centroid, area, bbox, mask-present, and validity metrics
- simple blob/ellipse summaries directly derived from the swim-bladder mask

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
- geometry primitives stored here should be recomputable from
  `masks_roi` plus the documented method/policy attrs
- downstream `analysis/subject_shape_runs` should consume refined masks and/or these
  mask-local primitives, not raw `subject_mask_runs`

Metric-QC policy:

- `components/<component>/metrics.attrs["schema_id"]` should be
  `refined_subject_component_mask_metrics_v1`.
- `components/<component>/metrics.attrs["qc_schema_id"]` should be
  `refined_subject_component_metric_qc_reasons_v1`.
- `components/<component>/metrics.attrs["qc_policy"]` records the conservative
  component-specific gates used to derive generated metric-QC reason tags.
- Generated metric-QC reason tags use the `needs_review_metric_*` prefix so
  refresh/backfill tools can replace generated tags without deleting manual
  review tags.
- `scripts/py -m fisheye.utils.backfill_refined_subject_mask_metrics` refreshes
  mask-local metrics and generated metric-QC reason tags for existing refined
  subject-mask runs without recreating mask pixels.

## Boundary With `analysis/subject_shape_runs`

`analysis/subject_shape_runs/<run>` is the analysis home for interpreted
biological geometry that requires a coordinate convention, anatomical polarity,
temporal context, track identity, or relationships between components.

Keep these out of `refined_subject_masks_runs` as canonical outputs:

- body centerline/spline used as an anatomical coordinate frame
- canonical body B-spline fits, including centerline or outline fits with
  smoothing/knot parameters
- canonical biological body length derived from a centerline or B-spline arc
  length
- head/tail-polarized body axis or heading inferred from masks
- body curvature or bend metrics
- swim-bladder position relative to body axis or centerline
- swim-bladder distance to eye pair, body centroid, or anatomical landmarks
- eye angles relative to body/head heading
- temporally smoothed or track-aligned shape metrics

Reasoning:

- `refined_subject_masks_runs` is the curated mask-pixel authority.
- `analysis/subject_shape_runs` is a deterministic derived-analysis layer.
- Recomputing interpreted shape metrics should not mutate or re-author the
  refined masks.
- The shape stage can carry its own method version, source refined-mask run,
  optional source keypoints/heading run, track/temporal context, and failure
  state.

Practical rule:

- If the value answers "what geometry did this one refined component mask have?",
  store it with `refined_subject_masks_runs`.
- If the value answers "what biological pose/shape/relationship does this
  animal have?", store it in `analysis/subject_shape_runs` or a more specific
  downstream analysis run.

Body B-spline rule:

- refined body components may store contours and non-canonical debug seeds
- refined body components may store approximate long-axis QC descriptors such as
  Feret diameter or PCA/ellipse major-axis length
- the canonical B-spline fit belongs in `analysis/subject_shape_runs` because it depends
  on fit method, knot/parameterization policy, smoothing, validity criteria, and
  usually anatomical polarity
- the canonical biological body length should be derived from the validated
  centerline/B-spline arc length in `analysis/subject_shape_runs`, not from raw
  mask area or an unqualified contour diameter

Current implementation note:

- When both `eye_left` and `eye_right` are present,
  `refined_subject_masks_runs` materializes:
  - `components/eye_left|eye_right/geometry/ellipse_params`
  - `components/eye_left|eye_right/geometry/ellipse_success`
  - `components/eye_left|eye_right/contours/{ptr,len,points_xy}`
  - `relations/eye_pair/metrics/{separation_px,separation_valid}`
- These arrays are derived from the refined subject-mask component masks during
  refined-run creation/finalization.

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
- legacy eye-specific retune/failure tooling may still target standalone
  historical refined-eye runs explicitly
- canonical eye saves and eye review-state changes in
  `refined_subject_masks_runs` may now refresh the matching
  `refined_eye_masks_runs/<run>` as a derived compatibility artifact
- derived compatibility refined-eye runs should be treated as read-only in
  legacy viewers so canonical eye authority does not drift back out of
  `refined_subject_masks_runs`

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

Implementation note as of 2026-04-02:

- canonical eye edits and eye component review-state updates now materialize a
  compatibility `refined_eye_masks_runs/<run>` view from the canonical
  `refined_subject_masks_runs/<run>` eye components when anatomical
  `eye_left`/`eye_right` components are available
- the compatibility run now serves legacy readers such as eye-specific profile,
  export, and visualization tools, while canonical authoring authority remains
  in `refined_subject_masks_runs`

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
- At what point should compatibility materialization of `refined_eye_masks_runs`
  become opt-in rather than routine once unified refined-subject eye writes are
  available?

## Related Docs

- [subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md)
- [subject_mask_registry_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_registry_contract.md)
- [eye_subject_mask_unification_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_subject_mask_unification_design.md)
- [subject_mask_refinement_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_refinement_todo.md)
- [derived_analysis_run_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/derived_analysis_run_contract.md)
- [subject_shape_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_shape_runs_contract.md)
- [review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md)
- [pose_kinematics_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_kinematics_run_design.md)
