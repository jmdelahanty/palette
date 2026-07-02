# Current Palette Pipeline Contract

<!-- contract-meta
version: 2
status: active
last_verified: 2026-07-01
-->

Purpose: define the current operator-facing source-of-truth contract for Palette
pipeline artifacts while the codebase is still migrating from eye-specific mask
stages to unified subject-mask components.

This document is the short current-state contract. Deeper design notes and TODOs
remain useful, but operator/query behavior should be judged against this file
first.

For a beginner-facing explanation of the coordinate systems, formulas, and
behavioral analytics used by these stages, start with
[`analytics_math_primer.md`](analytics_math_primer.md).

## Design Philosophy

Palette should keep these artifact roles distinct:

- Raw provenance snapshots: append-only inference or materialization outputs.
  They record what a model or method produced for a row-aligned source.
- Refined authority artifacts: curated or editable surfaces that operators and
  exports should trust for reviewed state.
- Compatibility surfaces: historical or adapter layouts kept readable for older
  archives, old viewers, and migration tooling. They are not competing
  canonical authorities.
- Refreshable caches: derived materializations that can be regenerated when
  sources change. They are not human curation authority.
- Training/export products: out-of-band products built from the current
  authority. They should be rebuilt after authority changes rather than treated
  as live pipeline state.

The default rule is:

- raw stages preserve provenance
- refined stages carry reviewed/editable authority
- registry and operator summaries should prefer the current authority and label
  compatibility state explicitly
- stale state is separate from review state

Recording-only archives follow the same artifact-role rules. They may lack
experiment/stimulus context, but they can still accumulate the full
non-stimulus analysis stack: detections, pose/keypoints, segmentation/refined
masks, subject shape, track kinematics, swim bouts, bout kinematics, and
non-stimulus exports. Stimulus runs and stimulus-response runs are optional
context-dependent layers and should be absent rather than faked when no
experiment context exists. The operational details live in
[`recording_analysis_pipeline_contract.md`](recording_analysis_pipeline_contract.md#recording-only-mode).

This current-state contract describes traditional top-level analysis/training
Zarr consumers. Clipped analysis shells are a new layout and require explicit
collection resolution before they should be treated as normal Crimson/operator
review targets. See
[`clipped_recording_consumer_mapping_contract.md`](clipped_recording_consumer_mapping_contract.md).

## Current Stage Contract

| Family | Raw provenance | Current refined authority | Compatibility/cache surfaces | Operator/query truth |
| --- | --- | --- | --- | --- |
| Import/video metadata | imported analysis/training zarr metadata | none | downsampled video products | imported archive metadata and manifest status |
| Detect | `detect_runs/<run>` | `refined_detect_runs/<run>/instances` | legacy refined subgroups such as `filtered`, `interpolated`, and `manual_*` | canonical curated refined rows when present; raw detect only as fallback or provenance |
| Detect quality | detect-run quality reports | refined detect review/status metadata | legacy detect-quality aliases | quality labels feed refine; review state belongs to refined detect |
| Crop | `crop_runs/<run>` | none in normal operation | geometry-only or repaired crop variants | current crop run that still matches selected detect/refined lineage |
| Keypoints | `keypoints_runs/<run>` | `refined_keypoints_runs/<run>` | legacy keypoint attrs such as singular `source_keypoint_run` | refined keypoints when present; metadata-driven pose and heading semantics |
| Raw segmentation | `subject_mask_runs/<run>` probability surfaces plus model/config/provenance | none | optional thresholded compatibility caches; historical `eye_masks_runs/<run>` data is read-only | unified subject-mask component availability for current mask state |
| Refined subject masks | `subject_mask_runs/<run>` sources plus component provenance | `refined_subject_masks_runs/<run>` | none; this is the canonical refined component surface | component availability, review state, and lifecycle from refined subject-mask component rows |
| Refined eye masks | historical `eye_masks_runs/<run>` when present | `refined_subject_masks_runs/<run>` for current eye review | `refined_eye_masks_runs/<run>` is historical compatibility layout | active eye geometry/export should use refined subject-mask or subject-shape eye components; legacy refined-eye data is inspectable history |
| Swim bladder | raw probability surfaces in `subject_mask_runs/<run>` | `refined_subject_masks_runs/<run>/components/swim_bladder` | coarse thresholded swim-bladder masks are compatibility/refinement caches | refined subject-mask swim-bladder component state |
| Subject shape | refined subject-mask component masks and optional mask-local geometry | none; derived deterministic analysis layer | `analysis/subject_shape_runs/<run>` as the coherent body/eyes/swim shape and shared body-frame surface; specialized downstream analysis runs may consume it | shape outputs must reference exact refined-mask source and any heading/keypoint/track inputs |
| Tail kinematics | ordered tail geometry from `analysis/subject_shape_runs` or future keypoint-derived tail posture | none; derived deterministic analysis layer | `analysis/tail_kinematics_runs/<run>` for body-frame tail angles, lateral deflections, and curvature summaries; Megabouts/ZebraZoom/Stytra views are adapters | tail traces must reference exact geometry source and record angle/sign/unit conventions |
| Eye angles | refined keypoints plus eye geometry from subject-shape/refined-subject/refined-eye sources | none; specialized deterministic analysis layer | `analysis/eye_angle_runs/<run>` with `schema_id = "analysis.eye_angle_runs"`, run schema v5, and output schema v7 | current runs prefer subject-shape eye geometry when available, use resolved major-axis orientation as canonical, derive gaze/minor-axis arrays plus BEAST/Bianco-compatible surfaces, expose `eye_angle_variant_schema` for UI representation selection, and record exact geometry/keypoint/body-frame lineage |
| Arena assignment/tracking | selected detect/refined lineage outputs | tracking QC/status metadata | older raw-detect-aligned assignments | assignment/tracking rows whose source lineage matches the selected detect/refined state |

## Mask-Specific Rules

The mask transition is the main place where the contract is not yet a finished
steady state.

Current rules:

- `subject_mask_runs` is the canonical raw multi-component mask family, with
  raw model output represented by probability surfaces plus model/config and
  provenance rather than thresholded masks.
- `refined_subject_masks_runs` is the canonical editable/refined component mask
  family for body, eyes, and swim bladder.
- The current implemented U-Net subject-mask path can train from merged
  subject-mask training artifacts, resolve trained models from the registry,
  write raw probability-first `subject_mask_runs/<run>` snapshots, and finalize
  `subject_v1_union` outputs into `subject_v1_lr`
  `refined_subject_masks_runs/<run>` candidates with `subject_body`,
  `eye_left`, `eye_right`, and `swim_bladder`.
- The smart finalizer is the canonical bridge from raw probabilities to refined
  binary component masks. It records cleanup reasons, component metrics,
  assignment provenance, Dask execution metadata, and review triage counts.
- Refined-subject eye geometry is now materialized from
  `refined_subject_masks_runs/<run>/components/eye_left|eye_right` and records
  `eye_geometry_status=computed` when the geometry/relations arrays are present.
- Direct mask-local primitives such as component contours, centroids, bboxes,
  areas, validity flags, and simple component shape descriptors belong with
  `refined_subject_masks_runs`.
- Interpreted biological geometry such as body centerlines/splines, canonical
  body B-spline fits, canonical body length from centerline/B-spline arc length,
  head/tail axes, swim-bladder position relative to body axis, analysis-facing
  eye component geometry, eye-pair relationships, and eye angles relative to
  heading belong in `analysis/subject_shape_runs` when they are part of coherent
  mask-derived body/eyes/swim geometry. Specialized downstream analysis runs
  should consume that surface when possible.
- Fish-relative coordinates should use the body-frame contract. Keypoint
  heading remains the fallback estimator, while shared mask/spline/hybrid body
  frames should materialize under
  `analysis/subject_shape_runs/<run>/body_frame/` when available.
- Track-level motion should come from
  `analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>`, not from
  legacy `analysis/movement_runs` or ad hoc subject-shape-origin differencing.
  Swim-bout event windows live in `analysis/swim_bout_runs`; new accepted
  swim-bout runs default to the compact-v2 tabular layout, while
  hierarchical-v1 remains an explicit compatibility option. Physical per-bout
  measurements live in linked `analysis/bout_kinematics_runs`. The
  Crimson-facing reader contract is
  [`crimson_track_motion_read_contract.md`](./crimson_track_motion_read_contract.md).
- Source stimulus timing and geometry should come from
  `analysis/stimulus_runs/<run>/steps/step_<i>`, not from downstream
  `stimulus_response_runs` or ad hoc protocol JSON parsing. Derived response
  runs join back to this source surface by `source_stimulus_run` and
  `step_index`. The Crimson-facing reader contract is
  [`crimson_stimulus_step_read_contract.md`](./crimson_stimulus_step_read_contract.md).
- Production assembly/export from `refined_subject_masks_runs` is
  approved-only by default; pending or missing component reviews require an
  explicit draft/QA override.
- Standalone `eye_masks_runs` / `refined_eye_masks_runs` production is retired.
  Historical groups remain readable in old zarrs and registry/status views, but
  no current workflow should create new standalone eye-mask runs.
- New eye-capable mask work should write `subject_mask_runs` and finalize into
  `refined_subject_masks_runs` components.
- `refined_eye_masks_runs` remains a historical compatibility layout only; it
  is not a manual review authority for new operator-facing eye state.
- Mask-level eye geometry and export consumers should use subject-shape or
  refined-subject eye components. Historical refined-eye groups can be inspected
  directly, but are not part of the live resolver path.
- Eye-angle analysis is a specialized downstream consumer. It now opts into
  `analysis/subject_shape_runs` as the preferred eye-geometry source when
  left/right eye ellipse geometry is present, records
  `source_geometry_kind`, and writes `schema_id = "analysis.eye_angle_runs"`
  plus `eye_angle_output_schema` to describe its current output groups. Schema
  v5 makes `preferred_angle_family="gaze"`,
  `preferred_eye_axis="ellipse_major"`, and `support/body_frame/` explicit.
  The major axis is canonical, while gaze/minor direction is derived from the
  resolved major axis; legacy major/minor arrays are retained for compatibility
  and QA. Output schema v7 adds `eye_angle_variant_schema` so marimo, Crimson,
  and other readers can select eye-frame, gaze, nasal-gaze, major-axis,
  centroid, or legacy representations from metadata. It also
  keeps `vergence_gaze_deg` as the v3-compatible total/axis-separation
  aggregate while adding `mean_eye_vergence_gaze_deg` for Johnson/BEAST-style
  mean per-eye convergence.

The target steady state is one segmentation orchestration surface that writes
one coherent probability-backed `subject_mask_runs/<run>` snapshot with
component-scoped method and provenance metadata. The U-Net subject-mask CLI now
implements that artifact shape for dense union-eye models, but the broader
component/method orchestration layer is still open. Thresholding, morphology,
review, and approval happen in `refined_subject_masks_runs/<run>`. Old
eye-specific stages are tolerated as transition surfaces, not design precedent.

## Registry And Operator Surface Rules

Registry and operator output should answer current availability and review state
from unified component surfaces wherever they exist.

Required behavior:

- Subject-mask component rows should rank available
  `refined_subject_masks_runs` components ahead of legacy eye rows.
- Partial refined-subject runs are valid. A refined run with only one reviewed
  component should be visible for that component and should not be hidden by a
  broader raw or legacy run.
- Stale lifecycle state must remain visible even when a row is otherwise the
  preferred component row.
- Subject-mask component registry views should project refined-run
  `source_subject_mask_stale_*` fields so query, training, and operator
  surfaces can explain why a component is stale without treating stale as a
  review label.
- `check_recording_steps` should label `eye_masks` and `refined_eye_masks` as
  legacy compatibility and report unified component summaries separately.
- Query tools may expose legacy eye-mask filters for diagnostics and historical
  training compatibility, but current mask-state filters should use subject-mask
  component views.

## Staleness Rules

Stale state is a source-lineage problem, not a review-state problem.

Required behavior:

- Raw provenance runs should not be edited to hide source changes.
- Curated refined rows may be preserved after a source change only when row
  identity is still stable.
- Preserved curated rows should be marked stale or queued for targeted review,
  not silently treated as clean.
- Refreshable caches can be regenerated or partially refreshed when lineage is
  stable.
- Identity-breaking changes should escalate to rerun or run-level invalidation.

For masks, the intended canonical stale payload is visible at the refined
subject-mask level and projected into registry/query/operator surfaces. Existing
eye-mask stale payloads are the precedent, not the final mask-wide answer.

## Current Known Gaps

These gaps are allowed transition state, but they should not be treated as the
desired design:

- There is no completed top-level `segmentation` orchestration step with a
  central method-capability table. The direct U-Net subject-mask CLI writes one
  coherent body/eyes/swim raw snapshot, but the broader component/method
  orchestration layer remains open.
- Some registry query and training-prep paths still expose legacy eye-mask
  filters as primary-looking options.
- Subject/refined-subject stale repair is not yet as complete as the eye-mask
  stale precedent, though component registry/query/operator projection now
  exposes refined-run source-stale state.
- Older operator docs may list component-specific segmentation commands before
  describing the unified target.
- Detect row-local stale repair now has a contract and validator, but downstream
  crop/keypoint/mask consumers still need a full audit for positional or legacy
  manual-subgroup assumptions.
- Smart-finalized refined runs are candidates until visual inspection and
  component approval are complete; generated `pending`/`needs_review` counts are
  triage state, not training approval.
- Temporal QC for abrupt area/centroid/component-count changes is planned as a
  second pass that flags rows without overwriting spatial masks.
- `analysis/subject_shape_runs` has a coherent body/eyes/swim writer with
  materialized `body_frame/`, snout-anchored centerline, B-spline, and tail
  geometry outputs. `analysis/eye_angle_runs` and
  `analysis/tail_kinematics_runs` now consume those surfaces when available;
  additional downstream consumers remain open.
- Subject-shape length stability QC is defined as a downstream analysis layer,
  but body/tail length distribution summaries, temporal delta flags, and
  multi-reason length-QC tags are not yet implemented.

## Review Checklist

When reviewing new pipeline work, ask:

1. What is the raw provenance artifact?
2. What is the refined or operator-authoritative artifact?
3. Is any legacy surface clearly labeled as compatibility?
4. Does the registry/query surface prefer the same authority as runtime/export?
5. Are partial component runs handled without hiding stale state?
6. Does source drift become explicit stale state rather than a review toggle?
7. Does the change move code toward one raw segmentation snapshot and one
   refined component authority?

## Related Documents

- [recording_analysis_pipeline_contract.md](recording_analysis_pipeline_contract.md)
- [derived_analysis_run_contract.md](derived_analysis_run_contract.md)
- [segmentation_pipeline_step_todo.md](segmentation_pipeline_step_todo.md)
- [subject_mask_refinement_todo.md](subject_mask_refinement_todo.md)
- [subject_mask_stage_unification_todo.md](subject_mask_stage_unification_todo.md)
- [subject_shape_runs_contract.md](subject_shape_runs_contract.md)
- [body_frame_contract.md](body_frame_contract.md)
- [body_spline_tail_anchor_design.md](body_spline_tail_anchor_design.md)
- [tail_kinematics_run_design.md](tail_kinematics_run_design.md)
- [tail_kinematics_tool_interop_design.md](tail_kinematics_tool_interop_design.md)
- [repo_wide_staleness_checklist.md](repo_wide_staleness_checklist.md)
- [repo_wide_staleness_gap_matrix.md](repo_wide_staleness_gap_matrix.md)
- [refined_detect_row_identity_contract.md](refined_detect_row_identity_contract.md)
- [src/fisheye/docs/provenance_workflow.md](../src/fisheye/docs/provenance_workflow.md)
