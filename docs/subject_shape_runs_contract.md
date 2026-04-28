# Subject Shape Runs Contract (Draft v1)
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-28
-->

Purpose: define the downstream deterministic analysis layer for biological
shape, pose, and cross-component relationships derived from canonical refined
subject masks.

## Scope

`analysis/subject_shape_runs/<run>` is the home for interpreted shape
outputs that should not be stored as mask-review metadata.

It should consume:

- `refined_subject_masks_runs/<run>` as the canonical mask-pixel authority
- optional refined-subject mask-local geometry primitives
- optional `refined_keypoints_runs/<run>` or heading/track runs when anatomical
  polarity, body heading, or temporal alignment is required

It should produce deterministic derived geometry and analysis-ready shape
features.

The first implementation is `fisheye.analysis.subject_shape_runs`. It writes
row-aligned component summaries, body principal-axis estimates, eye/swim ellipse
summaries, eye-pair relations, and swim/eye-to-body relations with optional
Dask worker-chunk execution. Body centerline and B-spline methods remain
follow-up shape methods under this same run family. The same run family is also
the preferred materialized home for shared body-frame arrays derived from
refined masks, centerlines, B-splines, keypoints, or hybrid estimators.

## Boundary Rule

Use `refined_subject_masks_runs/<run>` for mask-local primitives:

- component contours
- component centroids
- component area and bbox
- component mask-present and validity metrics
- simple mask-shape descriptors such as component count, hole fraction,
  solidity, and documented ellipse/PCA fits
- eye ellipse parameters and eye-pair separation when used as immediate
  refined-eye geometry/QC

Use `analysis/subject_shape_runs/<run>` for interpreted biology:

- body centerline or spline used as an anatomical coordinate frame
- body B-spline fit, including centerline or outline models with method-specific
  smoothing/knot parameters
- canonical biological body length derived from centerline or B-spline arc
  length
- head/tail-polarized body axis or heading inferred from masks
- body curvature, bend, width profile, or body-shape summaries
- swim-bladder position relative to the body axis or centerline
- swim-bladder distance to body centroid, eye pair, or anatomical landmarks
- analysis-facing eye component geometry when it is part of the same coherent
  body/eyes/swim subject-shape run
- eye-pair metrics that are consumed as biological geometry rather than
  immediate mask-local QC
- eye angles relative to body/head heading
- temporally smoothed or track-aligned shape metrics

Practical test:

- If the value is recomputable from one component mask without choosing an
  anatomical frame, keep it with `refined_subject_masks_runs`.
- If the value needs a coordinate convention, anatomical polarity, component
  relationship, track identity, temporal context, or smoothing policy, write it
  to `analysis/subject_shape_runs` or a more specific downstream analysis run.

Mask-quality prerequisite:

- `subject_body` mask-local QC belongs with `refined_subject_masks_runs`; see
  [subject_body_mask_qc_design.md](subject_body_mask_qc_design.md).
- Subject-shape writers should fail closed or mark warnings when source
  mask-level QC indicates a severe or review-required body mask, but
  `analysis/subject_shape_runs` should not be the primary authority for whether
  the refined mask pixels are plausible.

## Non-Goals

- Do not store raw model probabilities here.
- Do not edit or approve mask pixels here.
- Do not replace `refined_subject_masks_runs` as the mask authority.
- Do not collapse specialized analyses such as `analysis/eye_angle_runs` or
  `analysis/pose_kinematics_runs` into this stage without a clear migration
  decision.

## Intended Stage Relationship

```text
subject_mask_runs/<run>               # raw probability evidence
  -> refined_subject_masks_runs/<run> # canonical refined masks + mask-local geometry
  -> analysis/subject_shape_runs/<run> # interpreted biological shape geometry
```

Optional inputs:

```text
refined_keypoints_runs/<run>
analysis/track_kinematics_runs/<run>
tracking_runs/<run>
```

## Required Provenance

An `analysis/subject_shape_runs/<run>` writer should record:

- `schema_id = "analysis.subject_shape_runs"`
- `schema_version`
- `row_axis = "refined_subject_mask_rows"` for the first row-aligned writer
- `source_refined_subject_masks_run`
- `source_refined_subject_masks_stage = "refined_subject_masks_runs"`
- `source_mask_labels`
- `source_mask_label_schema_id`
- `source_mask_geometry_schema_id` when mask-local geometry was consumed
- method name and method version
- parameter/config hash or serialized config
- creation timestamp

Required when used:

- `source_refined_keypoints_run`
- `source_keypoint_heading_computation`
- `body_frame_schema_id`
- `body_frame_schema_version`
- `body_frame_estimator`
- `body_frame_source_refs`
- `source_tracking_run`
- `source_track_kinematics_run`
- `temporal_window`
- smoothing/filter method and parameters

## Proposed Layout

```text
analysis/subject_shape_runs/
  attrs:
    latest                         "<run_id>"
  <run_id>/
    attrs:
      schema_id                    "analysis.subject_shape_runs"
      schema_version               1
      source_refined_subject_masks_run
      source_mask_labels
      source_mask_label_schema_id
      method
      method_version
      created_at_utc
      row_axis                     "refined_subject_mask_rows"
      source_refs                  dict of exact input runs/paths
    row_index/
      frame_indices                (N,)
      detection_indices            (N,) optional
      source_refined_row_ids        (N,) optional
    body_frame/                     optional shared fish anatomical frame
      origin_xy                     (N, 2)
      forward_axis_xy               (N, 2)
      left_axis_xy                  (N, 2)
      heading_deg                   (N,)
      valid                         (N,)
      failure_reason_bytes          (N, width) optional uint8 utf8-null-terminated tags
      midline_xy                    (N, P, 2) optional
      arclength_px                  (N,) optional
    components/
      subject_body/
        centroid_xy                (N, 2) optional mirror/cache
        contour_ref                optional references into refined mask contours
        centerline_xy              (N, P, 2) optional
        centerline_valid           (N,) optional
        bspline_control_points_xy  (N, K, 2) optional
        bspline_sample_xy          (N, P, 2) optional
        bspline_knots              optional
        bspline_degree             scalar attr or dataset
        bspline_valid              (N,) optional
        centerline_arc_length_px   (N,) optional
        bspline_arc_length_px      (N,) optional
        axis_xy                    (N, 2) optional
        heading_rad                (N,) optional
        tail_tip_xy                (N, 2) optional
        tail_base_xy               (N, 2) optional
        tail_base_valid            (N,) optional
        tail_base_arclength_px     (N,) optional
        tail_base_failure_reason_bytes (N, width) optional
        tail_segment_arclength_px  (N,) optional
        body_arclength_px          (N,) optional
        tail_sample_s              (K,) optional normalized tail arclength samples
        tail_sample_xy             (N, K, 2) optional
        tail_tangent_xy            (N, K, 2) optional
        tail_normal_xy             (N, K, 2) optional
        tail_curvature_px_inv      (N, K) optional
        tail_width_px              (N, K) optional
        tail_width_valid           (N, K) optional
        curvature                  (N, P) optional whole-centerline curvature
        validity/
      swim_bladder/
        centroid_xy                (N, 2) optional mirror/cache
        ellipse_params             (N, 5) optional
        caudal_contour_point_xy    (N, 2) optional
        caudal_contour_projection_px (N,) optional
        caudal_contour_valid       (N,) optional
        caudal_contour_failure_reason_bytes (N, width) optional
        validity/
      eye_left/
        ellipse_params             (N, 5) optional mirror/cache
        validity/
      eye_right/
        ellipse_params             (N, 5) optional mirror/cache
        validity/
    relations/
      eye_pair/
        separation_px              (N,) optional mirror/cache
        separation_valid           (N,) optional
      swim_bladder_to_body/
        longitudinal_position      (N,) optional
        lateral_offset_px          (N,) optional
        distance_to_centerline_px  (N,) optional
      eyes_to_body/
        left_eye_angle_rad         (N,) optional
        right_eye_angle_rad        (N,) optional
```

This layout is intentionally permissive. The first implementation should write
only the arrays it can validate.

## Body Frame Placement

`analysis/subject_shape_runs/<run>/body_frame/` is the preferred shared
materialized location for fish anatomical body-frame outputs.

Reasoning:

- a body frame is deterministic derived biology, not reviewed mask-pixel
  authority
- the best estimator may come from body masks, centerlines, B-splines, keypoints,
  mask component centroids, or a hybrid of those sources
- downstream consumers such as eye angles, body/tail shape metrics, and
  bout-level metrics should be able to consume the same frame without
  duplicating sign and polarity conventions

The body-frame contract separates semantic anchors from estimators:

- semantic anchors name anatomy, such as `swim_bladder`, `eye_left`,
  `eye_right`, `subject_body`, `tail_tip`, or `snout_tip`
- estimators declare how those anchors were measured and how frame arrays were
  materialized
- outputs expose shared arrays such as `origin_xy`, `forward_axis_xy`,
  `left_axis_xy`, `heading_deg`, and `valid`

Keypoint-only datasets remain valid. A writer may materialize a body frame from
`pose_schema.metadata.heading_computation` when masks or body splines are not
available. Mask/spline estimators should preserve keypoint or mask-component
anchors for head/tail polarity and anatomical left/right resolution.

See [body_frame_contract.md](body_frame_contract.md).

## Component And Relation Organization

`analysis/subject_shape_runs` should preserve the same semantic component names
used by `refined_subject_masks_runs`, but the meaning is different:

- `refined_subject_masks_runs/components/<component>` owns reviewed mask pixels,
  mask-local QC, and component-local geometry that is directly recomputable from
  one mask channel.
- `analysis/subject_shape_runs/components/<component>` owns interpreted
  biological geometry derived from those component masks.

Use component groups for values whose primary subject is one semantic component:

- `components/subject_body` for centerlines, B-splines, body length, body axis,
  curvature, tail-normalized width profiles, and body-shape validity.
- `components/swim_bladder` for swim-bladder centroid/blob/ellipse summaries
  and component-specific validity.
- `components/eye_left` and `components/eye_right` for analysis-facing eye
  component geometry, ellipse/axis summaries, and component-specific eye
  validity consumed by coherent subject-shape analysis.

Use `relations/` for values whose meaning depends on more than one component or
an external coordinate frame:

- `relations/eye_pair` for cross-eye metrics such as separation.
- `relations/swim_bladder_to_body` for swim-bladder position along or relative
  to the body axis/centerline.
- `relations/eyes_to_body` for eye angles or offsets relative to body/head
  heading.

Component groups in `analysis/subject_shape_runs` are not approval surfaces. A
shape run may mark a component-derived value invalid or failed without changing
the source component's review state in `refined_subject_masks_runs`.

## Body B-Spline Policy

The canonical body B-spline fit belongs in
`analysis/subject_shape_runs`, not in `refined_subject_masks_runs`.

Reasoning:

- a B-spline is a fitted model, not just a direct mask primitive
- its output depends on smoothing, knot count, parameterization, resampling, and
  failure policy
- if the spline is used as a body coordinate frame, it also depends on anatomical
  polarity or heading source
- recomputing or improving the fit should create or update a derived analysis
  shape run without mutating the reviewed mask-pixel authority

Allowed refined-mask-side exception:

- a writer may store raw component contours or clearly marked non-canonical debug
  seeds with the refined body component
- those seeds must not be treated as the canonical body spline or body axis

Minimum recommended B-spline provenance:

- `source_refined_subject_masks_run`
- `source_component = "subject_body"`
- contour or mask source used for the fit
- spline method/version
- spline degree
- knot/parameterization policy
- smoothing or regularization parameters
- head/tail polarity source if the spline is oriented
- per-row validity/failure reason

## Body Length Policy

Palette should distinguish approximate mask-QC long-axis measurements from
canonical biological body length.

Mask-local approximations may live in `refined_subject_masks_runs`:

- `major_axis_length_px` from a documented PCA or ellipse fit
- `feret_diameter_px` from the maximum contour-point separation

Those values are useful for QC, triage, and rough size filtering, but they are
not the canonical biological body length because they are sensitive to contour
noise, fins, posture, and the chosen approximation.

Canonical body length should live in `analysis/subject_shape_runs`:

- `centerline_arc_length_px` when derived from a validated centerline
- `bspline_arc_length_px` when derived from a validated body B-spline

Required semantics:

- length units must be explicit (`px`, or calibrated physical units when
  available)
- the writer must record the source centerline/B-spline method and sampling
  convention
- invalid or ambiguous fits must set the length value to NaN and write a
  validity/failure reason
- if both an approximate long-axis metric and a spline/centerline length exist,
  downstream biological analyses should prefer the spline/centerline length

## Relationship To Existing Analysis Runs

`analysis/eye_angle_runs` computes interpreted eye angles from eye geometry plus
heading/keypoint context. It remains a valid specialized analysis run, but it
is not the first authority for mask-derived eye shape geometry in unified
body/eyes/swim workflows.

Current eye-angle v4 runs opt into `analysis/subject_shape_runs` as the
preferred source when left/right eye ellipse geometry is present. They record
`schema_id = "analysis.eye_angle_runs"`, `schema_version = 4`,
`method = "ellipse_and_centroid_eye_angles"`,
`row_axis = "keypoint_detection_rows"`, `source_geometry_kind`, and
`eye_angle_output_schema` so consumers can distinguish subject-shape,
refined-subject, and legacy refined-eye geometry sources. Schema v4 also
records `preferred_angle_family = "gaze"` and
`preferred_eye_axis = "ellipse_minor"` because the ellipse minor axis is the
biologically preferred eye-look direction in current overhead imagery, and it
materializes keypoint-derived `support/body_frame/` arrays for signed-angle
polarity. It retains the v3-compatible `vergence_gaze_deg` total/axis
separation and adds per-eye nasal gaze plus
`mean_eye_vergence_gaze_deg` for Johnson/BEAST-style comparisons.

`analysis/subject_shape_runs` should not force every specialized metric to move
immediately. It defines the mask-derived shape layer that can later feed or
replace specialized analyses when that migration is justified.

Recommended near-term approach:

- keep refined-subject eye contours, ellipse fits, and eye-pair checks in
  `refined_subject_masks_runs` when they are mask-local QC/source primitives.
- include `eye_left` and `eye_right` component geometry in
  `analysis/subject_shape_runs` when producing a coherent body/eyes/swim shape
  run.
- keep current eye-angle outputs in `analysis/eye_angle_runs`; eye-angle writers
  should consume `analysis/subject_shape_runs` when mask-derived eye geometry is
  available there, with refined-subject and refined-eye geometry retained as
  explicit compatibility fallbacks.
- do not create a separate eye-analysis authority for mask-derived eye geometry
  unless it is a downstream temporal, behavioral, or task-specific analysis.

## Open Questions

- Which body centerline method is the first supported implementation?
- Should body/eyes/swim shape outputs be track-aligned from the start, or
  remain row-aligned with refined masks until tracking is explicitly requested?
- What approval/quality threshold should be required before a mask/spline body
  frame supersedes a keypoint-only fallback?
- How should the first tail-anchor/spline implementation choose between raw
  centerline samples and B-spline samples for canonical body and tail length?
  See [body_spline_tail_anchor_design.md](body_spline_tail_anchor_design.md).
- What tail sampling density and width-probe policy should become the default
  for tail curvature and mask-width profiles?

## Related Documents

- [body_spline_tail_anchor_design.md](body_spline_tail_anchor_design.md)
- [body_frame_contract.md](body_frame_contract.md)
- [current_pipeline_contract.md](current_pipeline_contract.md)
- [derived_analysis_run_contract.md](derived_analysis_run_contract.md)
- [refined_subject_masks_runs_contract.md](refined_subject_masks_runs_contract.md)
- [subject_mask_refinement_todo.md](subject_mask_refinement_todo.md)
- [pose_kinematics_run_design.md](pose_kinematics_run_design.md)
- [src/fisheye/docs/eye_angle_conventions.md](../src/fisheye/docs/eye_angle_conventions.md)
