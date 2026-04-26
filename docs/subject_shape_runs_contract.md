# Subject Shape Runs Contract (Draft v1)
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-26
-->

Purpose: define the downstream deterministic geometry stage for biological
shape, pose, and cross-component relationships derived from canonical refined
subject masks.

## Scope

`subject_shape_runs/<run>` is the planned home for interpreted shape outputs
that should not be stored as mask-review metadata.

It should consume:

- `refined_subject_masks_runs/<run>` as the canonical mask-pixel authority
- optional refined-subject mask-local geometry primitives
- optional `refined_keypoints_runs/<run>` or heading/track runs when anatomical
  polarity, body heading, or temporal alignment is required

It should produce deterministic derived geometry and analysis-ready shape
features.

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

Use `subject_shape_runs/<run>` for interpreted biology:

- body centerline or spline used as an anatomical coordinate frame
- body B-spline fit, including centerline or outline models with method-specific
  smoothing/knot parameters
- canonical biological body length derived from centerline or B-spline arc
  length
- head/tail-polarized body axis or heading inferred from masks
- body curvature, bend, width profile, or body-shape summaries
- swim-bladder position relative to the body axis or centerline
- swim-bladder distance to body centroid, eye pair, or anatomical landmarks
- eye angles relative to body/head heading
- temporally smoothed or track-aligned shape metrics

Practical test:

- If the value is recomputable from one component mask without choosing an
  anatomical frame, keep it with `refined_subject_masks_runs`.
- If the value needs a coordinate convention, anatomical polarity, component
  relationship, track identity, temporal context, or smoothing policy, write it
  to `subject_shape_runs` or a more specific downstream analysis run.

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
  -> subject_shape_runs/<run>         # interpreted biological shape geometry
```

Optional inputs:

```text
refined_keypoints_runs/<run>
analysis/track_kinematics_runs/<run>
tracking_runs/<run>
```

## Required Provenance

A `subject_shape_runs/<run>` writer should record:

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
- `source_tracking_run`
- `source_track_kinematics_run`
- `temporal_window`
- smoothing/filter method and parameters

## Proposed Layout

```text
subject_shape_runs/
  attrs:
    latest                         "<run_id>"
  <run_id>/
    attrs:
      source_refined_subject_masks_run
      source_mask_labels
      source_mask_label_schema_id
      method
      method_version
      created_at_utc
    frame_indices                  (N,)
    detection_indices              (N,)
    source_refined_row_ids          (N,) optional
    body/
      centroid_xy                  (N, 2) optional mirror/cache
      contour_ref                  optional references into refined mask contours
      centerline_xy                (N, P, 2) optional
      centerline_valid             (N,) optional
      bspline_control_points_xy    (N, K, 2) optional
      bspline_knots                optional
      bspline_degree               scalar attr or dataset
      bspline_valid                (N,) optional
      centerline_arc_length_px     (N,) optional
      bspline_arc_length_px        (N,) optional
      axis_xy                      (N, 2) optional
      heading_rad                  (N,) optional
      curvature                    (N, P) optional
      validity/
    swim_bladder/
      centroid_xy                  (N, 2) optional mirror/cache
      ellipse_params               (N, 5) optional
      validity/
    relations/
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

## Body B-Spline Policy

The canonical body B-spline fit belongs in `subject_shape_runs`, not in
`refined_subject_masks_runs`.

Reasoning:

- a B-spline is a fitted model, not just a direct mask primitive
- its output depends on smoothing, knot count, parameterization, resampling, and
  failure policy
- if the spline is used as a body coordinate frame, it also depends on anatomical
  polarity or heading source
- recomputing or improving the fit should create or update a derived shape run
  without mutating the reviewed mask-pixel authority

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

Canonical body length should live in `subject_shape_runs`:

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

`analysis/eye_angle_runs` already computes interpreted eye angles from eye
geometry plus heading/keypoint context. That remains a valid specialized
analysis run.

`subject_shape_runs` should not force every specialized metric to move
immediately. It defines the mask-derived shape layer that can later feed or
replace specialized analyses when that migration is justified.

Recommended near-term approach:

- keep eye ellipse fits in `refined_subject_masks_runs`
- keep current eye-angle outputs in `analysis/eye_angle_runs`
- use `subject_shape_runs` first for body/swim shape features that do not
  already have a stable analysis home

## Open Questions

- Should `subject_shape_runs` live at the zarr root, like mask stage families, or
  under `analysis/subject_shape_runs` with other derived analyses?
- Which body centerline method is the first supported implementation?
- Should body/swim shape outputs be track-aligned from the start, or remain
  row-aligned with refined masks until tracking is explicitly requested?

## Related Documents

- [current_pipeline_contract.md](current_pipeline_contract.md)
- [refined_subject_masks_runs_contract.md](refined_subject_masks_runs_contract.md)
- [subject_mask_refinement_todo.md](subject_mask_refinement_todo.md)
- [pose_kinematics_run_design.md](pose_kinematics_run_design.md)
- [src/fisheye/docs/eye_angle_conventions.md](../src/fisheye/docs/eye_angle_conventions.md)
