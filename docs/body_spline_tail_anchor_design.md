# Body Spline Tail Anchor Design
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-28
-->

Purpose: define how Palette should represent mask/spline-derived tail geometry
without confusing it with pose-schema keypoints that may also use the semantic
label `tail_tip`.

## Design Decision

Palette should separate anatomical meaning from measurement source.

- `tail_tip` is a semantic anatomical label.
- A pose/keypoint run may measure `tail_tip` as a labeled keypoint.
- A subject-shape run may measure `tail_tip` as the posterior endpoint of a
  body-mask centerline or B-spline.
- These measurements should be comparable, but they should not overwrite each
  other or be stored in the same array family.

The mask/spline tail implementation should live under
`analysis/subject_shape_runs/<run>`, not under `refined_subject_masks_runs`.
Refined subject masks remain the reviewed mask-pixel authority. Subject-shape
runs own interpreted biological geometry.

## Core Concepts

### Semantic Tail Tip

`tail_tip` means the posterior/caudal anatomical endpoint of the animal.

This is a semantic concept, not a storage path. Writers must identify how that
semantic point was estimated.

Examples:

- `refined_keypoints_runs/<run>/keypoints_roi[..., tail_tip]`
  - semantic label: `tail_tip`
  - estimator family: keypoint model/manual keypoint refinement
- `analysis/subject_shape_runs/<run>/components/subject_body/tail_tip_xy`
  - semantic label: `tail_tip`
  - estimator family: subject-body centerline or B-spline posterior endpoint

### Caudal Swim-Bladder Anchor

The swim-bladder contour point furthest caudally should be stored as a distinct
anchor, not as the tail tip.

Recommended name:

```text
caudal_swim_bladder_contour_point_xy
```

Definition:

```text
the point on the swim-bladder contour with the minimum projection along the
body-frame forward axis, where forward points posterior-to-anterior
```

This point is useful as a stable internal landmark for defining where the tail
segment begins, but it is not itself the tail.

### Tail Base And Tail Segment

The tail segment should be defined relative to the body centerline or B-spline.

Recommended definitions:

- `tail_base_xy`: the point on the body centerline/spline associated with the
  caudal swim-bladder contour anchor.
- `tail_tip_xy`: the posterior endpoint of the oriented body centerline/spline.
- `tail_segment_arclength_px`: centerline/spline arc length from `tail_base_xy`
  to `tail_tip_xy`.
- `body_arclength_px`: centerline/spline arc length from the anterior endpoint
  to `tail_tip_xy`.

## Body Frame Requirement

"Caudal" is undefined until a body-frame polarity is known.

Near-term preferred polarity source:

```text
swim_bladder -> midpoint(eye_left, eye_right)
```

This matches the current body-frame contract:

- `forward_axis_xy` points posterior-to-anterior.
- `left_axis_xy` points anatomical left.
- scalar `forward_coordinate = dot(point - origin, forward_axis_xy)`.

With that convention:

- anterior points have larger forward-coordinate values.
- caudal/posterior points have smaller forward-coordinate values.
- the caudal swim-bladder contour point is the contour point with the minimum
  forward-coordinate value.

Future body-frame estimators may use keypoints, mask components, centerlines,
B-splines, or hybrid methods. The tail/spline writer must record the exact body
frame estimator and source refs used.

## Proposed Storage

The first implementation should extend `analysis/subject_shape_runs/<run>` with
optional arrays. Writers should omit arrays they cannot validate.

```text
analysis/subject_shape_runs/<run>/
  attrs:
    schema_id
    schema_version
    method
    method_version
    source_refined_subject_masks_run
    source_refined_keypoints_run optional
    body_frame_schema_id
    body_frame_schema_version
    body_frame_estimator
    body_frame_source_refs
    tail_geometry_schema_id         "analysis.subject_shape.tail_geometry"
    tail_geometry_schema_version    1
    tail_tip_semantic_label         "tail_tip"
    tail_tip_estimator              "subject_body_centerline_posterior_endpoint"
    tail_base_definition            "body_centerline_projection_of_caudal_swim_bladder_contour_point"
    caudal_anchor_definition        "min_projection_on_body_forward_axis"

  body_frame/
    origin_xy
    forward_axis_xy
    left_axis_xy
    heading_deg
    valid
    failure_reason_bytes
    midline_xy                     optional mirror of selected centerline/spline samples
    arclength_px                   optional mirror of selected body arclength

  components/
    swim_bladder/
      caudal_contour_point_xy
      caudal_contour_projection_px
      caudal_contour_valid
      caudal_contour_failure_reason_bytes

    subject_body/
      centerline_xy
      centerline_valid
      centerline_failure_reason_bytes
      bspline_control_points_xy
      bspline_sample_xy
      bspline_knots
      bspline_degree
      bspline_valid
      bspline_failure_reason_bytes
      head_endpoint_xy
      tail_tip_xy
      tail_base_xy
      tail_base_arclength_px
      tail_segment_arclength_px
      body_arclength_px
      curvature

  relations/
    keypoint_tail_to_spline_tail/
      distance_px                  optional when pose `tail_tip` exists
      valid
      failure_reason_bytes
```

## Relationship To Pose Schemas

Pose schemas may include a keypoint named `tail_tip`.

The spline implementation should not:

- backfill pose-schema `tail_tip` values from masks by default
- overwrite `refined_keypoints_runs/<run>/keypoints_roi[..., tail_tip]`
- assume a keypoint `tail_tip` and spline `tail_tip_xy` are equivalent

The spline implementation may:

- use pose `tail_tip` as an optional polarity or endpoint validation signal
- write a comparison metric between pose `tail_tip` and spline `tail_tip_xy`
- expose source refs so downstream consumers can choose either measurement

For body length, tail length, centerline curvature, and body-shape analytics,
the preferred source should be the spline-derived `tail_tip_xy` because it is
defined in the same geometry model as the body centerline.

For keypoint-only workflows, pose-schema `tail_tip` remains valid and should
continue to work without masks.

## Proposed Algorithm

### Stage 1: Anchor Extraction

Inputs:

- refined subject-body mask
- refined swim-bladder mask
- eye labels or keypoints sufficient to resolve body-frame polarity

Steps:

1. Resolve or compute body-frame axes for the row.
2. Extract the swim-bladder contour from the swim-bladder mask.
3. Project every swim-bladder contour point onto the body-frame forward axis.
4. Select the point with minimum forward projection as the caudal
   swim-bladder contour point.
5. Write the caudal anchor arrays and failure reasons.

Expected failure reasons:

- `missing_subject_body_mask`
- `missing_swim_bladder_mask`
- `missing_body_frame`
- `ambiguous_body_frame_polarity`
- `empty_swim_bladder_contour`
- `fragmented_swim_bladder_mask`

### Stage 2: Centerline And Spline

Inputs:

- refined subject-body mask
- body-frame polarity
- optional caudal swim-bladder contour point
- optional pose-schema `tail_tip` keypoint for validation

Steps:

1. Extract a subject-body skeleton or medial-axis centerline candidate.
2. Prune branches and choose the primary head-tail path.
3. Orient the path using body-frame polarity.
4. Fit a B-spline or sampled centerline model.
5. Define `head_endpoint_xy` as the anterior endpoint.
6. Define `tail_tip_xy` as the posterior endpoint.
7. Project the caudal swim-bladder contour point onto the oriented centerline.
8. Define `tail_base_xy` at that projection or nearest valid centerline point.
9. Compute body arclength, tail-segment arclength, and optional curvature.
10. Write validity and failure reasons instead of inventing values.

Expected failure reasons:

- `body_centerline_fit_failed`
- `body_centerline_branch_ambiguous`
- `tail_endpoint_ambiguous`
- `tail_base_projection_failed`
- `tail_tip_outside_body_mask`
- `spline_fit_failed`
- `source_row_stale`

## Visualization Requirement

The canary visualization should overlay:

- subject-body mask outline
- swim-bladder contour
- caudal swim-bladder contour point
- body-frame forward axis
- centerline or B-spline samples
- head endpoint
- tail base
- tail tip
- optional pose-schema `tail_tip` comparison point

This visualization should be created before treating the spline output as a
trusted analysis source.

## Implementation Checklist

### Documentation

- [x] Define semantic `tail_tip` versus source-specific measurements.
- [x] Define caudal swim-bladder contour anchor separately from tail tip.
- [x] Define tail base and tail segment relative to the subject-body
  centerline/spline.
- [ ] Update `subject_shape_runs_contract.md` proposed layout with final array
  names after implementation.
- [ ] Update `zarr_structure.md` once the writer exists.
- [ ] Add the tail/spline convention to the analytics primer after canary
  validation.

### Schema And Provenance

- [ ] Add tail-geometry schema attrs to subject-shape runs.
- [ ] Record body-frame estimator and source refs for every tail/spline run.
- [ ] Record whether pose `tail_tip` was absent, used for validation, or used
  as an estimator input.
- [ ] Store per-row validity arrays and stable failure reason tags.
- [ ] Add optional pose-tail-to-spline-tail comparison metrics.

### Writer Implementation

- [ ] Add a subject-shape method version for tail-anchor extraction.
- [ ] Implement caudal swim-bladder contour extraction.
- [ ] Implement body-frame projection for contour points.
- [ ] Implement centerline extraction from subject-body masks.
- [ ] Implement centerline orientation from body-frame polarity.
- [ ] Implement B-spline fitting or sampled centerline output.
- [ ] Compute `tail_tip_xy`, `tail_base_xy`, `body_arclength_px`, and
  `tail_segment_arclength_px`.
- [ ] Keep outputs row-aligned to refined subject-mask rows for the first pass.
- [ ] Do not mutate refined mask pixels or refined keypoint arrays.

### Validation

- [ ] Add in-memory unit tests for projection and caudal-anchor selection.
- [ ] Add in-memory unit tests for semantic source separation between pose
  `tail_tip` and spline `tail_tip_xy`.
- [ ] Add fixture tests for failure reasons on missing/ambiguous masks.
- [ ] Run a canary on the feeding recording.
- [ ] Compare spline `tail_tip_xy` against pose `tail_tip` when available.
- [ ] Inspect persisted overlay PNGs before using outputs downstream.

### Downstream Consumers

- [ ] Teach body/tail shape consumers to prefer subject-shape spline tail
  geometry when valid.
- [ ] Keep keypoint-only workflows using pose-schema `tail_tip`.
- [ ] Expose tail/spline outputs in review or visualization tooling.
- [ ] Avoid using spline outputs in bout/kinematics summaries until canary
  overlays and validity rates are reviewed.

## Open Questions

- Which centerline method should be first: skeleton longest path, distance
  transform ridge, contour-midpoint sampling, or another method?
- Should the first B-spline fit smooth the centerline, the body outline, or both?
- What validity thresholds should reject fragmented or branched body masks?
- Should the tail base be nearest-centerline projection to the caudal
  swim-bladder anchor, or the first centerline point caudal to that anchor?
- Should body arclength use raw centerline samples, the B-spline sample, or both
  with one marked preferred?
- When both pose `tail_tip` and spline `tail_tip_xy` exist, what distance should
  trigger `tail_endpoint_disagreement`?

## Related Documents

- [subject_shape_runs_contract.md](subject_shape_runs_contract.md)
- [body_frame_contract.md](body_frame_contract.md)
- [derived_analysis_run_contract.md](derived_analysis_run_contract.md)
- [derived_metrics_schema_contract.md](derived_metrics_schema_contract.md)
- [analytics_math_primer.md](analytics_math_primer.md)
- [current_pipeline_contract.md](current_pipeline_contract.md)
