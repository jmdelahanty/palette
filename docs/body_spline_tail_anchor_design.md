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

### Tail Sampling, Width, And Curvature

Tail curvature and width should be derived from a smoothed, ordered midline or
B-spline rather than directly from raw skeleton pixels.

Raw skeleton pixels are useful for discovering the centerline, but they are
jagged and branch-prone. Curvature is a derivative-like measurement, so it is
especially sensitive to pixel noise. The preferred workflow is:

```text
subject_body mask
  -> skeleton or medial-axis candidate
  -> cleaned ordered midline
  -> oriented midline with head/tail polarity
  -> B-spline or smoothed sampled centerline
  -> fixed arclength samples
  -> tangent, normal, curvature, and width profile
```

Tail sampling should use normalized arclength along the tail segment for the
first implementation:

```text
tail_sample_s = [0.0, ..., 1.0]
```

where `0.0` is `tail_base_xy` and `1.0` is `tail_tip_xy`. This makes tail
profiles comparable across frames even when body length or fitted sample counts
change.

At each tail sample:

1. evaluate the spline or smoothed centerline position
2. compute the local tangent direction
3. compute the perpendicular normal direction
4. intersect or probe the subject-body mask along the normal
5. store the measured mask width and validity

The same sampled curve should provide curvature. For B-splines, curvature should
prefer spline derivatives. For sampled centerlines without analytic
derivatives, curvature should use a documented finite-difference method.

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
    tail_sampling_schema_id         "analysis.subject_shape.tail_sampling"
    tail_sampling_schema_version    1
    tail_sample_domain              "tail_segment_normalized_arclength"
    tail_sample_count               integer
    curvature_method                "bspline_derivative" | "finite_difference"
    width_profile_method            "normal_mask_intersection"

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
      tail_sample_s
      tail_sample_xy
      tail_tangent_xy
      tail_normal_xy
      tail_tangent_angle_deg
      tail_curvature_px_inv
      tail_width_px
      tail_width_endpoints_xy
      tail_width_valid
      tail_width_failure_reason_bytes
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

### Stage 3: Tail Sampling, Width, And Curvature

Inputs:

- oriented centerline or B-spline
- `tail_base_xy`
- `tail_tip_xy`
- refined subject-body mask

Steps:

1. Define `tail_sample_s` as fixed normalized arclength positions from tail base
   to tail tip.
2. Evaluate `tail_sample_xy` at each position.
3. Compute tangent and normal vectors at each sample.
4. Compute `tail_tangent_angle_deg` using the body-frame angle convention.
5. Compute `tail_curvature_px_inv` from spline derivatives or documented
   finite differences.
6. Probe or intersect the subject-body mask along each normal vector.
7. Store `tail_width_px`, optional width endpoints, and per-sample validity.
8. Record all sampling, smoothing, derivative, and width-probe parameters in
   attrs.

Expected failure reasons:

- `missing_tail_base`
- `missing_tail_tip`
- `tail_segment_too_short`
- `tail_sample_outside_mask`
- `tail_width_intersection_failed`
- `tail_width_multiple_intersections`
- `tail_curvature_failed`
- `tail_spline_derivative_failed`

## Visualization Requirement

The canary visualization should overlay:

- subject-body mask outline
- swim-bladder contour
- caudal swim-bladder contour point
- body-frame forward axis
- centerline or B-spline samples
- tail sampling points
- tail normal width probes
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
- [x] Define tail-normalized sampling for width and curvature profiles.
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
- [ ] Store tail-sampling parameters: sample count, domain, smoothing source,
  curvature method, width-probe method, probe extent, and probe resolution.
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
- [ ] Add tail-normalized fixed arclength sampling.
- [ ] Compute spline/sample tangents and normals.
- [ ] Compute tail curvature from spline derivatives or finite differences.
- [ ] Compute tail width profiles from normal-line mask intersections.
- [ ] Keep outputs row-aligned to refined subject-mask rows for the first pass.
- [ ] Do not mutate refined mask pixels or refined keypoint arrays.

### Validation

- [ ] Add in-memory unit tests for projection and caudal-anchor selection.
- [ ] Add in-memory unit tests for semantic source separation between pose
  `tail_tip` and spline `tail_tip_xy`.
- [ ] Add fixture tests for failure reasons on missing/ambiguous masks.
- [ ] Add deterministic tests for tail-normalized sample positions.
- [ ] Add tests for width-profile probing on synthetic masks with known width.
- [ ] Add tests that curvature fails closed when derivatives are unavailable or
  degenerate.
- [ ] Run a canary on the feeding recording.
- [ ] Compare spline `tail_tip_xy` against pose `tail_tip` when available.
- [ ] Inspect persisted overlay PNGs before using outputs downstream.

### Downstream Consumers

- [ ] Teach body/tail shape consumers to prefer subject-shape spline tail
  geometry when valid.
- [ ] Keep keypoint-only workflows using pose-schema `tail_tip`.
- [ ] Expose tail/spline outputs in review or visualization tooling.
- [ ] Add tail-width and curvature overlays to the subject-shape canary viewer.
- [ ] Avoid treating tail-width/curvature profiles as behavior metrics until
  temporal alignment and smoothing policy are explicitly chosen.
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
- What default `tail_sample_count` is enough for curvature and width profiles:
  10, 20, or a length-dependent value?
- Should width profiles probe the full subject-body mask or only the caudal mask
  segment posterior to the swim-bladder anchor?
- Should curvature be stored in pixel inverse units only, or also calibrated
  physical inverse units when calibration is available?

## Related Documents

- [subject_shape_runs_contract.md](subject_shape_runs_contract.md)
- [body_frame_contract.md](body_frame_contract.md)
- [derived_analysis_run_contract.md](derived_analysis_run_contract.md)
- [derived_metrics_schema_contract.md](derived_metrics_schema_contract.md)
- [analytics_math_primer.md](analytics_math_primer.md)
- [current_pipeline_contract.md](current_pipeline_contract.md)
