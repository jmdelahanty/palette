# Body Frame Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-28
-->

Purpose: define a shared fish-relative coordinate-frame contract that can be
used by eye angles, body/tail shape, swim-bout metrics, and future
stimulus-response analyses without forcing every workflow to use the same input
modality.

## Design Decision

Palette should separate three concepts:

- **Semantic anchors**: labeled anatomical references such as `swim_bladder`,
  `eye_left`, `eye_right`, `subject_body`, `tail_tip`, or `snout_tip`.
- **Estimator**: the method that materializes frame geometry from those anchors,
  such as keypoints, mask component centroids, body-mask centerlines, or
  B-splines with keypoint polarity.
- **Body-frame outputs**: common arrays and conventions consumed by downstream
  analyses, regardless of estimator.

This keeps keypoint-only workflows valid while allowing mask/spline-derived
body frames to become the preferred source when they exist.

## Scope

This contract covers:

- what Palette means by a fish anatomical body frame
- where body-frame metadata and materialized arrays should live
- how keypoint, mask, and hybrid estimators should declare provenance
- how downstream analyses should reference a body frame

This contract does not replace:

- `pose_schema.metadata.heading_computation`
- `derived_metrics_schema`
- reviewed mask authority in `refined_subject_masks_runs`
- specialized downstream analyses such as `analysis/eye_angle_runs`

## Definition

Version 1 defines a two-dimensional fish anatomical frame:

- `origin`: an anatomical reference point used as the local frame origin.
- `forward_axis`: a unit vector pointing tail/body posterior toward head/anterior.
- `left_axis`: a unit vector perpendicular to `forward_axis`, pointing toward
  anatomical left.
- `heading_deg`: the scalar heading implied by `forward_axis`.
- `valid`: whether the frame was resolved for the row/sample.

The default semantic anchors for the current fish layout are:

- forward/polarity anchor: `swim_bladder -> midpoint(eye_left, eye_right)`
- left/right anchor: labeled `eye_left` and `eye_right`

These anchors may be measured by keypoints or by mask components. The contract
does not require masks.

## Coordinate Convention

Stored `*_xy` arrays use the same pixel coordinate space as their source rows:

- `x` increases rightward
- `y` increases downward
- coordinates must be isotropic pixels such as ROI or image pixels, not
  non-square normalized coordinates

`heading_deg` uses the existing keypoint heading convention:

```text
heading_deg = atan2(-forward_axis_y, forward_axis_x)
```

Implications:

- `0` degrees points toward image `+x`
- positive rotation is counter-clockwise in math coordinates
- the `y` component is negated when converting image vectors to angles

Fish-frame scalar coordinates should use:

```text
forward_coordinate = dot(vector_xy, forward_axis_xy)
left_coordinate    = dot(vector_xy, left_axis_xy)
```

where `vector_xy` is measured in the same image/ROI coordinate space as the
stored axes. This makes anatomical left positive without rotating or rewriting
the source image.

## Schema Placement

### Pose Schema

`pose_schema.metadata.heading_computation` remains the canonical keypoint
heading contract. It defines a keypoint-derived forward axis and heading scalar.

The pose schema may also expose stable semantic anchors when they are properties
of the skeleton itself. It should not own mask/spline estimator policy.

Use pose schema for:

- labeled keypoint semantics
- keypoint-derived heading computation
- keypoint dependency sets needed by editors/renderers

Do not use pose schema for:

- body-mask skeletonization method
- B-spline smoothing or knot policy
- mask-component fallback order
- temporal smoothing, filtering, or QC thresholds

### Subject Shape Runs

`analysis/subject_shape_runs/<run>/body_frame/` is the preferred materialized
home for body frames derived from refined subject masks, body centerlines,
B-splines, mask component centroids, or hybrid mask/keypoint estimators.

This placement is preferred because mask/spline body frames are deterministic
derived biology, not reviewed mask-pixel authority.

### Specialized Analysis Runs

Specialized analyses may compute or cache a local body frame when no reusable
subject-shape body frame exists. For example:

```text
analysis/eye_angle_runs/<run>/support/body_frame/
```

Such local caches must declare their estimator and source refs. They should be
treated as analysis-local support data, not as the canonical shared body-frame
surface.

### Future Dedicated Body-Frame Runs

A separate `analysis/body_frame_runs/<run>` family is not required now. It may
be justified later if multiple independent analyses need a shared body frame
that is not naturally part of a coherent `analysis/subject_shape_runs` product.

## Recommended Materialized Layout

For row-aligned runs:

```text
analysis/subject_shape_runs/<run>/
  attrs:
    body_frame_schema_id              "fish_anatomical_body_frame"
    body_frame_schema_version         1
    body_frame_estimator              "keypoint_head_axis" | "mask_component_axis" | "body_spline_with_anchor_polarity"
    body_frame_estimator_version      integer or stable string
    body_frame_coordinate_space       "roi_pixels" | "image_pixels"
    body_frame_angle_convention       "math_ccw_degrees_after_y_flip"
    body_frame_source_refs            dict of exact keypoint/mask/shape sources
  body_frame/
    origin_xy                         (N, 2)
    forward_axis_xy                   (N, 2)
    left_axis_xy                      (N, 2)
    heading_deg                       (N,)
    valid                             (N,)
    failure_reason_bytes              (N, width) optional uint8 utf8-null-terminated tags
    failure_reason                    (N,) optional compatibility string array
    midline_xy                        (N, P, 2) optional for spline/centerline estimators
    midline_valid                     (N,) optional
    arclength_px                      (N,) optional
```

Writers may omit optional arrays they cannot validate.

The body-frame origin is estimator-defined. For the current
`mask_component_axis` estimator, `origin_xy` is the eye-pair midpoint. That
origin is not the same as a rostral/nasal `snout_tip` landmark. Consumers that
need true snout-to-tail geometry should read the subject-shape snout/head/tail
fields, not infer the snout from `body_frame/origin_xy`.

## Recommended Metadata Shape

Run attrs should include a machine-readable contract payload when practical:

```json
{
  "schema_id": "fish_anatomical_body_frame",
  "schema_version": 1,
  "definition": {
    "origin": "estimator_defined_anatomical_origin",
    "forward_axis": "tail_or_body_posterior_to_head_or_anterior",
    "left_axis": "anatomical_left_perpendicular_to_forward",
    "heading_units": "degrees",
    "angle_convention": "math_ccw_degrees_after_y_flip"
  },
  "semantic_anchors": {
    "forward_from": "swim_bladder",
    "forward_to": ["eye_left", "eye_right"],
    "optional_rostral_anchor": "snout_tip",
    "left_reference": "eye_left",
    "right_reference": "eye_right"
  },
  "estimator": {
    "method": "keypoint_head_axis",
    "version": 1,
    "source_refs": {
      "refined_keypoints_run": "refined_keypoints_runs/..."
    }
  }
}
```

For a mask/spline estimator:

```json
{
  "estimator": {
    "method": "body_spline_with_anchor_polarity",
    "version": 1,
    "source_refs": {
      "subject_shape_run": "analysis/subject_shape_runs/...",
      "refined_subject_masks_run": "refined_subject_masks_runs/...",
      "polarity_keypoints_run": "refined_keypoints_runs/..."
    },
    "midline_source": "components/subject_body/bspline",
    "polarity_source": "swim_bladder_to_eye_midpoint",
    "left_right_source": "eye_left_eye_right_labels"
  }
}
```

## Estimator Families

### `keypoint_head_axis`

Uses keypoint labels to compute:

- origin: usually `midpoint(eye_left, eye_right)`
- forward axis: `swim_bladder -> midpoint(eye_left, eye_right)`
- left axis: perpendicular to forward, resolved by `eye_left`/`eye_right`

This is the keypoint-only fallback and should remain supported.

### `mask_component_axis`

Uses refined subject-mask component centroids or fitted component geometry:

- forward axis: `swim_bladder` component centroid toward eye-pair centroid
- left/right axis: labeled `eye_left` and `eye_right` components

This complements keypoints when component masks are available.

### `body_spline_with_anchor_polarity`

Uses a body mask centerline or B-spline for the midline geometry, then resolves
head/tail polarity and left/right direction from semantic anchors:

- midline: `subject_body` mask-derived centerline or B-spline
- polarity: keypoints or mask components identifying the head direction
- left/right: labeled eyes or another explicit anatomical side anchor

This is the expected preferred estimator for body/tail shape analyses once body
centerlines and splines are implemented.

## Validity

Recommended failure reasons:

- `missing_source_anchor`
- `missing_source_component`
- `degenerate_forward_axis`
- `ambiguous_polarity`
- `left_right_unresolved`
- `midline_fit_failed`
- `source_row_stale`

Downstream analyses must consult `valid` and reason arrays rather than
inferring all failure from NaN values.

## Relationship To Heading

`heading_computation` is the existing keypoint-derived scalar heading contract.
It is a valid body-frame estimator input, but it is not the whole body-frame
contract.

Near-term rule:

- if only keypoints are available, body-frame producers may use
  `pose_schema.metadata.heading_computation` as the estimator definition
- if mask/spline geometry is available and approved, subject-shape body-frame
  writers may prefer that geometry while preserving keypoint or mask-component
  anchors for polarity

## Relationship To `derived_metrics_schema`

`derived_metrics_schema` may say that a metric is measured in
`fish_anatomical_body_frame` coordinates, but it should not define or
materialize the body frame itself.

The body-frame source belongs in run attrs or a `body_frame/` support group.
Metric schemas should reference it.

## Migration Guidance

Short term:

- keep current keypoint heading semantics unchanged
- let `analysis/eye_angle_runs` declare any local body-frame support data it
  computes
- document `analysis/subject_shape_runs/<run>/body_frame/` as the future shared
  materialized home

Medium term:

- implement a reusable body-frame resolver that can consume, in order:
  1. `analysis/subject_shape_runs/<run>/body_frame`
  2. run-local `support/body_frame`
  3. keypoint heading metadata as a fallback estimator
- continue migrating eye-angle readers to schema v3 semantics, where axis
  orientation disambiguation is separate from biological convergence polarity
  and signed angles use the resolved body frame

Long term:

- prefer body-mask centerline/B-spline estimators for body/tail analyses
- keep semantic anchors such as `tail_tip` separate from source-specific
  measurements such as keypoint `tail_tip` or spline-derived `tail_tip_xy`
- keep keypoint-only body-frame production supported for datasets without masks
- materialize a dedicated `analysis/body_frame_runs` family only if reuse
  pressure justifies it

## Open Questions

- Which origin should be preferred for each downstream family: eye midpoint,
  swim bladder, body centroid, or spline arclength zero?
- Should `left_axis_xy` be stored explicitly, or always derived from
  `forward_axis_xy` plus side anchors?
- Should body-frame arrays be row-aligned to refined mask rows, keypoint rows,
  track samples, or all of the above through separate projection steps?
- What approval/quality threshold is required before a mask/spline-derived
  body frame supersedes a keypoint-derived fallback?

## Related Documents

- [body_spline_tail_anchor_design.md](body_spline_tail_anchor_design.md)
- [keypoint_heading_computation_contract.md](keypoint_heading_computation_contract.md)
- [subject_shape_runs_contract.md](subject_shape_runs_contract.md)
- [derived_analysis_run_contract.md](derived_analysis_run_contract.md)
- [derived_metrics_schema_contract.md](derived_metrics_schema_contract.md)
- [current_pipeline_contract.md](current_pipeline_contract.md)
- [src/fisheye/docs/eye_angle_conventions.md](../src/fisheye/docs/eye_angle_conventions.md)
