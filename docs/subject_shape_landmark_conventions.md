# Subject Shape Landmark Conventions
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-05-01
-->

Purpose: explain the landmark conventions used by
`analysis/subject_shape_runs`, especially the difference between mask-local
anatomical anchors and derived body-centerline landmarks.

## Core Rule

Separate three things:

- anatomical meaning
- measurement source
- storage path

For example, `tail_tip` is an anatomical meaning. It can be measured by a
keypoint model, a manual keypoint edit, a body-mask centerline, or a future
B-spline. Those measurements are comparable, but they are not interchangeable
unless their estimator and coordinate conventions are known.

The same rule applies to the rostral/nasal end of the fish. A pose schema may
use a semantic label such as `snout_tip`, `nose_tip`, or `rostral_tip`; a
subject-shape run may estimate the same anatomical concept from a body-mask
contour or centerline. Those values should be comparable, but they should not
overwrite each other.

## Storage Ownership

`refined_subject_masks_runs/<run>` owns canonical refined mask pixels and
mask-local caches.

Examples:

- `masks_roi`
- `mask_bitpacked`
- `mask_rle`
- `components/<component>/contours/{ptr,len,points_xy}`
- component area, centroid, bbox, mask-present, and simple mask-local QC

`analysis/subject_shape_runs/<run>` owns interpreted biological geometry
derived from those masks.

Examples:

- body frame axes
- rostral/snout contour anchor
- caudal swim-bladder anchor
- body centerline
- tail base
- tail tip
- body and tail arclength
- future B-spline, curvature, and width profiles

Subject-shape runs should not duplicate full component contours by default.
Full component contours remain mask-local caches under
`refined_subject_masks_runs`.

## Body Frame

Current mask-derived body-frame convention:

- `forward_axis_xy` points posterior-to-anterior.
- The near-term estimator is swim-bladder centroid to eye-pair midpoint.
- `left_axis_xy` points anatomical left.
- Coordinates are in ROI pixel `x, y` space unless attrs state otherwise.

With this convention:

- anterior/headward points have larger projection on `forward_axis_xy`
- posterior/caudal/tailward points have smaller projection on
  `forward_axis_xy`

The body frame is required before words such as `caudal`, `anterior`, `left`,
and `right` have deterministic geometric meaning.

Important distinction:

- `body_frame/origin_xy` is the estimator-defined frame origin. In the current
  mask-component estimator, this is the eye-pair midpoint.
- The eye-pair midpoint is useful for polarity and heading, but it is not the
  most rostral/nasal endpoint of the animal.
- Do not use `body_frame/origin_xy` as body length's anterior endpoint unless a
  run explicitly declares that convention.

## Rostral/Snout Tip

Storage:

```text
analysis/subject_shape_runs/<run>/components/subject_body/snout_tip_xy
```

Recommended companion fields:

- `components/subject_body/snout_tip_valid`
- `components/subject_body/snout_tip_failure_reason_bytes`
- `components/subject_body/snout_tip_estimator` attr, or an equivalent run attr

Meaning:

The most rostral/anterior anatomical point of the fish. This is the nose/snout
tip in overhead imagery, not the eye midpoint.

Recommended first mask-derived estimator:

```text
subject_body contour
  -> project contour points onto body-frame forward axis
  -> choose the point with maximum forward-axis projection
  -> store as snout_tip_xy
```

Pose/keypoint schemas may also contain a semantic `snout_tip` or equivalent
marker. That keypoint-derived value should remain in the pose/keypoint run.
Subject-shape may optionally compare the pose value to mask-derived
`snout_tip_xy`, but it should preserve both source measurements.

Related failure reasons:

- `missing_subject_body_mask`
- `missing_body_frame`
- `missing_subject_body_contour`
- `rostral_projection_failed`
- `snout_tip_outside_body_mask`
- `pose_snout_tip_missing`
- `pose_snout_tip_disagreement`

## Caudal Swim-Bladder Point

Storage:

```text
analysis/subject_shape_runs/<run>/components/swim_bladder/caudal_contour_point_xy
```

Meaning:

The point on the swim-bladder contour furthest toward the tail.

Current estimator:

1. take the refined `swim_bladder` mask
2. extract its contour
3. project each contour point onto the body-frame forward axis
4. choose the point with the minimum forward-axis projection

This point lies on the swim-bladder boundary. It is not the tail base and it is
not the tail tip. It is an internal anatomical anchor used to define where the
tail segment begins.

Related fields:

- `components/swim_bladder/caudal_contour_projection_px`
- `components/swim_bladder/caudal_contour_valid`
- `components/swim_bladder/caudal_contour_failure_reason_bytes`

## Body Centerline

Storage:

```text
analysis/subject_shape_runs/<run>/components/subject_body/centerline_xy
```

Meaning:

An ordered estimate of the fish body midline in ROI pixels.

Current estimator:

```text
subject_body mask
  -> skeletonize
  -> choose longest endpoint-to-endpoint skeleton path
  -> orient with body-frame polarity
  -> resample to a fixed point count
```

The current centerline is a skeleton-derived sampled polyline, not yet a
canonical B-spline. Future spline outputs should live in the same
`analysis/subject_shape_runs` family with their own method/provenance attrs.

Related fields:

- `components/subject_body/centerline_valid`
- `components/subject_body/centerline_failure_reason_bytes`
- `components/subject_body/body_arclength_px`
- `components/subject_body/head_endpoint_to_snout_distance_px`
- `components/subject_body/centerline_reaches_snout`
- `components/subject_body/centerline_snout_check_reason_bytes`

Older `centerline_xy` anterior endpoints may stop near the eye region if the
skeleton-derived path does not include the rostral/nose contour. That is a
centerline/skeleton estimator limitation, not evidence that the fish's
anatomical snout is at the eye midpoint.

Schema v3+ centerline/B-spline writers use the first approach: they extend the
body midline to a validated `snout_tip_xy` before resampling. The current
writer uses a bounded mask path rather than a strict straight-line bridge, and
it joins into a body-frame-guided medial head point rather than blindly using
the first skeleton endpoint. This prevents normal curved/rounded head masks
from being rejected because the straight chord leaves the body, and prevents
head-side skeleton branches from pulling the spline into off-axis mask
offshoots. If no bounded mask path can be found, the row is marked invalid with
a reason such as `snout_extension_no_mask_path`, `snout_extension_too_long`, or
`snout_extension_path_too_indirect` instead of writing a legacy head endpoint.

Schema v2/v5 runs used the second approach. They left `head_endpoint_xy` as the
anterior endpoint of the selected centerline/spline estimator, then wrote the
distance and validity check between that endpoint and `snout_tip_xy`.

## Head Endpoint

Storage:

```text
analysis/subject_shape_runs/<run>/components/subject_body/head_endpoint_xy
```

Meaning:

For schema v3+ subject-shape runs, the validated snout-anchored anterior
endpoint of the centerline/spline. When `centerline_valid = true`,
`head_endpoint_xy` should match `snout_tip_xy` within numerical tolerance.

For schema v2/v5 subject-shape runs, the anterior endpoint of the older
skeleton-derived centerline/spline estimator. It may not match `snout_tip_xy`.

Important distinction:

- `head_endpoint_xy` is source-specific geometry.
- `snout_tip_xy` is the preferred semantic rostral/nasal landmark when present.
- `body_frame/origin_xy` may be the eye midpoint or another estimator-defined
  origin.

These three points can be close, but they are not the same contract.

## Tail Base

Storage:

```text
analysis/subject_shape_runs/<run>/components/subject_body/tail_base_xy
```

Meaning:

The point on the body centerline associated with the caudal swim-bladder anchor.
This is the start of the tail segment for centerline/spline-based tail
measurements.

Current estimator:

1. compute `caudal_contour_point_xy` on the swim-bladder contour
2. project that point onto the oriented body centerline
3. store the projected centerline point as `tail_base_xy`

This point lies on the body centerline, not on the swim-bladder contour.

Use `tail_base_xy` for:

- tail-segment length
- tail-normalized sampling
- tail curvature
- future tail width profiles

Related fields:

- `components/subject_body/tail_base_valid`
- `components/subject_body/tail_base_arclength_px`
- `components/subject_body/tail_base_failure_reason_bytes`

## Tail Tip

Storage:

```text
analysis/subject_shape_runs/<run>/components/subject_body/tail_tip_xy
```

Meaning:

The posterior endpoint of the oriented body centerline or future B-spline.

Current estimator:

The tailward endpoint of the oriented skeleton-derived centerline path.

Important distinction:

- `tail_tip` in a pose schema is a semantic keypoint label.
- `tail_tip_xy` in `analysis/subject_shape_runs` is a mask/centerline-derived
  measurement of that semantic landmark.
- These should not overwrite each other.

Future analyses may compare keypoint `tail_tip` and shape-derived
`tail_tip_xy`, but they should preserve both source measurements.

## Relationship Between The Landmarks

The current derived tail geometry is:

```text
swim_bladder mask contour
  -> caudal_contour_point_xy          # point on swim-bladder boundary
  -> projection onto body centerline
  -> tail_base_xy                     # point on body centerline
  -> centerline from tail_base to tail_tip
  -> tail_segment_arclength_px
```

The body centerline also has:

```text
head_endpoint_xy -> ... -> tail_base_xy -> ... -> tail_tip_xy
```

So:

- caudal swim-bladder point: source anchor on the organ boundary
- snout tip: semantic rostral/nasal endpoint, preferably from body contour or
  pose/keypoint `snout_tip`
- head endpoint: current anterior endpoint of the centerline/spline estimator
- tail base: derived point on the body midline
- tail tip: posterior endpoint of the body midline/spline
- centerline: ordered body-axis representation connecting head and tail

## Schema Versioning Guidance

Documenting `snout_tip` did not change older archives. Older
`analysis/subject_shape_runs` with `schema_version = 1` may lack all
snout/rostral arrays.

Current materialized writer:

- `schema_version = 3`
- `method = "subject_shape_from_refined_masks_v8"`
- `method_version = 8`
- `snout_tip_estimator = "subject_body_contour_max_forward_projection_v1"`
- `centerline_method = "snout_anchored_skeleton_longest_endpoint_path_v1"`
- `centerline_skeleton_method = "skeleton_longest_endpoint_path_v1"`
- `centerline_snout_extension_method = "prepend_mask_path_to_body_frame_guided_join_v1"`
- `centerline_snout_join_method = "body_frame_lateral_min_head_region_v1"`
- `head_endpoint_semantics = "validated_snout_tip"`
- `centerline_snout_check_method = "head_endpoint_to_snout_distance_v1"`

The current writer materializes `snout_tip_xy`, `snout_tip_valid`, and
`snout_tip_failure_reason_bytes` as first-class subject-shape outputs. It also
materializes `head_endpoint_to_snout_distance_px`,
`centerline_reaches_snout`, and `centerline_snout_check_reason_bytes` as a
schema-level invariant check. For valid v8 centerlines the distance should be
approximately zero. Pose/keypoint `snout_tip` values should still remain in the
pose/keypoint run; record comparisons separately instead of copying them into
mask-derived fields.

## Crimson Visualization Guidance

For mask overlays:

- draw mask fills from the `refined_subject_masks_runs` logical mask store
  (`masks_roi` when dense is present, otherwise compact `mask_bitpacked` or
  `mask_rle` through `MaskStore`)
- resolve channels by `mask_labels`
- use persisted component contours when available
- fall back to client-side contours when persisted contours are missing

For subject-shape overlays:

- draw `body_frame/origin_xy` only as the frame origin/eye-midpoint origin when
  useful
- draw `snout_tip_xy` separately when present
- draw `head_endpoint_xy` separately from `snout_tip_xy` for schema v2/v5 runs;
  for schema v3+ runs they should overlap when the centerline is valid
- draw `centerline_xy` as the body midline
- draw `caudal_contour_point_xy` as the swim-bladder caudal anchor
- draw `tail_base_xy` as the start of the tail segment
- draw `tail_tip_xy` as the posterior endpoint

Recommended visual distinction:

- snout tip: marker at rostral/nose contour endpoint
- head endpoint: marker at the declared centerline/spline anterior endpoint;
  in schema v3+ this is the validated snout endpoint
- caudal swim-bladder point: marker on swim bladder, for source-anchor review
- tail base: marker on centerline, for tail-segment start
- tail tip: marker at posterior centerline endpoint
- centerline: line through body, preferably head-to-tail oriented

Crimson should not infer tail geometry from contours alone when a valid
subject-shape run provides these derived arrays.

## Failure Handling

Each derived landmark should be read with its validity field when available.

Examples:

- `caudal_contour_valid`
- `tail_base_valid`
- `centerline_valid`

If a validity flag is false, display the geometry as missing or failed and
surface the corresponding reason bytes if useful. Do not silently invent a
landmark from a nearby contour point.

## Related Documents

- [subject_shape_runs_contract.md](subject_shape_runs_contract.md)
- [body_spline_tail_anchor_design.md](body_spline_tail_anchor_design.md)
- [body_frame_contract.md](body_frame_contract.md)
- [refined_subject_mask_geometry_cache_and_propagation_design.md](refined_subject_mask_geometry_cache_and_propagation_design.md)
