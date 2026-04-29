# Subject Shape Landmark Conventions
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-29
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

## Storage Ownership

`refined_subject_masks_runs/<run>` owns canonical refined mask pixels and
mask-local caches.

Examples:

- `masks_roi`
- `components/<component>/contours/{ptr,len,points_xy}`
- component area, centroid, bbox, mask-present, and simple mask-local QC

`analysis/subject_shape_runs/<run>` owns interpreted biological geometry
derived from those masks.

Examples:

- body frame axes
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
- future tail curvature
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
- tail base: derived point on the body midline
- tail tip: posterior endpoint of the body midline/spline
- centerline: ordered body-axis representation connecting head and tail

## Crimson Visualization Guidance

For mask overlays:

- draw `masks_roi` fills from `refined_subject_masks_runs`
- resolve channels by `mask_labels`
- use persisted component contours when available
- fall back to client-side contours when persisted contours are missing

For subject-shape overlays:

- draw `centerline_xy` as the body midline
- draw `caudal_contour_point_xy` as the swim-bladder caudal anchor
- draw `tail_base_xy` as the start of the tail segment
- draw `tail_tip_xy` as the posterior endpoint

Recommended visual distinction:

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
