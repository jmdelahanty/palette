# Body Frame Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-07-19
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
- `axis_valid`: whether the frame was resolved for the row/sample.

`heading_deg` is a derived scalar presentation/analysis surface, not part of
the canonical version-1 body-frame geometry record. Consumers may derive it
from `forward_axis_xy` using the fixed angle convention below. If a producer
also persists a heading array, that array requires its own row-bound contract;
its presence does not establish body-frame authority.

### Canonical version-1 derivation authority

A canonical body-frame record is not established by plausible orthonormal
arrays alone. The frame must bind the exact source coordinate payload, its
canonical row identity, the labeled source schema, method-specific validity or
polarity arrays, the controlled estimator record, and the owning producer
manifest. Every payload is path-, dtype-, shape-, and content-digest-bound in
the same archive.

For `keypoint_head_axis` and `mask_component_axis`, version 1 uses exactly:

```text
origin  = (eye_left + eye_right) / 2
forward = normalize(origin - swim_bladder)
left    = the source-axis-aware perpendicular to forward
```

The row is valid only when all three labeled anchors are valid and finite, the
forward vector is non-degenerate, and
`dot(eye_left - eye_right, left) > 0`. The last check resolves the perpendicular
using labeled anatomical side; a numerically orthonormal but oppositely labeled
axis is invalid.

For `body_spline_with_anchor_polarity`, version 1 chooses the spline endpoint
nearest the eye midpoint as anterior and the farther endpoint as posterior,
then uses `normalize(anterior - posterior)`. Equal endpoint distances,
non-finite inputs, inconsistent eye-side polarity, or a forward axis that does
not point from the swim-bladder anchor toward the eye midpoint make the row
invalid. A different spline estimator requires a new controlled formula ID or
schema version.

For every method, invalid rows use `axis_valid = false` and all-NaN origin and
axis geometry. Writers must rederive these values from the bound inputs; they
must not certify caller-provided geometry from norm and orthogonality checks
alone.

The body-frame row identity is the exact identity of its source coordinate
rowset. An observation source uses `observation_instance/instance_key`; a
track-sample source uses `track_sample/track_sample_key`. Neither key is a
biological `subject_id`, and a same-length identity array is not a substitute
for the bound source identity.

The default semantic anchors for the current fish layout are:

- forward/polarity anchor: `swim_bladder -> midpoint(eye_left, eye_right)`
- left/right anchor: labeled `eye_left` and `eye_right`

These anchors may be measured by keypoints or by mask components. The contract
does not require masks.

## Coordinate Convention

Stored `*_xy` arrays use the exact coordinate profile of their bound source
descriptor. Canonical version 1 accepts only:

- `source_camera_image_px.top_left_y_down.v1`
- `physical_mm.source_camera_y_down.v1`

Both profiles use source-camera `+x` right and `+y` down. ROI-local,
model-input, normalized, arena-relative, canvas, projector, and generic
`image_pixels` inputs are not accepted by version 1, even when their numeric
ranges or axes look compatible. A future profile may admit one only through a
typed, direction-labelled transform lineage and a new controlled contract.

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

where `vector_xy` is measured in the same bound source-camera-pixel or
source-camera-physical frame as the stored axes. This makes anatomical left
positive without rotating or rewriting the source image.

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
    fish_anatomical_body_frame         sealed typed frame record
    fish_anatomical_body_frame_sha256  digest of the exact frame record
  body_frame/
    origin_xy                         (N, 2)
    forward_axis_xy                   (N, 2)
    left_axis_xy                      (N, 2)
    axis_valid                        (N,)
```

Those four arrays are mandatory and content-digest-bound by the frame record;
invalid rows have `axis_valid == false` and all-NaN values in all three geometry
arrays. Heading, failure reasons, midlines, and arclengths may be persisted as
separate derived surfaces, but they are not optional members of the canonical
body-frame geometry record and do not inherit its authority implicitly.

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

Downstream analyses must consult `axis_valid` rather than infer validity only
from NaN values. Failure-reason arrays, when present under a separate derived
contract, may refine diagnostics but cannot override `axis_valid`.

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

- A future schema may add origins other than the version-1 eye-pair midpoint;
  it must use a new controlled origin/formula contract.
- `left_axis_xy` is mandatory in version 1 and is rederived from exact labeled
  source anchors; changing that materialization requires a new schema version.
- Version 1 may bind either an observation `instance_key` rowset or a
  `track_sample_key` rowset. Cross-rowset projection is a separate explicit
  operation and cannot be inferred from equal row counts.
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
