# Swim-Bladder Polar Boundary Design

<!-- design-meta
status: active
last_verified: 2026-04-01
-->

Purpose: define a new traditional swim-bladder proposal family for cases where
the current patch-local threshold/blob method is the wrong shape prior.

## Problem Statement

The current swim-bladder tuner and materializer use a dark-region model:

- crop a patch around the `swim_bladder` keypoint
- optionally darken edges with Sobel
- threshold the patch
- select the connected component nearest the keypoint

That is implemented today in:

- [swim_bladder_mask_tuner.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/swim_bladder_mask_tuner.py)
- [swim_bladder_segmentation.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/segmentation/swim_bladder_segmentation.py)

This works only when the swim bladder behaves like a coherent dark blob.

Observed canary failure mode:

- the swim bladder is often better described by a darker contrast ring or
  boundary than by a uniformly dark filled interior
- the interior shading is inconsistent
- Sobel-enhanced thresholding still does not reliably produce one enclosed
  filled region

So the main issue is not missing background subtraction. The issue is that the
current method is region-thresholding when the useful signal is boundary-like.

## Design Summary

Recommended new method family:

- `subject_method_family = "swim_bladder_polar_boundary_v1"`

Core idea:

- use the swim-bladder keypoint as an approximate center
- search radially around that center for a boundary signal
- fit a smooth closed shape from those per-angle boundary estimates
- rasterize the closed shape into the proposal mask

This is intentionally a different method family from the current threshold path
rather than an extension of it.

The current threshold path remains:

- `subject_method_family = "swim_bladder_patch_threshold_v1"`

## Why Polar Boundary Is A Better Fit

This method matches the observed anatomy/image signal better when:

- the structure is roughly round or oval
- the keypoint is near the center
- the visible signal is a partial or noisy ring
- the interior intensity is not a stable discriminator

Advantages over the current blob-threshold path:

- does not require the interior to be consistently dark
- can tolerate incomplete rings better
- uses the keypoint prior directly
- gives a natural contour-oriented representation that can later support
  contour QC

## Non-Goals

This design does not:

- require background subtraction
- assume ellipse-first geometry as the final refined representation
- replace the current threshold method for all archives immediately
- define a learned swim-bladder model

## Proposed Algorithm

### Inputs

- grayscale ROI patch centered on the swim-bladder keypoint
- approximate center from the keypoint
- patch-local method parameters

### Preprocessing

Optional preprocessing should be lightweight and local:

- Gaussian blur or median smoothing
- optional contrast normalization
- optional Sobel / gradient magnitude image

The method should operate primarily on an edge/boundary response image, not a
darkness-only image.

### Radial Search

For each angle in a fixed angular grid:

1. trace outward from the center
2. sample boundary response along the ray
3. choose the best candidate radius subject to:
   - minimum radius
   - maximum radius
   - local response threshold
   - optional inward/outward consistency constraints

If a ray has no valid hit:

- record it as missing
- later fill or interpolate it from neighboring valid rays if the missing span
  is small enough

### Shape Regularization

After radial hits are collected:

- smooth the radius profile across angle
- reject obvious outlier radii
- interpolate short missing spans
- require a minimum valid-ray fraction before accepting the proposal

### Rasterization

Convert the smoothed polar profile into:

- a closed polygon or contour in patch coordinates
- a filled binary mask for the proposal

Optional later:

- fit an ellipse only as a diagnostic summary, not as the primary mask model

## Saved Tuning Parameters

Recommended saved metadata shape:

```json
{
  "subject_mask_tuning": {
    "components": {
      "swim_bladder": {
        "method": "polar_boundary_center_seed",
        "subject_method_family": "swim_bladder_polar_boundary_v1",
        "version": "1.0",
        "tuned_timestamp": "...",
        "tuned_parameters": {
          "roi_padding": 18,
          "angle_step_degrees": 8,
          "min_radius_px": 3,
          "max_radius_px": 18,
          "smoothing_sigma": 1.5,
          "response_threshold": 0.12,
          "max_missing_gap_degrees": 40,
          "min_valid_ray_fraction": 0.55,
          "gradient_mode": "sobel_magnitude",
          "prefilter_sigma": 1.0
        },
        "context": {
          "storage_component_name": "swim_bladder"
        },
        "output_labels": ["swim_bladder"],
        "storage_component": "swim_bladder"
      }
    }
  }
}
```

Notes:

- these parameters intentionally describe a different proposal family from the
  current threshold/blob tuner
- the tuning entry should remain attr-only like the other mask tuners
- stage-level git/environment/platform provenance should still live on the
  materialized source run, not the tuning entry

## Tuner UX Changes

The current dedicated swim-bladder tuner remains the correct UI surface.

Recommended UI additions:

- show a `Boundary Response` panel in addition to the raw patch
- show a `Radial Hits` or `Polar Profile` debug visualization
- show the final closed contour over the patch
- preserve the current keypoint-centered ROI context panel

Recommended controls:

- keep ROI navigation and save behavior unchanged
- replace threshold/blob-specific sliders with boundary-specific sliders when
  this method family is active
- optionally allow switching between:
  - `threshold_blob`
  - `polar_boundary`

That would let the same tuner compare method families on the same ROI.

## Materializer Integration

The materializer should remain:

- [swim_bladder_segmentation.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/segmentation/swim_bladder_segmentation.py)

Required behavior:

- inspect `subject_mask_tuning.components["swim_bladder"]`
- dispatch by `subject_method_family`
- support both:
  - `swim_bladder_patch_threshold_v1`
  - `swim_bladder_polar_boundary_v1`

That keeps old tuning entries runnable while allowing the new method family to
land without destructive migration.

## Output And Provenance Policy

The output stage does not change:

- raw source run lives in `subject_mask_runs/<run>`

Expected attrs should still include:

- `method`
- `config`
- `tuning_source`
- `tuning_timestamp`
- `tuning_entry_snapshot`
- stage provenance

Recommended run semantics can remain:

- `traditional_swim_bladder_inference`

The method family difference should be reflected by:

- `method`
- `config`
- `tuning_entry_snapshot.subject_method_family`
- stage provenance parameters

## Acceptance Criteria For A Canary Prototype

Before treating this method family as the new default, validate that it:

1. improves proposal closure on canary ROIs where the threshold/blob method
   fails because the interior is not uniformly dark
2. remains centered on the swim-bladder keypoint instead of drifting to nearby
   edges
3. produces stable masks across neighboring ROIs in the same recording
4. writes a reproducible source run with auditable parameters

Recommended first comparison:

- tune and materialize both method families on the same canary archive
- compare proposal masks in the patch viewer or refined-subject workflow
- decide whether the new method should replace the threshold family for this
  recording class or remain opt-in

## Rollout Plan

### Phase 1

- keep the current threshold/blob path as-is
- add this design doc
- validate the image prior on a few canary ROIs manually

### Phase 2

- add a second preview path to `swim_bladder_mask_tuner.py`
- save tuning with `subject_method_family = "swim_bladder_polar_boundary_v1"`
- do not remove the threshold family

### Phase 3

- extend `swim_bladder_segmentation.py` to dispatch by saved method family
- materialize canary raw source runs with both methods for comparison

### Phase 4

- if the new method is consistently better, make it the recommended default for
  swim-bladder tuning on new canary archives

## Open Questions

1. Is the swim-bladder keypoint consistently near the center of the visible
   ring in the recordings that matter most?
2. Is the target shape usually star-convex enough that polar interpolation is
   safe?
3. Do we need anisotropic radius priors tied to fish heading, or is isotropic
   radial search sufficient?
4. Should the first implementation use raw gradient magnitude only, or blend
   darkness and gradient response?

## Related Docs

- [swim_bladder_patch_review_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_patch_review_design.md)
- [subject_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_tuning_workflow.md)
- [subject_mask_refinement_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_refinement_todo.md)
