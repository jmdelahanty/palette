# Swim Bladder Patch Review Design
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-03-12
-->

Purpose: define a patch-local review and tuning workflow for swim-bladder masks,
using the existing swim-bladder keypoint as the spatial anchor.

## Why This Exists

Current refined subject-mask review in
[refined_subject_mask_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/refined_subject_mask_review.py)
is full-ROI and component-generic. That is the right canonical edit surface,
but it is not ideal for rapid, consistent swim-bladder labeling.

Swim-bladder labeling has a different operator task than eye review:

- one small anatomical target,
- one known anchor keypoint (`swim_bladder`),
- a need for tight local consistency across many ROIs,
- and likely different geometry summaries than eyes.

This motivates a patch-local workflow similar in spirit to
[visualize_eye_mask_patches.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/visualization/visualize_eye_mask_patches.py),
but not a literal extension of the eye-specific UI.

## Design Decision

Recommended approach:

1. add a sibling swim-bladder patch reviewer,
2. add a sibling batch wrapper for launching it,
3. keep writes canonical by saving into `refined_subject_masks_runs`,
4. add a local swim-bladder tuner for a reproducible traditional auto-proposal
   path.

Not recommended:

- forcing swim bladder into the eye patch reviewer UI,
- encoding swim bladder as a fake eye channel,
- or treating patch review as a replacement for
  `refined_subject_mask_review.py`.

## Relationship To Existing Stages

Near-term intended path:

```text
subject_mask_runs/<run>
  -> refined_subject_masks_runs/<run>
  -> visualize_swim_bladder_mask_patches.py   # localized review/edit helper
```

Optional traditional local-proposal path later:

```text
swim_bladder_mask_tuner.py
  -> analysis_metadata.attrs["subject_mask_tuning"].components["swim_bladder"]
  -> traditional swim-bladder segmentation/materialization
  -> refined_subject_masks_runs/<run>
```

## Proposed New Tools

### 1. `fisheye.visualization.visualize_swim_bladder_mask_patches`

Purpose:

- show a small keypoint-centered patch around the swim bladder,
- let the operator rapidly inspect/edit swim-bladder masks,
- and write edits back into `refined_subject_masks_runs/components/swim_bladder`.

Expected inputs:

- Zarr archive path
- `--refined-run <run>` or `--subject-run <run>` + auto-create refined run
- optional `--crop-run <run>`
- optional `--padding`, `--scale-percent`, `--brush`, `--frames`

Primary source arrays:

- `crop_runs/<run>/roi_images`
- `refined_keypoints_runs/<run>/keypoints_roi`
- `subject_mask_runs/<run>/masks_roi` or
  `refined_subject_masks_runs/<run>/masks_roi`

Required keypoint:

- `swim_bladder`

### 2. `fisheye.utils.review_swim_bladder_masks_batch`

Purpose:

- batch wrapper across many recordings, analogous to
  [review_eye_masks_batch.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/review_eye_masks_batch.py)

Expected filters:

- `--recursive`
- `--registry`
- `--zarr-use`
- `--subject-run`
- `--refined-run`
- `--review-state-filter`

### 3. `fisheye.tune.swim_bladder_mask_tuner`

Purpose:

- tune a traditional local swim-bladder proposal method centered on the
  `swim_bladder` keypoint
- save reusable parameters into
  `analysis_metadata.attrs["subject_mask_tuning"].components["swim_bladder"]`

Current status:

- implemented in
  [swim_bladder_mask_tuner.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/swim_bladder_mask_tuner.py)
- stores patch-local threshold/Sobel/morphology parameters under
  `subject_mask_tuning.components["swim_bladder"]`
- now has a traditional materializer in
  [swim_bladder_segmentation.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/segmentation/swim_bladder_segmentation.py)
  that converts the saved tuning into raw `subject_mask_runs`
- current method family is still threshold/blob-oriented
- a boundary-oriented successor is proposed in
  [swim_bladder_polar_boundary_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_polar_boundary_design.md)

## Patch Reviewer UI Contract

Recommended panels:

1. `ROI Crop`
   - full ROI grayscale image
   - with swim-bladder center marker

2. `Swim Bladder Patch`
   - local crop around the swim-bladder keypoint

3. `Stored Mask Patch`
   - current refined swim-bladder mask in patch coordinates

4. `Patch Overlay`
   - grayscale patch + current mask overlay

5. `ROI Overlay`
   - full ROI with swim-bladder mask overlay for context

Optional later:

6. `Proposal Patch`
   - local traditional/model proposal preview

Recommended controls:

- `n` / `p`: next / previous ROI
- `j` / `k`: jump `-10` / `+10`
- `[` / `]`: brush radius
- `LMB`: paint
- `RMB`: erase
- `s`: save current ROI
- `r`: reset current ROI to stored refined mask
- `a`: approve
- `N`: needs review
- `R`: reject
- `P`: pending
- `q`: quit

## Storage Contract

The patch reviewer should not create a separate swim-bladder-only runtime stage.
It should write to the canonical refined subject-mask stage:

- `refined_subject_masks_runs/<run>/masks_roi[:, swim_bladder_channel, :, :]`
- `components/swim_bladder/reason_bytes`
- `components/swim_bladder/mask_present`
- `components/swim_bladder/area_px`
- `components/swim_bladder/edit_applied`

Expected attrs to update:

- `component_review_statuses["swim_bladder"]`
- parent `refined_subject_mask_review_status` when appropriate
- optional `component_summary_statistics["swim_bladder"]`

## Geometry Scope

Unlike eyes, swim bladder should not assume ellipse-first geometry.

Recommended near-term geometry:

- area
- centroid
- bounding box
- contour availability flag

Possible later geometry:

- contour array / contour table
- major/minor axis
- circularity
- local intensity moments

These should remain component-specific and live under:

```text
refined_subject_masks_runs/<run>/components/swim_bladder/geometry/
```

when introduced.

## Tuner Design

The traditional local swim-bladder tuner is patch-local and keypoint-centered,
not ROI-wide.

Current note:

- the implemented saved metadata shape below describes the current
  `swim_bladder_patch_threshold_v1` family
- a new `swim_bladder_polar_boundary_v1` family is now proposed separately in
  [swim_bladder_polar_boundary_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_polar_boundary_design.md)

Recommended saved metadata shape:

```json
{
  "subject_mask_tuning": {
    "components": {
      "swim_bladder": {
        "method": "global_threshold_otsu",
        "subject_method_family": "swim_bladder_patch_threshold_v1",
        "version": "1.0",
        "tuned_timestamp": "...",
        "tuned_parameters": {
          "roi_padding": 18,
          "pre_threshold": 64,
          "sobel_strength": 0.10,
          "min_area": 12,
          "max_area": 180,
          "min_circularity": 0.35,
          "opening_radius": 1,
          "closing_radius": 1
        },
        "context": {
          "storage_component_name": "swim_bladder"
        }
      }
    }
  }
}
```

This should remain under the subject-mask tuning namespace rather than creating
a separate top-level `swim_bladder_mask_tuning` blob.

## Batch Review Policy

The batch wrapper should support:

- listing candidate archives without opening UI
- scoping by `training` vs `analysis`
- filtering by component review state
- operating only on runs where `swim_bladder` is available

Registry/query integration is desirable later, but the first implementation can
follow the filesystem-driven pattern already used by
`review_eye_masks_batch.py`.

## Recommended Build Order

1. `visualize_swim_bladder_mask_patches.py`
2. `review_swim_bladder_masks_batch.py`
3. `swim_bladder_mask_tuner.py`
4. `swim_bladder_segmentation.py`
5. shared helper extraction if eye/swim-bladder patch tools converge

## Deferrals

Explicitly deferred for now:

- a fully generic `visualize_component_mask_patches.py`
- registry surfacing for swim-bladder patch review status
- automatic swim-bladder geometry beyond area/centroid-level summaries
- any attempt to unify swim bladder into the eye patch UI
