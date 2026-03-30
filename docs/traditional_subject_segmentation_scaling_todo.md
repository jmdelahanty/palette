# Traditional Subject Segmentation Scaling TODO

## Status

Deferred.

The current traditional `subject_body` path is intentionally simple and
single-process:

- tuner:
  - [subject_mask_tuner.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/subject_mask_tuner.py)
- writer/materializer:
  - [subject_segmentation.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/segmentation/subject_segmentation.py)

That is acceptable for canary-scale training archives. On the current canary
training zarr, runtime is only a few seconds, so additional orchestration would
add complexity without immediate benefit.

## Current Operational Workflow

The current canary operator path is:

1. tune `subject_body` in
   [subject_mask_tuner.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/subject_mask_tuner.py)
2. optionally propagate `subject_mask_tuning` by `camera_id` with
   `apply_tuning_by_camera.py`
3. materialize a raw body-only run with
   [subject_segmentation.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/segmentation/subject_segmentation.py)
4. refine/edit masks in
   [refined_subject_mask_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/refined_subject_mask_review.py)

That workflow is documented in:

- [subject_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_tuning_workflow.md)

## Why This Exists

We will eventually want the same operational properties that other traditional
Palette stages already have:

- chunked processing
- optional Dask schedulers
- clearer progress reporting on large archives
- better behavior when processing many ROIs or full recordings

Existing precedents:

- traditional detect uses chunked Dask processing:
  - [detect_traditional.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/detection/detect_traditional.py)
- traditional eye masks use chunked ROI processing with Dask:
  - [eye_segmentation.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/segmentation/eye_segmentation.py)

`subject_segmentation.py` does not yet follow that pattern.

## Current Implementation Boundary

Today the traditional subject-body writer:

- reads one materialized crop run
- reads one background run
- optionally applies the saved dish mask
- loops over ROIs in-process
- writes one body-only `subject_mask_runs/<run>`

This is correct for now, but not the long-term scaling design.

## Deferred Goal

Add a Dask-capable chunked execution path for traditional subject segmentation
without changing the core segmentation algorithm.

That means:

- keep the same tuned parameters
- keep the same ROI-local background subtraction logic
- keep the same dish-mask gating behavior
- change only execution strategy and write orchestration

## Desired End State

The future traditional subject segmentation path should support:

- `scheduler = threads | processes | distributed | single-threaded`
- chunked ROI execution
- optional worker count controls
- progress reporting similar to traditional eye segmentation
- deterministic run outputs regardless of scheduler

## Proposed Refactor Shape

### 1. Separate per-ROI computation from writeback

Extract a pure worker helper that takes:

- ROI image chunk
- ROI coordinates
- background source info
- dish-mask projection inputs
- tuned parameters

and returns per-row results such as:

- binary body mask
- probability-like diff image
- Otsu threshold
- component counts
- area/bbox summaries

### 2. Add ROI chunking

Use chunked row batches rather than one ROI at a time.

Suggested starting point:

- `chunk_size = max(64, min(1024, ceil(n_rois / workers)))`

This should mirror the operational style already used in
[eye_segmentation.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/segmentation/eye_segmentation.py).

### 3. Keep writeback centralized

Do not let workers write directly into the zarr run on the first refactor.

Preferred first step:

- workers compute chunk results
- main process aggregates
- main process writes `subject_mask_runs/<run>`

That keeps correctness and debugging simpler.

If needed later, direct chunk writes can be evaluated separately.

### 4. Preserve canary-friendly single-process mode

Do not remove the simple path.

Even after adding Dask support, `single-threaded` should remain an explicit and
well-tested option for:

- debugging
- small training zarrs
- sandbox-friendly execution

## Scope Boundaries

This deferred work is about execution scaling only.

It is not for:

- changing the traditional subject-body algorithm
- adding new anatomy channels
- adding `refined_subject_masks_runs`
- switching to SAM or model-based segmentation
- redesigning the registry

## Success Criteria

When this TODO is revisited, the refactor should be considered successful only
if all of the following are true:

- outputs match the current single-process implementation within expected
  numerical tolerance
- dish-mask gating is preserved exactly
- tuned parameter loading behavior is unchanged
- progress/output ergonomics remain clear
- large ROI sets run faster or scale better than the current loop

## Deferred Checklist

- [ ] Extract a pure per-chunk traditional subject segmentation worker.
- [ ] Add scheduler and worker-count options to
      [subject_segmentation.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/segmentation/subject_segmentation.py).
- [ ] Add chunked Dask execution modeled after
      [eye_segmentation.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/segmentation/eye_segmentation.py).
- [ ] Keep a tested `single-threaded` fallback path.
- [ ] Add focused fake/in-memory tests for aggregation logic.
- [ ] Defer any real-zarr large-run validation to local execution if sandbox
      behavior is unstable.
