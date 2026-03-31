# Refined Subject-Mask Scheduler TODO

Purpose: make refined subject-mask execution scheduler-aware by default, using
the same general execution model as other Palette stages, without turning the
interactive editor into a Dask-bound UI loop.

Date anchored: 2026-03-30.

## Rollout Status

- Phase 1 is in place:
  - the ROI-local save path now shares one canonical apply kernel for single-ROI
    and multi-ROI updates
- Phase 2 is in place:
  - `src/fisheye/refinement/refine_subject_masks.py` now provides a
    scheduler-aware non-UI apply entrypoint
  - the entrypoint supports explicit `--run-name`, component/ROI scoping,
    scheduler options, and `--dry-run` plan output
  - the pipeline now exposes this as the `refined_subject_masks` stage

Canary validation on 2026-03-31:

- archive:
  - `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`
- source run:
  - `subject_masks_canary_sam_points_body_eyes_001`
- refined run:
  - `refined_subject_masks_canary_sam_points_body_001`
- component:
  - `subject_body`
- full batch apply result:
  - `changed_roi_count = 0`
  - `noop_roi_count = 227`

So the current batch engine appears safe on the canary refined run, and the
existing body-only refined metadata was already consistent with the stored masks.

## Current State

Today the refined subject-mask path has two execution modes that share one
semantic apply contract:

- local interactive apply in `src/fisheye/tune/refined_subject_mask_review.py`
  - `prepare_refined_subject_run(...)` creates or backfills the refined run
    synchronously
  - `save_refined_subject_roi(...)` updates one ROI row at a time through the
    shared apply kernel
  - `sync_refined_subject_mask_metadata(...)` updates touched ROI rows through
    that same shared apply kernel
  - the OpenCV review UI remains a single-user local loop
- scheduler-aware non-UI apply in
  `src/fisheye/refinement/refine_subject_masks.py`
  - supports chunked batch recompute with `scheduler`, `num_workers`, and
    `chunk_size`
  - writes through the driver after chunk computation
  - is exposed in the pipeline as the `refined_subject_masks` stage

So the remaining work here is no longer "make refinement scheduler-aware at
all." The remaining work is to broaden usage beyond the current body-only
canary and decide how far to push scheduler-backed recompute for future
multi-component refinement.

## Desired Direction

Refinement should behave like a real stage, not like a one-off serial helper.

Target principle:

- refinement execution is scheduler-aware by default
- the editor/inspector are frontends to that execution layer
- interactive painting stays local and responsive
- the "apply refinement" step is what uses the scheduler-backed engine

This means the architecture should separate:

1. edit capture
   - user changes pixels or ROI-local component masks
2. refinement apply
   - masks, QC metrics, geometry summaries, reasons, and review metadata are
     materialized into the canonical refined run

The second layer should look like other scheduler-aware Palette stages.

Important boundary:

- the pipeline/stage application layer should be scheduler-aware
- interactive saveback should stay synchronous and local by default

That means Paintera, Crimson, and the OpenCV review UI should not require Dask
to save a small touched ROI set.

## Existing Palette Pattern To Match

Other stages in Palette already follow a scheduler model such as:

- `threads`
- `processes`
- `distributed`
- `single-threaded`

Relevant examples:

- `src/fisheye/tracking/crop.py`
- `src/fisheye/refinement/refine_eye_masks.py`
- `src/fisheye/segmentation/eye_segmentation.py`

The refined subject-mask path should align with that vocabulary rather than
inventing a refinement-specific execution model.

## Recommended Execution Model

### 1. Keep the UI/saveback path local

The live review/editor loop should not require Dask round-trips for every brush
event or save action.

Interactive frontends should:

- collect changed ROIs or ROI chunks
- apply those touched chunks through a synchronous local refinement path
- reload the updated slices after the local apply finishes

This keeps interaction latency acceptable and avoids making Dask the user-facing
editing loop.

### 2. Make the refinement engine chunked

The engine should operate on disjoint ROI chunks, not single pixels.

Chunk tasks should own:

- one refined run
- one component or a defined component subset
- one ROI span / chunk

Each chunk task should read, update, and write only its assigned ROI range.

### 3. Share refinement semantics, not necessarily execution mode

All frontends should share one canonical refinement logic contract, but not
necessarily one execution mode.

Two modes are acceptable and desirable:

- local immediate apply
  - used by Paintera, Crimson, and the OpenCV review UI
- scheduler-backed stage apply
  - used by pipeline or batch execution

The same semantic outputs should be produced from both modes:

- canonical refined masks
- geometry and QC metrics
- reasons and review metadata
- consistent provenance updates where applicable

Likely call sites:

- the interactive refined-subject review tool
- Paintera/Crimson saveback hooks
- future batch repair/recompute tools
- future pipeline entrypoints

This avoids semantic drift while still allowing the interactive save path to
remain simple and latency-friendly.

## What The Engine Should Own

For each touched ROI/component chunk, the engine should own recomputation of:

- `masks_roi`
- `edit_applied`
- run-level `metrics/`
  - `mask_present`
  - `area_px`
  - `centroid_xy`
  - `centroid_valid`
  - `bbox_xyxy`
  - `bbox_valid`
- component-local metrics under `components/<component>/metrics/`
  - `component_count`
  - `largest_component_fraction`
  - `hole_count`
  - `hole_area_fraction`
  - `sigma_noise`
  - `curvature_var`
  - `ipr`
  - `solidity`
- `reason_bytes` / `reason`
- `updated_at_utc`

Shared attrs or aggregate summaries should be written once in the driver after
chunk tasks complete.

## Write-Safety Rules

To avoid the same distributed-write pitfalls seen elsewhere:

- workers must only write disjoint ROI chunks
- workers must not race on the same array region
- shared attrs must be finalized in one place after worker completion
- the engine should prefer passing `zarr_path + run_name + ROI slice` to workers
  rather than large in-memory arrays

That keeps worker payloads small and matches better distributed patterns used in
other Palette stages.

## Proposed Surface

The future refinement engine should accept stage-style execution options such as:

- `--scheduler threads|processes|distributed|single-threaded`
- `--num-workers`
- `--chunk-size`

Possible call sites for the scheduler-backed mode:

- future batch refresh/repair CLI
- future non-interactive refinement pipeline step

Paintera/Crimson saveback should remain on the local immediate-apply path.

We do not need to commit to the final public entrypoint name yet, but the
execution model should be shared.

## Provenance Expectations

Once refinement is scheduler-aware, the run should record scheduler metadata in
the same spirit as other stages, for example:

- `dask_scheduler`
- `dask_num_workers`
- `dask_chunk_size`
- `dask_version`

This can live in stage provenance or direct attrs, but it should be present.

## Suggested Rollout

### Phase 1: Extract a chunk-safe apply kernel

- Move the current ROI-local saveback logic behind a kernel that can operate on
  explicit ROI spans.
- Keep behavior identical to the existing serial save path.

### Phase 2: Add scheduler-aware non-UI execution

- Add a refinement apply entrypoint that accepts scheduler settings.
- Start with `single-threaded`, `threads`, and `processes`.
- Treat `distributed` as an extension, not the first milestone.

### Phase 3: Route frontends through the same engine

- Update the review UI, Paintera, and Crimson paths to share the same
  refinement semantics and metric/reason logic.
- Keep those saveback paths local and synchronous by default.
- Do not require Dask/distributed execution for small interactive saveback.

### Phase 4: Add chunk-level summary/finalization

- Make aggregate attrs or summaries finalize in the driver.
- Ensure provenance captures scheduler settings.

## Acceptance Criteria

- Refined subject-mask execution is no longer an editor-only serial code path.
- Interactive editing still feels local and responsive.
- Interactive saveback uses the same refinement semantics as the pipeline path
  without requiring scheduler startup.
- Scheduler-backed refinement exists at the stage/pipeline application layer.
- Scheduler settings are explicit and align with the rest of Palette.
- Worker writes are chunk-disjoint and do not race on shared attrs.

## Non-Goals

- Making every brush event itself a Dask task.
- Requiring Paintera/Crimson saveback to go through Dask/distributed execution.
- Turning the subject-mask inspector into a distributed application.
- Replacing Paintera with a general-purpose distributed editor.
- Defining the final full pipeline step name today.
