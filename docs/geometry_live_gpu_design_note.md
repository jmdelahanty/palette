# Geometry-Live GPU Design Note

Date: 2026-04-04
Type: Design note / follow-up TODO

## Purpose

Document what it would take to accelerate `geometry_live` ROI reads when a crop
run is `geometry_only` and the archive does not persist `raw_video/images_full`
inside the Zarr.

This is specifically about the case where:

- the canonical analysis archive keeps crop geometry and provenance,
- source frames are resolved from `source_video_path`,
- downstream ROI stages want to read ROIs directly without first building a
  temporary ROI cache.

## Current State

For geometry-only crop runs, `CropImageSource` currently does one of two things:

- if `raw_video/images_full` exists, read full frames from the Zarr and crop on
  CPU
- otherwise, open `source_video_path` and decode frames live, then crop on CPU

Current external-video live behavior is CPU-bound:

- decode uses decord with `ctx=cpu()` when available, otherwise OpenCV
- cropping is done in NumPy on CPU using per-ROI top-left coordinates
- downstream GPU inference waits on the live ROI read path

On the representative copied analysis archive benchmarked on `2026-04-04`:

- archive: `2026-01-28T22-15-03Z_arena_1_DefaultScreen_analysis.zarr`
- ROI count: `22,876`
- ROI size: `512x512`
- source frames: external MP4 via `source_video_path`
- `raw_video` group existed, but `images_full` / `images_ds` were not present

Observed behavior:

- materialized crop runs performed normally
- `geometry_live` ran far slower, with low GPU utilization
- operator observation: after roughly `20 minutes`, the keypoint benchmark was
  still not near completion

Interpretation:

- the current live path is decode/crop-bound, not model-bound
- for large full-frame external videos, pure `geometry_live` is not an
  acceptable hot path in its current CPU form

## Reusable Existing Pieces

The repo already contains most of the hard GPU-side implementation in the
temporary ROI cache materializer:

- external video GPU decode:
  - `VideoReader(..., ctx=gpu(0))`
- GPU grayscale conversion
- GPU ROI extraction using stored top-left ROI coordinates:
  - `_process_chunk_gpu_from_top_left(...)`
- chunked processing controls:
  - `gpu_chunk_frames`
- fallback handling when GPU/decord/kvikIO/CuPy are unavailable

This exists today in `src/fisheye/tracking/crop.py` under
`materialize_external_roi_cache(...)`.

That means a future GPU `geometry_live` implementation should reuse the cache
builder kernels and chunking strategy rather than inventing a second crop kernel
family.

## Proposed Goal

Add a GPU-capable live ROI read path for external-video geometry-only crop runs.

Target outcome:

- keep the canonical archive lean (`geometry_only` crop run)
- allow one-off ROI inference to avoid the current CPU decode/crop bottleneck
- still keep the temporary ROI cache as the preferred option when more than one
  downstream stage will reuse the same ROIs

This is an optimization for live ROI reads, not a replacement for the shared
temporary ROI cache.

## Staged Plan

### Phase 1: GPU decode/crop, CPU return

Implement a batch-oriented GPU live path inside `CropImageSource` for
`frame_source_kind == "source_video_path"`.

Behavior:

- group requested ROI rows by frame window
- decode those windows on GPU
- crop ROIs on GPU using stored top-left coordinates
- return the final ROI batch as CPU NumPy arrays

Why start here:

- fits the existing `CropImageSource.read_slice(...) -> np.ndarray` contract
- requires minimal downstream changes
- should remove the major CPU decode/crop bottleneck

Limitations:

- still pays a device-to-host copy
- downstream consumers may upload again to GPU

### Phase 2: GPU-resident ROI batch option

Add an optional path for downstream stages that can accept device-resident
batches directly.

Best first candidate:

- U-Net eye-mask inference

Less friendly current candidate:

- YOLO pose inference via Ultralytics `model.predict(...)`, which is currently
  driven by CPU/NumPy image batches and CPU grayscale-to-RGB expansion

This phase should only happen if Phase 1 leaves enough performance on the table
to justify the wider API surface change.

## API / Policy Sketch

Potential reader knobs:

- `roi_live_acceleration = auto | cpu | gpu`
- `roi_live_gpu_chunk_frames = <int>`

Proposed defaults:

- `auto` for future mixed-mode benchmarks and selected analysis workflows
- `cpu` fallback whenever GPU live decode/crop is unavailable

Important distinction:

- `roi_cache_policy` decides whether to materialize/reuse a temporary cache
- `roi_live_acceleration` decides how to execute pure live ROI reads when cache
  is not used

These should remain separate controls.

## Expected Impact

The likely effect of Phase 1 is:

- `geometry_live` becomes materially less bad for large external videos
- one-off geometry-only inference becomes more tolerable
- the gap between `geometry_live` and `geometry_cache_reuse` should narrow

But the temporary ROI cache should still remain the preferred path when:

- multiple downstream stages will reuse the same ROI set
- the first-stage cache build cost can be amortized
- stable local scratch is available

## Risks / Constraints

- GPU memory pressure when decoding `4512x4512` frames in chunks
- ensuring crop semantics exactly match the CPU path
  - clipping
  - padding
  - grayscale conversion
- preserving deterministic row ordering
- avoiding a second independent live-crop implementation that drifts from the
  cache builder
- not over-optimizing the YOLO path before clarifying whether the main target is
  keypoints, U-Net eye masks, or both

## Recommended Next Step

Implement Phase 1 only:

- external-video `geometry_live_gpu`
- reuse `_process_chunk_gpu_from_top_left(...)`
- keep `CropImageSource.read_slice(...)` returning NumPy
- add a focused benchmark comparing:
  - `geometry_live_cpu`
  - `geometry_live_gpu`
  - `geometry_cache_build`
  - `geometry_cache_reuse`

This should happen before any attempt to make the downstream inference wrappers
GPU-native end-to-end.
