# Detection Chunking Findings

## Scope
- Question: how merged detection-training Zarr chunking relates to RTX A6000 training performance.
- Dataset inspected: `detect_cedar_shadow_v001`.

## What Was Measured
- Merged image array chunking:
  - `/nvme1/training/datasets/detect_cedar_shadow_v001/zarr/detect_cedar_shadow_v001_merged.zarr/raw_video/images_ds`
  - `chunk_shape = [64, 640, 640]`
- Label/index arrays:
  - `bbox_norm_coords`: `chunk_shape = [8192, 4]`
  - `frame_indices`: `chunk_shape = [8192]`
  - `source_index/source_frame_idx`: `chunk_shape = [8192]`

## Loader Access Pattern
- Detection training reads one frame at a time by index in `__getitem__`:
  - `src/fisheye/training/zarr_yolo_dataset_loader.py:919`
- Bboxes and frame indices are pre-cached in RAM:
  - `src/fisheye/training/zarr_yolo_dataset_loader.py:752`
  - `src/fisheye/training/zarr_yolo_dataset_loader.py:778`

## Interpretation
- GPU batch size and Zarr chunk size are different knobs:
  - `batch` affects VRAM/compute.
  - Zarr chunks affect I/O and decompression work.
- With `images_ds` chunks of 64 frames and random frame access, each sample can trigger decode of a 64-frame chunk.
- This can increase CPU/decompression overhead and data-loader stalls even when GPU VRAM is sufficient.

## Current Practical Read
- For `yolo11n` on RTX A6000 at `imgsz=640`, `batch=128` can still fit and train.
- If throughput is lower than expected, chunk depth is a likely contributor to input-pipeline inefficiency.

## Suggested Direction
- Prefer smaller frame-depth chunks for merged training datasets with random sampling:
  - typical target range: `8-32` frames, often `16` as a good starting point for gray input.
- Keep index arrays chunked large (current `8192` is fine).

## Recommended Follow-Up
1. Add exporter control for merged frame chunk depth (for example `--merge-frame-chunk`).
2. Re-export one dataset with `frame_chunk=16`.
3. Compare training throughput (`it/s`) and GPU utilization vs current `64`.

## Detect Inference Runtime Note (2026-02-09)

### Observation
- For full-video detect on `4512x4512` HEVC input with GPU decode + YOLO GPU inference:
  - average decode/read time per batch was ~`121 ms`
  - average model inference time per batch was ~`58 ms`
- Practical implication: decode/read is currently the larger bottleneck than model forward time.

### Design read
- The current detect path already uses key accelerator features:
  - fused model
  - FP16 on CUDA
  - channels-last tensors
  - Decord GPU decode path
- Therefore, adding `torch.compile`/CUDA-graph style optimization may help model time, but likely gives limited end-to-end gain unless decode/read is also improved.

## Decord vs Native Decode Benchmark Plan

### Goal
- Measure whether native decode paths (for example, Crimson/FFmpeg pipelines) materially outperform current Decord decode in end-to-end detect throughput.

### Backends to compare
- Decord GPU decode (current production path).
- Decord CPU decode.
- OpenCV fallback decode.
- Crimson/native decode path (if available in benchmark environment).

### Fixed test conditions
- Same input recording (`.mp4`), same frame range, same sampling policy.
- Same YOLO model weights and detection parameters (`conf`, `iou`, `max_det`, `batch_size`, `imgsz`).
- Same GPU, same host load conditions, warmup run before timed run.

### Metrics
- Decode FPS (decode-only microbenchmark).
- End-to-end detect FPS (decode + preprocess + model + write).
- Batch-level timing split:
  - decode/read ms
  - inference ms
  - write ms
- GPU utilization and VRAM footprint.

### Success criteria
- Native decode path should show repeatable reduction in decode time and clear end-to-end FPS improvement (not just microbenchmark decode gains).
- No regression in detection outputs or provenance fields.
