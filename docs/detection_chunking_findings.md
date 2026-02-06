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
