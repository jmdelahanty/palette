# Detect Decode Backend Benchmark TODO

Purpose: compare Decord-based decode against native decode paths (including Crimson) for end-to-end detect throughput on production recordings.

## Why this exists

- Recent detect run timing showed decode/read cost larger than model inference cost.
- Before adding major inference-engine complexity, we should measure whether decode backend changes provide bigger gains.
- 2026-05-14 follow-up: the Decord GPU path feeds torch tensors to
  Ultralytics. For tensor inputs, canonical `detection.resize_dims` must be
  applied explicitly before `model.predict`; passing `imgsz` alone can leave
  inference running on source-resolution tensors. This is especially expensive
  for `4512x4512` recordings and can make `.pt` inference appear anomalously
  slow even when the model itself is fast.

## Scope

- In scope:
  - decode backend comparison for detect pipeline
  - timing instrumentation and reproducible benchmark protocol
- Out of scope:
  - model architecture changes
  - annotation schema changes

## Candidate backends

1. Decord GPU decode (`detect_yolo` current primary path).
2. Decord CPU decode.
3. OpenCV decode fallback.
4. Crimson/native FFmpeg decode path (external integration path).

## Benchmark fixture

- Recording: one representative `4512x4512` HEVC source recording.
- Model: fixed detect model path from registry resolver.
- Detect params: fixed `conf`, `iou`, `max_det`, `batch_size`, and resize dims.
- Hardware: single machine/GPU per run, no competing heavy jobs.

## Metrics to collect

- End-to-end detect FPS.
- Decode/read ms per batch.
- Explicit preprocess/resize ms per batch.
- Inference ms per batch.
- Write ms per batch.
- GPU utilization and VRAM usage.
- Output sanity:
  - detection count
  - frame coverage
  - mean confidence

## Execution protocol

1. Warm up each backend once (discard timing).
2. Run at least 3 timed repetitions per backend.
3. Report median and p90 for each timing metric.
4. Keep all non-backend knobs fixed.

## Integration tasks

1. Add benchmark runner script that emits one JSON summary per run.
2. Add backend selector abstraction for decode path (where feasible).
3. Add Crimson adapter path (decode-only or decode+detect feed) for apples-to-apples timing.
4. Add a report script/table formatter for side-by-side comparison.
5. Split production timing provenance into read, preprocess/resize, predict,
   postprocess, and write phases so CUDA synchronization artifacts do not hide
   where time is actually spent.

## Decision rule

- Prefer backend with best median end-to-end FPS and stable p90, provided output parity is acceptable.
- If decode-only is faster but end-to-end is not, defer backend swap.

## Operator notes

- Keep current Decord path as default until benchmark conclusion.
- Treat Crimson/native path as experimental behind explicit flag.
