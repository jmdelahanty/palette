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

## Current Cluster Decode Finding

2026-05-14 smoke benchmark on a Janelia L4 compute node:

- Video: `sickyfish_2026_02_23_16_23_35_cam2010093.mp4`
- Size: `172,841,839,775` bytes
- Resolution: `4512x4512`
- Frames tested: `1000`
- Resize: `640x640`
- Command shape:

```bash
scripts/py -m fisheye.diagnostics.benchmark_video_decode \
  <video.mp4> \
  --frames 1000 \
  --resize 640 640 \
  --batch-sizes 1 4 8 16
```

Results:

| Source path | OpenCV CPU | Decord CPU | Decord GPU single | Decord GPU batch=1 | Decord GPU batch=4 | Decord GPU batch=8 | Decord GPU batch=16 |
|-------------|------------|------------|-------------------|--------------------|--------------------|--------------------|---------------------|
| PRFS `/groups/...` | 27.8 fps | 28.8 fps | 100.2 fps | 96.6 fps | 97.5 fps | 93.9 fps | 90.1 fps |
| local `/tmp/...` copy | 28.0 fps | 30.9 fps | 100.7 fps | 96.4 fps | 97.0 fps | 94.1 fps | 90.9 fps |

The full-video copy to `/tmp` took `3m15s` and did not materially improve
sustained decode throughput. For this workload, streaming from PRFS was
effectively equivalent to local `/tmp` once the reader was open.

Operational policy from this measurement:

- Do not copy full videos to node-local scratch by default for single-pass
  detection.
- Keep one Decord `VideoReader` open per video/job; avoid reopening per batch.
- Use scratch for workflows that repeatedly reopen the same video, do heavy
  random seeks, or show measured PRFS throughput limits.
- The cluster environment validator's Decord GPU smoke is an environment check,
  not a sustained throughput benchmark. Use this benchmark utility for PRFS vs
  scratch decisions.

Decord startup still has a real one-time open cost for very large MP4s
(`~19-22s` in this smoke). That cost appears dominated by container/Decord/FFmpeg
initialization rather than PRFS versus local storage, because local `/tmp` did
not reduce it.

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
