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
4. PyNvVideoCodec sequential NVDEC path (`pynvvc_nv12_rgb` default candidate;
   `pynvvc_luma_rgb` fast diagnostic variant).
5. Crimson/native FFmpeg/NVDEC decode path (external Crimson diagnostic target).

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

Important follow-up: for the same `172 GB` MP4, Decord's one-time open cost was
much larger in production-scale compute-smoke runs (`~214-246s`). Source review
showed Decord's `VideoReader` unconditionally indexes keyframes by scanning
packets before returning. This makes Decord a poor fit for "start at frame 0,
stream sequentially once" jobs on very large MP4s.

2026-05-14 PyNvVideoCodec sequential probe on the same recording:

- Demuxer startup: `~0.03s`
- Decoder startup: `~0.13s`
- First frame: `~0.06s`
- Sequential luma decode: `1600` frames in `~13.5s` (`~118 fps`)
- Luma-to-RGB resize/preprocess: `~1900 fps` for `640x640`, batch `16`
- NV12-to-RGB resize/preprocess: `~680 fps` for `640x640`, batch `16`

`pynvvc_nv12_rgb` and `pynvvc_luma_rgb` are available in
`fisheye.diagnostics.detect_compute_smoke`. Both stream sequentially from frame
`0` through PyNvVideoCodec/NVDEC and feed tensor batches to YOLO. The NV12/RGB
backend performs BT.601 limited-range color conversion after resizing the Y, U,
and V planes and is the correctness-oriented default candidate. The luma/RGB
backend replicates the NV12 luma plane into RGB channels and remains a fast
diagnostic variant.

`--pipeline-mode producer` adds a bounded producer/consumer diagnostic for this
backend. The producer owns sequential PyNvVideoCodec demux/decode and buffers a
small number of decoded NV12 batches; the consumer preprocesses and runs YOLO.
This mode deliberately avoids per-batch global CUDA synchronization, because
that synchronization serializes decode and inference. Its per-stage timings are
therefore host/queue timings, while `summary.total_seconds` and
`summary.end_to_end_fps` remain the primary throughput metrics.

CUDA streams are not a substitute for synchronization; they move the required
synchronization to explicit event handoff points. The current producer mode does
not put preprocessing on a separate stream yet. If that is added, it must record
events between preprocess and inference and keep source/output tensors alive
until their producing streams have finished.

2026-05-14 compute-smoke follow-up:

- The compute-only detection smoke aligns with the production Decord-GPU tensor
  path by explicitly resizing tensor inputs to `detection.resize_dims`,
  passing the equivalent `imgsz`, enabling `torch.backends.cudnn.benchmark`,
  and converting the YOLO model to channels-last on CUDA.
- Be careful comparing historical `detect_yolo` per-batch inference timings
  with compute-smoke timings. Production currently times `model.predict()` host
  return without an explicit CUDA synchronization; some GPU work may synchronize
  later during result extraction. The compute smoke reports both
  `predict_return_seconds` and `inference_cuda_sync_seconds`, so it is better
  suited for diagnosing true GPU latency.
- For short smoke runs, use `steady_state_excluding_first_batch` rather than
  aggregate `inference_fps` when judging model throughput, because the first
  batch includes Ultralytics/PyTorch warmup effects.

2026-05-15 PyNvVideoCodec follow-up:

- The simple sequential `pynvvc_luma_rgb` backend avoided the Decord
  `VideoReader` open scan and processed 1600 frames at 72.9 end-to-end FPS on
  the sickyfish PRFS smoke video.
- The producer/consumer experiment was tested and is not currently beneficial.
  On the same 1600-frame smoke, producer mode dropped to 62.7 end-to-end FPS
  because queue wait/coordination outweighed overlap benefits.
- Production `detect_yolo` now accepts
  `--decode-backend auto|pynvvc_nv12_rgb|pynvvc_luma_rgb|decord_gpu|decord_cpu|opencv`.
  The default `auto` path now prefers `pynvvc_nv12_rgb` when CUDA and resize
  settings are available, then falls back to Decord/OpenCV.
- A fixed-frame `decord_gpu` vs `pynvvc_luma_rgb` parity run matched detection
  counts/classes but showed non-trivial box/score drift, so luma/RGB should not
  be the default correctness path. A corrected fixed-frame `decord_gpu` vs
  `pynvvc_nv12_rgb` parity run on frames `0 100 500 1000 1500` passed with
  bbox max drift `0.00156` and score max drift `0.00732`.
- Production `detect_yolo` writes `timing_summary` plus flat timing attrs for
  `read_decode`, `preprocess_resize`, `predict`, `postprocess`,
  `array_assembly`, and `zarr_write`, so backend comparisons can be audited from
  persisted detect runs rather than compute-smoke JSON alone.

2026-05-18 PyNvVideoCodec surface-reuse follow-up:

- The first clipped sleepyfish all-clips detect/refine collection exposed a
  correctness bug in the production PyNvVC detection path. Raw detections held
  then jumped in batch-aligned windows: `clip_000000` had continuous
  `frame_indices`, but only 8 unique bbox centers in the first 256 frames and
  exact plateau runs `[144, 16, 16, 16, 16, 16, 16, 16]`.
- Root cause: `detect_yolo` retained a list of `torch.from_dlpack(frame)` tensors
  returned by PyNvVideoCodec and preprocessed them after collecting the whole
  batch. Those tensors can reference decoder-owned reusable CUDA surfaces, so
  earlier list entries can point at overwritten pixels by the time preprocessing
  runs.
- Fix: the PyNvVC production path now consumes `PynvvcLumaRgbReader.iter_frames`,
  immediately preprocesses each decoded surface into owned resized RGB tensor
  memory, synchronizes that materialization before advancing decode, and then
  batches the owned tensors for YOLO. New runs record
  `pynvvc_surface_materialization=stream_preprocess_owned_batch_v1` in
  `parameters` and `timing_summary`.
- Regression coverage:
  `tests/unit/fisheye/test_detect_yolo_resize_contract.py::test_pynvvc_streamed_batch_materializes_frames_before_surface_reuse`
  mutates a fake reusable decoder surface after first yield and verifies the
  owned preprocessed batch still preserves distinct frame values.
- A 256-frame production-path smoke showed the fix removed the batch-window
  artifact: fixed `batch_size=16` matched fixed `batch_size=1` on 254/256
  frames and had 43 unique bbox centers in the first 256 frames.
- Full-clip timing on `sleepyfish_2026_05_05_17_45_30_cam2010093`
  `clip_000000` (54,000 frames, local A6000, fixed `pynvvc_luma_rgb`):

| Batch size | FPS | Total | Read/decode | Preprocess | Predict | Notes |
|------------|-----|-------|-------------|------------|---------|-------|
| `16` | `127.8` | `422.5s` | `288.1s` | `16.1s` | `111.1s` | Faster YOLO prediction but slower read/decode timing under conservative per-frame materialization. |
| `1` | `131.0` | `412.3s` | `33.0s` | `14.1s` | `357.4s` | Slightly faster end-to-end with current conservative sync policy. |

- Correctness comparison over the first 5,000 frames of the full clip:
  fixed batch 16 and fixed batch 1 matched exactly on 4,824/5,000 frames
  (`96.5%`), with max normalized bbox difference `0.00156`; plateau statistics
  were similar between batch sizes, so the historical 16-frame corruption pattern
  is gone.
- Current recommendation: for correctness-sensitive production clipped
  `pynvvc_luma_rgb` jobs, prefer `--batch-size 1` unless/until the conservative
  per-frame synchronization is replaced with a CUDA event / preallocated owned
  batch-buffer implementation. Batch 16 remains correct after the fix, but is
  not currently faster end-to-end in this measured full-clip smoke.
- Historical detect/refine runs created before this fix and lacking
  `pynvvc_surface_materialization=stream_preprocess_owned_batch_v1` should be
  considered suspect if they used PyNvVC with batch size greater than 1.

2026-05-15 Crimson/native decode follow-up:

- Crimson now has an external headless diagnostic target:
  `crimson_decode_smoke`.
- The target opens the same PRFS MP4 through Crimson's `FFmpegDemuxer`, decodes
  sequential frames with `NvDecoder`, and emits JSON with `decoder_backend`,
  `open_seconds`, `decode_seconds`, `total_seconds`, `frames_decoded`,
  `end_to_end_fps`, `gpu_name`, and `crimson_git_commit`.
- A 1600-frame local A6000 smoke on the sickyfish fixture produced:
  `open_seconds=0.024`, `decode_seconds=14.229`, `decode_fps=112.45`, and
  `end_to_end_fps=109.98`.
- Persisted JSON:
  `/groups/johnson/johnsonlab/jeremy/palette_smoke/logs/crimson_decode_smoke_20260515_sickyfish_cam2010093/crimson_decode_smoke_sickyfish_cam2010093_1600.json`.
- `scripts/report_detect_compute_smokes.py` accepts this Crimson decode-smoke
  JSON alongside Palette compute-smoke JSON.

Example LSF smoke using the sequential PyNvVideoCodec NV12/RGB backend:

These cluster examples assume the current preferred v003 detector has been
mirrored from `/nvme1/models/detect/...` to
`/groups/johnson/johnsonlab/jeremy/palette_models/detect/...`. If it has not
been copied yet, either copy the v003 run directory first or substitute the
available detector path explicitly.

```bash
scripts/submit_detect_compute_smoke_bsub.sh \
  --video /groups/johnson/johnsonlab/jeremy/palette_smoke/sickyfish_2026_02_23_16_23_35_cam2010093/cams/Cam2010093_sickyfish_2026_02_23_16_23_35_cam2010093.mp4 \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/weights/best.pt \
  --config configs/fisheye/yolo_detect_config.yaml \
  --log-dir /groups/johnson/johnsonlab/jeremy/palette_smoke/logs \
  --decode-backend pynvvc_nv12_rgb \
  --batch-size 16 \
  --max-batches 100 \
  --run-label sickyfish_cam2010093_pynvvc_nv12
```

Validate the result:

```bash
scripts/py scripts/check_detect_compute_smoke.py \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/<run-dir>/sickyfish_cam2010093_pynvvc_nv12.<JOBID>.json
```

Compare multiple compute-smoke JSON reports side by side:

```bash
scripts/py scripts/report_detect_compute_smokes.py \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/<decord-run>/*.json \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/<pynvvc-run>/*.json
```

Crimson/native decode smoke command shape:

```bash
cmake --build /home/delahantyj@hhmi.org/gitrepos/crimson-ui-monolith/build \
  --target crimson_decode_smoke -j 4

/home/delahantyj@hhmi.org/gitrepos/crimson-ui-monolith/release/crimson_decode_smoke \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/sickyfish_2026_02_23_16_23_35_cam2010093/cams/Cam2010093_sickyfish_2026_02_23_16_23_35_cam2010093.mp4 \
  --frames 1600 \
  --output-json /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/crimson_decode_smoke_20260515_sickyfish_cam2010093/crimson_decode_smoke_sickyfish_cam2010093_1600.json

scripts/py scripts/report_detect_compute_smokes.py \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/crimson_decode_smoke_20260515_sickyfish_cam2010093/*.json \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/<palette-compute-smoke-run>/*.json
```

Fixed-frame backend parity command:

```bash
scripts/validate_cluster_palette_env.sh --require-pynvvc

scripts/submit_detect_decode_backend_parity_bsub.sh \
  --video /groups/johnson/johnsonlab/jeremy/palette_smoke/sickyfish_2026_02_23_16_23_35_cam2010093/cams/Cam2010093_sickyfish_2026_02_23_16_23_35_cam2010093.mp4 \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/weights/best.pt \
  --config configs/fisheye/yolo_detect_config.yaml \
  --backend-a decord_gpu \
  --backend-b pynvvc_nv12_rgb \
  --frames 0 100 500 1000 1500 \
  --batch-size 16 \
  --max-bbox-diff 0.01 \
  --max-score-diff 0.05 \
  --log-dir /groups/johnson/johnsonlab/jeremy/palette_smoke/logs \
  --run-label sickyfish_cam2010093_decode_parity
```

The submitter also runs `scripts/validate_cluster_palette_env.sh --require-pynvvc`
inside the LSF job whenever either backend starts with `pynvvc_`; use
`--skip-env-check` only for low-level debugging.

The underlying Python command is:

```bash
scripts/py -m fisheye.diagnostics.compare_detect_decode_backend_predictions \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/sickyfish_2026_02_23_16_23_35_cam2010093/cams/Cam2010093_sickyfish_2026_02_23_16_23_35_cam2010093.mp4 \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/weights/best.pt \
  --config configs/fisheye/yolo_detect_config.yaml \
  --backend-a decord_gpu \
  --backend-b pynvvc_nv12_rgb \
  --frames 0 100 500 1000 1500 \
  --batch-size 16 \
  --max-bbox-diff 0.01 \
  --max-score-diff 0.05 \
  --fail-on-count-mismatch \
  --output-json /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/detect_decode_backend_parity_sickyfish_cam2010093.json
```

Validate the parity JSON:

```bash
scripts/py scripts/check_detect_decode_backend_parity.py \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/<run-dir>/sickyfish_cam2010093_decode_parity.<JOBID>.json
```

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

1. Done: add benchmark runner script that emits one JSON summary per run.
2. Done: add backend selector abstraction for decode path (where feasible).
3. Done externally: add Crimson/native decode-only adapter path for
   apples-to-apples timing (`crimson_decode_smoke` in the Crimson repo).
4. Done: add a report script/table formatter for side-by-side comparison.
5. Done for `detect_yolo`: split production timing provenance into read,
   preprocess/resize, predict, postprocess, array assembly, and Zarr write
   phases so CUDA synchronization artifacts do not hide where time is actually
   spent.

## Decision rule

- Prefer backend with best median end-to-end FPS and stable p90, provided output parity is acceptable.
- If decode-only is faster but end-to-end is not, defer backend swap.

## Operator notes

- Production `detect_yolo` now defaults to `--decode-backend auto`.
- `auto` prefers `pynvvc_nv12_rgb` when PyNvVideoCodec, CUDA, and resize dims
  are available. The PyNv path avoids Decord's long-video keyframe-index startup
  scan. It falls back to Decord/OpenCV when those prerequisites are unavailable.
- Decord remains the fallback path when PyNvVideoCodec, CUDA, or resize dims
  are unavailable, and it remains available explicitly through
  `--decode-backend decord_gpu` or `--decode-backend decord_cpu`.
- Do not remove the Decord fallback until fixed-frame prediction parity and at
  least one persisted `detect_run` comparison pass on representative videos.
- Treat Crimson/native path as experimental behind explicit diagnostic target.

## Crimson/native adapter status

Palette should not treat the Crimson/native FFmpeg/NVDEC path as part of the
default backend decision until its timing is compared with Palette compute-smoke
JSON on the same hardware. The current Crimson-side surface is a headless
decode-only diagnostic target, not a Palette production dependency.

Current Crimson-side acceptance criteria:

1. Done: add a headless decode-only timing command that can run on the
   same PRFS MP4 fixture without launching an interactive UI.
2. Done: emit JSON with at least: video path, frame count,
   open/init seconds, decode seconds, frames decoded, end-to-end FPS, GPU name,
   decoder implementation, and git commit.
3. Done: keep the Crimson path behind an explicit diagnostic executable; do not
   make it a Palette production dependency.
4. Done: compare the resulting JSON next to Palette compute-smoke reports with
   `scripts/report_detect_compute_smokes.py`; the formatter accepts both
   Palette compute-smoke JSON and the Crimson decode-smoke fields listed above.
5. Keep using Crimson timing only as an additional candidate in the backend
   decision; Palette default promotion still requires Palette output parity.

Crimson implementation note:

```text
Target: /home/delahantyj@hhmi.org/gitrepos/crimson-ui-monolith/release/crimson_decode_smoke
Source: /home/delahantyj@hhmi.org/gitrepos/crimson-ui-monolith/tools/crimson_decode_smoke.cpp
```
