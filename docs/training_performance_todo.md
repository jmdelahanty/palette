# Training Performance TODO

## Context
- Current detection training throughput: ~3.2 it/s at batch 128 on RTX A6000.
- CPU/IO likely the bottleneck (random Zarr chunk reads + decompression).
- GPU memory usage ~17 GB; GPU idle time suggests data pipeline stalls.

## Completed
- `persistent_workers` is now configurable in detect training and can be enabled in `training_params`.
- `chunk_cache_size` is now configurable in detect training and applied in the Zarr loader.
- Detect loader now supports `num_workers` and `prefetch_factor` in `training_params`.
- Detect loader now supports `chunk_locality_sampling` (chunk-aware batch sampling for read locality).
- Explicit DataLoader worker shutdown was added at train end to avoid persistent-worker exit hangs.
- `merge-copy-batch-size` is available in merged dataset export CLI.
- `train_detection --profile` now reports input-pipeline timing breakdown and writes `input_pipeline_profile.json` per run.

## Latest Findings (2026-02-07, detect_cedar_shadow_v006)
- Profile attribution:
  - `dataset_zarr_read` dominates input time (~447s of ~547s `__getitem__` total in profile run).
  - `preprocess_to_device` is small relative to loader time.
- Merged Zarr layout inspected:
  - `zarr_format=3`, `chunks=(64, 640, 640)`, `compressors=(ZstdCodec(level=0),)`.
- Throughput A/B (1 epoch window):
  - `chunk_locality_sampling=true` vs `false`: ~`1.3 it/s` vs `1.0 it/s` (~30% gain).
  - `chunk_cache_size=32` was slower than `64` (~`1.7 it/s` vs ~`1.8 it/s` peak).
  - `prefetch_factor=3` was less stable/slower at tail than `2`.
- Current best-known practical setting:
  - `batch=256`, `num_workers=8`, `prefetch_factor=2`, `chunk_cache_size=64`,
  - `persistent_workers=true`, `chunk_locality_sampling=true`.

## Short-Term Checks
- Inspect Zarr chunking + compression:
  - raw_video/images_ds chunks, compressor type, and chunk cache behavior.
- Confirm data loader parameters:
  - num_workers, prefetch_factor.
- Extend `--profile` to support multi-worker timing collection with low overhead.
- Clarify in logs/docs that "input pipeline" includes more than disk IO (disk + CPU prep + transfer staging).
- Run chunk-cache sweep:
  - `chunk_cache_size` in {0, 32, 64, 128}.
- Run merged export copy sweep:
  - `--merge-copy-batch-size` in {128, 256, 512}.
- Run batch sweep on A6000:
  - batch in {128, 192, 256}, compare throughput and convergence.
- Discuss/decide CUDA prefetcher design:
  - double-buffer next batch on a dedicated CUDA stream,
  - expected memory overhead (at least one extra batch on device),
  - rollout plan gated behind a config flag.

## Medium-Term Improvements
- Add configurable dataloader settings (workers, prefetch_factor) in training config.
- Consider lower num_workers if disk contention is high.
- Consider grouped sampling (reduce random chunk access).

## Storage Layout Options
- Create a training-optimized Zarr (smaller frame chunks, e.g., 1–4 frames).
- Optionally store a cached training dataset on fast storage.

## GPU-Direct (GDS) Path
- Investigate reading Zarr via GPU-direct storage on compatible machines.
- Check whether raw_video/images_ds is stored in a GPU-friendly chunk layout.
- Evaluate GPU-based decoding or pinned memory for transfer.
- Evaluate KvikIO feasibility for this pipeline:
  - storage/filesystem and driver prerequisites for GDS,
  - integration points with Zarr chunk reads,
  - benchmark plan vs current CPU->GPU transfer path.

## Validation
- Track it/s and GPU utilization for each change.
- Log throughput and stall time in training reports.

## Benchmark Protocol
- Use the same merged dataset and same random seed for all runs.
- Run a short fixed test window (for example, first 5 epochs) for each setting.
- Record:
  - epoch wall time,
  - average it/s,
  - GPU utilization and memory,
  - epoch-boundary idle gap.
- Promote only settings that improve throughput without degrading validation metrics.
