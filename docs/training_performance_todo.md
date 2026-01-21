# Training Performance TODO

## Context
- Current detection training throughput: ~3.2 it/s at batch 128 on RTX A6000.
- CPU/IO likely the bottleneck (random Zarr chunk reads + decompression).
- GPU memory usage ~17 GB; GPU idle time suggests data pipeline stalls.

## Short-Term Checks
- Inspect Zarr chunking + compression:
  - raw_video/images_ds chunks, compressor type, and chunk cache behavior.
- Confirm data loader parameters:
  - num_workers, persistent_workers, prefetch_factor.

## Medium-Term Improvements
- Add configurable dataloader settings (workers, prefetch_factor, persistent_workers) in training config.
- Consider lower num_workers if disk contention is high.
- Consider grouped sampling (reduce random chunk access).

## Storage Layout Options
- Create a training-optimized Zarr (smaller frame chunks, e.g., 1–4 frames).
- Optionally store a cached training dataset on fast storage.

## GPU-Direct (GDS) Path
- Investigate reading Zarr via GPU-direct storage on compatible machines.
- Check whether raw_video/images_ds is stored in a GPU-friendly chunk layout.
- Evaluate GPU-based decoding or pinned memory for transfer.

## Validation
- Track it/s and GPU utilization for each change.
- Log throughput and stall time in training reports.
