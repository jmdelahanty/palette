# Zarr on NFS: Audit + Design TODO

## Context (current assumptions)

- Filesystem: NFS.
- Workload: single node, write-once read-many after import.
- Archiving: rare (not a primary requirement yet).

## Current behavior in this repo

- Zarr v3 `LocalStore` is the default for new imports.
- Standard import path always writes **sharded** `raw_video` arrays (frame shards).
- kvikIO/GDS import path uses `use_sharding`; default `false` means **chunk-only**.
- Derived arrays (crops, detections, keypoints, masks, etc.) are chunked, not sharded.
- Many call sites open Zarr directly instead of using `open_zarr_root`.
- No explicit consolidated-metadata step is run after writes.

## Design goals for NFS

- Minimize metadata IOPS and inode count.
- Preserve good read performance for common access patterns (frame + ROI slices).
- Keep the pipeline UX simple and predictable.

## What I need to learn / measure

- NFS limits and behavior: metadata latency, inode caps, mount options.
- File counts per dataset: chunk files per array, per run, total.
- Access patterns in practice: how many chunks do visualizers read per action?
- Zarr v3 consolidated metadata support (if any) and its impact on open times.
- Sharding impact on read/write speed for `raw_video` and `crop_runs`.

## TODO: technical evaluation

1. Benchmark open + read on a representative dataset and log timings.
2. Count chunk files/inodes per dataset and per stage.
3. Test sharding on the kvikIO path for 4512x4512 frames.
4. Explore sharding for `crop_runs` (large, high-count arrays).
5. Decide on a default sharding policy for NFS (target shard size in MB).
6. Evaluate consolidated metadata support and add optional finalize step.
7. Route all open calls through `open_zarr_root` for future store changes.

## TODO: design changes (if needed)

- Align config semantics: either honor `use_sharding` everywhere or remove it and
  document always-sharded defaults.
- Record chunk/shard sizes for all large arrays in run metadata.
- Add an optional "pack/export" step (ZipStore or similar) for portability.
- Document storage modes and NFS-recommended settings.

## Success criteria

- Predictable open times and stable read throughput on NFS.
- File counts stay within NFS inode/metadata constraints.
- Storage layout is explicit and auditable in metadata.
