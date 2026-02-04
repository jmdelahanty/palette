# Zarr Sharding Design Note

## Purpose

Define how (and when) we use sharding to reduce NFS metadata pressure while
keeping the pipeline usable for a single-node workflow. Sharding remains
optional until we validate it across all import paths.

## Constraints and assumptions

- Current workloads are single node, write-once read-many.
- Future workloads may run on a cluster; NFS metadata is the risk.
- kvikIO/GDS path may not reliably support sharding (needs validation).

## Current behavior (as implemented)

- Standard import (`create_palette_zarr`) writes sharded `raw_video` arrays.
- kvikIO/GDS import uses `use_sharding`; default config now enables sharding.
- Derived arrays (crops, refined runs, masks) are chunk-only.

## Design principles

- Sharding should be explicit and auditable in metadata.
- Default behavior should remain stable while we validate.
- Prefer measurement-driven shard sizes over hard-coded guesses.
- Do not block usage if sharding is unsupported (fallback cleanly).

## Proposed policy (conceptual)

- Raw video:
  - Shard target size based on frame size (e.g., 128-256 MB per shard).
  - Record frames-per-shard and shard shape in `raw_video.attrs`.
- Crop runs:
  - Shard only the large image arrays (`roi_images`).
  - Smaller arrays remain chunk-only.
- Other runs:
  - Keep chunk-only until measured, then shard if file counts become large.

## Configuration approach (future)

- Add a storage section, for example:
  - `storage.sharding_enabled`: bool (default false or "auto")
  - `storage.shard_target_mb_raw`: int
  - `storage.shard_target_mb_crops`: int
  - `storage.shard_policy`: "raw_only" | "raw_and_crops"

## Validation steps

1. Measure file counts with and without sharding on NFS.
2. Compare open/read times (visualizers + crop access).
3. Validate that sharding works on the kvikIO/GDS path.
4. Confirm that downstream tools read sharded arrays without changes.

## Post-import packing (optional)

If sharding is unavailable at import time (e.g., kvikIO/GDS), a post-import
"pack" step can rewrite the dataset into a sharded store.

Notes:
- This is a full rewrite of array data into shard files (not a metadata toggle).
- Requires temporary storage during the pack.
- Best run once after the pipeline is complete (write-once/read-many).

## Decision points

- Should sharding be the default for raw_video on NFS?
- Should the kvikIO/GDS path disable sharding if unsupported, with a warning?
- Do we shard crops by default once validated?

## Recommended profiles

### Local / dev (fast import, fewer guarantees)

- Sharding: optional (off by default for faster imports).
- Use when working on a local NVMe and not sharing datasets.
- Example config: `configs/fisheye/import_local.yaml`

### Cluster / NFS (stable reads, fewer files)

- Sharding: on by default for `raw_video`.
- Use when datasets live on NFS or when metadata IOPS is a concern.
- Example config: `configs/fisheye/import_nfs.yaml`
