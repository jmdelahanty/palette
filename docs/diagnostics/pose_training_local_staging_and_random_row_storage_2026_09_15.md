# Pose training local staging and random-row storage — 2026-09-15

## Scope and preserved behavior

This change combines two explicitly separate performance/storage corrections:

1. Pose training may execute from a verified ephemeral node-local copy of an
   immutable merged training export.
2. Future merged keypoint training exports use a named storage profile whose
   ROI inner chunk contains one training sample.

Dataset rows, artifact splits, keypoint coordinates and visibility, skeleton
identity, logical dataset hashes, source lineage, augmentation, and scientific
training parameters remain unchanged. Staging is an execution decision and is
recorded outside logical dataset identity. Rechunking legitimately changes the
physical tree and storage-plan declaration while retaining the existing
logical-content hash contract.

Caching the selected crop run and `roi_images` array in the pose dataset is an
enforcement correction. Labels were already fixed at dataset construction, but
the old image path was re-resolved for every sample. A selector change during a
run could therefore pair cached labels with pixels from another crop run. The
dataset now binds both labels and pixels to the run selected at construction.

## Default staging policy

`PoseConfig.dataset_staging` is execution-only and defaults to:

```yaml
dataset_staging:
  mode: auto
  max_total_size_gib: 8.0
  min_free_space_gib_after_stage: 8.0
```

`auto` stages eligible merged training exports only when their durable source
is on `/groups` or `/nrs`. The full eligible set must fit the total-size and
free-space gates. The scratch search order is an LSF job directory below
`/scratch/$USER`, then `$TMPDIR`, then `/tmp`; shared filesystems are refused as
scratch.

The source is inventoried by relative path and size, all source file contents
are SHA-256 hashed, the tree is copied, and every staged file is hashed again.
The staged path is admitted only after the two complete trees match. The v1
staging receipt records the durable and effective paths, export eligibility,
logical dataset hash when present, physical bytes and file count, inventory and
content digests, limits, and cleanup policy. The pose runtime receipt is now v2
and embeds this staging receipt. Temporary data is removed after training and
also has process-exit cleanup.

Explicit immutable publications are admitted through their complete/immutable
root contract. Historical `_merged.zarr` artifacts that predate that contract
are admitted through a named `legacy_closed_merged_training_export`
compatibility classification when they carry the root training-export metadata
and merged crop/split surface. Live per-recording training Zarrs are not staged
by `auto`.

## Future export storage

The named `training_random_row_immutable_v1` profile replaces
`training_immutable_v1` for new merged keypoint training arrays. `roi_images`
retains `access_pattern=per_row`, uses one sample per independently decodable
inner chunk, and groups inner chunks into approximately 32 MiB indexed outer
shards. Other arrays retain the established one-MiB planning behavior. The
profile ID and every concrete chunk and shard shape remain in the export's
storage manifest and array metadata.

For 10,717 grayscale 192×192 images the concrete plan is:

- inner chunk: `1×192×192` (36,864 uncompressed bytes)
- outer shard: `911×192×192` (33,583,104 uncompressed bytes)
- estimated ROI payload objects: 12

## Measured evidence

The mounted source was
`pose_head_192_recovered_reviewed_v001_merged.zarr`, containing 10,717 rows.
Random rows used one cached Zarr array handle.

| Read path/layout | Random-row time |
|---|---:|
| `/groups`, existing 32-row inner chunks | 14.11 ms/item |
| verified `/tmp` copy, existing 32-row inner chunks | 4.73 ms/item |
| verified `/tmp` copy, disposable one-row inner-chunk encoding | 1.67 ms/item |

The disposable encoding took 23.2 seconds and was removed after the benchmark.
The existing local 32-row layout measured 4.46 ms/item in the directly paired
layout benchmark. The one-row layout therefore improved local random reads by
about 2.7×, while staging plus the future layout was about 8.4× faster than the
shared-storage source in these single-process measurements.

Physical-copy canaries completed and cleaned up for both retained layout kinds:

| Artifact | Compatibility | Bytes | Files | Physical content SHA-256 |
|---|---|---:|---:|---|
| recovered 192 crop export | explicit immutable | 204,381,969 | 78 | `6470a943b901177b96a6a65df1ae10d6fc06dc02c6308f255ba9c43848f5b2e9` |
| May production pose export | legacy closed export | 1,542,220,407 | 1,201 | `7fba6a47e6aa4493c655d94831daab4bff2b0e85db39078c541138eb8de6d2ce` |

No durable training dataset was written or mutated by these canaries.

## Validation

- 66 focused unit tests passed across storage planning, pose configuration,
  registry-backed preparation, loader behavior, staging, and runtime receipts.
- 4 real-Zarr merged materialized/recovered pose export integration tests
  passed outside the sandbox.
- Python compilation and `git diff --check` passed.

Repository-required remote CI remains required before integration, shared
checkout update, or any production activation.
