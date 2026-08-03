# Analysis-array byte-planning adapter (2026-08-03)

## Scope

`fisheye.shared.zarr.analysis_storage_planning` is the read-only policy bridge
between an exact maintained `AnalysisArrayDeclaration` and Palette's existing
`ArrayIntent -> StoragePlan` planner. It does not create Zarr arrays, select or
promote a storage profile, change a writer, activate a selector, or mutate a
registry.

The caller must supply:

1. an exact declaration (the Python object or its canonical JSON entry);
2. observed shape and fixed-width dtype facts;
3. an explicit statement of the logical access-unit semantics; and
4. a `StorageProfile` chosen by the caller's benchmark/publication policy.

The adapter validates the logical contract first. A dtype, fixed extent,
symbolic extent, rank, path, or canonical-manifest mismatch fails before the
physical planner runs.

## Byte and access-unit rule

The planner input uses the observed fixed-width NumPy dtype item size and the
full trailing shape of one logical row. For a growth axis of zero:

```text
access_unit_shape = (1, *max(1, observed_shape[1:]))
bytes_per_access_unit = dtype.itemsize * product(access_unit_shape)
```

Thus the same byte profile produces different row counts without per-array
constants:

| Logical record | Bytes per record | Approximate 1 MiB inner chunk |
|---|---:|---:|
| `bool[N]` | 1 | 1,048,576 rows |
| `int64[N]` | 8 | 131,072 rows |
| `float32[N,5,2]` | 40 | 32,768 rows (1.25 MiB, nearest allowed power of two) |
| `float32[N,2]` indexed points | 8 | 131,072 rows |

The adapter restricts outer sharding to the growth axis. A keypoint row, vector
row, fixed-width text row, or indexed point is never split along its trailing
axes merely to approach a byte target. Eager arrays beneath the profile's eager
cap remain one regular chunk. Empty arrays retain their row contract while
estimating zero payload objects and one array-metadata object.

## Lifecycle and write ownership

The declaration's `write_mode` is carried unchanged into `ArrayIntent`.
Immutable declarations are explicitly classified as
`immutable_snapshot_array`, making them eligible for indexed sharding under a
profile that enables immutable sharding. The resolved plan continues to require
`whole_shard_single_writer` for sharded output. This module does not authorize
parallel writes: writers adopting a receipt must still partition work on whole,
non-overlapping physical shards as required by `docs/dask_zarr_write_safety.md`.

## Receipt

`plan_analysis_storage(...)` returns a deterministic receipt containing:

- the complete caller-supplied storage profile;
- resolved shared symbolic dimensions;
- the exact logical declaration and observed facts for every present array;
- the resolved `StoragePlan`, including chunk and shard shapes;
- logical bytes, inner-chunk counts, payload-object estimates, per-array
  metadata-object counts, and sharded/empty array counts; and
- a canonical SHA-256 digest over the receipt payload.

Object estimates cover array payload objects and one `zarr.json` per array.
They deliberately exclude group/root metadata because a set of declarations
does not establish the final group tree. Payload counts are deterministic
shape-derived populated-object upper bounds; compression ratios and fill-value
elision are not guessed.

## Adoption checklist for a later writer change

- [ ] Freeze the analysis family's exact declaration and actual dimensions.
- [ ] Classify each array's access pattern and state its access-unit semantics.
- [ ] Run the adapter with an explicit candidate profile; do not use an implicit
      default.
- [ ] Persist and validate the digest-bound receipt in the run manifest.
- [ ] Create every array from the receipt rather than separate row constants.
- [ ] Ensure every parallel worker owns complete, non-overlapping physical
      chunks or shards.
- [ ] Compare direct and consolidated metadata after immutable publication.
- [ ] Benchmark representative eager, windowed, indexed, and whole-run reads.
- [ ] Promote a versioned profile only after producer and consumer gates pass.

No production family adopts a profile merely because this adapter exists.
