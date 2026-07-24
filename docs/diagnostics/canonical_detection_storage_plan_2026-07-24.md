# Canonical Detection Storage Planning Report

Status: Phase 3 planning input; not yet a production-writer layout

Date: 2026-07-24

## Result

The canonical detection schema now has one physical planning owner. Every array
is classified by access pattern and immutable lifecycle, while row depth is
derived from exact dtype, trailing record shape, and the
`published_http_v1` byte budgets. No writer-specific chunk or shard row count
is part of the stage definition.

For the representative Sleepyfish scale, all nine arrays use approximately
1 MiB uncompressed inner chunks and each complete array fits into one indexed
outer shard. The conservative estimate is nine payload objects and 20 total
stage objects after adding array and group metadata.

This is a planning result, not a promoted storage profile. Phase 4 has now
resolved `zstd_fast_v1` to an exact codec chain, added the shared array-creation
boundary, and completed an initial local regular-versus-sharded smoke. The full
chunk/shard byte sweep, controlled repetitions, request instrumentation, and
real Mac/VPN validation are still required.

## Representative Input

The report uses read-only metadata from:

```text
/groups/johnson/johnsonlab/jeremy/recordings/
sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/
sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr/
detect_runs/detect_2026-05-14_15-39-11
```

The run reports:

- acquisition frames: `1,188,000`;
- observed detections: `1,187,087`;
- source camera: `4512 x 4512`;
- current detection limit: `max_det=1`;
- current stored layout: Zarr v3 regular `1,024`-row chunks with
  `bytes + zstd(level=0, checksum=false)`.

Only the frame and detection cardinalities and source extent are planning
inputs. The historical run's one-detection limit, float64 normalized boxes,
missing vNext arrays, and physical layout are not carried into the canonical
contract.

The historical run has 5,809 physical files for its six arrays and run group.
Its logical chunk grids contain 6,962 payload positions; all-zero `class_ids`
chunks are fill-elided, which explains the lower physical count. These numbers
are contextual evidence, not a like-for-like benchmark against the new
nine-array schema.

## Access And Storage Classification

All row-aligned instance columns are `WINDOWED`. Their access unit is one
complete observation row, including all trailing coordinate components. Normal
reads are contiguous frame-row ranges obtained from the index or row windows
used by joins and validation.

`frame_row_offsets` is `EAGER`: Crimson may load the complete small index once,
while a future seek path may request two adjacent frame-boundary values. Its
physical access unit remains one offset element because adjacent frame pairs
overlap; an approximately 1 MiB inner chunk provides the caching unit.

All arrays are `IMMUTABLE`. Shards may combine chunks only along axis 0. A
parallel publisher must serialize array metadata creation, then assign every
complete outer shard to exactly one writer. Partial chunk and partial shard
writes are forbidden.

## Proposed Plans

Shapes use `N=1,187,087` detection rows and `F=1,188,000` frames. Byte sizes are
uncompressed logical sizes. Payload objects are conservative estimates before
fill-value elision.

| Path | Dtype / shape | Access | Logical MiB | Inner chunk | Outer shard | Inner chunks | Payload objects |
| --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `instances/frame_indices` | `int32 (N,)` | windowed | 4.53 | `(262144,)` = 1 MiB | `(1310720,)` = 5 MiB | 5 | 1 |
| `instances/source_acquisition_frame_index` | `int64 (N,)` | windowed | 9.06 | `(131072,)` = 1 MiB | `(1310720,)` = 10 MiB | 10 | 1 |
| `instances/instance_key` | `uint64 (N,)` | windowed | 9.06 | `(131072,)` = 1 MiB | `(1310720,)` = 10 MiB | 10 | 1 |
| `instances/bbox_norm_coords` | `float32 (N,4)` | windowed | 18.11 | `(65536,4)` = 1 MiB | `(1245184,4)` = 19 MiB | 19 | 1 |
| `instances/bbox_img_xyxy` | `float32 (N,4)` | windowed | 18.11 | `(65536,4)` = 1 MiB | `(1245184,4)` = 19 MiB | 19 | 1 |
| `instances/centers_img_xy` | `float32 (N,2)` | windowed | 9.06 | `(131072,2)` = 1 MiB | `(1310720,2)` = 10 MiB | 10 | 1 |
| `instances/scores` | `float32 (N,)` | windowed | 4.53 | `(262144,)` = 1 MiB | `(1310720,)` = 5 MiB | 5 | 1 |
| `instances/class_ids` | `int32 (N,)` | windowed | 4.53 | `(262144,)` = 1 MiB | `(1310720,)` = 5 MiB | 5 | 1 |
| `instances/frame_row_offsets` | `int64 (F+1,)` | eager + two-value seek | 9.06 | `(131072,)` = 1 MiB | `(1310720,)` = 10 MiB | 10 | 1 |

Totals:

- logical bytes: `90,225,924` (`86.05 MiB`);
- inner chunks addressable through shard indexes: `93`;
- outer-shard payload objects: `9`;
- array metadata objects: `9`;
- run and `instances` group metadata objects: `2`;
- conservative stage object total: `20`.

The shard shape can exceed the concrete logical extent on axis 0. That is
intentional: the edge shard is partial logically but still contains an integer
number of physical inner-chunk slots. At this scale, that yields one storage
object per array while preserving approximately 1 MiB range-decode units.

No representative array is small enough to use a single regular inner chunk.
For smaller runs, the shared planner intentionally leaves an array unsharded
when its complete logical extent fits in one chosen inner chunk; such an array
already has one payload object and gains nothing from indexed sharding.

## HTTP And Metadata Interpretation

Object count and request count are related but are not identical:

- direct metadata opening may require the run, `instances`, and nine array
  metadata objects;
- validated consolidated metadata should expose all of those records through
  one archive-root metadata request;
- each array is one payload object, but a windowed reader should issue a range
  request only for the necessary indexed inner chunk;
- an eager full-array read may require multiple ranges inside one shard unless
  the reader coalesces ranges or fetches the complete object.

Phase 4 must measure request count and transferred bytes rather than treating
the nine payload objects as proof of nine read requests.

## Implementation Boundary

Implemented now:

- exact access rules for every canonical detection path;
- schema-linked immutable `StoragePlan` generation;
- access-unit shape, growth axis, and shard axes in the serialized plan;
- conservative inner-chunk, shard, payload, metadata, and total-object counts;
- fail-closed checks that shards contain whole chunks and preserve complete
  trailing observation axes;
- JSON-safe planning manifests and deterministic representative tests;
- exact Zarr v3 `bytes(little) + zstd(level=0, checksum=false)` codec
  construction for regular chunks;
- exact Zarr v3 indexed-sharding construction with the same inner data codecs,
  `bytes + crc32c` index codecs, and the shard index at the end;
- a policy-owned array-creation boundary that accepts logical contracts and
  `StoragePlan` values rather than raw writer-specific chunk/shard literals;
- a safe disposable detection benchmark writer with exact digest validation,
  consolidated/direct opens, common benchmark envelopes, and physical file
  inventory.

Not implemented in this phase:

- production-writer adoption of the shared array factory;
- promotion of any benchmark codec/storage candidate to a production profile;
- publication or selector updates;
- production performance claims.

The production writer remains unchanged until the Phase 4 evidence gate passes.

Initial Phase 4 smoke evidence is recorded in
[`canonical_detection_storage_benchmark_smoke_2026-07-24.md`](canonical_detection_storage_benchmark_smoke_2026-07-24.md).
