# Sleepyfish Subject-Mask Storage And Sharding Strategy

**Date:** 2026-07-10
**Status:** current-state diagnostic and proposed benchmark plan; no storage
migration has been approved or applied

## Scope

This note records the measured storage state of the clipped-collection
subject-mask run for:

```text
sleepyfish_2026_05_05_17_45_30_cam2010095
```

The analysis archive is:

```text
/groups/johnson/johnsonlab/jeremy/recordings/
  sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/
  sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr
```

The note has two purposes:

1. distinguish logical tensor size from physical compressed storage;
2. propose a Zarr v3 indexed-sharding strategy that preserves current reads
   while reducing filesystem-object count.

It also records the related partial refined-finalizer failure because storage
sharding and compute sharding must not be conflated.

## Terminology

Three different meanings of "shard" are present in this workflow:

- **collection/input shard**: one clip-specific
  `subject_mask_shard_runs/<run>`;
- **compute shard**: one contiguous global row range assigned to a finalizer
  process;
- **Zarr storage shard**: one Zarr v3 storage object that contains multiple
  independently compressed inner chunks.

The current raw run uses collection shards and regular Zarr chunks. It does
not use the Zarr v3 sharding codec.

## Current Raw Probability Layout

The all-clips inference consists of `22` completed collection shards:

```text
subject_mask_shard_runs/
  subject_masks_sleepyfish_sm_allclips_20260708_02_clip_000000
  ...
  subject_masks_sleepyfish_sm_allclips_20260708_02_clip_000021
```

Together they contain `1,169,010` ROI rows. Each raw run stores:

```text
mask_probs_roi: [N, 3, 512, 512] uint8
mask_labels: [subject_body, eyes_union, swim_bladder]
probabilities_encoding: linear_uint8_0_255
chunk shape: [32, 1, 512, 512]
codec chain: bytes -> zstd(level=0)
```

One full, non-boundary chunk is `8 MiB` before compression:

```text
32 rows * 1 channel * 512 * 512 * 1 byte = 8 MiB
```

The complete logical probability surface is:

```text
1,169,010 * 3 * 512 * 512 bytes
= 919,346,872,320 bytes
= 856.21 GiB
```

That number is the decoded logical size, not the bytes occupied on disk.

### Representative Physical Measurement

For `clip_000000`, which has `54,000` rows:

| Measurement | Value |
| --- | ---: |
| Logical probability bytes | about `39.55 GiB` |
| Chunk files | `5,064` |
| Physical chunk payload bytes | `176,791,353` (`168.60 MiB`) |
| Minimum compressed chunk | `3,285` bytes |
| Median compressed chunk | `7,775` bytes |
| Mean compressed chunk | `34,911` bytes |
| 95th percentile compressed chunk | `112,538` bytes |
| Maximum compressed chunk | `341,656` bytes |
| Logical-to-physical ratio | about `240:1` |

The operator-observed clip footprint of roughly `200 MB` is consistent with
this measurement. An uncompressed outer tar does not expand these arrays:
each Zarr chunk is already compressed internally by Zstd.

Across all clips, the regular grid has `109,632` possible probability chunk
objects. Compression controls bytes well, but the number of small filesystem
objects remains high.

## Raw Mask Validity

The raw inference output itself is healthy:

| Component | Rows present | Coverage |
| --- | ---: | ---: |
| `subject_body` | `1,169,010` | `100.000%` |
| `eyes_union` | `1,168,980` | `99.997%` |
| `swim_bladder` | `1,169,004` | `99.999%` |
| any component | `1,169,010` | `100.000%` |
| all components | `1,168,974` | `99.997%` |

Direct reads found encoded probability values spanning `0..255`. The raw
writer's declared `linear_uint8_0_255` contract and the finalizer decoder's
`uint8 / 255` behavior agree. The raw probability run does not need inference
re-execution.

## Partial Refined Run Diagnostic

The corresponding refined run is:

```text
refined_subject_masks_runs/
  refined_subject_masks_sleepyfish_sm_allclips_20260708_02
```

Its dense shell has the expected logical shape:

```text
masks_roi: [1,169,010, 4, 512, 512] uint8
```

but only `40,448` rows contain any mask (`3.46%`). The nonempty rows form
exactly eight intervals:

```text
[0, 5,120)
[145,920, 151,040)
[292,096, 296,960)
[438,272, 443,392)
[584,448, 589,568)
[730,624, 735,744)
[876,800, 881,664)
[1,022,976, 1,028,096)
```

Those starts exactly equal the eight process-worker compute-shard boundaries.
Each worker wrote only `19` or `20` of its roughly `571` assigned `256`-row
chunks. The run has no completion contract/status, final timing summary, or
final row-count summary.

This is an interrupted finalizer publication, not probability-encoding drift.
The current implementation creates the visible refined run shell before
worker completion. A hard process or scheduler kill therefore leaves a
plausible-looking partial run in the canonical namespace.

### Whole-Collection Setup Inside Every Worker

At the time of the failed run, output computation was row-sharded but source
initialization was global. Every worker received all `22` collection-shard
names and independently:

1. opens all source groups;
2. reads the complete target crop identity arrays;
3. converts `1.17M` identities into nested Python tuples;
4. builds a global Python identity-to-row dictionary;
5. maps every source row into the target collection;
6. resolves collection-wide keypoint/mask row identity;
7. only then processes its assigned output rows.

The probability pixels remain lazy and chunked. The avoidable memory cost is
the repeated global routing state plus decoded chunk/component intermediates.
This is a likely memory amplifier for an eight-process, `32 GB` finalizer job.
The simultaneous stop could also reflect a queue runtime limit; retained LSF
accounting would be required to distinguish the external termination cause.

This routing amplifier has now been removed in the working tree. The parent
process performs the tuple/dictionary rebase once, converts the result into
compact unsigned numeric arrays, and passes that validated plan to workers.
Worker initialization still opens source groups but does not call the global
crop identity-map builder. Provenance records
`global_identity_map_builds=1` and `worker_identity_map_rebuilds=0`.

For the `1,169,010`-row sleepyfish collection, a real read-only dry plan produced
exactly `8,183,070` bytes (`7.80 MiB`): shard indices fit in `uint8`, clip-local
rows in `uint16`, and target crop rows in `uint32`. Eight copied worker payloads
are therefore roughly `62 MiB`, rather than eight Python tuple/dictionary
joins. The one-time parent plan completed in `18.16 s` with `812.6 MiB` process
peak RSS on `delahantyj-ws1`; that parent peak remains, but it is no longer
recreated independently inside every worker.
The virtual collection reader also now creates position indices only for the
requested slice; it no longer allocates a full `N`-row `arange` for every
`256`-row read.

### Eight-Worker Initialization Smoke

Read-only LSF jobs `153059271` and `153059276` exercised the real `22`-shard
collection on `h07u06.int.janelia.org`. Each job built the parent plan, started
eight workers, resolved the real refined-keypoint assignment context, read one
probability/keypoint row per worker, and held initialized state for `15 s`.
Neither job created a refined run or touched the registry.

| Measurement | Job 153059271 | Job 153059276 |
| --- | ---: | ---: |
| parent plan seconds | `15.31` | `14.35` |
| compact plan MiB | `7.80` | `7.80` |
| summed worker current RSS MiB | `3020.2` | `2994.0` |
| maximum individual worker RSS MiB | `380.1` | `377.2` |
| LSF process-tree maximum MiB | `2005` | `1985` |
| swap | none | none |

Median per-worker current RSS in the first smoke was `247.4 MiB` immediately
after fork, `344.6 MiB` after collection loading, `353.4 MiB` after keypoint
assignment, and `378.6 MiB` after the sampled probability read. All workers
reported keypoint identity mode `source_crop_row_ids_match`; no keypoint subset
dictionary was constructed. Summed per-process RSS double-counts shared pages,
which is why LSF's process-tree maximum was about `2 GiB` rather than `3 GiB`.

The second submission requested `2 GiB` per slot, but the site's `serial`
application/esub raised the effective eight-slot request to `120 GiB`
(`15 GiB` per slot). It therefore confirms measured usage, not enforcement of
a `16` or `32 GiB` limit. Even so, the approximately `2 GiB` process-tree
initialization footprint leaves substantial room for the mask and output
buffers that the finalizer A/B must measure next.

## Design Goals

The next raw-probability layout should:

- preserve exact `linear_uint8_0_255` values;
- retain efficient reads of one component over contiguous ROI rows;
- support the common `256`-row finalizer batch;
- reduce filesystem/object-store object count by at least an order of
  magnitude;
- keep parallel writers on non-overlapping physical write units;
- remain immutable after a raw inference run is marked complete;
- fail closed when publication or validation is interrupted.

Raw probability arrays are not display surfaces. Random single-row display
latency is therefore a diagnostic measurement, not a primary optimization
goal. The finalizer and validation readers consume them; display and editing
use refined masks.

## Proposed Zarr V3 Indexed-Sharding Layout

Keep the existing inner read chunk initially:

```text
inner chunk: [32, 1, 512, 512]
```

This shape matches current access:

- rows are processed in contiguous batches;
- components are finalized separately;
- the full `512 x 512` plane is normally needed.

Benchmark a small control plus substantially larger outer storage shards:

| Candidate | Outer shard | Inner chunks per shard | Uncompressed shard payload |
| --- | --- | ---: | ---: |
| A, control | `[512, 1, 512, 512]` | `16` | `128 MiB` |
| B | `[2048, 1, 512, 512]` | `64` | `512 MiB` |
| C | `[4096, 1, 512, 512]` | `128` | `1 GiB` |
| D | `[8192, 1, 512, 512]` | `256` | `2 GiB` |
| E, extreme | one padded clip-sized shard per channel | about `1,700` | about `13.2 GiB` |

Based on the measured clip-000000 mean compressed inner chunk, the average
encoded payload would be approximately `0.53`, `2.1`, `4.3`, or `8.5 MiB`
for candidates A through D. Actual size varies strongly with image content.

Each channel remains in a separate outer shard so component-only readers do
not touch the other two components. For a `54,000`-row clip, candidates A
through D require approximately `318`, `81`, `42`, or `21` probability payload
objects respectively, instead of about `5,064` regular chunk payloads. Across
`22` clips, candidates B through D would use roughly `1,782`, `924`, or `462`
payload objects instead of about `111k`.

This is a benchmark preference, not yet a schema default. The large decoded
size of an outer shard does not imply that readers decode it in full: indexed
sharding retains independent inner-chunk reads. It can, however, make writes
expensive. Zarr's efficient path is to write a complete storage shard at once;
incrementally updating inner chunks may repeatedly read and rewrite the shard.
Whole-shard encoder memory, temporary bytes, and write amplification must be
measured before promotion. Candidate E is included to expose implementation
limits, not as the expected default.

## Parallel Write Ownership

Zarr storage shards become the physical write-safety boundary.

For any sharded array:

- one worker must own every inner chunk in an outer storage shard;
- two workers must never update different inner chunks in the same outer
  shard concurrently;
- worker ranges must be rounded to outer-shard row boundaries;
- boundary remainders must have one deterministic owner;
- requested and effective worker chunking must be recorded in provenance.

Raw clipped-collection inference is a good candidate because each clip run is
write-once and normally has one owning inference process. Refined dense
`masks_roi` is not automatically a good candidate while it remains editable.
Dense refined masks should stay chunked during active review unless writes are
whole-shard aligned or occur through per-worker temporary outputs followed by
a deterministic merge.

For candidates B through E, the raw writer must not append one `32`-row batch
by repeatedly rewriting the same storage shard. Acceptable implementations
are:

- buffer or stage all encoded inner chunks for one storage shard and publish
  it once;
- write ordinary chunks to a temporary worker-local store, then pack them into
  indexed shards in one immutable publication pass;
- write one temporary output per worker and merge whole storage shards
  deterministically.

The temporary-store-and-pack option is the safest first implementation because
it preserves the existing inference writer and moves sharding into a validated
publication step.

## Collection Mapping Strategy

Storage sharding does not itself fix collection routing. The process finalizer
now computes the global mapping once and builds compact integer arrays:

```text
row_source_shard_index: [N] smallest fitting unsigned integer
row_source_local_index: [N] smallest fitting unsigned integer
source_crop_row_ids: [N] smallest fitting unsigned integer
```

Workers receive this plan and validate its schema, shard names, crop-run set,
row count, per-shard row counts, and index bounds. They do not rebuild Python
tuple/dictionary joins. A future shared-memory or row-sliced transport could
eliminate even the small numeric copy per worker, but that is no longer the
dominant memory risk. A more direct alternative remains per-clip finalization
into temporary outputs followed by an ordered, validated collection merge.

## Publication And Completion Rules

New finalizer runs must not become selectable merely because their array shell
exists.

Required lifecycle:

1. create a temporary or explicit `in_progress` run;
2. record the expected input shards, row count, inner chunks, and outer storage
   shards;
3. persist completed write-unit coverage;
4. validate all expected rows/components and source lineage;
5. validate compact/dense derived-store coherence;
6. stamp the strict completion contract;
7. promote or publish into the selectable namespace;
8. refresh registry projections only after promotion.

A killed job must leave an unselectable `in_progress` artifact. Registry
reconciliation must not report such a run as `ok` through legacy-completion
compatibility.

## Benchmark Matrix

Create new temporary arrays from one representative completed raw clip. Do not
rechunk the canonical run in place.

Benchmark:

| Layout | Inner chunk | Outer shard |
| --- | --- | --- |
| baseline | `[32,1,512,512]` | none |
| A | `[32,1,512,512]` | `[512,1,512,512]` |
| B | `[32,1,512,512]` | `[2048,1,512,512]` |
| C | `[32,1,512,512]` | `[4096,1,512,512]` |
| D | `[32,1,512,512]` | `[8192,1,512,512]` |
| E | `[32,1,512,512]` | one padded clip-sized shard per channel |

Measure:

- exact encoded byte equality after decode;
- physical bytes and storage-object count;
- full-clip sequential read throughput;
- `256`-row, one-component finalizer reads;
- one-row, one-component inspection latency;
- all-component one-row latency;
- write throughput;
- bytes rewritten per incremental input batch;
- temporary publication bytes;
- peak writer RSS;
- Zarr async/thread concurrency;
- tar creation, copy, extraction, recursive listing, and deletion time;
- recovery behavior after terminating a write mid-run.

Initial acceptance gates:

- exact probability equality;
- at least `20x` fewer probability storage objects;
- target roughly `100x` to `700x` fewer objects if candidates B through D
  satisfy resource gates;
- no more than `20%` regression in the dominant `256`-row component read;
- bounded writer RSS compatible with declared cluster memory;
- no repeated whole-shard rewrite for each inference batch;
- whole-storage-shard worker ownership proven in provenance;
- interrupted writes remain unselectable;
- final validation detects every missing output write unit.

## Rollout

1. Add a read/write benchmark using one copied clip.
2. Select an outer shard size from measured PRFS behavior.
3. Add explicit storage-layout attrs and provenance.
4. Add deterministic whole-shard worker partitioning.
5. Add strict incomplete/complete publication gates.
6. Enable sharding for new immutable raw probability runs only.
7. Validate one complete clipped collection.
8. Consider optional migration of old runs by copy-and-validate; never rechunk
   canonical arrays in place.
9. Evaluate frozen refined outputs separately from active editable outputs.

## Initial 8,192-Row Benchmark Set

The first real-data fixture was generated from rows `[0, 8192)` of
`clip_000000`. Each layout was built in a separate process and validated by an
exact SHA-256 digest over decoded `uint8` values.

Fixture location:

```text
/tmp/palette_sleepyfish_probability_sharding_benchmark
```

| Layout | Files | Stored MiB | Write seconds | Peak RSS MiB | 256-row component read MiB/s | Random row ms/read | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| regular chunks | `770` | `24.90` | `23.99` | `414.7` | `5262.6` | `2.65` | yes |
| 512-row shards | `50` | `24.91` | `14.16` | `648.1` | `2897.6` | `2.91` | yes |
| 2,048-row shards | `14` | `24.91` | `10.48` | `1479.5` | `2875.2` | `2.67` | yes |
| 4,096-row shards | `8` | `24.91` | `10.70` | `2524.4` | `2788.6` | `3.50` | yes |
| 8,192-row shards | `5` | `24.91` | `11.25` | `4580.1` | `2853.5` | `3.64` | yes |

The file count includes Zarr metadata; the 8,192-row probability array itself
uses three outer storage shards, one per channel.

These are fixture-generation measurements, not final performance claims. The
read timings in this table were collected only once while building each
variant. The first result nevertheless establishes:

- indexed sharding preserves exact probabilities;
- physical bytes are essentially unchanged;
- object count falls sharply;
- write time improved for this one-shot pack path;
- peak RSS grows approximately with outer-shard row count;
- 8,192 rows is feasible but consumes about `4.5 GiB` in one process;
- 2,048 to 4,096 rows is the more conservative production benchmark region.

## Repeated Local Read Benchmark

The same fixture was read for seven randomized rounds from local NVMe-backed
ext4. Before each cold pass, the harness requested eviction of every variant
file from the Linux client page cache with `POSIX_FADV_DONTNEED`, then opened
the Zarr fresh. The warm pass immediately repeated the same scan. Each pass
read component zero in `256`-row batches and materialized `2 GiB` of logical
`uint8` data. Cache advice completed without errors in every round.

| Layout | Cold median MiB/s | Cold median seconds | Warm median MiB/s | Warm median seconds |
| --- | ---: | ---: | ---: | ---: |
| regular chunks | `2205.5` | `0.929` | `2239.7` | `0.914` |
| 512-row shards | `1425.4` | `1.437` | `1683.8` | `1.216` |
| 2,048-row shards | `1416.3` | `1.446` | `1656.2` | `1.237` |
| 4,096-row shards | `1501.1` | `1.364` | `1613.7` | `1.269` |
| 8,192-row shards | `1477.7` | `1.386` | `1629.0` | `1.257` |

Regular chunks were about `30%` faster by median throughput than the best
sharded layout in this local bulk-read test. This is plausible on fast local
storage: regular chunks take a direct file-per-chunk path, while indexed
sharding adds shard-index lookup and byte-range extraction around the same
inner-chunk decoding. The object-count advantage is not the bottleneck in this
scan. The four sharded sizes are comparatively close and their rank changed
between independent pilot and finalized runs. There is no supported
size-dependent read-throughput winner; increasing shard size mainly increases
writer RSS.

The `2,048`-row layout is therefore the preferred sharded candidate from this
local screen, but not because of a statistically distinct read advantage. It
gives about `63x` fewer payload objects for a typical clip, had the fastest
one-shot write, and uses substantially less peak RSS than `4,096` or `8,192`
rows. The `512`-row candidate does not meet the initial `20x` object-reduction
gate. The `2,048`-row candidate is not yet the production default: it does not
clear the current `20%` read-regression gate on local NVMe, and the decisive
follow-up must repeat the benchmark on PRFS where small-object and metadata
costs differ.

`POSIX_FADV_DONTNEED` is advisory. It requests eviction only from the client
kernel's file-page cache; it cannot guarantee eviction from device,
filesystem-server, or storage-controller caches. These measurements therefore
represent repeatable local cold-ish and warm behavior, not power-on cold reads
or PRFS performance.

Reproduce the repeated read benchmark with:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_mask_probability_sharding_reads \
  /tmp/palette_sleepyfish_probability_sharding_benchmark \
  --repeats 7 \
  --batch-rows 256 \
  --component 0 \
  --random-seed 20260710
```

## PRFS-Native Benchmark Mode

A native PRFS benchmark must create each Zarr directly on PRFS. Copying a
completed local fixture to PRFS does not measure per-chunk or per-shard object
creation, directory inventory, or native write behavior. The benchmark tools
therefore inspect `/proc/self/mountinfo` and can fail closed unless the target
resolves to an NFS mount whose server has a `prfs` DNS label. The exact mount
source is retained in provenance.

Each variant summary now records source and destination mount point,
filesystem type, mount source, host, recursive inventory duration, and storage
tier. The repeated reader additionally records fresh Zarr metadata-open latency
and cache-advice traversal duration separately from array scan time. Cache
advice remains a cold-ish client hint, not proof that server-side PRFS caches
were flushed.

### Workstation-to-PRFS Result

The same five variants were created directly on PRFS from
`delahantyj-ws1.hhmi.org`. This path read the source sample from PRFS and wrote
the destination to PRFS, so write time is an end-to-end workstation PRFS result
rather than an isolated destination-write measurement.

| Layout | Files | Direct PRFS write seconds | Peak RSS MiB | Exact |
| --- | ---: | ---: | ---: | --- |
| regular chunks | `770` | `36.77` | `441.4` | yes |
| 512-row shards | `50` | `15.49` | `673.4` | yes |
| 2,048-row shards | `14` | `10.72` | `1479.9` | yes |
| 4,096-row shards | `8` | `10.57` | `2505.2` | yes |
| 8,192-row shards | `5` | `10.73` | `4563.1` | yes |

Seven randomized direct PRFS-to-workstation read rounds then produced:

| Layout | Cold-ish median MiB/s | Warm median MiB/s | Cache-advice traversal seconds |
| --- | ---: | ---: | ---: |
| regular chunks | `2660.7` | `2761.7` | `4.549` |
| 512-row shards | `1036.5` | `1061.2` | `0.309` |
| 2,048-row shards | `380.3` | `1039.0` | `0.080` |
| 4,096-row shards | `388.6` | `1072.5` | `0.053` |
| 8,192-row shards | `321.3` | `1070.6` | `0.031` |

Fresh metadata-open medians were approximately `8.5` to `9.0 ms` for every
layout. Regular chunks retained the fastest array scans, while larger indexed
shards incurred a strong first-pass penalty and all sharded warm scans
converged near `1 GiB/s`. Conversely, recursively visiting files for cache
advice took `4.55 s` for the regular layout and at most `0.31 s` for a sharded
layout. Array throughput and metadata-management cost must therefore remain
separate metrics.

These measurements are workstation-client results. Compute nodes mount the
same service through `cluster.prfs.janelia.org`, rather than the workstation's
`prfs.hhmi.org` endpoint, so cluster performance must be measured separately.

Build each variant in a separate process so peak RSS remains attributable to
that layout:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_mask_probability_sharding \
  /path/to/subject_mask_shard_runs/<run> \
  --output-root /groups/.../.palette_benchmarks/probability_sharding_20260710 \
  --layout sharded \
  --shard-rows 2048 \
  --sample-rows 8192 \
  --inner-chunk-rows 32 \
  --batch-rows 256 \
  --random-read-count 32 \
  --require-destination-storage-tier prfs \
  --overwrite
```

After building the regular control and all sharded variants in the same PRFS
directory, run:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_mask_probability_sharding_reads \
  /groups/.../.palette_benchmarks/probability_sharding_20260710 \
  --repeats 7 \
  --batch-rows 256 \
  --component 0 \
  --random-seed 20260710 \
  --require-storage-tier prfs
```

For the compute-node comparison, use
`scripts/submit_subject_mask_probability_sharding_benchmark_bsub.sh`. Its job
first runs the direct PRFS reader, then stages `regular.zarr` to verified local
scratch and writes all five layouts back to a unique PRFS output. Both source
and destination tiers are fail-closed, every output is digest-checked, and
each write layout runs in a separate process. LSF job `153059133` is the first
corrected sleepyfish run of this protocol; job `153059132` exited before reads
because the initial classifier did not yet recognize the compute-node PRFS DNS
name.

### Compute-Node PRFS Result

Corrected LSF job `153059133` completed successfully on
`h07u08.int.janelia.org` in `549 s`, with `4,077 MiB` maximum job memory and no
swap. It used the same seven-round read protocol as the workstation run.

| Layout | Cold-ish median MiB/s | Warm median MiB/s | Metadata open ms | Cache-advice traversal seconds |
| --- | ---: | ---: | ---: | ---: |
| regular chunks | `1636.5` | `1661.4` | `2.32` | `1.378` |
| 512-row shards | `888.7` | `913.0` | `2.45` | `0.084` |
| 2,048-row shards | `893.8` | `907.0` | `2.34` | `0.029` |
| 4,096-row shards | `891.1` | `898.8` | `2.37` | `0.016` |
| 8,192-row shards | `889.5` | `901.4` | `2.39` | `0.011` |

Unlike the workstation client, the compute node showed no large-shard
first-pass penalty: all four sharded sizes were effectively equivalent. Regular
chunks remained about `1.8x` faster for the dominant component scan. Sharding
reduced full-tree cache-advice traversal by `16x` to `125x`, demonstrating a
large metadata-management benefit that is separate from array throughput.

The job then copied the exact regular fixture from PRFS to local scratch in
`1.01 s` and wrote each variant from verified local storage to a new PRFS
directory:

| Layout | Files | Scratch-to-PRFS write seconds | Inventory seconds | Peak RSS MiB | Exact |
| --- | ---: | ---: | ---: | ---: | --- |
| regular chunks | `770` | `42.88` | `3.006` | `354.4` | yes |
| 512-row shards | `50` | `30.93` | `0.206` | `484.8` | yes |
| 2,048-row shards | `14` | `35.41` | `0.059` | `1239.0` | yes |
| 4,096-row shards | `8` | `31.54` | `0.032` | `2236.6` | yes |
| 8,192-row shards | `5` | `32.95` | `0.021` | `4248.7` | yes |

All five variants matched SHA-256
`a9c8d9c7e15fc40d894206b2a2b21627d80cc7162862f4e07f2657fb1197c09d`.
Decode validation took approximately `36 s` per variant and was recorded
separately from write time.

The compute result preserves `2,048` rows as the balanced **sharded candidate**
for the next end-to-end finalizer test, not as an approved production default.
It gives about `63x` payload-object reduction for a typical clip with only
`1.24 GiB` writer RSS. The `512`-row layout misses the initial `20x`
object-reduction gate, while `4,096` and `8,192` rows spend an additional `1.0`
and `3.0 GiB` of writer memory without improving reads. A `4,096`-row post-pack
layout remains a reasonable low-concurrency alternative if minimizing PRFS
objects outweighs worker memory.

All sharded layouts are approximately `45%` slower than regular chunks for the
dominant compute-node component scan, so they fail the current `20%` read
regression gate. Promotion requires an end-to-end finalizer comparison to
determine whether lower PRFS object-management and publication costs offset
that array-scan regression in the real workflow.

### Contract-Complete Finalizer A/B Fixture

LSF job `153059726` built the full `54,000`-row `clip_000000` fixture on
`h07u08.int.janelia.org` under:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
  sleepyfish_finalizer_ab_clip000000_20260711/
    regular.zarr
    shard_02048.zarr
    fixture_manifest.json
```

Both Zarrs contain independent, contract-complete copies of the rebased crop
lineage, refined-keypoint assignment rows, subject-mask row metadata, channel
availability, thresholds, and exact probabilities. Their fixture-local crop
and keypoint row IDs are identical. Only the physical probability layout
differs.

| Fixture | Probability layout | Total files | Stored MiB | Probability write seconds |
| --- | --- | ---: | ---: | ---: |
| regular | chunks `[32,1,512,512]` | `5,162` | `173.34` | `427.12` |
| sharded | inner `[32,1,512,512]`, outer `[2048,1,512,512]` | `179` | `173.42` | `282.33` |

The total file counts include identical crop, keypoint, subject-row, group, and
array metadata. The probability values in the source order and both
destinations all match SHA-256
`fd3bcf09c5e73d53925a3db4d8d1f6784a027a2298e440564b5402bb73cab97b`.
Both fixtures report keypoint identity mode `source_crop_row_ids_match` over all
`54,000` rows and produce identical process-shards finalizer dry-run plans for
`subject_body`, `eye_left`, `eye_right`, and `swim_bladder`.

Fixture construction completed in `1,132 s` with `1,648 MiB` LSF maximum
memory, `1,725,388 KiB` driver peak RSS, and no swap. The source analysis Zarr
and registry were read-only. This fixture is ready for the position-balanced
end-to-end finalizer A/B.

### Position-Balanced End-to-End Finalizer A/B

LSF job `153061487` ran four complete `54,000`-row finalizations directly
against the PRFS fixtures on `h07u30.int.janelia.org`. The order was regular,
sharded, sharded, regular, so each physical layout occupied each execution
position once. All runs used eight `process_shards` workers, `256` logical rows
per worker chunk, `128` rows per physical dense output chunk, cheap metrics,
dense `uint8` refined masks, and no optional contours or eye geometry.

```text
/groups/johnson/johnsonlab/jeremy/recordings/logs/
  subject_mask_finalizer_layout_ab/
    sleepyfish_finalizer_layout_ab_clip000000_20260711_02/
      reports/summary.json
```

| Execution position | Layout | Finalizer seconds | Process wall seconds | Peak process-tree RSS GiB |
| ---: | --- | ---: | ---: | ---: |
| 1 | regular | `338.32` | `376.22`* | `10.66` |
| 2 | sharded | `334.41` | `338.10` | `9.40` |
| 1 | sharded | `334.98` | `340.04` | `9.70` |
| 2 | regular | `331.99` | `336.04` | `9.97` |

The regular median finalizer time was `335.15 s`; the sharded median was
`334.69 s`. Sharding was therefore `0.14%` faster, which is operationally
indistinguishable from no runtime difference. The order-specific comparisons
swung from sharded `1.16%` faster to sharded `0.90%` slower. Median process-tree
RSS was `10.31 GiB` regular versus `9.55 GiB` sharded, a `7.4%` reduction.
CPU time differed by less than `1%`.

`*` The first regular process wall includes removal of an incomplete output
left by an earlier wrapper attempt whose final computation succeeded but whose
completion gate correctly rejected missing Git provenance. Internal finalizer
time is not affected. The corrected job executed from a clean node-local
checkout of commit `ba9e4d3` and all four runs completed normally.

After the timed runs, an exhaustive logical comparison read every corresponding
output element. Both repeat pairs contained `125` arrays per run and reported
zero mismatches. Validation took `738.38 s` and was excluded from finalizer
timing. The complete LSF job used `9,558 MB` maximum memory, no swap, and
finished successfully in `2,140 s` including validation.

This end-to-end result clears the final runtime gate for the `2,048`-row
probability-sharding candidate. Together with the approximately `63x`
probability-payload object reduction, unchanged stored size, faster fixture
construction, and exact output parity, it supports using `2,048`-row indexed
shards for new read-only probability-mask stores. A full collection canary
should still precede any bulk rewrite of the existing `22` raw shards.

From a workstation without `bsub`, submit through the configured login host:

```bash
scripts/submit_subject_mask_probability_sharding_benchmark_bsub.sh \
  --input-root /groups/.../.palette_benchmarks/probability_sharding_20260710 \
  --submit-host login1-citrus-poller
```

Reproduce one variant with:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_mask_probability_sharding \
  /path/to/subject_mask_shard_runs/<run> \
  --output-root /tmp/palette_sleepyfish_probability_sharding_benchmark \
  --layout sharded \
  --shard-rows 4096 \
  --sample-rows 8192 \
  --inner-chunk-rows 32 \
  --batch-rows 256 \
  --random-read-count 32 \
  --overwrite
```

## External Guidance

- Zarr recommends selecting chunk shape from the access pattern and notes that
  at least roughly `1 MB` uncompressed chunks often perform better:
  <https://zarr.readthedocs.io/en/latest/user-guide/performance/>.
- Zarr v3 sharding separates the independently readable inner chunk from the
  coarser physical write/storage shard:
  <https://zarr.readthedocs.io/en/latest/user-guide/arrays/#sharding>.
- The indexed-sharding specification was designed to reduce small-object
  overhead without forcing inefficiently large read chunks:
  <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/>.

## Immediate Operational Conclusion

- Keep the `22` completed raw probability shards; they are valid and compact
  on disk.
- Treat `refined_subject_masks_sleepyfish_sm_allclips_20260708_02` as failed or
  incomplete, not authoritative.
- Recover by rerunning finalization from the existing raw shards into a new run
  after the collection-mapping and publication risks are addressed.
- Do not rerun GPU inference merely to recover this refined output.
