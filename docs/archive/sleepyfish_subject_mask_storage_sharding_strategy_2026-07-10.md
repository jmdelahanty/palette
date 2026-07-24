# Sleepyfish Subject-Mask Storage And Sharding Strategy

**Date:** 2026-07-10
**Last updated:** 2026-07-12
**Status:** diagnostic, implementation, finalizer parity, full-collection
finalizer, post-pack inference, and double-buffered direct-write canaries
complete; the `2,048`-row direct-write layout is now the default for new raw
probability runs, existing runs remain unchanged, and no canonical storage
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

The note has three purposes:

1. distinguish logical tensor size from physical compressed storage;
2. record the collection-worker memory-amplifier fix;
3. select and validate a Zarr v3 indexed-sharding candidate that preserves
   finalizer behavior while reducing filesystem-object count.

It also records the related partial refined-finalizer failure because storage
sharding and compute sharding must not be conflated.

The implementation is recorded in four commits on branch `sun`:

- `ba9e4d3` — `Optimize subject-mask collection finalization`;
- `081c766` — `Add subject-mask finalizer layout benchmark`;
- `e7ad962` — `Default subject-mask derived outputs`;
- `cb78692` — `Benchmark complete subject-mask finalization matrix`.

The principal implementation and reproduction surfaces are:

- `src/fisheye/refinement/finalize_subject_masks.py` — compact parent-built
  collection worker plan and bounded collection slicing;
- `src/fisheye/diagnostics/benchmark_subject_mask_collection_worker_init.py`
  — read-only real-collection worker initialization diagnostic;
- `src/fisheye/diagnostics/benchmark_subject_mask_probability_sharding.py`
  and `benchmark_subject_mask_probability_sharding_reads.py` — layout
  construction and repeated read benchmarks;
- `src/fisheye/diagnostics/build_subject_mask_finalizer_ab_fixture.py` — exact
  contract-complete regular/sharded fixture builder;
- `scripts/submit_subject_mask_collection_worker_init_bsub.sh`,
  `submit_subject_mask_probability_sharding_benchmark_bsub.sh`,
  `submit_subject_mask_finalizer_ab_fixture_bsub.sh`, and
  `submit_subject_mask_finalizer_layout_ab_bsub.sh` — cluster wrappers;
- `scripts/submit_subject_mask_complete_finalizer_matrix_bsub.sh` — complete
  default-output 8/16-worker finalizer and publication matrix;
- `tests/unit/fisheye/test_finalize_subject_masks.py` and the two probability
  sharding benchmark test modules — deterministic regression coverage.

The focused finalizer module completed with `43 passed in 50.57 s`; the staged
patches also passed `git diff --check` and shell syntax validation.

All fixture construction and finalizer A/B writes were confined to
`.palette_benchmarks` and benchmark log directories. The canonical analysis
Zarr, its `22` raw subject-mask shards, and the registry were not modified by
the storage-layout benchmark.

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

This routing amplifier was removed in commit `ba9e4d3`. The parent
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
buffers. The later end-to-end finalizer A/B measured approximately `9.4` to
`10.7 GiB` peak process-tree RSS for a complete `54,000`-row clip.

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

## Initial Post-Pack Inference Writer

The first production-path implementation is now available as an explicit
opt-in:

```text
--mask-probs-chunk-rois 32 --mask-probs-shard-rois 2048
```

The U-Net writer does not incrementally update indexed shards. It writes
ordinary inference batches to the private `_mask_probs_roi_working` array,
waits for the asynchronous output queue to drain, and then copies complete
outer-shard row ranges into the canonical `mask_probs_roi` array. It computes
SHA-256 digests over all decoded source and destination probability bytes and
removes the working array only after the digests match. A mismatch raises
before run completion and preserves the working array for diagnosis.

Successful runs record `mask_probs_storage_layout=indexed_sharding_v1`, the
requested shard rows, inner and outer shapes, pack and validation durations,
and both digests in attrs and run provenance. Unsharded runs retain
`regular_chunks_v1`. Dense `masks_roi` remains ordinarily chunked; sharding is
limited to immutable raw probabilities.

The option is propagated through the recording batch runner, clipped-
collection planner, and LSF submission wrapper. Because the standard batch
workflow writes inference output on node-local scratch before atomic PRFS
publication, the temporary ordinary chunks do not increase PRFS object count.
Only the validated sharded run group is copied to PRFS. This should reduce
publication metadata/object overhead substantially, although end-to-end copy
time still requires a production canary. Direct-to-PRFS invocations remain
supported but perform both the working write and post-pack on PRFS.

At this stage the `2,048`-row layout remained opt-in until a complete clipped-
inference canary passed decoded-value, lineage, completion, object-count, and
publication-time checks. Existing completed probability runs were never
rewritten in place.

The first full-clip inference canary below passed those integrity and storage
gates but showed that the separate pack plus two-surface digest added `413.6 s`.
This post-pack implementation is retained only as a historical compatibility
helper; new sharded inference uses the double-buffer candidate below.

## Double-Buffered Direct Writer Candidate

The next implementation keeps exactly two channel-major host buffers, each
covering one `2,048`-row outer shard. The inference output thread fills one
buffer while a single shard writer hashes and writes the other. Each physical
`[2048,1,512,512]` channel shard is written once, so logical inference batches
never trigger partial-shard read-modify-write behavior.

For the three-channel `uint8` contract, each buffer is `1.5 GiB` and the two
buffers reserve `3.0 GiB`. The writer records both shapes and byte counts in
`mask_probs_shard_write`. It computes one SHA-256 per channel from the buffer
before reuse, rereads the completed destination once, computes the same
per-channel digests, and completes the run only when every digest matches.
This removes the full ordinary-chunk working array, the separate pack phase,
and the redundant source validation pass while preserving a full decoded-byte
storage check.

The repeated `54,000`-row canary using the same model, crop proxy, cache,
cluster queue, staging path, and publication gates as job `153064680` passed.
The measured result is recorded below and supported the production-default
rollout recorded after the canary.

## Full-Clip Sharded-Inference Canary

LSF job `153064680` ran the complete `54,000`-row `clip_000000` inference on
`h08u16.int.janelia.org` with the production batch path: the `14.2 GB` flat ROI
cache and inference outputs were staged to node-local scratch, probabilities
used inner chunks `[32,1,512,512]` and outer shards `[2048,1,512,512]`, and only
the completed sharded run was atomically copied to PRFS.

Published run:

```text
subject_mask_shard_runs/
  subject_masks_unet_registry_sleepyfish_prob_shard_canary_clip000000_20260712_04
```

Evidence:

```text
/groups/johnson/johnsonlab/jeremy/recordings/logs/
  subject_mask_probability_sharded_inference_canary/
  sm_sleepyfish_prob_shard_canary_clip000000_20260712_04/
```

Integrity and contract gates passed:

- completion status is `complete`; shard outputs remain selector-ineligible for
  canonical recording-level resolution and registry refresh remains deferred;
- the private working array was removed only after validation;
- source and destination decoded SHA-256 are identical:
  `fd3bcf09c5e73d53925a3db4d8d1f6784a027a2298e440564b5402bb73cab97b`;
- this digest is also the previously established full-clip source, regular-
  fixture, and sharded-fixture digest, proving exact probability parity with
  the prior raw run;
- all eight row-lineage arrays match both the proxy crop and prior raw run;
- all seven metric arrays match the prior raw run exactly;
- all `54,000` rows have at least one present component, and sampled rows span
  the full stored probability range `0..255` in every channel.

Storage and publication result:

| Measure | Prior regular run | New sharded run |
| --- | ---: | ---: |
| Stored size | `173 MiB` | `172 MiB` |
| Probability files | `5,065` | `82` |
| Total run files | `6,646` | `1,663` |

Probability object count fell `61.8x` (`98.4%`) while stored bytes remained
effectively unchanged. The production publisher scanned the staged run in
`0.096 s`, copied `1,663` files / `179,533,272` apparent bytes in `16.53 s`,
and completed the publish phase in `21.79 s`; atomic commit took `0.003 s`.

The performance gate is not yet sufficient to make this layout the default.
The prior regular raw run completed in `509.2 s`. The sharded canary's raw
stage took `883.9 s`, including `130.9 s` to pack shards and `282.6 s` to
reread and SHA-256 validate both complete decoded surfaces. LSF wall time was
`959 s`, CPU time `1,155 s`, maximum accounted memory `2,022 MB`, and no swap.
The object/publication improvement is decisive, but adding `413.6 s` of local
post-processing is too large to enable unconditionally.

The next writer optimization should compute the source SHA-256 incrementally
from the values already read during the shard-copy loop, then retain one full
post-write destination digest as the fail-closed storage check. This removes
one redundant decoded source pass without weakening exact validation. Repeat
the full-clip inference canary after that change before promoting sharding to a
production default.

## Double-Buffered Direct-Write Canary

LSF job `153064710` repeated the complete `54,000`-row `clip_000000` inference
on `h08u16.int.janelia.org`. It reused the same durable `14.2 GB` flat ROI
cache built by job `152007051`, staged that cache to compute-node scratch, and
used the same model, crop proxy, `gpu_l4` queue, batch size, node-local output
staging, and atomic PRFS publisher as the post-pack canary. It did not rebuild
the ROI cache and did not run inference on the login node.

Published run:

```text
subject_mask_shard_runs/
  subject_masks_unet_registry_sleepyfish_prob_shard_doublebuf_clip000000_20260712_01
```

Evidence:

```text
/groups/johnson/johnsonlab/jeremy/recordings/logs/
  subject_mask_probability_sharded_inference_canary/
  sm_sleepyfish_prob_shard_doublebuf_clip000000_20260712_01/
```

The writer reserved exactly two `[3,2048,512,512]` `uint8` buffers: `1.5 GiB`
each and `3.0 GiB` total. It wrote `26` complete row shards plus one partial
row shard. The run completed without creating `_mask_probs_roi_working`; LSF
maximum accounted memory was `4 GiB`, with no growth as successive buffers
were reused.

Integrity and contract gates passed:

- completion status is `complete` and publication validation is `ok`;
- the source and destination per-channel SHA-256 values are identical, and the
  aggregate `sha256_per_channel_then_sha256_v1` digest matches exactly:
  `e2db8728d79fae65fed2d3b552a61b3daad6c61ef317d604870796047f7641fc`;
- all eight row-lineage arrays and all seven metric arrays match the prior
  post-pack canary exactly;
- sampled probabilities on rows crossing `32`-row inner-chunk and `2,048`-row
  outer-shard boundaries match the prior run exactly, contain `37,051` nonzero
  values, and span the stored range `0..255`;
- all `54,000` rows report at least one nonempty component.

Storage and publication were unchanged from the successful post-pack layout:
the run occupies `172 MiB`, `mask_probs_roi` has `81` payload shard files plus
one metadata file, and the complete run has `1,663` files. The publisher copied
`179,538,078` apparent bytes in `17.25 s` and completed its publish phase in
`22.84 s`.

| Raw inference writer | Duration | Delta from regular | LSF max memory |
| --- | ---: | ---: | ---: |
| Prior regular chunks | `509.2 s` | baseline | not remeasured |
| Separate post-pack | `883.9 s` | `+374.7 s` (`+73.6%`) | `2.0 GiB` |
| Direct two-buffer shards | `628.2 s` | `+119.0 s` (`+23.4%`) | `4.0 GiB` |

Direct double-buffering recovered `255.7 s` (`28.9%`) versus post-pack. Its
full destination reread and digest took `150.5 s`; subtracting that terminal
validation leaves `477.7 s` for inference plus overlapped shard output, below
the `509.2 s` regular baseline. Thus direct shard construction itself is not
the remaining slowdown. The explicit fail-closed full reread is now the only
material premium and is responsible for more than the total `119.0 s` net
regression.

This canary clears the implementation, memory, exact-write, content,
lineage, storage-object, and atomic-publication gates. Enabling `2,048`-row
shards by default for new immutable raw-probability shard runs is now a rollout
choice, not a correctness blocker. Existing completed runs must remain
unchanged.

## Default Rollout Decision

On 2026-07-12 the `2,048`-row double-buffered indexed-sharding layout became
the default for new raw U-Net probability outputs. The direct inference CLI,
recording batch runner, clipped-collection planner, and LSF batch wrapper all
resolve to the same default. `--mask-probs-shard-rois` remains available for an
explicit alternative valid shard size, while `--no-mask-probs-sharding`
selects ordinary chunks for compatibility or diagnostics.

Runs record the effective layout plus `mask_probs_storage_policy` and
`mask_probs_default_shard_rois=2048` in attrs and provenance. The default does
not migrate, rechunk, or otherwise mutate any completed historical run.

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

At this intermediate stage, the compute result preserved `2,048` rows as the
balanced **sharded candidate** for the then-pending end-to-end finalizer test,
not as an approved production default. It gives about `63x` payload-object
reduction for a typical clip with only
`1.24 GiB` writer RSS. The `512`-row layout misses the initial `20x`
object-reduction gate, while `4,096` and `8,192` rows spend an additional `1.0`
and `3.0 GiB` of writer memory without improving reads. A `4,096`-row post-pack
layout remains a reasonable low-concurrency alternative if minimizing PRFS
objects outweighs worker memory.

All sharded layouts were approximately `45%` slower than regular chunks for the
isolated dominant compute-node component scan, so they failed the initial
`20%` read-regression gate at this stage. A decision was therefore deferred to
the end-to-end finalizer comparison recorded below, which found no material
workflow regression.

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
and registry were read-only. This fixture was then used for the
position-balanced end-to-end finalizer A/B below.

### Position-Balanced End-to-End Finalizer A/B

LSF job `153061487` ran four complete `54,000`-row finalizations directly
against the PRFS fixtures on `h07u30.int.janelia.org`. The order was regular,
sharded, sharded, regular, so each physical layout occupied each execution
position once. All runs used eight `process_shards` workers, `256` logical rows
per worker chunk, `128` rows per physical dense output chunk, cheap metrics,
dense `uint8` refined masks, and explicitly disabled eye geometry, full ragged
component contours, and sampled component contours.

This was a core-finalizer storage-layout benchmark, not a complete production
contract benchmark. As of 2026-07-11, eye geometry, full ragged component
contours, and sampled component contours are default finalized outputs. Narrow
diagnostics must opt out explicitly with `--no-write-*`; production timing and
the full-collection canary must include all three surfaces.

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
left by LSF job `153061484`, whose final computation succeeded but whose
completion gate correctly rejected missing Git provenance. Internal finalizer
time is not affected. The wrapper was corrected to carry a Git bundle; job
`153061487` executed from a clean node-local checkout of commit `ba9e4d3` and
all four runs completed normally.

After the timed runs, an exhaustive logical comparison read every corresponding
output element. Both repeat pairs contained `125` arrays per run and reported
zero mismatches. Validation took `738.38 s` and was excluded from finalizer
timing. The complete LSF job used `9,558 MB` maximum memory, no swap, and
finished successfully in `2,140 s` including validation.

This result clears the core mask-finalizer layout gate for the `2,048`-row
probability-sharding candidate, but it does not establish complete default-
surface runtime. Together with the approximately `63x`
probability-payload object reduction, unchanged stored size, faster fixture
construction, and exact core-output parity, it supports retaining `2,048` rows
as the candidate for new read-only probability-mask stores. Repeat the A/B with
all default derived surfaces, then run a full collection canary before any bulk
rewrite of the existing `22` raw shards.

### Complete Default-Output 8/16-Worker Matrix

LSF job `153061568` first ran one complete default-output finalization for each
layout at 8 and 16 workers on `h07u28.int.janelia.org`. Its exact comparison
exposed the degenerate-ellipse validity defect described below. After that
contract was corrected in commit `cb78692`, LSF job `153061604` repeated the
entire matrix on the same host and passed exhaustive parity. All cases used
`256` logical rows per worker chunk, `256` rows per dense output chunk, dense
`uint8` masks, cheap metrics, eye geometry, full ragged component contours,
sampled contours, and `process_shards` postcompute. Raw probabilities were
read directly from PRFS, refined outputs were written to node-local scratch,
and each completed run was copied and atomically committed to a benchmark-only
PRFS Zarr with the production publication helpers.

```text
/groups/johnson/johnsonlab/jeremy/recordings/logs/
  subject_mask_complete_finalizer_matrix/
    sleepyfish_complete_finalizer_matrix_clip000000_20260711_02/
      reports/summary.json
      published_outputs.zarr
```

| Layout | Workers | Finalizer seconds | Core process-shard seconds | Derived postcompute seconds | Peak process-tree RSS GiB | Average effective cores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| regular | 8 | `332.89` | `303.08` | `11.61` | `9.93` | `7.29` |
| sharded 2,048 | 8 | `332.81` | `303.62` | `11.59` | `9.49` | `7.28` |
| sharded 2,048 | 16 | `208.39` | `181.64` | `9.14` | `22.34` | `13.59` |
| regular | 16 | `210.03` | `182.73` | `9.43` | `21.19` | `13.65` |

The corrected matrix makes the layout result especially clear. Sharding was
`0.08 s` (`0.02%`) faster at 8 workers and `1.64 s` (`0.78%`) faster at 16
workers: both are operationally indistinguishable from no runtime difference.
Moving from 8 to 16 workers reduced sharded finalizer time by `37.4%` and
regular time by `36.9%`; scaling was useful but sublinear because per-chunk
time rose under PRFS and CPU contention.

The 16-worker runs approximately doubled process-tree RSS. The sharded case
peaked at `22.34 GiB`, so a production 16-worker request must include margin
above roughly `23 GiB`. LSF reported `16.4 GiB` maximum job memory and no swap;
the process-tree metric can count shared pages more than once. Sharding used
`4.5%` less process-tree RSS at 8 workers but `5.4%` more at 16 workers; there
is no consistent memory win.

Every run created `155` arrays in `13,122` files. Apparent output size was
about `151.28 MB` and allocated size `188.22 MB`, despite the much larger
logical dense-mask shape. Node-local-to-PRFS copy took `92.5-96.4 s`, the
post-copy inventory/validation scan took `17.2-18.3 s`, and atomic commit took
about `0.003 s`. Publication is therefore a separate, metadata-heavy
`110-115 s` phase that worker-count tuning does not improve.

The matrix runner requires all four completion checks—eye geometry, full
contours, sampled contours, and the run completion contract—before publication
is accepted. It then compares every corresponding output array exactly against
the regular 8-worker reference. The fixture Zarrs and registry remain
read-only.

The initial exact comparison passed all `155` arrays for regular versus
sharded input at 8 workers, but failed two float arrays for both 16-worker runs
at row `53,074`: right-eye ellipse parameters and the derived eye separation.
The masks, packed contours, statuses, and every other array were exact. The
offending right-eye mask contained only `7` pixels and its contour enclosed
`1 px²`, yet OpenCV returned a nominally successful ellipse with axes near
`0.26 x 0.06 px`. Its underdetermined angle varied by `13.29 degrees` with
worker partitioning. This is a shared eye-geometry validity bug, not storage-
layout drift. `measure_mask_ellipse()` now rejects fitted axes below one pixel
as `ellipse_invalid_params`. This reclassified `25` subpixel fits, reducing
paired ellipse success from `53,930` to `53,905` and increasing
`assigned_needs_review` from `54` to `79`.

Corrected job `153061604` compared each of the three candidate runs against the
regular 8-worker reference. Every comparison contained `155` arrays and zero
mismatches. Exhaustive parity took `452.06 s`. All four runs also passed eye
geometry, full-contour, sampled-contour, and completion-contract checks. The
job finished `DONE` with no swap.

### Full 22-Shard Collection Canary

LSF job `153063644` finalized all `1,169,010` rows from the existing 22 raw
probability shards with 16 workers. Core process-shard computation took
`5,164.82 s`; complete finalizer time was `5,805.09 s` (`202.82 rows/s`). The
job averaged `14.70` effective CPU cores (`91.89%` of its allocation), used no
swap, and LSF reported `20.59 GiB` maximum memory. The compact collection index
plan was built once in the parent and never rebuilt by workers.

The complete run contains `155` arrays in `298,677` files. Apparent size is
`2.85 GB` and allocated size is `2.93 GB`. Node-local-to-PRFS copy took
`2,109.59 s`; the atomic commit took `0.004 s`. Exhaustive source/publication
comparison read `364,522` chunks and found zero mismatches.

The original validation report rejected the canary because the 54,000-row
reference stored eye `reason_bytes` with width 207, while the full collection
needed widths 277 and 255. Decoded `reason` strings matched, common byte
columns matched, and every additional byte was zero. Fixed-width byte width is
therefore a run-local storage detail, not a semantic mismatch. The original
report remains preserved; corrected validation v2 compares shared bytes
exactly and accepts different trailing widths only when every extra byte is
zero. The validation-only replay compared 137 reference arrays with zero
mismatches and passed every gate in `242.16 s`.

An independent dense-content audit also passed. All `18,268` expected
`masks_roi` payload chunks exist. Subject-body coverage is `100%`; left- and
right-eye coverage are each `99.9801%` with 233 absent rows; swim-bladder
coverage is `99.9995%` with six absent rows. A total of 264 representative
dense masks across all 22 source clips matched stored `area_px` exactly. The
canary is classified as a data/finalizer/publication pass despite the original
LSF exit code caused by the validator false negative.

Corrected evidence is stored in:

```text
/groups/johnson/johnsonlab/jeremy/recordings/logs/
  subject_mask_full_collection_canary/
    sleepyfish_full_collection_canary_20260711_02/reports/
      validation.json
      validation_corrected.json
      dense_content_audit.json
      summary_corrected.json
```

### Canonical Copy Promotion

The completed canary did not need another finalizer run. Commit `e49e3d75`
added a copy-only promotion path, and LSF job `153075919` used it to import
`refined_subject_masks_sleepyfish_full_collection_canary_20260711_02` into the
canonical recording Zarr. The source was copied into a hidden sibling,
validated there, and exposed with one atomic rename. Inference, refinement,
metrics, geometry, and contour computation were not rerun.

The source and canonical inventories both contain `298,677` files and
`2,846,839,371` apparent bytes. Their relative-path/size digests and metadata
byte digests match exactly. Exhaustive decoded comparison covered all `155`
arrays in `364,522` batches with zero mismatches. The canonical contract then
passed with zero errors and zero warnings. Dense `masks_roi` is physically
present as `uint8` with shape `(1,169,010, 4, 512, 512)`.

All three selectors now name the full run:

- `latest`
- `latest_complete`
- `refined_subject_mask_review_status_latest`

The registry refresh completed successfully and reports the canonical
`refined_subject_masks` step as `ok`, sourced from
`copy_promoted_completed_refined_subject_mask_run`. The prior two-clip smoke
run remains preserved but is no longer selected.

The promotion took `8,853.00 s` total. Physical copy took `3,002.02 s`; the
remaining cost was dominated by inventories and exhaustive decoded equality.
Because dense masks contain about `1.2 TB` of logical pixels, comparing source
and destination decodes about `2.4 TB` even though each compressed tree is only
about `2.85 GB`. This was appropriate as a one-time recovery proof, but routine
same-filesystem promotion should use atomic copy, exact physical inventory or
payload digests, contract validation, and deterministic decoded sampling. It
should not make a full decoded equality pass the default.

### Next Benchmark: PRFS Input Staging

The next controlled experiment will separate the benefit of indexed storage
shards from the benefit of copying their physical files to node-local scratch.
It will use the existing contract-complete `54,054`-row clip fixture and the
selected `2,048`-row probability layout. All cases will run the same complete
default-output finalizer with `16` workers, `256` logical rows per worker
chunk, and node-local refined output. Only the input placement will change:

1. **Direct PRFS:** read the sharded fixture directly through the node-local
   PRFS overlay. This is the current baseline.
2. **Probability-only stage:** copy the three sharded probability arrays and
   their metadata to node-local scratch once, while leaving crop lineage,
   detections, and keypoints on PRFS.
3. **Full-source stage:** copy the complete contract fixture to node-local
   scratch before starting the finalizer.

Case 2 should exercise the existing production
`stage_finalization_input_to_scratch` path rather than a benchmark-only copy
implementation. That path copies the selected `subject_mask_runs/<run>` into
the local output overlay and keeps the remaining root-level inputs linked to
PRFS. Case 3 needs an explicit complete-fixture copy mode in the benchmark
runner.

For this full-clip benchmark, probability-only staging is a practical proxy
for whole-storage-shard staging: every probability row will be consumed, so
all outer shards are needed. A later partial-access implementation should
assign complete `2,048`-row outer storage shards to workers and stage each
physical shard once. Each staged outer shard then serves eight `256`-row
worker chunks. It must not let multiple workers copy the same shard or assign
workers ranges that split a physical output chunk.

Each case will record these phases separately:

- source inventory and staging seconds, bytes, and physical file count;
- target initialization time;
- core `process_shards` time and chunk-duration distribution;
- derived-surface postcompute time;
- total finalizer time after staging;
- end-to-end time from staging start through completed local output;
- peak process-tree RSS, LSF maximum memory, swap, CPU efficiency, and peak
  local-scratch use;
- output publication time as a separate invariant phase, not part of the input
  placement comparison.

The primary comparison is end-to-end time, `staging + finalization`, rather
than finalizer time alone. The secondary comparison is operational behavior:
chunk-duration variance and PRFS object pressure under 16 concurrent workers.
The staged variants must produce exactly the same `155` arrays as the direct
PRFS reference and pass eye-geometry, full-contour, sampled-contour, and run-
completion checks.

Adopt staging only if it has zero parity differences and either improves
median end-to-end time by at least `10%` over repeated passes or materially
reduces chunk-time variance and PRFS pressure without excessive scratch use.
No case may swap. Cache state and run order must be repeated or alternated so
one warm-page-cache pass cannot decide the result. Use three rotated blocks
(`A-B-C`, `B-C-A`, and `C-A-B`), create a fresh scratch destination for every
staged pass, request `POSIX_FADV_DONTNEED` for source and staged payload files
between passes, and record whether each advisory cache-eviction request was
accepted. The advisory is useful experimental control, not proof that PRFS or
all kernel caches were cold.

If probability-only staging wins, repeat the winning mode on the full
`1,169,010`-row collection and compare 16 versus 24 workers. Test 32 workers
only if the 24-worker run remains comfortably within its memory allocation and
does not increase end-to-end time. This sequence answers separately whether
local reads help and whether they recover useful scaling under PRFS
contention.

The first executable pilot combines the staging and worker-count questions in
a position-balanced `2 x 2` matrix on the contract-complete clip: direct versus
staged `subject_mask_runs` input, each at 16 and 24 workers. It runs the four
cases forward and then in reverse, includes staging in end-to-end time, keeps
all candidate outputs on node-local scratch, and requires exact parity across
the complete output surface. Full-source staging remains the follow-up only if
probability-run staging leaves a meaningful residual input cost.

```bash
scripts/submit_subject_mask_input_staging_workers_bsub.sh \
  --fixture-root "$FIXTURE_ROOT" \
  --queue local \
  --wait-for-job 153063644 \
  --submit-host login1-citrus-poller
```

LSF job `153064150` completed this matrix on `h07u05`. The staged source was
`177.46 MB` in 110 regular files and copied to scratch in `0.79-0.91 s` after
successful advisory cache eviction. Every run produced 155 arrays; all seven
comparisons against the first direct 16-worker run had zero mismatches.

| Input placement | Workers | Median end-to-end s | Two runs s | Median CPU efficiency | Peak process-tree RSS GiB |
| --- | ---: | ---: | --- | ---: | ---: |
| direct PRFS | 16 | `214.52` | `218.63`, `210.40` | `83.11%` | `22.42` |
| staged subject run | 16 | `207.08` | `207.13`, `207.03` | `85.48%` | `20.18` |
| direct PRFS | 24 | `173.64` | `174.58`, `172.70` | `82.87%` | `29.75` |
| staged subject run | 24 | `172.57` | `171.64`, `173.50` | `83.40%` | `30.56` |

At 16 workers, staging saved `7.44 s` (`3.47%`) by median end-to-end time. One
direct run paid a `10.10 s` target initialization versus approximately `4 s`
for every other case; after that outlier, the second paired staging advantage
was only `1.60%`. Staging was very stable across its two 16-worker runs, but the
sample is too small to treat variance reduction alone as a production gate.

At 24 workers, staging saved only `1.07 s` (`0.62%`) by median and lost one of
the two paired comparisons. Core process-shard time was essentially identical:
direct median `141.69 s` versus staged `141.00 s`. Physical-shard staging does
not materially accelerate this CPU-dominated clip once 24 workers are active,
so it fails the planned `10%` adoption threshold. Do not make input staging a
mandatory production step from this evidence; retain it as an optional full-
collection/high-contention experiment.

Moving from 16 to 24 workers reduced median end-to-end time by `19.06%` for
direct reads and `16.67%` for staged reads. Per-chunk median duration increased
from about `12.8 s` to `15.7 s`, showing the expected contention, but the extra
parallelism still reduced wall time. CPU efficiency relative to requested
workers stayed near `83%`. Peak process-tree RSS increased from approximately
`20-22 GiB` to `30-31 GiB`; LSF reported `26.15 GiB` maximum memory for the
overall job and no swap. A full-collection 24-worker run therefore needs
meaningful memory margin above 32 GiB and should remain a canary before becoming
the default.

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
  --shard-rows 2048 \
  --sample-rows 8192 \
  --inner-chunk-rows 32 \
  --batch-rows 256 \
  --random-read-count 32 \
  --overwrite
```

Submit the contract-complete finalizer layout A/B with:

```bash
FIXTURE_ROOT=/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/sleepyfish_finalizer_ab_clip000000_20260711
scripts/submit_subject_mask_finalizer_layout_ab_bsub.sh \
  --fixture-root "$FIXTURE_ROOT" \
  --queue local \
  --submit-host login1-citrus-poller
```

Submit the complete default-output 8/16-worker matrix with:

```bash
scripts/submit_subject_mask_complete_finalizer_matrix_bsub.sh \
  --fixture-root "$FIXTURE_ROOT" \
  --queue local \
  --submit-host login1-citrus-poller
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
  on disk. Do not rewrite them in place.
- Treat `refined_subject_masks_sleepyfish_sm_allclips_20260708_02` as failed or
  incomplete, not authoritative.
- The repeated collection identity-map construction is fixed in `ba9e4d3` and
  the eight-worker real-collection initialization smoke passed with about
  `2 GiB` LSF process-tree memory.
- The `2,048`-row indexed-sharding layout has cleared the single-clip storage,
  probability exactness, construction, core-finalizer, and complete default-
  output timing gates. The complete 22-shard collection finalizer canary also
  passed its identity, memory, publication, exhaustive parity, and dense-
  content gates. The layout remains the selected candidate for new immutable
  read-only probability-mask stores.
- The full-clip inference canary also passed exactness, lineage, content,
  completion, object-count, and atomic-publication gates, reducing probability
  files `61.8x`. The direct two-buffer writer then reduced raw-stage time from
  `883.9 s` to `628.2 s` at `4 GiB` maximum accounted memory, with exact
  source/destination digests and no working array. The remaining `150.5 s`
  full destination validation is the material premium over ordinary chunks.
  The production default is now the `2,048`-row double-buffered layout;
  ordinary chunks require the explicit `--no-mask-probs-sharding` override.
- Finalized runs include eye geometry and sampled component contours by
  default. Full ragged component contours are now an explicit compatibility or
  analysis opt-in. The historical `335 s` layout A/B excluded the required
  derived surfaces and must not be quoted as complete production time.
- The full `22`-shard canary exercised the compact parent identity plan,
  finalizer completion/publication behavior, exhaustive array parity, and
  independent dense-content validation at the complete `1,169,010`-row scale.
  Preserve the same gates for future collection or storage-layout canaries.
- The complete canary output has been imported into the canonical recording
  Zarr by validated copy and atomic rename. It is now the `latest`,
  `latest_complete`, and review-status-selected refined subject-mask run. The
  incomplete historical all-clips run remains unpromoted.
- Do not rerun GPU inference merely to recover this refined output.
