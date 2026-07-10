# Subject-Mask Finalizer And Publication Status

**Date:** 2026-07-09
**Status:** current cluster diagnostic, completed full-run storage/performance gates, and pending PRFS publication canary

## Scope

This note captures where the refined subject-mask finalization workflow stands
after adding the CPU-only finalization submitter and rerunning a full
GoodCopBadCop recording on the cluster.

It is not a storage contract change. The current subject-mask policy still
stands:

- dense `refined_subject_masks_runs/<run>/masks_roi` is the authoritative
  editable and training/export mask surface;
- bitpacked and RLE stores are optional derived display/archive caches;
- eye geometry and component contours remain important downstream derived
  surfaces.

Related docs:

- `docs/refined_subject_masks_runs_contract.md`
- `docs/diagnostics/subject_mask_finalization_parallelism_2026-06-20.md`
- `docs/diagnostics/subject_mask_contour_sampling_2026-07-08.md`
- `docs/clipped_collection_roi_cache_model_workflow.md`

Related Crimson docs, in the `crimson-ui-monolith` repository:

- `docs/crimson_mask_overlay_async_buffering_notes.md`
- `docs/crimson_subject_mask_editable_dense_storage_decision_2026-07-08.md`
- `docs/crimson_annotation_append_log_and_materialized_runs_2026-07-08.md`

## Current Working Path

The working cluster path is now:

1. Submit one CPU-only finalization array task per recording with
   `scripts/submit_subject_mask_finalization_batches_bsub.sh`.
2. Stage the target analysis Zarr to node-local writable scratch when available.
3. Run `fisheye.refinement.finalize_subject_masks` against the staged Zarr.
4. Validate the staged output.
5. Publish the completed refined run group back into the canonical PRFS Zarr.
6. Validate the published output.
7. Refresh subject-mask registry performance and component-quality views.
8. Clean up the staged output.

This avoids holding a GPU allocation during finalization. The wrapper defaults
are now production-oriented for this workload:

```text
NCORES=16
MEM_GB=32
MAX_ACTIVE=4
FINALIZE_CHUNK_SIZE=256
FINALIZE_DENSE_MASK_ROW_CHUNK=256
FINALIZE_EXECUTION_BACKEND=process_shards
FINALIZE_NUM_WORKERS=16
FINALIZE_POSTCOMPUTE_BACKEND=process_shards
MASK_STORAGE=dense_uint8
WRITE_EYE_GEOMETRY=1
WRITE_COMPONENT_CONTOURS=1
STAGE_OUTPUT_TO_SCRATCH=1
STAGE_FINALIZATION_INPUT_TO_SCRATCH=1
```

Scratch staging now falls back when `/scratch/$USER` exists but is not writable.
The failed first run exposed that some nodes have `/scratch/delahantyj` owned by
`root:root`; the runner now checks write and execute permissions before using
that path.

## Full-Recording Cluster Result

Recording:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop/zarr/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop_analysis.zarr
```

Source raw subject-mask run:

```text
subject_masks_unet_registry_subject_masks_bitpacked_only_canary_20260621_01
```

Published refined run:

```text
refined_subject_masks_smart_finalizer_subject_mask_finalization_finalizer_only_perf_gcbc_arena1_20260709_03
```

LSF job:

```text
153037024[1]
host: h07u21
status: DONE
```

Outcome:

| field | value |
|---|---|
| finalization status | `ok` |
| validation status | `ok` |
| publish status | `ok` |
| registry refresh status | `ok` |
| output staging status | `cleaned` |
| registry subject-mask performance rows | 17 |
| registry component-quality rows | 62 |

Workflow phase timings:

| phase | seconds |
|---|---:|
| `archive_total` | 1031.20 |
| `prepare_output_staging` | 5.62 |
| `finalization_subprocess` | 592.42 |
| `validate_outputs` | 0.04 |
| `publish_staged_outputs` | 425.07 |
| `validate_published_outputs` | 0.27 |
| `consolidate_metadata` | 4.81 |
| `cleanup_output_staging` | 2.95 |

Finalizer-reported performance:

| field | value |
|---|---:|
| rows total | 120221 |
| finalizer duration | 583.67 s |
| finalizer throughput | 205.97 rows/s |
| chunk count | 470 |
| chunk size | 256 |
| worker processes | 16 |
| dense mask chunks | `[256, 1, 512, 512]` |
| postcompute duration | 18.50 s |
| postcompute throughput | 6497.31 rows/s |

The current CPU-only finalizer is materially better than the recent modern
cluster canaries:

| baseline | finalizer seconds | rows/s | total seconds | publish seconds | interpretation |
|---|---:|---:|---:|---:|---|
| current `20260709_03` | 583.67 | 205.97 | 1031.20 | 425.07 | current reference |
| bitpacked-only canary `20260621_01` | 888.76 | 135.27 | 1301.00 | 391.09 | current finalizer is about 1.5x faster |
| dense+bitpacked canary `20260621_01` | 1212.94 | 99.12 | 1644.49 | 409.41 | current finalizer is about 2.1x faster |
| RLE chunk256 `20260620_01` | 822.05 | 146.25 | n/a | n/a | current finalizer is about 1.4x faster |

An older `20260619` run reported a faster raw finalizer rate, but it predates
the current dense/contour/eye-geometry metadata surface and is not a clean
apples-to-apples baseline.

## Benchmark Policy For Publication Changes

Small deterministic Zarr fixtures are the first correctness gate, not
performance evidence. They are appropriate for proving output equality,
revision behavior, stale markers, and safe write ownership, but they cannot
predict full-run object counts or PRFS publication time.

Use three benchmark levels:

| level | workflow | decision it supports |
|---|---|---|
| focused fixture | unit tests with a few rows | semantic parity, chunk layout, write-safety invariants, failure cases |
| local row-window canary | `fisheye.diagnostics.benchmark_subject_mask_full_finalizer`, normally 4096 contiguous real rows copied into a temporary local Zarr | fast comparison of compute backends and phase attribution |
| full staged run | `scripts/submit_subject_mask_finalizer_benchmark_bsub.sh`, which stages the complete subject-mask run to node-local scratch | production compute scaling and complete refined-output file/byte/chunk inventory |

The full staged benchmark must use the same source run, components, metric
level, contour/eye-geometry policy, worker count, and mask storage mode for all
variants. `process_shards` is now the only supported production parallel
backend. New compute-kernel or storage-layout canaries should compare against a
controlled replay of the exact pre-change `process_shards` source snapshot,
not against Dask. The benchmark report now records a one-pass refined-run
storage inventory in addition to finalizer timings, so the 120221-row result
can compare total files and the `components/`, `metrics/`, and `masks_roi`
subtrees before scratch cleanup.

Node-local finalization still does not measure PRFS publication. A layout may
be accepted only after a separate publication canary copies its complete staged
run into a hidden `.publish_tmp` target on PRFS, records copy and commit times,
validates the copied run, and removes or promotes the target according to the
canary plan. Publication telemetry now records file count, apparent/allocated
bytes, per-array chunk layouts, top-level subtree pressure, storage-scan time,
copy time, and commit time.

The current working-tree finalization-metric implementation uses `16384`-row
chunks and a driver-merged sealed write. Workers return fixed-size metric
payloads; only the driver writes the large physical metric chunks after worker
compute is complete. This avoids the unsafe case where several 256-row worker
tasks perform read-modify-write operations inside one 16384-row Zarr chunk.
Focused and broader unit suites pass. A clean-commit full node-local staged run
also completed successfully and reduced the refined-run inventory to 28119
files. A same-allocation position-balanced comparison found no performance
regression. The separate PRFS publication canary is still pending.

### Zarr Chunk-Size And Sharding Guidance

Zarr does not prescribe one universal chunk size. Its performance guide says
that chunks of at least about 1 MiB uncompressed often perform better with
Blosc, but also says the chunk shape must follow the actual slicing pattern.
That is a starting heuristic, not a requirement for small metric arrays:

- https://zarr.readthedocs.io/en/v3.1.5/user-guide/performance/
- https://zarr.readthedocs.io/en/v3.0.10/user-guide/arrays.html#sharding
- https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/

Zarr v3 sharding separates the independently readable inner chunk from the
storage-object/write shard. It can retain small reads while bundling many inner
chunks into fewer files. The tradeoff is write amplification and concurrency:
a full shard is the safe and efficient write unit, and partial updates may need
to read and rebuild the existing shard depending on the store and codec path.

Crimson uses TensorStore, whose Zarr v3 driver supports `sharding_indexed`, and
the Crimson build currently pins TensorStore `v0.1.64`. That makes a read canary
reasonable, but the existing Crimson runtime probe checks only general Zarr v3
support. Add an explicit sharded-array open/read test before changing a Palette
contract:

- https://google.github.io/tensorstore/driver/zarr3/index.html
- https://google.github.io/tensorstore/kvstore/zarr3_sharding_indexed/index.html

Apply sharding selectively:

| surface | current/proposed unsharded unit | sharding assessment |
|---|---|---|
| editable dense `masks_roi` | `[256,1,512,512]` uint8, 64 MiB uncompressed | first test a clone with 1-2 MiB inner chunks and the existing 64 MiB shape as the shard; this may improve single-row reads but does not reduce the current 1875 mask files and retains a large edit/write unit |
| sealed fixed-shape metrics | `[16384,...]`, generally only eight row chunks per array | prefer simple unsharded chunks first; sharding adds codec/index complexity after file count is already small |
| sampled fixed-K contours | target row chunks selected near 0.25-1 MiB depending on component | prefer unsharded unless the measured full-run inventory still warrants bundling |
| optional full ragged `points_xy` | test 65536-point (512 KiB) or 262144-point (2 MiB) chunks | reasonable sharding canary because the surface is immutable/derived and inner chunks can preserve bounded exact reads |

Do not make authoritative `masks_roi` sharded by default until a canary measures
Crimson paused-row and playback reads, one-row edit latency, full-shard rewrite
behavior on PRFS, concurrent-write safety, publication time, and recovery after
an interrupted write. Sharding is most immediately attractive for immutable
derived read surfaces; it is not a substitute for the sampled-contour policy or
for large simple chunks in small sealed metric arrays.

The working tree also tested an opt-in `dask_array_store` canary. The driver
created destination arrays and attrs, pure row-block tasks computed fixed-shape
payloads, every Dask source was rechunked to the physical grid of its Zarr
target, and one `dask.array.store()` graph wrote the related outputs. After the
results below showed substantially worse throughput and process behavior, that
backend and the older delayed-task Dask backend were removed from the active
finalizer API. They remain documented here only as benchmark history.

### Full Staged Backend Results

The completed full staged comparisons use the same 120221-row source and write
the same production-matched `[256, 1, 512, 512]` dense-mask layout:

| backend | workers | finalizer seconds | rows/s | fixed-shape compute seconds | output files |
|---|---:|---:|---:|---:|---:|
| current `process_shards`, expanded common metrics | 16 | 433.17 | 277.54 | 395.95 | 28119 |
| earlier `process_shards`, first metric layout | 16 | 279.18 | 430.63 | 255.78 | 48639 |
| delayed-task Dask processes | 16 | 451.72 | 266.14 | 414.55 | 48639 |

The earlier dense-row-128 comparison produced the same ordering:
`process_shards` took 347.15 seconds while delayed-task Dask threads took
906.93 seconds. That run is useful for backend comparison but not for the final
production file count because its dense mask chunk was `[128, 1, 512, 512]`.

The production-matched staged inventories are now:

| subtree | published reference | first metric layout | expanded common metrics |
|---|---:|---:|---:|
| full refined run | 66087 | 48639 | 28119 |
| `masks_roi` | 1875 | 1875 | 1875 |
| `components/` | 51385 | 33937 | 24619 |
| `metrics/` | 11257 | 11257 | 55 |
| `relations/` | n/a | 942 | 942 |

The expanded layout removes 20520 files from the first-layout staged run
(42.2%) and 37968 files from the published reference (57.5%). Run-level
`metrics/` alone falls by 11202 files; the additional component reduction comes
from large spatial/topology metric and fingerprint chunks.

This was not an unchanged-layout run. It already used the working-tree
`16384`-row, driver-merged layout for sealed component finalization metrics.
It did **not** yet apply the proposed large chunks to common run metrics and
other sealed component arrays, and it still wrote the production full ragged
contours. Therefore 48639 is the measured result of the first layout change,
not the expected endpoint of the publication cleanup. The expanded
common-metric canary is LSF job `153050747`, run from clean commit `cef17a9` on
`h07u24`; it completed with empty stderr, 90.1% sampled CPU efficiency across
16 allocated slots, and 28119 files.

Do not attribute the current canary's 433.17-second duration to the layout. The
279.18-second first-layout run executed on `h06u02`, while the current run
executed on `h07u24`, and nearly every compute phase—not just metric
persistence—was slower. The new driver-owned common metric writes themselves
took about 0.24 seconds total.

LSF job `153050823` resolved the performance question on `h07u24`. It compared
the exact restored `46b3fe1` source with clean candidate `cef17a9` over the same
4096-row local copy in `AB`, `BA`, `AB`, `BA` order:

| implementation | four finalizer runs, seconds | median seconds | median CPU seconds | median process-tree RSS |
|---|---|---:|---:|---:|
| restored baseline | 30.442, 28.868, 28.627, 28.487 | 28.748 | 195.985 | 8.68 GiB |
| expanded common metrics | 28.706, 28.472, 28.524, 28.409 | 28.498 | 194.010 | 8.74 GiB |

The candidate was about 0.9% faster by median finalizer time, used about 1.0%
less median CPU time, and used about 0.7% more median RSS. Treat that as
performance-neutral rather than a speedup. All 122 arrays matched exactly.

### Dask-Array Row-Window Result

The Dask-array design was tested on the documented 4096-row real-data canary.
Both completed candidates used 256-row compute chunks, physical dense chunks
`[256, 1, 512, 512]`, eight outer workers, production-matched one-thread native
library settings, no contour/eye-geometry postcompute, and the same copied local
input. Resource samples were recorded from inside the benchmark process tree at
two-second intervals.

| backend | finalizer seconds | rows/s | sampled finalizer effective cores | sampled CPU efficiency | peak process-tree RSS |
|---|---:|---:|---:|---:|---:|
| `process_shards`, 8 processes | 26.82 | 152.71 | 6.43 | 80.4% | 12.42 GB |
| `dask_array_store`, 8 threads | 66.41 | 61.68 | 5.68 | 71.0% | 8.63 GB |

Both outputs had the same 1100-file layout. The Dask-array thread run opened
the source context once and recorded 15 cache hits, so repeated Zarr opens were
successfully removed. It was nevertheless about 2.5 times slower and consumed
about twice the CPU time. More CPU activity in an earlier nested-thread run did
not translate into useful throughput.

Those 15 hits are application-level source-context cache hits: they mean an
already-open Zarr context was reused. They are unrelated to hardware L1/L2/L3
cache-hit counters and must not be cited as evidence of CPU-cache behavior.

The Dask-array process canary was stopped after 536 seconds without completing.
Its partial samples averaged 2.12 effective cores, or 26.6% of eight workers,
and reached 38.9 GB RSS. The local multiprocessing scheduler serialized one
large block payload across dozens of selector/rechunk/store consumers; the
result was graph fan-out, memory amplification, and low utilization.

Conclusion: keep `process_shards` as the only supported production parallel
backend. The Dask finalizer implementations have been retired; any future Dask
design should begin as a separate benchmark canary after materially redesigning
the fixed-shape batch payload and graph topology.

### In-Job Resource Telemetry

Future benchmark submissions wrap each finalizer command with local
process-tree instrumentation. This does not poll LSF, login nodes, or compute
nodes from an external agent. Each allocated job reads only its own process
tree and writes:

- `<variant>.resources.json`: exact aggregate child user/system CPU seconds,
  average effective cores, worker- and slot-relative CPU efficiency, peak
  sampled RSS/process/thread counts, exit status, and host/job context;
- `<variant>.resources.jsonl`: two-second utilization samples for separating
  copy, finalizer, postcompute, and storage stalls;
- the same resource summary embedded in the main variant JSON.

The full staged runs above were submitted before this wrapper was added, so
their saved JSON has phase timing but not the new process-tree samples. The
4096-row runs validate the new telemetry path.

### Compute-Kernel Optimization Result

The initial controlled benchmark used `numeric_struct_of_arrays_v1`; the
follow-up spatial-stat reuse candidate was identified as
`numeric_struct_of_arrays_spatial_reuse_v2`:

1. Each component block returns dense numeric arrays for masks, source masks,
   a fixed `float32[N, M]` metric matrix, `uint32` finalization reason flags,
   `int16` quality codes, `float32` quality scores, and `uint8` review codes.
   Production workers no longer build a result dataclass, metric dictionary, or
   base reason/review strings per row. Eye-assignment labels remain strings
   because that separate assignment surface has not yet been converted.
2. Simple thresholding, metric-QC predicates, finalization reason predicates,
   and review routing are vectorized across the block. Morphology, connected
   components, hole measurement, and pixel change/probability reductions remain
   in the row loop.
3. Connected-component labels and stats are reused across small-component
   filtering, component selection, and finalization topology metrics. Body and
   swim-bladder topology metrics reuse those results in the caller. Complete
   label reuse through eye assignment remains future work.
4. Numeric reason flags are decoded at the driver/persistence boundary. The
   worker payload omits `extra_labels` entirely for body and swim bladder, so it
   does not manufacture one `"clean"` Python string per row. Only the currently
   unavoidable eye-assignment labels cross that boundary as strings.
5. The 4096-row real-data canary was replayed from an exact pre-refactor source
   snapshot and every output array was compared chunk by chunk.

Full `(N,H,W)` vectorization of area/change/probability calculations was tried
and rejected. It performed repeated full-block memory sweeps and created large
temporary boolean arrays. A 32-row subblock reduced that pressure but still
lost to the original row-local calculation. The retained kernel fuses those
pixel operations into the morphology row loop, while each 512-by-512 mask has
a bounded working set, and vectorizes only the inexpensive block-level
decisions.

Controlled results used the same 4096 contiguous GoodCopBadCop rows, local
input copy, 256-row chunks, eight `process_shards` workers, one native-library
thread per process, dense output, and no contour/eye-geometry postcompute:

| implementation | finalizer seconds | rows/s | finalizer wall seconds | peak process-tree RSS |
|---|---:|---:|---:|---:|
| exact pre-refactor replay | 14.915 | 274.62 | 15.082 | 9.59 GB |
| full-block pixel vectorization | 18.852 | 217.28 | 19.090 | 11.21 GB |
| 32-row pixel subblocks | 15.735 | 260.30 | 15.909 | 10.88 GB |
| retained fused numeric kernel | 15.084 | 271.55 | 15.275 | 10.62 GB |

The retained kernel is effectively performance-neutral in this single
controlled replay: about 1.1% lower throughput and 1.3% higher finalizer wall
time, with about 10.8% higher peak process-tree RSS. All 138 arrays and 2293
read chunks were exactly equal, including floating metrics, reason ordering,
and masks. The earlier 26.82-second `process_shards` result came from a
different point in the working-tree history and is not valid causal evidence
for this refactor.

The isolated cluster promotion gate subsequently rejected the v2 candidate.
LSF job `153050432` ran committed candidate `fb64b70` against the frozen exact
pre-refactor source in symmetric `AB`, `BA`, `AB`, `BA` order on the same
4096-row scratch copy and eight-slot allocation:

| implementation | finalizer seconds, four runs | median seconds | mean seconds | median CPU seconds | median process-tree RSS |
|---|---|---:|---:|---:|---:|
| exact pre-refactor replay | 193.953, 89.178, 82.924, 105.533 | 97.355 | 117.897 | 707.843 | 10.67 GiB |
| v2 candidate | 181.978, 153.437, 121.563, 111.715 | 137.500 | 142.173 | 1031.896 | 10.62 GiB |

The candidate median was 41.2% slower and its symmetric-sequence arithmetic
mean was 20.6% slower. Median CPU time was 45.8% higher, CPU efficiency was
similar (88.4% versus 87.5% of eight allocated slots), and RSS was effectively
unchanged. The candidate was slower in three of four same-repeat comparisons.
All 122 arrays written under this no-contour/no-eye-geometry benchmark policy
matched exactly. Strong cold-cache/time drift means these data should not be
used to claim a precise micro-optimization cost, but they decisively fail to
show a production benefit. Do not promote the numeric/spatial-reuse kernel;
restore the replayed baseline and reintroduce only separately measured slices.

The local replay initially made the compact typed contract look like useful
performance-neutral architectural cleanup, but the cluster gate shows that
architecture alone is not sufficient reason to retain it in the production
hot path. The exact replayed baseline has now been restored; the rejected
implementation remains only in Git history and the benchmark artifacts. A
C++/pybind block loop is also not justified by these timings. Profile one
separately introduced slice at a time from the restored baseline.

No hardware cache-miss counters were collected in these runs. Statements about
keeping row-local data hot are working-set/locality reasoning supported by the
relative timing and RSS of the attempted layouts, not proof of a particular
L1/L2/L3 cache-hit rate.

### Current Compute Implementation Checklist

Active production path:

- [x] Keep `process_shards` as the only production parallel backend; each
  process owns contiguous, physical-chunk-aligned rows and opens the Zarr once.
- [x] Keep native OpenCV/BLAS thread pools at one thread per worker process.
- [x] Use the exact cluster-winning row-wise finalization result path.
- [x] Merge finalization-metric payloads on the driver and write sealed
  16384-row physical chunks without overlapping worker writes.
- [x] Reuse assignment-time eye ellipses and contours for persisted eye
  geometry instead of measuring them again during postcompute.
- [x] Read body/swim masks in contiguous chunks for contour postcompute rather
  than once per row.
- [x] Record in-job phase, process-tree CPU, CPU-efficiency, thread-count, and
  RSS telemetry and retain the position-balanced cluster A/B harness.

Rejected and removed from active source:

- [x] Numeric struct-of-arrays batch results and block-level reason/review
  vectorization.
- [x] Numeric reason-flag transport and driver-side reason decoding.
- [x] Block-scoped morphology kernels, reusable pixel scratch, and
  change-aware hole-fill helpers.
- [x] Connected-component and assigned-eye spatial-stat transport used to skip
  later dense spatial scans.

These candidates are not retained behind production flags or as parallel
implementations. Git history and the saved A/B run are the reproducibility
surface if a narrower idea is reconsidered.

Partially complete:

- [~] Assigned-eye component count and largest-component fraction are reused,
  but eye hole metrics still require a separate background-labeling pass.
- [~] Eye assignment still constructs reason/status strings per row. This is a
  cleanup opportunity, but current timing shows it is not a leading cost.

Not started:

- [ ] Call-level CPU profiling of the retained fused kernel.
- [ ] Hardware performance-counter measurement for cache/TLB/branch behavior.
- [ ] A compiled C++/pybind batch loop.

### Next Compute Implementation Order

1. Profile the restored baseline at call level to identify one measured hot
   transition or duplicate scan.
2. Implement only that one slice in an experiment branch or frozen source
   snapshot; do not add a second production backend or dormant hot-path code.
3. Require exact output parity and a position-balanced cluster improvement
   beyond run noise before promotion.
4. Consider a compiled loop only if profiling shows Python/native transitions
   remain material after simpler isolated changes. Rely on OpenCV/compiler SIMD
   before considering handwritten architecture-specific intrinsics.

Every slice must preserve exact masks, row identity, integer metrics, reason
flags/order, review codes, and chunk ownership. Approximate floating metrics
may use a schema-documented tolerance only when no thresholded decision can
flip. Promotion requires focused tests, exact/tolerant array parity as
appropriate, a controlled real-row benchmark improvement beyond run noise, and
no material RSS regression.

Do not repeat already rejected candidates without new evidence: Dask or thread
production backends, full-block/32-row pixel vectorization, the corrected
half-plane or foreground-sparse eye split, the current component fast path, or
the `connectedComponentsWithStats` eye selector. Each was slower or had an
unfavorable parity/risk tradeoff on real GoodCopBadCop rows.

### Isolated Kernel A/B Gate

Do not use the latest workstation timings as promotion evidence. The machine
was concurrently busy: one attempted run reached a one-minute load average of
about 43, and unrelated eye-assignment and Zarr-write phases slowed together.
A later candidate run reported 15.063 seconds and 271.93 rows/s, but even that
run shared the workstation with other work. These runs remain valid for output
generation and exact parity, not causal performance conclusions.

`scripts/submit_subject_mask_finalizer_kernel_ab_bsub.sh` now prepares a single
LSF job that:

1. snapshots the saved baseline and the clean committed candidate from the
   synchronized cluster-visible Palette checkout;
2. copies the same 4096-row real source window once to node-local scratch;
3. runs baseline/candidate in position-balanced `AB`, `BA`, `AB`, `BA` order
   inside the same eight-slot allocation with one native-library thread per
   worker;
4. records process-tree CPU/RSS samples separately for every run;
5. reports median finalizer time, resource-wrapper wall time, CPU seconds, and
   peak RSS; and
6. compares every output array exactly before exiting successfully.

Two setup probes were submitted but did not produce benchmark evidence:

```text
LSF 153050415: exited after baseline compute because strict Git provenance had
               no repository working directory
LSF 153050421: exited at startup because a workstation /home checkout is not
               mounted on compute nodes
```

The corrected harness uses
`/groups/johnson/johnsonlab/jeremy/gitrepos/palette` for provenance, refuses to
snapshot it when dirty, and archives only committed `src/fisheye` files. The
completed non-publishing gate is:

```text
run: /groups/johnson/johnsonlab/jeremy/recordings/logs/
     subject_mask_finalizer_benchmarks/
     subject_mask_finalizer_kernel_ab_spatialreuse_20260710_05
job: 153050432
candidate commit: fb64b704d9d731930da0f204c8b17864737b7715
```

`reports/parity.json` records zero mismatches across 122 arrays.
`reports/summary.json` records the failed performance gate summarized above.
The job completed in 1175 seconds with an empty stderr file.

## Why Publication Is Still Expensive

The remaining bottleneck is not the logical byte size of dense masks. The
published run is small by byte volume but large by filesystem object count:

| subtree | files | disk usage |
|---|---:|---:|
| full refined run | 66087 | 188M |
| `masks_roi` | 1875 | 76M |
| `components/` | 51385 | 99M |
| `metrics/` | 11257 | 9.5M |

Publication took `425.07 s` to publish a run that occupies only `188M` on disk.
That effective byte throughput is low because PRFS has to create and commit
tens of thousands of small filesystem objects. The dominant cost is metadata
traffic and per-file creation, not bulk data transfer.

The current publish implementation is safe but file-count-heavy:

- `_prepare_run_group_publish(...)` copies the staged completed run group into
  a hidden `.publish_tmp` directory under the target PRFS parent with
  `shutil.copytree(...)`;
- `_commit_run_group_publish(...)` then promotes the temporary group with
  `os.replace(...)`.

This gives a simple group-level publish boundary, but every Zarr chunk and
metadata file is still created individually on PRFS. System tools such as
`tar`, `rsync`, or `cp -a` may reduce Python overhead, but they do not remove
the underlying filesystem-object count unless the storage layout changes.

One important observation: dense `masks_roi` is not the worst offender. It is
only about `1.9k` files. The derived component subtree dominates file count,
especially component metrics, reason/fingerprint arrays, and ragged contour
outputs.

The component subtree is not uniform. A direct count on the published run gives
the following largest contributors:

| component surface | files |
|---|---:|
| `subject_body/contours` | 10438 |
| `subject_body/finalization_metrics` | 8948 |
| `swim_bladder/finalization_metrics` | 8848 |
| `swim_bladder/contours` | 3578 |
| `eye_left/contours` | 2175 |
| `eye_right/contours` | 2175 |
| `eye_left/geometry` | 940 |
| `eye_right/geometry` | 940 |

This makes the first storage targets clearer. Full ragged contours and sealed
finalization metrics account for much more file pressure than the authoritative
dense masks. Common run-level metrics are also expensive: six arrays under
`metrics/`, currently chunked by `[256, 1, ...]`, account for about `11.3k`
files because the component axis is physically separated.

## Next File-Count Layout Decision

The next optimization is storage layout, not another compute backend. Apply a
mutability-aware policy rather than one row chunk to every array.

| surface class | examples | proposed layout/write owner | edit behavior |
|---|---|---|---|
| authoritative pixels | `masks_roi` | keep `[256, 1, 512, 512]`; component-separated chunks | one accepted component edit rewrites one 64 MiB logical mask chunk |
| live edit control | `edit_applied`, `manual_override`, `source_row_stale`, `row_revision` | keep modest row chunks and component separation | updated synchronously under the run lock |
| fixed-shape derived metrics | run `metrics/*`, component spatial/topology metrics, source fingerprints | `[16384, 4, ...]` for run-level arrays and `[16384, ...]` for component arrays; driver-owned merged write | never synchronously rewritten by canonical saves; mark stale and refresh explicitly |
| derived eye geometry/relations | ellipse arrays and eye-pair metrics | `[16384, ...]`; driver/maintenance write | stale after relevant mask edits |
| sampled display contour | bounded fixed-K points plus validity/count | component-specific K; benchmark 256/1024/4096 row chunks | default derived contour; stale after relevant mask edits |
| optional full contour | ragged `contours/{ptr,len,points_xy}` | explicit analysis/archive/export build only | no in-place row edits; regenerate packed artifact |

The measured common `metrics/` arrays contain 11250 payload chunks plus seven
metadata files in the reference layout. The new-run writer uses
`[16384, 4, ...]`, requiring eight payload chunks per array: the full-run canary
measured 55 total files rather than 11257.
`process_shards` workers return their small fixed-shape metric/fingerprint
payloads without writing those arrays. The driver merges all rows, stacks the
full component axis, and writes each run metric once; component metrics and
fingerprints are likewise driver-owned `[16384, ...]` surfaces.

The edit-safety prerequisite is also implemented. External single-component
writeback, in-process interactive review saves, and browser checkpoint applies
persist only authoritative dense pixels plus minimal edit/revision/source-sync
state, then mark metrics and contours stale. Explicit metadata/source-sync
maintenance must opt into derived refresh. Existing runs are not silently
rechunked; the layout applies when a new refined run is created.

Full ragged contour points are the largest remaining single contributor:

| component | current `points_xy` payload files | point rows |
|---|---:|---:|
| subject body | 9494 | 38884575 |
| swim bladder | 2634 | 10785948 |
| eye left | 1234 | 5054400 |
| eye right | 1234 | 5053479 |

The current point chunk is effectively 4096 `(x, y)` rows. Across all four
components that produces 14596 payload files, plus the `ptr`/`len` row-index
chunks. These files should not be part of the default publication layout.

The contour policy decision is:

- default finalized/published runs materialize a fixed-K sampled contour cache;
- full ragged contours are opt-in for analysis, archive, or export builds;
- sampled contours are derived directly from dense masks, not by first writing
  the full ragged representation;
- eye ellipse/angle geometry continues to be measured from dense masks or
  assignment-time geometry, not from sampled display contours;
- interactive dense-mask saves mark sampled and full contour artifacts stale
  and do not regenerate either inside the synchronous save transaction.

Current sampled-K candidates are body `128`, swim bladder `32` for display or
`64+` for geometry-sensitive workflows, and each eye `64`. A fixed-size array
per component avoids ragged `ptr`/`len` reads and bounds Crimson's overlay I/O.
Benchmark sampled-contour row chunks `256`, `1024`, and `4096`; `1024` is the
leading candidate because its uncompressed point payload is about 1 MiB for a
body chunk, 512 KiB for an eye chunk, and 256 KiB for a swim-bladder chunk.

If optional full-ragged builds still need fewer files, their independent point
chunk alternatives are:

| point chunk | aggregate payload chunks | uncompressed bytes per chunk | tradeoff |
|---:|---:|---:|---|
| 65536 | about 915 | 512 KiB | leading optional full-ragged canary |
| 262144 | about 231 | 2 MiB | stronger publication reduction, heavier on-demand row reads |
| 1048576 | about 59 | 8 MiB | publication-oriented; too large for routine exact interactive contour reads unless visibility gating and caching are proven |

The first default-layout canary should combine large fixed-shape derived metric
chunks with sampled contours and omit full ragged contours. Compare full file
inventory, finalizer time, sampled single-row/window read latency, Crimson
overlay fidelity, and hidden-target PRFS copy time. Test 65536-point chunks only
in a separate opt-in full-ragged build.

## Crimson Live-Reader Implications

The best active-review policy is still dense-first. The published run's
`masks_roi` chunks are `[256, 1, 512, 512]`, and Crimson already prefers dense
`masks_roi` over bitpacked or RLE stores. Dense fixed-row addressing has been
the most predictable measured PRFS playback path, and it is also the only
editable mask-pixel authority.

The large `components/` tree is not entirely cold from Crimson's point of view,
however. Crimson currently:

1. defers component contours and eye ellipse geometry during initial archive
   loading;
2. requests those optional overlays automatically after the archive is loaded;
3. reads each component's full contour `ptr` and `len` arrays;
4. attaches ragged `points_xy` reads to mask-chunk preparation once contours
   are available.

That means optional derived geometry can create background and playback I/O
even when the user primarily needs mask fills. The recommended reader split is:

| read tier | surfaces | policy |
|---|---|---|
| required live placement | run attrs, `frame_indices`, `source_crop_row_ids`, crop frame/ROI placement, `available_channels` | read once, cache, and pin to one completed run |
| required mask display/edit | dense `masks_roi` | asynchronous row-window prefetch during playback; blocking exact read while paused/editing |
| optional display geometry | eye ellipse geometry, sampled or full contours | request only when the corresponding overlay is visible; keep a cache independent from mask fills |
| analytical/QC data | component metrics, finalization metrics, reasons, fingerprints, full geometry diagnostics | never load automatically for ordinary playback; fetch on demand for an inspection panel or worker |

Crimson should therefore keep mask-fill, contour, and eye-geometry caches
independent. A mask-fill cache miss should not trigger contour I/O. When dense
masks are already decoded, a locally derived display contour is also a useful
fallback that avoids a second PRFS read path. Stored full contours remain
available when exact contract geometry is explicitly requested.

The current Crimson implementation confirms where to split the paths:

- `src/red.cpp` calls
  `requestRefinedSubjectMaskOptionalOverlayPrefetch()` whenever the Zarr is
  loaded; the request becomes a no-op only after it has been requested, is
  loading, or has completed;
- `src/zarr_loader_eye_keypoint.cpp` defers optional overlays during the initial
  refined-mask load by default, but its background optional-overlay worker then
  reads the full `ptr` and `len` arrays for every available component and full
  eye `ellipse_params`/`ellipse_success` arrays;
- once optional contours are published into loader state, mask-chunk loading
  scans contour row metadata and reads a covering ragged `points_xy` interval
  for each component with rows in that chunk;
- the loader already tracks optional-overlay state separately, so the natural
  next change is to request contour and ellipse surfaces from explicit overlay
  visibility transitions instead of the unconditional loaded-Zarr frame-loop
  call.

This should be two independently requestable optional capabilities, not one
combined optional-overlay job: contour visibility should not load eye geometry,
and eye-axis visibility should not load contours. Cached dense masks can render
fills without either capability.

Run discovery should remain metadata-driven:

- resolve the parent `latest` or review-status pointer;
- require `palette_run_completion_status == "complete"` for normal user-facing
  selection;
- pin that exact run for the loaded session instead of following `latest`
  during playback;
- exclude hidden `.publish_tmp` directories and incomplete runs from any
  filesystem fallback;
- expose an explicit refresh/reopen action when a newer run is published.

Crimson currently resolves the parent pointers first, then accepts the first
candidate with readable attrs. If no pointer candidate exists, its filesystem
fallback enumerates directories containing `masks_roi`, bitpacked masks, or an
RLE group and sorts by name. That fallback does not currently require
`palette_run_completion_status == "complete"` and does not exclude dot-prefixed
`.publish_tmp` directories. The reader should centralize candidate validation
so both pointer and filesystem paths apply completion, hidden-name, required
surface, and stale-compact-cache checks before a run is pinned.

## Interactive Editing And Chunk Rewrite Policy

The concern about rewriting large chunks during interactive saves is valid.
Zarr writes are chunk writes: changing one logical row normally performs a
read-modify-write of every physical chunk intersecting that row. The correct
policy depends on whether an edit preserves row cardinality and on the logical
size of one physical chunk.

The three annotation families have different current behavior:

| artifact and operation | current/observed behavior | recommended write model |
|---|---|---|
| move/resize an existing bounding box | Crimson's current manual refined-detection save clears and rewrites the complete `instances/` subgroup | persist a correction event and add a Palette-owned row-update path for the existing stable row; avoid whole-group replacement for one box |
| add/delete a bounding box | changes row count and downstream identity/alignment | append a correction event, then materialize a new frame-sorted refined-detection run; never insert into the middle of active arrays |
| edit keypoints for an existing ROI | Crimson writes one row across keypoint coordinate and quality arrays | keep row-local writes; current `[1024, 5, 2]` float64 keypoint chunks are only about `80 KiB` logical per coordinate array and are not intrinsically too large |
| add keypoints for a new detection | normally follows a cardinality-changing bbox/crop addition | carry the addition through the correction log and create the new keypoint row during materialization |
| edit one existing mask component | Palette writes one dense `masks_roi[row, component]` plane, plus current metric/review updates | keep dense `masks_roi` authoritative, use component-separated chunks, write only the selected component, and keep the synchronous metadata set minimal |
| add a mask for a newly added detection/crop row | changes alignment across detection, crop, keypoint, and mask runs | create it in the next materialized run rather than resizing or inserting into the active refined mask run |

### Dense Mask Chunk Tradeoff

For a four-component, 120221-row, `512 x 512` dense `uint8` mask array, the
row chunk controls both single-edit amplification and file count:

| row chunk | logical bytes rewritten for one component chunk | maximum dense chunk files | playback covered at 100 FPS |
|---:|---:|---:|---:|
| 256 | 64 MiB | about 1880 | 2.56 s |
| 128 | 32 MiB | about 3760 | 1.28 s |
| 64 | 16 MiB | about 7516 | 0.64 s |
| 32 | 8 MiB | about 15028 | 0.32 s |

The file counts are maxima assuming every chunk is materialized; fill-value
elision can make the observed count smaller. Compression also makes the stored
payload much smaller than the logical bytes, but the writer still pays the
decode/modify/encode cost for the physical chunk.

`[1, 1, 512, 512]` is still the wrong default: it minimizes one-save
amplification but would allow roughly 480000 dense mask chunk files. The useful
next canary is `64` versus `128` rows with component chunks of `1`. A `64`-row
chunk is a plausible editable compromise, but it should be promoted only after
measuring cold random save latency, sequential playback continuity, cache
memory, and publication time on PRFS. A `32`-row chunk may have insufficient
prefetch headroom at 100 FPS when filesystem latency varies.

### Minimal Synchronous Mask Save

An accepted live mask save should synchronously persist only what is needed to
establish durable pixel truth and concurrency state:

1. validate the requested run, row, component, and expected row revision;
2. update only dense `masks_roi[row, component]`;
3. update minimal review state such as `edit_applied`, `row_revision`, editor,
   reason, and timestamp;
4. mark bitpacked/RLE masks, metrics, and contours stale for the affected
   row/component;
5. return the committed row revision and stale-surface status to Crimson.

Full metrics, fingerprints, compact masks, and contours should refresh during
explicit validation, promotion, or background maintenance. Computing a metric
for immediate UI feedback is fine, but it does not need to cause a collection
of synchronous Zarr writes before the pixel save can succeed.

Crimson's current mask writeback reads the full four-component row before
changing one component. The Palette write boundary should instead read and
compare only the selected component unless a validation rule truly requires
cross-component pixels. This preserves the dense authority while avoiding
three unnecessary component-chunk reads.

The current working-tree Palette writeback candidate implements that boundary:

- it takes an optional expected component-row revision and rejects a stale
  request before mutation;
- it holds a Palette-owned filesystem lock for the refined run while updating
  pixels, revisions, and run-level stale attrs;
- it reads and writes only `masks_roi[row, component]`, not the full component
  stack;
- it synchronously updates `edit_applied`, `manual_override` when present, and
  component-row revision/timestamp/reason tracking;
- it leaves existing metrics and contours unchanged and marks derived surfaces
  stale;
- it reports logical mask bytes, dense chunk shape, touched chunk count,
  uncompressed bytes covered by those chunks, lock wait/hold time, read time,
  authoritative-write time, validation time, and total save time.

The initial lock is intentionally refined-run scoped. This is conservative and
safe for both physical mask chunks and shared run attrs, but it serializes edits
that touch different chunks. Move to finer physical-chunk locks only after stale
scope is stored in row-local arrays or another transaction-safe structure so
concurrent writers cannot lose run-level stale metadata.

Crimson's current `SubjectMaskWritebackRequest` does not carry an expected row
revision. The command client parses revision values from Palette's response but
does not pass `--expected-row-revision`, and the edit session tracks only its
local preview revision. Before the optimistic check becomes mandatory, Crimson
must load the selected component's durable `row_revision`, pin it when editing
starts, send it with the apply request, and replace its pinned value with
Palette's returned `row_revision_after`. A stale-revision rejection should keep
the preview, reload canonical pixels/revision, and ask the user to reconcile or
reapply rather than silently overwriting the newer row.

### Cardinality Changes And Materialization

Adding or deleting detections is not primarily a chunk-size problem. It is an
identity and alignment problem. Inserting a detection row can shift later rows
and invalidate crop, keypoint, mask, contour, and track references.

The safe pattern is:

```text
live edit
-> append-only correction/audit event with stable target identity
-> Crimson overlays the frame-local correction on the pinned materialized run
-> Palette validates and accepts/rejects the event
-> Palette materializes accepted events into a new frame-sorted run
-> downstream keypoint/mask/metric workers derive new versioned runs
-> parent latest pointer moves only after validation and completion
```

For existing subject-mask rows, the accepted dense edit still mutates
`masks_roi`; an event log is an audit/recovery surface, not a competing mask
authority. New mask rows caused by detection cardinality changes belong in the
next materialized dense run.

### Concurrency And Failure Safety

Row revisions alone are not sufficient when multiple users can save rows that
share one physical chunk. Two processes can read the same old chunk, modify
different rows, and race to write stale chunk payloads. Palette's production
writeback boundary should therefore:

- serialize writes by physical `(run, array, chunk-coordinate)` ownership;
- require an expected row revision and reject/reload stale requests;
- re-read the authoritative row after acquiring the chunk lock;
- batch multiple edits to the same chunk when practical, writing it once;
- record a pending/committed/failed audit event because updates across multiple
  Zarr arrays are not one atomic transaction;
- write the authoritative pixels/coordinates first, then publish the new row
  revision and stale-derived-state marker.

Crimson should not write mutable Zarr chunks directly in production. A single
Palette command or service should own validation, locking, durable mutation,
revision updates, and recovery reporting.

### Mutable Versus Sealed Storage

One chunking or sharding policy should not serve every surface:

- authoritative mutable bbox/keypoint rows: modest unsharded row chunks;
- authoritative mutable dense masks: modest row chunks, component axis `1`,
  full spatial planes;
- append-only correction/audit events: append-oriented chunks with a frame
  lookup index;
- sealed analytical metrics and finalization outputs: large row chunks or
  modest Zarr shards;
- display-only sampled contours or bitpacked masks: row windows aligned to
  Crimson prefetch, and replaceable when stale;
- full ragged contours: sealed analysis/archive data, not part of the required
  live save or playback path.

Large shards should not contain mutable edit targets. Rewriting one inner chunk
inside a shard can amplify a small edit into a much larger shard update.

## Current Contour Position

The canonical writer now has an opt-in fixed-K sampled contour surface with
`points_xy[N,K,2]`, `valid[N]`, and `source_point_count[N]`. It uses 1024-row
physical chunks and can run through either serial or `process_shards`
postcompute. Full ragged contours are controlled independently. Production
wrappers retain full ragged output until Crimson reads the sampled schema; the
default flip is therefore a coordinated reader/writer rollout, not a Palette-
only change. The July 8 contour diagnostic supports these candidates:

| component | current candidate K | safe use |
|---|---:|---|
| `subject_body` | 128 | display and compact body-outline summary |
| `swim_bladder` | 32 for display, 64+ for geometry-sensitive work | compact display or conservative shape work |
| `eye_left` | 64 | review and geometry-facing sampled contour candidate |
| `eye_right` | 64 | review and geometry-facing sampled contour candidate |

These are not replacements for dense masks. Dense masks remain pixel authority,
and eye geometry continues to be computed from dense masks or assignment-time
geometry. Existing consumers that truly require every raw boundary vertex must
request a full-ragged build explicitly; display/summary consumers should move
to the fixed-size sampled representation.

## What Is Solved

- CPU-only subject-mask finalization can run as its own LSF job without a no-op
  GPU inference dependency.
- Finalization can write to local scratch, publish to PRFS, validate, refresh
  registry views, and clean up scratch.
- The scratch path is now checked for writability, not just existence.
- The current 16-worker process-sharded finalizer reaches about `206 rows/s` on
  a full GoodCopBadCop recording while preserving dense masks, eye geometry, and
  component contours.
- Registry performance and component-quality views are refreshed after publish.
- Publication reports now include run-group storage and copy/commit telemetry.
- The full finalizer benchmark now retains refined-output file-count and chunk
  layout telemetry before deleting its local staged archive.
- The new-run finalization and common-metric writers use driver-owned large
  sealed chunks without allowing workers to share a physical Zarr metric chunk;
  production promotion remains gated on the full-run canary.
- New run-level common metrics use `[16384, C, ...]`; component
  spatial/topology metrics and source fingerprints use `[16384, ...]`.
- The clean-commit 120221-row staged canary completed successfully with 28119
  files: 57.5% fewer than the published reference and 42.2% fewer than the
  first metric-layout staged run.
- Interactive and browser mask saves leave derived arrays byte-unchanged and
  mark their affected row/component scope stale; explicit maintenance refresh
  remains available.
- The Palette-owned single-component writeback candidate provides revision
  checks, conservative cross-process serialization, minimal authority writes,
  stale-derived markers, and write-amplification telemetry.

## What Is Not Solved

- Publication still takes several minutes because the refined run contains
  about `66k` files.
- Finalization-only runs currently do not create a refined-run handoff package
  on NRS; the workflow event reported `handoff_package_count=0`. NRS packages
  are still primarily tied to raw subject-mask inference handoff.
- Full ragged contours, reason columns, and remaining eye-geometry surfaces
  still need their default publication layout reduced.
- Fixed-K contours are implemented and unit-validated, but the full-run file
  inventory/fidelity canary and Crimson reader migration are still pending.
- The production wrapper still enables full ragged contour generation until
  Crimson can consume fixed-K sampled contours.
- Crimson's current manual bounding-box save rewrites the complete curated
  `instances/` subgroup rather than applying a row-local update.
- Dense mask save amplification has not yet been benchmarked for editable row
  chunks `64` versus `128` on PRFS.
- The working-tree writeback lock is run-scoped rather than per-physical-chunk;
  this is safe but limits concurrent saves until stale metadata is made
  transaction-safe at finer granularity.
- The PRFS hidden-target publication canary has not yet been run for the
  expanded common-metric layout.
- Optional Crimson contour and eye-geometry reads are deferred from startup but
  are still requested automatically rather than only when visible.

## Practical Next Steps

1. Use the existing publish telemetry in the next layout canary.
   Compare file count, total bytes, top-level subtree file counts, publish
   backend, and publish duration for the 28119-file expanded layout against both
   the 66087-file published reference and the 48639-file first-layout staged
   run.

2. Separate Crimson's mask-fill, contour, and eye-geometry read paths.
   Do not automatically attach optional contour reads to mask-chunk prefetch.
   Record independent cold-read, cache-hit, and dropped-overlay telemetry for
   each surface.

3. Add interactive write-amplification telemetry and canaries.
   For bbox, keypoint, and mask saves, record logical row bytes, physical chunk
   shape, touched chunk count, read/decode/write duration, and total save
   duration. Compare dense mask row chunks `64` and `128` on PRFS before changing
   the editable default.

4. Replace whole-group bbox saves with correction events and materialization.
   Keep a row-local Palette-owned path for moves/resizes of stable boxes. Route
   add/delete operations through the append-only correction log and publish a
   new frame-sorted run after accepted changes.

5. Add Palette-owned concurrency control.
   Use expected row revisions plus per-physical-chunk serialization. Add
   deterministic same-chunk batching and failure-recovery tests before enabling
   multi-user production writes.

6. Benchmark publish methods without changing the storage contract.
   Compare current Python `copytree`, system `cp -a`, `rsync -a`, and
   `tar | tar` extraction into a `.publish_tmp` directory. This will tell us
   how much time is Python overhead versus unavoidable PRFS metadata overhead.

7. Continue the sealed-derived layout conversion.
   Common metrics, component metrics, fingerprints, and finalization metrics
   now use driver-owned large chunks for new runs. Keep mutable
   bbox/keypoint/mask authorities on modest unsharded chunks; next evaluate
   reason columns and remaining sealed eye-geometry arrays without changing
   existing archives in place.

8. Promote the implemented sampled-contour surface.
   Run the full staged inventory/fidelity canary for body `K=128`, eyes `K=64`,
   and swim bladder `K=32`; migrate Crimson's overlay reader to the fixed-K
   schema; then make full ragged contour generation an explicit
   analysis/archive/export option.

9. Consider an NRS refined-run package path.
   If finalization produces a tar package on NRS after local scratch compute,
   PRFS publication can be delayed, retried, or benchmarked independently. This
   does not eliminate the final PRFS file creation cost, but it gives a clean
   handoff artifact and makes publish experiments safer.

10. Keep clipped-collection subject-mask finalization shard-aware.
   For clipped archives, per-clip refined outputs should remain independent
   until a collection finalizer validates all expected shards. The collection
   merge/publish step should avoid rewriting work already completed per clip.

## Current Recommendation

Keep the current CPU-only finalizer as the production path for now. It is
correct, validates cleanly, preserves required dense masks/eye geometry/contours,
and is significantly faster than the recent modern cluster canaries.

For live review, keep dense `masks_roi` as the authoritative editable surface,
but do not assume the finalizer's `256`-row production chunk is also the best
interactive chunk. Benchmark `64` and `128` with real single-row saves and
playback before choosing the editable default. Keep component axis `1`, avoid
sharding mutable arrays, and route all production mutation through a
Palette-owned, revision-checked, per-chunk-serialized write boundary.

The immediate Crimson-side win is to separate required mask-fill reads from
optional contour and eye-geometry reads. The immediate bbox-side win is to stop
rewriting the whole curated instances group for one existing-box edit and to
route cardinality changes through correction events plus materialization.

Publication optimization should target sealed derived-output layout and file
count, not the core connected-components primitive or dense mask authority.
The higher-leverage structural win is reducing the component and common-metric
subtree file count through larger chunks, packed derived arrays, sampled
contours, or modest shards used only for stable read/archive surfaces.
