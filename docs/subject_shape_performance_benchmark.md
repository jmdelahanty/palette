# Subject-shape compute and storage benchmark

This runbook separates four concerns that were previously coupled in
`analysis/subject_shape_runs`:

1. source-mask reads;
2. biological geometry computation;
3. logical compute/write blocks;
4. final physical Zarr shards and publication.

The benchmark tools never write to their authoritative source Zarrs. Compute
variants write bounded standalone outputs on node-local scratch. Storage
variants clone a completed immutable run after staging it to node-local
scratch, and change only indexed physical shard spans.

## Sleepyfish baseline

The completed `subject_shape_sleepyfish_core_canary_20260713_01` run contains
1,169,010 rows. It used eight Dask processes, 256-row worker blocks, and 4,567
tasks. Its persisted layout has 421,379 files (421,264 payload files), occupies
2,385,287,470 physical bytes, and stores 4,567 detailed task timing records in
a 3,959,291-byte root `zarr.json`.

The shape stage took 9,532.44 seconds (122.63 rows/s). Persisted task timings
attribute about 39% of summed worker time to centerline extraction. The
enclosing eight-slot LSF workflow reached 583 threads and used 138,974.59 CPU
seconds over 11,038 wall seconds, so native-library oversubscription must be
controlled before worker scaling is interpreted.

The modern writer persists a compact timing summary by default. Full per-chunk
records are embedded only when `--include-chunk-timings` is explicitly
requested for diagnostics, preventing routine run metadata from growing by
several megabytes.

The tail-kinematics materializer needs only a subset of the shape run, but that
subset still contains 60,532 physical files for 641,608,777 bytes. This is why
subject-shape physical layout materially affects downstream staging.

## Bounded compute benchmark

`fisheye.diagnostics.benchmark_subject_shape_compute` maps an aligned source
row window onto row zero of disposable standalone outputs. It invokes the real
component, body-frame, snout, caudal-anchor, centerline, spline, relation, and
Zarr-write paths. Each variant records separate source-read, compute, and
persist timings and hashes every output array against the first variant.

Variant syntax is:

```text
NAME:WORKERS:BLOCK_ROWS:NATIVE_THREADS[:FLAGS]
```

Supported flags are `crop` and `per-task-open`. Compute blocks must be a
multiple of the 256-row logical output chunk, so every process owns whole,
non-overlapping physical chunks.

The cluster wrapper's default 32,768-row matrix compares:

- per-task versus persistent source/output handles;
- 256-, 512-, and 1,024-row compute blocks;
- one versus two native threads per worker;
- 8, 16, and 32 worker processes;
- the foreground-cropped centerline prototype.

Render or submit it with:

```bash
scripts/submit_subject_shape_compute_benchmark_bsub.sh \
  --zarr /path/to/recording_analysis.zarr \
  --refined-run refined_subject_masks_run \
  --benchmark-id subject_shape_compute_canary_YYYYMMDD_01 \
  --submit
```

The wrapper refuses execution outside LSF, pins the clean Palette commit,
rejects thread budgets larger than the allocation, writes outputs only to
node-local scratch, and retains the report under the recording's
`.processing_logs/subject_shape_benchmarks` directory.

## Physical-shard benchmark

`fisheye.diagnostics.benchmark_subject_shape_sharding` preserves every logical
chunk, codec, group attribute, and decoded value while testing outer indexed
row shards of 16,384 through 1,048,576 rows. Copy tasks own complete,
non-overlapping physical shards. It benchmarks random-row, contiguous-window,
and full-scan reads and can optionally time checksum-validated publication
copies.

The cluster wrapper stages the immutable source run to node-local storage once
before constructing candidates:

```bash
scripts/submit_subject_shape_sharding_benchmark_bsub.sh \
  --source-run-path /path/to/analysis/subject_shape_runs/run \
  --benchmark-id subject_shape_sharding_canary_YYYYMMDD_01 \
  --submit
```

The full decoded comparison is acceptance evidence for a new layout, not a
normal production step. Production writers should use bounded local/final
contract validation rather than rereading historical runs.

## Interpretation and promotion gates

- Do not select a shard size from file count alone. Check random reads,
  contiguous reads, full scans, publication, and task/retry granularity.
- Do not interpret worker scaling until OpenCV and BLAS thread counts are
  explicit and the requested native-thread budget fits the LSF allocation.
- The foreground-crop centerline path may be promoted only if every decoded
  output array exactly matches the current method on representative real rows.
- If an algorithmic optimization changes decoded geometry, it is a new method
  version and requires a scientific validation decision rather than a storage
  migration.
- Final production design should retain small logical blocks, assemble large
  physical shards on node-local scratch, validate there, and publish the
  completed run atomically.

## Sleepyfish benchmark evidence (2026-07-15)

The bounded compute matrix ran as Citrus job `153101220` from Palette commit
`c910935895b30d72da82dd603e6f1600c0ebc06e`. It processed source rows
524,288–557,055 (32,768 rows); every output array from all eight variants had
the same decoded SHA-256 digest as the baseline.

| Variant | Wall seconds | Rows/s | Result |
| --- | ---: | ---: | --- |
| per-task open, 8 workers, 256 rows, 1 native thread | 61.89 | 529 | exact |
| persistent, 8 workers, 256 rows, 1 native thread | 60.65 | 540 | exact |
| persistent, 8 workers, 512 rows, 1 native thread | 59.97 | 546 | exact |
| persistent, 8 workers, 1,024 rows, 1 native thread | 59.98 | 546 | exact |
| persistent, 8 workers, 1,024 rows, 2 native threads | 60.21 | 544 | exact |
| persistent, 16 workers, 1,024 rows, 1 native thread | 31.41 | 1,043 | exact |
| persistent, 32 workers, 1,024 rows, 1 native thread | 18.60 | 1,762 | exact |
| cropped centerline, 32 workers, 1,024 rows, 1 native thread | 14.37 | 2,281 | exact |

Foreground cropping reduced summed centerline compute from 222.32 to 90.82
worker-seconds (59%) and reduced variant wall time by 23%, or increased
throughput by 29%. Two native threads did not help. Persistent handles and
larger compute blocks helped only modestly, while 16 and 32 single-threaded
workers scaled well. The job peaked at 36.7 GB. Its LSF record also exposed
retained native-library thread pools despite active `threadpoolctl` limits, so
the wrapper now sets an import-time native-thread ceiling before Python starts.

The physical-shard matrix ran as Citrus job `153101250`. The authoritative
source was read-only, the node-local stage was checksum-verified, and every one
of 102 arrays matched exactly in every candidate. Copying the existing 421,379
files to node-local storage took 922 seconds before independent checksum
validation. The source occupied 3.84 GB of allocated filesystem space for
2.39 GB of apparent data because most payload files were tiny.

| Outer row shard | Payload files | Largest shard | Random trial | 1,024-row windows | Full scan |
| ---: | ---: | ---: | ---: | ---: | ---: |
| unsharded source | 421,264 | 0.11 MB | 0.148 s | 0.098 s | 9.94 s |
| 16,384 | 6,913 | 7 MB | 0.185 s | 0.123 s | 8.41 s |
| 65,536 | 1,729 | 28 MB | 0.186 s | 0.121 s | 10.61 s |
| 131,072 | 865 | 55 MB | 0.185 s | 0.120 s | 10.61 s |
| 262,144 | 481 | 107 MB | 0.187 s | 0.121 s | 10.62 s |
| 524,288 | 289 | 212 MB | 0.186 s | 0.122 s | 10.63 s |
| 1,048,576 | 193 | 418 MB | 0.187 s | 0.122 s | 10.66 s |

The indexed sharding codec preserved 256-row inner-chunk access: interactive
timings were nearly flat as outer shards grew. The recommended balanced layout
is therefore 131,072 outer rows with 256-row inner chunks. It reduces payload
files about 487-fold and allocated space to 2.39 GB while keeping the largest
retry/rewrite unit near 55 MB. A 262,144-row layout is reasonable for
bulk-oriented immutable products, but 524,288 and 1,048,576 rows provide too
little additional benefit for their 212–418 MB retry units.

For production, use 1,024-row logical compute blocks, single-threaded native
libraries, and up to 32 worker processes on a 32-core node. Compute blocks must
not independently write into a shared 131,072-row physical shard. Either one
writer must own the complete outer shard, or workers must create temporary
block outputs that are assembled deterministically on node-local storage before
validation and atomic publication.
