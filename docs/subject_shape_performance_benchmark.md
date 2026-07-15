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
