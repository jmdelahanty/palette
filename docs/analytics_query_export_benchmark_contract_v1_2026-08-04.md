# Analytics Query Export Benchmark Contract V1

Date: 2026-08-04

Status: executable benchmark runner implemented; no physical profile, source
authority, selector, registry row, or canonical archive is promoted or mutated.

## Purpose

The four exact immutable query products now share one closed benchmark runner:

- `eye_trace_samples`;
- `kinematics_samples`;
- `activity_spatial_time_bins`; and
- `tail_trace_samples`.

The runner invokes each maintained publisher rather than reproducing its write
logic. It records the publisher's ten non-overlapping process-local phases,
wraps publication and every reader in process-tree resource telemetry, and
launches fresh-process reads against only the manifest-enumerated Parquet
parts.

Implementation:

```text
fisheye.diagnostics.benchmark_analytics_query_exports
```

## Closed Request

Every request is a canonical-JSON digest envelope. It binds:

- one of the four exact families;
- `representative_short` or `full_duration` scale;
- one recording analysis Zarr;
- exact named source runs and scope;
- exact family-specific publisher parameters;
- one immutable export-run identity;
- separate benchmark publication, node-local scratch, and evidence roots;
- deterministic random-frame and frame-window workloads;
- fresh-process repetition count and cache-state declaration; and
- requested workers, allocated slots, and process sampling interval.

Unknown or missing source/parameter fields fail closed. Every writable path
must contain an explicit benchmark namespace, must be outside the source
archive, and must not overlap the other writable roots. An existing export
manifest cannot be replaced.

`build-request` supplies the maintained CLI boundary. For example:

```bash
scripts/py -m fisheye.diagnostics.benchmark_analytics_query_exports \
  build-request \
  --family kinematics_samples \
  --scale full_duration \
  --zarr RECORDING_analysis.zarr \
  --track-kinematics-run EXPLICIT_RUN \
  --track-scope offline \
  --requested-sample-rate-hz 10 \
  --source-window-rows 131072 \
  --export-root /benchmark/publication \
  --scratch-root /node-local/benchmark/scratch \
  --benchmark-output-dir /benchmark/evidence \
  --export-run-id immutable_export_id \
  --output /benchmark/request.json
```

Execution is a separate command:

```bash
scripts/py -m fisheye.diagnostics.benchmark_analytics_query_exports \
  matrix --request /benchmark/request.json
```

## Publication Evidence

One fresh process calls the real maintained exporter. The result binds:

- the request digest;
- immutable manifest file and canonical payload digests;
- row and part counts;
- the exporter's exact final decoded validation;
- all ten publisher phase durations;
- publisher wall/CPU time and process-local peak RSS; and
- process-tree wall/CPU/RSS/thread samples from the outer wrapper.

Runtime telemetry is result-only. The runner rejects an immutable export
manifest containing that field.

## Reader Workloads

Each fresh-process reader first executes the complete maintained export
validator. It then performs:

1. footer/schema opens for every selected part;
2. deterministic random-frame predicates over family-specific hot columns;
3. deterministic contiguous frame windows over those columns; and
4. a bounded-batch full-column scan with one logical stream digest.

The frame extent comes only from Parquet row-group min/max statistics. Missing
statistics fail closed rather than triggering an unrecorded full axis scan.
The result records latency samples and p95/median summaries, decoded rows and
bytes, throughput, CPU, RSS, manifest/part object count, and apparent/allocated
filesystem bytes. All repetitions must agree on the manifest and full-scan
digest.

## Nonmutation And I/O Boundary

Before and after execution, the runner hashes direct `zarr.json` metadata for
the root, selected parents, and every selected source subtree. Any change
invalidates the matrix.

The local Parquet reader cannot report network requests or compressed network
transfer. Those fields remain exact JSON nulls. Linux process-requested file
bytes must be measured separately with `trace_storage_io`; mounted Crimson
request/transfer behavior remains separate consumer evidence.

No matrix result authorizes promotion. `promotion_authorized` is required to
remain false.

## Full-Duration Source Preflight

The existing Sleepyfish archive does not yet contain an authority that can
feed these exact query exporters without a benchmark-only compatibility or
scientific rematerialization:

- selected eye angles are compact v5; the query contract requires compact v7;
- selected tail kinematics are v1; the query contract requires v2;
- all inspected full-duration track authorities store `positions_mm` as
  `float32[N,2]`, while the frozen maintained track and kinematics-query
  contracts require `float64[N,2]`; and
- activity/spatial export inherits that exact track-position requirement even
  though its selected swim-bout v8 run correctly binds the older explicit
  track-manifest digest.

The real kinematics source binder rejected the newest sealed track authority
at `positions_mm` before publication. This is the desired fail-closed outcome.
The benchmark runner must not weaken the dtype, silently cast the live
authority, or stamp the historical archive as current.

The next fixture step is therefore a copied, selector-isolated benchmark
archive. Any lossless float32-to-float64 compatibility widening must be labeled
as such and cannot serve as production-authority or profile-promotion evidence.
Eye and tail require their maintained semantic rematerializers, not metadata
stamping.

## Validation

Focused tests cover all four closed requests, recomputed-digest field-set
tampering, benchmark-path isolation, the request CLI, deterministic multipart
random/window/full-scan reads, exact source-metadata guard coverage, and
fail-closed missing Parquet statistics. The current focused result is 10/10.

