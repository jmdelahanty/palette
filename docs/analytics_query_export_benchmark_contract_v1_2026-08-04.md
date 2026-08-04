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

For `kinematics_samples`, `representative_short` is an executable scale
contract rather than a free-form label. It requires one explicit half-open
acquisition-frame interval of exactly 200,000 frames. The maintained exporter
persists that interval in projection schema v2 and independent validation
rejects every output row outside it. `full_duration` requests reject a bounded
interval. Existing unbounded projection-v1 exports remain valid and unchanged.
The initial bounded implementation still streams and rehashes the complete
selected source surfaces, so its writer time is deliberately conservative; it
does not trade source-integrity validation for a better short-scale number.

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

A representative-short request additionally supplies:

```text
--scale representative_short \
--source-frame-start 0 \
--source-frame-stop-exclusive 200000
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
- process-tree wall/CPU/RSS/thread samples from the outer wrapper; and
- process-tree requested characters, read/write syscall counts, and
  OS-reported storage bytes from resource-telemetry v2.

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
transfer. Those fields remain exact JSON nulls. Resource-telemetry v2 records
process-tree requested characters/syscalls and OS-reported storage bytes. The
requested-character counter includes cache-served reads, while storage bytes
depend on what the host OS reports; neither may be labeled network transfer.
Path-attributed Linux reads remain the responsibility of `trace_storage_io`,
and mounted Crimson request/transfer behavior remains separate consumer
evidence.

No matrix result authorizes promotion. `promotion_authorized` is required to
remain false.

## Full-Duration Source Preflight

The existing Sleepyfish archive does not yet contain eye and tail authorities
that can feed all four exact query exporters without scientific
rematerialization:

- selected eye angles are compact v5; the query contract requires compact v7;
- selected tail kinematics are v1; the query contract requires v2.

The real kinematics source binder rejected the newest sealed track authority
at `positions_mm` before publication. That fail-closed result exposed a false
new float64 declaration: the crop contract, writer, and all 12 inspected
full-duration track authorities use float32. The maintained exact track and
kinematics-query contracts now require float32 without casting. See
`track_coordinate_precision_contract_correction_2026-08-04.md`.

Track and activity/spatial benchmarks must now use their real exact
full-duration authorities. Eye and tail still require selector-isolated
maintained semantic rematerialization, not metadata stamping. A later
full-duration canary preflight established a narrower upstream prerequisite:
recording-level shard finalization did not publish the canonical refined-
subject-mask coordinate ownership required to rematerialize subject-shape v4.
The subject-shape materializer correctly fails closed before eye/tail work
begins. See
`docs/diagnostics/eye_tail_query_export_source_prerequisite_2026-08-04.md`.
No benchmark may mutate or restamp the historical archive.

## Validation

Focused tests cover all four closed requests, recomputed-digest field-set
tampering, benchmark-path isolation, the request CLI, deterministic multipart
random/window/full-scan reads, exact source-metadata guard coverage, and
fail-closed missing Parquet statistics. The kinematics suite additionally
covers v1/v2 projection reconstruction, exact half-open range enforcement,
invalid range rejection, batch-boundary independence, and independent decoded
validation. The current combined focused result is 39/39. The first clean
representative-short execution and independent v1-slice/v2 equality receipt
are recorded in
`docs/diagnostics/kinematics_query_export_representative_short_2026-08-04.md`.
