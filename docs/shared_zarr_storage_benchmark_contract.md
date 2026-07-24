# Shared Zarr Storage Benchmark Contract

Status: initial implementation contract

Related logical-schema design:
[`shared_zarr_schema_catalog_design.md`](shared_zarr_schema_catalog_design.md).

## Purpose

Turn Palette's Zarr chunk, shard, codec, and publication choices into measured,
versioned decisions. Benchmarks consume a resolved `StoragePlan`; they do not
introduce independent row-count constants that can drift from production
policy.

This contract covers three distinct costs:

1. writing an array layout;
2. validating and publishing an immutable artifact;
3. reading the published artifact through real consumer access patterns.

Passing one phase does not imply success in the others.

## Existing Palette Coverage

Palette already has valuable dataset-specific benchmark implementations:

| Surface | Existing diagnostic | Useful coverage | Gap to close |
| --- | --- | --- | --- |
| keypoints | `benchmark_keypoint_sharding.py` | exact clone validation, random rows, ranges, full scan, file inventory | fixed row candidates; no common profile identity or consolidated-open comparison |
| columnar tables | `benchmark_columnar_zarr_sharding.py` | regular/sharded variants, bounded windows, full scan, source mutation guard | preserves existing inner chunks instead of sweeping byte-based plans |
| tail and subject shape | `benchmark_tail_kinematics_sharding.py`, `benchmark_subject_shape_sharding.py` | parallel whole-shard ownership, transfer benchmark, read patterns | dataset-specific result schema and shard-row candidates |
| subject-mask probabilities | `benchmark_subject_mask_probability_sharding.py`, `benchmark_subject_mask_probability_sharding_reads.py` | native writes, storage-tier identity, cold/warm attempts, component scans | no shared plan, HTTP request instrumentation, or publication timing |
| destination reads | `benchmark_zarr_destination_reads.py` | small direct read summary | limited access patterns and environment provenance |

These tools should become adapters around a shared case/result contract. They
should not be replaced before their real-data and write-safety coverage has
been preserved.

## Inputs

Every benchmark case records:

- case ID and benchmark schema version;
- Palette commit and dirty-state digest;
- array family and logical schema ID/version;
- logical shape, dtype, axes, and access unit;
- complete resolved `StoragePlan`;
- codec profile and exact resolved codec chain;
- source-data identity and content/sample digest;
- source and destination filesystem descriptions;
- execution host, CPU, memory, worker count, and relevant library versions;
- cache state and cache-eviction method;
- trial order, seed, warmup count, and measured repeat count.

A result without these fields is exploratory evidence, not profile-promotion
evidence.

## Logical Schema And Dtype

The logical array contract, not the storage profile, owns dtype. Each canonical
array path declares:

- logical schema ID and version;
- exact dtype/representation;
- rank, axis names, and fixed trailing dimensions;
- fill and null semantics;
- required or optional status;
- coordinate system and units where applicable.

The storage planner accepts that dtype and must preserve it. A benchmark must
fail before timing if the source dtype, proposed contract dtype, or decoded
destination dtype disagree.

Legacy alternatives remain explicit, version-keyed reader adapters. They are
not candidates that Crimson probes repeatedly when opening a current archive.

The first canonical detection-storage benchmark holds continuous detection
geometry at exact `float32`. It compares chunking, sharding, codecs, metadata,
and publication behavior without mixing in representation changes. `float16`
and quantized integer geometry are deferred until the canonical storage specs
are complete. A later representation study must use a new schema version or
representation ID and add numerical-error plus downstream-behavior acceptance
tests; it is not another storage-profile candidate in the initial sweep.

The detection-specific benchmark and publication gates are tracked in
[`canonical_detection_storage_implementation_checklist.md`](canonical_detection_storage_implementation_checklist.md).
The cluster execution workflow is tracked separately in
[`canonical_detection_storage_cluster_benchmark_checklist.md`](canonical_detection_storage_cluster_benchmark_checklist.md).

## Parameter Matrix

Initial inner-chunk sweep:

```text
128 KiB, 512 KiB, 1 MiB, 2 MiB uncompressed
```

Initial outer-shard sweep where applicable:

```text
8 MiB, 32 MiB, 128 MiB, 512 MiB uncompressed
```

Each candidate is derived again from shape, dtype, access unit, write mode, and
object budget. The benchmark labels a byte target; it does not pass a raw row
count.

Codec candidates are named, versioned profiles. A codec comparison must keep
logical shape, dtype, access unit, and workload identical.

## Write Phase

Measure:

- array creation time;
- payload write time;
- logical and physical MiB/s;
- peak resident memory;
- worker count and ownership unit;
- requested and effective worker chunking;
- partial-chunk or partial-shard writes;
- final payload and metadata object counts;
- apparent and allocated bytes;
- compressed-to-logical byte ratio;
- failure, retry, and cleanup behavior.

Parallel candidates are valid only when each worker owns complete,
non-overlapping chunks and, for sharded arrays, complete non-overlapping
shards.

## Publication Phase

Measure publication separately from payload generation:

- source open and inventory time;
- rechunk/shard copy time;
- decoded equality validation time;
- logical-schema and dtype validation time;
- provenance and manifest generation time;
- metadata consolidation time;
- transfer/copy time to the destination tier;
- final atomic commit time;
- total time until selector eligibility;
- peak memory, bytes transferred, and created objects.

Publication succeeds only after decoded equality or an explicitly documented
semantic migration, schema validation, and metadata validation pass.

## Read Workloads

Run access patterns that correspond to actual consumers:

| Pattern | Representative arrays | Measurement |
| --- | --- | --- |
| metadata open | archive root and capability manifest | root requests, metadata requests, time to usable schema |
| eager whole-array | small enums, lookup tables, run metadata | cold/warm latency and bytes transferred |
| windowed rows | timelines, keypoints, lineage, offsets | requested rows, decoded bytes, transfer bytes, cache reuse |
| per-row | masks and per-frame images | random frame/component latency and decode amplification |
| indexed values | contour/RLE values via pointer/length | index reads plus value-range reads |
| sequential playback | Crimson frame traversal | sustained frames/s, p50/p95/p99 latency, cache hit rate |
| trainer sampling | images, keypoints, masks, labels, indexes | random minibatches/s, prefetch behavior, worker scaling |
| full scan | validation and analysis consumers | throughput and peak memory |

Every read report distinguishes logical bytes returned, uncompressed bytes
decoded, compressed bytes transferred, and request count. Filesystem tests may
not be able to observe all four; HTTP Range tests must.

## Metadata Cases

Every publication candidate is measured twice when supported:

1. unconsolidated open, as a compatibility baseline;
2. consolidated root open, as the intended Crimson path.

The root schema/capability manifest states the expected array paths, dtypes,
optional features, and storage-policy versions. Consolidated Zarr metadata
provides the actual array metadata in one root view. Crimson validates expected
against actual once, then dispatches typed readers without per-array dtype
probing.

Record:

- root requests;
- additional metadata requests;
- manifest parse time;
- consolidated metadata parse time;
- schema/dtype validation time;
- total time to first required array and first usable frame.

## Environments

At minimum, retain separate evidence for:

- node-local NVMe/SSD;
- Palette's shared Linux filesystem tier;
- a request-logging HTTP Range server;
- the actual Mac/VPN path used by Crimson.

Results from one tier must not silently promote defaults for another. Cache
state and cache-eviction support must be explicit.

## Result Schema

The common envelope is `palette.storage_benchmark` with `schema_version: 1`:

```json
{
  "schema_id": "palette.storage_benchmark",
  "case_id": "keypoints_img__published_http_v1__chunk_1m__shard_32m",
  "phase": "read",
  "logical_schema": {"id": "...", "version": 1},
  "storage_plan": {"policy_version": "palette.storage_planner.v1"},
  "source_identity": {},
  "environment": {},
  "workload": {},
  "trials": [],
  "summary": {},
  "validation": {}
}
```

Dataset-specific diagnostics may add namespaced details, but promotion tooling
reads the common envelope.

## Profile Promotion Gate

A byte/codec profile becomes a production default only when:

- dtype and decoded-value validation pass;
- object count remains within budget;
- write and publication memory remain bounded;
- required access patterns meet agreed latency/throughput thresholds;
- no measured tier regresses beyond its accepted tolerance;
- Crimson decoder compatibility is proven;
- results identify the exact profile, schema, data, environment, and commit;
- the profile receives a new immutable version ID.

There is no universal winner across all array families. The shared planner
keeps the parameter vocabulary common while access class and lifecycle select
the appropriate result.

## Implementation Checklist

### Foundation

- [x] Add a pure byte-based `StoragePlan`.
- [x] Include logical shape, dtype, access class, and write mode in the resolved
      plan contract.
- [x] Add a read-only actual-versus-proposed layout comparison.
- [ ] Add logical schema IDs and canonical dtype descriptors per array path.
- [x] Add the common benchmark result envelope and JSON validation.

### Adapt Existing Benchmarks

- [ ] Make keypoint candidates consume `StoragePlan`.
- [ ] Make columnar candidates re-plan inner chunks as well as outer shards.
- [ ] Adapt tail and subject-shape transfer benchmarks to the common envelope.
- [ ] Adapt probability write/read benchmarks to the common envelope.
- [ ] Preserve all existing source-read-only and write-ownership guards.

### Add Missing Measurements

- [ ] Add publication and consolidation phase timing.
- [ ] Add consolidated-versus-unconsolidated metadata-open cases.
- [ ] Add a request-counting HTTP Range store/server.
- [ ] Add Crimson playback, scrub, and initialization traces.
- [ ] Add keypoint and mask training minibatch workloads.
- [ ] Report peak memory and compressed bytes consistently.

### Execute

- [ ] Establish current-layout baselines before migrating writers.
- [ ] Run the byte/shard sweep on representative small fixtures.
- [ ] Run bounded real-data canaries on local and shared storage.
- [ ] Run HTTP and Mac/VPN cases.
- [ ] Promote only evidence-backed profile versions.
