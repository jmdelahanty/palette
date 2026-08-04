# Derived analytics storage benchmark suite v1

Date: 2026-08-03

Status: benchmark planning contract implemented; no profile promoted.

`fisheye.shared.zarr.analysis_benchmark_suite` turns one exact
`AnalysisStoragePlanReceipt` into a deterministic writer/read/publish workload
suite. It does not create an archive or authorize production state changes.

Each suite is bound to:

- one maintained family ID;
- one exact logical dimension set;
- the complete digest-bound storage-plan receipt;
- a deterministic seed and repetition count;
- three per-array cases: complete materialization, the declared primary access
  pattern, and a full scan; and
- one whole-run validation/consolidation/copy/atomic-publication case.

Primary reads are derived from access class. Eager arrays are loaded whole,
windowed arrays use eight deterministic ranges of at most 4,096 rows, per-row
arrays use 128 deterministic complete records, and indexed arrays resolve 128
deterministic index rows into persisted value ranges at execution time. These
selections depend on logical shape, path, and seed—not chunk or shard shape—so
candidate layouts receive identical work.

The publication case is deliberately run-level. It measures manifest and
logical validation, direct/consolidated metadata comparison, copy, validation,
atomic publication, object/byte inventory, and peak RSS separately from array
materialization. A suite requires node-local compute, benchmark-only immutable
publication, rotated candidate order, declared cache state, decoded equality,
and selector/registry ineligibility.

The first required scales for frame-like families remain approximately 200,000
and 1,000,000 rows. Event/table families should supply dimensions representative
of both a small real recording and a full-duration Sleepyfish-style recording;
they must not fake one row per frame when their actual cardinality differs.

## Physical-I/O tracing

`fisheye.diagnostics.trace_storage_io` now provides the Linux process-tree
physical-read evidence boundary. It runs one existing benchmark command under
`strace -ff -yy` and GNU `time -v`, attributes successful `read`, `pread64`,
`readv`, `preadv`, and `preadv2` calls only to explicit target Zarr roots, and
retains digest-bound raw traces. When given `--stage-id` and
`--matrix-result`, the receipt deeply validates the family matrix through the
catalog and binds its normalized source/candidate identity and matrix digest.

The measurement scope is deliberately named `process_tree_file_syscalls`:

- measured bytes are the compressed file bytes requested by the traced
  process tree, not proven filesystem, SMB, HTTP, or network transfer bytes;
- memory-mapped page-fault I/O is not counted;
- strace overhead invalidates the traced command's latency as comparative
  performance evidence; use the normal balanced fresh-process matrix for
  latency and the traced execution for read counts/bytes/object attribution;
- GNU time supplies CPU and peak-RSS observations for the wrapped benchmark;
  and
- the trace receipt always remains nonpromoting.

Crimson/macOS still requires its own TensorStore/file metrics and mounted-path
consumer evidence. The Linux trace cannot substitute for that gate.

## Palette and Crimson consumer evidence

`fisheye.analysis_workflows.storage_consumer_evidence` defines the shared v1
receipt that a real Palette or Crimson consumer must emit. A receipt covers
one catalog stage, one representative scale, one exact validated family
matrix, and one consumer implementation revision. It translates the
matrix-owned relative source/candidate paths beneath the consumer's mounted
archive path rather than rewriting scientific run identity.

The required execution is balanced and process-first. Every repetition has
one source and one candidate trial, source/candidate order alternates, and
every trial has a distinct process identity. The receipt records exact-schema
opens, direct/consolidated metadata equivalence, explicit run selection, dtype
probe count, stale-publication count, production mutations, decoded and
workload result digests, and a fixed performance measurement surface:

- readiness;
- primary-read p95;
- full-scan duration and row throughput;
- peak RSS; and
- physical read operations/bytes when the consumer exposes them.

The validator derives the verdict. It requires equal decoded and workload
digests across both layouts, exact typed opens, metadata equivalence, explicit
selection, zero dtype probes, zero stale publications, successful processes,
and no production mutation. A dirty producer revision may record a compatible
debug result but is never evidence-eligible. The receipt always keeps
`promotion_authorized=false`; a separate versioned promotion gate remains
mandatory.

The catalog owns stage-specific validation through
`DerivedAnalysisStorageBenchmark.validate_consumer_evidence(...)`, preventing
a valid receipt for one family from being attached to another.

Next steps per family:

- [ ] Connect the scientific writer to the exact plan receipt behind an opt-in
      candidate profile.
- [ ] Implement the suite's node-local writer and read adapter.
- [ ] Copy through the existing exclusive benchmark publisher.
- [ ] Run balanced repetitions on node-local and shared storage.
- [ ] Add the real Palette reader workload.
- [ ] Add the Crimson consumer workload when the family is user-facing there.
- [x] Define one immutable, cross-language Palette/Crimson consumer evidence
      contract with exact matrix, revision, workload, path, trial, metric, and
      derived-verdict bindings.
- [ ] Compare exact decoded values and direct/consolidated metadata.
- [x] Implement process-tree file-read, CPU, RSS, raw-trace, and exact matrix
      binding for Linux benchmark commands.
- [ ] Run that tracer on representative candidate matrices and record object
      counts, apparent/allocated bytes, phase timing, and RSS.
- [ ] Make a separate versioned promotion decision; a passing benchmark never
      mutates a selector by itself.
