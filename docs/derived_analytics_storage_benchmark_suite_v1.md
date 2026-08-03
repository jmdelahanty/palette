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

Next steps per family:

- [ ] Connect the scientific writer to the exact plan receipt behind an opt-in
      candidate profile.
- [ ] Implement the suite's node-local writer and read adapter.
- [ ] Copy through the existing exclusive benchmark publisher.
- [ ] Run balanced repetitions on node-local and shared storage.
- [ ] Add the real Palette reader workload.
- [ ] Add the Crimson consumer workload when the family is user-facing there.
- [ ] Compare exact decoded values and direct/consolidated metadata.
- [ ] Record object counts, apparent/allocated bytes, phase timing, and RSS.
- [ ] Make a separate versioned promotion decision; a passing benchmark never
      mutates a selector by itself.
