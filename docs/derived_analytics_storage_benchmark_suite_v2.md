# Derived analytics storage benchmark suite v2

Date: 2026-08-04

Status: current executable benchmark-selection contract; no physical profile
or production selector is promoted by this document.

## Purpose

Suite v2 retains the deterministic source/candidate, publication, full-scan,
and access-class framework from suite v1 while correcting how primary reads are
selected and executed. It is implemented by
`fisheye.shared.zarr.analysis_benchmark_suite` and consumed by the typed
candidate-execution and read-matrix runners.

Historical suite-v1 documents remain structurally auditable. They are not
eligible as current timing or promotion evidence, and new execution requests
and trials fail closed when given them.

## Selection contract

Every array selection is derived from the candidate's validated storage-plan
receipt:

- the selection axis is `storage_plan.growth_axis`;
- the selection extent is the observed logical shape on that axis;
- both values and `selection_extent_source=observed_facts_growth_axis` are
  persisted in the suite; and
- source and candidate execute the same digest-bound logical selection.

Primary reads remain access-class-specific:

- `EAGER`: read the complete array;
- `WINDOWED`: read eight deterministic contiguous windows of at most 4,096
  growth-axis rows;
- `PER_ROW`: read 128 deterministic complete records on the growth axis; and
- `INDEXED`: resolve 128 deterministic table rows in one orthogonal-indexing
  operation with `execution_strategy=batched_orthogonal_index`.

The exact-tabular schemas do not yet expose a common persisted CSR pointer/length
index for every compact table. Their indexed workload therefore represents the
current batched-table consumer, not a nonexistent range-index object. A future
schema may add table-specific indexes and a newly versioned workload; that must
not be inferred by silently changing suite v2.

## Evidence boundary

The five-repetition matrix still provides fresh-process, order-rotated latency,
CPU, RSS, logical equality, metadata equivalence, object-count, and apparent /
allocated-byte evidence. OS and mounted-filesystem cache state must be declared
truthfully and is not reset by the runner.

Linux `trace_storage_io` evidence measures process-requested file bytes and
read-family syscalls for one explicitly bound matrix. It does not prove network
transfer, does not count mmap page faults, and its traced latency is invalid for
comparative performance claims. Crimson must provide separate mounted-macOS
TensorStore/file metrics for user-facing families.

Neither a benchmark suite, a passing matrix, nor a trace authorizes profile,
selector, registry, or canonical-data mutation. Promotion remains a separate,
versioned decision requiring the complete documented gates and rollback path.
