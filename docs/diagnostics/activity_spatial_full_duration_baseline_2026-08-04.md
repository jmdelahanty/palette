# Activity/Spatial Full-Duration Baseline — 2026-08-04

Status: clean-revision correctness and read matrix pass; publisher access
pattern rejected as the maintained performance default. No profile, selector,
registry authority, workflow default, or canonical archive was changed.

## Immutable Inputs

The benchmark-only source fixture is:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
analytics_query_exports/fixtures/
sleepyfish_activity_spatial_full_20260804_81a86dd4
```

It is bound to clean Palette commit
`81a86dd464985853cf1c46f2a2e43a6f30021114`. Its manifest digest is
`3979e5410b1cd7b53613821ce7b591e6dbd5e65eb8b1fc2054a66c09f4405876`.
The fixture contains 7,066 files and 352,487,479 apparent bytes. All 132
swim-bout arrays passed the closed lossless projection: 129 exact and three
fixed-UTF8 source-prefix/zero-padding widenings. Direct and consolidated
metadata were equivalent and `evidence_eligible=true`.

## Execution Boundary

The first v4 controller attempt failed before publication because the request
file was placed inside the matrix-owned evidence directory. That immutable
namespace remains failure evidence. The successful v5 request separates
`requests/` from `evidence/`:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
analytics_query_exports/
20260804_full_duration_v5_activity_clean_81a86dd4
```

Request digest:
`a00b85538275ed4f874d86f90bdeca3c007859cd88534cb90d2b6a5eabe07158`.
Matrix digest:
`c106d12ced066a74bbb17a9c3d2ec02cdf90628d4b20cdb2d75348d169ed9d69`.
The source guard hashed 268 direct metadata files before and after and found no
change.

## Baseline Result

Correctness passed for 7,920 globally aligned 5-second bins. The immutable
query product is one Parquet object plus one manifest object:

- 715,648 apparent bytes and 716,288 allocated bytes;
- median full scan: 44.0 ms;
- median random-frame p95: 6.10 ms;
- median window p95: 6.15 ms;
- median complete validation: 276 ms; and
- one identical full-scan logical digest across five fresh processes.

Publication was unacceptable:

- wall time: 2,445.6 seconds (40 minutes 45.6 seconds);
- node-local scratch Parquet phase: 2,437.4 seconds;
- source binding before/after: 3.73 and 3.85 seconds;
- process CPU: 835.5 seconds;
- average effective CPU: 0.341 cores; and
- peak process-tree RSS: 968,192,000 bytes.

The 30 FPS source and 5-second bins produce 150-frame bins. The baseline
implementation issued one source read window for each of approximately 7,920
bins, repeatedly reopening and decoding much larger physical source chunks.
During the live run `/proc/<pid>/io` reached at least 82,171,228,447 requested
read characters and 681,663 read syscalls before completion. Those two values
are diagnostic lower bounds, not immutable matrix fields: resource-telemetry
v1 did not persist process I/O counters.

## Correction

The exporter now binds a versioned extraction policy and reads consecutive
whole bins in a bounded source window. With the 131,072-row default and
150-frame bins, one read covers 873 bins/130,950 frames, reducing the expected
full-track read-window count from about 7,920 to nine without changing row
semantics. Unit evidence proves one-bin and multi-bin windows have identical
decoded payloads and columns, including IEEE NaN semantics.

Resource-telemetry v2 now persists process-tree requested characters,
read/write syscall counts, and OS-reported storage bytes. It explicitly marks
these counters as distinct from compressed network transfer.

The next admissible gate is a clean-revision v6 matrix using the same immutable
fixture, source runs, binning, row-group policy, workload, and five-process
reader matrix. It must match the baseline full-scan logical digest exactly and
materially reduce publication time and requested I/O. This document does not
promote either implementation.
