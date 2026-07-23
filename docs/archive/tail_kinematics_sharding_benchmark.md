# Tail-Kinematics Physical Sharding Benchmark

## Question

`analysis/tail_kinematics_runs` is a set of dense, row-aligned analytical
tensors, not a `compact_tabular_v2` table. Its existing 256-row logical chunks
support bounded interactive reads, but the first million-frame Sleepyfish
materialization used 16,384-row physical shards and produced 1,686 files for a
285 MB run. The benchmark asks whether larger physical shards reduce object and
transfer overhead without making representative reads unacceptably slower.

## Controlled layouts

The benchmark preserves each array's dtype, shape, logical chunk grid, codec,
fill value, and decoded bytes. Only row-aligned arrays change physical outer
shard span:

- 16,384 rows (materialized-canary baseline)
- 65,536 rows
- 131,072 rows
- 262,144 rows

Constants such as `tail_angle_sample_s` are copied without forcing them onto the
frame-row sharding scheme. Nested row lineage and revision arrays are included.

## Safety and interpretation

The authoritative source run is opened read-only and copied once to node-local
scratch. Candidate Zarrs remain disposable node-local artifacts. The driver
creates groups, arrays, attributes, and reports; process workers each own a
complete, non-overlapping outer row-shard stripe. Every candidate is decoded and
SHA-256 compared against the staged source before it is accepted.

Recorded outcomes include:

- build wall time and decoded write throughput with eight workers;
- physical file count, apparent/allocated bytes, and shard-size distribution;
- timed node-local-to-PRFS `rsync` publication followed by checksum validation;
- repeated random one-row reads;
- repeated 1,024-row contiguous windows;
- bounded full scans of representative scalar, angle, and XY arrays.

The read timings are explicitly cache-warm or cache-mixed node-local evidence.
They should not be interpreted as cold-cache shared-filesystem measurements.
The one-time shared-to-local staging time is recorded separately.

Implementation:

- `fisheye.diagnostics.benchmark_tail_kinematics_sharding`
- `scripts/submit_tail_kinematics_sharding_benchmark_bsub.sh`

The LSF wrapper refuses execution outside an allocation, pins a clean shared
Palette commit, refuses `/groups` as scratch, stages all source shards in one
`rsync`, and retains only JSON/timing/status files on shared storage. Each
publication candidate is copied into a job-specific hidden shared directory,
checked with a checksum-mode `rsync` dry run, and removed before the job exits.

## Sleepyfish result

The direct publication benchmark ran as Citrus job `153100876` against
`tail_kinematics_sleepyfish_node_local_canary_20260715_01`. All decoded-array
digests and post-transfer physical-file checksums matched.

| Outer shard rows | Total files | Publication seconds | MiB/s | Full scan seconds |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 1,686 | 13.70 | 19.9 | 3.95 |
| 65,536 | 444 | 5.01 | 54.4 | 5.98 |
| 131,072 | 237 | 3.20 | 85.0 | 6.37 |
| 262,144 | 145 | 2.41 | 113.2 | 6.26 |

Random-row and 1,024-row window timings were effectively unchanged at 262,144
rows. The selected production contract is therefore 262,144-row physical
shards with 16,384-row compute sub-blocks and the existing 256-row logical
chunks. This prioritizes publication and object-count performance while keeping
interactive bounded reads stable; consumers doing full scans retain a measured
roughly 58% local-read penalty.
