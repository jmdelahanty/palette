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

- 16,384 rows (current baseline)
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
`rsync`, and retains only JSON/timing/status files on shared storage.
