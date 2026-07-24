# Canonical Detection Storage: Four 200k Repetitions

Date: 2026-07-24

Status: complete benchmark evidence; no storage profile promoted

## Execution identity

- Palette commit:
  `5ff44a2cadd702f3156347b3a38118c1aad32730`
- Locked cluster checkout:
  `/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/shared-zarr-storage-policy-20260723-5ff44a2c`
- Workflow:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/workflows/sleepyfish_det_storage_200k_repetitions_20260724_01`
- Matrix fingerprint:
  `012346b8940425021c87ed3e84ac7b21cbd634112cb56f529e7df161e54e6688`
- LSF block array: `153169683`
- LSF finalizer: `153169684`, gated by successful completion of the block
  array
- Execution host: `h07u31`
- Scale: 200,000 frames and 199,734 instances
- Repetition indices: 1, 2, 3, and 4

The matrix retained eight unique physical candidates from twenty requested
labels. Larger shard targets collapsed to the same physical plans at this
scale, so every retained sharded candidate used an 8 MiB target shard.

## Workloads

Each candidate used separate subprocesses for writing, local-scratch reading,
and published PRFS reading. Each reader ran the following ordered suite:

1. direct and consolidated metadata-open trials;
2. complete eager `frame_row_offsets` reads;
3. 128 deterministic random frame-to-instance-slice reads;
4. 64 deterministic 32-row observation-range reads;
5. a complete sequential traversal in 700-frame windows;
6. per-array window and full-scan reads.

Every payload workload ran once under the label
`process_first_pass_os_cache_uncontrolled` and once in the same process under
the label `same_process_warm_pass_1`. No result is labelled as a controlled
cold-cache measurement.

## Correctness and isolation

- All four block reports completed successfully.
- All 32 candidate records passed exact array and consumer-workload checks.
- Both direct and consolidated metadata opens succeeded.
- Every block reported the frozen fixture unchanged.
- The final node-local scratch path was confirmed absent after each block.
- All five runtime statuses, including the finalizer, succeeded with return
  code zero.
- No LSF stderr file contained data.
- The aggregate records zero registry updates, zero selector updates, zero
  training artifacts, and no profile promotion.

Measured block totals were 339.1, 358.5, 314.8, and 315.5 seconds. The
approximately 14% maximum-to-minimum spread is enough to make a single fastest
observation an unreliable decision rule.

## Median results across four repetitions

`PRFS suite` is the fresh reader subprocess wall time. `Frame p95` is the
median, across repetitions, of each first-pass random-frame workload's p95
frame latency. `Rows p95` is defined analogously for random observation
ranges. The local pipeline includes materialization, exact validation,
consolidation, and the original per-array smoke reads; it is not a pure payload
write timer.

| Layout | Inner chunk | Payload objects | Local pipeline (s) | Publish (s) | PRFS suite (s) | Frame p95 (ms) | Rows p95 (ms) | Sequential FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regular | 128 KiB | 104 | 0.605 | 1.290 | 11.328 | 18.73 | 8.53 | 73,601 |
| regular | 512 KiB | 31 | 0.506 | 0.572 | 14.976 | 18.12 | 12.81 | 52,431 |
| regular | 1 MiB | 17 | 0.467 | 0.428 | 19.594 | 22.10 | 17.31 | 38,450 |
| regular | 2 MiB | 10 | 0.467 | 0.352 | 23.101 | 25.70 | 21.74 | 31,685 |
| sharded, 8 MiB target | 128 KiB | 8 | 0.564 | 0.318 | 16.990 | 19.19 | 14.44 | 44,763 |
| sharded, 8 MiB target | 512 KiB | 8 | 0.507 | 0.327 | 21.455 | 23.50 | 19.17 | 34,322 |
| sharded, 8 MiB target | 1 MiB | 8 | 0.477 | 0.318 | 23.139 | 25.77 | 22.01 | 31,977 |
| sharded, 8 MiB target | 2 MiB | 8 | 0.560 | 0.330 | 31.568 | 36.96 | 32.13 | 23,459 |

Median peak RSS was 200,388,608 bytes for every candidate. The benchmark input
is loaded before candidate materialization, so peak RSS did not discriminate
among these layouts.

## Findings

1. Inner chunk size matters independently of sharding. For these narrow
   detection arrays and small indexed reads, larger inner chunks consistently
   increased total PRFS suite time, random-range latency, and read
   amplification.
2. The 128 KiB regular layout was the fastest overall reader, but its 104
   payload objects and 1.29-second median publication time make it a poor
   object-count default.
3. The 128 KiB-inner, 8 MiB-shard layout is the provisional Pareto candidate.
   Relative to the declared 1 MiB regular control, it used 8 rather than 17
   payload objects, reduced median PRFS suite time by 13%, reduced median
   random-frame p95 by 13%, increased sequential throughput by 16%, and reduced
   publication time by 26%. Its median local pipeline was 21% slower, still
   within the predeclared 1.25 ratio.
4. The 2 MiB-inner sharded layout is dominated at this scale: it has the same
   eight-object count as the other sharded plans, while writing and reading
   more slowly than the 1 MiB-inner sharded plan.
5. All candidates exceeded the 700-FPS sequential target by a large margin.
   Random frame/range latency and total suite time are more discriminating for
   this schema than the 700-FPS threshold.
6. Complete eager offset reads took only about 5--9 ms across candidates.
   Chunk-size selection for offsets is therefore not a bottleneck in this
   200,000-frame PRFS workload.
7. Consolidated opens were approximately 2--3 ms for these nine-array stores.
   This does not measure Crimson's large-archive initialization problem: the
   benchmark is PRFS rather than HTTP and contains far fewer arrays/groups than
   a complete analysis archive.

## Decision boundary

The aggregate correctly refused profile selection because the new
consumer-workload schema has only four balanced repetitions and the declared
minimum is five. The earlier repetition-0 smoke can support write,
publication, object-count, and correctness observations, but it predates the
consumer workload schema and must not be treated as a fifth read repetition.

Before promotion:

- run at least one more balanced repetition of this workload version;
- apply the declared per-metric gates, including peak resource checks;
- carry the provisional frontier to full-duration data;
- benchmark request count, transferred bytes, and read amplification through
  HTTP Range requests; and
- validate the finalists in Crimson on the actual Mac/VPN path.

Evidence is stored in `matrix.json`, the four reports under
`reports/blocks/`, per-candidate local/write/publication/PRFS reports beside
the published candidates, and `aggregate.json` in the workflow root above.
