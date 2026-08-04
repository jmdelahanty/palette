# Swim-bout full-duration storage suite-v2 checkpoint

Date: 2026-08-04

Status: full-duration Palette benchmark passed; selector-ineligible and
nonpromoting. Short-scale and real Crimson consumer evidence remain open.

## Immutable inputs

- Palette revision: `22779af592c03be4f44a1544b1214a94c69c3d1a`
  from a clean detached worktree.
- Archive:
  `/tmp/.palette_benchmarks/derived_analytics_storage/swim_bouts_sleepyfish_full_20260804_e8f3e020/archive.zarr`
- Source run:
  `analysis/swim_bout_runs/swim_bouts_sleepyfish_exact_v8_eligible_source_20260804_e8f3e020`
- Candidate run:
  `analysis/swim_bout_runs/swim_bouts_sleepyfish_published_http_v1_20260804_e8f3e020`
- Cache declaration: `fresh_process_os_cache_uncontrolled`.
- Seed: 17; repetitions: five, with rotated fresh-process ordering.

The source and candidate are the same immutable pair used by the historical
suite-v1 checkpoint. Neither run was republished or modified.

## Corrected balanced matrix

Evidence directory:
`/tmp/.palette_benchmarks/derived_analytics_storage/swim_bouts_sleepyfish_full_20260804_e8f3e020/read_matrix_untraced_suite_v2_22779af5`

- Matrix payload digest:
  `61c51e64473fbd07b50f6173ef1b0efece4047f0ea844c4980a0594bc4756b19`.
- Suite-v2 payload digest:
  `9805baafdd269f03b39325f9fcd3a406dfdf30538fd13aaf89361a7f08d667cf`.
- Exact decoded equality: passed.
- Direct/consolidated metadata equivalence: passed.
- Manifest and storage-receipt validation: passed.
- Archive metadata read-only guard: unchanged.

| Median / inventory | Source | Candidate | Candidate change |
|---|---:|---:|---:|
| Primary access | 1.723 s | 0.577 s | -66.5% |
| Full scan | 1.455 s | 0.635 s | -56.4% |
| Peak RSS | 514.9 MB | 501.5 MB | -2.6% |
| Payload objects | 137 | 113 | -17.5% |
| Apparent bytes | 40,173,037 | 34,407,207 | -14.4% |
| Allocated bytes | 40,894,464 | 35,053,568 | -14.3% |

The median primary-access components all favored the candidate:

| Access class | Source | Candidate |
|---|---:|---:|
| Eager | 0.425 s | 0.165 s |
| Indexed batched rows | 1.224 s | 0.372 s |
| Windowed growth-axis ranges | 0.089 s | 0.034 s |

This resolves the suite-v1 30.164 s apparent primary-read regression. That
result came from 128 independent scalar reads for each indexed column rather
than the maintained batched consumer pattern.

## Process-tree physical-I/O trace

Evidence directory:
`/tmp/.palette_benchmarks/derived_analytics_storage/swim_bouts_sleepyfish_full_20260804_e8f3e020/read_trace_suite_v2_22779af5`

- Trace payload digest:
  `3252e8799fe400059c21760a06cb441ab806702b94fd70310ac34ae7838c4676`.
- Bound matrix payload digest: `61c51e64473fbd07b50f6173ef1b0efece4047f0ea844c4980a0594bc4756b19`.
- Process-requested file bytes: 244,556,761.
- Attributed read operations: 17,896.
- Unique attributed objects: 541.
- Metadata bytes / operations: 71,325,786 / 12,776.
- Payload bytes / operations: 173,230,975 / 5,120.
- Peak RSS: 517,828,608 bytes.

Against the historical scalar suite-v1 trace, process-requested bytes fell
89.97% (2,437,806,661 to 244,556,761) and attributed read operations fell
79.49% (87,267 to 17,896). Unique objects remain high because the traced command
also performs complete metadata validation and full scans of both layouts.

These are Linux process-tree file-syscall measurements, not proven filesystem,
SMB, HTTP, or network transfer bytes. Mmap page faults are not counted and
`strace` overhead invalidates the traced command's latency.

## Verdict and remaining gates

The existing `published_http_v1` candidate passes the corrected full-duration
Palette read gate and its physical-I/O sensitivity trace. No smaller-chunk
profile is justified by the obsolete scalar result, and no profile is promoted
here.

Still required before promotion:

- one representative short real recording / event cardinality;
- mounted Crimson consumer correctness and physical-I/O evidence if swim-bout
  data becomes user-facing there;
- complete writer/publication phase evidence under current hierarchical
  telemetry; and
- an explicit versioned promotion and rollback decision.

A metadata-only recording census found 120,221-143,305-frame GoodCopBadCop
swim-bout sources, but their track-kinematics authorities predate the required
digest-bound `track_motion_publication_manifest`. Current sealed track
authorities were found only for the full-duration Sleepyfish camera archives.
The short gate therefore remains missing; those legacy runs must not be stamped
or treated as current merely to manufacture benchmark evidence.
