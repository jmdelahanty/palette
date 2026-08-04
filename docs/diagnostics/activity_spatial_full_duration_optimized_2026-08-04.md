# Activity/Spatial Full-Duration Optimized Gate — 2026-08-04

Status: pass for the bounded multi-bin extraction implementation. This is an
execution-policy result, not a Zarr/Parquet physical-profile promotion, and it
changes no selector, registry authority, workflow activation, or canonical
archive.

## Paired Evidence

Both runs use the same clean immutable source fixture, 30 FPS source,
5-second/150-frame global bins, exact source-run map, 65,536-row Parquet group,
deterministic workloads, and five fresh reader processes.

Baseline matrix:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
analytics_query_exports/
20260804_full_duration_v5_activity_clean_81a86dd4/evidence/activity_spatial
```

Baseline matrix digest:
`c106d12ced066a74bbb17a9c3d2ec02cdf90628d4b20cdb2d75348d169ed9d69`.

Optimized matrix:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
analytics_query_exports/
20260804_full_duration_v6_activity_batched_ebd40847/evidence/activity_spatial
```

Optimized request digest:
`37982ac6d71eb279dac136cd6d3d9d7a48a550d908f33b0561bef94e62113213`.
Optimized matrix digest:
`f00f19823d3e8573346e3e1894d254c67000b879ca24c786552284f5aaaf3486`.
The optimized publication is bound to clean Palette commit
`ebd408476eb5e62d2e98850d940cb73b7cd7f030`.

## Correctness

Both matrices passed and contain exactly 7,920 rows. Their maintained
fresh-process full scans produced the same logical stream digest:

```text
8b08cc08eb36f6cdcf96a12d228c2bb3008c0493d0f3e44b2d4ef03bada1f0a1
```

The optimized manifest is export-envelope v3 and binds:

- `requested_source_window_rows=131072`;
- `bin_size_frames=150`;
- `effective_bins_per_source_window=873`;
- `effective_source_frame_span=130950`; and
- `consecutive_global_bins_bounded_source_window_v1`.

All five optimized readers agreed on the manifest and decoded stream. The
268-file source metadata guard was unchanged. `promotion_authorized=false`.

## Publication Result

| Measurement | One-bin baseline | Bounded multi-bin | Change |
|---|---:|---:|---:|
| Publisher wall time | 2,445.60 s | 41.23 s | 59.3x faster |
| Scratch Parquet phase | 2,437.40 s | 32.23 s | 75.6x faster |
| Process CPU | 835.54 s | 38.29 s | 95.4% lower |
| Peak process-tree RSS | 968,192,000 B | 862,969,856 B | 10.9% lower |
| Output objects | 2 | 2 | unchanged |
| Apparent output bytes | 715,648 B | 716,235 B | +587 B |

Resource-telemetry v2 measured the optimized publisher directly:

- 453,838,303 requested read characters;
- 15,598 read-like syscalls;
- zero OS-reported storage-read bytes, consistent with cache/mounted-filesystem
  reporting and not evidence of zero network transfer; and
- 1,929,216 OS-reported storage-write bytes.

The baseline runner used telemetry v1. Live `/proc` diagnostics before it
finished had already reached at least 82,171,228,447 requested characters and
681,663 read syscalls. Therefore the optimized implementation reduced these
diagnostic lower bounds by at least 181x and 43.7x respectively. Those are not
network-transfer ratios.

## Reader Result

The small immutable result remains efficient and did not materially regress:

| Measurement | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Random-frame p95 | 6.10 ms | 5.89 ms | 3.5% faster |
| Window p95 | 6.15 ms | 5.79 ms | 5.9% faster |
| Full scan | 44.02 ms | 45.28 ms | 2.9% slower |
| Complete validation | 276.31 ms | 277.11 ms | 0.3% slower |

The small scan/validation differences are immaterial relative to the 59x
publication improvement and identical decoded stream.

## Verdict

The bounded multi-bin reader is accepted as the maintained activity/spatial
export implementation. The one-bin implementation remains immutable benchmark
evidence, not a compatibility mode or rollback surface: both produce the same
portable query contract, and only internal source-read scheduling changed.

Remaining gates are the representative-short scale and the independent
full-duration eye/tail exporter fixtures. This result does not authorize any
storage-profile or production-selection change.
