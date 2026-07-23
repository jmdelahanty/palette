# Sleepyfish Columnar Zarr Sharding Benchmark

Date: 2026-07-16

## Decision

New arrays written through `fisheye.shared.zarr.columnar` use indexed outer
shards with a requested row span of 262,144 by default. The writer preserves
the existing logical chunks, rounds the requested shard span upward to the
logical row-chunk grid, and caps it at the array's useful chunk-grid extent.
Scalars, empty arrays, and arrays with only one logical row chunk remain
regular, unsharded arrays.

This is a physical-storage decision only. Field names, dtypes, shapes, logical
chunks, string encoding, and the structured-table read surface are unchanged.
The current writers are serial whole-array writers, so no workers share a
physical shard. Any future parallel writer must assign complete,
non-overlapping physical shards as required by `docs/dask_zarr_write_safety.md`.

## Sources

The source groups were opened read-only:

- `analysis/swim_bout_runs/swim_bouts_sleepyfish_core_canary_20260713_01`
- `analysis/bout_kinematics_runs/bout_kinematics_sleepyfish_core_canary_20260713_01`

Both live in the completed Sleepyfish canary recording
`sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr`.

Disposable candidates and JSON reports were written under
`/tmp/palette-columnar-sharding-benchmark`. The diagnostic is reusable:

```bash
scripts/py -m fisheye.diagnostics.benchmark_columnar_zarr_sharding \
  /path/to/completed/run/group \
  --output-root /tmp/columnar-sharding-benchmark \
  --shard-rows 65536 131072 262144 524288 \
  --scan-rows 65536 \
  --window-rows 1024
```

It clones every group and array serially, preserves logical chunks and source
attributes, checks paths/shapes/dtypes/chunks plus first/middle/last row values,
measures physical objects and read/write timings, and verifies that the source
metadata-file digest did not change. It does not perform an additional full
decoded-value digest pass.

## Results

Timings are single node-local, warm-or-mixed-cache observations. They are useful
for rejecting poor layouts, not as stable throughput guarantees.

### Swim-bout run

The source contains 133 arrays, of which 100 span multiple logical row chunks.
The run is 135.0 MiB decoded.

| Requested shard rows | Payload objects | Reduction | Write (s) | 1,024-row windows (s) | Full scan (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| disabled | 5,164 | — | 9.154 | 0.383 | 2.215 |
| 65,536 | 360 | 93.03% | 7.534 | 0.449 | 1.291 |
| 131,072 | 235 | 95.45% | 7.337 | 0.520 | 2.183 |
| **262,144** | **142** | **97.25%** | **7.682** | **0.439** | **2.079** |
| 524,288 | 121 | 97.66% | 8.519 | 0.432 | 2.111 |

At 262,144 rows, the object count is already near the floor. Doubling the shard
span removes only 21 additional payload objects while making the write slower.
The selected layout kept bounded-window latency close to the regular layout and
was faster for the full scan in this run.

### Bout-kinematics run

The source contains 115 arrays, of which 104 span multiple logical row chunks.
The run is 44.3 MiB decoded. Its large tables have 53,130 rows, so all tested
shard spans place each qualifying field in one aligned outer shard.

| Requested shard rows | Payload objects | Reduction | Write (s) | 1,024-row windows (s) | Full scan (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| disabled | 1,359 | — | 3.225 | 0.423 | 0.714 |
| 65,536 | 110 | 91.91% | 3.006 | 0.466 | 0.426 |
| 131,072 | 110 | 91.91% | 2.834 | 0.429 | 0.428 |
| **262,144** | **110** | **91.91%** | **3.011** | **0.500** | **0.477** |
| 524,288 | 110 | 91.91% | 3.194 | 0.584 | 0.495 |

All 712 representative value selections across both runs and all layouts
matched exactly. Both source metadata digests were unchanged after the
benchmark.

## Provenance Contract

Each array written by the shared helper records its logical chunk shape,
requested and effective shard rows, effective shard shape, physical layout,
alignment policy, and any reason sharding was skipped. Each columnar group also
records the policy and counts of sharded and regular fields. Callers can pass
`shard_rows=None` to explicitly retain regular chunks.
