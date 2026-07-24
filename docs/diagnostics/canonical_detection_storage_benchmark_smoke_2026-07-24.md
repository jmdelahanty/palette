# Canonical Detection Storage Benchmark Smoke

Status: exploratory Phase 4 smoke; no production profile selected

Date: 2026-07-24

## Result

The shared planner and array factory can materialize the complete nine-array
canonical detection schema as regular Zarr v3 chunks or indexed Zarr v3 shards.
Both 200,000-frame candidates decoded to the exact canonical dtypes and values.

At this short-video scale, indexed sharding combined the multi-chunk arrays
into one payload object each. It reduced the actual payload-file count from
`17` to `8` and total file count from `28` to `19`; the all-zero `class_ids`
payload was fill-elided in both layouts. Apparent storage was effectively
unchanged.

This single local, uncontrolled-cache A/B does **not** select a storage profile.
It establishes that the byte-derived design remains structurally defensible for
approximately 200,000 frames and identifies the tradeoff to test with repeated
remote reads.

## Safe Fixture

The source was copied from the historical noncanonical detection run:

```text
/groups/johnson/johnsonlab/jeremy/recordings/
sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/
sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr/
detect_runs/detect_2026-05-14_15-39-11
```

The benchmark source and outputs live only below:

```text
/tmp/palette-zarr-benchmarks/
sleepyfish_cam2010095_detection_20260724/
```

The copied tree was verified against the source with:

- `5,809` files;
- `8,317,265` apparent bytes;
- deterministic relative-path/content tree SHA-256
  `de59cb09cbe53866d2587719e5b8b06771536dceaa1fe1ec3cabfb7a7faf3bd3`;
- source-group `zarr.json` SHA-256
  `65e6f8a5d691aad0148e1f08d3c7c0e052fdf789167b49e47f2c1168cf3b7c3b`.

The fixture README and manifest mark it noncanonical, unregistered,
selector-ineligible, read-only at the source, and disposable at the outputs.

## Canonical Input

Both candidates used the same validated input:

- frames: `200,000`;
- observed detection rows: `199,734`;
- source geometry: `4512 x 4512`;
- canonical logical bytes: `15,181,920`;
- normalized boxes: source `float64` converted once to canonical `float32`;
- image boxes and centers: derived from the canonical `float32` boxes;
- instance keys: minted from the canonical `float32` representation;
- frame-row offsets: exact `int64` cumulative counts;
- all nine destination arrays: exact SHA-256 match to the validated canonical
  input.

## Candidates

Both candidates used approximately `1 MiB` uncompressed inner chunks and the
exact `zstd_fast_v1` data codec contract:

```text
Zarr format 3
bytes(endian=little)
zstd(level=0, checksum=false)
```

The sharded candidate additionally used indexed sharding with `bytes + crc32c`
index codecs and `index_location=end`. Its `32 MiB` target is an upper budget,
not padding: the concrete outer shards were approximately `2 MiB` or `4 MiB`
for arrays holding two or four inner chunks.

| Measure | Indexed sharding | Regular chunks | Sharded change |
| --- | ---: | ---: | ---: |
| planned inner chunks | 18 | 18 | none |
| planned payload objects | 9 | 18 | -50.0% |
| planned stage objects | 20 | 29 | -31.0% |
| actual payload files | 8 | 17 | -52.9% |
| actual total files | 19 | 28 | -32.1% |
| apparent bytes | 5,613,036 | 5,606,607 | +0.11% |
| summed array-write time | 0.163 s | 0.233 s | -29.8% |
| total candidate time | 0.461 s | 0.514 s | -10.4% |
| summed 1,024-row window reads | 0.039 s | 0.030 s | +29.3% |
| summed full-array reads | 0.042 s | 0.038 s | +11.4% |

Times are one observation each, came from separate processes, and used an
uncontrolled local filesystem cache. They are smoke diagnostics, not effect
estimates. Peak process RSS was approximately `296 MB`, including the in-memory
canonical input; it is not yet an isolated writer-memory measurement.

## Metadata Finding

Both direct and consolidated opens succeeded. Consolidation took about `15 ms`;
direct and consolidated group opens each took about `2-4 ms` for this tiny
standalone stage. That does not model Crimson opening a complete archive with
hundreds of groups and arrays.

Zarr Python `3.1.3` emitted an important compatibility warning: consolidated
metadata is not formally part of the Zarr v3 specification and may differ
across implementations. Palette can still make consolidation part of its
published-profile contract, but Crimson compatibility must be proven against
the exact consumer driver rather than inferred from Python behavior.

## Remaining Evidence Gate

- repeat and randomize regular/sharded trials;
- sweep `128 KiB`, `512 KiB`, `1 MiB`, and `2 MiB` chunks and relevant shard
  budgets;
- separate cold and warm reads;
- add individual-frame and observation-row lookup workloads;
- record request count, range count, decoded bytes, and transferred bytes;
- benchmark publication/copy separately from first materialization;
- run the winning candidates through the actual Crimson Mac/VPN path;
- only then promote a production storage profile or decide that two-to-four
  chunk arrays should remain regular at this scale.
