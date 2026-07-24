# Canonical Detection Storage: Five-Repetition Reduction

Date: 2026-07-24

Status: next-stage candidate selected; no production profile promoted

Follow-up: full-duration evidence is documented in
[`canonical_detection_storage_full_duration_reduction_2026-07-24.md`](canonical_detection_storage_full_duration_reduction_2026-07-24.md).

## Evidence Identity

- Benchmark Palette commit:
  `5ff44a2cadd702f3156347b3a38118c1aad32730`
- Reducer Palette commit:
  `987d97ae3bfa70f663ede1bafa2b13f744118a19`
- Frozen fixture:
  `sleepyfish_cam2010095_detect_20260724_v1`
- Fixture-manifest SHA-256:
  `ae1b65b1e5255168bed320cf0d099b16ef9966255c6aed098182e33bf653062a`
- Scale: 200,000 frames, 199,734 instances, 4512x4512 source geometry
- Repetition indices: 1 through 5
- Repetition-5 block/finalizer jobs: `153170737` / `153170738`
- Repetition-5 hosts: `h07u19` / `h07u11`

Repetition 5 reused the same locked checkout, fixture, seed, workload schema,
correctness gates, performance tolerances, requests, and eight physical
candidate fingerprints as repetitions 1--4. Its reviewed plan had no
destination collision. The block and success-dependent finalizer both
completed with return code zero; stderr was empty, the frozen fixture remained
unchanged, all exact array and consumer reads passed, and the exact node-local
scratch root was absent afterward.

The combined aggregate is:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
canonical_detection_storage/reductions/
sleepyfish_det_storage_200k_v2_repetitions_1_5_20260724_01/aggregate.json
```

It records five blocks, eight candidates, zero registry updates, zero selector
updates, zero training artifacts, and `profile_promoted = false`.

## Five-Repetition Results

All values below are medians except `PRFS p95`, which is the p95 across the
five fresh-reader subprocess times. `Frame p95` and `rows p95` are medians of
the per-repetition first-pass latency p95 values.

| Layout | Inner chunk | Pass | Objects | Local (s) | Publish (s) | PRFS median / p95 (s) | Frame p95 (ms) | Rows p95 (ms) | Sequential FPS |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regular | 128 KiB | no | 104 | 0.564 | 1.265 | 11.216 / 11.807 | 18.43 | 8.47 | 73,992 |
| regular | 512 KiB | no | 31 | 0.505 | 0.559 | 14.960 / 15.964 | 17.69 | 12.53 | 52,622 |
| regular control | 1 MiB | yes | 17 | 0.459 | 0.427 | 19.307 / 21.901 | 20.96 | 16.99 | 38,836 |
| regular | 2 MiB | no | 10 | 0.397 | 0.349 | 22.961 / 26.420 | 25.65 | 21.41 | 31,863 |
| sharded, 8 MiB target | 128 KiB | **yes** | **8** | 0.510 | 0.316 | 16.556 / 19.235 | 19.11 | 13.84 | 46,393 |
| sharded, 8 MiB target | 512 KiB | no | 8 | 0.492 | 0.326 | 21.422 / 22.752 | 22.65 | 18.93 | 34,359 |
| sharded, 8 MiB target | 1 MiB | no | 8 | 0.416 | 0.313 | 21.907 / 27.178 | 24.51 | 20.50 | 33,786 |
| sharded, 8 MiB target | 2 MiB | no | 8 | 0.543 | 0.329 | 30.678 / 33.474 | 33.95 | 32.13 | 24,072 |

Median write-phase peak RSS was 200,470,528 bytes for every candidate and was
therefore not discriminating.

## Gate Results

Only two candidates passed every predeclared gate:

1. the regular 1 MiB control; and
2. the 128 KiB-inner, 8 MiB-target-shard candidate.

The sharded 128 KiB candidate used 8 objects instead of the control's 17. Its
median local pipeline ratio was 1.110, publication ratio was 0.740, and peak
RSS ratio was 1.000. Every required median PRFS latency ratio was below 1.0,
and its highest cross-repetition p95 read ratio was 1.036, well below the 1.20
limit. It therefore won the declared lowest-object-count objective.

Rejections were explicit:

- regular 128 KiB failed only median publication time (`2.966x` control);
- regular 512 KiB failed only median publication time (`1.309x`);
- regular 2 MiB failed multiple median and p95 PRFS latency gates;
- sharded 512 KiB narrowly failed median total PRFS, indexed-range, and
  sequential-window gates (`1.110x` to `1.130x` where the limit is `1.10x`);
- sharded 1 MiB failed multiple median and p95 PRFS latency gates; and
- sharded 2 MiB failed broadly and was the slowest sharded reader.

The aggregate contains every per-metric ratio and rejection reason.

## Decision

Carry the 128 KiB-inner, 8 MiB-target-shard candidate and the regular 1 MiB
control to full-duration validation. Do not change the production profile yet.
After full-duration evidence, test request count, transferred bytes, range-read
amplification, metadata behavior, random scrub, forward playback, and eager
loads through HTTP Range and Crimson on the Mac/VPN path.
