# Canonical Detection Storage: Full-Duration Reduction

Date: 2026-07-24

Status: full-duration evidence complete; neither candidate promoted

## Execution Identity

- Palette commit:
  `a5fb2d765923cefe0764e8d192e867abb858194e`
- Locked cluster checkout:
  `/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/shared-zarr-storage-policy-20260723-a5fb2d76`
- Workflow:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/workflows/sleepyfish_det_storage_full_duration_20260724_01`
- Matrix fingerprint:
  `9c138b458bad342d3f224d290b63d048d0efcc16a17eae80b307b82ea79b88f2`
- Scale: 1,188,000 frames, 1,187,087 instances, 4512x4512 source
- Repetition indices: 0 through 4
- LSF block array / finalizer: `153170772[1-5]%1` / `153170773`
- Execution host: `h07u19`

The source was the existing immutable benchmark-only fixture
`sleepyfish_cam2010095_detect_20260724_v1`, manifest SHA-256
`ae1b65b1e5255168bed320cf0d099b16ef9966255c6aed098182e33bf653062a`.
Metadata confirmed the exact full frame and instance cardinalities before
submission.

The reviewed shortlist contained only:

1. the regular 1 MiB target-chunk control; and
2. the 200k winner, with 128 KiB target inner chunks in 8 MiB target shards.

All five blocks and the success-dependent finalizer completed with return code
zero. All ten candidate records passed exact array and consumer-read checks,
the fixture identity remained unchanged, all six stderr files were empty, and
all five exact node-local scratch roots were absent afterward. Block totals
ranged from 253.4 to 257.8 seconds. The aggregate records zero registry,
selector, and training updates and no profile promotion.

## Five-Repetition Results

All values are medians except `PRFS p95`, which is the p95 across five fresh
reader-subprocess times.

| Layout | Pass | Payload objects | Local pipeline (s) | Publish (s) | PRFS median / p95 (s) | Eager offsets first / warm (ms) | Frame p95 (ms) | Rows p95 (ms) | Sequential FPS |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regular 1 MiB control | yes | 88 | 1.750 | 1.258 | 65.969 / 66.255 | 22.52 / 19.71 | 25.51 | 15.96 | 42,244 |
| sharded 128 KiB / 8 MiB | no | 16 | 2.148 | 0.503 | 59.614 / 61.424 | 37.89 / 31.62 | 19.87 | 13.91 | 47,304 |

Median write-phase peak RSS was 468,373,504 bytes for both candidates. Planned
payload counts were 93 and 17; actual counts were 88 and 16 because populated
edge/fill behavior reduced them slightly.

## Gate Result

The regular control passed every gate. The sharded candidate passed:

- local pipeline (`1.228x`, limit `1.25x`);
- publication (`0.400x`, limit `1.25x`);
- peak RSS (`1.000x`, limit `1.25x`);
- total PRFS reader time (`0.904x` median and `0.927x` p95);
- random-frame latency (`0.779x` first-pass median ratio);
- indexed-range latency (`0.872x` first-pass median ratio); and
- sequential traversal (`0.893x` first-pass consumer-time ratio).

It failed only complete eager `frame_row_offsets` reads:

| Pass | Sharded | Control | Ratio | Limit |
| --- | ---: | ---: | ---: | ---: |
| first median | 37.89 ms | 22.52 ms | 1.683 | 1.10 |
| warm median | 31.62 ms | 19.71 ms | 1.604 | 1.10 |
| first cross-repetition p95 | 41.51 ms | 24.00 ms | 1.729 | 1.20 |
| warm cross-repetition p95 | 35.95 ms | 20.44 ms | 1.759 | 1.20 |

The frozen reducer therefore selected only the regular 1 MiB control for the
next stage. That formal result must not be changed by weakening a threshold
after observing it.

## Interpretation And Next Experiment

The result does not support a global regular-layout decision. The sharded plan
created about 82% fewer payload objects, published about 60% faster, shortened
the complete PRFS suite by about 10%, and improved every windowed/random read
metric. Its only regression was an absolute 12--16 ms increase while eagerly
loading the complete offsets array.

This exposes the remaining weakness in applying one target inner-chunk byte
budget to every access class. `frame_row_offsets` is `EAGER` and benefits from
the control's 1 MiB chunk. The instance columns are `WINDOWED` and benefited
from 128 KiB chunks.

Before HTTP/Crimson promotion testing, add one policy-derived hybrid candidate:

- `EAGER` arrays: 1 MiB target inner chunks;
- `WINDOWED` instance columns: 128 KiB target inner chunks;
- immutable outer shards: 8 MiB target;
- no per-array row literals or writer overrides.

Benchmark the hybrid, original sharded candidate, and regular control for five
full-duration repetitions under predeclared gates. This tests the intended
access-pattern axis of the shared planner without changing dtype, logical
schema, codec, or consumer workloads. Only after that comparison should the
frontier move to HTTP Range and Crimson Mac/VPN validation.
