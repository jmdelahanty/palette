# Sleepyfish Keypoint Sharding Canary

**Date:** 2026-07-12
**Status:** clone canary complete; storage candidate passed; inference-writer rollout not yet implemented

## Scope

The canary cloned one completed immutable clipped-collection keypoint run
without modifying its source:

```text
keypoint_shard_runs/
  keypoint_shard_sleepyfish_kp_allclips_20260708_01_clip_000000
```

The source contains `54,000` ROI rows, five keypoints, and 24 numeric arrays.
Its physical size is only `8,832,358` bytes, but its ordinary Zarr chunks
occupy 742 files, including 717 payload files.

## Candidate Layout

- ROI-domain arrays retain their existing `1,024`- or `16,384`-row inner
  chunks and use aligned `65,536`-row outer shards.
- The three frame-domain arrays retain `1,024`-row inner chunks and use aligned
  `262,144`-row outer shards.
- Every destination outer shard is written in one operation and then reread.
- Every array is validated using a decoded-byte SHA-256 digest.

One complete ROI-domain shard across all 21 ROI arrays is approximately
`25.95 MB`; two writer buffers would require approximately `51.9 MB`. One
frame-domain shard across all three frame arrays is approximately `3.15 MB`.

## Execution

LSF job `153064781` ran on `h07u22` in the `short` queue. The workload ran from
shared-checkout commit `1cb2aaf`; the login host was used only for submission
and status queries. LSF runtime was 21 seconds and maximum accounted memory was
84 MB.

Benchmark artifact and report:

```text
/groups/johnson/johnsonlab/jeremy/recordings/logs/keypoint_sharding_canary/
  sleepyfish_clip000000_20260712_01/
```

## Results

All 24 arrays passed exact decoded-byte equality. Run attributes and completion
metadata were copied into an isolated group stamped `benchmark_only=true`.

| Measure | Ordinary chunks | Indexed shards | Change |
| --- | ---: | ---: | ---: |
| Total files | 742 | 45 | `16.5x` fewer |
| Payload files | 717 | 20 | `35.9x` fewer |
| Apparent bytes | 8,832,358 | 8,867,727 | `+0.4%` |

Clone plus exhaustive destination validation took `4.26 s`. Sparse all-fill
arrays remained payload-free; each populated ROI array used one physical shard,
and each populated frame-domain array also used one physical shard for this
clip's nonzero range.

Seven repeated warm PRFS reads of `keypoints_roi` were stable:

| Read | Ordinary median | Sharded median | Ratio |
| --- | ---: | ---: | ---: |
| 256 random rows | `266 ms` | `501 ms` | `1.88x` |
| contiguous 1,024 rows | `1.48 ms` | `3.16 ms` | `2.14x` |
| full 54,000-row scan | `22.5 ms` | `49.0 ms` | `2.18x` |

Indexed-sharding codec/index handling roughly doubled these small-array read
times, matching the direction observed for probability-mask sharding. Absolute
latencies remain small, while filesystem-object pressure falls sharply.

## Decision And Next Step

The storage candidate passes the first canary. Do not rewrite the completed
source run. The next step is a parity-tested YOLO inference-writer canary using
two aligned `65,536`-row ROI buffers and complete-shard ownership. The serial
YOLO batch writer is the safest first production target. Traditional/Dask
keypoint writers must not write different logical slices within one physical
shard; they need complete-shard worker ownership or worker-local outputs plus a
deterministic merge.

Do not make sharding the keypoint default until the inference-writer canary
passes output parity, completion/publication, file-count, memory, and runtime
gates.
