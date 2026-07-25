# YOLO Detection Sharding Canary

**Date:** 2026-07-12
**Status:** direct writer and isolated cluster canary passed; indexed sharding enabled by default

## Design

YOLO detection differs from keypoint inference: the current detector already
materializes all results in memory and writes them only after video inference
finishes. The sharded path therefore does not need a second pair of streaming
buffers. It creates empty indexed-sharded arrays and writes each complete outer
shard once from the materialized tables.

The canary candidate, now the production default, uses:

```text
--detect-row-shard-rows 262144
--detect-frame-shard-rows 262144
```

The existing inner chunks remain unchanged. The five detection-row arrays share
one aligned outer grid; `frame_counts` and `n_detections` use an independent
frame-row grid. The writer hashes all seven source arrays, rereads every
destination shard, and requires exact decoded-byte SHA-256 equality before run
completion. Run attrs record `detect_storage_layout`, the requested shard rows,
and `detect_shard_write` timings and hashes.

Implementation commits:

- `30db8171` — writer, CLI/registry/batch propagation, and focused tests
- `cb8b9afa` — isolated LSF A/B harness and audit
- `b5ff083c` — storage-replay parity separated from repeated-inference variance

## Cluster Canary

The benchmark used the first 100 seconds (`10,000` frames) of GoodCopBadCop
camera `2010093`, the production YOLO11n model, PyNvVideoCodec NV12 decoding,
640×640 inference, batch size 16, confidence 0.4, IoU 0.8, and `max_det=1`.
Both inference runs were written to one isolated benchmark Zarr under:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
  yolo_detection_sharding_20260712_01/
```

No canonical recording run selector or registry row was changed.

| Job | Host | Result |
| --- | --- | --- |
| `153064929` | `h08u14` (NVIDIA L4) | both inference runs completed; initial cross-inference exact audit exited 1 |
| `153064941` | `h07u18` | materialized writer replay and exact audit passed |
| `153064942` | `h07u18` | read-only semantic audit of the two inference runs passed |

## Writer Parity And Storage

The ordinary run's 9,830-row materialized table was replayed through the
production sharded writer. All seven arrays matched exactly, including bounding
boxes, scores, frame mappings, counts, classes, and `instance_key`.

| Measure | Ordinary chunks | Indexed shards | Change |
| --- | ---: | ---: | ---: |
| Total files | 50 | 14 | `3.57x` fewer |
| Payload files | 42 | 6 | `7.0x` fewer |
| Apparent bytes | 180,379 | 203,122 | `+12.6%` |
| Allocated bytes | 198,144 | 207,360 | `+4.7%` |

`class_ids` was all fill-value zero and therefore required no physical payload
file. The apparent-byte increase is only 22.7 KB and is dominated by shard
indexes and metadata.

The replay writer spent `0.081 s` writing and `0.041 s` exhaustively rereading
and validating the output. The end-to-end sharded inference writer independently
reported exact internal source/destination hashes, `0.126 s` writing, and
`0.057 s` validation.

For this short clip, the frame arrays' 10,000-row inner chunk made the effective
outer frame shard 270,000 rows: requested shard rows are rounded upward to a
complete inner-chunk multiple. Normal longer recordings use 16,384-row geometry
chunks, for which 262,144 is already aligned.

## Runtime And Repeated-Inference Variance

| Measure | Ordinary inference | Sharded inference |
| --- | ---: | ---: |
| Inference duration | 81.0 s | 76.1 s |
| Whole command elapsed | 104.4 s | 88.2 s |
| Maximum process RSS | 2,058,956 KiB | 2,080,588 KiB |
| Total Zarr write timing | 0.478 s | 0.376 s |

The sharded pass ran second on the same node, so the faster result is consistent
with model/decoder warm-up and is not evidence that sharding speeds inference.
There is no material runtime or memory regression, and physical output writing
is negligible relative to decoding and inference.

The two independent GPU inference passes were not bit-deterministic: the first
reported 9,830 detections and the second 9,829. Frame 4,904 crossed the
confidence threshold only in the first pass, and shared-frame floating outputs
also differed at the bit level. The initial job therefore exited 1 under an
overly strict cross-inference audit. This is separate from storage correctness:
both writers validated their own materialized inputs, and the ordinary-to-
sharded storage replay was exact for every array.

## Decision

The indexed-sharded YOLO detection writer passes correctness, publication,
object-count, runtime, and memory gates. The 262,144-row layout is now the
default for immutable serial YOLO detection outputs. Use
`--no-detect-sharding` only for explicit compatibility tests or regular-chunk
benchmarks. Blob/traditional writers are unchanged.
