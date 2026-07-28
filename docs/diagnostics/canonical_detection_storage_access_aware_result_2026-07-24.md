# Canonical Detection Storage: Access-Aware Result And Crimson Handoff

Date: 2026-07-24

Status: cluster comparison complete; access-aware hybrid is the consumer-test
finalist; no production profile promoted

## Outcome

The access-aware hybrid is the best practical candidate from the cluster
comparison:

- `WINDOWED` instance columns use approximately 128 KiB uncompressed inner
  chunks;
- the `EAGER` `frame_row_offsets` array uses approximately 1 MiB uncompressed
  inner chunks; and
- every immutable array is stored in approximately 8 MiB uncompressed outer
  shards.

Compared with the regular 1 MiB unsharded control, the hybrid reduced payload
objects from 88 to 16, cut median publication from 1.192 to 0.514 seconds,
shortened the complete PRFS read subprocess from 66.035 to 58.501 seconds, and
improved random-frame, indexed-row, and sequential workloads. Its median local
write pipeline increased from 1.728 to 2.083 seconds and peak RSS was unchanged.

The hybrid missed the frozen relative gate only for the complete eager offsets
read. The absolute differences were small: `+5.619` ms on the first pass and
`+3.106` ms on the same-process warm pass. The frozen reducer correctly retained
the regular control as its formal next-stage selection; the result must not be
rewritten after observation. The hybrid nevertheless remains the Crimson
finalist because it passed the frozen p95 gates and the proposed consumer-side
absolute limits by a wide margin. No storage profile may be promoted until the
real Crimson/Mac mounted-network test is complete.

## Execution Identity

- Palette commit:
  `5def5d6de1ca477f1f91b53f4b328f02c6805295`
- Commit-pinned cluster checkout:
  `/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/shared-zarr-storage-policy-20260723-5def5d6d`
- Workflow:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/workflows/sleepyfish_det_storage_access_aware_full_20260724_03`
- Matrix fingerprint:
  `8628a442a682a090dc8e4c159d39cd5e270d93aa1835911fdce86c2e22d7dedd`
- Frozen reduction:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/reductions/sleepyfish_det_storage_access_aware_full_v1_repetitions_0_4_20260724_01/aggregate.json`
- Scale: 1,188,000 frames, 1,187,087 instances, 4512x4512 source
- Repetition indices: 0 through 4
- LSF block array / finalizer: `153171547[1-5]%1` / `153171548`
- Execution host: `h07u07`

The immutable benchmark-only fixture was
`sleepyfish_cam2010095_detect_20260724_v1`, manifest SHA-256
`ae1b65b1e5255168bed320cf0d099b16ef9966255c6aed098182e33bf653062a`.
All five blocks and all 15 candidate trials completed. The aggregate records no
registry, selector, training-artifact, or profile update. The source fixture was
unchanged, stderr was empty, and all exact node-local scratch roots were absent
after completion.

## Compared Layouts

| Label | Inner chunk policy | Outer shards |
| --- | --- | --- |
| `regular__chunk_1048576` | 1 MiB for every access class | none |
| `sharded__chunk_131072__shard_8388608` | 128 KiB for every access class | 8 MiB |
| `sharded__chunk_131072__eager_chunk_1048576__shard_8388608` | 128 KiB `WINDOWED`; 1 MiB `EAGER` | 8 MiB |

The hybrid's `int64 (1,188,001,)` offsets array has an outer shape of
`(1,048,576,)` and an inner shape of `(131,072,)`, corresponding to 8 MiB and
1 MiB uncompressed. As a representative instance column, `float32 (N,4)`
`bbox_img_xyxy` has outer shape `(524,288,4)` and inner shape `(8,192,4)`,
corresponding to 8 MiB and 128 KiB uncompressed.

The hybrid is Zarr v3. Its arrays use `sharding_indexed`; inner payload codecs
are little-endian `bytes` followed by Zstandard level 0 without a checksum.
Shard indexes use little-endian `bytes` followed by `crc32c`, stored at the end
of the shard. The root contains inline consolidated metadata. Crimson must
prove that its exact TensorStore build accepts this chain; Python success is
not a consumer compatibility result.

## Five-Repetition Results

Values are medians across the five balanced repetitions. `Frame p95` and `Rows
p95` are the per-operation p95 values reduced across repetitions.

| Layout | Apparent bytes | Objects | Local pipeline (s) | Publish (s) | PRFS reader (s) | Eager offsets first / warm (ms) | Frame p95 (ms) | Rows p95 (ms) | Sequential FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regular 1 MiB | 27,502,561 | 88 | 1.728 | 1.192 | 66.035 | 23.04 / 19.74 | 25.24 | 15.78 | 41,933 |
| uniform sharded 128 KiB / 8 MiB | 27,481,804 | 16 | 2.109 | 0.509 | 59.319 | 37.94 / 31.97 | 18.89 | 13.74 | 46,830 |
| access-aware hybrid | 27,500,336 | 16 | 2.083 | 0.514 | 58.501 | 28.66 / 22.85 | 19.81 | 13.55 | 47,792 |

Median peak RSS was 468,578,304 bytes for all three candidates. The hybrid
relative to the regular control had:

- `0.182x` as many payload objects;
- `1.206x` local pipeline time, within the frozen `1.25x` limit;
- `0.431x` publication time;
- `0.886x` complete PRFS reader time;
- `0.785x` random-frame p95 latency;
- `0.859x` indexed-row p95 latency; and
- `1.140x` sequential frame throughput.

## Frozen Gate Result

The hybrid passed every scalar and cross-repetition p95 gate. It failed only
the two median eager-offset ratios:

| Gate | Hybrid | Control | Ratio | Frozen limit | Absolute difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| first median | 28.66 ms | 23.04 ms | 1.244 | 1.10 | +5.62 ms |
| warm median | 22.85 ms | 19.74 ms | 1.157 | 1.10 | +3.11 ms |

Its first and warm cross-repetition p95 ratios were `1.159` and `1.176`, both
within the frozen `1.20` limit. The reducer therefore selected only the regular
control and set `profile_promoted=false` and `next_stage_only=true`.

The Crimson source review proposed, but has not yet established as a release
contract, these absolute consumer gates:

- complete selected-run offsets under 100 ms warm, tolerating at least 25 ms
  absolute regression;
- lightweight random analysis frame/window under 150 ms p95 on the mounted
  network path; and
- storage throughput at least twice the 700 FPS source rate.

The cluster hybrid satisfies those provisional limits. This does not replace
the missing Crimson measurement.

## Crimson Test Stores

Use the matching regular and hybrid stores from each repetition. This avoids
reusing one path for every process-first trial and preserves balanced ordering.
For example, repetition 000 contains:

- control:
  `.../frames_full/repetition_000/frames_full__regular__chunk_1048576__1d67be02d6c9.zarr`
- hybrid:
  `.../frames_full/repetition_000/frames_full__sharded__chunk_131072__eager_chunk_1048576__shard_8388608__3224926babc9.zarr`

The omitted prefix is:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/workflows/sleepyfish_det_storage_access_aware_full_20260724_03/candidates/sleepyfish_det_storage_access_aware_full_20260724_03`

On macOS, substitute the actual mounted Johnson Lab prefix for `/groups`; do
not copy the stores to the Mac's local SSD for the mounted-network result. The
stores are standalone, selector-ineligible benchmark groups rather than full
analysis archives. They are appropriate for a direct adapter benchmark. A
separate noncanonical full-archive fixture is required to measure complete
Crimson archive initialization and consolidated discovery.

## Crimson Validation Protocol

### 1. Freeze The Consumer Environment

- Record Crimson commit, build configuration, TensorStore version, macOS
  version, machine, VPN state, mount path, and whether the file or HTTP kvstore
  is in use.
- Test the current zero-byte TensorStore cache as a compatibility baseline and
  one fixed nonzero candidate cache, initially 128 MiB, as the intended
  operating configuration. Do not combine these samples.
- Run at least five process-first repetitions per layout. Rotate control and
  hybrid order, use the matching repetition path, and report median and p95.
- Call the first pass `process-first, OS/filesystem cache uncontrolled` unless
  cache eviction is actually enforced and recorded. Do not label it cold by
  assumption.

The current mounted PRFS path is TensorStore's file kvstore, not HTTP. Record
file-range behavior there. If HTTP delivery is intended, run the same protocol
again through a request-logging HTTP Range server and report it as a separate
storage tier.

### 2. Prove Schema And Codec Compatibility First

Open the canonical paths with their exact declared dtypes, without trying a
sequence of alternative types:

| Path | Exact dtype and shape |
| --- | --- |
| `instances/frame_indices` | `int32 (N,)` |
| `instances/source_acquisition_frame_index` | `int64 (N,)` |
| `instances/instance_key` | `uint64 (N,)` |
| `instances/bbox_norm_coords` | `float32 (N,4)` |
| `instances/bbox_img_xyxy` | `float32 (N,4)` |
| `instances/centers_img_xy` | `float32 (N,2)` |
| `instances/scores` | `float32 (N,)` |
| `instances/class_ids` | `int32 (N,)` |
| `instances/frame_row_offsets` | `int64 (F+1,)` |

Require the consolidated and explicit-metadata paths to describe the same
shape, dtype, chunks, shards, codecs, and logical schema. Verify offsets are
monotone, start at zero, end at `N`, and that every sampled slice
`offsets[f]:offsets[f+1]` contains only rows whose `frame_indices == f`.
Compare decoded values between the regular and hybrid layouts for all sampled
operations. A codec, CRC, dtype, or value mismatch is an immediate failure and
must not be averaged into performance results.

### 3. Exercise The Future Offsets Adapter

For only the selected run:

1. read the complete `frame_row_offsets` array once during adapter
   initialization;
2. retain it for the adapter's lifetime;
3. resolve each later frame entirely from the retained vector; and
4. verify through telemetry that scrubbing and playback cause no additional
   offset-array reads.

Do not reconstruct offsets by scanning `frame_indices` or reading
`frame_counts`. Do not reproduce Palette's diagnostic two-offset read on every
random frame: that workload was useful for comparing storage granularity, but
it is not the intended Crimson implementation.

Record selected-run metadata/open time, full-offset read time, retained bytes,
and time to first usable detection overlay. The hybrid promotion target is
warm offsets below 100 ms and no more than 25 ms absolute regression from the
control.

### 4. Run Both Contract-Shaped And UI-Shaped Reads

Use Palette's deterministic workload as the contract comparison:

- seed `20260724`;
- 128 random frames without replacement;
- 64 random 32-row observation ranges, using seed `20260725`;
- two passes in the same process; and
- full sequential traversal in 700-frame windows with a 700 FPS source target.

For each frame, derive the row slice from the retained offsets. The
contract-shaped pass reads all eight instance arrays so it remains comparable
with the Palette evidence. Add a second UI-shaped pass that reads only the
columns actually needed by the enabled Crimson overlay. Report these separately.

Also exercise actual Crimson behavior:

- selected-run initialization;
- deterministic random scrubbing with stale-generation cancellation;
- forward and reverse playback;
- chunk-aligned read-ahead of 0.5 and 1.0 seconds, rather than only 6--12
  frames; and
- at least one complete optional-array or full-scan operation that Crimson
  genuinely performs.

Primary provisional gates are random-frame p95 below 150 ms, forward storage
throughput of at least 1,400 frames/second, no playback deadline misses after
prefetch warmup, and no material full-archive initialization regression. The
last gate requires the future full-archive fixture; the standalone store cannot
establish it.

### 5. Measure Physical I/O And Caching

Expose TensorStore file metrics and, where possible, corroborate them with
`fs_usage` on macOS. Record at least:

- metadata file reads and typed-open attempts;
- file/range read count, batched-read count, transferred bytes, and latency;
- shard-index reads and repeated index reads;
- decoded/logical bytes returned to the adapter;
- TensorStore cache limit and observable cache hits;
- Crimson presentation-cache hits, cancellations, and evictions; and
- per-workload initialization, p50, p95, maximum, and throughput.

A one-frame request should fetch the required inner chunk ranges, not every
complete 8 MiB shard. The nonzero-cache run should demonstrate whether shard
indexes and decoded chunks are reused. If physical byte/range telemetry is not
available, record that as missing evidence rather than inferring it from wall
time.

### 6. Promotion Decision

Promote neither layout from Palette's cluster result alone. The hybrid is ready
for profile versioning only if:

- exact schema, codec, CRC, and decoded-value checks pass;
- the persisted-offset adapter replaces scans and repeated offset reads;
- the absolute initialization, scrub, playback, and throughput gates pass on
  the actual Mac/VPN mount;
- file/range evidence shows bounded read amplification;
- consolidated discovery is either consumed by Crimson or explicitly left as
  a separately tracked adapter task; and
- a noncanonical full-archive fixture shows that the profile does not worsen
  complete archive initialization.

If it passes, version the access-aware profile and then integrate the canonical
detection writer. If it fails, retain the complete trace and identify whether
the cause is physical layout, the zero-cache TensorStore context, serial
per-array reads, metadata probing, or application scheduling before changing
chunk or shard budgets.

## Benchmark-Harness Correction

Two earlier workflow IDs are preserved as non-selection evidence:

- `_01` was plan-only and was not submitted after review found candidate-order
  imbalance.
- `_02` submitted from commit `3406b8a2`; fail-closed plan validation detected
  that access-specific budgets were not forwarded to worker subprocesses. The
  remaining array elements were terminated, the forwarding bug was fixed in
  `5def5d6d`, and the complete comparison was rerun as `_03`.

Both affected only the dedicated benchmark namespace. Neither changed a
canonical archive, registry, selector, training artifact, or production
profile.

## Implementation Validation

The most comprehensive unit/contract invocation passed 77 tests. The focused
candidate-order invocation passed 15 tests. After correcting subprocess option
forwarding, the outside-sandbox real-Zarr lifecycle and matrix invocation
passed 16 tests. Static compilation and `git diff --check` also passed. These
invocations overlap and must not be presented as one summed test count.
