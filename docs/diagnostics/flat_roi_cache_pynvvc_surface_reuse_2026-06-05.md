# Flat ROI Cache PyNvVC Surface Reuse Diagnostic
<!-- contract-meta
status: current
last_verified: 2026-06-05
purpose: Record the GoodCopBadCop ROI-cache/keypoint jump investigation and the PyNvVideoCodec surface-lifetime fix.
-->

## Summary

GoodCopBadCop keypoints showed a persistent jump around frames 2399 to 2400 in
arena 3 even though the fish and refined detection bounding box were stable.
Crimson was rendering the persisted keypoints correctly; the bad value was
already present in `keypoints_img`.

Root cause: the flat ROI cache PyNvVideoCodec path retained decoded
decoder-owned frame tensors from `decode_next(N)` and cropped them later. The
decoder can reuse those surfaces, so earlier cache rows could be written from
later frame pixels while still being associated with the earlier row's declared
crop origin.

Fix: stream decoded frames with `iter_frames()`, crop immediately, clone/copy
only ROI tensors into owned staging buffers, then batch the owned ROI payloads
for host transfer and disk writes.

## Affected Example

Archive:

```text
/nvme1/recordings/2026-05-29T18-11-16Z_arena_3_GoodCopBadCop/zarr/2026-05-29T18-11-16Z_arena_3_GoodCopBadCop_analysis.zarr
```

Runs:

```text
crop_runs/crop_2026-06-04_21-12-28
keypoints_runs/keypoints_goodcopbadcop_traditional_v2_cache_20260604
refined_detect_runs/refined_detect_2026-06-04_15-39-00/instances
```

Cache:

```text
/nvme1/palette_roi_cache/goodcopbadcop_crop_cache_20260604/roi_cache/2026-05-29T18-11-16Z_arena_3_GoodCopBadCop_analysis__crop_2026-06-04_21-12-28__ee0d7b66731c.flat_roi_cache.json
```

Observed row pair:

```text
frame 2399 row 2281:
  crop origin: (3424, 2680)
  bbox centroid: (3680.10, 2936.33)
  swim bladder keypoint_img: (3668.66, 2912.23)
  keypoint minus bbox centroid: (-11.44, -24.09)
  cache pixel parity at declared origin: corr ~= 0.9996

frame 2400 row 2282:
  crop origin: (3424, 2680)
  bbox centroid: (3680.10, 2936.33)
  swim bladder keypoint_img: (3665.73, 3058.02)
  keypoint minus bbox centroid: (-14.37, +121.70)
  cache pixel parity at declared origin: corr ~= 0.184
  best-match offset: roughly (+8, -145)
```

The offset magnitude matched the persisted keypoint jump, showing that the
keypoint runner consumed cache pixels from a different effective crop origin
than the one later used to convert ROI keypoints back into image coordinates.

## Proof Of Mechanism

The pre-fix writer decoded a frame chunk, retained returned tensors, then
cropped after the decoder had advanced:

```text
frames = reader.decode_next(decode_count)
for frame_tensor in frames:
    crop frame_tensor later
```

Direct comparison showed:

- The existing cache row matched the old chunked decode/crop behavior.
- The same row decoded and cropped immediately with `reader.iter_frames()`
  matched the live CPU reference at the declared origin.

Therefore the bug was not in Crimson frame synchronization, not in
`keypoints_img` conversion, and not in the crop geometry arrays. It was a cache
pixel/origin mismatch caused by decoder surface reuse.

## Accepted Contract

For PyNvVideoCodec-backed flat ROI caches:

- Treat `torch.from_dlpack(frame)` tensors as decoder-owned borrowed surfaces.
- Do not retain decoded full-frame tensors across decoder advancement.
- Decode sequentially to the maximum requested frame.
- Skip frames that have no ROI rows.
- Crop frames with ROI rows immediately.
- Clone/copy only the derived ROI tensors into owned staging.
- Batch only owned ROI tensors/bytes for downstream host transfer and disk
  writes.

This keeps the persisted cache useful for multiple downstream consumers
including keypoints and segmentation while paying the decode cost once.

## Rejected Optimization

Batching crops after decoding a frame batch would require one of:

- a documented PyNvVideoCodec retained-surface guarantee;
- a lower-level decoder surface pool API where Palette explicitly owns/release
  surfaces;
- cloning every full decoded luma frame before decoder advancement.

With the current API, cloning full 4512x4512 luma frames is correct but too
expensive. It was tested during the investigation and did not improve observed
GoodCopBadCop cache throughput because full-frame ownership dominated the cost.

## Regeneration Policy

Any flat ROI cache built by the pre-fix `pynvvc_luma` writer should be treated
as suspect unless it passes pixel parity against a fresh reference. For the
GoodCopBadCop workflow, regenerate all four caches and rerun downstream
keypoint/segmentation runs that consumed them.

Use overwrite rather than manual deletion when possible:

```bash
scripts/py -m fisheye.utils.crop_flat_roi_cache_batch \
  /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
  --path-contains GoodCopBadCop \
  --source-type refined \
  --selection-policy full_recording \
  --workflow-id goodcopbadcop_crop_cache_20260604 \
  --cache-root /nvme1/palette_roi_cache \
  --cache-decode-backend pynvvc_luma \
  --roi-live-acceleration gpu \
  --overwrite-cache \
  --progress-stderr \
  --progress-interval-s 60 \
  --apply
```

Expected performance on the local A6000 during the investigation was about
175-180 ROI rows per second for this full-recording GoodCopBadCop workload,
roughly 13 minutes per arena. The speed is acceptable for correctness but not
yet an optimized final path.

## Transfer Optimization Follow-Up

The safe writer now batches only owned ROI tensors and transfers those batches
through reusable pinned host staging. The first transfer optimization copied
those pinned payloads into asynchronous writer-owned numpy buffers. A
2026-06-05 A6000 synthetic transfer check with a 1024x512x512 uint8 ROI batch
measured:

- pageable `.cpu().numpy()` path: mean about 0.116 seconds for 256 MiB;
- pinned staging path: mean about 0.010 seconds for 256 MiB.

This validates the transfer optimization, but it does not change the primary
runtime diagnosis for full GoodCopBadCop caches: sequential NVDEC decode remains
the dominant cost. The manifest timing block now separates
`gpu_cat_seconds_total` from `gpu_to_host_seconds_total` and records
pinned/pageable transfer counts so future cache artifacts can be compared
directly.

A follow-up optimization removed the pinned-to-numpy writer copy: the writer now
uses a small ring of pinned host buffers directly and waits for a buffer's write
future before reusing it. Contiguous row runs are written through the Python
buffer protocol rather than materializing a `bytes` payload first.

Arena 3 GoodCopBadCop full-cache timing showed the impact:

- pre-direct writer cache: 856.3 seconds, 165.7 rows/s;
- pinned transfer plus writer-owned numpy copy: 837.8 seconds, 169.4 rows/s;
- direct pinned-buffer writer: 813.3 seconds, 174.5 rows/s.

The direct pinned-buffer writer reduced `serialize_seconds_total` from about
30.4 seconds to about 0.05 seconds and `write_seconds_total` from about
55.7 seconds to about 22.2 seconds. Decode remained dominant at about
788 seconds, so future single-stream speedups require pipeline telemetry and
possibly a lower-level decode/crop pipeline rather than more host-copy tuning.

The follow-up telemetry build completed in about 791.1 seconds at 179.4 rows/s.
Key timing fields:

- `decode_seconds_total`: 765.4 seconds;
- `crop_seconds_total`: 18.3 seconds;
- `gpu_to_host_seconds_total`: 1.5 seconds;
- `transfer_submit_seconds_total`: 1.9 seconds;
- `serialize_seconds_total`: 0.05 seconds;
- `write_seconds_total`: 22.5 seconds, mostly hidden by async writing;
- `write_wait_seconds_total`: 0.10 seconds;
- `pinned_buffer_wait_seconds_total`: 0.001 seconds;
- `frame_lookup_seconds_total`: 0.17 seconds;
- `staging_append_seconds_total`: 2.1 seconds;
- `rows_mask_seconds_total`: 0.28 seconds;
- `progress_emit_seconds_total`: 0.28 seconds.

This rules out host transfer, writer backpressure, progress logging, and Python
row bookkeeping as the primary limit. The dominant bucket is the
`reader.iter_frames()`/`next(...)` path, which includes more than pure NVDEC
engine occupancy: CPU demux, packet feeding, Python/native boundary cost,
surface mapping, and synchronization can all be inside that wrapper-level
timing. If `nvidia-smi dmon` reports decode utilization around 50% during this
run, the next useful tests are either concurrent independent video streams or a
native decode/crop pipeline that can keep decode surfaces queued while crop,
transfer, and writes drain independently.

## PyNvVideoCodec API Check

Palette checked the installed PyNvVideoCodec package before pursuing a native
pipeline. Environment:

- package: `PyNvVideoCodec`;
- version: 2.1.0;
- relevant APIs present: `CreateDemuxer`, `CreateDecoder`,
  `SimpleDecoder.get_batch_frames`, `SimpleDecoder.get_batch_frames_by_index`,
  and `ThreadedDecoder.get_batch_frames`.

Bounded decode-only benchmark on the actual GoodCopBadCop arena 3 camera video
(`/nvme1/recordings/2026-05-29T18-11-16Z_arena_3_GoodCopBadCop/cams/Cam2010095_2026-05-29T18-11-16Z_arena_3.mp4`),
first 5,000 frames:

- `CreateDemuxer + CreateDecoder`: 183.9 fps;
- `SimpleDecoder.get_batch_frames(64)`: 182.4 fps;
- `ThreadedDecoder.get_batch_frames(64)`: 181.4 fps.

`ThreadedDecoder` is intended to decode in a background thread, but for this
single sequential high-resolution HEVC stream it did not improve raw frame
retrieval. Swapping the current cache reader to `ThreadedDecoder` is therefore
not expected to materially improve flat-cache build time.

A small retention probe found that both `SimpleDecoder` and `ThreadedDecoder`
kept sampled tensors stable after additional batches. That is encouraging, but
it is empirical behavior, not a sufficient contract to reintroduce retaining
decoded full-frame tensors across decoder advancement. The accepted safe policy
remains immediate ROI crop/clone from the current decoded surface.

Decision for now:

- Accept the current direct pinned-buffer flat-cache throughput as the
  production-safe single-video baseline.
- Scale production work as one video per GPU on the cluster.
- Do not add a `ThreadedDecoder` cache backend unless future benchmarks show a
  clear advantage on the target recording family.
- Defer native decode/crop pipeline work until single-video latency becomes a
  stronger bottleneck than correctness and operational simplicity.

## Follow-Ups

- Add a wrapper validation mode that samples parity after cache creation and
  fails if byte equality does not hold against the selected reference path.
- Consider a direct decode/crop-to-model path for workflows that do not need
  persisted ROI pixels.
- If persisted cache speed becomes limiting, investigate sparse seek decoding
  only for low-coverage recordings; for high-coverage recordings sequential
  decode remains reasonable because most frames have detections.
