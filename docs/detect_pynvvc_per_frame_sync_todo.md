# Detect path: per-frame CUDA sync in PyNvVC batch preprocessing

Status: open (measure before changing)
Date: 2026-06-10
Source: codebase review (docs/diagnostics/codebase_review_2026-06-10.md, Performance & IO)

## Where

- `src/fisheye/detection/detect_yolo.py` — `_read_and_preprocess_pynvvc_batch`
  (per-frame loop around lines 476-516)
- `src/fisheye/shared/pynvvc_luma_rgb.py` — `preprocess_luma_rgb` /
  `preprocess_nv12_rgb` (called with a one-element list per frame)

## CUDA execution model (the background that makes this make sense)

CUDA is a mail queue, not a phone call. Kernel launches (`F.interpolate`,
`.copy_`, `.mul`) are *enqueued* on a stream and return to Python in
microseconds; the GPU drains the queue at its own pace. The CPU's job is to
keep the queue full. `event.synchronize()` blocks the host until the GPU has
finished everything enqueued on that stream **up to where the event was
recorded** — a full stop: GPU drains, goes idle, CPU wakes, refills queue.

A CUDA event marks a *position in a stream*, so syncing it waits on the whole
preprocess chain queued before it — not just "this frame's materialization" as
the in-code comment claims. It is narrower than a device-wide sync, but wider
than the comment believes.

## Current behavior and its three costs

Per frame, the loop: decodes (NVDEC), runs the full preprocess chain as a
batch-of-1 (`preprocess_luma_rgb([frame_tensor], ...)`), copies into the
preallocated `processed_batch`, then records and synchronizes a CUDA event.

1. **Host/device ping-pong** — one full stop per frame (256 per io-batch);
   GPU idles during Python loop overhead, CPU idles during GPU work.
2. **Batch-of-1 kernels** — `F.interpolate` on `[1,1,H,W]` per frame; kernel
   launch overhead (~5-10 us) unamortized, low occupancy. `processed_batch` is
   already preallocated for the whole batch but is fed one frame at a time.
3. **Lost NVDEC/compute overlap** — NVDEC is a separate hardware engine;
   frame N+1 could decode while frame N preprocesses, but the per-frame sync
   serializes the two engines.

## Why the sync exists (and what part of the reasoning is correct)

PyNvVideoCodec returns tensors backed by decoder-owned surfaces that may be
recycled on the next decode call. Reading a recycled surface produced a real
keypoint-corruption bug (see
docs/diagnostics/flat_roi_cache_pynvvc_surface_reuse_2026-06-05.md). The
safety contract is correct and host-side: surface reuse is triggered by the
host calling `next()`, so the host must not advance until the rescuing copy
has *executed* — only a host sync guarantees that. The mistake is scope, not
principle:

> The contract requires only that the surface bytes are copied into owned
> memory before the next decode. It does not require that resize, channel
> expansion, or normalization have finished.

## Planned fix: rescue minimally per frame, batch the work

Phase 1 (per frame, minimal): D2D-copy the raw luma plane (or NV12 frame)
into a preallocated owned staging buffer (~20 MB at ~1 TB/s = tens of us),
record an event, sync on that copy only, then advance the decoder.

Phase 2 (once per batch, fully async): one batched
`F.interpolate([N,1,H,W])` + expand + normalize over the staging buffer, no
syncs, flowing straight into inference on the stream.

```python
staging = torch.empty((max_batch_frames, source_height, source_width),
                      device=device, dtype=torch.uint8)
copied = torch.cuda.Event()
for i in range(max_batch_frames):
    frame_tensor = next(frame_iter)
    staging[i].copy_(frame_tensor[:source_height, :])  # minimal rescue
    copied.record(stream)
    copied.synchronize()                               # waits on a memcpy only

luma = staging[:actual_count].unsqueeze(1).to(dtype)
resized = F.interpolate(luma, size=(h, w), mode="bilinear", align_corners=False)
batch = resized.expand(-1, 3, -1, -1).mul(1.0 / 255.0) \
               .contiguous(memory_format=torch.channels_last)
```

The same restructuring applies to the NV12 path (currently three float32
plane interpolations per frame, batch-of-1).

General principle: when a safety rule forces serialization, isolate the
smallest operation the rule actually constrains, serialize only that, and
batch everything else.

## Future work (do these BEFORE and WITH the change)

- [ ] **Profile first with Nsight Systems**: `nsys profile` a short detect run
      and inspect the timeline. If NVDEC decode is the long pole at 20MP, the
      compute side has idle headroom and this fix buys little — decide from
      the timeline, not from this doc.
- [ ] **Replace host-clock GPU timing with CUDA events.** The existing
      `preprocess_seconds` numbers are only accurate *because* the per-frame
      sync forces execution to finish inside the timed region.
      `time.perf_counter()` around async launches measures enqueue time
      (microseconds), not execution time — after removing the syncs the
      timing dict will report near-zero while wall-clock stays real. Measure
      GPU stages with paired `torch.cuda.Event(enable_timing=True)`:
      `start.record(); ...; end.record(); end.synchronize();
      start.elapsed_time(end)` — or torch.profiler. Audit other timing dicts
      that bracket GPU work with perf_counter for the same hazard (e.g.
      flat_roi_cache write-path timings).
- [ ] **Verify pixel parity after the change** using the existing
      pynvvc pixel-parity test pattern
      (tests/unit/fisheye/test_training_crop_pynvvc_pixel_parity.py) so the
      surface-reuse bug class cannot silently return.
- [ ] **Benchmark before/after** on a real 20MP recording (frames/sec for the
      read+preprocess phase, end-to-end detect wall-clock) and record the
      numbers here.
