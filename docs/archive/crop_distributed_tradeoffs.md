# Crop Distributed Tradeoffs

## Context
The crop stage writes `roi_images` (and related arrays) by detection index. When
running with a distributed scheduler, multiple workers may write into the same
Zarr chunk file unless detection boundaries align with chunk boundaries.

With **single fish / one detection per frame**, detection boundaries align to
frame boundaries, so matching the crop chunk size to the frame chunk size is
safe. With **multiple detections per frame**, boundaries are no longer aligned
and parallel chunk writes can corrupt data (e.g., all-black crops).

## Safe Today (Single Detection Per Frame)
- Use `crop.scheduler: distributed`.
- Ensure `roi_images.chunks[0]` matches the frame chunk size (usually
  `import.chunk_size`).
- This keeps each worker writing to disjoint Zarr chunks.

## Future Multi-Fish Per Frame: Options

### Option 1: Auto-fallback when unsafe
- Detect frames with >1 detection and force a non-distributed scheduler
  (`processes` or `single-threaded`).
- Pros: simple, safe, minimal code changes.
- Cons: slower for large datasets.

### Option 2: Detection-chunk tasks
- Build crop tasks by detection index ranges aligned to Zarr chunk boundaries.
- Each task decodes the frames needed for that detection slice, then writes
  a contiguous detection slice.
- Pros: distributed + safe even with multi-fish.
- Cons: more complex; decoding becomes less sequential; more bookkeeping.

### Option 3: Chunk-level locks
- Keep frame-based tasks but lock per Zarr chunk file before writing.
- Pros: maintains current task structure; prevents corruption.
- Cons: lock contention reduces speed; extra infra complexity.

### Option 4: Chunk size = 1 (always safe)
- Force detection chunk size to 1.
- Pros: trivially safe for any concurrency.
- Cons: massive number of small files; lower I/O efficiency.

## Recommendation When Multi-Fish Arrives
- Start with **Option 1** (auto-fallback) for correctness.
- Only invest in **Option 2** if distributed speed becomes a bottleneck.
