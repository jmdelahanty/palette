# Zarr Sharding Design Note

## Purpose

Define how (and when) we use sharding to reduce NFS metadata pressure while
keeping the pipeline usable for a single-node workflow. Sharding remains
optional until we validate it across all import paths.

## Constraints and assumptions

- Current workloads are single node, write-once read-many.
- Future workloads may run on a cluster; NFS metadata is the risk.
- kvikIO/GDS path may not reliably support sharding (needs validation).

## Current behavior (as implemented)

- Standard import (`create_palette_zarr`) writes sharded `raw_video` arrays.
- kvikIO/GDS import uses `use_sharding`; default config now enables sharding.
- Derived arrays (crops, refined runs, masks) are chunk-only.

## Design principles

- Sharding should be explicit and auditable in metadata.
- Default behavior should remain stable while we validate.
- Prefer measurement-driven shard sizes over hard-coded guesses.
- Do not block usage if sharding is unsupported (fallback cleanly).

## Proposed policy (conceptual)

- Raw video:
  - Shard target size based on frame size (e.g., 128-256 MB per shard).
  - Record frames-per-shard and shard shape in `raw_video.attrs`.
- Crop runs:
  - Shard only the large image arrays (`roi_images`).
  - Smaller arrays remain chunk-only.
- Other runs:
  - Keep chunk-only until measured, then shard if file counts become large.

## Configuration approach (future)

- Add a storage section, for example:
  - `storage.sharding_enabled`: bool (default false or "auto")
  - `storage.shard_target_mb_raw`: int
  - `storage.shard_target_mb_crops`: int
  - `storage.shard_policy`: "raw_only" | "raw_and_crops"

## Validation steps

1. Measure file counts with and without sharding on NFS.
2. Compare open/read times (visualizers + crop access).
3. Validate that sharding works on the kvikIO/GDS path.
4. Confirm that downstream tools read sharded arrays without changes.

## 2026-02-10 Crop kvikIO observation and TODO

Observed during `crop_batch` on analysis archives (GPU + kvikIO):

- `external_use_sharding=false` sustained higher crop throughput (~70-80 crops/s on 4512x4512, ROI 512x512).
- `external_use_sharding=true` showed earlier slowdown in long runs.
- Warning appears in both cases:
  - `zarr.buffer.gpu.NDBuffer ... does not support __cuda_array_interface__ ... falling back to slow copy`
- With sharding enabled, an additional warning can appear:
  - `zarr.buffer.gpu.Buffer ... does not support __cuda_array_interface__ ... falling back to slow copy`

Current interpretation:

- The warning is emitted when GPU-buffer mode is enabled and Zarr receives CPU/NumPy arrays.
- In crop flow, likely sites are metadata/aux writes (for example `save_crop_metadata(...)`) and CPU coordinate writes, not the core ROI image path itself.
- Sharding can add additional overhead for this incremental write pattern, so non-sharded chunks are currently the safer/high-throughput operator default.

Confirmed trace (2026-02-10):

- Warning A (`NDBuffer`, `core.py:425`) maps to generic array writes under GPU buffer mode (for example metadata/aux arrays written from NumPy while `zarr.config.enable_gpu()` is active).
- Warning B (`Buffer`, `core.py:178`) is sharding-specific in this workflow:
  - `roi_images` sharded arrays use `sharding_indexed` with index codecs including `crc32c`.
  - The `crc32c` codec path builds CPU arrays (`np.append(...)`) and then calls `prototype.buffer.from_array_like(...)`, which emits the GPU Buffer slow-copy warning under GPU prototype mode.

TODO (no behavior change yet):

1. Keep crop default at `external_use_sharding=false` for production until warning pressure and throughput regressions are resolved.
2. Split crop writes into:
   - GPU-heavy ROI writes (kvikIO path)
   - CPU metadata/aux writes in a CPU-buffer context.
3. Evaluate sharding-index codec/prototype handling so shard index writes do not force GPU Buffer slow-copy warnings on CPU-built byte buffers.
4. Re-benchmark `sharding=false` vs `sharding=true` after steps 2-3.
5. Update default profile guidance if warning volume and throughput improve.
6. Keep `--require-kvikio` enabled for production crop batches.

## GPU CRC32C feasibility and integration plan (2026-02-10)

Findings:

- Current Zarr CRC path in our stack is CPU-oriented (`zarr.codecs.crc32c_` calling Python `crc32c`).
- `sharding_indexed` does not require CRC32C for correctness; CRC32C is an integrity layer (default in current Zarr index codec chain).
- NVIDIA nvCOMP exposes GPU CRC APIs and supports CRC32-C (Castagnoli polynomial), but there is no drop-in `zarr-python` GPU CRC32C backend in our current environment.
- The observed crop warnings are buffer-prototype conversion warnings, not checksum mismatches.

Near-term recommendation:

- Keep index CRC32C enabled for safety.
- Keep crop sharding default off for production (`external_use_sharding=false`) until warning/perf behavior is stabilized.
- Add explicit post-run validation so integrity checks remain operator-visible.

Proposed integration plan:

1. Phase 0: post-run integrity checks (low risk)
   - Add a lightweight audit command that forces shard-index decode/validation for `roi_images` arrays (exercises current CRC32C checks).
   - Optionally add a full payload hash manifest (`sha256`) for archival verification.
2. Phase 1: pluggable CRC backend abstraction
   - Introduce a small internal CRC interface (`cpu` now, `gpu` optional later).
   - Add config flag (`storage.crc32c_backend=cpu|gpu|auto`) with deterministic fallback to CPU.
3. Phase 2: GPU prototype (experimental)
   - Build a focused nvCOMP adapter for CRC32C over GPU buffers.
   - Scope first to shard-index related bytes path to minimize blast radius.
   - Record backend provenance in run attrs (`crc32c_backend`, `crc32c_backend_effective`).
4. Phase 3: validation and go/no-go
   - Correctness parity against CPU CRC32C on representative archives.
   - Throughput/latency benchmarks with and without sharding.
   - Promote only if measurable end-to-end gain exceeds complexity cost.

Risks and caveats:

- Additional CUDA/nvCOMP dependency surface and deployment complexity.
- Limited potential speedup for small shard-index payloads.
- Warning elimination may be better solved by CPU/GPU prototype routing than by GPU CRC itself.

## Post-import packing (optional)

If sharding is unavailable at import time (e.g., kvikIO/GDS), a post-import
"pack" step can rewrite the dataset into a sharded store.

Notes:
- This is a full rewrite of array data into shard files (not a metadata toggle).
- Requires temporary storage during the pack.
- Best run once after the pipeline is complete (write-once/read-many).

## Decision points

- Should sharding be the default for raw_video on NFS?
- Should the kvikIO/GDS path disable sharding if unsupported, with a warning?
- Do we shard crops by default once validated?

## Recommended profiles

### Local / dev (fast import, fewer guarantees)

- Sharding: optional (off by default for faster imports).
- Use when working on a local NVMe and not sharing datasets.
- Example config: `configs/fisheye/import_local.yaml`

### Cluster / NFS (stable reads, fewer files)

- Sharding: on by default for `raw_video`.
- Use when datasets live on NFS or when metadata IOPS is a concern.
- Example config: `configs/fisheye/import_nfs.yaml`
