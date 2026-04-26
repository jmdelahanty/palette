# KvikIO / GPUDirect Storage Subject-Mask Experiment

Purpose: evaluate whether KvikIO/GPUDirect Storage should become an output
backend for U-Net subject-mask inference.

## Current Decision

Do not route the subject-mask operator through KvikIO/GDS by default yet.

Reasons:

- the normal fast path is already stable and reasonably fast:
  probability-first output, async Zarr writes, no dense `masks_roi`
- a profiled arena 2 canary still shows possible IO upside, but IO is not the
  only bottleneck
- KvikIO is installed and GDS is visible outside the Codex sandbox, but both
  `kvikio.CuFile` and `kvikio.zarr.GDSStore` currently segfault during Python
  interpreter teardown on this workstation after successful writes
- embedding that teardown failure in the Palette operator would make otherwise
  successful inference runs exit as failed

## Observed Baseline

Command shape:

```bash
scripts/py -m fisheye.segmentation.infer_unet_subject_masks \
  /nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr \
  --resolve-model-from-registry \
  --registry /nvme1/palette_registry.sqlite \
  --model-coverage-class dense_all_components \
  --model-component-coverage-key body+eyes+swim_bladder \
  --model-label-schema-id subject_v1_union \
  --run-name subject_masks_unet_registry_default_fast_profile_2026-04-26 \
  --crop-run crop_2026-02-10_21-44-51 \
  --assignment-keypoint-group refined_keypoints_runs \
  --assignment-keypoint-run refined_keypoints_2026-03-02_13-46-40 \
  --device 0 \
  --batch-size 128 \
  --mask-probs-dtype uint8 \
  --mask-probs-chunk-rois 32 \
  --profile-timings \
  --overwrite
```

Result:

- `19,235` ROIs processed in `98.5s`
- `sync_after_forward`: `59.36s`
- `metric_compute`: `30.12s`
- `output_write_probs`: `25.22s`
- `roi_read`: `22.42s`
- `d2h_copy`: `14.33s`
- `model_forward`: `1.09s`
- `h2d_copy`: `0.34s`
- `output_queue_drain`: `0.22s`

Interpretation:

- there is enough `output_write_probs` and `d2h_copy` time to justify
  experimentation
- metric computation is also a major remaining CPU-side cost
- a GDS writer alone will not address the full bottleneck stack

## Local KvikIO Findings

Outside the Codex sandbox, the current `palette-py311` environment reports:

- `kvikio` is installed
- `kvikio.cufile_driver.get("is_gds_available")` is `True`
- `/nvme1` is mounted as `ext4` with `data=ordered`

Observed probes:

- `kvikio.zarr.GDSStore` can open a Zarr store and write data, but the child
  process exits with signal 11 during interpreter teardown
- `kvikio.CuFile.pwrite(...)` can write valid host-buffer and GPU-buffer data,
  but the child process exits with signal 11 during interpreter teardown
- forcing `os._exit(0)` after a successful child write avoids the teardown
  crash, which strongly suggests a KvikIO/cuFile cleanup issue rather than a
  failed write

## Teardown Root-Cause Assessment

Additional isolated probes narrowed the unsafe behavior:

- `import kvikio` exits cleanly
- `kvikio.cufile_driver.get("is_gds_available")` exits cleanly
- creating and closing a `kvikio.CuFile` handle is enough to exit with signal
  11, even without writing bytes
- `KVIKIO_COMPAT_MODE=ON` exits cleanly for the same open/close probe
- `KVIKIO_COMPAT_MODE=AUTO` on `/nvme1` successfully opens the file, then
  exits with signal 11 during normal interpreter teardown
- `os._exit(0)` after successful GDS writes avoids the crash

Current local versions:

- `kvikio`: `25.08.00`
- `libcufile`: `(1, 14)`
- cuFile driver major/minor reported by KvikIO: `2.27`

Most likely cause:

- a native KvikIO/cuFile/GDS cleanup issue triggered after a non-compat
  `CuFile` handle has been created
- plausible mechanisms include destructor/finalizer ordering, CUDA/cuFile
  context lifetime, `nvidia-fs` cleanup, or KvikIO/libcufile version skew

Less likely causes:

- Palette Zarr metadata or array layout, because `CuFile` open/close alone
  reproduces the crash
- CuPy/GPU buffers, because host-buffer `CuFile` probes also crash
- write alignment or file payload size, because zero-byte/no-write probes still
  crash after handle creation
- failed IO, because files are written correctly before teardown

Decision:

- defer further KvikIO/GDS integration work for now
- revisit after a KvikIO/CUDA/cuFile environment update, or if profiling shows
  `output_write_probs`/`d2h_copy` has become the dominant remaining bottleneck
- do not add KvikIO to the main inference process while normal teardown exits
  with signal 11

## Diagnostic Command

Run outside the Codex sandbox:

```bash
scripts/py -m fisheye.diagnostics.benchmark_kvikio_gds \
  --scratch-dir /tmp \
  --size-mib 64
```

This runs risky KvikIO work in child processes and reports whether each child
exits cleanly, fails, or dies from a signal. It also runs `os._exit` variants to
distinguish write-path failure from interpreter-teardown failure.

The command returns non-zero if any child probe fails or dies from a signal. On
the current workstation that non-zero exit is expected because the normal
teardown probes hit signal 11.

Use `--skip-normal-teardown` only when measuring write throughput and not
teardown safety.

## Future Backend Criteria

Only add an operator-facing KvikIO/GDS backend after all are true:

- `benchmark_kvikio_gds` normal-teardown probes exit with status `success`
- `GDSStore` or a lower-level writer can preserve the Palette Zarr v3 contract
  without custom fragile chunk metadata logic
- a canary shows material improvement over the current fast default path
- the backend records explicit attrs such as `output_store_backend`,
  `kvikio_gds_available`, and whether GPU buffers were written directly
- normal Zarr remains the default fallback

If KvikIO remains teardown-unsafe but raw CuFile write speed is compelling, the
only defensible design is an isolated helper process with a strict artifact
contract. Do not put unsafe KvikIO teardown into the main inference process.
