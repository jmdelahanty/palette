# TensorRT INT8 Export TODO

Goal: add post-training INT8 export with calibration that fits the current ONNX → trtexec pipeline.

## Decisions
- **Quantization path**: PTQ (calibration) first; skip QAT for now.
- **Calibration data source**: representative `images_ds` frames for detection.
- **Preprocessing**: must match inference exactly (resize/letterbox, gray→3‑channel, normalization).
- **Calibration set size**: target 500–2000 images; include varied lighting/contrast.

## Required steps
1) **Calibration dataset selector**
   - Choose frames from Zarr (sample across time).
   - Record dataset list + frame indices in a manifest.

2) **Calibration cache builder**
   - Script to load frames, apply preprocessing, feed batches to TRT calibrator.
   - Output: `calibration.cache` file.
   - Save SHA256 + metadata (input shape, preprocessing settings).

3) **INT8 engine build**
   - Use `trtexec --int8` and point it at the cache (confirm exact flag with `trtexec --help`).
   - Enforce fixed input shape; avoid dynamic shapes in calibration.

4) **Provenance logging**
   - Extend engine manifest:
     - `calibration_cache_path`, `calibration_cache_sha256`
     - `calibration_manifest_path` (frames used)
     - `calibration_method` (entropy/minmax)
     - preprocessing params + input shape

5) **Quality validation**
   - Compare INT8 vs FP16 on a held‑out subset.
   - If accuracy drops: increase calibration set size or revisit preprocessing.

## Open questions
- Should we store a small calibration Zarr manifest in the registry?
- How large should the calibration set be per camera/rig?
- Do we want a `--calib-only` mode to generate cache without building an engine?
