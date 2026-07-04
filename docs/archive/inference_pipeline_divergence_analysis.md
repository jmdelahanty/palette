<!-- ARCHIVED 2026-07-04: dated point-in-time snapshot / spent work ticket, retained for history only. -->

# Inference Pipeline Divergence Analysis

**Date:** 2026-03-05
**Scope:** All YOLO-based inference paths in `palette/src/fisheye/`

## Overview

The fisheye module has three core inference engines, each with multiple entry
points. This document maps the pipelines, identifies where their image
preprocessing diverges, and proposes consolidation opportunities — with a
particular focus on image resizing/letterboxing and whether `imgsz` should be
read automatically from model metadata.

### Status Update (2026-03-05)

Detection entry points now use `resize_dims` as the canonical inference-size
parameter and treat `imgsz` as a legacy alias:

- Canonical: `resize_dims` (`[height, width]`)
- Legacy alias: `imgsz` (normalized into `resize_dims`)
- Conflict guard: providing both with different values raises an error
- Legacy `video.resize` pre-resize is still supported as fallback
  (`[width, height]`), but ignored when canonical detection sizing is set

---

## 1. The Three Core Engines

| Engine | Implementation | Input source | Model type |
|--------|---------------|--------------|------------|
| **Detection** | `detection/detect_yolo.py` → `detect_yolo()` | Raw video frames (Decord or OpenCV) | YOLO Detection |
| **Keypoints** | `detection/detect_keypoints_yolo.py` → `detect_keypoints_yolo()` | Zarr ROI crops (grayscale `(N, H, W)`) | YOLO Pose |
| **Eye Masks** | `segmentation/eye_segmentation_yolo.py` → `segment_eye_masks_yolo()` | Zarr ROI crops (grayscale `(N, H, W)`) | YOLO Segmentation |

## 2. Entry Points Per Engine

Each engine is reachable through several entry points. The wrappers are thin
pass-throughs — they parse CLI args and forward to the engine function. No
preprocessing divergence exists *between* entry points for the same engine.

### Detection

| Entry point | File |
|-------------|------|
| Direct CLI | `detection/detect_yolo.py` (`if __name__ == "__main__"`) |
| Inference wrapper | `inference/predict_detections.py` |
| Registry runner | `utils/run_detect_with_registry_model.py` |
| Batch runner | `utils/run_detections_batch.py` |

### Keypoints (Pose)

| Entry point | File |
|-------------|------|
| Direct CLI | `detection/detect_keypoints_yolo.py` (`if __name__ == "__main__"`) |
| Inference wrapper | `inference/predict_pose.py` |
| Registry runner | `utils/run_keypoints_with_registry_model.py` |
| Batch runner | `utils/run_keypoints_batch.py` |

### Eye Masks (Segmentation)

| Entry point | File |
|-------------|------|
| Direct CLI | `segmentation/eye_segmentation_yolo.py` (`if __name__ == "__main__"`) |
| Inference wrapper | `inference/predict_eye_masks.py` |
| Registry runner | `utils/run_eye_masks_with_registry_model.py` |
| Batch runner | `utils/run_eye_masks_batch.py` |

### Standalone Script

`scripts/predict_with_ultralytics.py` is a separate, self-contained detection
script that uses OpenCV and writes Ultralytics-format output (not Zarr). It has
its own resize logic (`cv2.resize` to a `target_shape`).

---

## 3. Image Preprocessing Per Engine

### 3a. Detection (`detect_yolo`)

The detection path is the most complex because it supports three video backends
and has **two independent resize mechanisms**:

**Video decode backends (in priority order):**

1. Decord GPU — `VideoReader(path, ctx=gpu(0))`
2. Decord CPU — `VideoReader(path, ctx=cpu())`
3. OpenCV fallback — `cv2.VideoCapture(path)`

**Resize mechanism 1 — legacy `video.resize` (manual pre-resize):**

Configured in YAML under `video.resize` (`[width, height]` legacy order).
In current detection code this is a fallback-only path:

- Applied only when canonical detection sizing is not set
  (`detection.resize_dims`, CLI `--resize-dims`, or legacy `--imgsz`)
- Ignored when canonical detection sizing is set

Current default config (`configs/fisheye/yolo_detect_config.yaml`) uses:

- `detection.resize_dims: [640, 640]`
- `video.resize: null`

When active, this is a **squish resize** — it forces exact `[w, h]` dimensions
with no aspect-ratio preservation.

**Resize mechanism 2 — canonical `resize_dims` mapped to YOLO `imgsz`:**

Canonical `[h, w]` detection sizing is normalized and passed as YOLO `imgsz`.
For numpy/list inputs, Ultralytics then applies its internal
aspect-ratio-preserving letterbox behavior.

**Other preprocessing:**
- BGR-to-RGB conversion (OpenCV backend only): `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
- Normalization to `[0, 1]`: `chunk.mul_(1.0 / 255.0)` (Decord GPU path only;
  Ultralytics handles this internally for numpy/list inputs)

### 3b. Keypoints (`detect_keypoints_yolo`)

- Load grayscale ROI crops from Zarr: `crop_group["roi_images"]` shape `(N, H, W)`
- Grayscale-to-RGB: `np.repeat(img[..., None], 3, axis=2)` via `_repeat_to_rgb()`
  (`detect_keypoints_yolo.py:277-280`)
- Pass list of `(H, W, 3)` numpy arrays to `model.predict(..., imgsz=imgsz)`
- YOLO handles letterboxing internally

### 3c. Eye Masks (`segment_eye_masks_yolo`)

- Identical input loading and grayscale-to-RGB conversion as keypoints
- Pass list of `(H, W, 3)` numpy arrays to `model.predict(..., imgsz=imgsz_resolved)`
- Post-inference: mask prototype upsampling, sigmoid, adaptive thresholding,
  connected component analysis, resize masks back to ROI dimensions
  (`_resize_prob_masks`, `eye_segmentation_yolo.py:431-443`)

---

## 4. `imgsz` Resolution Logic — Current Behavior

None of the engines read `imgsz` from the model's own metadata. Here is what
each engine does when the user does not provide a CLI sizing override:

| Engine | Default `imgsz` behavior | Code location |
|--------|-------------------------|---------------|
| **Detection** | Uses config-driven `detection.resize_dims` / `detection.imgsz` when present; otherwise falls through to Ultralytics default (640). The shipped config sets `detection.resize_dims: [640, 640]`, so most runs explicitly apply 640 via config rather than relying on the implicit fallback. | `detect_yolo.py:441-481`, `configs/fisheye/yolo_detect_config.yaml:25` |
| **Keypoints** | `max(roi_h, roi_w)` | `detect_keypoints_yolo.py:396` |
| **Eye Masks** | `max(roi_h, roi_w)` | `eye_segmentation_yolo.py:833-834` |

Ultralytics YOLO models store their trained image size in model metadata,
accessible via `model.overrides['imgsz']` or `model.model.args['imgsz']`. None
of the engines query this value.

---

## 5. Identified Divergences

### 5a. Legacy pre-resize path can still introduce aspect-ratio distortion

The strongest distortion risk today is the legacy `video.resize` fallback path,
which performs a squish resize before inference when canonical detection sizing
is not set.

Current detection logic does **not** stack canonical `resize_dims/imgsz` and
`video.resize` by default; when canonical detection sizing is set,
`video.resize` is ignored.

Distortion risk remains if users rely on legacy pre-resize for non-square input
frames.

### 5b. `imgsz` fallback differs between detection and crop-based pipelines

- Detection typically runs at configured 640 (`detection.resize_dims: [640, 640]`
  in the shipped config); if sizing is omitted from both CLI and config, it
  falls through to Ultralytics' built-in 640
- Keypoints/eye-masks default to `max(roi_h, roi_w)` — which depends on
  cropping parameters and has nothing to do with what the model was trained at

If a pose model was trained at `imgsz=256` but ROIs happen to be 120x80, the
inference `imgsz` becomes 120 — a significant resolution mismatch. Conversely,
if ROIs are 400x300, `imgsz` becomes 400, which may exceed what the model saw
during training.

### 5c. No engine reads `imgsz` from the model

This is the root cause of 5b and a broader source of training/inference size
mismatch risk. The model knows its trained input size, but inference code does
not currently query it.

### 5d. Duplicated grayscale-to-RGB conversion

Both `detect_keypoints_yolo.py` and `eye_segmentation_yolo.py` have their own
`_repeat_to_rgb()` implementation with identical logic.

### 5e. `_normalize_imgsz` exists only in detection

`detect_yolo.py` has a `_normalize_imgsz()` helper that handles list/tuple/int
coercion (`detect_yolo.py:286-307`). The other two engines do a simpler
`int(imgsz)` cast. If someone passes `--imgsz 256 256` to detection it works;
the same syntax would fail for keypoints or eye masks.

---

## 6. Training/Inference Parity

Training uses Ultralytics' built-in `preprocess_batch()` which applies
letterboxing to `imgsz`. The custom `ZarrYOLODataset` loader does **not** resize
images itself — it loads at original crop size and lets YOLO handle the rest.

This means training consistently uses **YOLO's aspect-ratio-preserving
letterbox**. Inference should do the same, but:

- Detection's `video.resize` path introduces a squish that training never applies
- Keypoints/eye-masks default `imgsz` to ROI dimensions rather than the trained
  size, so the letterbox target can differ from training

---

## 7. Aspect Ratio vs. Padding: The Resize Tradeoff

Letterboxing preserves aspect ratio but introduces padding pixels that the CNN
must process. Squishing eliminates padding but distorts features. This section
examines the tradeoff and the practical options.

### 7a. The three resize strategies

**Full square letterbox** (default YOLO behavior when `imgsz` is a single int):
Resize the longest side to `imgsz`, pad the shortest side with gray to make the
image square. Simple and correct, but potentially wasteful — a 1920x1080 frame
letterboxed to 640x640 is 44% padding.

**Rectangular inference** (`rect=True` in Ultralytics `model.predict()`):
Resize the longest side to `imgsz`, pad the shortest side only to the **nearest
stride multiple** (stride=32 for YOLO). That same 1920x1080 frame becomes
640x360, padded to 640x384 — only 6% padding. Aspect ratio is perfectly
preserved with near-zero waste.

**Squish resize** (what `video.resize: [640, 640]` currently does):
Force both dimensions to the target. Zero padding, zero waste, but geometric
distortion — circles become ellipses, angles shift. This is a distribution
shift from training, where YOLO uses letterboxing.

### 7b. Quantifying the padding cost

| Input aspect ratio | Full square waste | Rect (stride-32) waste |
|-------------------|-------------------|----------------------|
| 1:1 (square)      | 0%                | 0%                   |
| 4:3 (1440x1080)   | 25%               | 2%                   |
| 16:9 (1920x1080)  | 44%               | 6%                   |
| 2:1 (extreme)     | 50%               | 3%                   |

Rectangular inference recovers nearly all of the speed loss from letterboxing.
The remaining few percent of padding pixels (to reach stride alignment) is
negligible in practice.

### 7c. When squishing is tolerable

For small aspect-ratio deviations (images that are already close to square),
squishing causes minimal distortion and models tend to tolerate it. The ROI
crops in the keypoints and eye-mask pipelines are bounding-box regions around
individual fish — they are typically close to square, so the difference between
letterbox and squish is small for those pipelines.

For the detection pipeline processing full video frames, the deviation is larger
(fisheye cameras are typically 16:9 or 4:3), so squishing matters more and
should be avoided.

### 7d. Recommendation

**Use rectangular inference (`rect=True`) for detection where Ultralytics is
fed numpy/list frames, and validate equivalent behavior on the Decord-GPU
`torch.Tensor` path before making it the global default.** This preserves aspect
ratio and can recover most padding overhead.

**For ROI crops** (keypoints, eye masks), standard letterboxing is typically
fine. The crops are small and close to square, so padding overhead is usually
modest.

**De-emphasize the manual `video.resize` squish path.** Keep it only as explicit
legacy fallback, not as a preferred default.

The key mental model: **padding is cheap, distortion is expensive.** A few
percent of wasted FLOPs on gray pixels is almost always preferable to changing
the geometric relationships the model learned during training. Rectangular
inference makes that "few percent" truly negligible.

### 7e. Batching constraint

One caveat: batched inference requires all images in a batch to share the same
dimensions. If input images have varying aspect ratios, rectangular inference
works best when batches are sorted by aspect ratio (Ultralytics does this
automatically when `rect=True` in validation/prediction). For video inference
where every frame has the same resolution, this is not a concern — all frames
share the same aspect ratio and will get the same rectangular padding.

---

## 8. Proposed Consolidations

> Note: P2 below is informed by the tradeoff analysis in section 7.

### P1. Auto-read `imgsz` from model metadata (all engines)

Add a shared helper that queries the model's trained `imgsz` and preserves
either square or rectangular `[h, w]` metadata:

```python
def resolve_imgsz(
    model: YOLO,
    user_imgsz: Optional[int | list[int] | tuple[int, int]] = None,
) -> int | list[int]:
    """Return user override, or the model's trained imgsz."""
    normalized_user = _normalize_imgsz(user_imgsz)
    if normalized_user is not None:
        return normalized_user

    model_args = getattr(getattr(model, "model", None), "args", None)
    raw_candidates = [
        getattr(model, "overrides", {}).get("imgsz"),
        model_args.get("imgsz") if isinstance(model_args, dict) else getattr(model_args, "imgsz", None),
    ]
    for raw in raw_candidates:
        normalized = _normalize_imgsz(raw)
        if normalized is not None:
            return normalized

    return 640  # Ultralytics default
```

All three engines would call this after loading the model, replacing their
current fallback logic. This ensures the model always receives inputs at the
resolution it was trained on unless explicitly overridden, including models
trained with rectangular `[h, w]` sizes.

### P2. Remove `video.resize` from detection (or make it aspect-ratio-aware)

The manual pre-resize is redundant when `imgsz` is passed to `model.predict()`.
Options:

- **Remove it entirely.** Let YOLO's internal letterbox handle sizing. Simpler,
  correct by default.
- **Replace with aspect-ratio-preserving resize.** If there is a performance
  reason to downscale before hitting YOLO (e.g., reducing memory for very
  high-res video), use the letterbox logic already in `import_video.py`
  (`_compute_letterbox_dims`) instead of a squish resize.

### P3. Shared grayscale-to-RGB utility

Extract the duplicated `_repeat_to_rgb()` into a shared module (e.g.,
`fisheye/utils/image_ops.py` or similar) and import from both keypoints and
eye-mask engines.

### P4. Unified `imgsz` normalization

Move `_normalize_imgsz()` from `detect_yolo.py` to a shared location and use
it in all three engines so they all handle `int`, `list[int]`, and `tuple`
inputs consistently.

---

## 9. Priority Ranking

| Priority | Item | Rationale |
|----------|------|-----------|
| **High** | P1 — Auto-read `imgsz` from model | Eliminates training/inference mismatch for all pipelines with minimal code change |
| **High** | P2 — Remove/fix `video.resize` | Prevents silent aspect-ratio distortion in detection |
| **Low** | P3 — Shared `_repeat_to_rgb` | Code dedup; no behavioral impact |
| **Low** | P4 — Shared `_normalize_imgsz` | Code dedup; minor consistency improvement |

---

## 10. Pre-Materialized Crops vs. Live Cropping: Critical Evaluation

This section evaluates whether the current architecture — pre-materializing ROI
crops into Zarr (`crop_runs/<run>/roi_images`) — is a sound tradeoff versus
computing crops on demand from full frames and detection bounding boxes.

### 10a. What the crop datastream actually costs

**Code complexity:**

| Component | Lines | Purpose |
|-----------|-------|---------|
| `tracking/crop.py` | ~2,560 | Core crop creation (GPU/CPU paths, Dask parallelism, kvikIO, provenance) |
| `utils/crop_batch.py` | ~680 | Batch runner with signature-based staleness detection |
| `registry/step_cascade.py` | ~50 | Cascade invalidation when crops change |
| `shared/keypoint_stale.py` | ~200 | Granular staleness marking for downstream stages |

That is **~3,500 lines** of infrastructure to maintain a cache. This is not
"store some pixels" — it is a full lifecycle management system with GPU decode
paths, Dask-distributed writes, signature hashing, cascade invalidation, and
provenance tracking.

**Storage cost:**

Each crop is `(roi_h, roi_w)` uint8 — typically 512x512 = 262 KB per crop.
For a recording with 100K detections, that is ~26 GB of crop data (before lz4
compression, which typically achieves ~60-70% of original for uint8 image data).
This is stored *in addition to* the source video.

**Operational cost:**

In the standard batch path, detection changes (new model, refined bboxes,
interpolation) generally trigger full crop regeneration. The batch runner
detects staleness via signature comparison and reruns the crop stage, after
which downstream keypoints are typically invalidated and must be re-run.

That said, the repo does have targeted repair utilities for limited subset
updates (for example, patching selected crops or recomputing refined/interpolated
ROIs), so "no incremental update path" would be too strong as a blanket claim.

### 10b. How crops are actually consumed

A practical read-pattern summary is:

| Consumer class | Typical read pattern |
|----------|-------------|
| Core inference stages (YOLO/traditional keypoints, segmentation) | Usually single-pass per stage run |
| Training exports / merged artifact builders | Usually one pass per export job |
| Training loaders (merged datasets) | Repeated reads across epochs |
| Tuning/diagnostics/visualization tools | Repeated or random on-demand reads |

So the "read once then idle" characterization is accurate for some production
pipelines, but not for all repo workflows. There are meaningful recurring
consumers (tuners, diagnostics, visualizers, and dataset loaders) that can read
`roi_images` multiple times depending on usage.

The storage-duplication argument still stands, but should be framed as
workload-dependent rather than universal.
### 10c. Steelmanning the case for live cropping

Your boss's position is stronger than the initial framing suggests. Here is why:

**1. The "reuse" argument does not hold for inference.**

The core justification for pre-materialized crops was "I wouldn't have to
perform a 2-stage network." But the current architecture already performs two
stages sequentially — detection, then crop, then pose inference. The crop step
just happens to persist its output to disk between stages. Replacing it with an
in-memory crop-and-forward would eliminate I/O without changing the computational
cost. The detection bboxes and frame indices are already stored in the Zarr;
computing `frame[y1:y2, x1:x2]` from a decoded frame is negligible compared to
model inference time.

**2. The "reproducibility" argument has a gap.**

Pre-materialized crops guarantee "same pixels every run" only if you never
regenerate them. But the architecture requires full regeneration whenever
detections change — and detections change frequently (new models, refinement,
interpolation). After regeneration, the crop run has a new timestamp and new
pixels. The reproducibility guarantee is already scoped to a specific run, which
is exactly what you would get from a deterministic live-crop function that takes
(frame, bbox, roi_size) → crop.

**3. The training argument is real, but the current pipeline is still coupled to
per-recording crops.**

Training ultimately benefits from merged training artifacts, but the current
export/preparation path still sources ROI tensors from per-recording
`crop_runs/roi_images`. So while a future export step could crop on the fly
from video + bboxes and still emit the same merged training zarr, that is not
how the repo works today. Eliminating per-recording crop storage would require
refactoring the upstream export/build pipeline, not just changing the final
training reader.

**4. The TensorRT argument is valid but narrower than the current dependency
graph.**

Crop-based TensorRT models do need stable crop datasets for calibration and
validation. In principle that could be satisfied by a merged training zarr
alone, but in the current codebase those merged artifacts are still built from
per-recording materialized crops. So this is a plausible future simplification,
not something the present pipeline already supports.

**5. The infrastructure tax is real and compounds.**

3,500 lines of crop lifecycle code is a significant maintenance surface. Every
schema change, every new detection source type, every new storage backend must
account for crop materialization. The cascade invalidation system exists
*because* of the materialized cache — if crops were computed on demand, there
would be nothing to invalidate.

### 10d. Where live cropping would struggle

**Random frame access in compressed video is slow.** Crops require decoding
specific frames from video. Sequential access (processing all frames in order)
is fast with Decord/OpenCV, but random access (training shuffles samples across
frames) requires seeking, which is expensive for inter-frame codecs like H.264.
This is the strongest argument for materialization and applies most directly to
training/merged-artifact workflows. In the current repo it also matters for
some per-recording review/tuning tools that jump around ROIs, even though the
main sequential inference path is a better fit for live cropping.

**Full frames may not be in the Zarr.** The current schema makes
`raw_video/images_full` optional. Some recordings may only have the external
video file or downsampled frames. Live cropping would need a reliable path back
to source pixels, which may mean keeping the video file path in metadata (it
already is, as `video_source_path` in crop run attrs).

### 10e. Verdict

**Per-recording `crop_runs/roi_images` materialization remains technically
sound, but looks over-provisioned as a universal default.**

It remains a strong fit when:

- ROI tensors are reused heavily (interactive tuning/review loops)
- reproducibility requires persisted ROI pixels for a specific run
- workflows depend on tools that currently require `roi_images`

It is weaker as a default when:

- workloads are primarily one-pass inference
- per-recording crop runs are generated once and rarely revisited
- storage and lifecycle complexity dominate the runtime savings

### 10f. Recommended direction

Update note, 2026-06-04: this section predates the mixed-mode reader migration.
YOLO pose, YOLO eye masks, U-Net eye masks, U-Net subject masks, subject
segmentation, SAM subject masks, and several visualizers now consume
`CropImageSource` and can read geometry-only crop runs or flat ROI cache
manifests. See `docs/crop_reader_geometry_only_inventory_2026-05-16.md` for the
current reader inventory and
`docs/geometry_only_crop_workflow_cache_design.md` for the active cache design.

1. **Add a compatibility phase before any default switch.** Today, multiple
   components still hard-require `crop_runs/<run>/roi_images` (YOLO
   keypoints/eye masks, traditional pipelines, tuners, and several diagnostics).

2. **Introduce live cropping as an opt-in inference path first.** Start with the
   core sequential path (detection -> keypoints/eye masks), validate correctness
   and throughput, then expand to adjacent tools.

3. **Keep crop materialization as an explicit export/cache operation.** The merged
   training zarr remains the right long-lived artifact for repeated training
   epochs.

4. **Retain optional per-recording materialization for heavy review/tuning loops.**
   Treat this as an explicit cache (`--materialize-crops`), not a universal
   default.

5. **Add retention/GC policy for materialized crop runs.** Bound storage
   duplication and clean stale cache runs automatically.
