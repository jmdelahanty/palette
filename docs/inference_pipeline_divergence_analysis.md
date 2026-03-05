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

**Resize mechanism 1 — `video.resize` (manual pre-resize):**

Configured in YAML under `video.resize` (e.g., `[640, 640]`). Applied *before*
YOLO sees the frame:

- Decord GPU path: `F.interpolate(chunk, size=resize_dims, mode='bilinear')`
  (`detect_yolo.py:841-846`)
- Decord CPU path: `cv2.resize(frame, tuple(resize_dims))` per frame
  (`detect_yolo.py:867-869`)
- OpenCV path: `cv2.resize(frame, tuple(resize_dims))`
  (`detect_yolo.py:914-915`)

This is a **squish resize** — it forces frames to exact `[w, h]` dimensions
with **no aspect ratio preservation**. The current `yolo_detect_config.yaml`
sets this to `[640, 640]`.

**Resize mechanism 2 — `imgsz` (YOLO internal letterbox):**

Passed to `model.predict(..., imgsz=imgsz)` (`detect_yolo.py:462-463`).
Ultralytics internally applies **aspect-ratio-preserving letterboxing** with
gray padding to fit the image into the target size.

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
each engine does when the user does not provide `imgsz`:

| Engine | Default `imgsz` behavior | Code location |
|--------|-------------------------|---------------|
| **Detection** | Falls through to Ultralytics default (640) | `detect_yolo.py:373` — `_normalize_imgsz(None)` returns `None`, so `imgsz` is omitted from `predict_kwargs` |
| **Keypoints** | `max(roi_h, roi_w)` | `detect_keypoints_yolo.py:396` |
| **Eye Masks** | `max(roi_h, roi_w)` | `eye_segmentation_yolo.py:833-834` |

Ultralytics YOLO models store their trained image size in model metadata,
accessible via `model.overrides['imgsz']` or `model.model.args['imgsz']`. None
of the engines query this value.

---

## 5. Identified Divergences

### 5a. Detection has two stacking resize operations

The `video.resize` squish and `imgsz` letterbox can both be active
simultaneously. If `yolo_detect_config.yaml` has `video.resize: [640, 640]` and
the user also passes `--imgsz 640`:

1. Frame is squished from original aspect ratio to 640x640 (distorted)
2. YOLO letterboxes 640x640 to 640x640 (no-op, but the distortion persists)

If the original video is not square, the fish are distorted before the model
ever sees them. The model was likely trained on letterboxed (not squished) data,
so this is a **training/inference mismatch**.

### 5b. `imgsz` fallback differs between detection and crop-based pipelines

- Detection defaults to Ultralytics' built-in 640
- Keypoints/eye-masks default to `max(roi_h, roi_w)` — which depends on
  cropping parameters and has nothing to do with what the model was trained at

If a pose model was trained at `imgsz=256` but ROIs happen to be 120x80, the
inference `imgsz` becomes 120 — a significant resolution mismatch. Conversely,
if ROIs are 400x300, `imgsz` becomes 400, which may exceed what the model saw
during training.

### 5c. No engine reads `imgsz` from the model

This is the root cause of both 5a and 5b. The model knows its trained input
size, but the inference code never asks.

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

**Use rectangular inference (`rect=True`) for detection on video frames.** This
is a one-argument change to `model.predict()` and reclaims nearly all the
padding overhead while preserving aspect ratio exactly.

**For ROI crops** (keypoints, eye masks), standard letterboxing is fine. The
crops are small and close to square, so padding overhead is already minimal.

**Remove the manual `video.resize` squish.** It solves a performance problem
that `rect=True` solves correctly without introducing distortion.

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

Add a shared helper that queries the model's trained `imgsz`:

```python
def resolve_imgsz(model: YOLO, user_imgsz: Optional[int] = None) -> int:
    """Return user override, or the model's trained imgsz."""
    if user_imgsz is not None:
        return user_imgsz
    try:
        return int(model.overrides['imgsz'])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return int(model.model.args['imgsz'])
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return 640  # Ultralytics default
```

All three engines would call this after loading the model, replacing their
current fallback logic. This ensures the model always receives inputs at the
resolution it was trained on unless explicitly overridden.

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
