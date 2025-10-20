Here’s a clean, **Codex-friendly TODO markdown** you can drop into the repo (e.g., `TODO_optimize_softmask_pipeline.md`).
It summarizes the next optimization steps and clarifies dependencies so future you (or another engineer) can prompt Codex or GPT-5 to pick up the thread easily.

---

# 🧠 TODO: Optimize YOLO Soft-Mask Pipeline (Deferred Work)

**Context:**
Current implementation in `src/fisheye/segmentation/eye_segmentation_yolo.py` captures *ingredients* (protos, coeffs, boxes) and lazily computes soft masks at `_pop_cached_prob_masks_any`.
Ultralytics’ native `process_mask_native()` still executes fully (matmul + sigmoid + crop + upsample), so total work is duplicated.
We rely on the native path for correct high-res fallbacks and telemetry, so forcing `upsample=False` is unsafe until we replace or patch the result tensor.

---

## ✅ Baseline (now)

* Hook captures `_MaskCacheEntry(proto, coeffs, boxes, shape_hw, upsample)` for each ROI.
* `_pop_cached_prob_masks_any()` computes soft masks lazily on cache pop.
* `_resolve_soft_masks()` uses **cache-first**, then reconstruct, then native `masks.data`.
* All downstream consumers still see high-res `result.masks` from Ultralytics.
* Performance improved vs binary fallback, but native path still recomputes everything.

---

## 🧩 Next Optimization Steps

### 1️⃣ **Eliminate duplicate matmul safely**

* Idea: Intercept `result.masks.data` and replace with our computed high-res masks *after* inference.
* Then `process_mask_native` could be run with `upsample=False` (proto-grid size).
* Requires verifying:

  * `result.masks.orig_shape` and `masks.data` consistency.
  * Any Ultralytics post-processing (e.g. NMS or Retina pipeline) isn’t broken.
* Expected gain: remove one full `coeffs @ protos + upsample` pass per ROI.

### 2️⃣ **Optional: Full patch of `process_mask_native`**

* Implement a lightweight replacement:

  * Skip Ultralytics matmul entirely when cache path active.
  * Return a tensor placeholder to satisfy model internals.
* High risk across UL versions — isolate behind `--fast-softmask` flag.

### 3️⃣ **GPU-resident lazy compute**

* Avoid `.cpu()` inside `_pop_cached_prob_masks_any`.
* Perform `F.interpolate` on GPU, then transfer once with `to("cpu", non_blocking=True)` after upsample.
* Measure using NVTX or `torch.cuda.Event` timing.

### 4️⃣ **Batch reconstruction**

* Instead of per-ROI compute, batch soft-mask reconstruction per model call (`results` list).
* Combine coeffs/protos across instances; fewer kernel launches.

### 5️⃣ **Profiling & Benchmarks**

* Add optional timing hooks (rich or `torch.cuda.Event`) for:

  * Hook capture latency
  * `_compute_native_soft_masks`
  * Ultralytics native call
* Compare:

  * (a) current full path
  * (b) cache-only
  * (c) cache + native upsample=False (after safe interception)

---

## 🚧 Safety Notes

* Fallbacks and external consumers (e.g., contour tools) still **depend on high-res `result.masks.data`**.
  Never disable native upsample until:

  1. We explicitly overwrite `result.masks.data` with our high-res tensor.
  2. All downstream modules are verified against the new source.

* Keep version guards:

  ```python
  from ultralytics import __version__ as UL_VERSION
  if UL_VERSION < "8.1.0": ...
  ```

---

## 🧩 Prompt Hooks for Codex / GPT-5

To resume this work later:

```
# Resume optimization work:
"Explain how to intercept result.masks.data safely in Ultralytics YOLO to remove duplicate matmul without breaking fallbacks."

# Benchmark guidance:
"Instrument torch.cuda.Event timing around _compute_native_soft_masks and native process_mask_native to compare kernel cost."

# Fast-path prototype:
"Design a context-managed fast softmask patch that disables native upsample but injects our computed high-res tensor back into result.masks.data."
```
