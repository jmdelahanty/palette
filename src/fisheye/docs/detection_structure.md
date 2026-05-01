# Palette Detection Run Layout

This note summarizes how detection results are stored under `detect_runs/` for both
the traditional (blob-based) and YOLO pipelines. Use it as a reference when writing
downstream tooling or inspecting Zarr archives by hand.

---

## 1. Where to Look

```
<archive>.zarr/
  detect_runs/
    @latest = "detect_2025-02-12_19-04-41"   # helper for CLI tools
    detect_2025-02-12_19-04-41/
      frame_indices
      bbox_norm_coords
      frame_counts
      ...
```

All detection runs live under `detect_runs/<run_name>`.  The parent group keeps a
`latest` attribute so interactive tools can pick the most recent run automatically.

---

## 2. Shared Datasets

Both detection modes emit the same core geometry arrays:

| Dataset              | Shape / dtype            | Level          | Meaning |
| -------------------- | ------------------------ | -------------- | ------- |
| `frame_indices`      | `(N,) int32`             | detection      | Frame index for each detection (0-based). Matches the imported video frame order. |
| `bbox_norm_coords`   | `(N, 4) float64`         | detection      | Normalised `[cx, cy, width, height]` in **ROI pixels ÷ ROI dims**. Values lie in `[0, 1]`. |
| `frame_counts`       | `(num_frames,) int32`    | per-frame      | Number of detections per source frame (derived from `frame_indices`). |

Notes:
- `frame_indices`/`bbox_norm_coords` are parallel arrays – use the same index to access matching detections.
- `frame_counts` is convenient when plotting occupancy over time or when you need to re-aggregate per-frame stats quickly.

### Optional datasets

| Dataset        | Emitted by | Purpose |
| -------------- | ---------- | ------- |
| `scores`       | YOLO only  | Per-detection confidence (`float32`). Traditional blobs do not produce classifier scores. |
| `n_detections` | YOLO only  | Duplicate of `frame_counts` kept for backwards compatibility with legacy tooling. |

Downstream code should **check for existence** before consuming optional datasets:

```python
group = root["detect_runs"][run_name]
scores = group["scores"][:] if "scores" in group else None
```

---

## 3. Run-Level Attributes

Every detection run records provenance to make later refinement/audit effortless:

| Attribute                 | Traditional (`blob`)                         | YOLO (`yolo`) |
| ------------------------- | -------------------------------------------- | ------------- |
| `detection_method`        | `"blob"`                                     | `"yolo"`      |
| `detect_timestamp_utc`    | ISO timestamp when the run finished          | same          |
| `duration_seconds`        | Wall clock runtime                           | ✓             |
| `parameters`              | Serialized parameter dict (e.g. bg-sub settings) | Detection thresholds, resize dims, batch size |
| `summary_statistics`      | Frame coverage, distribution, totals         | Coverage, totals, min/mean/max score |
| `git_*` / environment     | Commit hash, hostname, dependency versions   | Git + runtime hardware info (GPU, CUDA, etc.) |

### Method-specific extras

- **Traditional (blob)** runs also store:
  - `detection_source`: `"zarr_video"` (operates on the imported grayscale video stack).
  - `source_background_run`: which background model was used.
  - `dask_scheduler`: scheduler mode when multiple workers were used.
  - `code_version`: full metadata block including numpy/skimage versions.

- **YOLO** runs additionally record:
  - `detection_source`: typically `"external_video"` (reads straight from the source video).
  - `model_path`, `model_name`, `model_type` (always `"yolo_object_detection"`).
  - `parameters` include network-specific thresholds (`conf`, `iou`, `max_det`, batch size, resize dims).
  - Hardware summary: `gpu_available`, `gpu_name`, CUDA version, plus an embedded JSON (`_full_environment_info`) mirroring the training runtime.

Because the attribute dictionaries can evolve, prefer `dict.get(...)` accessors instead of indexing by key blindly.

---

## 4. Working With the Data

### Loading detections

```python
import zarr
import numpy as np

root = zarr.open("experiment.zarr", mode="r")
run_name = root["detect_runs"].attrs["latest"]
det_group = root[f"detect_runs/{run_name}"]

frame_idx = det_group["frame_indices"][:]
bboxes = det_group["bbox_norm_coords"][:]
scores = det_group["scores"][:] if "scores" in det_group else None
```

### Reconstructing per-frame lists

```python
num_frames = det_group.attrs.get("total_frames", det_group["frame_counts"].shape[0])
frame_to_slice = [[] for _ in range(num_frames)]
for det_id, frame in enumerate(frame_idx):
    frame_to_slice[frame].append(det_id)
```

### Filtering by confidence (YOLO)

```python
if scores is not None:
    keep = scores >= 0.4
    frame_idx = frame_idx[keep]
    bboxes = bboxes[keep]
    scores = scores[keep]
```

---

## 5. Downstream Stages

After detection you will typically see the following folders appear alongside `detect_runs/`:

| Group                  | Produced by                            | Purpose |
| ---------------------- | -------------------------------------- | ------- |
| `detect_quality_runs/` | `scripts/py -m fisheye.refinement.detect_quality` | Stores per-frame and per-detection quality labels. |
| `refined_detect_runs/` | `scripts/py -m fisheye.refinement.refine_detect` | Canonical curated detect runs with sparse `instances/` rows and `source_detections/` audit data. |
| `tracking_runs/`       | `scripts/py -m fisheye.tracking.arena_assignment` | Tracking outputs derived from arena assignment for the current single-subject-per-arena workflow. |

Each of these records the source detection run in their attributes (e.g. `source_detect_run`) so the lineage always points back to one of the `detect_runs/<run>` entries described above.

Current downstream tooling should treat `detect_runs/` as the raw detect
artifact layer. When a refined run exists, the canonical curated read surface
is `refined_detect_runs/<run>/instances`; legacy subgroup reads are
compatibility-only.

---

## 6. Quick Reference

| Field               | Traditional | YOLO | Notes |
| ------------------- | ----------- | ---- | ----- |
| `frame_indices`     | ✓           | ✓    | Detection-level |
| `bbox_norm_coords`  | ✓           | ✓    | `[cx, cy, w, h]` in ROI-relative units |
| `frame_counts`      | ✓           | ✓    | Per-frame occupancy |
| `n_detections`      | –           | ✓    | Alias of `frame_counts` |
| `scores`            | –           | ✓    | Confidence scores (float32) |
| `detection_method`  | `"blob"`    | `"yolo"` | Use to branch in tooling |
| `parameters`        | ✓           | ✓    | Contents differ by detector |
| `summary_statistics`| ✓           | ✓    | JSON-friendly summary dict |

Keeping the conventions aligned between the two pipelines means most analysis code can treat them interchangeably – just guard optional datasets and read the `detection_method` attribute when behaviour truly must diverge.

---

Happy detecting!
