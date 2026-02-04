# Training Data Workflow (Sampled Imports + Detection Iteration)

This document describes a practical, repeatable workflow for building YOLO training data from large videos **without** full-resolution imports. It favors small, sampled Zarrs for training + QC and uses detection runs on raw videos for scaling.

## Overview
- **Raw MP4s remain the source of truth** for full-resolution data.
- **Sampled Zarr imports** provide a compact, repeatable training dataset with metadata.
- **Detection-only Zarrs** (via `detect_yolo`) avoid full imports and still retain provenance (`source_video_path`).
- **Refinement + QC** is a first-class step before training updates.
- **Iteration**: train → run detect on more videos → refine → curate → retrain.

## Recommended Workflow

### 1) Sampled import for training frames (full + downsampled)
Use a config with `import.resolutions: both` (e.g. `configs/fisheye/import_local.yaml` or `configs/fisheye/default.yaml`).

```bash
python -m fisheye.capture.import_video /path/to/video.mp4 \
  --config configs/fisheye/import_local.yaml \
  --training-data \
  --frame-step 100 \
  --zarr-path /path/to/output/training_sample.zarr
```

Notes:
- `--training-data` **requires** `--frame-step`.
- The import stores `raw_video/original_frame_indices`, mapping sampled frames back to original video indices.
- Keep the sampled Zarr in the recording’s `zarr/` folder or a dedicated training workspace.

### Batch import from `recordings/` layout (camera videos)
If recordings are organized as:

```
recordings/<session_uuid_protocol>/
  raw/<session>.h5
  cams/Cam<id>*.mp4
  zarr/
```

you can batch import sampled training Zarrs using:

```bash
python src/fisheye/utils/import_recordings_training.py /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --dry-run
```

Apply the imports:

```bash
python src/fisheye/utils/import_recordings_training.py /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --apply
```

Optional: rich-formatted dry-run output:

```bash
python src/fisheye/utils/import_recordings_training.py /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --dry-run \
  --rich
```

Defaults:
- Input video: `cams/*.mp4` (camera video).
- Output Zarr: `zarr/<h5_stem>.zarr`.

Optional registry registration:

```bash
python src/fisheye/utils/import_recordings_training.py /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --apply \
  --register
```

Optional: mirror stimulus H5 into the Zarr (when available):

```bash
python src/fisheye/utils/import_recordings_training.py /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --apply \
  --import-stimulus
```

By default, stimulus import is skipped if `analysis/stimulus_runs` already exists. Use `--stimulus-always` to force another run, or `--stimulus-run-name` + `--stimulus-overwrite` to control replacement.

### 2) Run YOLO detection directly on the raw video
This creates a **detection-only** Zarr with `source_video_path` metadata.

```bash
python -m fisheye.detection.detect_yolo /path/to/video.mp4 \
  --model /path/to/model.pt \
  --output /path/to/output/detect_runs.zarr
```

### 3) Refine + QC detections
```bash
python -m fisheye.refinement.refine_detect /path/to/output/detect_runs.zarr
python -m fisheye.tracking.assign_ids /path/to/output/detect_runs.zarr
python -m fisheye.visualization.detection_visualizer /path/to/output/detect_runs.zarr
```

### 4) Crop full-resolution ROIs for pose/segmentation
Cropping uses `raw_video` if present, otherwise `source_video_path` stored in metadata.

```bash
python -m fisheye.tracking.crop /path/to/output/detect_runs.zarr \
  --config configs/fisheye/default.yaml
```

### 5) Generate training config + manifest
Use downsampled frames for detection training and register the dataset in the registry if desired.

```bash
python -m fisheye.diagnostics.prepare_detect_training \
  /path/to/training_sample.zarr \
  --input-format gray \
  --source-type filtered \
  --out-config /path/to/out/train_detect.yaml \
  --out-manifest /path/to/out/train_manifest.json \
  --register
```

Versioned convention (auto paths):

```bash
python -m fisheye.diagnostics.prepare_detect_training \
  /path/to/training_sample.zarr \
  --input-format gray \
  --source-type filtered \
  --set-name detect_base \
  --register
```

This writes:
- `runs/configs/detect/detect_base_v###.yaml`
- `runs/manifests/detect/detect_base_v###.manifest.json`

List versions:

```bash
python src/fisheye/utils/list_training_versions.py
python src/fisheye/utils/list_training_versions.py --name detect_base
```

### 6) Train + iterate
- Train from the generated config + manifest.
- Use the new model to re-run detection on additional videos.
- Refine, QC, and add to the next training iteration.

## Operational Notes
- **Avoid full imports** unless you truly need all frames. Detection runs can stay lightweight.
- **Keep paths stable** on the cluster: `source_video_path` must be readable from compute nodes.
- **Metadata matters**: sampled imports are deterministic (every Nth frame), which makes QC reproducible.
- **Provenance**: use `--register` in `prepare_detect_training` to log datasets into the registry.
- **Registry wiring (current)**: import writes metadata into the Zarr, but does not register it.
  - Register explicitly with `--register` (batch import) or `python -m fisheye.registry.scan`.
  - Rich provenance requires stimulus metadata (see `import_stimulus_to_zarr`).

## When to Use Full Imports
- You need **all frames** for downstream analyses that cannot be reconstructed from detection outputs.
- You need **full-resolution, frame-by-frame** features that are not derivable from raw video + detections.

---
For broader registry/provenance context, see `docs/detection_training_plan.md`.
