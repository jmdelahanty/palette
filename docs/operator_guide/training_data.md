# Training Data

This guide covers how to create labeled training datasets from your analysis
Zarrs and how to train detection, pose, and segmentation models. It assumes
you have already run the [analysis pipeline](pipeline_workflow.md) and have
Zarrs with detections, keypoints, and/or masks.

## Overview

There are three model types you can train:

| Model | What it learns | Input data needed |
|-------|----------------|-------------------|
| **Detection** (YOLO) | Fish bounding boxes | Downsampled frames + detection labels |
| **Pose** (YOLO) | Anatomical keypoints | Cropped ROIs + keypoint annotations |
| **Eye masks** (U-Net) | Eye segmentation | Cropped ROIs + eye mask labels |

Each follows the same pattern:

```
analysis zarrs with labels
       |
       v
  1. review & approve labels (interactive)
       |
       v
  2. prepare training config + manifest
       |
       v
  3. (optional) merge multiple datasets into one
       |
       v
  4. train the model
       |
       v
  5. export (ONNX / TensorRT)
```

---

## Step 1: Review and approve labels

Before data enters a training set, it should be reviewed. Each label type
has its own review tool with an interactive UI.

### Detection review

```bash
scripts/py -m fisheye.tune.detect_review path/to/zarr/..._analysis.zarr
```

This is review and manual correction on the refined detect surface. Raw detect
artifact labeling already happened earlier via `detect_quality` plus
`refine_detect`; this review step decides whether the curated refined run is
acceptable for downstream use.

Modes:
- Default: browse detections and set approval status
- `--retune`: adjust blob detection parameters (traditional method only)
- `--manual`: draw or erase bounding box corrections

Keyboard shortcuts in the UI: **a** (approve), **n** (needs review),
**r** (reject), **p** (pending).

### Keypoint review

```bash
scripts/py -m fisheye.tune.keypoint_review path/to/zarr/..._analysis.zarr
```

Modes:
- Default: browse keypoints and set approval status
- `--retune`: recalculate usable keypoints mask
- `--manual`: draw corrections
- `--audit`: update summary statistics

### Eye mask review

```bash
scripts/py -m fisheye.tune.eye_mask_review path/to/zarr/..._analysis.zarr
```

Same modes as keypoint review. Tune threshold and morphology parameters,
then approve.

### Batch review

To generate a list of recordings that need review:

```bash
scripts/py -m fisheye.utils.generate_review_list /nvme1/recordings \
  --recursive --stage crop --review-state missing \
  --output crop_review_list.txt
```

Then review them:

```bash
scripts/py -m fisheye.utils.review_keypoints_batch /nvme1/recordings \
  --recursive --manual
```

---

## Step 2: Prepare training config and manifest

The prepare step scans your Zarrs, validates that labels exist and are
approved, and writes two files:

- A **config YAML** with training hyperparameters and dataset paths
- A **manifest JSON** listing every dataset, its source Zarr, and metadata

### Detection training prep

```bash
scripts/py -m fisheye.diagnostics.prepare_detect_training \
  /path/to/zarr1.zarr /path/to/zarr2.zarr \
  --source-type refined \
  --input-format gray \
  --set-name my_detect_set \
  --dry-run
```

Remove `--dry-run` to write the files. With `--set-name`, outputs go to
versioned paths:
- `runs/configs/detect/my_detect_set_v001.yaml`
- `runs/manifests/detect/my_detect_set_v001.manifest.json`

Or specify paths explicitly:

```bash
scripts/py -m fisheye.diagnostics.prepare_detect_training \
  /path/to/zarr1.zarr /path/to/zarr2.zarr \
  --source-type refined \
  --input-format gray \
  --out-config /path/to/train_detect.yaml \
  --out-manifest /path/to/train_detect.manifest.json
```

Key options:
- `--source-type`: which detection source family to use. `refined` is the
  current canonical curated surface; `filtered`, `interpolated`, and `manual`
  are legacy compatibility options.
- `--input-format`: frame format (`gray` or `rgb`)
- `--require-review-state` / `--require-review-intended-use`: gate on refined
  detect review metadata from the registry's refined detect review surface
- `--max-interpolated-detections-rate`: legacy compatibility gate for older
  refined archives that still carry interpolation-heavy outputs
- `--allow-unapproved`: skip review-state checks (not recommended for
  production training)

### Listing existing versions

```bash
scripts/py src/fisheye/utils/list_training_versions.py
scripts/py src/fisheye/utils/list_training_versions.py --name my_detect_set
```

---

## Step 3: Merge datasets (optional)

When training from multiple recordings, merge them into a single training Zarr
with train/val splits.

### Detection merge

```bash
scripts/py -m fisheye.utils.export_detect_training_zarr \
  --manifest /path/to/train_detect.manifest.json \
  --merge \
  --out-zarr /path/to/detect_merged.zarr \
  --split 0.8/0.2 \
  --seed 42
```

### Keypoint/pose merge

```bash
scripts/py -m fisheye.utils.export_keypoint_training_zarr \
  --manifest /path/to/pose.manifest.json \
  --merge \
  --out-zarr /path/to/pose_merged.zarr \
  --split 0.8/0.2 \
  --seed 42
```

### Eye mask merge

```bash
scripts/py -m fisheye.utils.export_eye_mask_training_zarr \
  /path/to/source.zarr \
  /path/to/eye_mask_merged.zarr \
  --label-mode lr \
  --input-format gray \
  --split-train 0.8 \
  --split-val 0.2 \
  --split-seed 42
```

`--label-mode` controls how eye masks are encoded:
- `lr`: separate left/right eye channels
- `union`: single combined channel

---

## Step 4: Train

All trainers take the config YAML from the prepare step.

### Detection

```bash
scripts/py -m fisheye.training.train_detection \
  /path/to/train_detect.yaml \
  --run-name my_detect_v1 \
  --no-log-registry
```

### Pose / keypoints

```bash
scripts/py -m fisheye.training.train_pose \
  /path/to/pose_config.yaml \
  --run-name my_pose_v1 \
  --no-log-registry
```

### Eye masks

```bash
scripts/py -m fisheye.training.train_eye_masks \
  /path/to/eye_mask_config.yaml \
  --run-name my_eye_masks_v1
```

All trainers:
- Use Ultralytics YOLO under the hood (detection and pose) or a custom U-Net
  (eye masks)
- Write weights, training reports, and metrics to the project/run directory
- Support `--export-onnx` and `--export-trt` to export immediately after
  training

The `--no-log-registry` flag skips writing the training run to the model
registry. Omit it if you have a registry set up and want to track runs there.

---

## Step 5: Export

After training, export the model for inference.

### ONNX + TensorRT (recommended)

```bash
scripts/py -m fisheye.training.export_detection \
  /path/to/runs/detect/my_detect_v1 \
  --export-trt
```

This:
1. Loads the best weights from the run directory
2. Exports to ONNX with embedded metadata (run ID, config hash, system info)
3. Builds a TensorRT engine optimized for your GPU

### ONNX only

```bash
scripts/py -m fisheye.training.export_onnx \
  /path/to/runs/detect/my_detect_v1/weights/best.pt
```

### ONNX to TensorRT (standalone)

If you already have an ONNX file:

```bash
scripts/py -m fisheye.training.onnx_to_tensorrt \
  /path/to/model.onnx \
  --output-dir /path/to/output
```

---

## Sampled imports for training frames

If you want to create training Zarrs from scratch (rather than using analysis
Zarrs from the pipeline), you can do a sampled import that grabs every Nth
frame:

```bash
scripts/py -m fisheye.capture.import_video /path/to/video.mp4 \
  --config configs/fisheye/default.yaml \
  --training-data \
  --frame-step 100 \
  --zarr-path /path/to/training_sample.zarr
```

`--training-data --frame-step 100` imports every 100th frame (both full
resolution and downsampled). This is useful for building compact training
datasets without importing entire videos.

### Batch sampled import

```bash
scripts/py src/fisheye/utils/import_recordings_training.py /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --dry-run
```

Remove `--dry-run` and add `--apply` to execute.

---

## Iteration workflow

Training is iterative. A typical cycle:

1. **Train** a detection model on your current labeled data.
2. **Run inference** with the new model on unlabeled recordings (see
   [pipeline workflow](pipeline_workflow.md)).
3. **Review** the new detections — approve good ones, correct bad ones.
4. **Add** the newly reviewed data to your next training set.
5. **Retrain** with the expanded dataset.

Each iteration improves model performance on your specific rig, dish design,
and fish species.

---

## Advanced: model registry

If you have the Palette model registry set up, you can automate dataset
selection and model tracking. The registry-based prepare commands
(`prepare_detect_training_from_registry`,
`prepare_keypoint_training_from_registry`,
`prepare_eye_mask_training_from_registry`) query the registry to find approved
datasets matching your criteria (dish design, rig, review state) and build
configs automatically.

The registry also tracks training runs, model exports, and their provenance.
See `docs/training_data_workflow.md` for the full registry-based workflow.
