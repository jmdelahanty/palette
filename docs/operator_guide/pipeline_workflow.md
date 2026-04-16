# Pipeline Workflow

This guide covers every step of the Palette analysis pipeline after your
recordings have been [organized](organize_recordings.md). Each step reads from
the analysis Zarr and appends its outputs there, so the Zarr is the single
source of truth for a recording's analysis state.

Recommended pre-import order:

1. organize recordings into the standard recording directory layout
2. run the [video diagnostics preflight](video_diagnostics.md) against the
   organized recording to check `raw/`, `cams/`, and paired `Cam..._meta.csv`
   camera metadata files
3. if `Media` passes, import the recording into its analysis Zarr
4. continue with detection and downstream analysis stages

## Overview

```
organize recordings
       |
       v
recommended video diagnostics preflight
       |
       v
  1. import (create analysis zarr, import metadata + stimulus)
       |
       v
  2. detect (YOLO object detection on camera video)
       |
       v
  3. detect quality (grade detection run, flag artifacts)
       |
       v
  4. refine detect (filter artifacts, write sparse curated instances)
       |
       v
  5. crop (extract ROI patches from detections)
       |
       v
  6. keypoints (anatomical landmark detection on crops)
       |
       v
  7. refine keypoints (correct eye swaps, compute geometry)
       |
       v
  8. eye masks (segment eyes from crops)
       |
       v
  9. subject masks (full body segmentation)
       |
       v
  10. track kinematics (consolidate into per-track metrics)
       |
       v
  11. analysis (swim bouts, stimulus response, heatmaps)
```

Steps 1-7 can be run together via the batch pipeline command. Steps 8+
currently run as separate commands.

---

## Running the full pipeline (batch)

The most common way to process recordings is the batch pipeline, which handles
steps 1 through 7 for every recording under your recordings root. You just
need a YOLO model file (`.pt`) — ask Jeremy for the current best model if you
don't have one.

### Dry-run first

```bash
scripts/py -m fisheye.utils.import_recordings_analysis \
  --recursive \
  --model /path/to/best.pt \
  --conf 0.25 \
  --iou 0.7 \
  --refine-detect \
  --keypoints \
  --refine-keypoints
```

With no `--apply` flag, this prints a plan showing each recording it found,
what camera video it matched, and whether the analysis zarr already exists.

### Apply

```bash
scripts/py -m fisheye.utils.import_recordings_analysis \
  --recursive \
  --model /path/to/best.pt \
  --conf 0.25 \
  --iou 0.7 \
  --refine-detect \
  --keypoints \
  --refine-keypoints \
  --apply
```

This processes every recording that doesn't already have an analysis zarr.
To reprocess existing recordings, add `--overwrite`.

### For a single recording

```bash
scripts/py -m fisheye.utils.run_recording_analysis_pipeline \
  --recording-dir "$PALETTE_RECORDINGS_ROOT/2026-01-28T19-36-18Z_arena_1_Feeding" \
  --model /path/to/best.pt \
  --conf 0.25 \
  --iou 0.7 \
  --refine-detect \
  --keypoints \
  --refine-keypoints \
  --apply
```

### Advanced: using the model registry

If you have access to the Palette model registry, you can let the pipeline
automatically select the best model for each recording instead of specifying
one explicitly. This is optional — most users should use an explicit model
path as shown above.

```bash
scripts/py -m fisheye.utils.import_recordings_analysis \
  --recursive \
  --model-source registry \
  --refine-detect \
  --keypoints \
  --refine-keypoints \
  --register \
  --apply
```

---

## Step-by-step reference

Use these when you need to run or re-run an individual stage.

### 1. Import

Creates the analysis Zarr archive and imports video metadata and stimulus
events from the H5 file. This is handled automatically by the batch pipeline
and the single-recording pipeline — you rarely need to run it standalone.

**What it writes to the Zarr:**
- `raw_video` group with video dimensions, fps, codec, frame count
- `analysis/stimulus_runs/` with stimulus event tables (if H5 has stimulus data)

### 2. Detect

Runs YOLO inference on the camera video to find fish in every frame.

```bash
scripts/py -m fisheye.detection.detect_yolo \
  path/to/cams/Cam2010093_2026-01-28T19-36-18Z_arena_1.mp4 \
  --output path/to/zarr/2026-01-28T19-36-18Z_arena_1_Feeding_analysis.zarr \
  --model /path/to/best.pt \
  --conf 0.25 \
  --iou 0.7 \
  --batch-size 32 \
  --write-raw-video-metadata
```

**What it writes:** `detect_runs/{run_id}/` with bounding boxes, scores, class
labels, and coverage statistics.

### 3. Detect quality

Labels raw detection artifacts (jumps, blips, multi-detection) on a detect run.

```bash
scripts/py -m fisheye.refinement.detect_quality \
  path/to/zarr/2026-01-28T19-36-18Z_arena_1_Feeding_analysis.zarr
```

Automatically uses the latest detect run. To target a specific run:

```bash
scripts/py -m fisheye.refinement.detect_quality \
  path/to/zarr/..._analysis.zarr \
  --run detect_run_id
```

**What it writes:** `detect_runs/{run_id}/quality_reports/{quality_run_id}/` with
artifact labels, quality summary metadata, and threshold provenance.

**Batch:**
```bash
scripts/py -m fisheye.utils.detect_quality_batch /nvme1/recordings --recursive --apply
```

### 4. Refine detect

Consumes raw detect artifact labels, filters bad candidates, and writes the
canonical sparse curated detect surface.

```bash
scripts/py -m fisheye.refinement.refine_detect \
  path/to/zarr/..._analysis.zarr \
  --config configs/fisheye/default.yaml
```

The current sparse-first workflow filters flagged raw detections and writes
`refined_detect_runs/{run_id}/instances` plus
`refined_detect_runs/{run_id}/source_detections`. Interpolation metadata is
retained only for legacy compatibility.

**What it writes:** `refined_detect_runs/{run_id}/instances` for curated bbox
rows and `refined_detect_runs/{run_id}/source_detections` for candidate/audit
rows.

**Batch:**
```bash
scripts/py -m fisheye.utils.refine_detect_batch /nvme1/recordings --recursive --apply
```

### 5. Crop

Extracts ROI image patches around each detection for downstream pose and
segmentation models.

```bash
scripts/py -m fisheye.utils.crop_batch \
  path/to/zarr/..._analysis.zarr \
  --apply
```

Or batch across all recordings:

```bash
scripts/py -m fisheye.utils.crop_batch \
  /nvme1/recordings \
  --recursive \
  --apply
```

The default ROI size is 512x512 (set in `configs/fisheye/default.yaml`). The
crop source defaults to the canonical current refined surface
`refined_detect_runs/{run_id}/instances`, with legacy fallback only for older
archives.

**What it writes:** `crop_runs/{run_id}/` with ROI image arrays.

### 6. Keypoints

Detects anatomical landmarks (eyes, body points) on each crop.

```bash
scripts/py -m fisheye.utils.run_keypoints_batch \
  path/to/zarr/..._analysis.zarr \
  --config configs/fisheye/default.yaml \
  --apply
```

The default config uses the `traditional` (geometry-based) method. The
`yolo` method is also available via the config.

**What it writes:** `keypoints_runs/{run_id}/` with landmark coordinates and
confidence scores.

### 7. Refine keypoints

Corrects left/right eye swaps and computes diagnostic geometry metrics.

```bash
scripts/py -m fisheye.refinement.refine_keypoints \
  path/to/zarr/..._analysis.zarr \
  --config configs/fisheye/default.yaml
```

**What it writes:** `refined_keypoints_runs/{run_id}/` with swap-corrected
coordinates and geometry metrics.

**Batch:**
```bash
scripts/py -m fisheye.utils.refine_keypoints_batch /nvme1/recordings --recursive --apply
```

### 8. Eye masks

Segments eye regions from each crop. Three methods are available: U-Net
(primary), traditional (color-based), and YOLO.

```bash
# U-Net (recommended)
scripts/py -m fisheye.segmentation.infer_unet_eye_masks \
  path/to/zarr/..._analysis.zarr \
  /path/to/eye_mask_checkpoint.pt \
  --batch-size 256

# Batch across recordings
scripts/py -m fisheye.utils.run_eye_masks_batch \
  /nvme1/recordings \
  --recursive \
  --method unet \
  --model /path/to/eye_mask_checkpoint.pt \
  --apply
```

**What it writes:** `eye_masks_runs/{run_name}/` with binary masks,
probability maps, and per-frame metrics (area, centroid, bounding box).

#### Refine eye masks

```bash
scripts/py -m fisheye.refinement.refine_eye_masks \
  path/to/zarr/..._analysis.zarr
```

### 9. Subject masks

Segments the full fish body. Two approaches: U-Net (produces a 3-channel mask
for body, eyes, and swim bladder) or SAM (Segment Anything Model).

```bash
# U-Net subject masks
scripts/py -m fisheye.segmentation.infer_unet_subject_masks \
  path/to/zarr/..._analysis.zarr \
  /path/to/subject_mask_checkpoint.pt \
  --batch-size 256

# SAM subject masks (batch)
scripts/py -m fisheye.utils.run_sam_subject_masks_batch \
  /nvme1/recordings \
  --recursive \
  --apply
```

**What it writes:** `subject_mask_runs/{run_name}/` with multi-channel masks
and per-frame metrics.

#### Swim bladder segmentation

Specialized segmentation for the swim bladder using polar boundary detection.
Runs after subject masks are available.

```bash
scripts/py -m fisheye.segmentation.swim_bladder_segmentation \
  path/to/zarr/..._analysis.zarr

# Batch
scripts/py -m fisheye.utils.run_swim_bladder_segmentation_batch \
  /nvme1/recordings \
  --recursive \
  --apply
```

### 10. Track kinematics

Consolidates detections, keypoints, and arena assignments into unified
per-track kinematic metrics (position, heading, speed, acceleration).

```bash
scripts/py -m fisheye.analysis.track_kinematics \
  path/to/zarr/..._analysis.zarr \
  --smooth-seconds 1.0
```

**What it writes:** `analysis/track_kinematics_runs/{run_name}/` with
per-track arrays for position (px and mm), heading, angular velocity, speed
(raw/filtered/smoothed), and acceleration.

### 11. Analysis

These steps produce final behavioral metrics. Each requires track kinematics
to exist.

#### Swim bout statistics

```bash
scripts/py -m fisheye.analysis.swim_bout_statistics \
  path/to/zarr/..._analysis.zarr
```

**What it writes:** Per-bout metrics (duration, distance, mean/peak speed) and
inter-bout interval analysis.

#### Stimulus response

Computes per-trial behavioral metrics aligned to stimulus events. Requires
stimulus data to have been imported during step 1.

```bash
scripts/py -m fisheye.analysis.stimulus_response \
  path/to/zarr/..._analysis.zarr
```

**What it writes:** Per-fish, per-bout, and per-frame metrics including
heading alignment and grating-specific measures.

#### Training heatmaps

Generates spatial distribution plots across stimulus phases
(pre/training/post).

```bash
scripts/py -m fisheye.analysis.plot_training_heatmaps_zarr \
  path/to/zarr/..._analysis.zarr \
  --output-dir /path/to/figures/
```

---

## Configuration

Most steps read defaults from `configs/fisheye/default.yaml`. Key sections:

| Section | Controls |
|---------|----------|
| `import` | Frame chunk size, sharding, downsampling resolution |
| `detect` | Thresholds, dish mask geometry |
| `crop` | ROI size (default 512x512), scheduler, source type |
| `keypoints` | Method (traditional/yolo), geometry filters |
| `refine_detect` | Raw artifact filtering, sparse curated detect writes |
| `refine_eye_masks` | Area filtering, chunk size |
| `refine_subject_masks` | Component selection, scheduler |

Override any config with `--config path/to/custom.yaml` on the relevant
command.

## Logs

The batch pipeline writes JSONL logs to
`$PALETTE_LOG_ROOT/import_recordings_analysis/` (or
`$PALETTE_RECORDINGS_ROOT/logs/import_recordings_analysis/`). Each log line
includes the recording, step, return code, and command that was run.

Disable logging with `--no-log`.

## What the Zarr looks like after a full run

```
2026-01-28T19-36-18Z_arena_1_Feeding_analysis.zarr/
  raw_video/                          (video metadata)
  analysis/
    stimulus_runs/{run}/events        (stimulus event tables)
    track_kinematics_runs/{run}/      (per-track metrics)
  detect_runs/{run}/                  (raw bounding boxes)
  detect_runs/{run}/quality_reports/{quality_run}/
                                      (raw detect artifact labels)
  refined_detect_runs/{run}/instances (canonical curated detections)
  refined_detect_runs/{run}/source_detections
                                      (candidate/audit detections)
  crop_runs/{run}/roi_images          (ROI patches)
  keypoints_runs/{run}/               (landmarks)
  refined_keypoints_runs/{run}/       (swap-corrected landmarks)
  eye_masks_runs/{run}/               (eye segmentation)
  subject_mask_runs/{run}/            (body segmentation)
```

Every run group includes provenance metadata tracing back to its source runs,
model versions, and configuration.
