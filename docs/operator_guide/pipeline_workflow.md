# Pipeline Workflow

This guide covers every step of the Palette analysis pipeline after your
recordings have been [organized](organize_recordings.md). Each step reads from
the analysis Zarr and appends its outputs there, so the Zarr is the single
source of truth for a recording's analysis state.

For the current source-of-truth contract across canonical, refined,
compatibility, and cache surfaces, see
[current_pipeline_contract.md](../current_pipeline_contract.md).

For cluster migration status and implementation gaps across detect, pose,
segmentation, and refinement stages, see
[cluster_pipeline_migration_checklist.md](../cluster_pipeline_migration_checklist.md).

Recommended acquisition-to-analysis order:

1. organize recordings into the standard recording directory layout
2. run the [video diagnostics preflight](video_diagnostics.md) against the
   organized recording to check `raw/`, `cams/`, and paired `Cam..._meta.csv`
   camera metadata files
3. run the [H5 diagnostics preflight](h5_diagnostics.md) against the organized
   recording to verify raw Citrus H5 importability and optional section health
4. if both preflights pass, import the recording into its analysis Zarr
5. create or verify the dish mask on the imported analysis Zarr
6. continue with detection and downstream analysis stages

If you run diagnostics through `organize_recordings --run-video-diagnostics`
and/or `--run-h5-diagnostics`, the organizer writes a `preflight` summary into
`recording_manifest.json`. The import entry points below now block only when
that stored manifest preflight is `fail`. They do not block on `warn`.

Use `--allow-preflight-failures` only when you explicitly want to override a
recorded failed preflight.

For repeatable operator smoke checks on real data, use the shared fixture
convention documented in [test_data.md](test_data.md) rather than writing test
artifacts back into `/nvme1/recordings`.

## Overview

```
organize recordings
       |
       v
recommended video diagnostics preflight
       |
       v
recommended H5 diagnostics preflight
       |
       v
  1. import (create analysis zarr, import metadata + stimulus)
       |
       v
  2. dish mask (create/verify arena geometry)
       |
       v
  3. detect (YOLO object detection on camera video)
       |
       v
  4. detect quality (grade detection run, flag artifacts)
       |
       v
  5. refine detect (filter artifacts, dish-mask gate, write sparse curated instances)
       |
       v
  6. crop (extract ROI patches from detections)
       |
       v
  7. keypoints (anatomical landmark detection on crops)
       |
       v
  8. refine keypoints (correct eye swaps, compute geometry)
       |
       v
  9. eye masks (segment eyes from crops)
       |
       v
  10. subject masks (full body segmentation)
       |
       v
  11. track kinematics (consolidate into per-track metrics)
       |
       v
  12. analysis (swim bouts, stimulus response, heatmaps)
```

Detection through refined keypoints can be run together via the batch pipeline
command when dish masks already exist or have been imported from acquisition
metadata. If masks still need manual tuning, run the import-only command first,
tune/verify masks, then start the detect/refine pipeline. Steps 9+ currently
run as separate commands.

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
what camera video it matched, whether the analysis zarr already exists, and
whether a recorded manifest preflight would block the run.

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
  --registry /nvme1/palette_registry.sqlite \
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
When you do run import as its own step, pass `--registry` so registry-backed
review tools can see the new analysis zarrs immediately:

```bash
scripts/py -m fisheye.utils.import_organized_recordings_analysis \
  --organize-log "$ORGANIZE_LOG" \
  --registry /nvme1/palette_registry.sqlite \
  --apply
```

**What it writes to the Zarr:**
- `raw_video` group with video dimensions, fps, codec, frame count
- `analysis/stimulus_runs/` with stimulus event tables (if H5 has stimulus data)

### 2. Dish mask

Create or verify the arena/dish mask before detection/refinement for production
runs. Raw detection can technically run without this metadata, but
`refine_detect` uses `analysis_metadata.attrs["dish_mask"]` to gate bbox centers
and preserve outside-dish candidates in `source_detections` with reason
`outside_dish_mask`. If you add the mask after refining detections, regenerate
detect-quality/refined-detect outputs for that run.

Interactive tuning:

```bash
scripts/py -m fisheye.tune.mask_tuner \
  path/to/zarr/..._analysis.zarr \
  --registry /nvme1/palette_registry.sqlite
```

For full-resolution tuning, add `--full`. If Orange/Citrus wrote a runtime dish
mask and Palette imported it into `analysis_metadata.attrs["dish_mask"]`,
operators should still visually verify it before starting detect/refine. See
[citrus_dish_mask_handoff.md](citrus_dish_mask_handoff.md).

**What it writes:** `analysis_metadata.attrs["dish_mask"]` plus mask tuning
metadata. With `--registry`, successful saves also mark `dish_mask` as `ok` in
`recording_step_status`; without it, registry maintenance can discover the Zarr
attribute later.

To iterate masks missing according to the registry:

```bash
scripts/py -m fisheye.utils.review_dish_masks /nvme1/recordings \
  --source registry \
  --only-missing \
  --registry /nvme1/palette_registry.sqlite
```

### 3. Detect

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

### 4. Detect quality

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

### 5. Refine detect

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
When a dish mask is present, refinement also marks outside-dish candidates as
`outside_dish_mask` while preserving them in `source_detections`.

**What it writes:** `refined_detect_runs/{run_id}/instances` for curated bbox
rows and `refined_detect_runs/{run_id}/source_detections` for candidate/audit
rows.

**Batch:**
```bash
scripts/py -m fisheye.utils.refine_detect_batch /nvme1/recordings --recursive --apply
```

### 6. Crop

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

**What it writes:** `crop_runs/{run_id}/`. Analysis archives may use
`crop_storage_mode=geometry_only`, while training/export artifacts should remain
materialized.

For local workflows that will immediately run keypoints or segmentation on
geometry-only crops, build the flat ROI cache in the same serial pass:

```bash
scripts/py -m fisheye.utils.crop_flat_roi_cache_batch \
  /nvme1/recordings \
  --recursive \
  --source-type refined \
  --selection-policy full_recording \
  --workflow-id local_crop_cache_YYYYMMDD \
  --cache-root /nvme1/palette_roi_cache \
  --cache-decode-backend pynvvc_luma \
  --roi-live-acceleration gpu \
  --apply
```

The cache is workflow-local scratch data consumed by downstream ROI-model
stages via `--roi-cache-manifest`; the durable stage remains `crop_runs`.

Example handoff to direct ROI-model commands:

```bash
ZARR=/path/to/recording_analysis.zarr
CACHE_MANIFEST=/nvme1/palette_roi_cache/local_crop_cache_YYYYMMDD/roi_cache/<cache>.flat_roi_cache.json

scripts/py -m fisheye.detection.detect_keypoints_yolo "$ZARR" \
  --model /path/to/keypoint_model.pt \
  --roi-cache-manifest "$CACHE_MANIFEST"

scripts/py -m fisheye.segmentation.infer_unet_subject_masks "$ZARR" \
  --checkpoint /path/to/subject_mask_model.pt \
  --roi-cache-manifest "$CACHE_MANIFEST"
```

The manifest is the API boundary. Downstream stages validate it against the
selected archive and crop run before memory-mapping the payload.

### 7. Keypoints

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

### 8. Refine keypoints

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

### 9. Eye masks

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

### 10. Subject masks

Segments subject-mask components. The current recommended full-component path is
the U-Net subject-mask model, which writes raw probability surfaces for body,
eyes union, and swim bladder into `subject_mask_runs/<run>`. SAM/traditional
paths remain useful for body-only or component-specific workflows.

```bash
# U-Net subject masks
scripts/py -m fisheye.segmentation.infer_unet_subject_masks \
  path/to/zarr/..._analysis.zarr \
  --resolve-model-from-registry \
  --registry /nvme1/palette_registry.sqlite \
  --model-coverage-class dense_all_components \
  --model-component-coverage-key body+eyes+swim_bladder \
  --model-label-schema-id subject_v1_union \
  --crop-run <crop_run> \
  --assignment-keypoint-group refined_keypoints_runs \
  --assignment-keypoint-run <refined_keypoints_run> \
  --device 0 \
  --batch-size 128 \
  --mask-probs-dtype uint8 \
  --mask-probs-chunk-rois 32

# SAM subject masks (batch)
scripts/py -m fisheye.utils.run_sam_subject_masks_batch \
  /nvme1/recordings \
  --recursive \
  --apply
```

For cluster SAM3 smokes, prefer the bsub wrapper and a bounded apply:

```bash
scripts/submit_sam_subject_masks_bsub.sh \
  --zarr /groups/.../recording_analysis.zarr \
  --crop-run <crop_run> \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run <refined_keypoints_run> \
  --output-run <planned_subject_mask_run> \
  --sam3-root /groups/johnson/johnsonlab/jeremy/gitrepos/sam3 \
  --checkpoint /groups/johnson/johnsonlab/jeremy/models/sam3/sam3.pt \
  --python-bin /groups/ahrens/home/delahantyj/miniforge3/envs/palette-sam3/bin/python \
  --apply \
  --apply-limit 16 \
  --profile-timings \
  --no-hf-download \
  --submit
```

The wrapper is dry-run by default and requires `--apply-limit` unless
`--allow-full-apply` is passed. SAM3 is currently an optional external checkout,
not a Palette submodule. For cluster jobs, prefer the `/groups` checkpoint plus
`--no-hf-download` so execution does not depend on Hugging Face auth/cache state
on compute nodes.

**Recommended U-Net mode:** probability-first raw output. By default the U-Net
path writes `mask_probs_roi`, component/row metadata, and metrics; it does not
materialize the dense thresholded `masks_roi` cache. It also uses async output
and no Rich progress repaint by default. Use `--write-masks-roi` for dense
binary compatibility output, `--no-async-output` for serial writer debugging,
and `--progress` for interactive terminal progress.

**What it writes:** `subject_mask_runs/{run_name}/` with multi-channel
probabilities, per-frame metrics, model/config/provenance metadata, and
assignment-keypoint lineage for later `eyes_union -> eye_left/eye_right`
finalization.

Finalize the raw probabilities into canonical refined component masks:

```bash
scripts/py -m fisheye.refinement.finalize_subject_masks \
  path/to/zarr/..._analysis.zarr \
  --source-run <subject_mask_run> \
  --refined-run <refined_subject_mask_run> \
  --chunk-size 64 \
  --metric-level cheap \
  --execution-backend dask_worker_chunks \
  --scheduler processes \
  --num-workers 48
```

This writes `refined_subject_masks_runs/<run>` with
`["subject_body", "eye_left", "eye_right", "swim_bladder"]` when the raw source
uses `subject_v1_union` and has usable assignment keypoint lineage. The fast
default writes canonical masks, cleanup metrics, reasons, review triage, and
provenance; expensive shape-QC metrics and eye geometry can be added
explicitly. Dense per-component `source_seed_masks_roi` arrays are diagnostic
intermediates and are omitted by default. Add `--retain-source-seeds` for
troubleshooting runs where comparing seed masks against finalized masks is
worth the extra write/storage cost.

Add or refresh refined-subject eye geometry after the fast finalizer:

```bash
scripts/py -m fisheye.utils.backfill_refined_subject_eye_geometry \
  path/to/zarr/..._analysis.zarr \
  --zarr-use analysis \
  --apply
```

Use `--metric-level full --write-eye-geometry` on the finalizer only when the
operator intentionally wants expensive shape-QC and eye-ellipse relation writing
folded into the same run creation command.

For large subject-mask runs, keep the row finalizer on `--execution-backend
process_shards` and use sharded postcompute for expensive derived artifacts:

```bash
scripts/py -m fisheye.refinement.finalize_subject_masks \
  path/to/zarr/..._analysis.zarr \
  --source-run <subject_mask_run> \
  --refined-run <refined_subject_mask_run> \
  --chunk-size 256 \
  --metric-level cheap \
  --execution-backend process_shards \
  --num-workers 8 \
  --write-eye-geometry \
  --write-component-contours \
  --postcompute-backend process_shards \
  --postcompute-chunk-size 256 \
  --postcompute-num-workers 8
```

`--postcompute-backend process_shards` affects only eye geometry and body/swim
contour materialization. Canonical masks and metrics are still produced by the
main finalizer backend. The subject-mask batch workflow defaults this expensive
postcompute step to `process_shards`; the lower-level finalizer CLI keeps
`serial` available for historical-path debugging.

Generated refined runs are candidates until reviewed. Do not treat a
smart-finalized run as training-approved solely because all components are
available; check component review state and reason/triage counts first.

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

### 11. Track kinematics

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

### 12. Analysis

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
