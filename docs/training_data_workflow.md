# Training Data Workflow (Sampled Imports + Detection Iteration)

This document describes a practical, repeatable workflow for building YOLO training data from large videos **without** full-resolution imports. It favors small, sampled Zarrs for training + QC and uses detection runs on raw videos for scaling.

## Overview
- **Raw MP4s remain the source of truth** for full-resolution data.
- **Sampled Zarr imports** provide a compact, repeatable training dataset with metadata.
- **Detection-only Zarrs** (via `detect_yolo`) avoid full imports and still retain provenance (`source_video_path`).
- **Refinement + QC** is a first-class step before training updates.
- **Iteration**: train → run detect on more videos → refine → curate → retrain.

## Recording Analysis Stage Contract

For recording analysis archives, the canonical stage order is:

1. import/create analysis archive + metadata/stimulus
2. detect
3. refine (optional)
4. registry rescan (optional)

Single recording wrapper:

```bash
scripts/py -m fisheye.utils.run_recording_analysis_pipeline \
  --recording-dir "$REC" \
  --model-source registry \
  --registry /nvme1/palette_registry.sqlite \
  --apply
```

Batch wrapper:

```bash
scripts/py -m fisheye.utils.import_recordings_analysis /nvme1/recordings \
  --recursive \
  --model-source registry \
  --registry /nvme1/palette_registry.sqlite \
  --apply
```

Contract reference: `docs/recording_analysis_pipeline_contract.md`.

## Recommended Workflow

### 1) Sampled import for training frames (full + downsampled)
Use the batch wrapper for new sampled training Zarrs. Its default backend is
`--decode-backend pynvvc-luma`, which decodes with PyNvVideoCodec, stores raw
NV12 Y/luma frames as `uint8`, writes `raw_video/original_frame_indices`, and
stamps the `orange_mono_pynvvc_luma_uint8_v1` pixel contract.

The direct `capture.import_video --training-data` path is retained as a legacy
Decord-derived path, not the preferred path for new long-lived training assets.

```bash
scripts/py -m fisheye.utils.import_recordings_training /path/to/recordings \
  --target-sampled-frames 200 \
  --dry-run
```

Notes:
- `--target-sampled-frames` resolves a per-recording `frame_step` from source
  frame-count metadata.
- The import stores `raw_video/original_frame_indices`, mapping sampled frames back to original video indices.
- Keep the sampled Zarr in the recording’s `zarr/` folder or a dedicated training workspace.
- PyNvVC sampled imports are GPU-only and fail before creating the final Zarr if
  CUDA/PyNvVideoCodec is unavailable.
- The older Decord path is still available as
  `--decode-backend legacy-decord --allow-legacy-decode-contract` for explicit
  legacy backfills only.

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
scripts/py -m fisheye.utils.import_recordings_training /nvme1/recordings \
  --target-sampled-frames 200 \
  --dry-run
```

Apply the imports:

```bash
scripts/py -m fisheye.utils.import_recordings_training /nvme1/recordings \
  --target-sampled-frames 200 \
  --register \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --apply
```

`--target-sampled-frames` resolves a per-recording frame step from the source
frame count, using `recording_manifest.json` when available and external
recorder summary sidecars as a legacy fallback. For example, a 140k-frame
recording with `--target-sampled-frames 200 --skip-tail-frames 200` resolves to
roughly `--frame-step 699`.
Use `--recursive` only when recordings are nested deeper than
`<recordings_root>/<recording>/raw/*.h5`; broad recursive scans over PRFS can be
slow.
Use `--limit 1` with `--apply` for the first smoke before running a whole batch.

Optional: rich-formatted dry-run output:

```bash
scripts/py -m fisheye.utils.import_recordings_training /nvme1/recordings \
  --target-sampled-frames 200 \
  --dry-run \
  --rich
```

Defaults:
- Input video: `cams/*.mp4` (camera video).
- Output Zarr: `zarr/<recording>_training.zarr`.

Optional registry registration:

```bash
scripts/py -m fisheye.utils.import_recordings_training /nvme1/recordings \
  --target-sampled-frames 200 \
  --register \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --apply
```

Optional: mirror stimulus H5 into the Zarr (when available):

```bash
scripts/py -m fisheye.utils.import_recordings_training /nvme1/recordings \
  --target-sampled-frames 200 \
  --import-stimulus \
  --stimulus-quiet \
  --register \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --apply
```

By default, stimulus import is skipped if `analysis/stimulus_runs` already exists. Use `--stimulus-always` to force another run, or `--stimulus-run-name` + `--stimulus-overwrite` to control replacement.

Current GoodCopBadCop smoke:

```bash
scripts/py -m fisheye.utils.import_recordings_training /groups/johnson/johnsonlab/jeremy/recordings \
  --path-contains GoodCopBadCop \
  --target-sampled-frames 200 \
  --limit 1 \
  --import-stimulus \
  --stimulus-quiet \
  --register \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --apply
```

If the smoke output and registry row look correct, remove `--limit 1` to build
the full current GoodCopBadCop set. The June 2026 GoodCopBadCop recordings
resolve to roughly `frame_step=697..700`; the older May 29 batch resolves from
external-recorder summaries to `frame_step=717`.

Cluster PyNvVC import:

```bash
scripts/submit_import_recordings_training_bsub.sh \
  --root /groups/johnson/johnsonlab/jeremy/recordings \
  --path-contains GoodCopBadCop \
  --target-sampled-frames 200 \
  --limit 1
```

The submitter is dry-run by default. Inspect the generated run directory, then
add `--submit` on an LSF login node. It submits one GPU array task per recording,
uses `--decode-backend pynvvc-luma`, imports stimulus, and registers outputs in
the PRFS registry by default.

### Acquisition crop-video pose training
External-IPC recordings can include acquisition-time crop videos under
`derived/external_crop_recorder/`. For pose training that should match Orange's
runtime crop stream, append sampled crop-video frames into the same
`<recording>_training.zarr` that stores sampled full-frame detector-training
frames:

```bash
scripts/py -m fisheye.utils.import_recordings_training /groups/johnson/johnsonlab/jeremy/recordings \
  --path-contains RedScare \
  --target-sampled-frames 200 \
  --include-acquisition-crop-video \
  --acquisition-crop-run-prefix crop_red_scare_acquisition_crop_video_training \
  --import-stimulus \
  --register \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --apply
```

This creates:

- `raw_video/images_full` and `raw_video/images_ds` from the full camera video
  for detector training.
- `crop_runs/<crop_red_scare...>/roi_images` from the acquisition crop video for
  pose-labeling/training.
- crop lineage arrays including `source_crop_xywh`,
  `source_crop_video_frame_indices`, `source_crop_local_frame_ids`,
  `source_training_row_indices`, and `source_recording_frame_ids`.
- only crop-video rows with `has_detection=true` and `blank_frame=false`; use
  `source_training_row_indices` because crop rows can be fewer than full-frame
  sampled rows.
- crop QC boxes (`bbox_roi_xyxy`, `bbox_norm_coords`) from realtime crop-meta
  detection geometry.

The crop-video pixel contract is currently `orange_mono_pynvvc_luma_uint8_v1`
with `frame_format_confirmation_status=pending_orange_confirmation`; Orange
still needs to confirm the exact frame representation handed to the crop-video
encoder and future pose model.

Cluster submitter:

```bash
scripts/submit_import_recordings_training_bsub.sh \
  --path-contains RedScare \
  --target-sampled-frames 200 \
  --include-acquisition-crop-video \
  --acquisition-crop-run-prefix crop_red_scare_acquisition_crop_video_training \
  --max-active 4 \
  --submit
```

Contract reference: `docs/acquisition_crop_pose_training_workflow.md`.

### 2) Run YOLO detection directly on the raw video
This creates a **detection-only** Zarr with `source_video_path` metadata.

```bash
scripts/py -m fisheye.detection.detect_yolo /path/to/video.mp4 \
  --model /path/to/model.pt \
  --resize-dims 768 1280 \
  --output /path/to/output/detect_runs.zarr
```

Notes:
- `--resize-dims` is the canonical inference-size knob (`[height width]`).
- `--imgsz` is still accepted as a legacy alias and normalizes into `resize_dims`.

### 3) Refine + QC detections
```bash
scripts/py -m fisheye.refinement.refine_detect /path/to/output/detect_runs.zarr
scripts/py -m fisheye.tune.detect_review /path/to/output/detect_runs.zarr
scripts/py -m fisheye.tracking.arena_assignment /path/to/output/detect_runs.zarr
scripts/py -m fisheye.visualization.detection_visualizer /path/to/output/detect_runs.zarr
```

For sampled training Zarrs, approved detection review writes the detection data
profile. Add the image-domain profile when comparing new recordings against the
existing training pool:

```bash
scripts/py -m fisheye.utils.training_image_profile \
  /path/to/training_sample.zarr \
  --apply \
  --sync-registry \
  --registry /nvme1/palette_registry.sqlite
```

This captures brightness, contrast, sharpness, clipping, illumination, and
fish/background contrast metrics in
`analysis/training_image_profile_runs/<run>`.

### 4) Crop full-resolution ROIs for pose/segmentation
Cropping uses `raw_video` if present, otherwise `source_video_path` stored in metadata.

```bash
scripts/py -m fisheye.tracking.crop /path/to/output/detect_runs.zarr \
  --config configs/fisheye/default.yaml
```

### 5) Generate training config + manifest
Use downsampled frames for detection training and register the dataset in the registry if desired.

```bash
scripts/py -m fisheye.diagnostics.prepare_detect_training \
  /path/to/training_sample.zarr \
  --input-format gray \
  --source-type refined \
  --out-config /path/to/out/train_detect.yaml \
  --out-manifest /path/to/out/train_manifest.json \
  --register
```

Versioned convention (auto paths):

```bash
scripts/py -m fisheye.diagnostics.prepare_detect_training \
  /path/to/training_sample.zarr \
  --input-format gray \
  --source-type refined \
  --set-name detect_base \
  --register
```

This writes:
- `runs/configs/detect/detect_base_v###.yaml`
- `runs/manifests/detect/detect_base_v###.manifest.json`

List versions:

```bash
scripts/py -m fisheye.utils.list_training_versions
scripts/py -m fisheye.utils.list_training_versions --name detect_base
```

### 6) Train + iterate
- Train from the generated config + manifest.
- Use the new model to re-run detection on additional videos.
- Refine, QC, and add to the next training iteration.

For production training, treat the config and manifest as the immutable inputs
to the run. Pass both to the trainer explicitly, keep registry logging enabled,
and write outputs to a stable project directory:

```bash
scripts/py -m fisheye.training.train_detection \
  /nvme1/training/datasets/<set_id>/<set_id>.yaml \
  --manifest /nvme1/training/datasets/<set_id>/<set_id>.manifest.json \
  --set-id <set_id> \
  --registry /nvme1/palette_registry.sqlite \
  --project /nvme1/models/detect/<set_id> \
  --run-name <set_id>_yolo11n_trt_YYYYMMDD \
  --export-trt \
  --trt-precision fp16 \
  --trtexec /usr/local/TensorRT-10.0.1.6/bin/trtexec \
  --trt-profiling
```

Run long GPU jobs under `tmux`, `screen`, or a cluster scheduler. Do not rely
on a foreground Codex/tool session to keep a multi-hour training job alive:

```bash
tmux new-session -d -s palette_detect_train '
cd /home/delahantyj@hhmi.org/gitrepos/palette &&
env MPLCONFIGDIR=/tmp/matplotlib-training \
    ULTRALYTICS_CONFIG_DIR=/tmp/ultralytics-training \
  scripts/py -m fisheye.training.train_detection ... \
    > /tmp/palette_detect_train.log 2>&1
'
tail -f /tmp/palette_detect_train.log
```

The detection trainer snapshots the run inputs under `<run>/inputs`, writes a
training report YAML, and records registry rows for the training run plus model
exports unless `--no-log-registry` is passed. With `--export-trt`, ONNX export
is implied and TensorRT export runs after training completes or early-stops.

## Operational Notes
- **Avoid full imports** unless you truly need all frames. Detection runs can stay lightweight.
- **Keep paths stable** on the cluster: `source_video_path` must be readable from compute nodes.
- **Metadata matters**: sampled imports are deterministic (every Nth frame), which makes QC reproducible.
- **Provenance**: use `--register` in `prepare_detect_training` to log datasets into the registry.
- **Registry wiring (current)**: import writes metadata into the Zarr, but does not register it.
  - Register explicitly with `--register` (batch import) or `scripts/py -m fisheye.registry.scan`.
  - Rich provenance requires stimulus metadata (see `import_stimulus_to_zarr`).

## Registry Hygiene (Before Training)

Run this before kicking off new training jobs:

```bash
scripts/py -m fisheye.registry.maintenance --registry /nvme1/palette_registry.sqlite --reconcile-registry
```

Inspect current registry state:

```bash
scripts/py -m fisheye.utils.check_training_registry --registry /nvme1/palette_registry.sqlite --all --limit 100
```

Dry-run safety check:

```bash
scripts/py -m fisheye.registry.maintenance --registry /nvme1/palette_registry.sqlite --reconcile-registry --dry-run
```

## When to Use Full Imports
- You need **all frames** for downstream analyses that cannot be reconstructed from detection outputs.
- You need **full-resolution, frame-by-frame** features that are not derivable from raw video + detections.

---
For broader registry/provenance context, see `docs/detection_training_plan.md`.
For merged-export schema/CLI constraints, see `docs/detection_merged_export_contract.md`.

## One-Command Build (Detect: Registry -> Preflight -> Merged Zarr)

Use the pipeline wrapper to run selection + preflight + merged export in one invocation:

```bash
scripts/py -m fisheye.utils.run_detect_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --source-type refined \
  --input-format gray \
  --model-input gray \
  --out-config /tmp/detect_build.yaml \
  --out-manifest /tmp/detect_build.manifest.json \
  --build-dataset \
  --merge-out-zarr /nvme1/datasets/detect/detect_build_merged.zarr \
  --merge-out-dir /nvme1/datasets/detect/detect_build \
  --merge-split 0.8/0.2 \
  --merge-seed 42 \
  --merge-overwrite
```

Notes:
- `--build-dataset` is a convenience alias for `--export-merged` + `--aggregate-training-data-card`.
- `--export-merged` requires `--out-manifest` (the export step consumes that manifest).
- `--export-merged` cannot be combined with `--dry-run` because preflight dry-run does not write files.
- If the preflight source manifest would collide with the merged manifest path,
  the wrapper preserves the source-selection manifest as
  `<set_id>.source.manifest.json`. The merged Zarr root provenance points at
  that immutable source manifest and stores its SHA-256 hash.
- `fisheye.utils.prepare_detect_training_from_registry` is prepare-only and no longer launches merge/train.
- Registry-selected detection exports gate on `refined_detect_review_current`,
  which is the semantic current reviewed-refined surface. The current view
  prefers explicitly reviewed rows over newer unreviewed refined attempts, so a
  cleanup/refinement rerun does not silently displace an approved training
  surface.
- Registry-selected detection exports deduplicate source Zarrs by physical path
  before calling the lower-level preparer. This avoids double-counting labels
  when historical rescans have both base and path-hash dataset IDs for the same
  Zarr.
- Sampled recording-only training Zarrs can be exported with `--source-type
  refined` directly from `refined_detect_runs/<run>/instances`; they do not need
  `crop_runs` or crop-review approval.
- `--aggregate-training-data-card` uses the preflight manifest dataset list and
  `detection_data_profile_latest` rows to write `<set_id>.data_card.json`.
- Data-card aggregation now also writes plot PNGs by default to
  `<set_id>.data_card.plots/` (histograms + center heatmap). Use
  `--data-card-no-plots` in pipeline mode or `--no-plots` in standalone mode to disable.
- Adjust center-heatmap coarsening with `--data-card-plot-heatmap-bin-factor` (pipeline)
  or `--plot-heatmap-bin-factor` (standalone aggregation/plot tool).
- For a full "where artifacts live" matrix (filesystem vs zarr visualizations),
  see `docs/artifact_storage_map.md`.

Standalone data-card aggregation:

```bash
scripts/py -m fisheye.utils.aggregate_detection_training_data_card \
  --manifest /nvme1/training/datasets/<set_id>/<set_id>.manifest.json \
  --registry /nvme1/palette_registry.sqlite
```

Standalone plot rendering from an existing data-card JSON:

```bash
scripts/py -m fisheye.utils.plot_detection_training_data_card \
  --card /nvme1/training/datasets/<set_id>/<set_id>.data_card.json
```

Default center-heatmap plotting uses coarser bins (`--heatmap-bin-factor 2`) for readability.
Use `--heatmap-bin-factor 1` for full-resolution bins.

Open existing plots (or generate missing ones, then open):

```bash
scripts/py -m fisheye.utils.plot_detection_training_data_card \
  --card /nvme1/training/datasets/<set_id>/<set_id>.data_card.json \
  --view
```

Regenerate all plot files and then open them:

```bash
scripts/py -m fisheye.utils.plot_detection_training_data_card \
  --card /nvme1/training/datasets/<set_id>/<set_id>.data_card.json \
  --view \
  --force
```

## One-Command Build (Pose: Registry -> Preflight -> Optional Train)

Use the keypoint pipeline wrapper to run selection + preflight and optionally launch training:

```bash
scripts/py -m fisheye.utils.run_keypoint_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --source-type filtered \
  --input-format gray \
  --model-input gray \
  --keypoint-run latest_traditional \
  --set-name cedar_shadow_pose \
  --export-merged \
  --merge-split 0.8/0.2 \
  --merge-seed 42 \
  --merge-overwrite \
  --register \
  --train
```

Include keypoint data-card aggregation + plot controls in pipeline mode:

```bash
scripts/py -m fisheye.utils.run_keypoint_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --source-type filtered \
  --input-format gray \
  --model-input gray \
  --keypoint-run latest_traditional \
  --set-name cedar_shadow_pose \
  --export-merged \
  --data-card-output /nvme1/training/datasets/pose/cedar_shadow_pose.data_card.json \
  --data-card-split train \
  --data-card-plot-dir /nvme1/training/datasets/pose/cedar_shadow_pose.data_card.plots \
  --data-card-plot-prefix cedar_shadow_pose_train \
  --data-card-plot-heatmap-bin-factor 2 \
  --data-card-view
```

Dry-run preflight (no files written):

```bash
scripts/py -m fisheye.utils.run_keypoint_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --source-type filtered \
  --input-format gray \
  --model-input gray \
  --keypoint-run latest_traditional \
  --dry-run
```

Notes:
- `--train` requires a manifest with non-empty `set_id` to avoid unlinked runs.
- `--set-name` is recommended when using `--train` so `set_id` is generated deterministically.
- `--train` cannot be combined with `--dry-run`.
- `--export-merged` requires a written preflight manifest and cannot be combined with `--dry-run`.
- keypoint data-card aggregation is auto-enabled for `--export-merged`; disable with
  `--no-aggregate-training-data-card`.
- use `--aggregate-training-data-card` to require data-card aggregation even without
  `--export-merged` and fail if the keypoint aggregator module is unavailable.
- `--aggregate-training-data-card` cannot be combined with `--dry-run`.
- `--model-input` defaults to `--input-format` if omitted.
- `--merge-row-gate-policy` defaults to `auto` in pose merged export.
- `--register-registry` defaults to `--registry` when `--register` is set.

For a pose/keypoint-focused operator runbook (query -> prepare -> pipeline -> validate),
see `docs/keypoint_training_workflow.md`.

## Model Export CLI Choice

Use `fisheye.training.export_detection` as the default export CLI for trained detect runs.

- `export_detection` (preferred):
  - resolves run artifacts (`weights`, report, manifest) from a run directory
  - exports ONNX and/or TensorRT
  - embeds lightweight ONNX `metadata_props` (`run_id`, `set_id`, `manifest_sha256`, `system_hostname`, `torch_version`, `cuda_version`, `exported_at_utc`)
  - supports registry updates via `--log-registry --registry ...`

- `export_onnx` (low-level/manual):
  - direct ONNX export from weights (or `--run-dir`)
  - optional metadata embedding via `--meta-run-id`, `--meta-set-id`, `--meta-manifest-sha256`
  - useful for manual debugging or custom export experiments
  - does **not** write export rows to the training registry

Example (preferred, TRT + registry logging):

```bash
scripts/py -m fisheye.training.export_detection \
  /nvme1/models/detect/<set_id>/<run_id> \
  --export-trt \
  --log-registry \
  --registry /nvme1/palette_registry.sqlite
```

Artifact note:
- TensorRT manifest uses `<run_id>_<precision>.tensorrt.manifest.json`.
- TensorRT manifest `build_env` includes export host/device context (hostname, TensorRT version, torch/cuda versions, plus parsed `trtexec` device details when available: selected device name/id/UUID, compute capability, SMs, memory).
