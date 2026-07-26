# Training Data

This guide covers how to create labeled training datasets from analysis Zarrs or
sampled training Zarrs, and how to train detection, pose, and segmentation
models. The normal analysis-Zarr path assumes you have already run the
[analysis pipeline](pipeline_workflow.md) and have Zarrs with detections,
keypoints, and/or masks. Recording-only sampled training Zarrs may use an
initial prediction artifact, but that unbound artifact must be canonically bound
before review.

## Overview

There are four model types you can train:

| Model | What it learns | Input data needed |
|-------|----------------|-------------------|
| **Detection** (YOLO) | Fish bounding boxes | Downsampled frames + detection labels |
| **Pose** (YOLO) | Anatomical keypoints | Cropped ROIs + keypoint annotations |
| **Eye masks** (U-Net) | Eye segmentation | Cropped ROIs + eye mask labels |
| **Subject masks** (U-Net) | Body, eyes union, swim bladder | Cropped ROIs + subject-mask component labels |

Each follows the same pattern:

```
analysis zarrs with labels OR sampled training zarrs with seeded labels
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
- Default: browse detections, make manual corrections, and optionally approve
  on exit with `a`
- `--retune`: adjust blob detection parameters (traditional method only)

Keyboard shortcuts in the UI: **n** / **p** (next/previous), **c** (clear
detection for this frame/slot), **r** (reset current edit), **a** (approve),
**q** (save changes and quit).

For a sampled training Zarr that starts from raw frames only, create unbound
prediction evidence first:

```bash
scripts/py -m fisheye.utils.predict_training_detections \
  /path/to/training.zarr \
  --registry /nvme1/palette_registry.sqlite \
  --model-run-id <registered_detect_run_id> \
  --run-name detect_seed_<model_or_date> \
  --apply
```

This command writes immutable, selector-free
`detection_artifact_runs/<run>`. It does not publish `detect_runs`, and its
training/model-frame boxes are not source-camera coordinate authority. Do not
pass the artifact to `refine_detect --detect-run` or copy/relabel it under
`detect_runs`. An explicit canonical binding/promotion path must validate its
frame-source lineage and publish a distinct canonical detect run first. If that
path is unavailable, stop before review.

After canonical binding, `refine_detect` can use sampled-import passthrough to
write the canonical `refined_detect_runs/<run>/instances` surface, disable
jump/blip filters, and omit a detect-quality report. The refinement step still
applies a tuned
`analysis_metadata.attrs["dish_mask"]` when one is present, just as it does for
full analysis Zarrs: outside-dish seed detections are retained in
`source_detections` for audit but excluded from the approved `instances` surface
as `outside_dish_mask`. After that, review the refined surface with
`detect_review` and approve either in the UI (`a`) or with
`accept_detect_review --state approved --intended-use training`.
Approval for `intended_use=training` materializes the canonical detection data
profile (`analysis/detection_profile_runs/<run>`) and attempts to sync the
registry projection automatically when the dataset is registered. The profile
stores bbox center heatmaps, size/aspect histograms, source-content hashes, and
run-lineage fingerprints used by training data cards.

After approval, run the training image profile when you want to compare image
statistics across training Zarrs before retraining:

```bash
scripts/py -m fisheye.utils.training_image_profile \
  /path/to/training.zarr \
  --apply \
  --sync-registry \
  --registry /nvme1/palette_registry.sqlite
```

This writes `analysis/training_image_profile_runs/<run>` and records intensity,
contrast, sharpness, clipping, illumination-gradient, and optional
fish/background contrast metrics. It complements the detection data profile:
detection profiles describe label geometry; training image profiles describe the
sampled pixels. See `docs/training_image_profile_schema_contract.md`.

After canonical binding, for one-fish-per-frame training Zarrs add
`--per-frame-top-k 1` if the seed detector produces multiple candidates per
sampled frame. This keeps only the highest-confidence candidate in `instances`
while retaining the lower-scoring raw candidates in `source_detections` as
`duplicate` rows for audit/review.

For a resumable queue of pending detection-training archives:

```bash
scripts/py -m fisheye.utils.review_detect_batch \
  --registry /nvme1/palette_registry.sqlite \
  --zarr-use training \
  --path-contains _training.zarr \
  --queue-output /tmp/pending_detect_training_zarrs.txt \
  --details-output /tmp/pending_detect_training_zarrs.tsv \
  --state-file /tmp/detect_review_batch_state.json \
  --all \
  --reviewer "$USER"
```

Resume after interruption with the same state file:

```bash
scripts/py -m fisheye.utils.review_detect_batch \
  --registry /nvme1/palette_registry.sqlite \
  --zarr-use training \
  --path-contains _training.zarr \
  --state-file /tmp/detect_review_batch_state.json \
  --resume \
  --all \
  --reviewer "$USER"
```

This wrapper does not bulk-approve labels; it opens `detect_review` one Zarr at
a time and records progress for restart.

### Keypoint review

```bash
scripts/py -m fisheye.tune.keypoint_review path/to/zarr/..._analysis.zarr
```

Modes:
- Default: browse keypoints and set approval status
- `--retune`: recalculate usable keypoints mask
- `--manual`: draw corrections
- `--audit`: update summary statistics

For the current 5-point `traditional_v2` migration, the preferred labeling path
is not direct use of the existing 5-point model on new PyNvVC-luma crops. Seed
a `traditional_v2` refined run from the reliable 3-point refined run with
`extend_keypoint_skeleton`, then manually complete `snout_tip` and `tail_tip`.
The seed run is expected to have `refined_success=false` and
`usable_keypoints=false` until those new points are completed.

Example manual completion from an SSH/tmux session with X forwarding available:

```bash
scripts/py -m fisheye.tune.keypoint_review \
  /path/to/training.zarr \
  --manual \
  --refined-run refined_keypoints_traditional_v2_seed_pynvvc_luma_v1_20260517_cam2010093 \
  --review-intended-use training
```

Manual review supports arbitrary labels from the selected run metadata. For
`traditional_v2`, use number keys `1` through `5` to select
`swim_bladder`, `eye_left`, `eye_right`, `snout_tip`, and `tail_tip`, then save
with `s`. Use `n`/`p` for navigation, `a` to approve when complete, `x` for
`fish_present_no_keypoints`, and `d` for detection issues. Do not approve
experimental low-confidence 5-point model outputs unless they have been
visually corrected.

### Eye component mask review

Standalone eye-mask review has been retired. Review current eye labels through
the refined subject-mask review surface, targeting `eye_left` and `eye_right`
components when the selected refined run provides them.

```bash
scripts/py -m fisheye.tune.refined_subject_mask_review \
  path/to/zarr/..._analysis.zarr \
  --components eye_left eye_right
```

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

Prefer the registry-driven wrapper when building real training sets. It queries
the registry for matching source Zarrs, gates on refined-detect review status,
then calls the lower-level preparer with the resolved path list.

```bash
scripts/py -m fisheye.utils.prepare_detect_training_from_registry \
  --registry /nvme1/palette_registry.sqlite \
  --path-contains fish_2026 \
  --source-type refined \
  --input-format gray \
  --require-review-state approved \
  --require-review-intended-use training \
  --dry-run
```

Remove `--dry-run` or use `run_detect_training_pipeline --build-dataset` to
write the config, manifest, and optional merged training Zarr. The registry
selector deduplicates multiple registry rows that point at the same physical
Zarr path and prefers rows that have both an approved refined-detect quality
projection and a detection data profile.

For clipped replacements of full-video sampled training Zarrs, keep the
registry wrapper default:

```bash
--training-sample-duplicate-policy prefer-clipped
```

This computes a fingerprint from each training Zarr's
`raw_video/original_frame_indices` plus recording/camera identity. If the
original sampled Zarr and the clipped training Zarr contain the same parent
frames, the clipped `source_layout="rolling_clips"` copy is selected and the
original is skipped. Use `--training-sample-duplicate-policy error` for audit
runs when you want duplicate parent-frame samples to fail closed. Use
`keep-all` only when deliberate double-counting is intended.

To inspect or select only clipped training Zarrs, use the registry source-layout
filter:

```bash
scripts/py -m fisheye.utils.registry_query \
  --registry /nvme1/palette_registry.sqlite \
  --zarr-use training \
  --source-layout rolling_clips
```

Sampled recording-only training Zarrs usually do not have `crop_runs`. For
`--source-type refined`, the preparer reads the approved
`refined_detect_runs/<run>/instances` surface directly; crop approval is not
required for this sampled-training path because there is no crop stage to
approve.

Inventory note: the approved detector-training corpus reached 60 source
training zarrs on 2026-05-16. At that point, 52 were legacy/crop-bearing
training zarrs with a `pynvvc_luma_v1` materialized crop run, and the 8
`sickyfish`/`sleepyfish` sampled training zarrs were detection-only. That
snapshot is now stale for active development: those 8 sampled zarrs are being
promoted into crop/keypoint training sources as explicit crop geometry and
PyNvVC-luma crops are added. Check each archive's current `crop_runs.latest`,
`keypoints_runs.latest`, and `refined_keypoints_runs.latest` before assuming it
is detection-only.

Metadata-only inspection command:

```bash
for z in /nvme1/recordings/{sickyfish_*,sleepyfish_*}/zarr/*_training.zarr \
         /nvme1/recordings/{sickyfish_*,sleepyfish_*}/zarr/*_clipped_training.zarr; do
  [ -d "$z" ] || continue
  echo "$z"
  jq -r '.attributes.latest // "no crop_runs latest"' "$z/crop_runs/zarr.json" 2>/dev/null || true
  jq -r '.attributes.latest // "no keypoints_runs latest"' "$z/keypoints_runs/zarr.json" 2>/dev/null || true
  jq -r '.attributes.latest // "no refined_keypoints_runs latest"' "$z/refined_keypoints_runs/zarr.json" 2>/dev/null || true
done
```

The direct path-based preparer is still useful for explicit lists or debugging:

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

Do not use the direct path-based preparer for the final all-available export
when both original and clipped replacements exist unless you have manually
removed duplicates from the path list. It does not query registry
`source_layout` metadata before building the manifest.

Key options:
- `--source-type`: which detection source family to use. `refined` is the
  current canonical curated surface; `filtered`, `interpolated`, and `manual`
  are legacy compatibility options.
- `--input-format`: frame format (`gray` or `rgb`)
- `--require-review-state` / `--require-review-intended-use`: registry-wrapper
  gates on refined-detect review metadata; use these on
  `prepare_detect_training_from_registry` or `run_detect_training_pipeline`
- `--max-interpolated-detections-rate`: registry-wrapper compatibility gate for
  older refined archives that still carry interpolation-heavy outputs
- `--allow-unapproved`: direct-preparer escape hatch to skip on-disk review
  checks; not recommended for production training

### Detection model input size

When seeding labels on sampled training Zarrs, choose the frame source that
matches the model input size when possible. The registry can report this for
trained/exported detection models:

- `training_runs.final_metrics_json` can include `imgsz_h` and `imgsz_w` for
  the trained `.pt` model.
- `onnx_models` and `tensorrt_models` record exported `input_shape`, `img_h`,
  and `img_w`.
- `detect_model_performance_latest` records the `inference_width` and
  `inference_height` used by model-backed detection runs.

For example, to inspect current detection model dimensions:

```bash
sqlite3 /nvme1/palette_registry.sqlite \
  "select run_id,set_id,model_path,substr(final_metrics_json,1,500) \
   from training_runs \
   where task_type='detect' and status='success';"

sqlite3 /nvme1/palette_registry.sqlite \
  "select run_id,set_id,path,input_shape,img_h,img_w \
   from onnx_models \
   where path like '%detect%';"
```

The current `detect_cedar_shadow_v007` detector is registered with
`imgsz_h=640`, `imgsz_w=640`, and ONNX/TensorRT `input_shape=[1, 3, 640, 640]`.
For sampled training Zarrs that already contain `raw_video/images_ds` at
`640x640`, prefer that array for label seeding instead of reading
`raw_video/images_full` and resizing again. If the stored downsampled frame
shape does not match the model, use the full-resolution array and let the
inference path perform the documented resize/letterbox transform.

Registry schema migration 49 normalizes the trained model input contract
directly on `training_models` and exposes a shared `model_input_shapes` query
view:

```bash
sqlite3 /nvme1/palette_registry.sqlite \
  "select artifact_kind,run_id,task_type,artifact_path,input_shape,img_h,img_w \
   from model_input_shapes \
   where task_type='detect' and img_h=640 and img_w=640;"
```

See
[`../model_input_shape_registry_design.md`](../model_input_shape_registry_design.md)
for the schema and backfill plan.

### Generate an unbound detection-prediction artifact

Use `predict_training_detections` to run a registered detector directly over
frames stored inside a training Zarr. The utility resolves the model input
shape through `model_input_shapes`, prefers `raw_video/images_ds_rgb` or
`raw_video/images_ds` when the stored frame shape matches the detector, and
falls back to `raw_video/images_full` with YOLO resizing when needed.

Dry-run first:

```bash
scripts/py -m fisheye.utils.predict_training_detections \
  /path/to/training.zarr \
  --registry /nvme1/palette_registry.sqlite \
  --model-run-id omnifin0_cedar_shadow_v007_detect_20260206-235656_25f3fbcb
```

Write the immutable `detection_artifact_runs/<run>` evidence:

```bash
scripts/py -m fisheye.utils.predict_training_detections \
  /path/to/training.zarr \
  --registry /nvme1/palette_registry.sqlite \
  --model-run-id omnifin0_cedar_shadow_v007_detect_20260206-235656_25f3fbcb \
  --run-name detect_seed_v007 \
  --apply
```

The written run records the selected model, `input_shape_status`, frame-source
array and exact extent, sampled-frame mapping (`source_frame_indices` when
available), dense run-local `artifact_row_id`, array-specific unbound numeric
semantics, prediction parameters, and a sealed live payload inventory. It never
advances a detector selector and cannot be consumed as an ordinary detect run.

Before refinement, an explicit canonical binding/promotion step must validate
the artifact's frame-source lineage and publish a distinct canonical
`detect_runs/<run>`. Do not assume that a `640x640` training frame can be mapped
to source-camera pixels by a resolution ratio. If no approved binding path is
available, fail closed rather than creating `refined_detect_runs` from the
artifact.

Refinement, not prediction, is responsible for applying the dish-mask spatial
gate. This keeps raw model predictions immutable while making the curated
training surface respect the tuned dish geometry.

### Listing existing versions

```bash
scripts/py -m fisheye.utils.list_training_versions
scripts/py -m fisheye.utils.list_training_versions --name my_detect_set
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

### Subject mask merge

For the unified subject-mask model, prepare source selection from the registry
and then export one merged training zarr:

```bash
scripts/py -m fisheye.utils.prepare_subject_mask_training_from_registry \
  --registry /nvme1/palette_registry.sqlite \
  --require-component subject_body \
  --require-component eyes_union \
  --require-component swim_bladder \
  --out-config /path/to/subject_mask_training.yaml \
  --out-manifest /path/to/subject_mask_training.manifest.json \
  --set-name subject_masks_dense_all_components

scripts/py -m fisheye.utils.run_subject_mask_training_pipeline \
  --config /path/to/subject_mask_training.yaml \
  --manifest /path/to/subject_mask_training.manifest.json \
  --registry /nvme1/palette_registry.sqlite \
  --export-merged \
  --merge-out-zarr /path/to/subject_mask_merged.zarr \
  --subject-label-schema subject_v1_union
```

Subject-mask training artifacts use `training_task="subject_masks"` and store
supervision in `subject_mask_runs/<run>/masks_roi` plus
`target_valid_channels`. The target schema is encoded in the artifact attrs, so
the loader/trainer should derive `names` and `nc` from the artifact instead of
requiring hand-edited config defaults.

Use the source audit when checking that registry-selected sources still match
on-disk masks:

```bash
scripts/py -m fisheye.utils.audit_subject_mask_training_sources \
  /path/to/subject_mask_training.manifest.json
```

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

#### Current preferred detection baseline

As of 2026-05-16, the preferred detector baseline is:

```text
set_id: detect_all_available_detect_training_v003
run_id: detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1
status: success
input: RGB, NCHW, [1, 3, 640, 640]
best epoch: 41
best validation metrics: P=0.9794, R=0.9787, mAP50=0.9838, mAP50-95=0.7539
```

Primary artifacts:

```text
/nvme1/models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/weights/best.pt
/nvme1/models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/exports/onnx/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1.onnx
/nvme1/models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/exports/tensorrt/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1_fp16.engine
```

The v003 retry reproduced the previous v002 baseline exactly at the metric
level, but v003 is the cleaner preferred baseline because it was built from the
current 60 approved source training zarrs and has clean `success` rows in
`training_runs`, `training_models`, `onnx_models`, and `tensorrt_models`.
The interrupted `detect_all_available_detect_training_v003_yolo11n_trt_20260516_tmux`
attempt should be treated as superseded by the retry run above.

For production detection runs, prefer leaving registry logging enabled and pass
the manifest and set ID explicitly. Long-running GPU jobs should be launched in
`tmux` or an equivalent scheduler job, not as a foreground Codex/tool process,
because a client/session interruption can terminate foreground child processes.

Example: train from a merged registry-built dataset and export through FP16
TensorRT in one run:

```bash
tmux new-session -d -s palette_detect_train_v003 '
cd /home/delahantyj@hhmi.org/gitrepos/palette &&
env MPLCONFIGDIR=/tmp/matplotlib-training \
    ULTRALYTICS_CONFIG_DIR=/tmp/ultralytics-training \
  scripts/py -m fisheye.training.train_detection \
    /nvme1/training/datasets/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003.yaml \
    --manifest /nvme1/training/datasets/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003.manifest.json \
    --set-id detect_all_available_detect_training_v003 \
    --registry /nvme1/palette_registry.sqlite \
    --project /nvme1/models/detect/detect_all_available_detect_training_v003 \
    --run-name detect_all_available_detect_training_v003_yolo11n_trt_YYYYMMDD \
    --export-trt \
    --trt-precision fp16 \
    --trtexec /usr/local/TensorRT-10.0.1.6/bin/trtexec \
    --trt-profiling \
    > /tmp/detect_all_available_detect_training_v003_train_trt_YYYYMMDD.log 2>&1
'
```

Monitor:

```bash
tmux attach -t palette_detect_train_v003
tail -f /tmp/detect_all_available_detect_training_v003_train_trt_YYYYMMDD.log
```

Expected detection-training outputs:

- `weights/best.pt` and `weights/last.pt`: trained Ultralytics weights.
- `results.csv`, plots, and validation images in the run directory.
- `<timestamp>_detection_training_report.yaml`: effective config, source
  manifest summary, final metrics, and artifact paths.
- `inputs/`: snapshotted config and manifest used for the run.
- `exports/onnx/<run_id>.onnx` and
  `exports/onnx/<run_id>.onnx.manifest.json` when ONNX export is enabled.
- `exports/tensorrt/<run_id>_<precision>.engine` and
  `exports/tensorrt/<run_id>_<precision>.tensorrt.manifest.json` when
  TensorRT export succeeds.
- Registry rows in `training_runs`, `training_models`, `onnx_models`, and
  `tensorrt_models` when `--no-log-registry` is not used.

Operational checks before launching a TensorRT run:

```bash
scripts/py -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
/usr/local/TensorRT-10.0.1.6/bin/trtexec --version
```

Post-run artifact check:

```bash
RUN=/nvme1/models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_YYYYMMDD
find "$RUN" -maxdepth 4 -type f \( -name best.pt -o -name results.csv -o -name "*.onnx" -o -name "*.engine" -o -name "*.manifest.json" \) | sort
```

Strict JSON check for export manifests:

```bash
scripts/py - <<'PY'
import json
from pathlib import Path
root = Path("/path/to/training/run")
bad = []
for p in root.rglob("*.json"):
    try:
        json.loads(p.read_text())
    except Exception as exc:
        bad.append((p, exc))
print("bad_json", len(bad))
for path, exc in bad[:20]:
    print(path, exc)
raise SystemExit(1 if bad else 0)
PY
```

### Pose / keypoints

```bash
scripts/py -m fisheye.training.train_pose \
  /path/to/pose_config.yaml \
  --run-name my_pose_v1 \
  --no-log-registry
```

### Subject masks

```bash
scripts/py -m fisheye.segmentation.train_unet_subject_masks \
  /path/to/subject_mask_training.yaml \
  --manifest /path/to/subject_mask_training.manifest.json \
  --set-id subject_masks_dense_all_components_v001 \
  --registry /nvme1/palette_registry.sqlite \
  --run-name my_subject_masks_v1 \
  --device cuda:0 \
  --val-preview-thresholds 0.5,0.25,0.1
```

The pipeline wrapper can run the same trainer after export with `--train`. It
also forwards useful operator options such as `--epochs`, `--device`,
`--tb-logdir`, and `--no-progress`.

All trainers:
- Use Ultralytics YOLO under the hood (detection and pose) or a custom U-Net
  (eye masks and subject masks)
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
frame. See [sampled_import.md](sampled_import.md) for the full operator guide,
including the recording-only/video-only path.

```bash
scripts/py -m fisheye.utils.import_sampled_training_pynvvc \
  /path/to/video.mp4 \
  /path/to/training_sample.zarr \
  --source-frame-count 54000 \
  --frame-step 100 \
  --config configs/fisheye/default.yaml
```

`--frame-step 100` imports every 100th frame (both full
resolution and downsampled). This is useful for building compact training
datasets without importing entire videos.

### Batch sampled import

```bash
scripts/py -m fisheye.utils.import_recordings_training /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --dry-run
```

Remove `--dry-run` and add `--apply` to execute.

For organized recording-only directories that have camera videos but no H5 or
protocol source, use `scripts/py -m fisheye.utils.intake_video_only_recording`
instead. That wrapper stamps `experiment_context_status = "absent"` and can
register the resulting training Zarr in the registry.

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
`prepare_subject_mask_training_from_registry`) query the registry to find approved
datasets matching your criteria (dish design, rig, review state) and build
configs automatically.

The registry also tracks training runs, model exports, and their provenance.
See `docs/training_data_workflow.md` for the full registry-based workflow.
