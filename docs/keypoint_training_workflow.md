# Keypoint Training Workflow

Purpose: provide an operator workflow for pose/keypoint training that mirrors the detect workflow shape:

- registry query
- prepare-only config/manifest generation
- one-command pipeline (optional merged export and train)
- post-run registry checks

All commands below use repository-standard `scripts/py`.

## 1) Query Registry Candidates

Dataset query with keypoint-quality gates:

```bash
scripts/py -m fisheye.utils.registry_query \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --zarr-use training \
  --model-input gray \
  --keypoint-review-state approved \
  --keypoint-review-intended-use training \
  --keypoint-usable-rate-min 0.70 \
  --limit 20
```

Quick keypoint-quality summary:

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry /nvme1/palette_registry.sqlite \
  --view keypoint-quality \
  --show-keypoint-quality
```

## 2) Prepare-Only Flow

Generate pose config + manifest without training:

```bash
scripts/py -m fisheye.utils.prepare_keypoint_training_from_registry \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --source-type filtered \
  --input-format gray \
  --model-input gray \
  --keypoint-run latest_traditional \
  --require-review-state approved \
  --require-review-intended-use training \
  --min-usable-keypoints-rate 0.70 \
  --set-name cedar_shadow_pose \
  --out-config /tmp/pose_build.yaml \
  --out-manifest /tmp/pose_build.manifest.json \
  --register
```

Dry-run preflight:

```bash
scripts/py -m fisheye.utils.prepare_keypoint_training_from_registry \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --source-type filtered \
  --input-format gray \
  --keypoint-run latest_traditional \
  --dry-run
```

## 3) One-Command Pipeline Flow

Preflight + merged export + train:

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

Optional ONNX/TRT after train:

```bash
scripts/py -m fisheye.utils.run_keypoint_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --source-type filtered \
  --input-format gray \
  --keypoint-run latest_traditional \
  --set-name cedar_shadow_pose \
  --train \
  --export-onnx \
  --export-trt
```

Dry-run preflight (no files written):

```bash
scripts/py -m fisheye.utils.run_keypoint_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --source-type filtered \
  --input-format gray \
  --keypoint-run latest_traditional \
  --dry-run
```

## 4) Direct Merged Export + Validation

Export merged Zarr from an existing manifest:

```bash
scripts/py -m fisheye.utils.export_keypoint_training_zarr \
  --manifest /tmp/pose_build.manifest.json \
  --merge \
  --out-zarr /tmp/pose_build_merged.zarr \
  --split 0.8/0.2 \
  --seed 42
```

Validate merged output:

```bash
scripts/py -m fisheye.utils.validate_keypoint_training_zarr \
  /tmp/pose_build_merged.zarr \
  --expected-input-format gray
```

## 5) Registry Checks After Train

Show model/run/export surfaces:

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry /nvme1/palette_registry.sqlite \
  --view models \
  --limit 20
```

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry /nvme1/palette_registry.sqlite \
  --view onnx \
  --limit 20
```

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry /nvme1/palette_registry.sqlite \
  --view tensorrt \
  --limit 20
```

## 6) Behavior and Defaults (Current)

- `--model-input` defaults to `--input-format` in prepare/pipeline wrappers.
- `--set-name` is auto-generated when omitted and `--out-config` is not provided.
- `--register-registry` defaults to `--registry` when `--register` is set.
- `run_keypoint_training_pipeline` is prepare-first:
  - `--train` runs `fisheye.training.train_pose` after preflight.
  - `--export-merged` requires a written manifest and cannot be used with `--dry-run`.
  - `--train` cannot be combined with `--dry-run`.
- Merged pose export defaults:
  - `--merge-row-gate-policy auto` (prefers refined `usable_keypoints` when available).
  - `--trt-precision fp16` when `--export-trt` is enabled (unless overridden).
