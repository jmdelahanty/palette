# Eye-Mask Training Workflow

Purpose: operator workflow for eye-mask dataset build + training, aligned with
detect/keypoint parity behavior.

All commands use repository-standard `scripts/py`.

## 1) Query Candidate Datasets

```bash
scripts/py -m fisheye.utils.registry_query \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --zarr-use training \
  --model-input gray \
  --eye-mask-review-state approved \
  --eye-mask-review-intended-use training \
  --eye-mask-usable-rate-min 0.70
```

Quick profile/performance checks:

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry /nvme1/palette_registry.sqlite \
  --view eye-mask-profile
```

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry /nvme1/palette_registry.sqlite \
  --view eye-mask-performance
```

## 2) Build Dataset (Merged Export + Card + Plots)

Recommended baseline for LR training with explicit negatives:

```bash
REG="/nvme1/palette_registry.sqlite"

scripts/py -m fisheye.utils.run_eye_mask_training_pipeline \
  --registry "$REG" \
  --dish-design cedar \
  --rig-id omnifin0 \
  --zarr-use training \
  --input-format gray \
  --model-input gray \
  --label-mode lr \
  --eye-stage refined_eye_masks_runs \
  --require-review-state approved \
  --require-review-intended-use training \
  --set-name cedar_shadow_omnifin0_auto_gray_lr \
  --build-dataset \
  --merge-row-gate-policy usable_plus_explicit_negatives \
  --merge-explicit-negative-ratio 1000 \
  --merge-split 0.8/0.2 \
  --merge-seed 42 \
  --data-card-force-plots
```

Notes:
- `--build-dataset` enables merged export + data-card aggregation.
- `--merge-explicit-negative-ratio 1000` approximates “no practical cap”.
- Aggregation is fail-closed by default on stale/missing profile rows.

## 3) Refresh/Repair Profile Rows (When Needed)

If aggregation reports missing/stale `eye_mask_data_profile_latest` rows:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --refresh-eye-mask-profiles
```

or targeted sync:

```bash
scripts/py -m fisheye.utils.sync_eye_mask_profile_registry \
  --registry "$REG" \
  --zarr-use training \
  --apply
```

Then rerun pipeline aggregation stage.

## 4) Train U-Net

From generated config:

```bash
SET_ID="eye_mask_cedar_shadow_omnifin0_auto_gray_lr_v001"
CONFIG="/nvme1/training/datasets/$SET_ID/$SET_ID.yaml"

scripts/py -m fisheye.segmentation.train_unet_eye_masks "$CONFIG" \
  --run-name "${SET_ID}_unet" \
  --output-dir "/nvme1/models/eye_masks/$SET_ID" \
  --device cuda:0 \
  --tb-logdir "/nvme1/models/eye_masks/$SET_ID/tensorboard" \
  --no-compile
```

Registry parity behavior:
- When invoked via `run_eye_mask_training_pipeline --train`, trainer receives
  `--manifest`, `--set-id`, and `--registry`.
- Trainer writes lifecycle status to registry: `in_progress`, `failed`,
  `success`.

## 5) TensorBoard

```bash
scripts/py -m tensorboard.main \
  --logdir "/nvme1/models/eye_masks/$SET_ID/tensorboard" \
  --port 6006 \
  --host 0.0.0.0
```

If TensorBoard initially shows no dashboards, wait for first event writes.

## 6) Common Troubleshooting

- `torch.compile` failures:
  - rerun with `--no-compile`.
- CUDA OOM:
  - lower `training_params.batch_size` in the dataset config YAML.
- Stale/missing profile rows:
  - run maintenance refresh/sync commands above, then rerun aggregation.

## 7) Post-Run Checks

Check model/run rows:

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry "$REG" \
  --view models \
  --limit 20
```

Inspect profile-linked card state:

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry "$REG" \
  --view eye-mask-profile \
  --show-eye-mask-profile
```
