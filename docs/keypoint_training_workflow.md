# Keypoint Training Workflow

Purpose: provide an operator workflow for pose/keypoint training that mirrors the detect workflow shape:

- registry query
- prepare-only config/manifest generation
- one-command pipeline (optional merged export and train)
- post-run registry checks

For external-tool interoperability, see
[`pose_coco_interoperability.md`](pose_coco_interoperability.md). Palette Zarr
remains the canonical pose store; COCO/YOLO-style outputs are export views.

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
  --input-format gray \
  --keypoint-run latest_traditional \
  --dry-run
```

If your approved training pool spans more than one skeleton lineage, add an
explicit selector so preflight keeps only the intended annotation source instead
of failing on mixed skeleton signatures. Use the 5-point selector only for
completed and approved `traditional_v2` refined runs, not for seed runs that
still have missing `snout_tip` / `tail_tip` values:

```bash
scripts/py -m fisheye.utils.prepare_keypoint_training_from_registry \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --input-format gray \
  --model-input gray \
  --keypoint-run latest_traditional \
  --skeleton-id pose_skel_traditional_v2 \
  --require-review-state approved \
  --require-review-intended-use training \
  --dry-run
```

Troubleshooting:

- If preflight fails with `keypoint_quality row is stale for filesystem mtime`, refresh registry quality rows and rerun:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry /nvme1/palette_registry.sqlite \
  --refresh-keypoint-quality
```

- If preflight fails with `stale keypoint_quality row: expected refined_run ..., observed ...` and the archive contains both an older refined run and a later migrated refined run for the same `source_keypoints_run`, the current resolver in `prepare_keypoint_training_from_registry.py` can pick the wrong refined run when `created_utc` ties. Current workaround:
  1. repair the migrated refined runs in batch:

```bash
scripts/py -m fisheye.utils.repair_keypoint_training_refined_run_ties \
  --registry /nvme1/palette_registry.sqlite
```

This repair utility matches migrated refined runs using the `traditional_v2_seed`
name pattern, so it covers both plain names like `..._traditional_v2_seed` and
numbered variants like `..._traditional_v2_seed_001`.

  2. rerun `--refresh-keypoint-quality`
  3. rerun preflight

- Needed Palette patch:
  `prepare_keypoint_training_from_registry.py` should break refined-run ties the same way the registry `keypoint_quality_current` view does, instead of sorting only on `created_utc`.

## 3) One-Command Pipeline Flow

Preflight + merged export + train:

```bash
scripts/py -m fisheye.utils.run_keypoint_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
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

Preflight + merged export + keypoint data-card aggregation (plots + view):

```bash
scripts/py -m fisheye.utils.run_keypoint_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
  --input-format gray \
  --model-input gray \
  --keypoint-run latest_traditional \
  --set-name cedar_shadow_pose \
  --export-merged \
  --data-card-output /tmp/cedar_shadow_pose.data_card.json \
  --data-card-split train \
  --data-card-plot-dir /tmp/cedar_shadow_pose.data_card.plots \
  --data-card-plot-prefix cedar_shadow_pose_train \
  --data-card-plot-heatmap-bin-factor 2 \
  --data-card-view
```

If aggregation fails with stale/missing `keypoint_data_profile_latest` rows:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry /nvme1/palette_registry.sqlite \
  --refresh-keypoint-profiles

scripts/py -m fisheye.utils.check_training_registry \
  --registry /nvme1/palette_registry.sqlite \
  --view keypoint-profile \
  --no-rich
```

Then rerun pipeline. Emergency overrides (not recommended for production):
- `--data-card-allow-profile-mtime-mismatch`
- `--data-card-allow-profile-fallback-scan`

Optional ONNX/TRT after train:

```bash
scripts/py -m fisheye.utils.run_keypoint_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --dish-design cedar \
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

## Pose Runtime Contract

Pose training is fail-closed at three boundaries:

- `training_params` rejects unknown fields, including misspelled loss or
  augmentation names;
- the trainer compares requested loss, optimizer, augmentation, worker, seed, and
  image-size values with the effective Ultralytics arguments before the first
  epoch;
- the first runtime batch must be three-channel `uint8` at the declared model
  input shape and must become `float32` in `[0, 1]` after normalization.

The training, batch-diagnostic, and integrity-audit entry points all use the same
pose loader builder. Native ROIs are transformed with the shared reversible
`identity`/`pad_to_size` contract. The default checked-in profile uses a 512x512
model input: 512x512 pixels remain unchanged and smaller crops are centered with
zero padding rather than resized.

Pose augmentation is opt-in through `training_params.augment`. Palette applies
the configured photometric, affine, flip, and erasing operations in the Zarr
loader and transforms boxes and keypoints with the image. Directional landmarks
must be bound by label-name pairs such as `eye_left`/`eye_right`; keypoint indices
are never hard-coded.

Every successful run writes `pose_training_runtime_receipt.json` beside the
weights. Its digest is recorded in the training report and registry metrics. The
receipt contains the starting-model digest and architecture, source and model
input shapes, effective arguments, loader worker policy, active augmentation,
and the observed first-batch tensor contract.

Before a full run, inspect the exact configured loader without training:

```bash
scripts/py -m fisheye.training.diagnose_pose_batch /path/to/pose_build.yaml \
  --batch-size 8
```

## 6) Behavior and Defaults (Current)

- `--model-input` defaults to `--input-format` in prepare/pipeline wrappers.
- `--set-name` is auto-generated when omitted and `--out-config` is not provided.
- `--register-registry` defaults to `--registry` when `--register` is set.
- `run_keypoint_training_pipeline` is prepare-first:
  - `--train` runs `fisheye.training.train_pose` after preflight.
  - `--export-merged` requires a written manifest and cannot be used with `--dry-run`.
  - `--train` cannot be combined with `--dry-run`.
  - `--aggregate-training-data-card` cannot be combined with `--dry-run`.
  - keypoint data-card aggregation is auto-enabled for `--export-merged` unless
    `--no-aggregate-training-data-card` is set.
  - keypoint data-card aggregation is fail-closed on stale profile rows by
    default; use `--data-card-allow-profile-mtime-mismatch` only for
    controlled recovery.
  - missing keypoint profile rows fail closed by default; use
    `--data-card-allow-profile-fallback-scan` only for controlled fallback.
  - if the keypoint data-card aggregator module is unavailable, auto-aggregation is skipped;
    use `--aggregate-training-data-card` to require it and fail closed.
- Merged pose export defaults:
  - `--merge-row-gate-policy auto` (prefers refined `usable_keypoints` when available).
  - `--trt-precision fp16` when `--export-trt` is enabled (unless overridden).
