# Detect vs Pose Merged-Set Behavior Checklist

Purpose: verify that pose merged-set creation behavior matches detection workflow expectations, while keeping pose-specific quality gates intact.

## How to Use

- Run this checklist whenever `prepare_*_from_registry`, `run_*_pipeline`, exporter, validator, or registry write paths change.
- Mark each item PASS/FAIL and capture command output paths in the run notes.

## 1) Entry-Point Parity

- [ ] Detect preflight exists and runs: `fisheye.utils.prepare_detect_training_from_registry`.
- [ ] Pose preflight exists and runs: `fisheye.utils.prepare_pose_training_from_registry` (alias) and `fisheye.utils.prepare_keypoint_training_from_registry`.
- [ ] Detect wrapper exists and runs: `fisheye.utils.run_detect_training_pipeline`.
- [ ] Pose wrapper exists and runs: `fisheye.utils.run_pose_training_pipeline` (alias) and `fisheye.utils.run_keypoint_training_pipeline`.
- [ ] Detect train entrypoint exists and runs: `fisheye.training.train_detection`.
- [ ] Pose train entrypoint exists and runs: `fisheye.training.train_pose` (alias) and `fisheye.training.train_keypoints`.

## 2) Preflight Selection Semantics

- [ ] Detect preflight only selects source datasets (never training-purpose merged artifacts).
- [ ] Pose preflight only selects source datasets (never training-purpose merged artifacts).
- [ ] Pose review/quality gates are applied at SQL level using `keypoint_quality_current`.
- [ ] Pose selector semantics are preserved (`latest_traditional`, `latest_yolo`, fallback policy).
- [ ] Quality exclusions are reported with explicit reasons.

## 3) Fail-Closed Pose Validation

- [ ] For each selected pose row, preflight re-checks Zarr metadata before export/train.
- [ ] Preflight fails if refined run is missing/stale/diverged from registry row.
- [ ] Preflight fails if review state/intended use diverges from selected row.
- [ ] Preflight fails if usable/total/rate diverges from selected row.

## 4) Generated Artifact Parity

- [ ] Detect and pose both emit `<set_id>.yaml` and `<set_id>.manifest.json`.
- [ ] Detect and pose merged export both emit `<set_id>_merged.zarr`.
- [ ] Detect and pose merged export both emit `<set_id>_merged.summary.json`.
- [ ] Split behavior (`train/val`) is deterministic for same seed and inputs.

## 5) Registry Linkage Parity

- [ ] `training_sets` row is written with consistent `set_id`, `config_path`, `manifest_path`.
- [ ] Merged dataset is registered with `zarr_purpose="training"`.
- [ ] Training set links selected dataset IDs and merged dataset ID.
- [ ] Pose runs appear in `check_training_registry --view models`.

## 6) Runtime Wrapper Behavior

- [ ] Detect wrapper can run preflight -> merged export -> optional train in one command.
- [ ] Pose wrapper can run preflight -> merged export -> optional train in one command.
- [ ] Pose wrapper “Next:” command points to `fisheye.training.train_pose`.
- [ ] Pose training params are sanitized so Ultralytics-only args are forwarded and loader-only args are not.

## 7) Observability and Audit

- [ ] `check_training_registry --view keypoint-quality --show-keypoint-quality` shows pass/exclude reasons.
- [ ] Integrity check reports stale/divergent keypoint-quality rows.
- [ ] Backfill/refresh maintenance commands report inserted/updated/deleted/unchanged counts.

## Suggested Command Set (Operator Smoke Test)

```bash
scripts/py -m fisheye.utils.prepare_pose_training_from_registry \
  --registry /nvme1/palette_registry.sqlite \
  --base-config configs/fisheye/pose_config.yaml \
  --source-type filtered \
  --input-format gray \
  --keypoint-run latest_traditional \
  --require-review-state approved \
  --require-review-intended-use training \
  --min-usable-keypoints-rate 0.70 \
  --dry-run

scripts/py -m fisheye.utils.check_training_registry \
  --registry /nvme1/palette_registry.sqlite \
  --view keypoint-quality \
  --show-keypoint-quality
```

## Exit Criteria

- [ ] All checklist items pass on a real registry.
- [ ] No training-purpose detect merged datasets are considered as pose source candidates.
- [ ] Pose selection is deterministic and auditable from registry + manifest alone.
