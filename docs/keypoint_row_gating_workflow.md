# Keypoint Row Gating Workflow

This document explains how pose row filtering works after the merged row-gate changes.

## Summary

Pose selection now has two levels:

1. Dataset-level gating (registry/preflight):
   - Uses reviewed refined quality metadata (`review_state`, `review_intended_use`, `usable_keypoints_rate`).
2. Row-level gating (merged export):
   - Chooses which individual ROI/keypoint rows are included in the merged training Zarr.

Row-level filtering is now finalized during merged export, not deferred to loader-only behavior.

## Dataset-Level Gating (Preflight)

`prepare_keypoint_training_from_registry` chooses which source datasets/runs are allowed, based on:

- review requirements (`--require-review-state`, `--require-review-intended-use`)
- minimum usable threshold (`--min-usable-keypoints-rate`)

This stage decides *which datasets/runs* are eligible, not which rows inside each dataset.

## Row-Level Gating (Merged Export)

`export_keypoint_training_zarr` applies row gating policy when building merged pose datasets.

CLI:

- `--row-gate-policy auto` (default)
- `--row-gate-policy refined_usable`
- `--row-gate-policy raw_success`

Policy semantics:

1. `auto`:
   - Use refined `usable_keypoints` mask when available for the selected keypoint run.
   - Fallback to raw `detection_success` if refined usable mask is unavailable.
2. `refined_usable`:
   - Require refined `usable_keypoints`.
   - Fail if no compatible refined mask is available.
3. `raw_success`:
   - Always use raw `detection_success`.

## What Gets Written

Merged keypoint run attrs include:

- `method = "merged_export"`
- `row_gate_applied = true`
- `row_gate_policy = <policy|mixed>`
- `row_gate_counts = {...}`

Merged manifest/summary include row-gate provenance:

- requested/applied policy
- per-policy counts
- per-source row-gate stats (selected, total, refined run, etc.)

## Loader Behavior After Export

For merged pose exports with:

- `method=merged_export`
- `row_gate_applied=true`

the loader does **not** apply an additional `detection_success` row filter.

Reason: row inclusion was already finalized at export.

## Why Counts May Differ

You may still see:

- source keypoint rows (`keypoints_roi` in source run) > merged samples

That difference is expected when row gating excludes rows before merge output is written.

## Diagnostics

Use:

```bash
scripts/py src/fisheye/utils/check_training_sample_accounting.py <training_config.yaml>
```

It reports:

- selected keypoint run
- row gate policy/applied flag
- keypoint rows
- raw success rows
- final valid rows
- sampled/train/val counts

## Pipeline Usage

Through wrapper:

```bash
scripts/py -m fisheye.utils.run_pose_training_pipeline \
  --registry /nvme1/palette_registry.sqlite \
  --source-type filtered \
  --input-format gray \
  --keypoint-run latest_traditional \
  --require-review-state approved \
  --require-review-intended-use training \
  --min-usable-keypoints-rate 0.70 \
  --export-merged \
  --merge-row-gate-policy auto
```

Direct merged export (advanced):

```bash
scripts/py -m fisheye.utils.export_keypoint_training_zarr \
  --manifest /path/to/preflight.manifest.json \
  --merge \
  --row-gate-policy auto
```

## Recommended Default

Use `auto` for normal curated workflows:

- honors refined usable mask when present,
- preserves backward compatibility via raw-success fallback.

Use `refined_usable` for strict review-only training.
