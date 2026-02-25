# Eye-Mask Profile Registry Ops Runbook

Purpose: triage and repair eye-mask training data-card aggregation failures caused by missing
`eye_mask_data_profile_latest` rows, and document when fallback mode is appropriate.

## 1. Failure Signature

Typical aggregator failure:

```text
Training data card aggregation failed: Missing eye_mask_data_profile_latest rows for dataset_id(s): <dataset_id>.
Run scripts/py -m fisheye.utils.sync_eye_mask_profile_registry --registry <REGISTRY> --apply.
To continue without profile rows, rerun with --allow-profile-fallback-scan.
```

Typical sync output for affected datasets:

```text
missing_profile    <dataset_id>    -    <zarr_path>    analysis/eye_mask_profile_runs missing
```

Working example from incident:

- `dataset_id`: `2026-01-28T19-22-28Z_arena_1:zc66de17bea1b`
- `zarr_path`: `/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_training.zarr`

## 2. Root Cause

This failure mode occurs when:

1. A source training zarr does not yet contain a usable `analysis/eye_mask_profile_runs/<run>/attrs['profile_summary']`.
2. Therefore `sync_eye_mask_profile_registry` cannot upsert that dataset into `eye_mask_data_profile`.
3. The data-card aggregator (registry-first, fail-closed by default) stops on missing profile rows.

In this incident, the eye-mask source run data existed on disk, but profile rows were absent until
backfill + sync were performed for the specific dataset.

## 3. One-Dataset Fix Commands

Use `scripts/py` and repair only the failing dataset.

```bash
REG=/nvme1/palette_registry.sqlite
DS='2026-01-28T19-22-28Z_arena_1:zc66de17bea1b'
ZARR='/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_training.zarr'
MANIFEST='/nvme1/training/datasets/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v001/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v001.manifest.json'
CARD='/nvme1/training/datasets/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v001/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v001.data_card.json'
```

Pick a valid source run (prefer refined, fallback to raw):

```bash
REFINED_RUN="$(find "$ZARR/refined_eye_masks_runs" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -n1)"
RAW_RUN="$(find "$ZARR/eye_masks_runs" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -n1)"

if [ -n "$REFINED_RUN" ]; then
  SRC="refined_eye_masks_runs/$REFINED_RUN"
elif [ -n "$RAW_RUN" ]; then
  SRC="eye_masks_runs/$RAW_RUN"
else
  echo "No eye-mask runs found under $ZARR"
  exit 1
fi
```

Backfill profile run into that zarr:

```bash
scripts/py -m fisheye.utils.backfill_eye_mask_profiles "$ZARR" \
  --zarr-use training \
  --source-eye-mask-path "$SRC" \
  --registry "$REG" \
  --apply
```

Sync exactly that dataset row:

```bash
scripts/py -m fisheye.utils.sync_eye_mask_profile_registry \
  --registry "$REG" \
  --dataset-id "$DS" \
  --apply
```

## 4. Verification Steps

Confirm latest-profile row now exists:

```bash
scripts/py -m fisheye.utils.registry_query \
  --eye-mask-data-profile-latest \
  --profile-dataset-id "$DS" \
  --json | jq 'length'
```

Expected: `1`

Re-run aggregation (no fallback):

```bash
scripts/py -m fisheye.utils.aggregate_eye_mask_training_data_card \
  --manifest "$MANIFEST" \
  --output "$CARD"
```

Optional quick card sanity check:

```bash
jq '{set_id, created_at_utc, dataset_count:.selection.dataset_count, total_rois:.quality.total_rois}' "$CARD"
```

## 5. When `--allow-profile-fallback-scan` Is Actually Needed

Use `--allow-profile-fallback-scan` only when you explicitly choose to proceed even though
`eye_mask_data_profile_latest` is missing/incomplete.

Use fallback when:

1. You need an unblock immediately.
2. You cannot backfill/sync missing profile rows yet.

Do not use fallback when:

1. Missing rows can be repaired quickly (preferred).
2. You need strict registry-first provenance parity.

Fallback example:

```bash
scripts/py -m fisheye.utils.aggregate_eye_mask_training_data_card \
  --manifest "$MANIFEST" \
  --output "$CARD" \
  --allow-profile-fallback-scan
```

## 6. How To View the Generated Data Card and Plots

The eye-mask aggregator supports `--view` and `--force`.

Open existing/generated plots after aggregation:

```bash
scripts/py -m fisheye.utils.aggregate_eye_mask_training_data_card \
  --manifest "$MANIFEST" \
  --output "$CARD" \
  --view
```

Force regenerate plots and open:

```bash
scripts/py -m fisheye.utils.aggregate_eye_mask_training_data_card \
  --manifest "$MANIFEST" \
  --output "$CARD" \
  --view --force
```

Pipeline equivalent flags:

```bash
scripts/py -m fisheye.utils.run_eye_mask_training_pipeline \
  ... \
  --aggregate-training-data-card \
  --data-card-view \
  --data-card-force-plots
```

View JSON directly:

```bash
jq . "$CARD" | less
```
