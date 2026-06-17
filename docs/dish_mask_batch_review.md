# Dish Mask Batch Apply + Manual Review

This workflow is for propagating a tuned dish mask across many recordings and then **visually verifying** each target with the interactive tuner. In practice the batch apply gets you close, but **manual review is still required** to ensure the dish boundary is correct for each recording.

## Why this exists

- Dishes are consistent in design but **center/scale can drift** per camera and session.
- A single tuned mask often transfers well, but **not perfectly**.
- We use a batch apply to reduce manual work, then **review and correct** each Zarr.

## Data sources

- Dish mask is stored on each Zarr in:
  - `analysis_metadata/attrs['dish_mask']`
- Batch apply groups by `experimental_chamber` (dish type).
  - This comes from H5 root attr `experimental_chamber` (new recordings)
  - For older recordings it is backfilled from:
    - `/calibration_snapshot/arena_config_json.selected_dish_type_name`

## Quick workflow

1) **Tune one representative recording**
```bash
scripts/py -m fisheye.tune.mask_tuner /nvme1/recordings/<recording>/zarr/<recording>.zarr \
  --registry /nvme1/palette_registry.sqlite
```

The tuner always writes `analysis_metadata.attrs["dish_mask"]`. Passing
`--registry` also marks `recording_step_status.dish_mask=ok` for the matching
dataset immediately; otherwise registry maintenance can refresh the status
later.

2) **Backfill chamber (older recordings only)**
```bash
scripts/py -m fisheye.utils.backfill_experimental_chamber /nvme1/recordings --recursive --apply
```

3) **Batch apply dish mask by chamber**
```bash
scripts/py -m fisheye.utils.apply_dish_mask_by_chamber /nvme1/recordings \
  --recursive \
  --source /nvme1/recordings/<recording>/zarr/<recording>.zarr \
  --registry /nvme1/palette_registry.sqlite \
  --apply
```

4) **Manual review / correction**
```bash
scripts/py -m fisheye.utils.review_dish_masks /nvme1/recordings \
  --recursive \
  --chamber cedar \
  --registry /nvme1/palette_registry.sqlite
```

## Batch apply details

`apply_dish_mask_by_chamber.py`:

- Re-detects the circle **per target** using the Hough params from the source dish mask.
- Uses `images_ds` by default for speed (pass `--full` for full-res).
- Uses a mid-frame by default (pass `--frame` to force a specific index).
- Skips targets with existing dish masks unless `--overwrite` is used.
- With `--registry`, successful target saves also upsert `dish_mask=ok`.
- Logs JSONL by default:
  - `$PALETTE_LOG_ROOT/apply_dish_mask_by_chamber`
  - or `<recordings_root>/logs/apply_dish_mask_by_chamber`

## Manual review details

`review_dish_masks.py`:

- Lists candidates, then launches the tuner for each Zarr one at a time.
- After you close the tuner, it prompts to continue or quit.
- Helpful flags:
  - `--only-present` to check the batch-applied ones
  - `--only-missing` to find gaps
  - `--chamber <name>` to focus on a dish type
  - `--full` or `--frame` for better fidelity
  - `--registry <sqlite>` to mark successful interactive saves in the registry
  - `--start`, `--limit` for partial passes

Examples:
```bash
# Only those that already have masks (verification pass)
scripts/py -m fisheye.utils.review_dish_masks /nvme1/recordings --recursive --only-present \
  --registry /nvme1/palette_registry.sqlite

# Only missing masks (fix pass)
scripts/py -m fisheye.utils.review_dish_masks /nvme1/recordings --recursive --only-missing \
  --registry /nvme1/palette_registry.sqlite
```

## Notes / best practices

- **Always review** after batch apply—small offsets matter for downstream detection.
- For consistent results, review all recordings from a single `experimental_chamber` in one pass.
- If you re-run batch apply after manual fixes, use `--overwrite` cautiously or you’ll overwrite manual corrections.

## Related tools

- `src/fisheye/utils/check_recording_steps.py` — quick status table for which recordings have dish masks and other tuning steps.
- `src/fisheye/utils/apply_tuning_by_camera.py` — apply tuning by camera_id (not dish type).
