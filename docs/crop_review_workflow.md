# Crop Review Workflow

This note captures how to inspect ROI crops and record approvals for training
data.

Registry-backed crop review/status planning is tracked in:
`docs/crop_review_registry_todo.md`.

## Quick review (single recording)

```
python -m fisheye.visualization.visualize_crops /path/to/recording.zarr
```

Keys:
- Left/Right: navigate crops
- `a`: set `crop_review_status` to approved (default manual/training)
- `n`: mark `needs_review`
- `r`: mark `rejected`
- `p`: mark `pending`
- `u`: cycle `intended_use` (`training` ↔ `full_recording`) for subsequent writes

## Batch review

```
python -m fisheye.utils.review_crops /nvme1/recordings --recursive
```

Use `--file-list` to review a specific subset.
Use `--review-intended-use full_recording` (plus optional `--review-state`,
`--review-method`, `--reviewer`, `--review-notes`) to forward review defaults to
each launched viewer.

## Generating review lists

To collect crops missing approval into a file list:

```
python -m fisheye.utils.generate_review_list /nvme1/recordings \
  --recursive \
  --stage crop \
  --review-state missing \
  --output crop_review_list.txt
```

By default, crop review lists include only crop runs with
`status=completed`. To include other run states (for example debugging),
set `--crop-run-status any`.

## Keypoint review (batch)

```
python -m fisheye.utils.review_keypoints_batch /nvme1/recordings --recursive --manual
```

Use `--retune` or `--audit` to switch modes.

## Notes on contrast

The crop viewer locks contrast to 0–255 for grayscale crops to avoid per-frame
auto-scaling that can make some frames appear brighter/dimmer even when the
pixel values are identical.
