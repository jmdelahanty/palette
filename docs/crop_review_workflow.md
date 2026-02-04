# Crop Review Workflow

This note captures how to inspect ROI crops and record approvals for training
data.

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

## Batch review

```
python -m fisheye.utils.review_crops /nvme1/recordings --recursive
```

Use `--file-list` to review a specific subset.

## Generating review lists

To collect crops missing approval into a file list:

```
python -m fisheye.utils.generate_review_list /nvme1/recordings \
  --recursive \
  --stage crop \
  --review-state missing \
  --output crop_review_list.txt
```

## Keypoint review (batch)

```
python -m fisheye.utils.review_keypoints_batch /nvme1/recordings --recursive --manual
```

Use `--retune` or `--audit` to switch modes.

## Notes on contrast

The crop viewer locks contrast to 0–255 for grayscale crops to avoid per-frame
auto-scaling that can make some frames appear brighter/dimmer even when the
pixel values are identical.
