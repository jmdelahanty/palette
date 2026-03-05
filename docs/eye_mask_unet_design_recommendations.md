# Eye-Mask U-Net Design Recommendations

## Purpose

Capture a practical recommendation for the current eye-mask U-Net architecture and a low-risk upgrade path.

## Current Implementation

Code paths:

- `src/fisheye/segmentation/unet.py`
- `src/fisheye/segmentation/train_unet_eye_masks.py`

Current model:

- Custom lightweight U-Net (`UNetSmall`), not a built-in torchvision model.
- Encoder/decoder with skip connections.
- Two-conv blocks with `BatchNorm2d + ReLU`.
- Base width: `base_channels=32`.
- Output channels:
  - `1` for `label_mode=union`
  - `2` for `label_mode=lr`

## Assessment

This is a good baseline for eye-mask segmentation:

- Architecture is standard and reliable for dense mask prediction.
- Implementation is simple and maintainable.
- It is expressive enough for this task without adding unnecessary complexity.

Known tradeoffs:

- Memory usage is high at larger batch sizes.
- BatchNorm can become less stable as batch size is reduced.

## Recommended Path

### 1) Keep current model for baseline

First establish a reference run with the existing architecture and reasonable batch size.

Why:

- Fastest path to get a valid model.
- Gives a reference point for any future architecture changes.

### 2) If needed, apply this v2 sequence

Order changes one at a time for clean attribution:

1. Reduce width: `base_channels 32 -> 16`.
2. Replace normalization: `BatchNorm2d -> GroupNorm`.
3. Add light dropout (`p=0.1`) in bottleneck and optional decoder blocks.

Why this order:

- Width reduction is the largest memory/throughput lever.
- GroupNorm is usually more robust at smaller effective batch sizes.
- Dropout is optional regularization after core stability is confirmed.

## Suggested Experiment Matrix

Minimum set:

1. Baseline: `base=32`, BatchNorm, no dropout.
2. Memory-first: `base=16`, BatchNorm, no dropout.
3. Small-batch-stable: `base=16`, GroupNorm, no dropout.
4. Regularized: `base=16`, GroupNorm, dropout `0.1`.

Keep all other training settings fixed while comparing.

## Evaluation Criteria

Prioritize validation metrics:

- `val_dice` should increase.
- `val_loss` should decrease.
- Train/val gap should remain controlled (no obvious overfitting).

Secondary criteria:

- No OOM at target batch size.
- Acceptable epoch time.

## Implementation Notes

Likely edit points:

- `src/fisheye/segmentation/unet.py`
  - normalization layers in `_DoubleConv`
  - optional dropout insertion
- `src/fisheye/segmentation/train_unet_eye_masks.py`
  - `UNetSmall(..., base_channels=...)` instantiation

Treat each v2 change as a separate commit so regressions are easy to isolate.
