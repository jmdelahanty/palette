<!-- ARCHIVED 2026-07-04: documents eye-mask code deleted in commit 4a85e5d (eye-mask stage severance). Retained for history only; NOT current. Live replacement: docs/eye_subject_mask_unification_design.md. -->

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

## Output / Storage Policy

Recommended operational policy:

- For analysis inference, prefer `label_mode=union` when downstream refinement
  will assign left/right identity from keypoints.
- Keep mask outputs ROI-local rather than persisting full-frame masks.
- Treat probability semantics and physical storage encoding separately:
  - semantic value: probability in `[0,1]`
  - physical encoding: `float16` or quantized `uint8`

Recommended defaults:

- Analysis runs:
  - `label_mode=union`
  - `batch_size=256` on the current RTX A6000 analysis host
  - `mask_probs_chunk_rois=32`
  - `mask_probs_dtype=uint8`
  - perform ROI normalization on GPU after transfer rather than on CPU
- Higher-fidelity artifacts or workflows that need soft probabilities with less
  quantization:
  - keep `mask_probs_dtype=float16`
  - lower batch sizes may still be preferred when GPU memory is shared or
    stability headroom matters more than peak throughput

Encoding contract:

- `float16`/`float32`: direct probability storage
- `uint8`: linear quantization with decode rule `p = stored / 255`
- prefer `uint8` over `int8` for quantized probabilities because probability
  values are nonnegative and map naturally to `0..255`
- Writers should emit `probabilities_dtype` and `probabilities_encoding`
  attrs so downstream readers can normalize back to `[0,1]`

This lets analysis archives use smaller/faster probability storage without
changing the semantic contract seen by refinement, visualization, or training
export readers.

Observed benchmark outcome on the `2026-01-28T19-22-28Z_arena_1_DefaultScreen`
smoke archive:

- warm-cache `geometry_only` U-Net inference with
  `batch_size=256`, `mask_probs_chunk_rois=32`, `mask_probs_dtype=uint8`, and
  GPU-side normalization reached about `106s` wall time for `23,287` ROIs
- `sync_after_forward` is now the dominant timing stage, which means queued
  U-Net GPU compute is the main remaining bottleneck
- ROI reads and probability writes are no longer the limiting cost on this
  workload after the cache-layout and `uint8`/GPU-normalization changes

Operational implication:

- analysis-time optimization work should now focus on the inference backend /
  model compute path before pursuing more aggressive cache I/O changes

Decision rule:

- choose the physical storage dtype based on operational needs
  (write bandwidth, archive size, fidelity needs),
- keep the semantic contract explicit so readers always know whether a stored
  array represents direct float probabilities or quantized probabilities that
  must be decoded back to `[0,1]`.

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
