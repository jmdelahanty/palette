# Video Pixel And Model Input Contract
<!-- contract-meta
status: current
last_verified: 2026-07-06
purpose: Clarify the difference between persisted video/crop pixels and model-input tensors, especially PyNvVideoCodec luma versus NV12-to-RGB detection preprocessing.
-->

## Summary

Palette has two related but distinct contracts:

- Persisted pixel artifacts store source-aligned image data.
- Model-input tensors are runtime products derived from those persisted or
  decoded pixels for a specific model.

For Orange monochrome recordings, the preferred persisted ROI/crop pixel
contract is `pynvvc_luma_v1` / `orange_mono_pynvvc_luma_uint8_v1`: decoded
PyNvVideoCodec NV12 Y/luma plane, stored as `[N,H,W] uint8` before model-specific
resize, letterbox, channel replication, or normalization.

Detection is different today. Current YOLO detection inference uses
`pynvvc_nv12_rgb` as the correctness-oriented PyNvVideoCodec backend because the
existing detector expects RGB-like full-frame tensors and fixed-frame parity
favored NV12-to-RGB conversion over luma replication. The `pynvvc_luma_rgb`
detection path remains useful as a fast diagnostic variant, but it is not the
default correctness path for current detector runs.

## Current Contracts

### Crop, Keypoint, And Mask Inputs

Crop-derived stages consume ROI pixels. For new Orange mono training and cache
artifacts, the canonical persisted surface is:

```text
name: orange_mono_pynvvc_luma_uint8_v1
shape: [roi, roi_height, roi_width]
dtype: uint8
source: Orange camera MP4 decoded by PyNvVideoCodec
source_encoder_boundary: NV12
mono_semantics: camera intensity copied to NV12 Y plane; UV neutral 128
color_conversion: raw NV12 Y/luma plane crop; no RGB reconstruction
```

Encoded stream metadata has its own contract. Orange prepares NV12 for NVENC
(`Y` plane plus interleaved neutral `UV`), but FFmpeg/ffprobe may report decoded
HEVC frames as `yuv420p` (`Y`, `U`, and `V` planes). That is a memory-layout
difference after decode, not a semantic problem. The important stream VUI/range
metadata is `color_range`: future Orange encodes should tag these monochrome
full-range camera samples as `pc`/full range, not `tv`/limited range. Palette
records observed stream fields separately as `video_color_range`,
`video_color_space`, `video_color_transfer`, and `video_color_primaries`; model
input code should continue to rely on the explicit Orange mono8 full-range
pixel contract rather than letting an RGB decoder reinterpret the Y plane.

Model-specific tensorization happens later:

- read `[N,H,W] uint8` luma ROI pixels,
- apply an explicit reversible model-input transform,
- replicate luma to three channels when the model expects `3` channels,
- scale by `/255` and convert to the model dtype/layout.

This keeps crop caches and training Zarrs independent of a specific model input
shape, engine batch profile, or tensor layout.

### Reversible Model-Input Transforms

Native persisted crop pixels do not need to match a trained model's input size.
When they differ, inference must use a declared reversible transform and write
predictions back in native crop coordinates.

Current supported transform modes live in
`src/fisheye/shared/model_input_transform.py`:

```text
identity
  native crop size == model input size

pad_to_size
  center-pad native crops with zero-valued pixels to the model input size
  scale remains 1.0
  keypoints/boxes are mapped back by subtracting [pad_left, pad_top]
  mask logits/probabilities are cropped back before writing masks_roi/mask_probs_roi

auto
  identity when sizes match
  pad_to_size when a larger model input size is requested
```

For RedScare-style acquisition crop videos, this means native `384x384` crop
frames can be fed to a model trained at `512x512` by padding to `512x512`, then
writing keypoints and masks back to the native `384x384` crop coordinate system.
This is preferred over resizing for bootstrapping labels because it preserves
pixel scale. The prediction run must record `model_input_transform`,
`model_input_shape_hw`, and `native_roi_shape_hw`.

Smaller-input models are still a valid optimization target, but they should be
trained with the same explicit transform used at inference. Do not silently mix
native `348/384/512` crop images as if they were equivalent model inputs.
Either train at native size, train with a recorded resize/letterbox/pad policy,
or export separate model rows whose input contract records the intended shape
and transform.

### Detection Inputs

Detection consumes full video frames and immediately constructs YOLO input
tensors. Current production `detect_yolo` supports both PyNvVideoCodec paths:

- `pynvvc_nv12_rgb`: decodes NV12 and reconstructs RGB-like tensors from Y, U,
  and V planes using the limited-range BT.601 conversion implemented in
  `shared/pynvvc_luma_rgb.py`.
- `pynvvc_luma_rgb`: decodes the Y/luma plane, resizes it, and replicates luma
  into three channels.

The current default correctness path is `pynvvc_nv12_rgb` when CUDA,
PyNvVideoCodec, and resize dimensions are available. This is not because
Orange mono videos need chroma for biological signal. It is because the current
detector was trained and validated against RGB-like preprocessing, and the
recorded parity check showed lower box/score drift for NV12-to-RGB than for
luma replication.

## Preferred Future Direction

Best practice for this codebase is to converge detection, crop, keypoint, and
mask inputs onto the same source-aligned luma contract, but only after detector
training and validation make that true.

The preferred future detector contract is:

```text
source pixels: pynvvc_luma_v1 / orange_mono_pynvvc_luma_uint8_v1
preprocess: luma -> resize/letterbox -> replicate to 3 channels -> /255
tensor: FP32 or engine-selected dtype, NCHW, model-specific input size
```

That would make detection and downstream ROI stages share one pixel source and
one mental model: mono camera intensity enters the pipeline as decoded NV12
luma, while model-specific preprocessing is explicit and artifact-specific.

Do not switch existing production detection to luma replication solely for
architectural neatness. The migration should require:

- a detector trained on luma-replicated inputs or a validated fine-tune from the
  current detector,
- fixed-frame and full-run parity/quality comparisons against the current
  `pynvvc_nv12_rgb` detector,
- persisted model metadata that records the source pixel contract and
  preprocessing transform,
- training exporters that refuse to mix incompatible pixel contracts,
- deployment/runtime code that applies the same luma-replicated preprocessing
  used during training.

## Practical Rule

Until a luma-trained detector is accepted:

- Use `pynvvc_luma_v1` for new persisted ROI crops, keypoint training, and mask
  training artifacts from Orange monochrome recordings.
- Use `pynvvc_nv12_rgb` for production full-frame YOLO detection inference.
- Use `pynvvc_luma_rgb` detection only for controlled diagnostics, ablations, or
  a detector explicitly trained and validated for that preprocessing path.
- Treat normalized CHW tensors, RGB replication, and TensorRT optimization
  profiles as model/runtime artifacts, not persisted crop-cache artifacts.
- Treat padding/resizing/letterboxing as model-input transforms. Prediction
  outputs remain in native crop coordinates unless a writer explicitly declares
  a different output coordinate space.

## Current Metadata Audit

The 2026-06-05 pixel-contract audit found no persisted image surface labeled
`pynvvc_nv12_rgb`; that backend remains a detection inference tensorization path
in current code. The audit did find substantial under-labeling of persisted raw
video images, crop runs, and merged training exports.

Use `docs/diagnostics/pixel_contract_audit_2026-06-05.md` as the implementation
checklist for making this contract enforceable.

## References

- `docs/geometry_only_crop_workflow_cache_design.md`
- `docs/diagnostics/pixel_contract_audit_2026-06-05.md`
- `docs/training_crop_representation_migration.md`
- `docs/subject_mask_training_artifact_contract.md`
- `docs/detect_decode_backend_benchmark_todo.md`
- `src/fisheye/shared/pynvvc_luma_rgb.py`
- `src/fisheye/detection/detect_yolo.py`
