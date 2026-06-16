# Native TensorRT Inference TODO

Last verified: 2026-06-05

## Goal

Add a Palette-owned TensorRT inference backend for detection/keypoint/mask
models, instead of relying on Ultralytics `.engine` inference as the production
runtime.

Ultralytics remains useful for training, PyTorch inference, ONNX export, and
quick compatibility checks. It should not be the only TensorRT runtime boundary
because `.engine` inference through Ultralytics requires the Python TensorRT
package and may attempt dependency auto-installation unless explicitly disabled.

## Current Finding

A local A6000 keypoint TensorRT engine was built successfully from the 5-keypoint
pose model:

```text
run_id: pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2
engine: /nvme1/models/pose/pose_all_registry_reviewed_v2_keypoints_20260520_v001/pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2/exports/tensorrt/pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2_fp16.engine
profile: images min=1 opt=1024 max=1024, 512x512
precision: fp16
builder_optimization_level: 0
GPU: NVIDIA RTX A6000, compute capability 8.6
```

The engine output contract is raw YOLO pose output:

```text
output0: FLOAT, shape=(batch, 20, 5376)
```

This means the runtime must implement YOLO pose postprocessing outside the
engine:

- decode box/object/keypoint predictions,
- apply confidence filtering,
- apply NMS or otherwise select the best pose per ROI,
- map keypoints back into ROI/image/normalized coordinate spaces,
- preserve Palette's existing keypoint run arrays and attrs.

This engine is not an EfficientNMS/postprocessed engine.

## Design Decisions

- **Native runtime boundary**: Palette should own TensorRT execution and
  postprocessing for batch analysis pipelines.
- **Ultralytics guard**: any path that still imports Ultralytics should set or
  require `YOLO_AUTOINSTALL=false` by default to prevent hidden dependency
  mutation.
- **Dependency boundary**: native Python TensorRT inference requires importable
  TensorRT Python bindings. `trtexec` can build/benchmark engines without those
  bindings, but Palette cannot decode model outputs in Python without either
  Python TensorRT bindings or a separate C++ runtime wrapper.
- **Backend selection**: expose TensorRT as an explicit backend, not an implicit
  behavior triggered by a `.engine` suffix.
- **Parity first**: do not run full production batches until a small PT-vs-TRT
  parity smoke passes on fixed ROI-cache inputs.

## Proposed CLI/API Shape

Detection/keypoint inference should accept a backend flag:

```bash
scripts/py -m fisheye.detection.detect_keypoints_yolo \
  /path/to/analysis.zarr \
  --model /path/to/model.engine \
  --inference-backend native-trt \
  --roi-cache-manifest /path/to/cache.flat_roi_cache.json \
  --input-mode tensor \
  --batch-size 1024 \
  --imgsz 512
```

Candidate backend values:

- `ultralytics`: current PyTorch/Ultralytics path, accepts `.pt` and possibly
  other Ultralytics-supported artifacts.
- `native-trt`: Palette-owned TensorRT execution and Palette-owned YOLO
  postprocessing.
- `auto`: allowed only after parity coverage is strong; until then prefer
  explicit backend selection.

## Implementation Checklist

### Phase 0: Safety Guard

- Add a startup guard in Palette Ultralytics entry points:
  - set `YOLO_AUTOINSTALL=false` before importing Ultralytics when possible, or
  - fail clearly if `YOLO_AUTOINSTALL` is true and a `.engine` path is passed.
- Add a test or CLI smoke proving `.engine` inference does not trigger
  dependency mutation.
- Document that installs of TensorRT Python bindings are explicit environment
  work, not automatic runtime behavior.

### Phase 1: TensorRT Runtime Loader

- Add `fisheye.inference.tensorrt_runtime` or equivalent.
- Load serialized engine and inspect:
  - TensorRT version,
  - input tensor names,
  - output tensor names/shapes,
  - dynamic profile min/opt/max shapes,
  - dtype and device memory requirements.
- Provide a small API:
  - `load_engine(path)`,
  - `infer(batch_tensor_or_numpy, input_name="images")`,
  - `close()`.
- Support dynamic batch sizes up to the engine profile max.
- Record runtime metadata in run attrs:
  - engine path/sha256,
  - manifest path/sha256,
  - selected profile,
  - TensorRT version,
  - GPU name/UUID/compute capability,
  - batch size and input shape.

### Phase 2: YOLO Pose Postprocessing

- Implement postprocessing for raw YOLO pose output `(B, C, anchors)`.
- Confirm channel layout for the current 5-keypoint model:
  - bbox channels,
  - object/class confidence channels,
  - keypoint x/y/conf channels.
- Match Ultralytics semantics closely enough for scientific parity:
  - confidence threshold,
  - IoU threshold,
  - max detections,
  - best detection per ROI selection.
- Return the same per-ROI intermediate fields consumed by
  `detect_keypoints_yolo`:
  - keypoints in ROI coordinates,
  - keypoint confidence,
  - pose bbox in ROI coordinates,
  - detection confidence,
  - success flag.

### Phase 3: Integration With Existing Keypoint Writer

- Refactor `detect_keypoints_yolo` so the batch loop delegates prediction to a
  backend object.
- Keep crop/ROI-cache reading, ROI override handling, output array creation,
  lineage copying, and registry sync unchanged.
- Add backend-specific timing buckets:
  - `input_prepare`,
  - `trt_h2d`,
  - `trt_execute`,
  - `trt_d2h`,
  - `postprocess`,
  - `result_write`.
- Keep `source_roi_cache_*` attrs and timing profile attrs unchanged so existing
  consumers continue to work.

### Phase 4: Parity Tests And Smokes

- Build a deterministic 32-ROI smoke from a flat ROI cache.
- Run PyTorch/Ultralytics `.pt` inference and native TRT inference on identical
  input tensors.
- Compare:
  - success count,
  - selected detection confidence,
  - bbox ROI coordinates,
  - keypoint ROI coordinates,
  - heading derived from keypoints.
- Define tolerances separately for FP16 numeric drift and postprocess ordering.
- Add a Zarr-writing smoke on one small/temporary run name only after
  non-mutating parity passes.

### Phase 5: Registry And Model Resolution

- Teach model resolution to distinguish:
  - trained `.pt` model,
  - ONNX artifact,
  - TensorRT export row,
  - target-specific deployment artifact.
- For native TRT keypoints, resolve from `tensorrt_models` or
  `model_deployment_artifacts`, not from `training_models`.
- Refuse incompatible engines:
  - wrong task type,
  - missing/unknown output contract,
  - unsupported input shape,
  - unsupported TensorRT version,
  - profile max batch below requested batch size.

## Open Questions

- Should detection TensorRT engines continue to use EfficientNMS in-engine while
  keypoint TensorRT engines use raw-output postprocessing, or should both use a
  unified raw-output path?
- Do we want a C++ runtime wrapper for TensorRT inference to avoid maintaining
  Python TensorRT bindings in `palette-py311`?
- Should TensorRT runtime engines be built with CUDA Graph benchmarking enabled
  for `trtexec` reports, even though CUDA Graph is a runtime measurement option
  rather than an engine-build accelerator?
- Should high-effort engines be built only for final deployment hardware, while
  local workstation engines use low-effort builder settings for fast iteration?

## Immediate Next Slice

1. Add the Ultralytics autoinstall guard for `.engine` paths.
2. Add a minimal TensorRT import/runtime probe utility:

```bash
scripts/py -m fisheye.utils.check_tensorrt_engine \
  /path/to/model.engine \
  --manifest /path/to/model.tensorrt.manifest.json
```

3. Implement non-mutating native inference on one batch of cached ROIs.
4. Add PT-vs-TRT parity reporting before writing any Zarr keypoint run.
