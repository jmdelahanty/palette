<!-- ARCHIVED 2026-07-04: dated point-in-time snapshot / spent work ticket, retained for history only. -->

# Orange A16 TensorRT Engine Handoff

Purpose: provide a copy/paste task for an Orange agent to build a
hardware-specific A16 TensorRT deployment engine from a Palette ONNX export.

Palette treats ONNX as the portable model artifact and TensorRT engines as
hardware/runtime-specific deployment artifacts. A local A6000 engine is useful
for Palette workstation smoke tests, but Orange A16 deployment should use an
engine built on A16 hardware.

## Prompt For Orange Agent

```text
Read-only first, then implement only if the repo/machine already has the needed
TensorRT tooling.

Context:
Palette now treats ONNX as the portable model artifact and TensorRT engines as
hardware/runtime-specific deployment artifacts. I need an Orange A16 high-effort
TensorRT detect engine built from a Palette-exported ONNX, suitable for Orange
inference on A16 hardware.

Goal:
Build an Orange/A16 FP16 high-effort TensorRT engine from this ONNX:

ONNX: <PASTE_ONNX_PATH>
ONNX sha256: <PASTE_ONNX_SHA256>
Palette run_id: <PASTE_RUN_ID>
NMS contract: conf=<CONF>, iou=<IOU>, topk=<TOPK>
Expected input: images FP32 1x3x640x640
Expected outputs: num_dets, bboxes, scores, labels

Orange preprocessing remains:
Mono/luma source frame
  -> letterbox resize to 640x640
  -> padding value 114
  -> divide by 255.0
  -> replicate luma into B, G, R planes
  -> planar NCHW FP32 tensor

Build requirements:
- Build on an NVIDIA A16 GPU, not A6000/L4.
- Use TensorRT 10.0.1 if available.
- Use FP16.
- Use high-effort build knobs:
  --builderOptimizationLevel=5
  --avgTiming=32
  --profilingVerbosity=detailed
- Use an explicit A16 device, e.g. --device=<A16_DEVICE_ID>.
- Preserve EfficientNMS_TRT in-engine behavior.
- Do not change Orange runtime code unless needed for validation.

Suggested trtexec shape:
trtexec \
  --device=<A16_DEVICE_ID> \
  --onnx=<ONNX_PATH> \
  --saveEngine=<OUTPUT_ENGINE_PATH> \
  --fp16 \
  --builderOptimizationLevel=5 \
  --avgTiming=32 \
  --profilingVerbosity=detailed

Deliverables:
1. Engine file path.
2. Engine sha256.
3. Full trtexec command used.
4. Full trtexec log path.
5. TensorRT manifest JSON containing:
   - schema/version
   - Palette run_id
   - source ONNX path and sha256
   - engine path and sha256
   - precision
   - input/output contract
   - NMS conf/iou/topk
   - TensorRT version
   - CUDA version
   - selected GPU name/id/UUID/compute capability/SM count/memory
   - builderOptimizationLevel=5
   - avgTiming=32
   - profilingVerbosity=detailed
   - command
6. Standalone benchmark summary from trtexec if available.
7. Orange app-level smoke/validation summary if feasible:
   - cameras tested
   - frame count
   - steady detect p95
   - steady YOLO total p95
   - infer p95
   - drops/gaps/errors
8. Suggested Palette registration command:
   scripts/py -m fisheye.utils.register_model_deployment_artifact \
     --run-id <RUN_ID> \
     --manifest-path <A16_TRT_MANIFEST> \
     --deployment-runtime orange \
     --target-hardware-class A16 \
     --status candidate \
     --apply

Important:
- Do not mark the artifact preferred unless Orange validation passes.
- If validation passes, explain whether it is safe to promote from candidate to
  validated or preferred.
- Do not overwrite the existing default engine until validation is complete.
```

## Palette SCP Template

Run from the Palette workstation:

```bash
ONNX=/nvme1/models/detect/detect_all_available_detect_training_v004/detect_all_available_detect_training_v004_yolo11n_trt_20260520/exports/onnx/detect_all_available_detect_training_v004_yolo11n_trt_20260520.onnx
MANIFEST=/nvme1/models/detect/detect_all_available_detect_training_v004/detect_all_available_detect_training_v004_yolo11n_trt_20260520/exports/onnx/detect_all_available_detect_training_v004_yolo11n_trt_20260520.onnx.manifest.json

scp "$ONNX" "$MANIFEST" <orange_user>@<orange_host>:/home/jeremy/orange_data/detect/
```

Verify on the Orange machine:

```bash
sha256sum /home/jeremy/orange_data/detect/detect_all_available_detect_training_v004_yolo11n_trt_20260520.onnx
```

## Palette Registration After Handoff

After the Orange agent returns the engine manifest, register it in Palette:

```bash
scripts/py -m fisheye.utils.register_model_deployment_artifact \
  --registry /nvme1/palette_registry.sqlite \
  --run-id detect_all_available_detect_training_v004_yolo11n_trt_20260520 \
  --manifest-path <A16_TRT_MANIFEST_FROM_ORANGE> \
  --deployment-runtime orange \
  --target-hardware-class A16 \
  --status candidate \
  --apply
```

If Orange validation passes and this should become the selected deployment
engine, re-register with:

```bash
scripts/py -m fisheye.utils.register_model_deployment_artifact \
  --registry /nvme1/palette_registry.sqlite \
  --run-id detect_all_available_detect_training_v004_yolo11n_trt_20260520 \
  --manifest-path <A16_TRT_MANIFEST_FROM_ORANGE> \
  --deployment-runtime orange \
  --target-hardware-class A16 \
  --status validated \
  --apply
```

Only use `--status preferred` after explicitly deciding that this engine should
replace the current Orange default.
