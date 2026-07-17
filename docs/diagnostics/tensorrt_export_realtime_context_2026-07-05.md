# TensorRT export path — purpose, status, and decision rule (2026-07-05)

status: active — context note; supersedes the "half-built acceleration infra" reading
of the TensorRT surface in the 2026-07-01 review's lower-severity list.

## The correction that motivated this note

A 2026-07-05 assessment initially read the TensorRT export surface as dead weight
(export side exists, offline inference never consumes engines, trtexec regexes broken
without anyone noticing) and suggested it as a deletion candidate under the subtraction
rule. **That reading was wrong about purpose.** Maintainer clarification:

> The TensorRT engines are built so the maintainer can run **realtime, closed-loop
> behavior from the acquisition library**. The export path is load-bearing for that
> use — it is not an abandoned offline-pipeline optimization.

**Standing instruction: do not delete or "clean up" the ONNX/TensorRT export path**
(`training/export_onnx.py`, `training/onnx_to_tensorrt.py`, `training/export_shared.py`,
export hooks in `train_detection.py`/`train_pose.py`/`train_keypoints.py`, and the
deployment-artifact registration in the registry). Fix it, don't fold it.

## What exists today (verified against `sun` @ `8af2443`)

- **Export side (load-bearing, realtime):** `.pt → ONNX → TensorRT engine` via
  `export_onnx.py` and `onnx_to_tensorrt.py`, shared helpers in `export_shared.py`,
  engines registrable as model deployment artifacts
  (`utils/register_model_deployment_artifact.py`, registry tables).
- **Offline pipeline (PyTorch, deliberate):** `detection/detect_yolo.py:805` loads the
  checkpoint via `YOLO(model_path)` and runs PyTorch with FP16 (`model.half()`,
  `detect_yolo.py:823`). Offline batch does **not** consume engines, and that is the
  intended split, not drift.
- **Known defect:** the trtexec version-parsing regexes in
  `training/export_shared.py:70,73,78` contain double-escaped `\\s`/`\\d` in raw
  strings, so they can never match and `tensorrt_version` records `None`. Fix is
  assigned (Slice D, `agents_todo/brief_review_remediation_wave_2026-07-05.md`).
  Consequence until fixed: registered engine artifacts under-record the TensorRT
  version they were built with — exactly the metadata that matters for engine
  compatibility (see risks below).

## The intended split

| Path | Runtime | Why |
|---|---|---|
| Realtime closed-loop (acquisition library) | TensorRT engine | Latency-bound; per-frame inference must keep up with acquisition; TRT is the right tool |
| Offline batch pipeline (bsub stages) | PyTorch `.pt` + FP16 | Throughput-bound; scaled horizontally across cluster jobs; decode/IO/zarr typically dominate, not model forward |

Decision rule (from the 2026-07-05 assessment, unchanged by the correction):
TensorRT is a latency tool. Use it for the realtime path. Adopt it for offline batch
only if (a) per-stage profiling proves a stage is forward-pass-bound after the cheap
ladder (FP16 verified engaged, batch size pushed until decode saturates,
`torch.compile`), and (b) GPU-hours are actually the constrained resource. Neither has
been demonstrated; no offline TRT adoption is planned.

## Risks to manage on the realtime path (open, not yet enforced)

These inherit the repo's silent-wrong-data and provenance discipline; none are done yet:

1. **Engine identity in provenance.** A `.engine` is a derived artifact pinned to GPU
   architecture + TensorRT version. Registry deployment-artifact rows already record
   content hashes, and Palette now has a read-only helper to verify those hashes against
   the on-disk engine file. Realtime-side enforcement is still external to this repo: the
   acquisition library must check the engine it actually loads against the registered
   content hash and build context (source `.pt` hash, ONNX opset, TRT version, GPU arch,
   precision flags).
2. **Numeric parity.** FP16/INT8 engines shift outputs relative to the `.pt`. Realtime
   detections that feed closed-loop behavior AND later analysis must be comparable to
   the offline path — `docs/diagnostics/realtime_offline_detection_comparison_design_2026-06-17.md`
   is the relevant design. A parity census (same recording, `.pt` vs `.engine`,
   detections within stated tolerance) should gate any engine promoted for closed-loop
   use.
3. **Engine/driver rebuild matrix.** Engines silently fail or refuse to load across
   TRT/driver upgrades and GPU models. Recording (1) makes staleness detectable instead
   of a mystery at acquisition time.

## Pointers

- Realtime/offline comparison design: `docs/diagnostics/realtime_offline_detection_comparison_design_2026-06-17.md`
- Provenance enforcement (epoch gate, artifact-hash follow-up): `docs/archive/provenance_finalization_enforcement_design.md`, `docs/archive/provenance_enforcement_roadmap.md`
- Regex fix + hygiene assignment: `agents_todo/brief_review_remediation_wave_2026-07-05.md` (Slice D)
