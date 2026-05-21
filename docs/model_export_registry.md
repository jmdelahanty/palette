# Model Export Registry (V2 Design Notes)

## Why this exists
We already record:
- training runs in `training_runs`
- trained detector artifacts in `training_models`
- ONNX exports in `onnx_models`
- TensorRT exports in `tensorrt_models`
- hardware/runtime-specific deployment artifacts in
  `model_deployment_artifacts`

This is a strong foundation, but most deployment compatibility fields still live in JSON
manifests or `metadata_json`. That makes frequent questions harder than they should be:

- "Which FP16 TensorRT engines support batch >= 8 at 640x640?"
- "Which ONNX exports were generated at opset 11?"
- "Which engines were built on TRT 10.x for compute capability 8.6?"

Goal: keep JSON as full provenance, while promoting a small set of high-value fields into
typed columns for fast filtering.

For trained-model input shape normalization, see
[`model_input_shape_registry_design.md`](model_input_shape_registry_design.md).
That design extends this export-focused registry surface by making
`training_models` own the trained `.pt` artifact input contract while keeping
`onnx_models` and `tensorrt_models` responsible for export-specific input
contracts.


## Current registry shape (canonical)
- `training_runs`: run lifecycle + config/manifest/model/metrics links.
- `training_models`: one row per detector training run (`run_id` primary key).
- `onnx_models`: one row per run's ONNX artifact (`run_id` primary key).
- `tensorrt_models`: one row per `(run_id, precision)`.
- `model_deployment_artifacts`: one row per target-specific deployable
  artifact, such as an Orange/A16 FP16 TensorRT engine built from a Palette
  ONNX export.

Legacy:
- `model_exports` exists for compatibility/backfill history and should not be treated as the
  long-term query surface.

Operationally, the preferred detection-training path is:

1. Build a registry-selected training config and manifest.
2. Train with `fisheye.training.train_detection` while passing `--manifest`,
   `--set-id`, and `--registry`.
3. Keep registry logging enabled.
4. Use `--export-trt` for deployment builds; this implies ONNX export.

The run directory remains the complete artifact bundle. The registry is the
fast query/index surface, not the only source of truth. The JSON manifests
written beside ONNX and TensorRT artifacts retain full build provenance,
including paths, hashes, input shape, output contract, export command, and build
environment.

## Deployment Artifacts

TensorRT engines are compiled deployment artifacts, not portable model
descriptions. A trained Palette checkpoint may produce one ONNX model, but that
ONNX can legitimately have multiple TensorRT engines:

- an A6000 engine for local workstation smoke tests,
- an A16 engine for Orange acquisition/runtime deployment,
- an L4 engine for cluster batch inference,
- an INT8 candidate for a future calibration experiment.

`tensorrt_models` is intentionally still a compact export inventory keyed by
`(run_id, precision)`. It is not sufficient as the deployment selector because
two FP16 engines can share the same trained model but target different GPUs.
The companion table `model_deployment_artifacts` owns this target-specific
identity.

Important fields:

- `artifact_id`: stable deployment artifact identifier.
- `run_id`: trained model/training run that owns the deployment artifact.
- `source_onnx_run_id`, `source_onnx_path`, `source_onnx_sha256`: portable
  model artifact used as the build input.
- `artifact_kind`: usually `tensorrt_engine`.
- `deployment_runtime`: e.g. `orange`.
- `target_hardware_class`: e.g. `A16`, `L4`, or `A6000`.
- `target_gpu_name`, `target_compute_capability`: concrete build/runtime GPU
  identity when available.
- `precision`, `engine_path`, `engine_sha256`, `manifest_path`,
  `manifest_sha256`: deployable engine identity.
- `status`: `candidate`, `validated`, `preferred`, or `deprecated`.
- `trtexec_path`, `trt_version`, `cuda_version`,
  `builder_optimization_level`, `avg_timing`, `profiling_verbosity`,
  `cuda_graph`: TensorRT build strategy and environment.
- `nms_conf`, `nms_iou`, `nms_topk`: baked detection NMS settings.
- `validation_summary_json`: app-level validation summary, such as Orange
  steady p95 latency, drop/gap/error counts, and validation recording IDs.
- `metadata_json`: full provenance payload; typed columns are only the fast
  query surface.

Registering an externally built Orange/A16 engine should use the deployment
artifact table rather than overwriting the local TensorRT export row:

```bash
scripts/py -m fisheye.utils.register_model_deployment_artifact \
  --registry /nvme1/palette_registry.sqlite \
  --run-id detect_all_available_detect_training_v004_yolo11n_trt_20260520 \
  --manifest-path /path/to/a16_engine.tensorrt.manifest.json \
  --deployment-runtime orange \
  --target-hardware-class A16 \
  --status candidate \
  --apply
```

For Orange production use, the preferred artifact should be the engine built on
the target hardware class, not a local workstation engine. A6000-built engines
remain useful for local smoke/review, but they should not be marked as the
preferred Orange/A16 deployment artifact.

## Current preferred detector baseline

As of 2026-05-16, the preferred detector baseline is:

```text
set_id: detect_all_available_detect_training_v003
run_id: detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1
task_type: detect
status: success
input_shape: [1, 3, 640, 640]
input_color_space: rgb
```

Artifact paths:

```text
best.pt:
/nvme1/models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/weights/best.pt

ONNX:
/nvme1/models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/exports/onnx/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1.onnx

TensorRT FP16:
/nvme1/models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/exports/tensorrt/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1_fp16.engine
```

Best validation checkpoint was epoch 41:

```text
precision: 0.9794
recall: 0.9787
mAP50: 0.9838
mAP50-95: 0.7539
```

This run reproduced the prior v002 baseline metrics exactly, but v003 is the
preferred operational baseline because it has a clean source-set identity and
successful registry rows in `training_runs`, `training_models`, `onnx_models`,
and `tensorrt_models`.

## Design principles
- Keep normalized artifact tables (`training_models`, `onnx_models`, `tensorrt_models`).
- Keep `metadata_json` and manifest files as full audit records.
- Promote only fields needed for high-frequency deployment filters.
- Make new columns nullable to preserve backward compatibility.


## Proposed queryable columns
These are additions to existing canonical tables.

### `onnx_models`
- `opset` INTEGER
- `input_shape` TEXT
- `img_h` INTEGER
- `img_w` INTEGER
- `max_batch` INTEGER
- `dynamic_shapes` INTEGER
- `file_size_bytes` INTEGER
- `exporter_torch_version` TEXT
- `exporter_cuda_version` TEXT
- `exporter_hostname` TEXT

### `tensorrt_models`
- `input_shape` TEXT
- `img_h` INTEGER
- `img_w` INTEGER
- `max_batch` INTEGER
- `dynamic_shapes` INTEGER
- `file_size_bytes` INTEGER
- `trt_version` TEXT
- `cuda_version` TEXT
- `compute_capability` TEXT
- `gpu_name` TEXT
- `gpu_uuid` TEXT
- `system_hostname` TEXT

### Optional later
- `output_contract_json` (or hash) if we want contract-level filtering in SQL.
- `plugins_hash` for plugin-compatibility search.


## Source of truth mapping
- ONNX manifest:
  - export settings (`opset`, `input_shape`, imgsz)
  - build environment (`torch`, `cuda`, host)
  - output contract
- TensorRT manifest:
  - precision + engine path/sha
  - build environment (`trt_version`, GPU identity, compute capability, host)
  - input/output contract reference
- Fallbacks:
  - infer from `metadata_json` and artifact paths when manifests are missing.


## Backfill strategy
1. Schema migration:
   - add nullable columns above to `onnx_models` and `tensorrt_models`.
2. Backfill command:
   - read each row's manifest if available, then metadata JSON fallback.
   - populate capability columns.
3. Integrity check:
   - extend `--check-integrity` to validate column/manifest consistency (warning-level first).


## Query examples (target state)
```sql
-- FP16 TRT engines for 640x640 with batch >= 8
SELECT run_id, path
FROM tensorrt_models
WHERE precision = 'fp16'
  AND img_h = 640
  AND img_w = 640
  AND max_batch >= 8;

-- ONNX exports built at opset 11
SELECT run_id, path
FROM onnx_models
WHERE opset = 11;

-- Engines compatible with a given deployment class
SELECT run_id, path, trt_version, compute_capability
FROM tensorrt_models
WHERE trt_version LIKE '10.%'
  AND compute_capability = '8.6';

-- Preferred Orange A16 deployment engines
SELECT run_id, artifact_id, engine_path, engine_sha256
FROM model_deployment_artifacts
WHERE deployment_runtime = 'orange'
  AND target_hardware_class = 'A16'
  AND status = 'preferred';
```


## Implementation plan
1. Add schema columns + indexes on common filters:
   - `onnx_models(opset, img_h, img_w)`
   - `tensorrt_models(precision, img_h, img_w, max_batch)`
   - `tensorrt_models(trt_version, compute_capability)`
2. Update writers:
   - populate columns at write-time in export/train flows.
3. Add backfill in maintenance CLI.
4. Add query CLI surface (`registry models ...`) or extend existing registry query tools.
5. Update docs and examples.


## Tradeoffs
- Pros:
  - Fast SQL filters for deployment decisions.
  - Less JSON parsing in operational tooling.
- Cons:
  - More schema upkeep when exporter metadata changes.
- Mitigation:
  - Keep JSON as canonical full detail and only promote stable, high-value fields.
