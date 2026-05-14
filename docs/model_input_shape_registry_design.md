# Model Input Shape Registry Design
<!-- contract-meta
status: design
last_verified: 2026-05-13
purpose: Design for normalizing model input-shape metadata in the Palette registry.
-->

## Summary

Palette should normalize model input-shape metadata in the registry. The current
registry already records input shape for exported artifacts in `onnx_models` and
`tensorrt_models`, but the trained model artifact in `training_models` does not
have typed shape columns. For current detection models, the trained shape is
recoverable from JSON fields such as `training_runs.final_metrics_json.imgsz_h`
and `imgsz_w`, but consumers should not need to parse JSON or infer model shape
from a previous inference run.

The recommended design is additive:

- `training_models` owns the trained `.pt` artifact input contract.
- `onnx_models` owns ONNX export input contract.
- `tensorrt_models` owns TensorRT engine input contract.
- A query view exposes all model artifacts with a consistent input-shape surface.

This keeps full provenance in JSON while promoting the small set of stable fields
that downstream tooling needs for fast, reliable queries.

## Problem

Training-data and inference tools need to answer simple questions:

- Which registered detector expects 640x640 input?
- Can this sampled training Zarr use `raw_video/images_ds`, or must it read
  `raw_video/images_full` and resize?
- Which pose model was trained at a different input size than its ONNX export?
- Which exported engine is compatible with a deployment target?

Today those answers are uneven:

- `onnx_models` and `tensorrt_models` already have `input_shape`, `img_h`,
  `img_w`, `max_batch`, and `dynamic_shapes`.
- `training_models` stores trained artifact metadata, but does not expose typed
  input-shape columns.
- `training_runs.final_metrics_json` can contain `imgsz_h` and `imgsz_w`, but
  JSON parsing should be a fallback, not the normal query path.
- `detect_model_performance_latest` records inference dimensions, but those are
  a run-time execution fact, not the trained model input contract.

## Non-Goals

- Do not encode raw recording frame shape here. Raw frame shape belongs to the
  recording/Zarr metadata.
- Do not replace full training manifests or metrics JSON. The registry columns
  are a query surface, not the full audit record.
- Do not infer biology or label semantics from shape. Shape only describes the
  tensor contract expected by a model artifact.
- Do not make inference-run dimensions authoritative for a model artifact. They
  are useful validation evidence but not the source of truth.

## Canonical Ownership

`training_models` should own trained model input metadata because it is the row
that represents the trained artifact. Export tables may agree with it, but they
are not the source of truth for the `.pt` model.

Recommended ownership:

| Table | Artifact | Input-shape responsibility |
| --- | --- | --- |
| `training_models` | trained model, usually `.pt` | canonical trained artifact input contract |
| `onnx_models` | ONNX export | export-specific input contract |
| `tensorrt_models` | TensorRT engine | engine-specific input contract |
| `detect_performance` | inference execution | observed inference size for a run |

## `training_models` Columns

Registry migration 49 adds nullable columns to preserve backward compatibility:

| Column | Type | Meaning |
| --- | --- | --- |
| `input_shape` | `TEXT` | JSON list for the network input shape, usually `[1, C, H, W]`. |
| `input_layout` | `TEXT` | Tensor layout such as `NCHW`. |
| `input_channels` | `INTEGER` | Network channel count. For YOLO RGB-style models this is normally `3`. |
| `img_h` | `INTEGER` | Model input image height in pixels. |
| `img_w` | `INTEGER` | Model input image width in pixels. |
| `max_batch` | `INTEGER` | Known or assumed maximum batch for the artifact, if meaningful. |
| `dynamic_shapes` | `INTEGER` | Boolean integer. `1` if dynamic shape is explicitly supported. |
| `input_dtype` | `TEXT` | Expected tensor dtype when known, for example `float32`. |
| `input_color_space` | `TEXT` | Expected input color interpretation when known, for example `rgb` or `gray`. |
| `input_shape_source` | `TEXT` | Where the normalized fields came from. |
| `input_shape_status` | `TEXT` | `explicit`, `inferred_from_imgsz`, `export_backfill`, `unknown`, or `conflict`. |

For YOLO detector and pose models, `input_shape=[1, 3, H, W]`,
`input_layout=NCHW`, `input_channels=3`, and `input_color_space=rgb` are the
expected registry-level network contract even when source videos or training
Zarr arrays are grayscale. The grayscale-to-network-input conversion remains an
inference/preprocessing responsibility and should be documented separately.

## Shape Source Precedence

Writers and backfill tools should record where shape information came from.
Recommended precedence:

1. Explicit normalized fields passed by the training writer.
2. Training report or final metrics fields such as `effective_imgsz`,
   `imgsz_h`, and `imgsz_w`.
3. Training metadata or manifest fields such as `training_params.imgsz`.
4. Matching export metadata in `onnx_models` or `tensorrt_models`.
5. Optional model inspection, if the model file is accessible and dependencies
   are available.
6. Unknown.

If multiple sources disagree, do not silently pick one. Record the best
available value, mark `input_shape_status='conflict'`, and preserve the conflict
details in `metadata_json` or a maintenance report.

## Query View

Migration 49 adds the `model_input_shapes` view, which exposes one row per model
artifact with consistent columns:

| Column | Meaning |
| --- | --- |
| `artifact_kind` | `training`, `onnx`, or `tensorrt`. |
| `run_id` | Training run identifier. |
| `set_id` | Training dataset/set identifier. |
| `task_type` | `detect`, `pose`, `subject_mask`, etc. |
| `artifact_path` | Path to the model artifact. |
| `artifact_sha256` | Artifact checksum when available. |
| `input_shape` | JSON list shape. |
| `input_layout` | Tensor layout. |
| `input_channels` | Network channel count. |
| `img_h` | Input height. |
| `img_w` | Input width. |
| `max_batch` | Max batch when known. |
| `dynamic_shapes` | Dynamic-shape support. |
| `input_dtype` | Expected dtype when known. |
| `input_color_space` | Expected input color interpretation. |
| `input_shape_source` | Source used to populate shape fields. |
| `input_shape_status` | Confidence/status classification. |

The view should not collapse trained and exported artifacts into one ambiguous
row. A trained `.pt` model and its ONNX export are related but distinct
artifacts and may legitimately have different deployment properties.

## Consumer Rules

Consumers should treat normalized shape fields as the first query surface.

For sampled training Zarr label seeding:

1. Resolve the model row from the registry.
2. Query `model_input_shapes` for `artifact_kind='training'` unless an exported
   artifact is explicitly requested.
3. If a sampled array exactly matches `img_h` and `img_w`, prefer that array.
4. If no sampled array matches, read `raw_video/images_full` and let the
   inference path perform its documented resize or letterbox transform.
5. Preserve the chosen source array and resize path in run provenance.

For deployment tools:

1. Query `artifact_kind='onnx'` or `artifact_kind='tensorrt'`.
2. Filter by `img_h`, `img_w`, `max_batch`, `dynamic_shapes`, precision, and
   hardware compatibility columns.
3. Do not assume the trained artifact shape is identical to every exported
   artifact unless the registry records that agreement.

## Backfill Behavior

Migration 49 backfills existing rows by parsing `final_metrics_json` and
`metadata_json`, then uses matching `onnx_models` and `tensorrt_models` only as
fallback evidence. If training metadata and export metadata disagree, the
training row keeps the training-derived value, marks
`input_shape_status='conflict'`, and stores conflict details in `metadata_json`.

A later registry maintenance pass should add explicit checks for missing or
conflicting model input shapes on successful model rows.

## Writer Updates

Writer surface status:

- `PaletteRegistry.record_training_model(...)`: accepts normalized input-shape
  fields and persists them into `training_models`.
- Detection training writer: pass effective YOLO `imgsz` as
  `input_shape=[1, 3, H, W]`.
- Pose training writer: records effective YOLO pose `imgsz` in final metrics,
  which `record_training_model(...)` normalizes with the same layout convention.
- Subject-mask training writer: pass its actual model input contract once the
  subject-mask registry surface is finalized.
- Export writers: keep using the existing export-specific columns and preserve
  artifact-specific values.

## Validation Plan

Focused tests cover the implemented registry behavior:

- Registry migration test: `training_models` has the new nullable columns.
- Training writer test: detector and pose training registration write
  `input_shape`, `img_h`, `img_w`, `input_layout`, and `input_channels`.
- Backfill unit test: `imgsz_h/imgsz_w` in `final_metrics_json` resolves to
  `[1, 3, H, W]` with `input_shape_status='inferred_from_imgsz'`.
- Conflict test: training metrics and export metadata disagree, and the
  backfill reports `conflict` instead of silently overwriting.
- Query view test: `model_input_shapes` returns separate rows for trained, ONNX,
  and TensorRT artifacts.
- Consumer smoke: sampled label seeding chooses `raw_video/images_ds` when its
  array shape exactly matches the selected model.

## Open Decisions

- Whether to implement the schema for all task types immediately or start with
  detector and pose rows. Recommendation: make the schema general, then backfill
  detector and pose first.
- Whether `input_color_space` should be required for all models. Recommendation:
  nullable initially, but training writers should populate it when known.
- Whether model file inspection should be part of normal backfill. Recommendation:
  use it as an optional diagnostic only; registry JSON and writer-provided
  values should be the normal path.

## Implementation Status

The registry implementation slice is `schema_version` migration 49
(`model_input_shape_registry`). The migration adds nullable input-shape columns
to `training_models`, backfills existing rows from training metrics/metadata or
export rows, and creates the `model_input_shapes` view. This is a formal
registry schema bump because Palette tracks registry schema changes in the
`schema_version` table and `PRAGMA user_version`.

## Remaining Work

The next implementation layer can rely on `model_input_shapes` for detector and
pose input shape lookup. Remaining work before broadening the surface:

1. Add registry maintenance checks for `unknown` or `conflict` input shapes.
2. Add subject-mask model input contracts once that registry surface is
   finalized.
3. Continue hardening `fisheye.utils.predict_training_detections` with real
   training-Zarr/model smokes as new sampled recordings are imported.
