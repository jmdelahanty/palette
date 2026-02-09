# Model Export Registry (V2 Design Notes)

## Why this exists
We already record:
- training runs in `training_runs`
- trained detector artifacts in `training_models`
- ONNX exports in `onnx_models`
- TensorRT exports in `tensorrt_models`

This is a strong foundation, but most deployment compatibility fields still live in JSON
manifests or `metadata_json`. That makes frequent questions harder than they should be:

- "Which FP16 TensorRT engines support batch >= 8 at 640x640?"
- "Which ONNX exports were generated at opset 11?"
- "Which engines were built on TRT 10.x for compute capability 8.6?"

Goal: keep JSON as full provenance, while promoting a small set of high-value fields into
typed columns for fast filtering.


## Current registry shape (canonical)
- `training_runs`: run lifecycle + config/manifest/model/metrics links.
- `training_models`: one row per detector training run (`run_id` primary key).
- `onnx_models`: one row per run's ONNX artifact (`run_id` primary key).
- `tensorrt_models`: one row per `(run_id, precision)`.

Legacy:
- `model_exports` exists for compatibility/backfill history and should not be treated as the
  long-term query surface.


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
