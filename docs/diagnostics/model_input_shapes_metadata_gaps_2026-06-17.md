# Why `model_input_shapes` has empty precision / dtype / shape fields

**Date:** 2026-06-17
**Scope:** Read-only investigation of the registry (`/nvme1/palette_registry.sqlite`, 350 datasets at time of writing).
**Method:** Three parallel code-tracing passes + live-data verification. No code or data was modified.

## TL;DR

`model_input_shapes` is a **VIEW**, not a table — a `UNION ALL` of `training_models`,
`onnx_models`, and `tensorrt_models`. "Why is this cell empty" therefore has a
different answer per `artifact_kind`. The empty values fall into three causes:

1. **Structurally always-null** — the view hardcodes `NULL` because no source column exists.
2. **Never recorded on disk** — the exporters/training never capture the value anywhere.
3. **Data-dependent (recoverable)** — the source exists for some runs and not others.

Most empty cells are **correct-by-design**, not bugs. The one genuinely actionable
gap is a pose run whose shape lives in its on-disk manifest but was never synced
into the registry.

## Observed data gaps (live, 34 artifact rows)

| Field | onnx | tensorrt | training | Note |
|---|---|---|---|---|
| `artifact_precision` | **9/9 empty** | 0/9 | **16/16 empty** | only tensorrt has it |
| `input_dtype` | **9/9 empty** | **9/9 empty** | 7/16 empty | almost never set |
| `input_shape` (+ layout/channels/img_h/img_w/max_batch/dynamic_shapes) | 1/9 | 1/9 | **7/16 empty** | |
| `input_color_space` | 1/9 | 1/9 | 7/16 | |

`input_shape_status` values: training → `unknown` (7), `inferred_from_imgsz` (6),
`export_backfill` (3); onnx/tensorrt → mostly `explicit`, 1 each `unknown`.

## How the view is assembled

`CREATE VIEW model_input_shapes AS` in `src/fisheye/registry/db.py` (~1946–2050):

- `SELECT 'training' ...` from `training_models` — carries the only real shape/dtype columns.
- `SELECT 'onnx' ...` from `onnx_models` — **hardcodes `NULL AS artifact_precision`, `NULL AS input_dtype`**.
- `SELECT 'tensorrt' ...` from `tensorrt_models` — `trt.precision AS artifact_precision`, **`NULL AS input_dtype`**.

Verified against the live DB:

```
model_input_shapes object type: view
onnx_models:     has dtype=False  has precision=False
tensorrt_models: has dtype=False  has precision=True
training_models: has dtype=True   has precision=False
```

## Cause 1 — structurally always-null (no source column)

| Field | Empty for | Reason |
|---|---|---|
| `artifact_precision` | training, onnx | `training_models` and `onnx_models` have no `precision` column; the view writes literal `NULL`. Only `tensorrt_models.precision` exists. |
| `input_dtype` | onnx, tensorrt | Neither table has any `dtype` column; the view writes literal `NULL`. |

These cannot be fixed by a backfill — they require schema columns + capture logic.

## Cause 2 — never recorded on disk

- **Input dtype is not written by any exporter.** The ONNX manifest records *output*
  dtypes (e.g. `num_dets: INT32`), not input dtype. Training reports omit it entirely.
  The registry's only dtype source is a fallback inference in `db.py` (~1759): for
  `detect`/`pose` with a known shape, dtype is auto-set to `float32`. That is why the
  only non-null `input_dtype` values appear on training rows.
- **Precision is only meaningful for a built TensorRT engine** (fp16/int8/fp32). A
  `.pt` or `.onnx` is not precision-built, so `NULL` there is correct. Engine precision
  is recorded in the manifest `export.precision` and resolved by
  `_infer_tensorrt_precision()` (`db.py` ~6459), defaulting to `fp16`.

## Cause 3 — data-dependent: the 7 `unknown` training shapes

Training shape resolution order (`_shape_fields_from_training_payloads`, `db.py` ~1633–1781):

```
explicit            (input_shape in final_metrics/metadata)
  → inferred_from_imgsz   (imgsz in args.yaml / training report)
    → export_backfill     (copy shape from this run's onnx/tensorrt row)
      → unknown           (no source found)
```

The 7 `unknown` training rows (verified):

```
[eye_mask    ] exports=['training']                     eye_mask_cedar_shadow_..._v001
[pose        ] exports=['training']                     omnifin0_cedar_shadow_v001_pose_...
[pose        ] exports=['onnx','tensorrt','training']   omnifin0_cedar_shadow_v004_pose_...   <-- anomaly
[subject_mask] exports=['training']                     subject_masks_union_canary_v001
[subject_mask] exports=['training']                     subject_masks_union_all_components_v001
[pose        ] exports=['training']                     pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_01
[pose        ] exports=['training']                     pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry1
```

Two kinds:

- **6 `.pt`-only runs** (eye/subject mask UNets + pose). UNet training writes no
  `args.yaml`/training report with `imgsz`, and these have no onnx/trt export to
  backfill from → genuinely no source. Expected.
- **1 anomaly** — `omnifin0_cedar_shadow_v004_pose_...` has onnx **and** tensorrt
  exports with manifest paths set, yet their `input_shape` is `NULL`:

```
v004 pose — per-artifact in the view:
  [training] shape=None  status=unknown
  [onnx    ] shape=None  src=onnx_models.input_shape      status=unknown
  [tensorrt] shape=None  src=tensorrt_models.input_shape  status=unknown
raw rows:
  onnx_models:     input_shape=None img_h=None img_w=None manifest=set
  tensorrt_models: input_shape=None img_h=None img_w=None manifest=set
```

So the on-disk manifest sidecar exists (with `export.input_shape`) but the shape was
never extracted into the table at registration time.

## Where the metadata lives on disk

| Artifact | File | Carries |
|---|---|---|
| Training `.pt` | `args.yaml` (`imgsz: 640`), `*_training_report.yaml` (`training_history.effective_imgsz`) | shape only — no dtype, no precision |
| ONNX | `exports/onnx/<name>.onnx.manifest.json` → `export.input_shape`, `imgsz`, `opset`, `nms` | shape; output dtypes only |
| TensorRT | `exports/tensorrt/<name>_<prec>.tensorrt.manifest.json` → `export.precision`, `export.input_shape`, `trt`/`build_env` | shape + the only precision record |
| UNet masks | `best_model.pt` + `validation_previews/` only | nothing — no metadata sidecar |

Writers: `src/fisheye/training/train_detection.py` (onnx manifest ~1387, trt manifest
~1473, training report ~2014); registry ingest via `record_model_export()` →
`record_onnx_model()` / `record_tensorrt_model()` (`db.py` ~7180 / ~7282), reading
manifests through `_resolve_shape_fields()` (`db.py` ~6961). Migration-time training
backfill: `_backfill_training_model_input_shapes()` / `_export_input_shape_fallback()`
(`db.py` ~6141 / ~6301), invoked by migration 049.

## Assessment

- `artifact_precision` empty for `.pt`/`.onnx`, `input_dtype` empty for `.onnx`/`.engine`:
  **expected and correct.** No source exists; values would be constant (float32 /
  not-applicable). Low value to "fix."
- 6 mask/`.pt`-only `unknown` shapes: **expected** — UNet training emits no shape
  metadata. Fixable only by having UNet training write a report, or recording the known
  mask input size at registration.
- **v004 pose run: actionable.** The shape is in the on-disk manifest but unsynced.
  `src/fisheye/utils/backfill_pose_onnx_registry_metadata.py` exists to re-read those
  manifests and backfill `onnx_models` — a dry-run would show how many export rows it
  repopulates.

## Possible follow-ups (not done here)

1. Dry-run `backfill_pose_onnx_registry_metadata.py` to quantify recoverable export-row shapes.
2. Surface `input_shape_status` as a color-coded column on the `/models` browser page.
3. (Larger) Add input-size capture to UNet mask training reports so mask `.pt` rows resolve.
4. (Schema) Decide whether `input_dtype`/precision are worth carrying for onnx — likely not.
