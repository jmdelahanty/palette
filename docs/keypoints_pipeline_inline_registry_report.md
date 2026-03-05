# Pipeline Inline Registry Update Opportunities

## Summary

The processing pipeline (detect → crop → keypoints → keypoint refinement →
eye masks → eye mask refinement → ID assignment) has inconsistent registry
integration. Some stages update the registry as they execute; others are silent
and require a separate rescan or batch script to make the registry aware of
their completion. This report identifies the gaps across both the keypoints and
eye masks pipelines, documents the existing infrastructure, and recommends
specific changes.

---

## Current state: who talks to the registry?

| Stage | Inline step status? | Performance/quality refresh? | Source tag |
|---|---|---|---|
| Detect | No | No | — |
| Crop | **Yes** | No (rescan only) | `runtime_crop` |
| Detect quality | **Yes** | No | `runtime_detect_quality` |
| Refined detect | **Yes** | No | `runtime_refine_detect` |
| Keypoints (traditional) | **No** | No | — |
| Keypoints (YOLO) | **No** | No | — |
| Keypoints refinement | **Error path only** | Batch script only | `run_keypoints_batch_auto_review` |
| Eye masks (YOLO/UNet) | **No** | No | — |
| Eye masks refinement | **Yes** | **No** (neither perf nor quality) | `runtime_refine_eye_masks` |
| ID assignment | **Yes** | No | `runtime_assign_ids` |

### What "inline" means

An inline registry write happens during stage execution, immediately after the
zarr is written. It calls `upsert_recording_step_status()` to update the
`recording_step_status` table and fires `invalidate_downstream_steps()` so
dependent stages are marked `"missing"`. The user can see progress in the
registry TUI or wide view without running a separate rescan.

### What happens today without inline writes

1. Stage completes → writes zarr groups and attrs.
2. Nothing tells the registry.
3. User must either:
   - Run a batch script that happens to write status (only some do).
   - Run a full rescan via `registry_rescan` (expensive: reads entire arrays,
     DELETE+INSERT across 7+ tables per zarr).
   - Run `check_training_registry --view recording-steps-wide` which inspects
     zarrs directly but doesn't update the registry.

---

## Infrastructure that already exists

All the machinery for inline writes is production-ready. Nothing new needs to be
built — the stages just need to call it.

### Status ledger API

`src/fisheye/registry/status_ledger.py` — `upsert_recording_step_status()`

Writes one row to `recording_step_status` (latest state) and appends to
`recording_step_status_history` (audit trail). Idempotent via
`ON CONFLICT ... DO UPDATE`. Accepts:

- `dataset_id`, `step_name`, `status` (required)
- `run_name`, `method`, `coverage_pct`, `source`, `zarr_mtime_ns` (optional)
- `review_status_json`, `details_json` (optional structured metadata)

### Step cascade

`src/fisheye/registry/step_cascade.py` — `invalidate_downstream_steps()`

Dependency graph:
```
detect → refined_detect → crop → keypoints → refined_keypoints → {eye_masks, id_assignment}
eye_masks → refined_eye_masks
```

When a stage completes with `status="ok"`, its dependents are automatically
marked `"missing"`. This already works for stages that call it.

### Dataset ID resolution

`src/fisheye/registry/db.py` — `resolve_dataset_id(root, zarr_path)`

Returns `(dataset_id, session_uuid)` from zarr attrs. Every stage that writes
inline uses this.

### Stage provenance helpers

`src/fisheye/shared/stage_provenance.py` — `build_stage_provenance()`,
`write_stage_provenance()`

Already used by crop, keypoints YOLO, refined keypoints, eye masks YOLO, and
refined eye masks to write provenance attrs to zarr groups.

### Targeted refresh methods

`src/fisheye/registry/db.py` has per-table refresh methods that open one zarr,
extract rows, and replace them — much cheaper than a full rescan:

- `refresh_detect_performance_for_dataset()`
- `refresh_crop_quality_for_dataset()` (via `replace_crop_quality`)
- `refresh_keypoint_performance_for_dataset()`
- `refresh_keypoint_quality_for_dataset()`
- `refresh_eye_mask_performance_for_dataset()`
- `refresh_eye_mask_quality_for_dataset()`

`run_keypoints_batch.py` calls the keypoint variants (lines 695-702).
No batch script calls the eye mask variants.

---

## Keypoints pipeline gaps

### Gap K1: Keypoints detection — no step status write

**Where**: `src/fisheye/detection/detect_keypoints_yolo.py` (YOLO path) and
`src/fisheye/detection/detect_keypoints_traditional.py` (traditional path).

**What's missing**: After writing `keypoints_runs/<run>/` to zarr, neither
function tells the registry. The zarr attrs already contain everything needed
(`method`, `summary_statistics` with success rate, run name via `latest`).

**Recommendation**: Add ~30 lines at the end of each detection function,
following the crop pattern (`src/fisheye/tracking/crop.py` lines 215-250).

**Design consideration**: The registry connection. Crop receives the registry as
a parameter. The keypoints detection functions currently don't accept a registry
argument. Add an optional `registry: Registry | None` parameter — cleanest,
matches the crop pattern. When `None`, skip the write (backwards compatible).

### Gap K2: Keypoints refinement — success path is silent

**Where**: `src/fisheye/refinement/refine_keypoints.py`

**What's missing**: `_emit_refined_keypoint_status()` exists (lines 257-302) and
correctly writes step status + cascade invalidation. But it is only called on
**error paths**. The success path at ~line 815 writes zarr attrs and returns
without emitting status.

**Recommendation**: Call `_emit_refined_keypoint_status()` on the success path
with `status="ok"`. The function already accepts all needed parameters. This is
approximately a 5-line fix.

**Why it matters**: `run_keypoints_batch.py` compensates by writing
`refined_keypoints` status itself (line 642), but only in the batch workflow.
Interactive/pipeline runs that go through `create_refined_keypoint_run()`
directly leave the registry stale.

### Gap K3: Keypoints performance/quality refresh is batch-only

**Where**: `run_keypoints_batch.py` lines 695-702.

**What's missing**: `refresh_keypoint_performance_for_dataset()` and
`refresh_keypoint_quality_for_dataset()` are only called from the batch script.
Interactive/pipeline runs don't refresh these tables.

---

## Eye masks pipeline gaps

### Gap E1: Eye masks segmentation — no step status write

**Where**: `src/fisheye/segmentation/eye_segmentation_yolo.py` (YOLO),
`src/fisheye/segmentation/infer_unet_eye_masks.py` (UNet), and
`src/fisheye/utils/run_eye_masks_with_registry_model.py` (model resolution
wrapper).

**What's missing**: After writing `eye_masks_runs/<run>/` to zarr, none of these
functions tell the registry. The model resolution wrapper writes extensive
provenance attrs to the zarr (model selection details, candidate list, scores)
but never calls `upsert_recording_step_status()`.

The zarr attrs after segmentation contain everything the registry needs:
`method`, `total_rois`, `successful_roi_pairs`, `successful_roi_pair_rate`,
`source_crop_run`, `source_keypoints_run`, `duration_seconds`.

**Recommendation**: Same pattern as Gap K1. Add an optional `registry` parameter
to the segmentation entry points. Write step status with
`step_name="eye_masks"` on completion.

### Gap E2: Eye masks refinement — no performance/quality refresh

**Where**: `src/fisheye/refinement/refine_eye_masks.py`

**What's partially working**: `_emit_refined_eye_masks_status()` (lines 108-158)
correctly writes `recording_step_status` with `step_name="refined_eye_masks"`
and `source="runtime_refine_eye_masks"`. This is called on the **success path**
(line 2403). Registry write failures are caught and warned but don't fail the
stage.

**What's missing**: Unlike the keypoints batch script which calls both
`refresh_keypoint_performance_for_dataset()` and
`refresh_keypoint_quality_for_dataset()`, the eye masks refinement calls
**neither** `refresh_eye_mask_performance_for_dataset()` **nor**
`refresh_eye_mask_quality_for_dataset()`. These tables only get populated during
a full rescan.

**Recommendation**: After the existing `_emit_refined_eye_masks_status()` call
on the success path, add calls to both refresh functions. The data is already in
the zarr — the refresh functions just extract it.

### Gap E3: Eye masks refinement — no error status emission

**Where**: `src/fisheye/refinement/refine_eye_masks.py` main() (lines 2528-2530)

**What's missing**: If `refine_eye_masks()` raises an exception, the main()
function catches it, prints an error, and exits with code 1 — but does NOT write
error status to the registry. This means the registry has no record of a failed
refinement attempt.

**Contrast with refined keypoints**: `refine_keypoints.py` calls
`_emit_refined_keypoint_status()` on error paths. Eye masks refinement has the
equivalent function but doesn't call it on failure.

**Recommendation**: Add an `except` block in `refine_eye_masks()` or `main()`
that calls `_emit_refined_eye_masks_status(status="error", ...)` before
re-raising. The function already handles its own exceptions internally, so this
is safe.

### Gap E4: Batch script doesn't update registry after completion

**Where**: `src/fisheye/utils/run_eye_masks_batch.py`

**What's missing**: The batch script uses the registry for **discovery**
(querying datasets with `require_steps_ok=['crop', 'keypoints']`) but does not
update the registry after each recording completes. Compare with
`run_keypoints_batch.py` which calls both step status upsert and
performance/quality refresh.

**Recommendation**: Add a `_sync_eye_mask_registry_rows_after_run()` function
(mirroring the keypoints batch pattern) that calls step status upsert +
performance/quality refresh after each recording.

---

## Detect pipeline gap

### Gap D1: Detect stage — no step status write

**Where**: `src/fisheye/detection/detect_yolo.py` or equivalent detect entry
points, and `src/fisheye/utils/run_detections_batch.py`.

**What's missing**: The detection stage writes `detect_runs/<run>/` to zarr but
never updates `recording_step_status`. The batch script
(`run_detections_batch.py`) also does not write step status — it only logs JSON
output.

**Note**: This is the **first stage** in the pipeline. Its silence means the
cascade never fires from the start — downstream stages are never auto-marked
`"missing"` when a new detection run completes.

---

## Crop pipeline gap

### Gap C1: Crop stage doesn't refresh crop_quality

**Where**: `src/fisheye/tracking/crop.py` lines 215-250.

**What's already working**: Crop writes step status inline with
`source="runtime_crop"`. This is the reference implementation for other stages.

**What's missing**: It doesn't call `refresh_crop_quality_for_dataset()`.
The quality table only gets populated during a full rescan.

---

## Shared helper opportunity

The inline registry write pattern is repeated identically across stages. A
shared helper would reduce boilerplate and ensure consistency:

```python
# Proposed: src/fisheye/shared/registry_stage_complete.py

def emit_stage_completion(
    registry: Registry,
    root: zarr.Group,
    zarr_path: Path,
    *,
    step_name: str,
    run_name: str,
    method: str,
    source: str,
    coverage_pct: float | None = None,
    details: dict | None = None,
) -> None:
    """Write step status + cascade after a pipeline stage completes."""
    dataset_id, session_uuid = resolve_dataset_id(root, zarr_path)
    registry.upsert_dataset(
        dataset_id, session_uuid=session_uuid, zarr_path=zarr_path
    )
    upsert_recording_step_status(
        registry,
        dataset_id=dataset_id,
        step_name=step_name,
        status="ok",
        run_name=run_name,
        method=method,
        coverage_pct=coverage_pct,
        source=source,
        zarr_mtime_ns=int(zarr_path.stat().st_mtime_ns),
        details_json=details,
    )
    invalidate_downstream_steps(
        registry, dataset_id=dataset_id, step_name=step_name, source=source,
    )
```

Every stage would then reduce to one call:

```python
emit_stage_completion(
    registry, root, zarr_path,
    step_name="keypoints",
    run_name=run_name,
    method="yolo_pose",
    source="runtime_detect_keypoints_yolo",
    coverage_pct=summary.get("success_rate_percent"),
)
```

Stages that already have their own `_emit_*_status()` functions
(refined keypoints, refined eye masks) could be migrated to use this helper or
kept as-is if they have stage-specific logic beyond the common pattern.

---

## Cost of a full rescan (for context)

`register_from_root()` in `db.py` (line 9492) does a **complete extraction** for
one zarr:

- Opens every `*_runs` group
- Reads entire arrays into memory (`detection_source[:]`, `frame_counts[:]`,
  `usable_keypoints[:]`, etc.)
- Computes statistics (coverage %, percentiles, interpolation rates)
- DELETE+INSERT rows across 7+ quality/performance tables

There is **no incremental logic**. The `zarr_mtime_ns` column is stored but
never used to skip unchanged data. Every rescan re-reads everything.

For a batch of N zarrs, cost is **O(N × array_size)** — typically 10-100ms per
zarr but adds up across hundreds of recordings.

Inline step status writes avoid this entirely — they write one row to
`recording_step_status` using metadata already available from the stage
execution. No zarr re-reading required.

---

## Priority ordering

| Priority | Gap | Effort | Impact |
|---|---|---|---|
| 1 | Shared `emit_stage_completion` helper | ~40 lines, new file | Foundation for all other fixes |
| 2 | Keypoints detection step status (K1) | ~30 lines × 2 files | Biggest blind spot in the pipeline |
| 3 | Eye masks segmentation step status (E1) | ~30 lines × 2 files | Second biggest blind spot |
| 4 | Keypoints refinement success path (K2) | ~5 lines | Fixes the interactive/pipeline path |
| 5 | Eye masks refinement perf/quality refresh (E2) | ~10 lines | Brings parity with keypoints batch |
| 6 | Eye masks refinement error status (E3) | ~10 lines | Records failed attempts |
| 7 | Detect stage step status (D1) | ~30 lines | Completes cascade chain from the start |
| 8 | Eye masks batch registry sync (E4) | ~40 lines | Mirrors keypoints batch pattern |
| 9 | Crop quality refresh (C1) | ~10 lines | Nice-to-have |
| 10 | Keypoints perf/quality at stage time (K3) | ~10 lines | Nice-to-have; batch script covers it |

Items 1-6 are high value. Items 7-10 are incremental improvements that can be
done opportunistically.

---

## Asymmetry summary

### Eye masks vs keypoints comparison

| Aspect | Keypoints | Eye masks |
|---|---|---|
| Segmentation/detection step status | **No** | **No** |
| Refinement success step status | Error path only | **Yes** (inline) |
| Refinement error step status | Error path emits | **No** |
| Performance refresh | Batch script (lines 695-702) | **Never** |
| Quality refresh | Batch script (lines 695-702) | **Never** |
| Batch script registry sync | Partial (refined_keypoints only) | **None** |

The eye masks pipeline is actually better than keypoints for inline refinement
status (it emits on success), but worse for performance/quality table refresh
(never happens outside a full rescan).

---

## Related docs

- `docs/shared_helpers_refactor_todo.md` — broader shared helper deduplication
- `docs/recording_step_status_parallel_agents_contract.md` — step status API
  contract
- `src/fisheye/registry/status_ledger.py` — the step status write API
- `src/fisheye/registry/step_cascade.py` — downstream invalidation graph
- `src/fisheye/tracking/crop.py` lines 215-250 — the reference implementation
  for inline registry writes
- `src/fisheye/refinement/refine_eye_masks.py` lines 108-158 —
  `_emit_refined_eye_masks_status()` (eye masks inline write reference)
