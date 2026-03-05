# Shared Helpers & Deduplication TODO

Initial audit: 2026-02-24
Updated: 2026-03-04 (status refresh + C1/C2 alignment cleanup)

Scan covered ~150+ files across `src/fisheye/` and `src/` standalone scripts
(vendored `decord/` excluded).

---

## Status Key

- [ ] Not started
- [~] Partial / in progress
- [x] Done

---

## 2026-03-04 Snapshot (Code-Verified Refresh)

Current state after direct code-audit verification of CRITICAL/HIGH/MEDIUM/LOW sections.

| Item | Status | Notes |
|---|---|---|
| C1. Registry step status helper | [x] Done | Shared `emit_stage_completion()` covers all current `_emit_*_status` wrappers and the direct inference/batch/manual sync status paths touched in this slice. |
| C2. Dataset metadata extraction helper | [x] Done | `extract_dataset_metadata(...)` is centralized in `registry/db.py`, reused via shared stage helpers, and direct resolve+attrs extraction callsites in active paths (including `registry/db.py` registration) are migrated. |
| C3. Normalization helper consolidation | [~] Partial | Shared `type_conversions.normalize_attr()` exists, but many private `_normalize_attr/_status_text/_decode_attr` variants remain. |
| C4. Batch logging/timestamp consolidation | [~] Partial | Added shared `batch_logging.py`; migrated 4 core batch runners to shared logger/run-id helpers. |
| 1. Zarr run resolution helper | [ ] Not started | Diagnostic/read paths still duplicate run resolution logic. |
| 2. Promote `open_zarr_root()` | [~] Partial | Helper exists, but adoption is limited and most call sites still use raw `zarr.open(...)`. |
| 3. Data-card aggregation dedup | [ ] Not started | Detection/keypoint aggregate scripts still largely parallel implementations. |
| 4. Model loading/device helper | [ ] Not started | YOLO model load/device placement remains copy-pasted across many files. |
| 5. CLI shared argument builders | [~] Partial | Added shared builders for apply/dry-run, logging, and registry discovery; migrated 4 core batch runners. |
| 6. Plot save/finalize helper | [ ] Not started | `tight_layout/savefig/close` boilerplate remains duplicated. |
| 7. Registry zarr discovery factory | [~] Partial | Added shared `zarr_discovery.py`; migrated discovery in detection/crop/keypoint/eye-mask runners. |
| 8. Root/log-dir resolution | [~] Partial | Added shared `environment.py`; migrated root/log-dir helpers in 4 core batch runners. |
| 9. Ensure-group pattern | [~] Partial | `require_group` usage exists, but many files still use `if not in root: create_group`. |
| 10. Bbox/keypoint utilities | [ ] Not started | No shared `bbox_utils.py`; duplicated local implementations remain. |
| 11. Provenance recording wrapper | [~] Partial | Base stage provenance helper exists, but many callsites still manually gather git/env/platform payloads. |
| 12. Array validation/gap helpers | [ ] Not started | No shared `array_validation.py`; duplicated NaN/gap/monotonic checks remain. |
| 13. Rich progress factory | [~] Partial | Repeated `_progress()` wrappers and `Progress(...)` setups remain; no shared factory. |
| 14. Lineage tracking wrapper | [~] Partial | Keypoint lineage helper exists, but generic `build_source_attrs()` is missing and ad-hoc source attrs remain. |
| 15. Data quality/coverage computation | [~] Partial | Some utilities exist in `refinement/utils.py`, but no shared cross-stage metrics helper yet. |
| 16. Batch loop abstraction | [~] Partial | Plan dataclasses and loop/summarize blocks remain duplicated across 4 core batch scripts. |
| 17. Sampled import metadata helper | [ ] Not started | Identical helper still duplicated in detect-quality and refine-detect. |
| 18. Pydantic schema consolidation | [~] Partial | Training uses v2 config; legacy `src/config_models.py` remains and appears orphaned in-repo. |
| 19. Data-card plot unification | [ ] Not started | Duplicate plotting logic spans detection, keypoint, and eye-mask data-card scripts. |
| 20. Color scheme constants | [ ] Not started | No shared color constants module; repeated hardcoded palettes remain. |

### Phase A Progress (2026-03-04)

- Added `src/fisheye/shared/type_conversions.py` with shared normalization/numeric coercion helpers.
- Added `src/fisheye/shared/registry_stage_complete.py` with `emit_stage_completion(...)` and shared registry-completion flow.
- Centralized `extract_dataset_metadata(...)` in `src/fisheye/registry/db.py` and reused it from shared stage-completion utilities.
- Shared stage-completion helper is now wired through current `_emit_*_status` wrappers in:
  - `detection/detect_keypoints_yolo.py`
  - `detection/detect_keypoints_traditional.py`
  - `segmentation/eye_segmentation_yolo.py`
  - `segmentation/infer_unet_eye_masks.py`
  - `refinement/refine_eye_masks.py`
  - `refinement/refine_detect.py`
  - `refinement/refine_keypoints.py`
  - `refinement/detect_quality.py`
  - `tracking/crop.py`
  - `tracking/assign_ids.py`
  - `inference/predict_pose.py`
  - `utils/keypoint_retry.py`
  - `utils/run_detect_with_registry_model.py`
  - `utils/run_eye_masks_with_registry_model.py`
- Additional direct status-write migrations now use `emit_stage_completion(...)`:
  - `inference/predict_detections.py`
  - `inference/predict_eye_masks.py`
  - `utils/run_keypoints_batch.py`
  - `utils/run_eye_masks_batch.py`
  - `tune/keypoint_failure_review.py`
- C2 cleanup slice migrated direct dataset metadata extraction to shared helper in:
  - `diagnostics/prepare_detect_training.py`
  - `refinement/refine_keypoints.py`
  - `registry/db.py` (`register_from_root`)

### Phase B Progress (2026-03-03)

- Added `src/fisheye/shared/batch_logging.py` with shared `JsonLogger`,
  `utc_now()`, and `make_run_id()`.
- Added `src/fisheye/shared/environment.py` with shared
  `resolve_recording_roots()` and `resolve_log_dir()`.
- Added `src/fisheye/shared/zarr_discovery.py` with shared registry-backed
  zarr discovery for paths and camera-aware entries.
- Added `src/fisheye/cli/shared_args.py` with shared argparse builders for
  apply/dry-run, logging args, and registry discovery args.
- Migrated 4 core batch runners to shared helpers:
  - `utils/run_detections_batch.py`
  - `utils/crop_batch.py`
  - `utils/run_keypoints_batch.py`
  - `utils/run_eye_masks_batch.py`

### Audit Metrics (2026-03-04)

- `zarr.open(...)`: 254 call sites across 160 files.
- `open_zarr_root(...)` usage outside helper definition: 3 call sites.
- `emit_stage_completion(...)`: 21 call sites across 20 files.
- direct `upsert_recording_step_status(...)` outside helper/ledger/cascade definitions: 1 call site (callback wrapper in `run_eye_masks_batch.py` for test hook compatibility).
- `PALETTE_RECORDINGS_ROOT` references: 66 occurrences across 43 files.
- `JsonLogger` references in 19 files.
- `_utc_now` helper copies in 32 files.
- `_run_id` helper copies in 17 files.
- `if ... not in root` + `root.create_group(...)`: 12 files under `src/fisheye`, 18 across `src`.
- Local `_progress()` wrappers: 9 files.
- `Progress(...)` constructions: 37 call sites across 30 files.

### Scope Corrections From Re-Audit

- Item 19 is broader than originally written: data-card plotting duplication is across 3 scripts (detection, keypoint, eye-mask), not only 2.
- Item 12 should explicitly include `check_smoothed_orientation_nan.py` alongside distance/speed NaN checks.
- Item 13 duplication extends well beyond the 9 previously called out batch scripts.

### Open Questions / Decisions (2026-03-04)

**Resolved in current implementation:**
1. `emit_stage_completion(...)` now supports a live `Registry` instance,
   path-like input, or env-discovered registry via `registry=...` +
   `auto_registry_from_env`.
2. Dataset metadata extraction is centralized in `registry/db.py` and now
   flows through canonical dataset-id resolution paths (including
   hash-suffixed IDs for source recordings).

**Still open:**
1. For reporting and success metrics, should counts target `src/fisheye` only
   or all of `src` including standalone scripts?
2. What is the standard error policy for registry writes in shared helpers:
   warn-and-continue, or hard-fail for selected stages?
3. Should `src/config_models.py` be removed, deprecated, or retained for
   external compatibility?
4. For Items 12/15, should shared metric helpers live under
   `fisheye/shared/*` or expand `fisheye/refinement/utils.py`?

### Recommended Next Implementation Plan (from current baseline)

#### Phase A (In Progress: Finish Remaining Critical Work)

1. C3: replace remaining local `_normalize_attr/_status_text/_as_text/_decode_attr`
   helpers with shared conversions.
2. C4: continue batch logging/timestamp migration beyond the 4 core runners.
3. Add focused regression tests where helper migration touches monkeypatchable
   status-write hooks.

#### Phase B (High Priority Consolidation)

1. Item 1: introduce read-path `resolve_zarr_run(...)` and migrate diagnostics.
2. Item 2: promote `open_zarr_root()` across remaining raw `zarr.open(...)`
   call sites.
3. Item 5: expand CLI shared argument builders into detection/model/batch/store
   groups.
4. Item 4: add YOLO model loading/device placement helper and migrate major
   inference/training call sites.

#### Phase C (Medium Priority Pipeline Hygiene)

1. Item 16: extract shared batch plan/execute/summarize loop.
2. Item 11: add higher-level provenance recording wrapper.
3. Items 12/13/14: consolidate array validation, progress factory, and lineage
   source-attr writes.
4. Item 17: deduplicate sampled-import metadata reader in refine-detect paths.

#### Phase D (Reporting & Plotting Cleanup)

1. Item 3: consolidate data-card aggregation helpers/statistics.
2. Item 6: standardize figure save/finalize helpers.
3. Item 19: unify data-card plotting module(s), including eye-mask path.
4. Item 20: consolidate color scheme constants.
5. Item 18: finalize Pydantic schema consolidation decision.

---

## CRITICAL PRIORITY

Items with the highest line savings and widest blast radius. These affect the
core pipeline execution path and should be tackled before the HIGH items.

### C1. Registry Step Status Emission Helper

- [x] Create `emit_stage_completion()` in `fisheye/shared/registry_stage_complete.py`
- [x] Migrate 13+ `_emit_*_status()` functions across detection, tracking,
      refinement, and segmentation stages

**Pattern (repeated 13 times, ~50 lines each):**
```python
def _emit_crop_step_status(*, root, zarr_path, status, run_name, method,
                           coverage_pct, review_status, details, console):
    try:
        zarr_file = Path(zarr_path).expanduser().resolve()
        dataset_id, session_uuid = resolve_dataset_id(root, zarr_file)
        recording_id = _normalize_attr(root.attrs.get("recording_id")) or ...
        zarr_use = _normalize_attr(root.attrs.get("zarr_use"))
        zarr_purpose = _normalize_attr(root.attrs.get("zarr_purpose"))
        registry_path = RegistryPaths.from_env(Path.cwd()).path
        registry = Registry(registry_path)
        try:
            registry.upsert_dataset(dataset_id, session_uuid, zarr_path, ...)
            upsert_recording_step_status(registry, dataset_id, recording_id,
                                         step_name, status, run_name, method, ...)
            if status == "ok":
                invalidate_downstream_steps(registry, dataset_id, step_name, ...)
        finally:
            registry.close()
    except Exception as exc:
        if console: console.print(f"[yellow]Warning:[/yellow] {exc}")
```

**Proposed helper:**
```python
def emit_stage_completion(
    root: zarr.Group | None,
    zarr_path: Path,
    *,
    step_name: str,
    status: str,
    source: str,
    run_name: str | None = None,
    method: str | None = None,
    coverage_pct: float | None = None,
    review_status_json: dict | None = None,
    details_json: dict | None = None,
    console: Console | None = None,
    registry: Registry | Path | str | None = None,
    metadata: DatasetMetadata | None = None,
    upsert_dataset_row: bool = True,
) -> bool:
    """Write step status + cascade after a pipeline stage completes."""
```

**Affected files:**
- `tracking/crop.py` — `_emit_crop_step_status()` (~55 lines)
- `refinement/refine_detect.py` — `_emit_refined_detect_status()` (~50 lines)
- `refinement/refine_keypoints.py` — `_emit_refined_keypoint_status()` (~50 lines)
- `refinement/refine_eye_masks.py` — `_emit_refined_eye_masks_status()` (~87 lines)
- `refinement/detect_quality.py` — `_emit_detect_quality_status()` (~48 lines)
- `detection/detect_keypoints_yolo.py` — `_emit_keypoint_step_status()` (~50 lines)
- `detection/detect_keypoints_traditional.py` — `_emit_keypoint_step_status()` (~50 lines)
- `segmentation/eye_segmentation_yolo.py` — `_emit_eye_masks_status()` (~50 lines)
- `segmentation/infer_unet_eye_masks.py` — `_emit_eye_masks_status()` (~60 lines)
- `tracking/assign_ids.py` — `_emit_tracking_step_statuses()` (~50 lines)
- `inference/predict_pose.py` — `_emit_keypoint_status()` (~50 lines)
- `utils/run_detect_with_registry_model.py` — `_emit_detect_step_status()` (~77 lines)
- `utils/run_eye_masks_with_registry_model.py` — `_emit_eye_masks_failure_status()` (~30 lines)

**Additional direct-path migrations (2026-03-04):**
- `inference/predict_detections.py` — detect step status writer now calls `emit_stage_completion(...)`
- `inference/predict_eye_masks.py` — eye-mask status writer now calls `emit_stage_completion(...)`
- `utils/run_keypoints_batch.py` — refined keypoint auto-review sync now calls `emit_stage_completion(...)`
- `utils/run_eye_masks_batch.py` — post-run eye-mask/refined-eye-mask sync now calls `emit_stage_completion(...)`
- `tune/keypoint_failure_review.py` — manual review sync now calls `emit_stage_completion(...)`

**Cross-reference:** `docs/keypoints_pipeline_inline_registry_report.md` (Priority 1)
recommends this same helper as the foundation for closing all inline-registry gaps.

**Est. savings:** ~600 lines across 13 files.

---

### C2. Dataset Metadata Extraction Helper

- [x] Create `extract_dataset_metadata(root, zarr_path)` as a shared helper in
      `fisheye/registry/db.py` and consume it via `shared/registry_stage_complete.py`
- [x] Migrate remaining direct resolve + attr extraction paths in active code paths

**Pattern (repeated 14+ times, ~6 lines each):**
```python
dataset_id, session_uuid = resolve_dataset_id(root, zarr_path)
recording_id = _normalize_attr(root.attrs.get("recording_id")) or _normalize_attr(session_uuid)
zarr_use = _normalize_attr(root.attrs.get("zarr_use"))
zarr_purpose = _normalize_attr(root.attrs.get("zarr_purpose"))
```

**Proposed helper returns a NamedTuple or dataclass** with `dataset_id`,
`session_uuid`, `recording_id`, `zarr_use`, `zarr_purpose`.

**Affected files:** Same 13 as C1 plus `registry/db.py` register_from_root.

**Est. savings:** ~200 lines across 14 files.

---

### C3. Normalization Helper Consolidation

- [x] Promote `_decode_attr()` from `registry/db.py` to
      `fisheye/shared/type_conversions.py` as `normalize_attr()`
- [ ] Replace remaining private copies of `_normalize_attr`, `_status_text`,
      `_as_text`, `_decode_attr` across the codebase

**Pattern (repeated with minor name variations in 33+ files):**
```python
def _normalize_attr(value):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None
```

Variants: `_normalize_attr`, `_status_text`, `_as_text`, `_decode_attr`,
`_as_float`, `_coerce_positive_float`. All do byte-safe string/float coercion.

**Est. savings:** ~300 lines across 33 files.

---

### C4. Batch Logging & Timestamp Consolidation

- [x] Create `fisheye/shared/batch_logging.py` with:
  - `JsonLogger(log_dir, run_id)` — JSONL event logger
  - `utc_now() -> str` — ISO 8601 UTC timestamp
  - `make_run_id() -> str` — `YYYYMMDDTHHMMSSZ_<pid>` run identifier
- [~] Migrate 15-29 files that define identical private copies

**Pattern: `JsonLogger` class (identical in 15+ files):**
```python
class JsonLogger:
    def __init__(self, log_dir, run_id):
        self._path = log_dir / f"{run_id}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
    def log(self, **kwargs):
        kwargs.setdefault("ts_utc", _utc_now())
        kwargs.setdefault("run_id", self._run_id)
        with open(self._path, "a") as f:
            f.write(json.dumps(kwargs, default=str) + "\n")
```

**Pattern: `_utc_now()` (identical in 29 files):**
```python
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
```

**Pattern: `_run_id()` (identical in 15 files):**
```python
def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{os.getpid()}"
```

**Affected files (non-exhaustive):**
- `utils/run_detections_batch.py` (lines 56-77)
- `utils/run_keypoints_batch.py` (lines 75-88)
- `utils/run_eye_masks_batch.py` (lines 56-69)
- `utils/crop_batch.py` (lines 64-77)
- 11+ additional batch/utility scripts

**Est. savings:** ~450 lines across 15-29 files.

---

## HIGH PRIORITY

### 1. Zarr Run Resolution Helper

- [ ] Create `resolve_zarr_run(root, parent_path, run_name, fallback_to_latest=True)` in `fisheye/shared/zarr_helpers.py`
- [ ] Migrate ~20 diagnostic scripts that copy-paste the resolve-latest-or-named block

**Pattern (repeated ~20+ times):**
```python
parent = root["analysis"]["movement_runs"]["offline"]
run_name = run_name or parent.attrs.get("latest")
if not run_name:
    raise SystemExit("No 'latest' attribute")
if run_name not in parent:
    raise SystemExit(f"Run not found. Available: {sorted(parent.group_keys())}")
return parent[run_name], run_name
```

**Affected files (non-exhaustive):**
- `fisheye/diagnostics/check_smoothed_distance_nan.py`
- `fisheye/diagnostics/check_smoothed_speed_nan.py`
- `fisheye/diagnostics/check_smoothed_orientation_nan.py`
- `fisheye/diagnostics/validate_centroids.py`
- `fisheye/diagnostics/check_chaser_periodicity.py`
- `fisheye/diagnostics/plot_chaser_alignment.py`
- 15+ additional diagnostics

**Note:** `get_run_group()` in `shared/zarr/schema.py` handles *creation*, but
not *resolution of existing runs*. 14+ pipeline files already use
`get_run_group()` for creation. The gap is on the read/resolve side.

**Est. savings:** ~400 lines across 20+ files.

---

### 2. Promote Existing `open_zarr_root()`

- [ ] Audit 200+ raw `zarr.open()` call sites in `src/fisheye/`
- [ ] Replace with `open_zarr_root()` from `fisheye/utils/zarr_io.py` where appropriate
- [ ] Consider a grep-based pre-commit check to discourage raw `zarr.open()` in new code

**Current state:** Helper exists, handles Zarr v2/v3 compat, but only 3 files import it.

---

### 3. Data Card Aggregation Deduplication

- [ ] Extract shared functions into `fisheye/shared/type_conversions.py`:
  - `_normalize_text(value) -> Optional[str]`
  - `_as_float(value) -> Optional[float]`
  - `_as_int(value) -> Optional[int]`
  - `_parse_json_mapping(value) -> Optional[dict]`
- [ ] Extract shared functions into `fisheye/shared/statistics.py`:
  - `weighted_mean(values) -> Optional[float]`
  - `numeric_stats(values) -> Optional[dict]`
  - `aggregate_histograms(summaries, histogram_names) -> dict`
- [ ] Refactor `aggregate_detection_training_data_card.py` to use shared modules
- [ ] Refactor `aggregate_keypoint_training_data_card.py` to use shared modules

**Current state:** The two aggregate files are ~95% identical. The two plot files
(`plot_detection_training_data_card.py`, `plot_keypoint_training_data_card.py`)
also share matplotlib Agg init, bar-chart styling, heatmap rendering, colorbar
config, and save boilerplate.

---

### 4. Model Loading & Device Placement Helper

- [ ] Create `load_yolo_model(path, device=None, fp16=False)` in `fisheye/utils/model_loader.py`
- [ ] Migrate ~24 call sites across detection, keypoint, segmentation, training, and inference

**Pattern (repeated ~24 times):**
```python
model = YOLO(str(model_path))
model.to(device)
torch.backends.cudnn.benchmark = True
model.model = model.model.to(memory_format=torch.channels_last)
model.half()
```

**Affected files (non-exhaustive):**
- `fisheye/detection/detect_yolo.py`
- `fisheye/detection/detect_keypoints_yolo.py`
- `fisheye/segmentation/eye_segmentation_yolo.py`
- `fisheye/training/train_detection.py`
- `fisheye/training/train_pose.py`
- `fisheye/inference/predict_detections.py`
- `fisheye/inference/predict_pose.py`

---

### 5. CLI Shared Argument Builders

- [~] Create `fisheye/cli/shared_args.py` with reusable argument-group functions:
  - Implemented: `add_apply_dry_run_args`, `add_log_args`,
    `add_registry_discovery_args`.
  - Remaining: `add_detection_args(parser)` (`--conf`, `--iou`, `--max-det`),
    `add_model_args(parser)` (`--model`, `--device`), broader
    `add_batch_args(parser)` coverage, and `add_zarr_store_args(parser)`.
- [~] Migrate existing scripts to use shared builders

**Affected files (non-exhaustive):**
- `fisheye/detection/detect_keypoints_yolo.py`
- `fisheye/detection/detect_yolo.py`
- `fisheye/inference/predict_pose.py`
- `fisheye/inference/predict_detections.py`
- `fisheye/utils/run_detections_batch.py`
- `fisheye/utils/crop_batch.py`
- `fisheye/utils/run_keypoints_batch.py`
- `fisheye/utils/run_eye_masks_batch.py`
- 20+ diagnostic scripts

---

### 6. Plot Save / Finalize Helper

- [ ] Create lightweight helpers in `fisheye/utils/plot_helpers.py`:
  - `save_figure(fig, path, dpi=150, bbox_inches="tight", close=True)`
  - `create_subplot_grid(n_panels, max_cols=4) -> (fig, axes)`
- [ ] Unify DPI defaults (currently 150 in most places, 180 in data cards)
- [ ] Migrate highest-traffic files first

**Current state:** ~150+ occurrences of the `tight_layout` / `savefig` / `close`
sequence across 68+ files.

---

### 7. Registry Zarr Discovery Factory

- [x] Create `discover_zarrs(source, registry_path, scope_paths, **filters)` in
      `fisheye/shared/zarr_discovery.py`
- [x] Migrate 4 batch scripts that each implement `_discover_zarrs_from_registry()`

**Pattern (repeated in 4 files, ~60 lines each):**
```python
def _discover_zarrs_from_registry(
    registry_path: Path,
    scope_paths: Sequence[Path],
    rig_id: Optional[str],
    arena_id: Optional[str],
    camera_id: Optional[str],
    path_contains: Optional[str],
    skip_existing: bool,
) -> List[Path]:
    registry = Registry(registry_path)
    try:
        rows = registry.query_datasets(rig_id=rig_id, arena_id=arena_id, ...)
        # filter by scope, deduplicate, sort
    finally:
        registry.close()
```

**Affected files:**
- `utils/run_detections_batch.py` — `_discover_analysis_zarrs_from_registry()` (lines 169-228)
- `utils/crop_batch.py` — `_discover_zarrs_from_registry()` (lines 364-425)
- `utils/run_keypoints_batch.py` — similar discovery function
- `utils/run_eye_masks_batch.py` — similar discovery function

**Est. savings:** ~200 lines across 4 files.

---

### 8. Root Path & Log Dir Resolution

- [x] Create `resolve_recording_roots()` and `resolve_log_dir()` in
      `fisheye/shared/environment.py`
- [~] Migrate 33+ files that duplicate the `PALETTE_RECORDINGS_ROOT` env lookup

**Pattern (repeated in 33+ files):**
```python
def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]
```

**Est. savings:** ~200 lines across 33 files.

---

## MEDIUM PRIORITY

### 9. "Ensure Group Exists" Pattern

- [ ] Standardise on `require_group()` or a thin wrapper across the codebase
- [ ] Audit ~40 files that use `if X not in root: root.create_group(X)`

---

### 10. Bbox / Keypoint Coordinate Utilities

- [ ] Create `fisheye/shared/bbox_utils.py`:
  - `clip_keypoints(kp, width, height)`
  - `extract_best_detection(result)`
  - `extract_keypoint_confidences(keypoints, det_idx, n_keypoints)`
  - `normalize_coords(coords, width, height)`
- [ ] Consolidate ~15 scattered implementations

---

### 11. Provenance Recording Wrapper

- [ ] Add a higher-level wrapper around `build_stage_provenance()` in
  `fisheye/shared/stage_provenance.py` that auto-gathers `git`, `platform`,
  and `environment` dicts from `get_git_info()` / `get_environment_info()`
- [ ] Collapse ~30 lines per call site into ~5 across 12+ files

---

### 12. Array Validation / Gap Detection Helpers

- [ ] Create `fisheye/shared/array_validation.py`:
  - `nan_summary(array) -> dict` (NaN count, indices, percentage)
  - `detect_frame_gaps(ids) -> list[dict]` (`np.diff` + `np.where` pattern)
  - `check_monotonicity(array) -> bool`
- [ ] Migrate implementations from:
  - `diagnostics/check_smoothed_distance_nan.py`
  - `diagnostics/check_smoothed_speed_nan.py`
  - `chaser_analysis/check_frame_monotonicity.py`
  - `chaser_analysis/h5_gap_checker.py`
  - `detection_gap_analyzer.py`
  - `multi_roi_gap_interpolator.py`

---

### 13. Rich Console / Progress Bar Factory

- [ ] Create a small factory in `fisheye/utils/console.py`:
  - `make_progress(**overrides) -> Progress` (standardised columns)
- [ ] 27+ files instantiate the same `Progress(SpinnerColumn(), BarColumn(), ...)` pattern

Confirmed across 9 batch scripts: `run_keypoints_batch.py`,
`run_eye_masks_batch.py`, `crop_batch.py`, `refine_keypoints_batch.py`,
`refine_detect_batch.py`, `detect_quality_batch.py`, `retune_detect_batch.py`,
`compute_backgrounds_batch.py`, `prune_zarr_runs.py`, and 18+ others.

---

### 14. Lineage Tracking (Source Run Attrs)

- [ ] Extend `fisheye/shared/provenance_attrs.py` with a generic
      `build_source_attrs()` helper
- [ ] Migrate 8+ files that manually write `source_detect_run`,
      `source_refined_run`, `source_crop_run`, `source_keypoints_run`,
      `source_quality_run`, `source_eye_masks_run`, `source_stimulus_run`

**Current state:** `provenance_attrs.py` only handles `source_keypoints_run`
lineage. Other source attrs are written ad-hoc:

| Attr | Files writing it |
|---|---|
| `source_detect_run` | crop.py, assign_ids.py, refine_keypoints.py, keypoint_yolo.py |
| `source_refined_run` | crop.py (3x), keypoint_yolo.py, refine_keypoints.py |
| `source_crop_run` | refine_keypoints.py (multiple), visualization files |
| `source_quality_run` | refine_detect.py, detect_quality.py |
| `source_eye_masks_run` | visualization files, eye_segmentation |
| `source_stimulus_run` | refine_online_detect.py |

**Est. savings:** ~150 lines across 8+ files.

---

### 15. Data Quality & Coverage Computation

- [ ] Create `fisheye/shared/quality_metrics.py` or extend
      `fisheye/refinement/utils.py` with:
  - `compute_coverage_stats(presence_mask) -> dict`
  - `compute_detection_distribution(frame_counts) -> dict`
  - `compute_success_rate(success_mask, total_expected) -> dict`
  - `compute_nan_statistics(array) -> dict`
- [ ] Migrate 5+ files that compute these manually

**Pattern: detection distribution (repeated in detect_traditional.py, visualize
files, quality reports):**
```python
distribution = {
    'frames_with_0': int(np.sum(frame_counts == 0)),
    'frames_with_1': int(np.sum(frame_counts == 1)),
    'frames_with_2': int(np.sum(frame_counts == 2)),
    'frames_with_3_to_5': int(np.sum((frame_counts >= 3) & (frame_counts <= 5))),
    'frames_with_6_plus': int(np.sum(frame_counts >= 6)),
}
```

**Est. savings:** ~120 lines across 5+ files.

---

### 16. Batch Processing Loop Abstraction

- [ ] Create `fisheye/shared/batch_processor.py` with a plan-based batch loop
- [ ] Migrate 4 batch scripts that follow the same plan/execute/summarize pattern

**Pattern (repeated in 4 files, ~100-150 lines each):**
1. Build a list of `Plan` dataclasses (status, reason, zarr_path, prerequisites)
2. Iterate plans: check status, log events, call pipeline function with try/except
3. Count ok/skipped/missing/failed
4. Print summary statistics

**Affected files:**
- `utils/run_detections_batch.py` (lines 713-856)
- `utils/crop_batch.py` (lines 666-747)
- `utils/run_keypoints_batch.py` (lines 2037+)
- `utils/run_eye_masks_batch.py` (similar pattern)

**Note:** Plan dataclasses (`DetectPlan`, `KeypointPlan`, `EyeMaskPlan`,
`CropPlan`) are structurally identical. Status constants (`STATUS_OK`,
`STATUS_SKIPPED`, `STATUS_MISSING`, `STATUS_FAILED`) are defined in some files
but inline strings in others.

**Est. savings:** ~300 lines across 4 files.

---

### 17. Input Validation: Sampled Import Metadata

- [ ] Create shared `read_sampled_import_meta()` in
      `fisheye/shared/zarr_helpers.py`
- [ ] Migrate `detect_quality.py` and `refine_detect.py` which have identical
      38-line implementations

**Pattern (identical in 2 files at ~38 lines each):**
```python
def _read_sampled_import_meta(root):
    # ... reads sampled import attrs, parses JSON, returns (is_sampled, meta)
```

**Est. savings:** ~38 lines (one file fully deduplicated).

---

## LOW PRIORITY / NICE-TO-HAVE

### 18. Consolidate Pydantic Config Schemas

- [ ] Audit `src/config_models.py` (Pydantic v1 legacy) vs
  `fisheye/training/config.py` (Pydantic v2)
- [ ] Consolidate to single source-of-truth if both are still imported

### 19. Data Card Plot Unification

- [ ] Unify `plot_detection_training_data_card.py` and
  `plot_keypoint_training_data_card.py` into a configurable
  `data_card_plotting.py` module
- [ ] Shared patterns: matplotlib Agg init, bar-chart styling, heatmap
  rendering, colorbar config, subplot grid with hidden unused axes

### 20. Color Scheme Constants

- [ ] Consolidate hardcoded hex colours (`#2E6F95`, `#4A7C59`, `#123146`, etc.)
  into a `fisheye/utils/color_schemes.py` or a dict constant
- [ ] 15+ files define ad-hoc colour palettes

---

## Suggested New Module Layout

```
src/fisheye/shared/
    registry_stage_complete.py  # emit_stage_completion (C1), uses extract_dataset_metadata from registry/db (C2)
    type_conversions.py         # normalize_attr, _as_float, _as_int, safe_json_parse (C3)
    batch_logging.py            # JsonLogger, utc_now, make_run_id (C4)
    zarr_helpers.py             # resolve_zarr_run, ensure_group, get_latest_run (1)
    zarr_discovery.py           # discover_zarrs (registry + filesystem) (7)
    environment.py              # resolve_recording_roots, resolve_log_dir (8)
    array_validation.py         # nan_summary, detect_frame_gaps, check_monotonicity (12)
    bbox_utils.py               # clip_keypoints, extract_best_detection, normalize_coords (10)
    quality_metrics.py          # coverage_stats, detection_distribution, nan_statistics (15)
    batch_processor.py          # plan-based batch loop abstraction (16)
    statistics.py               # weighted_mean, numeric_stats, aggregate_histograms (3)
    provenance_attrs.py         # [EXPAND] build_source_attrs for all lineage types (14)
    stage_provenance.py         # [EXPAND] higher-level auto-gather wrapper (11)

src/fisheye/utils/
    model_loader.py             # load_yolo_model(path, device, fp16) (4)
    plot_helpers.py             # save_figure, create_subplot_grid, configure_colorbar (6)
    data_card_plotting.py       # unified plotting for detection + keypoint data cards (19)
    console.py                  # make_progress() factory (13)

src/fisheye/cli/
    shared_args.py              # add_detection_args, add_model_args, add_batch_args,
                                # add_registry_discovery_args (5)
```

---

## Impact Estimate

| Area                          | Status | Est. lines removable | Files affected |
|-------------------------------|--------|---------------------:|---------------:|
| Registry status emission (C1) | Done   |                 ~600 |            13  |
| Batch logging + timestamps (C4) | Partial |              ~450 |         15-29  |
| Zarr run resolution (1)       | Open   |                 ~400 |            20+ |
| Normalization helpers (C3)    | Partial |               ~300 |            33  |
| Data card dedup (3)           | Open   |                 ~300 |              4 |
| Batch processing loop (16)    | Partial |               ~300 |              4 |
| Argparse groups (5)           | Partial |               ~500 |            25+ |
| Plot save boilerplate (6)     | Open   |                 ~300 |            68+ |
| Dataset metadata (C2)         | Done   |                 ~200 |            14  |
| Registry zarr discovery (7)   | Done   |                 ~200 |              4 |
| Model loader (4)              | Open   |                 ~200 |            24  |
| Root path resolution (8)      | Partial |               ~200 |            33  |
| Lineage tracking (14)         | Partial |               ~150 |              8 |
| Data quality metrics (15)     | Partial |               ~120 |              5 |
| Type conversion dedup (3)     | Partial |               ~100 |            10+ |
| **Gross potential (historical, includes completed items)** | N/A | **~4,020+** | **150+** |
| **Approx remaining (excludes completed C1/C2/7)** | N/A | **~3,020+** | **120+** |

---

## Related docs

- `docs/keypoints_pipeline_inline_registry_report.md` — identifies registry
  gaps that C1 (`emit_stage_completion`) would close
- `docs/recording_step_status_parallel_agents_contract.md` — step status API
  contract
- `src/fisheye/shared/zarr/schema.py` — existing `get_run_group()` helper
- `src/fisheye/shared/stage_provenance.py` — existing provenance helper
- `src/fisheye/shared/provenance_attrs.py` — existing lineage helper (keypoints only)
- `src/fisheye/registry/status_ledger.py` — the step status write API
- `src/fisheye/registry/step_cascade.py` — downstream invalidation graph
