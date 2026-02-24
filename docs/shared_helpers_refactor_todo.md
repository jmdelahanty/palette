# Shared Helpers & Deduplication TODO

Audit date: 2026-02-24

Scan covered ~150+ files across `src/fisheye/` and `src/` standalone scripts
(vendored `decord/` excluded).

---

## Status Key

- [ ] Not started
- [x] Done

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

---

### 2. Promote Existing `open_zarr_root()`

- [ ] Audit 187+ raw `zarr.open()` call sites in `src/fisheye/`
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

- [ ] Create `fisheye/cli/shared_args.py` with reusable argument-group functions:
  - `add_detection_args(parser)` — `--conf`, `--iou`, `--max-det` (8+ scripts)
  - `add_model_args(parser)` — `--model`, `--device` (6+ scripts)
  - `add_batch_args(parser)` — `--batch-size`, `--apply`/`--dry-run` (5+ scripts)
  - `add_zarr_store_args(parser)` — `store` positional + `--run` (10+ diagnostic scripts)
- [ ] Migrate existing scripts to use shared builders

**Affected files (non-exhaustive):**
- `fisheye/detection/detect_keypoints_yolo.py`
- `fisheye/detection/detect_yolo.py`
- `fisheye/inference/predict_pose.py`
- `fisheye/inference/predict_detections.py`
- `fisheye/utils/run_detections_batch.py`
- `fisheye/utils/crop_batch.py`
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

## MEDIUM PRIORITY

### 7. "Ensure Group Exists" Pattern

- [ ] Standardise on `require_group()` or a thin wrapper across the codebase
- [ ] Audit ~40 files that use `if X not in root: root.create_group(X)`

---

### 8. Bbox / Keypoint Coordinate Utilities

- [ ] Create `fisheye/shared/bbox_utils.py`:
  - `clip_keypoints(kp, width, height)`
  - `extract_best_detection(result)`
  - `extract_keypoint_confidences(keypoints, det_idx, n_keypoints)`
  - `normalize_coords(coords, width, height)`
- [ ] Consolidate ~15 scattered implementations

---

### 9. Provenance Recording Wrapper

- [ ] Add a higher-level wrapper around `build_stage_provenance()` in
  `fisheye/shared/stage_provenance.py` that auto-gathers `git`, `platform`,
  and `environment` dicts from `get_git_info()` / `get_environment_info()`
- [ ] Collapse ~30 lines per call site into ~5 across 12+ files

---

### 10. Array Validation / Gap Detection Helpers

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

### 11. Rich Console / Progress Bar Factory

- [ ] Create a small factory in `fisheye/utils/console.py`:
  - `make_progress(**overrides) -> Progress` (standardised columns)
- [ ] 27+ files instantiate the same `Progress(SpinnerColumn(), BarColumn(), ...)` pattern

---

## LOW PRIORITY / NICE-TO-HAVE

### 12. Consolidate Pydantic Config Schemas

- [ ] Audit `src/config_models.py` (Pydantic v1 legacy) vs
  `fisheye/training/config.py` (Pydantic v2)
- [ ] Consolidate to single source-of-truth if both are still imported

### 13. Data Card Plot Unification

- [ ] Unify `plot_detection_training_data_card.py` and
  `plot_keypoint_training_data_card.py` into a configurable
  `data_card_plotting.py` module
- [ ] Shared patterns: matplotlib Agg init, bar-chart styling, heatmap
  rendering, colorbar config, subplot grid with hidden unused axes

### 14. Color Scheme Constants

- [ ] Consolidate hardcoded hex colours (`#2E6F95`, `#4A7C59`, `#123146`, etc.)
  into a `fisheye/utils/color_schemes.py` or a dict constant
- [ ] 15+ files define ad-hoc colour palettes

---

## Suggested New Module Layout

```
src/fisheye/shared/
    type_conversions.py    # _normalize_text, _as_float, _as_int, safe_json_parse
    statistics.py          # weighted_mean, numeric_stats, aggregate_histograms
    zarr_helpers.py        # resolve_zarr_run, ensure_group, get_latest_run
    array_validation.py    # nan_summary, detect_frame_gaps, check_monotonicity
    bbox_utils.py          # clip_keypoints, extract_best_detection, normalize_coords

src/fisheye/utils/
    model_loader.py        # load_yolo_model(path, device, fp16)
    plot_helpers.py         # save_figure, create_subplot_grid, configure_colorbar
    data_card_plotting.py   # unified plotting for detection + keypoint data cards
    console.py             # make_progress() factory

src/fisheye/cli/
    shared_args.py         # add_detection_args, add_model_args, add_batch_args
```

---

## Impact Estimate

| Area                     | Est. lines removable | Files affected |
|--------------------------|---------------------:|---------------:|
| Zarr run resolution      |                 ~400 |            20+ |
| Data card dedup          |                 ~300 |              4 |
| Model loader             |                 ~200 |            24  |
| Argparse groups          |                 ~500 |            25+ |
| Plot save boilerplate    |                 ~300 |            68+ |
| Type conversion dedup    |                 ~100 |            10+ |
| **Total**                |           **~1,800+**|       **100+** |
