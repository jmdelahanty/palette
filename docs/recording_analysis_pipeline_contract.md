# Recording Analysis Pipeline Contract
<!-- contract-meta
version: 2
status: active
last_verified: 2026-05-09
-->

Purpose: define the canonical, operator-first contract for analysis processing per recording.

For the repo-wide current source-of-truth contract across detect, keypoints,
masks, registry, query, and operator surfaces, see
[current_pipeline_contract.md](current_pipeline_contract.md). This document is
the narrower import/detect/refine analysis pipeline contract.

Date anchored: 2026-02-09.

Related detect batch contract:
- `docs/detect_batch_analysis_zarr_parallel_agents_contract.md`

## Goals

- Keep stage responsibilities explicit and composable.
- Keep execution migration-safe and fail-closed on ambiguous inputs.
- Support single-recording and batch orchestration with consistent behavior.

## Canonical Stage Tools

- Stage 1 import tool:
  - Module: `fisheye.utils.import_recording_analysis`
  - Responsibility: ensure `*_analysis.zarr`, import video metadata, import
    stimulus metadata when an H5/protocol source is available.
  - Explicit non-goal: no detect or refine orchestration.
- Stage 2 dish-mask tuning/import:
  - Module: `fisheye.tune.mask_tuner` for operator tuning; acquisition-provided
    masks should be imported into the same Zarr attr surface.
  - Responsibility: populate or verify
    `analysis_metadata.attrs["dish_mask"]` before production detect/refine.
  - Current behavior note:
    - raw detect can run without a dish mask, but `refine_detect` applies the
      dish-mask bbox-center gate only when this metadata exists.
    - adding/changing the dish mask after refinement means detect-quality and
      refined-detect outputs for that run should be regenerated.
- Stage 3 detect tool:
  - Modules:
    - `fisheye.detection.detect_yolo` (explicit model path/config)
    - `fisheye.utils.run_detect_with_registry_model` (registry model resolution + detect run provenance)
  - Responsibility: append one detect run to analysis Zarr.
- Stage 4 detect-quality tool (required before refine for production):
  - Module: `fisheye.refinement.detect_quality`
  - Responsibility: append one raw-detect quality report under the selected
    detect run with `quality_flags` and `detection_quality_labels`.
  - Current behavior note:
    - blob/traditional detect path writes quality as part of detect.
    - YOLO detect path requires explicit `detect_quality` stage invocation.
- Stage 5 refine tool:
  - Module: `fisheye.refinement.refine_detect`
  - Responsibility: consume raw-detect quality labels, filter raw detections,
    and append sparse curated refined detect outputs; keep raw detect
    immutable.
  - Current behavior note:
    - when `analysis_metadata.attrs["dish_mask"]` exists, outside-dish
      candidates are filtered from curated instances but retained in
      `source_detections` with reason `outside_dish_mask`.
    - interpolation is disabled in the normal sparse-first refined-detect path.
- Stage 6 registry tool:
  - Module: `fisheye.registry.db.Registry.scan_zarr`
  - Responsibility: rescan/update registry metadata for the resulting analysis Zarr.

## Canonical Orchestrators

- Batch import-only orchestrator:
  - Module: `fisheye.utils.import_organized_recordings_analysis`
  - Behavior: resolve already organized recording directories, then run
    `process_recording_import` per recording.
  - Explicit non-goal: no detect, refine, keypoints, or registry scan
    orchestration.
- Single recording orchestrator:
  - Module: `fisheye.utils.run_recording_analysis_pipeline`
  - Required execution order:
    1. import (`process_recording_import`)
    2. dish-mask tune/import/verify before production detect/refine
    3. detect (`run_detect_yolo` or `run_detect_registry_model`)
    4. detect quality (`fisheye.refinement.detect_quality`)
    5. refine (`run_refine_detect`, optional)
    6. register (optional)
- Batch orchestrator:
  - Module: `fisheye.utils.import_recordings_analysis`
  - Behavior: resolve many recording plans, then run the full
    single-recording analysis pipeline per plan.
  - Naming warning: despite `import_` in the module name, `--apply` is not
    import-only; it proceeds into detect/refine unless those pipeline options
    are changed. Use `import_organized_recordings_analysis` when the desired
    action is only creating/importing analysis archives.

## Stage Order Invariant

The required order is:

1. `import_recording_analysis`
2. dish-mask tune/import/verify
3. detect
4. `detect_quality`
5. refine (optional)
6. register (optional)

Rationale: refine should consume explicit quality labels, not inferred "all clean"
fallbacks. Detect/refine should also run against an archive that already has
analysis purpose, imported metadata context, and the dish-mask geometry needed
for outside-dish gating.

## Recording-Only Mode

Some recordings have a camera video but no experiment/H5/protocol source. These
recordings are valid inputs for experiment-agnostic processing, but they are not
valid inputs for stimulus-aligned analyses.

`recording-only` means "no experiment/stimulus context is required"; it does
not mean the archive is limited to raw video metadata. A recording-only analysis
archive can still accumulate the normal non-stimulus Palette datasets:
detections, refined detection instances, crops, keypoints, refined keypoints,
raw segmentation/probability outputs, refined subject masks, subject shape,
track kinematics, swim-bout runs, bout-kinematics runs, and non-stimulus
exports/visualizations. The optional boundary is stimulus context and the
analysis families that require it.

Supported entry points:

- Single recording:
  - `scripts/py -m fisheye.utils.import_recording_analysis --recording-only ...`
  - `scripts/py -m fisheye.utils.run_recording_analysis_pipeline --recording-only ...`
- Batch:
  - `scripts/py -m fisheye.utils.import_organized_recordings_analysis --recording-only ...`
  - `scripts/py -m fisheye.utils.import_recordings_analysis --recording-only ...`
  - Equivalent explicit form: `--no-import-stimulus`

Behavior:

- `raw/*.h5` is required by default and whenever stimulus import is enabled.
- `--recording-only` disables stimulus import and allows the planner to resolve
  an archive from `cams/*.mp4` alone.
- Recording-only archives are marked at the root with:
  - `session_uuid`, `recording_id`, `recording_name`, `recording_path`
  - `recording_type = "behavior"`, `recording_subtype = "free"`,
    `behavior_mode = "free"` unless already provided
  - `artifact_schema_id = "recording_analysis_v1"` unless already provided
  - `experiment_context_status = "absent"`
  - `experiment_context_source = "none"`
  - `stimulus_runs_available = false`
- If an H5 is available, import keeps `experiment_context_status = "present"`
  and records `source_h5` / `source_h5_path`.
- Video-only training archives created with
  `fisheye.utils.intake_video_only_recording` use the same experiment-context
  fields with `artifact_schema_id = "video_only_v1"`.
- Registry scans are self-contained for these archives: when root attrs carry
  recording context, `Registry.scan_zarr` upserts a `recordings` row and exposes
  `experiment_context_status`, `experiment_context_source`, and
  `stimulus_runs_available` through `dataset_context_current`.

Valid downstream stages without experiment context:

- detect, detect-quality, refined detect
- crop, keypoints, refined keypoints
- masks, refined masks, subject shape
- track kinematics, swim-bout detection, bout kinematics
- detection coverage and whole-recording movement/bout exports

Readers, notebooks, and GUI tools should treat the following groups as optional
on recording-only archives and disable the corresponding UI panels or analysis
selectors rather than failing archive load:

- `analysis/stimulus_runs`
- `analysis/stimulus_response_runs`
- stimulus-aligned visualization artifacts

Other analysis families remain optional in the ordinary staged-processing sense:
they may be absent until produced, but their absence does not imply that the
archive is "video only" or ineligible for later detections, pose, segmentation,
or kinematics.

Invalid or non-meaningful without experiment context:

- `analysis/stimulus_response`
- OMR/concentric-grating/looming metrics
- step-level protocol summaries
- stimulus-aligned cross-recording exports

Do not synthesize fake stimulus runs for recording-only archives. If a no-stimulus
baseline needs step-like annotations, import a real protocol/stimulus context or
write a separate explicitly named baseline annotation run with its own
provenance.

## Detect Quality Guardrails (Best Practice)

- Always run `detect_quality` after each new detect run and before refine.
- Auto-run `detect_quality` when missing (or fail closed in strict mode).
- Treat `detect_quality` as a raw artifact-labeling stage, not as the refined
  review/approval stage.
- Treat quality as stale when detect run identity changes:
  - mismatch between `refined_detect_runs/<run>.attrs["source_detect_run"]` and
    the detect run you intend to refine.
  - missing `quality_reports.attrs["latest"]` for that detect run.
- Persist quality threshold provenance (`jump_threshold`,
  `blip_gap_threshold`) in quality report attrs for reproducibility.
- Prefer scaled thresholds for cross-resolution consistency:
  - `--threshold-mode scaled --threshold 100 --threshold-reference-width 640`
  - Interpreted as "100 px at 640px width", scaled per recording resolution.
- Log and surface the quality run ID used by refine (`source_quality_run`).
- For batch workflows, continue per-recording on failure, but classify failures
  as `failed_step=detect_quality` when this stage fails.

## Failure Semantics

- Input resolution is fail-closed for ambiguous single-recording inputs:
  - multiple `cams/*.mp4` without explicit `--video`
  - multiple `raw/*.h5` without explicit `--h5`
  - missing `raw/*.h5` when stimulus import is enabled
- Input resolution permits missing H5 only in recording-only / no-stimulus
  mode. In that mode the pipeline must not run stimulus import.
- Single-recording orchestrator:
  - stop immediately on first failed stage
  - return non-zero
  - report `failed_step` and `returncode` where available
  - treat `recording_manifest.json` `preflight.status=fail` as a blocking
    `failed_step=preflight_gate` unless `--allow-preflight-failures` is passed
- Batch orchestrator:
  - continue to next recording when one recording fails
  - summarize `ok/failed/skipped/missing`
  - recordings with blocking manifest preflight failures are planned as
    `missing` unless `--allow-preflight-failures` is passed
  - return non-zero if any recording failed

## Idempotency and Data Safety

- Import stage:
  - archive creation is idempotent (`mode="a"` when archive exists)
  - stimulus import defaults to skip when runs already exist unless `--stimulus-always`
- Detect stage:
  - append-only detect runs; existing runs remain immutable
- Detect-quality stage:
  - append-only quality reports under detect runs; existing quality reports
    remain immutable
- Refine stage:
  - append-only refined runs/manual groups; source detect runs remain immutable
- Registry stage:
  - rescan updates registry view of the archive path/metadata; does not require destructive rewrites

## Model Resolution Contract

- `--model-source explicit`:
  - detect uses explicit `--model` and/or detect config behavior.
- `--model-source registry`:
  - detect uses `run_detect_with_registry_model`
  - resolver currently targets `task=detect`
  - run provenance for selected model is written on the detect run attrs

## Logging Contract

- Batch pipeline writes JSONL logs unless `--no-log`.
- Expected high-level events:
  - `run_start`, `recording_plan`, `recording_start`
  - stage events (`video_metadata_imported`, `stimulus_result`, `detect_result`, `refine_result`)
  - terminal events (`recording_ok`, `recording_failed`, `recording_skipped`, `run_end`)

## Operator Runbook (Current)

- Single recording dry-run:
  - `scripts/py -m fisheye.utils.run_recording_analysis_pipeline --recording-dir "$REC" --dry-run`
- Dish-mask tune/verify after import and before production detect/refine:
  - `scripts/py -m fisheye.tune.mask_tuner "$ANALYSIS_ZARR" --registry /nvme1/palette_registry.sqlite`
  - The save always writes `analysis_metadata.attrs["dish_mask"]`; `--registry`
    additionally upserts `recording_step_status.dish_mask=ok` for the matching
    dataset. Without `--registry`, the next registry maintenance/backfill pass
    can discover the Zarr attr.
- Single recording apply with registry model + register:
  - `scripts/py -m fisheye.utils.run_recording_analysis_pipeline --recording-dir "$REC" --model-source registry --registry /nvme1/palette_registry.sqlite --register --apply`
  - If detect path is YOLO and pipeline wrapper does not auto-run quality:
    - `scripts/py -m fisheye.refinement.detect_quality "$ANALYSIS_ZARR" --save`
- Batch apply:
  - `scripts/py -m fisheye.utils.import_recordings_analysis /nvme1/recordings --recursive --model-source registry --registry /nvme1/palette_registry.sqlite --apply`
- Batch detect-quality stage (when needed separately):
  - `scripts/py -m fisheye.utils.detect_quality_batch /nvme1/recordings --recursive --apply`

## Out of Scope (Current Contract)

- Multi-camera recordings in one recording directory are fail-closed in this workflow.
- 3D/multi-view analysis layout is tracked separately in `docs/multicamera_3d_analysis_todo.md`.
