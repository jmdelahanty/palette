# Keypoint Retune Notes

This note records recent changes around keypoint retuning so future agents can follow the workflow and understand the storage decisions.

## Overview
- Added a failure-retune mode to the existing keypoint tuner to re-run keypoint detection on failed ROIs only.
- Retuning never touches detections or raw keypoint runs; it only edits the refined run.
- Retunes write a per-ROI `retune_id` plus a `retune_params` mapping on the refined run for auditability.
- Reason strings are written with slice-based assignment to avoid VLenUTF8 scalar write errors.

## Files touched
- `src/fisheye/tune/keypoint_tuner.py`
  - Failure retune mode via the keypoint review entrypoint.
  - Batch apply with optional parallel compute (ThreadPoolExecutor), single-writer.
  - Sample-based evaluation (`e`), full evaluation (`E`).
  - UI overlay indicates retune mode.
  - Uses tuned parameters from `analysis_metadata.keypoint_tuning` on startup.
  - VLenUTF8 reason write fixed with slice assignment.
  - Sanitizes `reason` array to plain strings before retuning.
- `src/fisheye/refinement/refine_keypoints.py`
  - Added `retune_id` array (int32, default -1) in refined runs.
- `src/fisheye/docs/zarr_structure.md`
  - Documented `retune_id` and `retune_params` attrs in refined runs.

## Retune flow
1. Run `refine_keypoints` to create a refined run and `failure_indices`.
2. Launch retune mode (review entrypoint):
   - `python -m fisheye.tune.keypoint_review /path/to.zarr --retune`
   - target a run with `--refined-run refined_keypoints_YYYY-MM-DD_HH-MM-SS`
3. Use sliders to adjust thresholds. Actions:
   - `e`: evaluate a sample (default 300) of failures.
   - `E`: evaluate all failures (slow).
   - `a`: apply to all remaining failures.
4. Successful ROIs are written into the refined run and tagged with `retune_id`.

## Postprocess summary
The review entrypoint recomputes `summary_statistics.postprocess` after retune or
manual correction so coverage reflects the current refined state.

## Deprecations
Legacy entrypoints for retune/manual review have been removed. Use
`python -m fisheye.tune.keypoint_review --retune|--manual` instead.

## Metadata and arrays
- `refined_keypoints_runs/<run>/retune_id` stores the parameter set label for each ROI.
- `refined_keypoints_runs/<run>.attrs["retune_params"]` maps `retune_id` -> parameter dict.
- `analysis_metadata.keypoint_tuning` remains untouched (original run only).

## Performance notes
- Apply uses parallel compute only (ThreadPoolExecutor) and single-threaded writes for safety.
- Threads can be slower than processes due to GIL, small ROI overhead, and library oversubscription.
- Use `--apply-batch-size` and `--apply-workers` to tune throughput.

## VLenUTF8 write fix
- Zarr VLenUTF8 expects array-like writes, not scalar strings.
- Use slice assignment:
  - `reason_arr[idx:idx+1] = np.array([reason], dtype=object)`
- A sanitize pass converts existing values to plain strings before retuning.

## Known follow-ups
- If reason writes fail elsewhere, apply the same slice-write pattern.
- Consider a process-based compute option for faster CPU-heavy workloads (still single writer).
