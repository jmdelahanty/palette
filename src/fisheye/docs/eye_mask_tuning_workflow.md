# Eye Mask Tuning Workflow

This document describes the current operator procedure for tuning and reviewing eye masks.

## Scope

- Traditional segmentation parameter tuning (`fisheye.tune.eye_mask_tuner` + rerun segmentation).
- Refined eye mask failure correction (`fisheye.tune.eye_mask_review --retune/--manual/--audit`).
- Works for both `_analysis.zarr` and curated `_training.zarr` archives when eye-mask runs are present.

## Preconditions

1. Archive contains `crop_runs/<run>/roi_images`.
2. Archive contains keypoint runs (`refined_keypoints_runs` preferred, else `keypoints_runs`).
3. If refining, archive already contains `eye_masks_runs/<run>`.

## Recommended Procedure (Traditional Source Masks)

1. Tune traditional eye-mask parameters interactively.

```bash
scripts/py -m fisheye.tune.eye_mask_tuner <archive>.zarr
```

Useful controls in tuner UI:
- `n`/`p`: next/previous ROI
- slider controls: `ROI Padding`, `PreThresh`, `Sobel %`, morphology, separation bounds
- `s`: save tuned parameters
- `q` or `ESC`: quit

Saved output:
- `analysis_metadata.attrs["eye_mask_tuning"]`

2. Re-run traditional eye segmentation so tuned parameters are used.

```bash
scripts/py -m fisheye.segmentation.eye_segmentation <archive>.zarr
```

Notes:
- Segmenter auto-loads `analysis_metadata.attrs["eye_mask_tuning"]` when present.
- Use `--crop-run` and `--keypoint-run` to pin specific sources.

3. Create a refined eye-mask run.

```bash
scripts/py -m fisheye.refinement.refine_eye_masks <archive>.zarr
```

Common options:
- `--source-run <eye_masks_run>`
- `--keypoint-run <run>`
- `--force-refine-traditional` to disable traditional fast-path copy behavior
- `--allow-latest-keypoint-fallback` only for legacy archives missing keypoint lineage attrs

4. Retune failed refined masks in batch.

```bash
scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --retune
```

Useful controls in retune UI:
- `e`: evaluate sample of remaining failures
- `E`: evaluate all remaining failures
- `a`: apply current parameters to remaining failures
- `n`/`p`: next/previous failure

Writes:
- per-ROI updates to `masks_roi`, ellipse arrays, contour arrays
- `retune_id` array and run attr `retune_params`
- `metrics/reason` tags include `retuned`

5. Manually correct remaining failures.

```bash
scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --manual
```

Useful controls in manual UI:
- `1`/`2`: active eye (left/right)
- mouse draw (`LMB`) / erase (`RMB`)
- `[` and `]`: brush size
- `s`: save current ROI correction
- `r`: reset current ROI to original masks
- `n`/`p`: next/previous failure
- `a`: set eye-mask review status using `--review-*` args (default: `approved/manual/training`)
- `N` / `R` / `P`: mark `needs_review` / `rejected` / `pending`

Writes:
- per-ROI updates to masks/ellipses/contours
- `metrics/reason` tags include `manual_correction`
- run attrs: `eye_mask_review_status` and parent `eye_mask_review_status_latest`

6. Recompute postprocess summary.

```bash
scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --audit
```

Writes:
- `refined_eye_masks_runs/<run>.attrs["summary_statistics"]["postprocess"]`
- `...["postprocess_updated_utc"]`

## Procedure Differences for YOLO/U-Net Eye-Mask Sources

1. Skip `eye_mask_tuner`; it tunes traditional threshold/morphology parameters only.
2. For model-based runs in `_analysis.zarr` archives, missing `analysis_metadata.attrs["eye_mask_tuning"]` is expected.
   - YOLO/U-Net tuning is performed through refinement/review (`--retune`, `--manual`), not traditional threshold metadata.
3. Run your model-based segmentation pipeline as usual.
4. Start at refinement + review:
   - `scripts/py -m fisheye.refinement.refine_eye_masks <archive>.zarr --source-run <model_eye_run>`
   - `scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --retune`
   - `scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --manual`
   - `scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --audit`

## Quick Verification Commands

1. List recent eye-mask runs and runtime metadata.

```bash
scripts/py -m fisheye.diagnostics.show_eye_mask_runs <archive>.zarr --limit 5
```

2. Check required arrays/attrs on eye-mask stages.

```bash
scripts/py -m fisheye.diagnostics.check_eye_masks <archive>.zarr
```

3. Print latest refined summary stats.

```bash
scripts/py -c "import zarr; r=zarr.open_group('<archive>.zarr', mode='r'); p=r.get('refined_eye_masks_runs'); n=p.attrs.get('latest') if p is not None else None; g=p[n] if (p is not None and n in p) else None; print('refined_run=', n); print('summary_statistics=', None if g is None else g.attrs.get('summary_statistics'))"
```
