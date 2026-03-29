# Subject Mask Tuning Workflow

This document describes the current operator workflow for subject-mask tuning
and canary body-mask generation.

## Scope

- Traditional `subject_body` tuning and materialization.
- Batch propagation of `subject_mask_tuning` by `camera_id`.
- First-pass refinement/review of `subject_body` masks.

This is the current practical workflow for canary training zarrs. It is not a
contract doc.

## Current Stage Split

- Tuning metadata lives in:
  - `analysis_metadata.attrs["subject_mask_tuning"]`
- Raw subject masks live in:
  - `subject_mask_runs/<run>`
- Refined editable subject masks live in:
  - `refined_subject_masks_runs/<run>`

For the raw stage contract, see
[subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md).

## Preconditions

1. Archive contains a materialized crop run with `roi_images`.
2. Archive contains a background run.
3. Archive has dish-mask tuning if dish gating should be applied.
4. Archive is a training or analysis zarr where `analysis_metadata` attrs are
   writable.

## Recommended Procedure: Traditional `subject_body`

### 1. Tune `subject_body` interactively

```bash
scripts/py -m fisheye.tune.subject_mask_tuner <archive>.zarr --component subject_body
```

What the UI currently shows:

- cropped ROI
- ROI-local dish mask preview when present
- stored mask preview if a subject-mask run is provided
- dish-masked traditional proposal

Saved output:

- `analysis_metadata.attrs["subject_mask_tuning"].components["subject_body"]`

Notes:

- The tuner preview now applies the saved dish mask so it matches the
  materializer behavior.
- Saving writes tuning metadata only. It does not write a new
  `subject_mask_runs` run.

### 2. Optionally propagate tuning by camera

If the same camera setup is shared across recordings, you can copy the
subject-mask tuning to other zarrs with the same `camera_id`.

Dry run:

```bash
scripts/py -m fisheye.utils.apply_tuning_by_camera \
  /nvme1/recordings \
  --source <source_training>.zarr \
  --recursive \
  --keys subject_mask_tuning \
  --merge-dicts
```

Apply:

```bash
scripts/py -m fisheye.utils.apply_tuning_by_camera \
  /nvme1/recordings \
  --source <source_training>.zarr \
  --recursive \
  --keys subject_mask_tuning \
  --merge-dicts \
  --apply
```

Important behavior:

- `--keys subject_mask_tuning` copies only the subject-mask tuning payload.
- `--merge-dicts` recursively merges dict-like payloads instead of replacing the
  whole top-level attr.
- This is useful when the source only updates
  `components.subject_body`, while targets may already contain unrelated entries
  such as `components.eyes_union`.
- Without `--merge-dicts`, a top-level overwrite would replace the entire
  `subject_mask_tuning` object.
- By default, the tool only targets the same `zarr_use` as the source.

### 3. Materialize a raw subject-mask run

```bash
scripts/py -m fisheye.segmentation.subject_segmentation \
  <archive>.zarr \
  --run-name traditional_subject_masks_canary_001
```

Use `--overwrite` if rerunning after retuning:

```bash
scripts/py -m fisheye.segmentation.subject_segmentation \
  <archive>.zarr \
  --run-name traditional_subject_masks_canary_001 \
  --overwrite
```

Current behavior:

- reads the saved `subject_body` tuning
- reads the selected crop run and background run
- applies dish-mask gating when dish-mask tuning exists
- writes a body-only `subject_mask_runs/<run>`
- records `run_semantics = "traditional_subject_body_inference"`

### 4. Refine / review the body masks

```bash
scripts/py -m fisheye.tune.refined_subject_mask_review \
  <archive>.zarr \
  --subject-run traditional_subject_masks_canary_001 \
  --components subject_body
```

This creates or reopens a `refined_subject_masks_runs/<run>` entry and lets the
operator paint or erase masks per ROI.

## Eye-Union Note

`subject_mask_tuner --component eyes_union` is a separate path.

Current behavior:

- uses the eye-specific threshold/Sobel method
- stores tuning under `subject_mask_tuning.components["eyes_union"]`
- mirrors `eye_mask_tuning` for compatibility with the existing traditional eye
  segmentation path

That is useful for eye workflows, but it is not the current body-mask
materialization path.

## Swim-Bladder Note

Swim bladder now has a dedicated local tuner:

```bash
scripts/py -m fisheye.tune.swim_bladder_mask_tuner <archive>.zarr
```

Current behavior:

- centers a local patch on the `swim_bladder` keypoint
- tunes a traditional threshold/Sobel/morphology proposal on that patch
- stores parameters under
  `subject_mask_tuning.components["swim_bladder"]`

This is intentionally separate from the ROI-wide `subject_mask_tuner` because
the operator task is patch-local and keypoint-centered rather than whole-ROI.

The saved tuning is ready for future traditional swim-bladder materialization,
and that materializer now exists:

```bash
scripts/py -m fisheye.segmentation.swim_bladder_segmentation \
  <archive>.zarr \
  --run-name traditional_swim_bladder_masks_canary_001
```

Current materializer behavior:

- reads `subject_mask_tuning.components["swim_bladder"]`
- centers each proposal on the chosen swim-bladder keypoint run
- writes a raw `subject_mask_runs/<run>` entry with only the
  `swim_bladder` channel available
- records `run_semantics = "traditional_swim_bladder_inference"`

## Current Batch Propagation Recommendation

For subject-body tuning, prefer:

- source zarr from the camera you tuned interactively
- `--keys subject_mask_tuning`
- `--merge-dicts`
- dry run before `--apply`

That gives the safest behavior for mixed subject-mask payloads.

## Current Limitations

- `subject_segmentation.py` is currently single-process and canary-oriented.
- The batch copier groups by `camera_id`, not by dish design or protocol.
- `subject_mask_tuning` propagation is attr-based; it does not automatically
  rerun segmentation after copying.
- `refined_subject_masks_runs` exists, but body/swim-bladder curation is still
  early and should be treated as a canary workflow.

For deferred scaling work, see
[traditional_subject_segmentation_scaling_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/traditional_subject_segmentation_scaling_todo.md).
