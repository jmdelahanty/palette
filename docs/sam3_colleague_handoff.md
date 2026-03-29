# Using The Palette SAM3 Subject-Mask Workflow On New Data

## Purpose

This note is for a collaborator who wants to reuse the SAM3 subject-mask work
done here on their own data.

The short version is:

- the Palette-specific integration lives in `palette`,
- SAM3 is used mostly as an upstream runtime,
- Paintera is optional and only matters if they want manual mask cleanup inside
  Palette-style Zarr stores,
- and no custom SAM3 training dataloaders were required for the workflow that
  was verified.

## Short Answer

If a collaborator asks "where is the real work for using SAM3 on our data?",
the answer is:

- **Primary repo:** `palette`
- **Runtime dependency:** `sam3`
- **Optional editor:** `paintera`

The key Palette entrypoints are:

- `src/fisheye/utils/run_sam_subject_masks.py`
- `src/fisheye/visualization/visualize_sam_subject_prompts.py`

These are the files that:

- read Palette training Zarr data,
- resolve crop runs and keypoint runs,
- construct SAM3 prompts,
- import the local SAM3 image runtime,
- run inference, and
- write new `subject_mask_runs/<run>` outputs back into the training archive.

## What We Did **Not** Need

We did **not** need to add a custom SAM3 dataloader or a Palette-specific
SAM3 training pipeline.

Specifically, the verified workflow did **not** depend on editing:

- `sam3/train/data/*`
- `sam3/train/*`
- SAM3 collators
- SAM3 dataset classes

So if a collaborator wants to reuse the current approach, they should **not**
start by forking SAM3's training/data stack.

The correct first move is to either:

- feed their data through the Palette-style wrapper in `palette`, or
- adapt the Palette wrapper to their own storage format.

## What Was Modified In `sam3`

On this machine, the only local code change found in `sam3` itself was:

- `sam3/model_builder.py`

That patch is a runtime compatibility convenience around packaged resource
lookup. It is **not** the core integration.

If a collaborator can use upstream SAM3 unchanged, that is preferable.

In other words:

- the Palette/SAM3 workflow does **not** fundamentally depend on a custom SAM3
  fork,
- but a local `sam3/model_builder.py` patch may still be helpful if their
  environment hits the same resource-resolution issue.

Two local SAM3 notebooks were also modified on this machine, but they are
examples and not part of the production Palette integration path.

## Required Data Contract

The current `run_sam_subject_masks.py` utility expects a Palette-style training
Zarr.

Minimum contract for the crop run:

- `crop_runs/<crop_run>/roi_images`
- `crop_runs/<crop_run>/roi_coordinates_full`
- `crop_runs/<crop_run>/bbox_norm_coords`
- `crop_runs/<crop_run>/frame_indices`
- `crop_runs/<crop_run>/detection_indices`
- `crop_runs/<crop_run>/detection_source`

Minimum contract for the keypoint run:

- `refined_keypoints_runs/<run>/keypoints_roi`
  - or `keypoints_runs/<run>/keypoints_roi`
- matching `frame_indices`
- matching `detection_indices`
- keypoint labels in attrs such as `keypoint_labels`

Additional root/crop metadata required for detect-box prompting:

- frame width/height resolvable from root or crop attrs

Important current limitation:

- even if the collaborator wants points-only prompting, the current wrapper
  still expects the crop-run arrays listed above, including `bbox_norm_coords`
  and the lineage arrays

So the simplest reuse path is:

- convert their data into a Palette-like training Zarr,
- or adapt `resolve_sam_subject_inputs(...)` inside
  `src/fisheye/utils/run_sam_subject_masks.py`

## Recommended Reuse Path

### Option 1: Their data can be represented as a Palette-style training Zarr

This is the easiest case.

They can:

1. materialize ROI crops into `crop_runs/<run>`
2. provide aligned keypoints in `refined_keypoints_runs/<run>` or
   `keypoints_runs/<run>`
3. run the existing `palette` SAM3 wrapper

### Option 2: Their data is not in Palette format

Do **not** start by hacking SAM3 dataloaders.

Instead, choose one of these:

1. write a small adapter that exports their data into a temporary
   Palette-compatible training Zarr
2. or adapt `run_sam_subject_masks.py` so its input resolver can read their
   native storage format

That keeps the SAM3-specific logic centralized and avoids creating a second
integration stack inside `sam3`.

## Commands To Reuse

### Inspect before writing

```bash
scripts/py -m fisheye.utils.run_sam_subject_masks \
  /path/to/training_like.zarr \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run <their_refined_run> \
  --output-run <planned_subject_mask_run> \
  --json
```

Use the inspect step first to verify:

- crop/keypoint row alignment
- prompt counts
- SAM3 runtime availability
- planned output contract

### Apply with point prompts plus detect-derived box

```bash
scripts/py -m fisheye.utils.run_sam_subject_masks \
  /path/to/training_like.zarr \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run <their_refined_run> \
  --output-run <planned_subject_mask_run> \
  --sam3-root /path/to/sam3 \
  --apply
```

### Apply with points only

```bash
scripts/py -m fisheye.utils.run_sam_subject_masks \
  /path/to/training_like.zarr \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run <their_refined_run> \
  --output-run <planned_subject_mask_run> \
  --sam3-root /path/to/sam3 \
  --no-box-prompt \
  --apply
```

## Output Contract

The current SAM3 wrapper writes a new:

- `subject_mask_runs/<run>`

with the existing Palette subject-mask schema:

- `label_schema_id = "subject_v1_union"`
- `mask_labels = ["subject_body", "eyes_union", "swim_bladder"]`
- typically only `subject_body` available for this phase-1 SAM3 workflow

So collaborators should expect a Palette subject-mask run, not a new standalone
SAM3-native result format.

## Viewer Support

For inspection inside `palette`:

- `src/fisheye/visualization/visualize_sam_subject_prompts.py`

Example:

```bash
scripts/py -m fisheye.visualization.visualize_sam_subject_prompts \
  /path/to/training_like.zarr \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run <their_refined_run> \
  --subject-run <their_subject_mask_run>
```

This shows:

- ROI crop
- prompt points
- optional box prompt
- stored `subject_body` mask overlay

## Paintera Is Optional

Paintera is only needed if the collaborator wants interactive manual cleanup of
Palette mask channels inside a Palette training Zarr.

That work lives in the `paintera` repo, not the SAM3 integration layer.

If they only want automatic SAM3 mask generation, they can ignore Paintera
entirely.

## Recommended Message To A Collaborator

If someone asks how to reuse this on their own data, the clean summary is:

> Start in `palette`, not in SAM3. The current working integration is a
> Palette-side wrapper that reads Palette-style training Zarr data, constructs
> SAM3 prompts, runs the upstream SAM3 runtime, and writes back a standard
> `subject_mask_runs/<run>`. Paintera is optional and only matters for manual
> cleanup. If your data is not already in Palette Zarr form, adapt the Palette
> wrapper or export a temporary Palette-like training Zarr. Do not begin by
> rewriting SAM3 dataloaders.

## Related Notes

- `docs/sam3_subject_mask_canary_plan.md`
- `docs/paintera_palette_subject_mask_workflow.md`
- `../sam3/docs/palette_zarr_subject_segmentation_workflow.md`
- `../sam3/docs/palette_zarr_sam_integration_todo.md`
