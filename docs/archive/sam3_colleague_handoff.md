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

If the goal is actual SAM3 fine-tuning rather than prompt-based SAM3 inference,
see `docs/sam3_finetuning_from_palette_data.md`. Fine-tuning should start from
a Palette-to-COCO/SAM3 export adapter over reviewed dense subject-mask training
data, not from the current `run_sam_subject_masks.py` runtime wrapper.

## Short Answer

If a collaborator asks "where is the real work for using SAM3 on our data?",
the answer is:

- **Primary repo:** `palette`
- **Runtime dependency:** `sam3`
- **Optional editor:** `paintera`

The key Palette entrypoints are:

- `src/fisheye/utils/run_sam_subject_masks.py`
- `src/fisheye/utils/run_sam_subject_masks_batch.py`
- `src/fisheye/visualization/visualize_sam_subject_prompts.py`

These are the files that:

- read Palette training Zarr data,
- batch-discover training Zarr archives when needed,
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

## Output Semantics

The current SAM3 wrapper is a prompt-based body-mask creator, not the same
cleanup/refinement path used after U-Net subject-mask inference.

For each eligible ROI, Palette asks SAM3 for candidate masks, selects the
candidate with the highest SAM-predicted quality score, and writes:

- `masks_roi[:, subject_body] = selected_logits > 0`
- `mask_probs_roi[:, subject_body] = sigmoid(selected_logits)`
- `metrics/sam_quality_score[:, subject_body] = selected_candidate_score`

The wrapper does not then run Palette's U-Net smart-finalizer cleanup. In
particular, SAM3 creation currently does not apply additional Palette-side
morphology closing, hole filling, keep-largest-component cleanup, or
removed-mass / changed-area finalization metrics.

That contrast is intentional. U-Net `subject_mask_runs` commonly become
`refined_subject_masks_runs` through `smart_finalize_subject_masks_v1`, where
those cleanup policies are explicit and parameterized. SAM3 outputs should stay
as selected SAM masks unless a later, explicit SAM3-refinement policy is added
with its own recorded parameters.

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
  --crop-run <their_crop_run> \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run <their_refined_run> \
  --output-run <planned_subject_mask_run> \
  --sam3-root /path/to/sam3 \
  --apply-limit 16 \
  --apply
```

Omit `--apply-limit` only when intentionally running the full eligible row
surface. For production/full runs, record that decision in the run notes or
submission manifest.

### Apply with points only

```bash
scripts/py -m fisheye.utils.run_sam_subject_masks \
  /path/to/training_like.zarr \
  --crop-run <their_crop_run> \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run <their_refined_run> \
  --output-run <planned_subject_mask_run> \
  --sam3-root /path/to/sam3 \
  --no-box-prompt \
  --apply-limit 16 \
  --apply
```

### Submit a bounded cluster smoke

Use the bsub wrapper when the SAM3 checkout and optional checkpoint are visible
from Janelia compute nodes:

```bash
scripts/submit_sam_subject_masks_bsub.sh \
  --zarr /groups/.../recording/zarr/recording_analysis.zarr \
  --crop-run <crop_run> \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run <refined_keypoints_run> \
  --output-run <planned_subject_mask_run> \
  --sam3-root /groups/johnson/johnsonlab/jeremy/gitrepos/sam3 \
  --checkpoint /groups/johnson/johnsonlab/jeremy/models/sam3/sam3.pt \
  --python-bin /groups/ahrens/home/delahantyj/miniforge3/envs/palette-sam3/bin/python \
  --apply \
  --apply-limit 16 \
  --profile-timings \
  --no-hf-download \
  --submit
```

The wrapper is dry-run by default. It refuses `--apply` without either
`--apply-limit N` or `--allow-full-apply`, which prevents accidental
full-recording SAM writes while the runtime remains experimental.

### Make SAM3 visible to cluster jobs

Do not make SAM3 a required Palette submodule yet. For cluster use, maintain an
external SAM3 checkout on shared storage, for example:

```bash
/groups/johnson/johnsonlab/jeremy/gitrepos/sam3
```

The local workstation checkout can be mirrored there with `rsync` once the
intended SAM3 state is chosen. Prefer excluding ad-hoc `outputs/` directories
and preserving the source checkout state:

```bash
rsync -a --delete \
  --exclude outputs/ \
  --exclude .ipynb_checkpoints/ \
  /home/delahantyj@hhmi.org/gitrepos/sam3/ \
  /groups/johnson/johnsonlab/jeremy/gitrepos/sam3/
```

If `--checkpoint` is not supplied, the SAM3 runtime may try its own default or
Hugging Face checkpoint resolution. For reproducible cluster jobs, use the
compute-node-visible checkpoint path and pass `--no-hf-download`:

```bash
/groups/johnson/johnsonlab/jeremy/models/sam3/sam3.pt
```

This file is the `sam3.pt` checkpoint from the gated Hugging Face
`facebook/sam3` repository. It was placed on `/groups` on `2026-07-01` so LSF
jobs do not depend on compute-node Hugging Face auth/cache state.

The Palette cluster environment also needs SAM3's runtime Python dependencies.
Prefer an isolated SAM3-specific clone of `palette-py311`, then pass it to the
bsub wrapper with `--python-bin`. At minimum, the Python environment used by
the bsub job must import:

```bash
PALETTE_PYTHON=/groups/ahrens/home/delahantyj/miniforge3/envs/palette-sam3/bin/python \
  scripts/py - <<'PY'
for name in ("huggingface_hub", "iopath", "timm", "einops", "torch", "torchvision", "PIL"):
    __import__(name)
    print(name, "ok")
PY
```

On 2026-07-01, the cluster SAM3 environment was verified on a compute node with
`numpy 2.2.6`, `zarr 3.1.6`, `opencv 4.13.0`, `torch 2.5.1`, `torchvision
0.20.1`, `huggingface_hub`, `iopath`, `timm`, `einops`, `PIL`, and
`pycocotools` importable. The environment is intentionally a SAM3-specific
clone of `palette-py311` with SAM3 installed as an editable external checkout.

The first bounded RedScare v3 training-Zarr cluster apply also succeeded on
`2026-07-01`:

- job `151924691`
- host `e10u08`
- target:
  `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare/zarr/2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr`
- crop run:
  `crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_1_RedScare`
- keypoint run:
  `refined_keypoints_training_review_red_scare_traditional_v3_seed_20260625_01`
- output run:
  `sam_subject_masks_redscare_training_a1_v3_apply16_smoke_20260701_01`
- result: `16/16` selected rows segmented, all non-empty, stderr empty,
  duration about `30.7s`

The current reviewed RedScare arena-1 training surface is not that 16-row smoke
run; it is the already approved composed refined run:

```text
refined_subject_masks_runs/refined_subject_masks_sam3_body_existing_eye_swim_red_scare_v3_canary_20260628_01
```

That refined run combines the cleaned SAM3 `subject_body` source with the
existing reviewed `eye_left`, `eye_right`, and `swim_bladder` components and has
all four channels populated for `200/200` sampled training rows.

### Batch dry run across training archives

```bash
scripts/py -m fisheye.utils.run_sam_subject_masks_batch \
  /nvme1/recordings \
  --recursive \
  --zarr-use training \
  --sam3-root /path/to/sam3
```

Use the batch dry run first to verify:

- which archives are eligible
- which crop/keypoint runs will be used
- which output run names are planned
- which archives are skipped because the target output already exists

### Batch apply across training archives

```bash
scripts/py -m fisheye.utils.run_sam_subject_masks_batch \
  /nvme1/recordings \
  --recursive \
  --zarr-use training \
  --sam3-root /path/to/sam3 \
  --apply
```

If a collaborator wants a fixed output run name across the batch, add:

```bash
  --output-run <planned_subject_mask_run>
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
