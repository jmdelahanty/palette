# SAM3 Fine-Tuning From Palette Data

Last verified: 2026-06-25

## Purpose

Document the practical path for fine-tuning Meta SAM3 from Palette training data
without confusing it with the existing Palette SAM3 inference wrapper.

This note is intentionally design-level. It does not make SAM3 a required
Palette dependency and does not add a Palette-owned SAM3 training stack.

## Current State

Palette currently uses SAM3 as an optional external runtime/pseudo-label
generator:

- Palette resolves ROI crops, keypoints, and detection geometry.
- Palette builds SAM3 prompts.
- SAM3 runs as a sibling checkout/runtime dependency.
- Palette writes outputs back to `subject_mask_runs/<run>`.

That working integration lives in:

- `src/fisheye/utils/run_sam_subject_masks.py`
- `src/fisheye/utils/run_sam_subject_masks_batch.py`
- `docs/archive/sam3_colleague_handoff.md`
- `docs/sam3_subject_mask_canary_plan.md`

The existing verified workflow did not require modifying SAM3 dataloaders,
collators, or training code.

## Official SAM3 Training Surface

The official `facebookresearch/sam3` repository now includes a training path:

- `README_TRAIN.md`
- `sam3/train/train.py`
- Hydra configs under `sam3/train/configs/`
- COCO-style image dataset loaders under `sam3/train/data/`

The official examples are built around Roboflow/ODinW-style datasets and Hydra
configuration files. The trainer can run locally or via its cluster launcher.

Important implication: SAM3 fine-tuning is not a one-command extension of
Palette's current prompt wrapper. The clean boundary is an export adapter from
Palette training zarrs into a SAM3-supported dataset format.

## Recommended First Target

Start with a narrow subject-body fine-tune.

Use:

- images: ROI crops from Palette training zarrs
- masks: dense reviewed `subject_body` masks
- category text: `fish` or `zebrafish`
- task: single-class instance/semantic-ish subject-body segmentation

Avoid in the first canary:

- eye-left / eye-right component segmentation
- swim-bladder segmentation
- multi-component masks
- direct training from compact analysis stores (`mask_bitpacked` or `mask_rle`)
- direct SAM3 training from Palette zarrs

Rationale:

- `subject_body` is the highest-value, most SAM-like object mask.
- The current Palette SAM3 path has historically produced body masks first.
- Dense training zarrs are already the canonical editable/reviewable training
  source.

## Palette Source Contract

Use training zarrs, not compact analysis zarr mask stores, as the first export
source.

Required Palette source surfaces:

```text
crop_runs/<crop_run>/roi_images
crop_runs/<crop_run>/frame_indices
crop_runs/<crop_run>/roi_coordinates_full
subject_mask_runs/<run>/masks_roi
subject_mask_runs/<run>.attrs["mask_labels"]
subject_mask_runs/<run>.attrs["label_schema_id"]
```

Optional but recommended:

```text
refined_subject_masks_runs/<run>/masks_roi
refined_subject_masks_runs/<run>/source_crop_row_ids
refined_subject_masks_runs/<run>.attrs["component_review_statuses"]
refined_subject_masks_runs/<run>.attrs["source_crop_run"]
```

Source selection policy:

1. Prefer reviewed/refined dense training masks when available.
2. Require explicit label lookup by `mask_labels`; do not assume channel index.
3. For the first exporter, include only rows where `subject_body` is present and
   approved/usable.
4. Hold out validation rows by recording or by explicit split manifest, not by
   random row shuffling alone.

Training-zarr policy:

- Training zarrs should stay dense `uint8` for reviewed labels.
- If the source is an analysis zarr with compact masks, materialize through
  `MaskStore` into a dense training/export surface before producing SAM3 data.

## SAM3 Export Format

The practical first export should be COCO-style:

```text
sam3_palette_export/
  images/
    <image_id>.png or <image_id>.jpg
  annotations_train.json
  annotations_val.json
  metadata.json
```

COCO fields should include:

- `images`
  - `id`
  - `file_name`
  - `width`
  - `height`
- `annotations`
  - `id`
  - `image_id`
  - `category_id`
  - `bbox` in COCO `xywh`
  - `area`
  - `segmentation` as COCO RLE or polygons
  - `iscrowd = 0`
- `categories`
  - one category first: `{id: 1, name: "fish"}`

Use COCO RLE for masks unless there is a strong reason to emit polygons.
RLE preserves raster masks directly and avoids contour simplification choices.

Record Palette provenance in `metadata.json`:

- source zarr path
- source dataset id if registry-backed
- crop run
- mask run
- label schema id
- exported label
- source row ids
- frame indices
- source mask storage format
- Palette commit
- SAM3 export script version

## SAM3 Config Direction

Start from the official Roboflow/COCO image-training config and adjust:

- dataset root: Palette SAM3 export directory
- train/val annotation JSON paths
- category prompt/name: `fish` or `zebrafish`
- checkpoint path / Hugging Face access
- experiment log dir
- batch size / workers for the target GPU
- segmentation loss enabled

The current official Roboflow fine-tuning config includes a commented
segmentation-loss section and defaults that are detection-oriented in places.
For Palette subject-body masks, verify that mask loss is actually enabled
before treating any run as a segmentation fine-tune.

Do not silently run a box-only fine-tune and call it mask fine-tuning.

## Validation Plan

Minimum canary:

1. Export one small reviewed training zarr to SAM3 COCO format.
2. Load the export through SAM3's COCO loader.
3. Train for a short smoke run.
4. Run inference on held-out ROI crops.
5. Compare against held-out reviewed masks.
6. Write predictions back into a throwaway Palette training zarr as
   `subject_mask_runs/<sam3_finetune_run>`.
7. Inspect in the normal Palette/Crimson/web review path.

Metrics:

- Dice
- IoU
- mask-present rate
- false positive area outside body
- boundary/contour qualitative review

Qualitative validation remains required. A high Dice score can still hide bad
eye/tail boundary behavior that matters downstream.

## Recommended Implementation Order

1. Add a read-only export planner:
   `scripts/py -m fisheye.utils.prepare_sam3_subject_body_finetune_export`
2. Add an apply mode that writes COCO-style images/annotations.
3. Add an export validator that loads the COCO JSON and decodes a sample of RLE
   masks.
4. Add a minimal SAM3 config template outside the main Palette runtime path.
5. Run one local/cluster smoke using the sibling `sam3` checkout.
6. Only after a successful smoke, decide whether to wrap SAM3 training jobs in
   Palette cluster submission helpers.

## Risks

- SAM3 training requires a separate environment (`python>=3.12`, recent PyTorch,
  CUDA, and Hugging Face checkpoint access). Do not fold this into
  `palette-py311`.
- Official configs are open-vocabulary/concept-oriented; they may need careful
  changes to train masks rather than boxes.
- Palette ROI crops are grayscale/luma-derived while SAM3 expects RGB-like
  images. The exporter should explicitly document whether it writes replicated
  luma RGB, raw grayscale PNG, or another representation.
- Fine-tuning SAM3 may be slower and operationally heavier than training the
  current U-Net subject-mask models.
- SAM3 may be better as a pseudo-label generator than as the production runtime
  model for high-throughput analysis.

## Decision Guidance

Use SAM3 fine-tuning if:

- reviewed subject-body labels remain limited,
- zero-shot/prompted SAM3 is close but not good enough,
- pseudo-label quality matters more than runtime speed,
- and maintaining a separate SAM3 environment is acceptable.

Prefer Palette's native subject-mask models if:

- the goal is high-throughput inference,
- labels are already curated in dense training zarrs,
- or the model must integrate cleanly with current registry/model deployment.

## Related Docs

- `docs/archive/sam3_colleague_handoff.md`
- `docs/sam3_subject_mask_canary_plan.md`
- `docs/subject_mask_training_artifact_contract.md`
- `docs/subject_mask_runs_contract.md`
- `docs/refined_subject_masks_runs_contract.md`
