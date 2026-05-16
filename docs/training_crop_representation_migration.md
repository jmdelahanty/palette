# Training Crop Representation Migration

## Purpose

Migrate reviewed training examples to a crop-pixel representation that matches
the cluster inference/cache path, then export those examples to retrain
detection-adjacent ROI models, pose/keypoint models, and mask models without
mixing incompatible pixel contracts.

The immediate target is Orange monochrome camera video. Orange's current camera
recording path prepares encoder input as NV12 in CUDA memory. For monochrome
cameras, the camera intensity is copied into the NV12 Y plane and the UV plane
is filled with neutral 128 before NVENC writes compressed H.264/HEVC packets.
That makes a PyNvVideoCodec luma crop a source-aligned representation for these
videos, even though it does not byte-match historical Decord-derived training
crops.

## Current Situation

Historical training Zarrs commonly contain:

- `raw_video/images_full`: sampled full-resolution frames imported through the
  historical Decord/RGB-to-grayscale path.
- `crop_runs/<run>/roi_images`: materialized crops of `raw_video/images_full`.
- reviewed detections, keypoints, and masks whose coordinates are ROI-local or
  full-frame-local to the existing crop geometry.

Recent parity checks showed:

- `crop_runs/<run>/roi_images` exactly matches crops from
  `raw_video/images_full`.
- PyNvVideoCodec raw luma and reconstructed-RGB candidates do not byte-match
  those historical crops.
- The mismatch is a representation/decoder-history mismatch, not a crop
  geometry or row-mapping mismatch.

For model training this is acceptable only if we avoid mixing representations:
train-time crop pixels and inference-time crop pixels must use the same
contract.

## Target Representation

Preferred v1 contract for Orange monochrome camera ROI crops:

```text
name: orange_mono_pynvvc_luma_uint8_v1
source: Orange camera MP4 decoded by PyNvVideoCodec
source_encoder_boundary: NV12
mono_semantics: camera intensity copied to NV12 Y plane; UV neutral 128
image_representation: uint8_grayscale_roi_v1
shape: [roi, roi_height, roi_width]
padding: zero outside source-frame bounds
color_conversion: raw NV12 Y/luma plane crop; no RGB reconstruction
```

Open decision:

- Whether to use raw Y values or a limited-to-full luma expansion. The current
  cluster cache uses raw luma. If we choose range expansion later, it must be a
  distinct `roi_pixel_contract` and require a new crop run plus model retrain.

Non-goal:

- Do not overwrite historical crop runs. They remain valid for historical model
  provenance and regression comparisons.

## Zarr Migration Shape

For each reviewed training Zarr, create a new materialized crop run:

```text
crop_runs/
  <old_crop_run>/                         # historical pixels, unchanged
  <new_pynvvc_luma_crop_run>/
    roi_images                            (N, H, W) uint8
    frame_indices                         copied from old crop run
    roi_coordinates_full                  copied from old crop run
    source_refined_row_ids                copied when present
    detection_indices                     copied when present
    bbox_norm_coords                      copied when present
    crop_bbox_norm_coords                 copied when present
    attrs:
      crop_storage_mode                   "materialized"
      source_crop_run                     "<old_crop_run>"
      crop_migration_kind                 "pixel_representation_migration"
      label_coordinate_transform          "identity"
      frame_index_mapping                 "raw_video/original_frame_indices" | "direct"
      source_video_path                   resolved MP4 path used for decode
      roi_pixel_contract                  <orange_mono_pynvvc_luma_uint8_v1>
      source_roi_pixel_contract           <old crop contract>
      decoder_backend                     "PyNvVideoCodec"
      decoder_backend_version             recorded if available
      orange_video_semantics              "mono_nv12_y_uv128"
```

For sampled training Zarrs, crop `frame_indices` are often local sampled-frame
indices. The generator must map them through `raw_video/original_frame_indices`
before decoding the source MP4.

## Label Migration Policy

The crop geometry is unchanged, so the label transform is identity. Labels
should still be represented as new run surfaces or explicit migrated aliases so
exporters can require that labels reference the new crop run.

### Bounding Boxes

Full-frame detection boxes remain in full-frame coordinates and do not change.
If a crop run contains crop-local bbox arrays, copy them unchanged because ROI
top-left and ROI size are unchanged.

Required provenance:

```text
source_crop_run: <new_pynvvc_luma_crop_run>
source_bbox_run/source_refined_detect_run: <old reviewed detection/refined run>
label_coordinate_transform: identity
```

Detection training that uses downsampled full frames is not directly affected by
ROI crop migration. Detection-adjacent exports must still record the image
contract they train on.

### Keypoints

Keypoint coordinates are ROI-local. If the new crop run keeps identical
`roi_coordinates_full`, `roi_size`, and row order, keypoint arrays can be copied
or linked unchanged.

Preferred migration:

```text
keypoints_runs/<new_run> or refined_keypoints_runs/<new_run>
  arrays copied from old reviewed run
  attrs:
    source_keypoints_run/refined_keypoints_run: <old_run>
    source_crop_run: <new_pynvvc_luma_crop_run>
    source_crop_signature: <new_signature>
    source_roi_pixel_contract: <new contract>
    label_coordinate_transform: identity
    migrated_from_crop_run: <old_crop_run>
```

Do not silently pair old keypoint runs with new crop pixels unless the exporter
also writes this identity migration provenance into the training manifest.

### Masks

Mask arrays are ROI-local dense labels. With unchanged ROI geometry, mask pixels
can be copied unchanged.

Applicable stages:

- `eye_masks_runs`
- `refined_eye_masks_runs`
- `subject_mask_runs`
- `refined_subject_masks_runs`

Preferred migration:

```text
<mask_stage>/<new_run>
  masks_roi / mask_probs_roi / component metadata copied as appropriate
  attrs:
    source_<mask_stage>_run: <old_run>
    source_crop_run: <new_pynvvc_luma_crop_run>
    source_roi_pixel_contract: <new contract>
    label_coordinate_transform: identity
    migrated_from_crop_run: <old_crop_run>
```

Review approvals may be carried forward only if validation confirms row count,
row identity, ROI geometry, and label arrays are unchanged. The carried-forward
review status should reference the original reviewed run rather than pretending
the labels were newly reviewed.

## Export Requirements

Training exporters must be able to select by crop pixel contract, not just by
latest crop run.

Required manifest fields:

```json
{
  "crop_run": "<new_pynvvc_luma_crop_run>",
  "roi_pixel_contract": {
    "name": "orange_mono_pynvvc_luma_uint8_v1"
  },
  "source_label_run": "<migrated_label_run>",
  "source_label_transform": "identity",
  "source_video_path": "<mp4>",
  "frame_index_mapping": "raw_video/original_frame_indices"
}
```

Model registry entries should record:

- training image/crop pixel contract name and full JSON contract,
- expected inference crop pixel contract,
- crop representation migration run/version,
- source training set manifest,
- decoder backend and version where available.

Runtime inference should warn or fail when a model trained on one
`roi_pixel_contract` is asked to consume another.

## Validation Gates

Before marking migrated training data usable:

1. Inventory every selected training Zarr:
   - source MP4 exists on the target filesystem,
   - camera/video is Orange monochrome or otherwise explicitly compatible,
   - old crop run and reviewed label runs exist.
2. Verify geometry identity:
   - `frame_indices`, `roi_coordinates_full`, ROI size, and row count match the
     source crop run.
3. Verify pixel generation:
   - new `roi_images` are reproducible from source MP4 plus recorded contract,
   - row order is stable,
   - checksums or sample parity reports are stored.
4. Verify label identity:
   - copied keypoint/mask/bbox arrays are byte-identical to source labels,
   - label run attrs point to the new crop run,
   - review status lineage points back to the original approved source.
5. Export smoke:
   - detection/keypoint/mask exporters select the new crop contract,
   - merged training artifacts validate,
   - a tiny train or dataloader smoke reads the new images and labels.
6. Model smoke:
   - train a small or short-run model,
   - run predictions on held-out reviewed examples,
   - compare performance against historical model and inspect overlays.

## Batch Migration Workflow

Use the batch wrapper for reviewed per-recording training Zarrs. It is
intentionally dry-run by default and writes JSONL so failures can be inspected
per archive before any bulk write.

Dry-run against approved registry training Zarrs:

```bash
scripts/py -m fisheye.utils.batch_migrate_training_crop_pixel_contract \
  --registry /nvme1/palette_registry.sqlite \
  --zarr-use training \
  --required-review-state approved \
  --required-review-intended-use training \
  --jsonl-report /tmp/pynvvc_training_crop_migration_dryrun.jsonl \
  --summary-json /tmp/pynvvc_training_crop_migration_dryrun.summary.json
```

Apply only after the dry-run report is clean:

```bash
scripts/py -m fisheye.utils.batch_migrate_training_crop_pixel_contract \
  --registry /nvme1/palette_registry.sqlite \
  --zarr-use training \
  --required-review-state approved \
  --required-review-intended-use training \
  --apply \
  --jsonl-report /tmp/pynvvc_training_crop_migration_apply.jsonl \
  --summary-json /tmp/pynvvc_training_crop_migration_apply.summary.json
```

Default behavior:

- Candidate discovery uses `zarr_use=training` and skips merged/exported
  training artifacts under training dataset roots.
- Registry-discovered candidates are gated by keypoint review state
  `approved` and intended use `training` unless `--approval-family none` is
  passed.
- The target crop run name is `<source_crop_run>_pynvvc_luma_v1`.
- Migrated label run names use the same `_pynvvc_luma_v1` suffix and point at
  the new crop run with `label_coordinate_transform=identity`.
- By default, the batch wrapper migrates each label family's latest source run
  only. Pass `--all-label-runs` only when doing a full archival migration of
  historical label runs; it is much slower and creates many more arrays.
- `crop_runs/latest` and label `latest` pointers are not changed unless
  `--set-latest` is passed.
- Existing completed target crop runs with the expected
  `orange_mono_pynvvc_luma_uint8_v1` contract are reused, so the batch is
  idempotent. Existing incomplete or mismatched target crop runs require
  inspection or `--overwrite`.
- After `--apply`, the wrapper runs a small PyNvVC parity sample by default;
  use `--parity-sample-count 0` to skip it.

Decode access policy:

- `fisheye.utils.regenerate_training_crops_pynvvc` supports
  `--decode-mode auto|sequential|indexed`.
- `auto` keeps the sequential demux/decode path for dense frame windows and
  uses PyNvVideoCodec `SimpleDecoder.get_batch_frames_by_index(...)` for sparse
  sampled training Zarrs.
- The indexed path is the correct default for existing training imports whose
  crop rows are sampled every ~100 source frames. Sequentially decoding
  `0..max(source_frame_indices)` is correct but too slow for that sparse
  layout.
- The effective access mode is recorded in `crop_runs/<run>.attrs` as
  `decode_mode_requested` and `decode_mode_effective`.

## Implementation Checklist

1. Add a crop regeneration utility.
   - Inputs: training Zarr, old crop run, source video override, target run
     name, pixel contract.
   - Output: new materialized crop run using PyNvVideoCodec luma.
   - Must support sampled training Zarr frame mapping.
   - Must use indexed PyNvVC reads for sparse sampled training rows.
   - Initial implementation: `scripts/py -m
     fisheye.utils.regenerate_training_crops_pynvvc`.
2. Add label migration utilities.
   - Keypoint run identity migration.
   - Subject/eye mask run identity migration.
   - Optional bbox/crop-local metadata migration.
3. Add validators.
   - Crop geometry identity validator.
   - Label identity validator.
   - Pixel contract/provenance validator.
4. Update training exporters.
   - Accept `--crop-pixel-contract` or equivalent selector.
   - Require label runs to reference the selected crop run or have explicit
     identity migration provenance.
   - Write contract fields into manifests.
5. Update registry surfaces.
   - Track crop pixel contract for training candidates.
   - Allow filtering training candidates by crop contract.
   - Record model expected input contract.
6. Batch migrate training Zarrs.
   - Dry-run inventory first.
   - Apply to all approved training Zarrs.
   - Refresh registry quality/profile rows.
   - Initial implementation: `scripts/py -m
     fisheye.utils.batch_migrate_training_crop_pixel_contract`.
7. Export and retrain.
   - Export detection/keypoint/mask datasets from the migrated crop contract.
   - Train new models.
   - Compare held-out and full-video smoke performance.

## Open Questions

- Should the first production contract be raw luma or limited-to-full-range
  luma?
- Do all existing training Zarr source videos have enough metadata to prove
  Orange monochrome NV12-Y semantics, or do we need manual allowlists?
- Should migrated label runs physically copy arrays or use lightweight alias
  runs with source references?
- Should old review approvals be carried forward automatically after identity
  validation, or should migrated datasets require a small visual re-review
  sample?
- Should model training use single-channel grayscale inputs, or expand luma to
  three identical channels for compatibility with current YOLO/UNet code paths?

## Recommended First Slice

Implement the crop regeneration utility and run it on one known reviewed
training Zarr. Then migrate one keypoint run and one subject-mask/refined-mask
run by identity, export a small manifest from those migrated surfaces, and run
the dataloader/model smoke before batch migration.
