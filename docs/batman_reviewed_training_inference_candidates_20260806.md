# Batman reviewed training inference candidates — 2026-08-06

Status: **COMPLETE, SELECTOR-INELIGIBLE BENCHMARK ARTIFACT**

This checkpoint validates that accepted sampled detections can provide exact,
lossless crop pixels to the existing terminal keypoint and subject-mask
producers. It does not approve either model output as training labels and does
not activate any production selector or registry row.

## Artifact

- Published artifact:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/batman_training_canary_20260806_v1/2026-07-21T19-38-32Z_arena_2_Batman_reviewed_inference_candidates_v1.zarr`
- Published v005-profile comparison artifact:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/batman_training_canary_20260806_v1/2026-07-21T19-38-32Z_arena_2_Batman_reviewed_inference_candidates_v2.zarr`
- Source crop run: `crop_runs/crop_reviewed_348_images_full_v1`
- Positive crop rows: 181
- Reviewed negative frames: 19; these intentionally have no crop or inference row
- Crop-materialization binding SHA-256:
  `e73d1b84de3a7e9eb77de3007a53acc82c8eee565983eed9595e477dc415a3b7`
- Candidate-publication receipt SHA-256:
  `38c023b44f07a2561ca1e299972a60361e3ad6e3117a70139aaf5ea7267bd0e9`
- Physical content SHA-256:
  `8f4b0bbdc98e8c8ae609e637bc1bbbaaf56ed7db6e757c90d9b0f540c83f501e`
- Physical inventory: 813 files, 4,288,419,987 bytes
- Palette commit: `0af2a32effbe943b4b9695ecd10b3b79e8fa6e19`

The artifact was copied from node-local `/nvme1` storage to a hidden sibling,
fully hashed, validated through direct and consolidated metadata, and made
visible with a same-parent atomic rename.

## Keypoint terminal

- Run:
  `keypoint_shard_runs/keypoints_batman_training_reviewed_348_terminal_v1`
- Model run: `pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2`
- Model SHA-256:
  `cce63d534a8f1491db1e2c71cb9236768c445722013dc39faeaf62a9d0a9a377`
- Ordered schema: `traditional_v2` with five labels:
  `swim_bladder`, `eye_left`, `eye_right`, `snout_tip`, `tail_tip`
- Pixel transform: centered zero padding from 348x348 to 512x512
  (82 pixels on every side), followed by network preprocessing to 256x256
- Model stride: 32
- Execution device: CPU
- Result: 0 successful poses and 181
  `no_pose_detection_above_threshold` failures
- Runtime: approximately 1.6 seconds for model inference

The zero-pose result is scientific evidence that this model is incompatible
with the enforced 348-to-512-to-256 profile. It is not evidence that the model
cannot respond to Batman crops generally, and it is not a crop, identity, or
publication failure. The terminal retains one failure-code row for every
accepted crop identity and is intended to validate machinery while a model is
retrained on the intended 348-pixel crop domain.

The terminal is explicitly `legacy_noncanonical`, uses float64 working arrays,
and may only feed the strict v2 finalizer. It is not itself a canonical
keypoint-v2 publication.

## v005 tensor-profile comparison

The same 181 reviewed crops were rerun with the exact successful v005 input
profile while retaining the identical model weights and threshold:

- Run:
  `keypoint_shard_runs/keypoints_batman_training_reviewed_348_v005_tensor_terminal_v1`
- Input mode: `tensor`
- Pixel transform: centered zero padding from 348x348 to 352x352
  (two pixels on every side)
- Submitted `imgsz`: 352
- Model stride: 32
- Confidence threshold: 0.25
- Execution device: CPU
- Successful poses: 136/181 (75.14%)
- Explicit failures: 45/181, all
  `no_pose_detection_above_threshold`
- Mean successful pose confidence: 0.8406
- Mean successful keypoint confidence: 0.9990
- Runtime: approximately 3.3 seconds

The earlier full-recording v005 run used this profile and reached 70.93% yield.
The reviewed-crop result therefore reproduces its behavior closely. Because the
model SHA-256 is identical between the 0% and 75.14% runs, this comparison
isolates preprocessing profile as the cause of the yield collapse. It does not
by itself establish landmark accuracy; the 136 predictions require visual or
manual review before use as training labels.

The comparison artifact records:

- comparison receipt SHA-256:
  `c900247af3e2cc92be8a1c7367d30ffda8b0448b0dcad5ba1db79c8d909c54e4`
- physical content SHA-256:
  `3d9cfe189ba23bf486c2bb05cb122f0803d7bfe9dee417b97576ceb165d4ffb4`
- physical inventory: 861 files, 4,288,650,487 bytes
- direct/consolidated declaration digest:
  `82a529117703c5e37c9815870d8dd0dd27116dc79d9dc0dafa06884049da3fd0`

All row identities, frame counters, failure fills, and coordinate bounds passed.
The v1 artifact was not modified; v2 is a separate immutable derivative.

## Subject-mask terminal

- Run:
  `subject_mask_shard_runs/subject_masks_batman_training_reviewed_348_terminal_v1`
- Model run: `subject_masks_union_all_components_v001`
- Model SHA-256:
  `217da20cd6ed780f5efe2c16add7cb932f40f08aac2f6e44795c0c381283839c`
- Components: `subject_body`, `eyes_union`, `swim_bladder`
- Pixel transform: centered zero padding from 348x348 to 512x512
- Execution device: CPU
- Payload: `uint8 [181, 3, 348, 348]` probability masks
- Result: all 181 rows contain nonzero probability payloads
- Runtime: approximately 43.1 seconds

These outputs require review before they become segmentation training labels.
The run contains probability masks only; it does not publish authoritative
dense refined masks, quality, refinement, contours, or display caches.

## Validation evidence

- Every `instance_key`, `source_refined_row_id`, source frame index, and source
  acquisition frame index in both terminal runs exactly matches the 181-row
  reviewed crop input.
- Both terminal runs bind the exact crop-materialization digest above.
- Both runs have `palette_run_completion_status=complete` and
  `stage_selector_eligible=false`.
- Neither terminal parent has `latest`, `latest_complete`, `authoritative_run`,
  or `selected_run` set.
- Direct and consolidated declarations match exactly:
  - crop subtree: 16 nodes, declaration digest
    `6d1c5a1006adf62a1d4faf38d8041fbd676adcfa312ecb51e0636d43f35814af`
  - keypoint subtree: 26 nodes, declaration digest
    `80cd9088ccdfc7f5f4985fede094d29458de9fba4247696c1344315ca0f48b58`
  - subject-mask subtree: 25 nodes, declaration digest
    `289c4db82e6555d02838e7a595512c6e69ef6da50b52602c3438d55b7a392c10`
- The final root publication receipt is identical through direct and
  consolidated reads.

Consolidation emitted Zarr's standard warning that consolidated metadata is
not yet part of the Zarr v3 specification and a warning for the existing
`worker_semantic_receipt.json` sidecar. Both were recorded; neither changed the
validated inline metadata result.

## Follow-up checklist

- [ ] Retrain or fine-tune the five-point pose model with reviewed 348x348 crops
      while preserving the explicit 348-to-model transform contract.
- [x] Rerun the keypoint terminal with the v005 352-tensor profile and require
      nonzero, reviewable successes; 136/181 passed inference.
- [ ] Review the 136 successful keypoint predictions before accepting any as
      training labels.
- [ ] Review sampled subject-mask predictions before accepting them as dense
      training labels.
- [ ] Export accepted keypoints and masks into the training authorities; do not
      promote these terminal predictions directly.
- [ ] Use strict finalizers to produce canonical float32 keypoint-v2 and dense
      refined subject-mask surfaces when accepted labels exist.
- [ ] Keep the benchmark artifact outside production selectors and registry
      activation unless a separate promotion gate authorizes it.
