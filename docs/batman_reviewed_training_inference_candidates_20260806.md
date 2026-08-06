# Batman reviewed training inference candidates — 2026-08-06

Status: **COMPLETE, SELECTOR-INELIGIBLE BENCHMARK ARTIFACT**

This checkpoint validates that accepted sampled detections can provide exact,
lossless crop pixels to the existing terminal keypoint and subject-mask
producers. It does not approve either model output as training labels and does
not activate any production selector or registry row.

## Review-artifact bridge checkpoint

The sampled-training candidates now have an exact bridge into reviewable
surfaces without rerunning inference or misrepresenting their compact frame
axis as recording-level crop-v2:

- the crop materialization is wrapped in
  `palette.keypoint.training_crop_source_manifest` and retains separate
  sampled-local and acquisition-camera frame domains;
- raw keypoints, keypoint quality, refined keypoints, and body frame are
  republished through the existing strict v2 schemas and byte planner;
- the strict refined-keypoint output is an immutable snapshot, while an open
  `edit_delta_runs` generation binds edits by `instance_key`;
- terminal probability masks are finalized into dense `uint8 masks_roi` under
  an explicit selector-ineligible `editable_draft` lifecycle; and
- the complete copied training artifact is built on bounded node-local
  scratch, verified, copied to a hidden sibling, and made visible by atomic
  rename.

A real 181-row, 4.0 GiB local end-to-end smoke passed on 2026-08-06. It
preserved the 19 rowless negative-frame decisions, created all four strict
keypoint surfaces, created one open keypoint delta generation, and created the
dense four-component refined-mask draft (`subject_body`, `eye_left`,
`eye_right`, and `swim_bladder`) from the three raw model channels. The
resulting root, every new base, and the editable mask run remained
selector-ineligible; no registry or production selector was modified. This
local smoke is implementation evidence, not a durable published canary.

The maintained keypoint review backend is now delta-aware for strict immutable
bases. It requires exactly one active generation bound to the selected base,
verifies partition schemas, payload digests, key/hint identity, operation
semantics, and deterministic merge order, and builds a private in-memory
overlay. Saves publish a complete immutable partition before changing that
overlay. Legacy non-immutable runs retain their compatibility writer, but an
immutable base cannot be approved in place: its generation must first be
frozen and compacted into a new strict snapshot. The subject-mask reviewer can
use the dense editable draft after its task/session metadata is created.

A synthetic correction against an ordinary-copy Batman artifact proved the
boundary on 2026-08-06. The reviewer appended five landmark events, reopened
the artifact to reconstruct the same correction, and retained the refined-base
tree SHA-256 unchanged at
`0110866252e2c2ed3ccf1cbd5f7dc395e23aca150a6c5db70dfae3f6faae82b7`.
This is local integration evidence, not an accepted scientific edit.

Compaction additionally requires the full crop run manifest, not only its
digest in downstream keypoint manifests. The review-artifact publisher now
persists that manifest on `crop_runs/<run>.attrs.run_manifest` and validates
its digest against the refined-keypoint source binding. Review artifacts made
before this fix, including the local smoke above, require a metadata-only
republish/migration before they can be compacted; the compactor fails closed
instead of reconstructing missing provenance.

A full ordinary-copy republish exercised the corrected publisher across all
review surfaces. It completed with 181 review rows and receipt digest
`bd42c6029372e71f1b4b850d016c3f33111a034fa3728563ce218294a7896bdb`;
the persisted crop manifest's canonical digest was exactly the refined run's
bound digest
`9c6e55ad5443553952c1c49959f07c4b9d472fe6733b77edcd807e2d75dce8b3`.

After rebuilding that exact manifest for an ordinary-copy fixture and proving
its digest matched the already-bound value, a frozen-generation compaction
smoke passed. The selector-ineligible successor changed exactly landmark 4 of
row 0, preserved every other coordinate, bound the immediate parent manifest
and snapshot IDs, and recorded five events from one partition. Its local
receipt digest is
`e7d4bac8156bbe9485a0ca7963d53e9d16bf06ea18fdae7a8bc97e558ab4da79`.
No source selector, registry row, or production artifact changed.
The smoke uses the compactor's provisional manual-acceptance QC rule and is
therefore integration evidence only. Authority activation remains blocked
until the combined reviewed-publication gate binds and replays the exact
review-QC policy.

The combined immutable-publication boundary is now implemented by
`fisheye.utils.publish_reviewed_training_artifact_candidate`. It copies and
inventories the active review artifact, imports the receipt-bound compacted
keypoint successor, seals approved dense subject masks only in the copy,
consolidates the completed immutable artifact, verifies the consolidated
keypoint and mask declarations, and atomically publishes it without selectors
or registry changes. It refuses pending masks, non-frozen delta evidence,
tampered receipts, stale manifests, and concurrent source changes.

A disposable end-to-end smoke published
`/tmp/palette-reviewed-candidate-smoke.HJh9sg/published/reviewed.zarr` with
receipt digest
`8b2c4df5ee2876288e822d9827fa48c3a8f1a10dfce7501dd0cb4f1907233224`.
The source remained `editable_draft`; the candidate was `sealed_snapshot`,
`immutable_snapshot`, consolidated, and selector-ineligible. The smoke changed
approval fields only on its disposable source copy to exercise the gate, so it
is machinery evidence and not a scientific review or activation decision.

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

## Preprocessing contract decision

The comparison is now encoded as pose model-input contract v2 rather than as
an informal command-line choice. The model-specific contract is checked in at
`docs/diagnostics/batman_keypoint_v2_candidate_20260805/pose_model_input_contract_v2.json`.
It retains the immutable training evidence and exact weights digest from v1,
then adds two evidence-bound profiles for native 348x348 inputs:

- accepted: prepared tensor, centered zero pad to 352x352, `imgsz=352`, stride
  32, no Ultralytics spatial preprocessing;
- rejected: numpy-list, centered zero pad to 512x512, Ultralytics resize to
  256x256.

The accepted profile is limited to selector-ineligible candidate inference
because the comparison establishes detection yield, not landmark accuracy.
The whole-recording planner selects a profile by the exact native cache shape.
The terminal worker independently reconstructs that selection, reproduces a
digest-bound preprocessing probe, and rejects command-line geometry or mode
overrides. A model/native-shape pair without one exact accepted profile now
fails closed; training-source dimensions are no longer used to invent a
runtime transform.

Contract builders remain generic. Future trained models should publish their
reviewed native, submitted, and network extents with the model package. The
runtime consumes those dimensions from the contract rather than using a
Batman-, crop-, or stride-specific conditional.

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
- [x] Freeze the accepted and rejected profiles in a digest-bound v2 model
      contract and make exact native-shape selection fail closed.
- [ ] Review the 136 successful keypoint predictions before accepting any as
      training labels.
- [ ] Review sampled subject-mask predictions before accepting them as dense
      training labels.
- [ ] Export accepted keypoints and masks into the training authorities; do not
      promote these terminal predictions directly.
- [x] Use strict finalizers to produce float32 keypoint-v2 immutable bases,
      an instance-key edit generation, and dense editable refined subject-mask
      surfaces without relabelling the sampled frame axis as crop-v2.
- [ ] Publish the review artifact from an immutable Palette revision and hand
      its exact run paths to the maintained review clients.
- [x] Route maintained keypoint review reads and writes through the bound delta
      generation; never mutate the strict refined-keypoint base in place.
- [x] Implement and smoke the keypoint half of compaction into a new immutable,
      selector-ineligible, parent-bound refined snapshot.
- [x] Seal accepted dense masks only in a copied artifact and orchestrate the
      compacted-keypoint plus dense-mask publication as one immutable,
      selector-ineligible candidate.
- [x] Persist the complete bound crop manifest needed by the strict keypoint
      successor publisher; fail compaction on pre-fix artifacts until migrated.
- [ ] Keep the benchmark artifact outside production selectors and registry
      activation unless a separate promotion gate authorizes it.
- [ ] Bind and replay the exact manual keypoint QC policy, obtain real mask
      approvals, and run the resulting immutable candidate through Crimson
      before any authority activation.
