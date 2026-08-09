# Batman subject-mask review and retraining checkpoint — 2026-08-09

## Outcome

Four selector-ineligible subject-mask training-review artifacts are ready for
browser labeling. Each artifact contains 200 lossless 384×384 acquisition-crop
rows and a dense editable `uint8 [200,4,384,384]` `masks_roi` authority with
ordered components:

1. `subject_body`
2. `eye_left`
3. `eye_right`
4. `swim_bladder`

The immutable evidence chain is retained separately as raw subject masks,
canonical refined masks, subject-mask quality, and a bundle manifest. Browser
edits target only the dense editable refined run. No production selector,
registry activation, or analysis authority changed.

## Pinned inference contract

- model set: `subject_mask_cedar_shadow_omnifin0_gray_subject_v1_union_c6ff03ae_v001`
- model run: `subject_masks_union_all_components_v001`
- checkpoint SHA-256: `217da20cd6ed780f5efe2c16add7cb932f40f08aac2f6e44795c0c381283839c`
- validation Dice recorded by the model artifact: `0.9470`
- trained labels: `subject_body`, `eyes_union`, `swim_bladder`
- trained ROI shape: 512×512 grayscale
- native Batman ROI shape: 384×384 grayscale
- input transform: centered zero padding to 512×512, 64 pixels on every side
- output coordinate mapping: crop model output back to native 384×384 ROI
- probability representation: `uint8 [200,3,384,384]`
- probability inner chunks: `[32,1,384,384]`
- probability outer shard: `[2048,1,384,384]` (one shard per sampled artifact)

The padding transform validates the machinery without resampling pixels, but it
does not make the old 512-crop model distribution-equivalent to native
384-crop training. These predictions are review seeds, not model-promotion
evidence. The reviewed 384 data should be included when retraining the next
subject-mask model.

## Terminal inference results

All four terminal runs completed on the workstation A6000 in approximately
1.7–1.9 seconds per 200 rows. All were selector-ineligible, digest-bound to the
exact crop materialization and model artifact, and stored under
`subject_mask_shard_runs`.

| Arena | Body present | Eyes-union present | Swim-bladder present |
|---|---:|---:|---:|
| 1 | 200 | 200 | 197 |
| 2 | 200 | 200 | 185 |
| 3 | 200 | 200 | 196 |
| 4 | 200 | 200 | 200 |

Missing predictions remain explicit review cases. They were not silently
filled or converted into positive labels.

## Published review artifacts

| Arena | Artifact | Receipt digest |
|---|---|---|
| 1 | `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T20-12-57Z_arena_1_Batman/zarr/2026-07-21T20-12-57Z_arena_1_Batman_subject_mask_review_384_v1_training.zarr` | `e78a6efd1b2a159299e9c649e76f3bc35202ca4610edebaf67efe6c10f007592` |
| 2 | `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T20-12-57Z_arena_2_Batman/zarr/2026-07-21T20-12-57Z_arena_2_Batman_subject_mask_review_384_v1_training.zarr` | `595a94b5fee76209ffc2407f35535ea10d2e897e1ab4b25ee5f7c06854c5ad31` |
| 3 | `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T20-12-57Z_arena_3_Batman/zarr/2026-07-21T20-12-57Z_arena_3_Batman_subject_mask_review_384_v1_training.zarr` | `35c41bf82095a59b53452169a074a40b22e93c55f10836c1b51462257f6ccfac` |
| 4 | `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T20-12-57Z_arena_4_Batman/zarr/2026-07-21T20-12-57Z_arena_4_Batman_subject_mask_review_384_v1_training.zarr` | `e07b9d45be3f79e0ff266c670c90f72418dc086e45b6b1bbc62106876d6fbb3f` |

The physical size is approximately 4.19 GB per artifact because the current
training artifact also retains its sampled full-resolution images. Avoiding
that duplication remains a deferred storage optimization; it does not affect
the logical mask contract.

## Labeling tasks and backups

- task manifest: `docs/diagnostics/batman_subject_mask_review_tasks_20260809.json`
- task import: 16 applied, zero warnings
- tasks per recording: body, left eye, right eye, swim bladder
- rows per component task: 200
- labeling DB backup:
  `/home/delahantyj@hhmi.org/.palette/backups/labeling_work_before_batman_subject_masks_20260809.sqlite`
- exact mutable-Zarr backups:
  `/nvme1/palette_staging/labeling_backups/batman_201257_20260809/arena{1,2,3,4}`
- backup coverage: both keypoint and subject-mask review artifacts for every
  recording, eight targets total
- fixed-user server: `http://127.0.0.1:8797/my-datasets?expected_user=delahantyj`

Every restore remains operator-only and requires pausing the affected recording
assignment before replacing the current Zarr.

## Path to retraining and Batman re-analysis

- [x] Publish lossless acquisition-crop training materializations.
- [x] Publish keypoint prediction/review artifacts.
- [x] Publish dense subject-mask prediction/review artifacts.
- [x] Create guarded browser tasks and exact pre-labeling backups.
- [ ] Complete keypoint and all four component-mask review tasks.
- [ ] Freeze review state and validate dense mask geometry/QC invariants.
- [ ] Compact reviewed mutable state into new immutable training snapshots.
- [ ] Build separate task-specific merged corpora for detection, five-point
  pose, and subject masks, with recording-level train/validation/test splits.
- [ ] Retrain all three model families with exact input/preprocessing and
  augmentation provenance.
- [ ] Evaluate on held-out Batman recordings/cameras before registry promotion.
- [ ] Run one selector-ineligible full-recording Batman canary through Palette
  and Crimson.
- [ ] Promote model and storage authorities only after the canary passes.
- [ ] Re-run the Batman DAG: detection → refined detection → crop geometry →
  keypoints/quality/refinement/body frame → subject masks/quality/refinement →
  tracking/kinematics → chaser analytics.

