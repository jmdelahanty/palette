# Five-keypoint v3 Batman-training checkpoint — 2026-08-07

Verdict: **PASS FOR A SELECTOR-INELIGIBLE MODEL CANDIDATE; PRODUCTION MODEL
PROMOTION IS NOT YET AUTHORIZED**

## What this checkpoint proves

Palette composed and published an immutable 61-source, five-keypoint training
corpus, placed the reviewed Batman contribution in the training partition, read
the corpus through the production pose loader, and completed a contract-bound
30-epoch warm-start run on the local RTX A6000.

No training-dataset registry row, model-registry row, production selector, or
analysis archive was changed.

## Immutable corpus

Published Zarr:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/keypoint_merged_v3/five_point_reviewed_full_v3_batman_train_v003/pose_five_point_reviewed_full_v3_batman_train_v003_merged.zarr`

Local training copy:

`/nvme1/palette_staging/keypoint_training/five_point_reviewed_full_v3_batman_train_v003_local_20260807/pose_five_point_reviewed_full_v3_batman_train_v003_merged.zarr`

The immutable publication contains:

- 61 exact sources;
- 12,704 full-supervision poses and 63,520 labeled points;
- 29 indivisible leakage groups;
- 9,448 training rows, 3,256 validation rows, and no test rows;
- no train/validation row overlap;
- exact `pose_skel_traditional_v2` five-keypoint ordering;
- checked float32 keypoint storage; and
- centered zero-padding of the reviewed 348×348 Batman crops onto the 512×512
  training canvas without resizing.

Batman is source index 52. Its 181 rows occupy merged rows 10,786 through
10,966. All 181 are in `train_indices`; none are in validation or test.

Publication hashes:

- source manifest: `897b3df739900036d7b83de563a830d101d4a2d77a84cac5a4a67af725226c26`;
- merged manifest: `7209e932f9a392db2ad67aa9f6d51da29021fe1df4a22caa063e5aebc763a6a9`;
- summary: `7f9bc191ee45e2f7f38fd7232d9b2dc7584999b4ca7fa16aa58f69e5db6a8336`.

The v3 writer-side validator passed after publication. The production pose
reader then loaded the artifact-controlled split and produced a diagnostic batch
of images `[16,3,512,512]`, boxes `[16,4]`, and keypoints `[16,5,3]`.

## Exact training contract

Run directory:

`/nvme1/palette_staging/keypoint_training_runs/pose_five_point_reviewed_full_v3_batman_train_v003_adamw_warm_30e_20260807`

The immutable run-input snapshots are under its `inputs/` directory. The
execution-manifest SHA-256 is
`6da887b887794dd2e4c69f72e4467d50dfc4ba2e0aff454999340b4e0d18e283` and the
training-config SHA-256 is
`77b2ae5a895498b81e2506e6579744ffb69799432bc40da2b9a55a4b75267046`.

The starting model was the explicit May five-keypoint checkpoint. Its SHA-256
is `cce63d534a8f1491db1e2c71cb9236768c445722013dc39faeaf62a9d0a9a377`.

The effective training contract was:

- 30 epochs, batch 64, 512×512 model input, seed 42;
- AdamW, learning rate 0.001, momentum/beta1 0.9, and configured weight decay
  0.0005;
- pose/kobj/box/cls/dfl weights 12/1/7.5/0.5/1.5;
- eight persistent training workers with two prefetched batches;
- deterministic, unaugmented validation; and
- registry logging disabled.

Batch 64 intentionally matches Ultralytics' nominal batch size. The runtime
receipt confirms the actual decayed AdamW parameter group used 0.0005 rather
than the silently doubled 0.001 observed in the earlier batch-128 diagnostic.

## Augmentation result

Augmentation is active in Palette's custom pose loader. The exact receipt
records:

- `palette_single_sample_pose_augmentation_v1`;
- ±5-degree rotation;
- 5% translation;
- 10% scale variation;
- 10% grayscale intensity variation;
- 50% horizontal flip with `eye_left`/`eye_right` semantic swapping; and
- 5% erasing.

The first training batch was verified as raw uint8 `[64,3,512,512]` normalized
to float32 in `[0,1]`. Ultralytics reports `augment=false` intentionally: its
separate mosaic, mixup, cutmix, copy-paste, auto-augment, and multi-scale stack
is neutralized so it cannot apply an undocumented second transform pipeline.

## Training result

The run completed all 30 epochs in 1,354.7 seconds of recorded epoch time
(0.376 hours reported by Ultralytics). The best row by pose mAP50-95 was epoch
29:

- pose mAP50: 0.99499;
- pose mAP50-95: 0.99482;
- box mAP50-95: 0.88120.

The final epoch reported pose mAP50-95 0.99480 and box mAP50-95 0.88788. A
fresh validation of `best.pt` rounded both pose mAP50 and pose mAP50-95 to
0.995.

Artifact hashes:

- runtime receipt: `027e38d151b9f1bf69f545b4d4c8a5591e70e68927b104795987e0b119f64da3`;
- runtime payload: `82ce69081506c205434d0ab5f5075b8f1ec63c124776d3161673d920c720ff13`;
- `results.csv`: `10ad9fe4629bf11aca1444a39f7067352abb49321947938184e734157818ce7e`;
- `weights/best.pt`: `6a35ed911cd2c4284d4ef33b73b3a7ea29a3772a4a9ae7acac27510f126e0580`;
- `weights/last.pt`: `6f3c6d2e2ccb9586e7573864060b25c28bad53bf07efa04cd8a0cb0bb13e4f79`;
- training report: `a97fe5059ab093b2cb844394af70bf0b0126e453979c81193e196da4e07000f7`.

The runtime receipt status is `verified`; requested and effective trainer
arguments match; the starting-model content, pose schema, optimizer class and
parameter groups, preprocessing, loader behavior, and augmentation behavior
are all captured.

## Why this is not a promotion decision

1. All 181 reviewed Batman rows are training rows. A second independently
   labeled Batman recording is needed for a quantitative Batman-domain test.
2. The warm-start checkpoint previously saw much of the historical corpus.
   Therefore the six held-out historical leakage groups are disjoint from this
   fine-tuning run but are not necessarily pristine with respect to the
   warm-start model.
3. The warm-start model was not evaluated on this exact v003 validation split
   before training. The current metrics establish candidate quality, not a
   controlled before/after improvement.
4. The historical v003 source manifest records the skeleton ID and labels but
   does not embed the exact ordered edge list. This run correctly failed closed
   until an execution manifest supplied and hashed those edges. Palette now
   requires future source manifests to resolve and persist the exact edge list,
   propagates it through merged Zarr metadata and successor manifests, and
   rejects missing, reordered, conflicting, or post-write-tampered edges. This
   hardening is prospective and does not rewrite the immutable v003 artifact.

## Remaining gates

- [x] Persist and validate exact ordered skeleton edges in future v3 source and
  merged manifests and merged Zarr metadata.
- [ ] Evaluate the starting and candidate checkpoints on the same untouched
  source groups with one frozen evaluation command.
- [ ] Label at least one additional Batman recording for an independent
  Batman-domain gate.
- [ ] Run source-matched Batman full-recording inference and visual inspection,
  including edge/padded crops.
- [ ] Only after those gates, publish an immutable model artifact and consider
  registry/selector activation through a separate reviewed change.
