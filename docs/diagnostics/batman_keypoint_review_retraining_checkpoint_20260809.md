# Batman keypoint review and retraining checkpoint — 2026-08-09

## Decision

Training-image sharding and randomized-minibatch storage optimization are deferred.
Palette trains from an immutable merged training Zarr; minibatch order belongs to the
trainer.  The current priority is to add the newly reviewed Batman observations to a
successor merged corpus and retrain the five-keypoint pose model.

This deferral does not weaken logical schemas, provenance, atomic publication, or
immutable-base review.  It postpones only a physical-layout optimization that should be
benchmarked against training throughput later.

## Implementation checkpoint

- `79973be6` adds an atomic keypoint-only training-review publisher.  It publishes the
  strict raw-keypoint, quality, refined-keypoint, and body-frame bases, then creates one
  instance-key-bound delta generation for browser review.  It does not require or invent
  a subject-mask terminal.
- `771be597` adds the exact
  `sampled_acquisition_crop_video_hybrid` source binding used by lossless acquisition
  crop videos plus explicitly recorded full-frame fallback rows.
- The combined keypoint-plus-mask publisher remains strict and unchanged in purpose.
- The reviewed-keypoint compactor accepts either the combined review receipt or the
  narrower keypoint-only receipt.
- Keypoint review validation recomputes the immutable base `instance_key` digest and
  rejects a delta whose count or digest differs.

Focused validation passed:

- 62 review, delta, compaction, reviewed-export, and merged-corpus tests;
- 5 exact hybrid-provider and keypoint-only review tests;
- Ruff, Python compilation, and `git diff --check`.

## Published review artifacts

All artifacts contain 200 rows, remain selector-ineligible, defer registry activation,
and use direct metadata while their delta generation is mutable.  Their immutable base
generation was validated before entering mutable review.

| Arena | Receipt SHA-256 | Physical bytes | Review artifact |
| --- | --- | ---: | --- |
| 1 | `541d3c064579d5e7f40381652a8598bd6355f87bae830b0c40789017b20edd7d` | 4,180,087,336 | `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T20-12-57Z_arena_1_Batman/zarr/2026-07-21T20-12-57Z_arena_1_Batman_keypoint_review_384_v1_training.zarr` |
| 2 | `d0e0a0d62a962cfc5fd9cdd77f28f43622444435802b57a00e4b1a0d6d9e0595` | 4,179,920,646 | `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T20-12-57Z_arena_2_Batman/zarr/2026-07-21T20-12-57Z_arena_2_Batman_keypoint_review_384_v1_training.zarr` |
| 3 | `9b066fd52462c9e6a35594c4852882e0bd8965e06b31caaf64b824bed34e0c92` | 4,179,408,270 | `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T20-12-57Z_arena_3_Batman/zarr/2026-07-21T20-12-57Z_arena_3_Batman_keypoint_review_384_v1_training.zarr` |
| 4 | `d51a8cb888dd1007c9d4e99938d9495bef564cc9f3f87102b5a73c3dad4759ad` | 4,180,044,527 | `/groups/johnson/johnsonlab/jeremy/recordings/2026-07-21T20-12-57Z_arena_4_Batman/zarr/2026-07-21T20-12-57Z_arena_4_Batman_keypoint_review_384_v1_training.zarr` |

The approximately 4.18 GB per artifact is dominated by the 200 persisted
4512×4512 full-resolution training frames copied from the source training artifact.  It
is not keypoint-array overhead.  The 384×384 pose pixels are sourced from the lossless
acquisition crop video where the exact provider contract permits it; provider-specific
fallback rows remain explicit.

## Web review

The labeling-store backup immediately preceding task creation is:

`/home/delahantyj@hhmi.org/.palette/backups/labeling_work_before_batman_keypoints_20260809.sqlite`

Four warning-free tasks were assigned to `delahantyj`:

- `keypoints-batman-201257-arena1-384-review-v1`
- `keypoints-batman-201257-arena2-384-review-v1`
- `keypoints-batman-201257-arena3-384-review-v1`
- `keypoints-batman-201257-arena4-384-review-v1`

Every task uses `include_all=true` and `filter_mode=all`.  This reviews all 800
observations, including the 12 rows whose terminal inference recorded an explicit pose
failure.  Browser edits append delta partitions; they do not mutate the immutable
refined base.

## Required sequence after human review

- Confirm all four tasks are complete and no review session remains active.
- Freeze and validate each delta generation.
- Compact each immutable base plus delta into a new immutable refined snapshot.
- Publish four reviewed keypoint training-artifact candidates atomically.
- Compose a successor task-specific merged corpus with exact source receipts and
  acquisition-group split protection.
- Train the five-keypoint model on the local A6000 with the explicit 512 model canvas,
  configured augmentations, and effective hyperparameters recorded in provenance.
- Compare against the prior candidate on a recording-group-disjoint evaluation set
  before model promotion.

No production selector, model authority, registry entry, or source training artifact was
modified by this checkpoint.
