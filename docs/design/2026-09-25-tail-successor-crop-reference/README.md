# Tail successors reference their crop run instead of copying it

- **Status:** accepted
- **Owner:** labeling/Apply work (session palette-12, for the user); last reviewed 2026-09-25
- **Builds on:** `docs/design/2026-09-24-background-apply-effects/README.md`;
  the tail successor refresh (`fisheye.training.mask_tail_apply_refresh`,
  `fisheye.training.recovered_mask_review_payload`)
- **Related:** `docs/design/2026-09-25-review-archive-sharding/README.md`

## Problem

Every subject-mask Apply regenerates a training-tail successor version: five
new runs published into the archive. One of them is a crop run written with
`pixel_operation: "identity_copy"`, a byte-for-byte copy of the existing,
immutable crop images.

Measured on the arena 2 recovered training archive (Apply `0e6bcf1e`, 2 rows):

| Successor run | Files | Size |
|---|---:|---:|
| `crop_runs/recovered_full_roi_<v>` (identity copy of crops) | 209 | 25 MB |
| `keypoints_runs/..._seed_<v>` | 238 | 318 KB |
| `refined_keypoints_runs/..._edit_<v>` | 238 | 318 KB |
| `subject_mask_runs/recovered_masks_<v>` (mask snapshot) | 216 | 370 KB |
| `refined_subject_masks_runs/recovered_masks_edit_<v>` | 138 | 415 KB |

On `/groups`, this Apply's background update took 110 s live, and 114 s in a
private `/groups` replay. Publishing the five runs took 60 s, and the cost
tracks file count, not bytes. Copying the files concurrently did not help on
this mount (114 s to 108 s), so the fix is to publish less.

Each Apply also adds another 25 MB crop copy to `/groups`, which is at 88%
capacity.

## Why the copy is unnecessary

Crop runs are observation row sets: immutable once published, with their
identity bound by digests. A successor that copies one learns nothing new. It
only gains a second name for identical pixels.

The **mask snapshot is different, and stays.** Refined subject masks are
mutable review surfaces. Snapshotting them into an immutable
`subject_mask_runs/...` run is what pins the successor's source, and it is a
real provenance boundary.

## Design

1. **No crop run in a successor.** The successor's keypoint seed, keypoint
   edit, mask snapshot and mask edit runs all record
   `source_crop_run = <existing crop run>`, together with that crop run's
   identity digest, which is already part of the tail refresh's source proof.
2. **Row identity is unchanged.** Successor rows are the crop run's rows in
   the same order, as today (`source_crop_row_ids`). Only the location of the
   pixels changes.
3. **Consumers resolve crops through `source_crop_run`,** which is how every
   other stage already finds its crops. Consumers that currently expect the
   successor's own crop run need migrating:
   - the review task scope (`crop_run` in task scope);
   - the training exporters;
   - the review session builder;
   - the tail source proof (`_capture`).

   Each consumer gets a test through its real path.
4. **Versioning.** The successor version digest currently hashes a proof that
   names the copied crop run. The new policy gets a new policy identifier, for
   example `new_mask_seed_and_review_version_reference_crop_v2`, so old and new
   successors are never confused. Existing successors with copied crops stay
   valid and readable. Nothing historical is rewritten.

## What must be preserved

- **Crop pixels are bound by digest,** so a successor can never silently point
  at changed pixels. The crop run is immutable, and its identity is
  re-verified in the source proof.
- **Existing tests keep passing:** manual-point carry-over, preserved labels,
  refusal on source change during publication, and training eligibility.
- **Training export produces byte-identical training samples** for a
  successor before and after the change. This parity test is required.

## Expected effect (estimate until measured)

- **Per Apply:** one fewer run publication, 209 fewer files and 25 MB less
  written. That is roughly 20% of publish time on this archive.
- **Storage:** no crop growth per Apply.
- **Sharding** (the related design) reduces the remaining four runs' file
  counts.

## Decisions needed

1. Adopt the policy above with a new successor policy identifier, and leave
   existing copied-crop successors untouched?
2. Should the existing copied crops be reclaimed later? Only through a
   separate, authorized cleanup that re-points nothing silently. Proposed: not
   now.

## Decision log

- 2026-09-25: accepted by the user: new successor policy id; existing copied-crop successors untouched; no reclamation now. Implemented together with review-archive sharding as one successor format change.

- 2026-09-25: opened. The user agreed crops should be referenced, not copied.
