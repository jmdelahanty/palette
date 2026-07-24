<!-- ARCHIVED 2026-07-04: documents eye-mask code deleted in commit 4a85e5d (eye-mask stage severance). Retained for history only; NOT current. Live replacement: docs/archive/eye_subject_mask_unification_design.md. -->

# Why the segmentation U-Net can't learn left vs. right eye — investigation (2026-06-21)

## Problem statement

The project has labeled training data with separate `eye_left` / `eye_right` segmentation
masks. A U-Net segmenter fails to learn the distinction: it only ever segments "both eyes" —
effectively predicting the eyes-union in both channels rather than assigning each eye to its
anatomical side. The **keypoint** pipeline, by contrast, assigns L/R consistently. A colleague
suggested the zebrafish's near-symmetric eyes as the cause; this investigation tests that and the
mechanical alternatives.

## Method

Read-only pass over the segmentation training stack: the model (`segmentation/unet.py`), the two
trainers (`segmentation/train_unet_{eye,subject}_masks.py`), the losses
(`training/losses.py`), the dataloaders (`training/zarr_{eye,subject}_mask_dataset.py`), the
label-generation path (`refinement/subject_eye_assignment.py`, `assemble_refined_subject_masks.py`,
`refine_eye_masks.py`), the training-export (`utils/export_eye_mask_training_zarr.py`), and the
crop geometry (`tracking/crop.py`). Three angles were checked in parallel: augmentation,
label-consistency, and the model/loss/input. All claims below are grounded in code that was read;
nothing was modified.

---

## Executive summary

The two "obvious bug" hypotheses are **ruled out**, which reframes the failure as fundamental
rather than a silent data-corruption:

- **Not an augmentation/flip bug.** The subject-mask U-Net path has *zero* spatial augmentation —
  no flip, rotation, or transpose — so nothing scrambles handedness at train time. (The classic
  "flip image+mask without swapping the L/R channels" failure mode cannot occur because no flip
  exists.)
- **Not bad labels.** `eye_left`/`eye_right` masks are anatomically consistent, keypoint-derived,
  fail-closed, and inherit the same L/R convention the keypoints use.

The **leading explanation**: the per-channel sigmoid loss makes "predict the eyes-union in both
channels" the loss-optimal *degenerate* solution unless the network can reliably bind each eye to
its anatomical side — and it cannot, because (a) crops are axis-aligned (not heading-normalized),
so L/R is not a fixed image side, and (b) a per-pixel FCN is architecturally weak at the global
*relational* binding ("which side of the body axis is this eye"), which is exactly the task a
coordinate-regression keypoint head does naturally. That is why keypoints succeed where the U-Net
fails.

A **second, independent, checkable** suspect: per-channel supervision masking
(`MaskedBCEDiceCriterion`) zeroes the gradient on channels marked invalid; if `eye_left`/`eye_right`
are rarely supervised, the model is never penalized for L/R confusion — same symptom, even with
perfect data.

**Update (data inspection, same day): the suspects are now resolved empirically.** Direct
inspection of the live L/R training data
(`/nvme1/training/datasets/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v002`) confirms the
labels are pristine and fully supervised, *and* it is impossible for the network to recover
handedness from its input. The leading explanation is now a verified, quantified diagnosis — see
**"Empirical verification"** below.

---

## Empirical verification (2026-06-21 data inspection)

The relevant L/R dataset on disk is the eye-mask path, not subject masks: the only `lr` datasets are
`eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v001/v002`; the sole subject-mask dataset is
`subject_v1_**union**` (single eyes channel). So L/R training runs through the legacy eye-mask U-Net
(`train_unet_eye_masks.py`, `out_channels=2`) on `eye_masks_runs/merged_eye_masks/masks_roi` of shape
`(10788, 2, 512, 512)`. Measurements (full dataset for counts; 400-row random sample for
pixel-level stats):

| Check | Result | Verdict |
|---|---|---|
| `ellipse_success` per channel | `eye_left` = **10788/10788**, `eye_right` = **10788/10788**, both = 10788, neither = 0 | Supervision suspect **REFUTED** — every row fully supervises both eyes |
| Inter-channel IoU (`eye_left` ∩ `eye_right`) | mean = median = max = **0.000**; 100% of rows disjoint; 0% identical | Labels are clean separate blobs, **not a hidden union** |
| Mean mask area | `eye_left` ≈ 194px, `eye_right` ≈ 191px | Two similarly-sized, well-formed eye blobs |
| Is `eye_left` centroid image-left of `eye_right`? | **43.8%** of rows (≈ coin flip) | "Left eye" is **not a fixed image side** — crops are not heading-normalized |
| Eye separation | **18.9px** (median 18.8, p10–p90 16.6–21.7); both centroids ≈ (255, 257) in 512×512 | Eyes are tiny, dead-center, ~19px apart |
| U-Net bottleneck (confirmed 4 down-stages, ÷16 → 32×32) | 18.9 / 16 ≈ **1.2px apart** at the bottleneck | The two eyes **merge into one cell** where global/orientation context lives |

**Conclusion:** the data is pristine — clean, disjoint, fully-supervised L/R targets on every row —
and the model still collapses to the union. So it is definitively not labels, not supervision, not
augmentation. It is an input/architecture impossibility, now concrete:

- Assigning L vs R requires inferring the fish's **heading** (since "left eye" is on either image
  side, 43.8 / 56.2). Heading inference needs **global context** → only available deep in the net
  (the ÷16 bottleneck, large receptive field).
- But at that bottleneck the two 19px-apart eyes sit **~1.2px apart — the same cell.** Where the
  eyes are spatially separable (shallow, high-resolution layers) there is no orientation context;
  where there is orientation context (bottleneck) the eyes have already merged.
- Caught between those, per-channel sigmoid BCE makes the **symmetric hedge** — predict the
  eyes-union in both channels — the loss-optimal solution. Exactly the observed symptom. (The eye
  trainer's `overlap_weight` defaults to 0, so nothing even nudges the channels apart.)

This is the clean contrast with keypoints: the pose net regresses the whole
`{swim_bladder, eye_left, eye_right}` constellation as coordinates and resolves L/R via heading — the
eyes are separable at heatmap resolution and handedness is decided at the constellation level, not by
per-pixel channel binding on a merged feature cell.

> Reproduce: open the merged zarr `eye_masks_runs/merged_eye_masks`; read `ellipse_success` (N×2),
> and over a sample of rows compute per-channel mask area, `eye_left`∩`eye_right` IoU, the sign of
> `centroid_x(eye_left) − centroid_x(eye_right)`, and `eye_separation` vs the 512px crop / ÷16
> bottleneck.

---

## What we ruled out

### 1. Augmentation — not the cause (definitive)

The subject-mask U-Net path applies **no spatial augmentation at all**.
`SubjectMaskChunkedDataset.__getitem__` (`training/zarr_subject_mask_dataset.py:405`) returns the
zarr sample verbatim; `_load_chunk` only normalizes intensity (HWC→CHW move + `/255` + `nan_to_num`),
no geometric op. The trainer's collate (`train_unet_subject_masks.py:282`) only stacks tensors. The
`fliplr`/`flipud` fields exist on the config base (`training/config.py:75`) but are **never read**
by the subject-mask trainer. (The real flips/rotations in `training/zarr_yolo_dataset_loader.py:1478`
belong to the *detection* loader, a different pipeline that does not emit L/R eye channels.)

> Consequence: the classic flip-without-`flip_idx` bug is absent. It would only become relevant if
> augmentation were *added* later — at which point a horizontal flip / arbitrary rotation **must**
> swap `eye_left`↔`eye_right`.

### 2. Labels — anatomically consistent (definitive)

`assign_eyes_union_to_lr` (`refinement/subject_eye_assignment.py:249`,
method `subject_eyes_union_keypoint_assignment_v1`) splits the `eyes_union` mask by **nearest eye
keypoint** — a Voronoi/bisector split anchored on the `eye_left`/`eye_right` keypoint coordinates,
so it rotates with the fish (heading-invariant, not image-space). It is fail-closed: rows without
valid, distinct eye keypoints emit *empty* L/R masks plus a failure status
(`failed_missing_eye_keypoints` / `failed_coincident_eye_keypoints`, ~`:332-346`), never a guessed
assignment. Eye keypoint indices are resolved **by label name**
(`assemble_refined_subject_masks.py:485` → `resolve_required_keypoint_indices_from_attrs([...
"eye_left","eye_right"])`), so the mask channels share the exact convention the working keypoint
pipeline uses. No image-position ("leftmost blob") rule exists anywhere in the path.

> One defensive note: `training/zarr_eye_mask_dataset.py:330` reads `masks_roi[:channels]`
> **positionally** without remapping by `mask_labels`. Correct today (producers always write
> `(eye_left, eye_right)` in that order), but a future producer writing a different channel order
> would silently mislabel — worth an assert, not a current bug.

---

## The leading explanation

### The loss rewards hedging to the union

`training/losses.py` implements per-channel **sigmoid BCE + soft Dice** (`BCEDiceCriterion`,
`MaskedBCEDiceCriterion`). Each eye channel is supervised independently against its own binary
target. There is **no** permutation-invariant / Hungarian matching (so the model cannot swap
channels for free), but there is also **no signal telling the network which eye belongs in which
channel beyond the per-channel target itself**. If the network cannot reliably determine an eye
blob's anatomical side from its input, the loss-minimizing prediction is ~0.5 on both eyes in both
channels → at threshold, the eyes-union in both channels. This is exactly the observed symptom.

Mitigations are weak or absent: `overlap_weight` (penalizing `probs[:,0]*probs[:,1]`) exists only in
the legacy `BCEDiceCriterion` and defaults to `0.0`; the **live** subject trainer uses
`MaskedBCEDiceCriterion`, which has **no overlap term at all**. There is no `pos_weight`. The output
head is a single `Conv2d(b, out_channels, 1)` with both eye channels initialized identically
(`unet.py:113`) — nothing breaks the L/R symmetry.

### Why the network can't bind handedness

1. **Crops are axis-aligned, not heading-normalized.** `tracking/crop.py:315` extracts an
   axis-aligned box centered on the detection (`center − roi_w//2`); there is no `warpAffine` /
   heading rotation anywhere in the crop path (this is the documented "axis-aligned subject mask
   geometry"). So a fish facing north vs. south presents the *same* anatomical eye on *opposite*
   image sides. "Left eye" is therefore not a fixed image position — recovering it requires global
   orientation reasoning inside the crop.
2. **A per-pixel FCN is architecturally poor at global relational binding.** Assigning "the eye on
   the port side of the body axis → channel 0" requires propagating a global orientation decision to
   each eye pixel and breaking the L/R symmetry of two near-identical blobs. This is the kind of
   absolute-position / global-relation task FCNs are known to handle badly, whereas coordinate
   regression with global context handles it naturally.

### The telling contrast with keypoints

The pose network predicts the whole `{swim_bladder, eye_left, eye_right}` constellation **jointly**,
and the keypoint pipeline explicitly uses the **heading vector** to assign L/R
(`detection/detect_keypoints_traditional.py:349` rotates eye points by heading and assigns L/R by
the sign of the rotated coordinate). The `swim_bladder` keypoint anchors the anterior–posterior axis
and fixes the constellation's chirality. Global binding is natural for a coordinate-regression head;
it is not for a per-pixel segmentation head. So the colleague's "symmetry" intuition is half right —
the eyes *are* near-identical, but the failure isn't that they can't be *segmented*; it's that
nothing in the U-Net's input + architecture lets it decide *which side* each eye is on, and the loss
rewards hedging.

---

## A second, independent suspect — CHECKED AND REFUTED

`MaskedBCEDiceCriterion` multiplies both BCE and Dice by a per-channel `valid` mask
(`training/losses.py:110-118`); a channel with `valid=0` contributes **zero gradient**. The concern
was that `eye_left`/`eye_right` might be largely unsupervised, so the model is never penalized for
L/R confusion.

**Refuted by data inspection** (see "Empirical verification"): on the live L/R dataset every row
fully supervises both eyes (`ellipse_success` = 10788/10788 for each channel), and the legacy eye
trainer uses unmasked `BCEDiceCriterion` anyway. Supervision is not the cause.

---

## Design tension worth naming

The pipeline **already** derives `eye_left`/`eye_right` from `eyes_union` downstream via keypoints
(`assign_eyes_union_to_lr`), and a `subject_v1_union` schema exists for exactly that. Training the
U-Net to learn L/R directly (`subject_v1_lr`) re-solves — with a model architecturally bad at it — a
problem the pipeline already solves robustly with keypoints. The lowest-risk fix is therefore to
**segment `eyes_union` and split L/R from keypoints**, rather than ask the U-Net to learn handedness.

---

## Open questions — now answered by data inspection

1. **Which model is being trained?** → The **legacy eyes-only U-Net** (`train_unet_eye_masks.py`,
   `out_channels=2`). The only L/R datasets on disk are `eye_mask_*_lr_b9164009_v001/v002`; the sole
   subject-mask dataset is `subject_v1_union` (no L/R). There is no `subject_v1_lr` dataset, so the
   L/R failure is the eye-mask path.
2. **What is in the crop?** → A 512×512 grayscale ROI with the eyes tiny (~14px) and dead-center,
   19px apart. The fish body is plausibly in-frame, but crops are **not heading-normalized**
   (eye_left is image-left only 43.8% of the time), and the eyes merge at the ÷16 bottleneck where
   orientation context lives — so the heading cue is effectively unusable by this architecture.
3. **Per-channel supervised-row counts?** → Full: 10788/10788 for both eyes. Not the cause.

## Candidate fixes (matched to the confirmed cause)

- **Use `eyes_union` + keypoint split** (`assign_eyes_union_to_lr`) — lowest risk; reuses what
  already works and is the path the `subject_v1_union` dataset already takes. The pipeline solves
  handedness with keypoints; re-deriving it in a segmenter fights the architecture. **Recommended.**
- **If L/R segmentation is genuinely required, inject the handedness signal as input:** concatenate
  the `eye_left`/`eye_right`/`swim_bladder` **keypoint heatmaps as extra input channels** to the
  U-Net. This hands the network the resolved handedness (and the orientation anchor); it then only
  refines masks around given peaks. Highest-leverage architectural fix.
- **Rotation-normalize crops to canonical heading** — necessary but **not sufficient on its own**: a
  vanilla U-Net is translation-equivariant, so even with the left eye always on the same side it
  cannot use "which side" without an absolute-position signal. Pair it with **CoordConv / positional
  encoding**.
- **Mitigate the bottleneck merge** — a tighter crop around the head (larger eyes) or fewer
  downsampling stages keeps the two eyes spatially separable at the layers that carry orientation
  context. Helps, but the keypoint-heatmap-input or union+split routes are cleaner and address the
  ambiguity more directly.

---

## Appendix — conversational summary (as delivered)

> This was a productive pass — and the most useful result is what we *ruled out*. Two "obvious bug"
> hypotheses are dead, which reframes the problem from "a silent data corruption" to something more
> fundamental.
>
> **Ruled out:** It's not an augmentation/flip bug — the subject-mask U-Net path has zero spatial
> augmentation (no flip/rotation; `fliplr`/`flipud` config fields are inherited but never read). And
> it's not bad labels — `eye_left`/`eye_right` masks are anatomically consistent, keypoint-derived
> (`assign_eyes_union_to_lr`, nearest-eye-keypoint Voronoi split), fail-closed, and inherit the same
> convention the keypoints use.
>
> **Leading explanation:** the per-channel sigmoid loss makes "predict the eyes-union in both
> channels" the loss-optimal degenerate solution unless the network can reliably bind each eye to
> its anatomical side. It can't, because (a) crops are axis-aligned, not heading-normalized
> (`crop.py:315`), so "left eye" isn't a fixed image side; and (b) a per-pixel FCN is architecturally
> weak at the global relational binding "which side of the body axis is this eye" — exactly the task
> a coordinate-regression keypoint head does naturally. That's why keypoints succeed (the pose net
> predicts the `{swim_bladder, eye_left, eye_right}` constellation jointly and uses heading to assign
> L/R, `detect_keypoints_traditional.py:349`) and the U-Net fails.
>
> **Second, checkable suspect:** `MaskedBCEDiceCriterion` zeroes gradient on channels marked invalid
> (`losses.py:110`); if `eye_left`/`eye_right` are rarely supervised, the model is never penalized for
> L/R confusion — same symptom. Check the channel supervision counts. (The live subject trainer also
> has no overlap penalty.)
>
> **Design tension:** the pipeline already splits `eyes_union → L/R` via keypoints downstream, so the
> U-Net may be re-solving a solved problem with a model that's bad at it. Cheapest fix: segment
> `eyes_union`, derive L/R from keypoints.
>
> **Open questions:** which model/schema (legacy eyes-only 2ch vs `subject_v1_lr`)? what's in the
> crop (eyes-only vs whole fish)? what are the per-channel supervised-row counts? Fixes:
> rotation-normalize crops to canonical heading; feed keypoint heatmaps / a heading channel; or use
> `eyes_union` + the keypoint split that already exists.

---

*Companion reports: `docs/diagnostics/codebase_review_2026-06-20.md`,
`docs/diagnostics/utils_consolidation_review_2026-06-20.md`.*
