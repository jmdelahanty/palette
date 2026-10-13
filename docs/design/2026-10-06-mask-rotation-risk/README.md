# Rotating subject masks: measured risk before building a Rotate tool

- **Status:** evidence for a decision; nothing is built.
- **Owner:** labeling work (session palette-12, for the user). Last reviewed 2026-10-06.
- **Context:** a labeler (savinim) asked for mask rotation in the subject-mask editor. Move (#256) shipped without rotation, because rotating a binary mask resamples it and could damage training labels.
- **Classification:** read-only measurement. No code, data or archive changes.

## Question

How much does rotating a binary mask change it? Does that change break the mask-derived tail keypoints that training labels depend on?

## Method

Both experiments open every archive and the store read-only. The scripts are in this folder.

1. **Shape** (`rotation_experiment.py`)
   - **Masks:** 600 of savinim's applied masks, 200 each of `subject_body`, `eye_left` and `eye_right`, from the three Danionella review runs `refined_subject_masks_danionella_strong_single_inframe_review_20260804`. Median areas: body 2,913 px, eyes about 160 px.
   - **Rotation:** about the mask centroid by 0, 5, 10, 20, 30, 45 and 90°, on a padded canvas so nothing leaves the frame.
   - **Three methods:**
     - plain pixel rotation (nearest neighbour);
     - smoothed pixel rotation (bilinear, threshold 0.5);
     - outline rotation (sub-pixel boundary at level 0.5, rotated exactly and redrawn with even-odd fill).
   - **Measured:** area change, new pieces (8-connected), new holes, round-trip IoU (θ then −θ, two resamples), and new skeleton endpoints for bodies.
2. **Tail derivation** (`tail_derivation_check.py`)
   - **Data:** 16 tail-review archives in the labeling store (`tail_review_preview.sqlite`), one mask version per archive (the run its current mask task uses), paired with its keypoint run. 3,353 rows; 3,342 derive a valid tail at baseline.
   - **Derivation:** the production `derive_tail_seed`, legacy first and then the head-anchored fallback for `snout_extension_*` failures, exactly as mask Apply does.
   - **Rotation:** the whole row (every mask channel plus the head points) is rotated rigidly on a padded canvas. That isolates resampling damage from any misalignment with the image.
   - **Measured:** tails that stop deriving (and why), and how far the derived tail stations move once rotated back.

## Results

### 1. Shape change

One rotation changes very little. The round-trip figures below include two resamples.

| Body (n = 200) | Plain pixels | Smoothed pixels | Outline |
|---|---|---|---|
| Area change, median (worst 5%) | 0.06–0.16% (≤0.52%) | 0.09–0.18% (≤0.51%) | 0.07–0.22% (≤0.69%) |
| Splits into pieces | 0.5–4.5% | 0.5–2.5% | 1.0–3.5% |
| New holes | 0.5–**11.5%** | 0–2% | 0–1% |
| New skeleton endpoint (spur) | 18–74% | 15–48% | 13–40% |
| Round-trip IoU, median | 0.984–0.996 | 0.984–0.996 | 0.981–0.996 |

Eyes (n = 200 each), all methods:
- area change about 0.6% median, roughly 1 px on a 160 px mask, and 1.3–3.6% in the worst 5%;
- no splits;
- new holes at most 1%, and only with plain pixels.

At exactly 90°, plain pixel rotation is lossless.

### 2. Tail derivation

| Method | Angle | Tails broken (of 3,342 valid) | Main reasons | Tail station shift, median (worst 5%) |
|---|---|---|---|---|
| Smoothed | 10° | 1.0% (32) | split body 22, ambiguous head endpoint 10 | 0.83 px (1.77) |
| Smoothed | 20° | 1.3% (44) | split body 33, ambiguous head 10 | 1.04 px (2.17) |
| Smoothed | 45° | 1.7% (57) | split body 50, ambiguous head 4 | 0.74 px (1.58) |
| Plain | 10° | 1.4% (47) | split body 26, ambiguous head 16 | 0.80 px (1.80) |
| Plain | 20° | 1.7% (57) | split body 40, ambiguous head 13 | 1.04 px (2.23) |
| Plain | 45° | **4.1%** (136) | split body 65, **ambiguous skeleton endpoint 53** | 0.75 px (1.61) |
| Smoothed, then keep largest body piece | 10° | 0.30% (10) | ambiguous head 10 | 0.84 px (1.79) |
| Smoothed, then keep largest body piece | 20° | 0.33% (11) | ambiguous head 10, ambiguous tail 1 | 1.04 px (2.17) |
| Smoothed, then keep largest body piece | 45° | 0.21% (7) | ambiguous head 4, skeleton 2, tail 1 | 0.75 px (1.62) |

About 10 rows per setting newly derive a tail after rotation, which shows how close those rows sit to a derivation threshold. A few dozen rows switch between the legacy and head-anchored method.

## Findings

1. **Geometry is preserved.** When the tail still derives, its stations move about 1 px (median), and at most about 2 px for the worst 5%. That is within the size of a crop pixel and smaller than ordinary editing variation.
2. **Failures come from fragmentation, and one existing click fixes them.** Most tails that break do so because the rotated body splits into pieces (`fragmented_subject_body_mask`): the thin tail tip breaks off.
   - Keeping the largest body piece, which is exactly what *Remove stray pieces* (`R`) does, removes every fragmentation failure. Broken tails fall from 1.0–1.7% to 0.2–0.3%.
   - That residual matches the roughly 11 rows per setting that newly derive a tail after rotation, so the net effect on tail derivation is approximately zero.
   - The remaining failures are mostly `ambiguous_head_endpoint`: rows already near a derivation threshold.
3. **Plain pixel rotation should not be used.** It is the only method that creates many holes (up to 11.5% of bodies) and ambiguous skeletons (53 broken tails at 45°).
4. **Outline rotation is no better than smoothed rotation.** It costs more complexity for the same results, so smoothed rotation is the preferred method.
5. **Eyes are safe.** They are compact and never split.
6. **Not measured:**
   - Misalignment. A labeler rotating only the body mask, rather than the whole row, when the fish's actual pose differs is a separate risk: a plausible-looking but wrong mask. It is the same risk as copy-previous, and only review guards against it.
   - Effects on a trained model.

## Recommendation, if a Rotate tool is built

- **Resampling:** smoothed (bilinear, threshold 0.5) rotation about the mask centroid. Resample once per rotation session from the session's original mask, like Move, so repeated adjustment never compounds.
- **After each rotation:** show the piece count and the area change. If the body split, say so and offer *Remove stray pieces*. With that step, tail derivation is unaffected within measurement noise.
- **Provenance:** record the rotation (angle and method) in the saved checkpoint's metadata, so rotated rows can be audited or excluded from training later.
- **Scope:** no rotation for keypoints; masks only.

## Reproduce

```
scripts/py docs/design/2026-10-06-mask-rotation-risk/rotation_experiment.py rows.json
PYTHONPATH=src scripts/py docs/design/2026-10-06-mask-rotation-risk/tail_derivation_check.py tail.json \
    [pixel_nearest,pixel_bilinear | pixel_bilinear_keep_largest]
```

Both scripts read the live archives under `/groups` and the labeling store, read-only. Results depend on the masks present when they are run.
