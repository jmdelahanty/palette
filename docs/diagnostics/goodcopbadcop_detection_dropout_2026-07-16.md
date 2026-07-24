# GoodCopBadCop detection dropout: three mechanisms (2026-07-16)

Investigation of missing fish detections in the GoodCopBadCop chaser cohort,
triggered by the freeze-curve avoidance readout (see
`docs/archive/goodcopbadcop_avoidance_readout_survey.md`), whose immobility estimate is
censored by dropout. What began as "the detector loses frozen fish" turned out
to be **three distinct mechanisms, two of which are cheap refinement-stage
bugs, not detection failures.**

## Scope and data

- 12 reachable analysis zarrs on `/groups/johnson/johnsonlab/jeremy/recordings`
  (3 sessions, all 2026-06-14, 4 arenas each). The May 2026-05-29 session's 8
  zarrs are registered but not on local disk (paths under `/nvme1` that don't
  resolve), so this audit covers 12 of the ~20 registered / 32-fish cohort.
- Fish centroid feeding all metrics = **offline refined detection** bbox center
  (`method: offline_detection_to_chaser_distance`,
  `source_detection_kind: refined`), not the live tracker. Raw + refined detect
  runs are both floored at confidence **0.40**.
- Per-frame position = geometric center of the refined detect bounding box (not
  a subject-mask centroid).

## Summary of the three mechanisms

| mechanism | where | nature | fix | needs model work? |
|---|---|---|---|---|
| 1. Jump-anchor cascade | `detect_quality.py` | real on-track fish mislabeled "jump" | fix anchor update | no (re-refine) |
| 2. Dish-mask under-fit (+ corner-gate bug) | `refine_detect.py` | real wall fish outside an undersized mask | +5% buffer + gate fix | no (re-refine) |
| 3. Genuine inference misses | model | no box emitted at the 0.40 floor | lower floor + retrain | yes |

## Per-recording dropout

`drop%` = frames with no refined detection. `in-raw` = missing frames that
have a ≥0.40 box in the raw run (dropped by refinement — recoverable without
inference). `absent` = missing from raw too (genuine inference miss).

| recording | drop% | pre% | train% | post% | miss | in-raw | absent |
|---|--:|--:|--:|--:|--:|--:|--:|
| 21-12-08 arena_1 | 14.15 | 6.65 | 17.62 | 20.97 | 19814 | 10563 | 9251 |
| 21-12-08 arena_2 | 25.43 | 2.40 | 36.83 | 44.64 | 35611 | 27231 | 8380 |
| 21-12-08 arena_3 | 3.38 | 0.14 | 13.11 | 3.83 | 4738 | 1047 | 3691 |
| 21-12-08 arena_4 | 5.99 | 0.08 | 44.77 | 0.48 | 8394 | 9 | 8385 |
| 21-50-10 arena_1 | 0.47 | 1.06 | 0.14 | 0.00 | 662 | 13 | 649 |
| 21-50-10 arena_2 | 0.58 | 0.61 | 0.02 | 0.67 | 819 | 20 | 799 |
| 21-50-10 arena_3 | 0.00 | 0.00 | 0.00 | 0.00 | 5 | 0 | 5 |
| 21-50-10 arena_4 | 5.56 | 0.76 | 10.20 | 9.16 | 7791 | 150 | 7641 |
| 22-33-50 arena_1 | 2.31 | 4.03 | 4.03 | 0.12 | 3226 | 2678 | 548 |
| 22-33-50 arena_2 | 1.28 | 0.71 | 0.11 | 2.23 | 1789 | 289 | 1500 |
| 22-33-50 arena_3 | 2.09 | 0.48 | 2.81 | 3.46 | 2916 | 59 | 2857 |
| 22-33-50 arena_4 | 0.19 | 0.13 | 0.03 | 0.29 | 261 | 123 | 138 |
| **cohort** | **5.12** | **1.42** | **10.81** | **7.16** | 86026 | 42182 | 43844 |

Dropout is heterogeneous (0–25%) and concentrated in the **training/chase
epoch** (mean 10.8%), consistent with the fish freezing/relocating during the
chase. Pre is clean (1.4%). Cohort-wide, **49% of missing frames are
recoverable without re-inference** (in-raw), 51% are genuine misses.

## Mechanism 1 — jump-anchor cascade (`detect_quality.py`)

`identify_temporal_artifacts` (`src/fisheye/refinement/detect_quality.py:314-335`)
compares each detection to `last_valid_pos` — the last *non-jump* position —
and on a jump **does not update the anchor** (line 331). Correct for a
one-frame spike, but there is no recovery from a *sustained* relocation: when
the fish relocates across a detection gap (moves while undetected) it lands
>threshold (705 px effective) from the frozen anchor, is flagged a jump, the
anchor stays frozen, and **every subsequent on-track detection is flagged until
the fish wanders back** — a cascade. The computed per-step displacement
(`distances`, line 308) that would distinguish real fish from teleports is
never used for the flag.

Cohort-wide cascade sweep (jump-flagged detections in consecutive runs ≥50):

| recording | jump | cascade (≥50) | longest run |
|---|--:|--:|--:|
| 21-12-08 arena_2 | 23264 | **17989** | 1093 (~11 s) |
| 22-33-50 arena_1 | 2668 | 1489 | 297 |
| 21-12-08 arena_3 | 995 | 995 | 995 |
| (8 others) | 0–268 | 0 | — |
| **cohort** | 27316 | **20473 (75%)** | |

75% of jump flags are cascades, but **88% of all cascade frames are arena_2
alone.** These are real on-track fish (median displacement to the previous
detection ≈ 0). Recoverable by fixing the anchor update — no re-inference.

## Mechanism 2 — dish-mask under-fit + corner-gate bug (`refine_detect.py`)

Two compounding issues drop real fish at the arena rim:

**(a) The dish mask is systematically ~4–5% too small.** Masks are Hough
auto-detections stored in `analysis_metadata.dish_mask`. Real (clean) fish
detections reach a maximum of only 1.03–1.05× the mask radius in every
recording, i.e. the mask is fit a few percent *inside* the true arena. `%out` =
clean fish currently outside the mask:

| recording | mask r | % fish outside | max reach | buffer for 99.9% |
|---|--:|--:|--:|--:|
| 21-12-08 arena_1 | 0.473 | 19.9% | 1.05 | +4% |
| 21-12-08 arena_2 | 0.475 | 18.2% | 1.05 | +4% |
| 22-33-50 arena_2 | 0.480 | 13.1% | 1.04 | +3% |
| 22-33-50 arena_1 | 0.484 | 10.4% | 1.03 | +3% |
| 21-12-08 arena_4 | 0.475 | 8.5% | 1.04 | +4% |
| 21-50-10 arena_4 | 0.478 | 6.3% | 1.04 | +3% |
| 21-12-08 arena_3 | 0.478 | 1.2% | 1.03 | +2% |
| 21-50-10 arena_2 | 0.480 | 1.6% | 1.03 | +2% |
| 22-33-50 arena_3 | 0.478 | 3.3% | 1.04 | +3% |
| 21-50-10 arena_3 | 0.478 | 0.3% | 1.02 | +2% |
| 21-50-10 arena_1 | 0.478 | 0.1% | 1.02 | +0% |
| 22-33-50 arena_4 | 0.480 | 0.1% | 1.03 | +0% |

10 of 12 masks are undersized. `%out` tracks **wall-hugging**: the recordings
where the fish spent the most time at the rim lose the most, because a mask 4%
too small clips the detection pile-up at the wall. So the mask has been
systematically censoring high-thigmotaxis, wall-proximal frames cohort-wide.

**(b) The gate tests the box corner, not its center.**
`_dish_mask_inside_bbox_centers` (`refine_detect.py:321-335`) takes
`bboxes[:, :2]` as the center, but `bbox_norm_coords` is `[x, y, w, h]`, so
`[:, :2]` is the **top-left corner**. `_apply_dish_mask_quality_gate`
(`:352-418`) relabels clean-but-outside detections to
`_DISH_MASK_QUALITY_LABEL`; `filter_detections` (`:620-689`) then keeps only
label==0 and drops them (the raw quality report still shows them "clean," which
is why they looked dropped-for-no-reason). Verified on arena_1: the corner test
reproduces the observed drop **exactly (10,321)**; the true-center test would
drop 26,033. So the corner is *not* the over-drop driver — the undersized mask
is — but it is a real semantic bug that makes clipping uneven (shifts the test
point up-left by ~half a box).

Also note `refine_detect.py:663` overwrites the accumulated `keep_mask` with
`detection_quality_labels == 0`, making the `filters` parameter dead code —
jumps/blips/dish-outside are always dropped regardless of configuration.

**Fix for mechanism 2:** enlarge the mask radius by **~+5% (cohort-safe; worst
max reach = 1.05)** *and* fix the corner→center gate. If the corner test is
left in, size the buffer to ~+7–8% to compensate. Re-refine — no re-inference.
A second agent is adding a buffer to the mask fit as of 2026-07-16; +5% is the
recommended magnitude, paired with the gate fix.

## Mechanism 3 — genuine inference misses

51% of missing frames (43,844 cohort-wide) have no box in the raw run at all —
the model emitted nothing at its 0.40 floor. Dominant in arena_4 (train
44.77%), 21-50-10 arena_4, 22-33-50 arena_3. These are the only ones that
require **re-inference at a lower confidence floor and/or a retrained model.**
Adding training data + retraining (planned) is the durable fix for this half;
it will NOT recover mechanisms 1 or 2, which are post-inference filtering.

## Impact on the science

- The freeze-curve "avoidance" result (mid-band immobility +0.177 pre→post) was
  **RETRACTED 2026-07-17 as a raw-tracking-noise artifact** — it vanishes on the
  smoothed signal (Δ+0.004, p=0.85; see `avoidance_readout_survey.md` item 1). So
  it was never a dropout/censoring issue: the metric was thresholding raw centroid
  speed at the noise floor. `chaser_response_regimes` now classifies immobility on
  `speed_smoothed_mm`. (The escape result, by contrast, survives the clean signal:
  12× during chase, 12/12, p=0.0005.)
- The dish mask preferentially censored **wall-proximal / high-thigmotaxis**
  frames. So the thigmotaxis result (+0.138, session-RE p=0.040) and any
  wall/edge-referenced metric must be **re-checked after the mask buffer +
  re-refine**, not treated as settled.
- Immobility is scored as bbox-center speed < 1.0 mm/s on validly-tracked
  frames — a locomotor proxy, not ethological freezing (no tail/cardiac axis on
  this rig; see survey doc).

## Fix sequencing

1. **Cheap, no model work (re-refine only):** dish-mask +5% buffer + corner→
   center gate fix (recovers ~real fish in 10/12 recordings, esp. arena_1);
   jump-anchor fix in `detect_quality.py` (recovers ~18k in arena_2). Also fix
   the `refine_detect.py:663` keep_mask overwrite.
2. **Model work:** lower inference floor + retrain for the genuine-miss half.
3. **Re-run** thigmotaxis on the re-refined data and update the numbers. (The
   freeze curve is retracted — see above — so it is no longer part of this
   sequence; only thigmotaxis and any wall/edge metric need the re-refine.)

## Code references

- `src/fisheye/refinement/detect_quality.py:307-345` — jump/blip labeling (anchor cascade at 331).
- `src/fisheye/refinement/refine_detect.py:321-335` — `_dish_mask_inside_bbox_centers` (corner-vs-center bug).
- `src/fisheye/refinement/refine_detect.py:352-418` — `_apply_dish_mask_quality_gate` (relabels outside-dish).
- `src/fisheye/refinement/refine_detect.py:620-689` — `filter_detections` (keep_mask overwrite at 663).
- `src/fisheye/analysis/chaser_response_regimes.py` — freeze-curve consumer (immobility threshold, dropout QC).
