# GoodCopBadCop chase epoch: tracking failure, the clamp artifact, and the freeze regime

**Date:** 2026-07-14
**Recording:** `2026-06-14T21-12-08Z_arena_4_GoodCopBadCop` (the only recording in the store
with a materialized `chaser_distance_run` — 1 of 57 analysis archives)
**Run:** `analysis/chaser_distance_runs/goodcopbadcop_chaser_distance_v1_20260617`

## Summary

Three findings, in descending order of how much they should change what you do.

1. **45% of the chase epoch has no fish**, in a single contiguous 80.5-second block —
   **but it is a confidence-threshold artifact and is recoverable.** The fish is visible in
   every frame, and the same model detects it correctly at 0.28–0.35 confidence, just under
   the 0.40 threshold. Re-run detection at `conf ~0.20–0.25`. See §1.
2. **The hole at the centre of the radial profile is the dot, not the fish.** The chaser
   controller clamps the dot when its edge reaches the fish, so the minimum attainable
   distance is the dot radius (2.0 mm). Observed minimum: 2.01 mm, zero frames below.
3. **The fish's response to the chase is to freeze, and it freezes harder the closer the
   dot gets.** This is real structure and it is the opposite of an approach routine.

Finding 2 remains: that hole is stimulus geometry, not behavior. Finding 3 is real, and its
freeze rates are **lower bounds** until the detection is re-run — because the model's recall
collapses precisely when the fish freezes.

## 1. The 80.5-second tracking hole — RECOVERABLE (updated 2026-07-14)

**Resolved: this is a confidence-threshold artifact, not a detection failure.** Traced through
every stage:

| stage | frames with data (in the gap) | empty |
|---|---|---|
| raw YOLO (`detect_runs`) | **1** | **8053** |
| refined (`refined_detect_runs`) | 0 | 8054 |
| `chaser_distance_run` `fish_valid` | 0 | 8054 |

Not a plumbing loss: refinement dropped only **9** detections across the whole recording, and
nothing was lost downstream. But the fish is **plainly visible in every gap frame** (see the
camera crops) — motionless, pressed nose-first into the dish rim among the bubbles.

Re-running the *same model* (`detect_all_available_detect_training_v004`, yolo11n) on the
dropped frames at `conf=0.01`:

| frame | in gap | top score | box centre | dist from fish | verdict |
|---|---|---|---|---|---|
| 61700 | no | 0.614 | (1150, 417) | 115 px | passed |
| 62000 | **yes** | **0.282** | (1008, 503) | 52 px | sub-threshold |
| 65000 | **yes** | **0.317** | (1015, 513) | 51 px | sub-threshold |
| 67000 | **yes** | **0.310** | (1016, 512) | 49 px | sub-threshold |
| 69700 | **yes** | **0.345** | (1016, 516) | 53 px | sub-threshold |
| 69900 | no | 0.530 | (1011, 538) | 72 px | passed |

The boxes are correctly on the fish (~143x106 px, matching the before/after boxes), parked and
stable — exactly what a motionless fish should produce. The detections simply sit **below the
0.40 confidence threshold** (the minimum surviving score in the entire recording is 0.401).

**Fix: re-run detection at `conf ~0.20-0.25`.** The dish-mask gate, the single-fish constraint
and the tracker will absorb any added false positives. That recovers essentially the whole
80.5 s.

**But note the bias this exposes.** The model's confidence collapses to ~0.30 *specifically*
when the fish is motionless and pressed into the rim/bubble band — i.e. exactly when it is
freezing, which is the behavior of interest. Whatever threshold is used, recall is
anti-correlated with the response. These 8053 frames are ideal hard-positive training data.

**Recommended QC:** in a single-fish dish the fish cannot leave. A contiguous run of
`n_detections == 0` longer than a second or two is a detector failure *by construction* and
should fail loudly rather than propagate as `fish_valid = False`.

---

## 1b. The original write-up (superseded above)

Fish tracking dropout by epoch:

| epoch | dropout | note |
|---|---|---|
| `pre_event` | 0.08% | objects static |
| `training_event` | **44.77%** | chaser pursuing |
| `post_event` | 0.48% | objects static |

The training dropout is **not** many short gaps. It is one block: training frames
704–8757, i.e. t = 617.3 s to 697.8 s, 80.5 seconds, 44.7% of the epoch.

Across that entire block **the fish moved 1.1 mm** — last seen at (14.8, 6.9) mm, first
seen again at (14.1, 7.8) mm. It did not leave the arena. It stopped.

Ruled out as causes:

- **Dot occlusion.** The chaser stayed a median of 8.9 mm away throughout the gap and never
  came within 3 mm. It was not sitting on top of the fish.
- **Motion blur.** The fish was motionless, not fast. It was still at 0.87 mm/s when it
  reappeared.

The remaining explanation is that a perfectly stationary fish at the arena edge is not
being detected. Note that the fish is immobile ~45% of the time in `pre_event` and is
tracked fine there, so stillness alone is not sufficient — position (extreme edge, outside
the nominal 40 mm circle, in a corner of the square region) is likely the other half.

**This is the priority.** The detector fails specifically on the behavior that is the
response. It is not missing data; it is the main result.

## 2. The clamp artifact

Protocol: `chasers[0].radius_mm = 2.0`, `target_radius_mm = 1.0`, and the chaser state
carries a `target_clamped_pos` field.

Observed during `training_event`: minimum distance **2.01 mm**, **zero** frames below
2 mm, p1 = 2.84 mm.

A behavioral flee threshold does not land on the dot's radius to two decimal places. The
controller stops the dot when its edge reaches the fish centroid. The 0–2 mm hole in the
radial occupancy profile is therefore **stimulus geometry**, and any reading of it as "the
fish refuses to swim over the dot" is wrong.

Note also `size_scale_max = 2.5` with `size_scaling_full_distance_mm = 3.0`: the dot grows
to 2.5× as it closes, so at contact it is a ~5 mm-radius disc. This does not appear to
break detection (frames at 0–3 mm are tracked fine), but it is worth knowing.

`chaser_response_regimes` reads `radius_mm` from the protocol, records it on the `chasers/`
group, marks `distance_floor_is_clamp`, and emits
`distance_floor_at_chaser_radius:<epoch>:<chaser>`.

## 3. What the fish actually does: it freezes

Decomposing the change in distance onto the fish→chaser axis, `training_event`, chaser 0
(the aggressive/pursuing object), on the surviving 55% of frames:

| distance | **P(frozen)** | escape gain (v_r when swimming) | heading away |
|---|---|---|---|
| 2–4 mm | **0.62** | −2 to −5 mm/s | 64–72% |
| 4–8 mm | 0.61 | −12 mm/s | 50–55% |
| 8–15 mm | 0.59 | −14 mm/s | 63% |
| 15–30 mm | 0.49 | −23 mm/s | 68% |
| 30–50 mm | 0.45 | −13 mm/s | 75% |

`freeze_index` (near − far) = **+0.204**. `approach_fraction` = **0.309**.

Three things:

- **The fish never swims toward the dot at any distance.** Radial velocity is negative
  everywhere; only 31% of swimming frames head even partly toward it. There is no approach
  routine to find.
- **Immobility rises monotonically as the dot closes** — 45% frozen when it is far, 62%
  when it is on top of the fish.
- **Escape is *weakest* when the dot is closest** (+1.27 mm/s separation contribution at
  2–4 mm, vs +7.70 at 15–30 mm). If this were flight it would be the other way round.

And the "shell" at 2–4 mm in the radial occupancy (19.2% of frames, 29.7× enriched) is a
pursuit equilibrium, not a preferred radius: at 2–4 mm the fish contributes +1.27 mm/s to
the gap and the chaser contributes −1.38 mm/s. They cancel. It is a dot parked against a
motionless fish.

**All of these freeze numbers are lower bounds** — the 80-second hole is the fish freezing
so completely the tracker lost it, so the hardest freezes are precisely the ones excluded.

## What is invalid as a result

- Any near-zone occupancy, dwell, or radial-density statistic for `training_event`. It is
  computed on 55% of frames, non-randomly selected, and the pile-up at the floor is the
  controller.
- The 18.2× near-zone enrichment for `training_event` in `chaser_radial_occupancy`. That is
  the chaser's controller closing the loop on the fish; the module flags it
  (`closed_loop_null`) precisely so it is not read as behavior.
- Any claim that this fish avoids the aggressive chaser. The assay's two primary endpoints
  disagree in sign on this fish (`nearzone_occ_specificity` = +0.023 reads as avoidance;
  `approach_p05_specificity` = −5.54 mm reads as the opposite — the fish's closest
  approaches to the aggressive object got 2.6 mm *closer* after training). Baseline
  near-zone occupancy for the aggressive object was already 0.045%, ~35× below chance, so
  there was no dynamic range for a decrease. n = 1.

## What to do

1. **Fix detection on stationary fish.** Until the tracker holds a frozen fish at the arena
   edge, the chase epoch cannot be analyzed. This blocks everything else.
2. **Materialize the stack across the cohort.** 1 of 57 archives has a
   `chaser_distance_run`. `src/fisheye/group_statistics/goodcopbadcop.py` already defines
   the right cross-recording endpoints and has one row to work with.
3. **Reconsider `r_zone_mm = 5.0`.** At this fish's baseline it has no dynamic range. The
   CDF ladder and the approach percentiles discriminate better and are already persisted.
4. **Go bout-level.** Bin swim bouts by distance-to-chaser at onset and look at the
   repertoire (`swim_bouts`, `bout_kinematics`, megabouts). A regime switch in *which bouts
   the fish uses* is a far stronger claim than a change in mean speed.

## Components written for this

- `chaser_radial_occupancy` (`docs/chaser_radial_occupancy_contract.md`) — area-normalized
  rings around the moving chaser, fixing the geometric bias in
  `epoch_distributions/hist_density`.
- `chaser_response_regimes` (`docs/chaser_response_regimes_contract.md`) — the freeze and
  escape-gain curves above, with tracking QC and the clamp detection as first-class outputs.
