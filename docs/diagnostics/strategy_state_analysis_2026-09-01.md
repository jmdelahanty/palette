# Strategy-state analysis — individual pre→post change on the goodbatbadbat cohort

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-01
implementation: committed-module-unmerged
-->

**Scope:** individual-fish strategy clustering, pre/post window decoding,
direction decomposition, and time-resolved twin-corrected avoidance on the
phase-B validated-behavior export (`goodbatbadbat-validated-behavior-phase-b-
20260901-b45aa6a5`, 80 admitted recordings). All exploratory: unit=recording,
subject/batch identity historically unavailable, no clustering adjustment.
Companion docs: `validated_behavior_export_assessment_2026-09-01.md` (export
audit + fix branches), `chaser_analytics_roadmap_2026-08-10.md` (A/B series).

**Implementation:** committed module
`src/fisheye/group_statistics/validated_behavior_strategy_states.py` + CLI, on
branch `agent/palette/validated-behavior-strategy-states-20260901` (abba9d3f,
27/27 tests). Consumes the twin-excess and censored-IBI products as validated
parquet paths (not code imports), so it merges independently of the twin-nulls
and role-contrasts branches; at runtime it needs their outputs. Real-run
outputs: `/tmp/strategy-states-v001` (+ scratch figures in the session
scratchpad `strategy_clustering/`). Decoder windows are pluggable so the A4
anticipation probe can reuse it once the schedule importer lands.

## Design decisions that matter

- **Corner-fair features only** (7): twin-excess near-zone fraction and
  distance p50 (aggressive, keypoint), wall-distance p50, occupancy entropy,
  bout rate, |net bout heading change|, censored IBI frac>2s. The parked
  chaser moves corners between pre and post, so absolute-position or raw
  chaser-distance features would manufacture "transitions" from the stimulus
  schedule.
- **One pooled model across pre+post** so cluster labels are comparable
  between epochs; posterior-weighted flows so soft assignments don't fabricate
  crisp transitions.
- **Training epoch is never clustered** (180 s, closed-loop pursuit = each
  fish received a fish-dependent stimulus). The middle Sankey tier is the
  measured trial responder class instead: escape-dom (esc_frac≥0.75) /
  freeze-leaning (freeze_frac≥0.25) / mixed.
- **Decoder = LORO logistic regression** on 60 s windows, 12 stimulus-blind
  bout/IBI features; window-level CV is forbidden (autocorrelation).

## Results

1. **Population structure is explorer-vs-punctuated, not the hypothesized
   avoidant/explorer/thigmotactic triad.** GMM (BIC k=3 = 2 clusters + 1
   outlier recording `2026-08-11T18-29-34Z_arena_4`): "active explorer" vs
   "punctuated/low-activity" (bout rate 24 vs 58/min, IBI>2s ×10, lower
   entropy). Avoidance features load ≈0 on the separating axis — baseline
   avoidance varies continuously, not as a type. Bootstrap ARI ~0.51–0.63:
   a discretized axis, not natural kinds.
2. **Asymmetric transition (Sankey):** 17/69 explorers → punctuated after
   training; 1/10 the reverse (dependence p=0.0002, arena-stratified).
   Responder class predicts conversion beyond pre-cluster and arena
   (p≈0.0024–0.0036); held-out validation: converters had **~zero escape
   rate during training trials** vs 22.3/min for stayers (p=0.0003).
   Escape *quality* among escapers (latency, separation gain) does not
   predict conversion. Avoidance features show NO converter/stayer
   difference — conversion is locomotor, not spatial.
3. **Direction decomposition:** the punctuated axis is −entropy/−bout-rate/
   +IBI-tail. esc_frac predicts movement along it (ρ=−0.44 to −0.46,
   p<1e-4); freeze_frac weakly positive (ρ=+0.26). The per-fish decoder AUC
   tracks the parallel component (ρ≈0.6–0.7) and NOT orthogonal or total
   displacement — the decoder is a punctuated-axis odometer, not a general
   change detector.
4. **Disposition (causality caution):** pre-epoch locomotor features predict
   training escape fraction (entropy ρ=+0.37/+0.41, bout rate +0.35, IBI
   tail −0.30; LORO pre→responder AUC 0.64–0.69). Pathway is partly trait
   continuity + amplification, not pure induction; responder still adds
   information beyond pre-cluster (p≈0.003).
5. **The post state is strongest immediately and decays:** decoder P(post)
   0.56 in minute 1 → ~0.50 by minutes 8–10 (per-fish decay p=0.017).
   Post minutes 1–3 vs pre: per-fish median AUC **0.83**, 48/80 fish >0.75
   (vs 0.665 over the full 10 min). **Recommendation: post minutes 1–3 is
   the standard aftereffect readout window; the full window measures decay.**
6. **Avoidance/kinematics dissociation (per-minute twin excess,
   `timeresolved_twin_excess.parquet`):** post-epoch distance excess is flat
   (~+3 mm; early vs late p=0.37; early-post minus pre +0.07 mm, p=0.52;
   converters if anything less avoidant, n.s.). The kinematic aftereffect
   comes and goes with no change in spatial relation to the dot.
7. **NEW: pre-epoch avoidance ramp.** Twin-corrected distance excess to the
   parked dot climbs from ≈0 (minute 1) to +8…+11 mm by minutes 8–10
   (late-pre p=0.003/1e-5/2e-4). Innate avoidance develops over ~5 min of
   exposure — sensitization or exploration-first dynamics — and is already
   asymptoted/absent as a ramp in post. Pre-epoch avoidance metrics are
   time-weighted; single-epoch means hide this.
8. **Color note (registry-verified):** both goodbatbadbat chasers are BLACK
   [0,0,0,1] — the role contrast is color-clean (GoodCopBadCop was red
   aggressive vs blue inert, color-confounded), and the innate avoidance here
   is to a black dot, generalizing the red-cohort precedent.

## One-paragraph synthesis

Fish that fail to escape during training — partly predictable from a
pre-existing punctuated lean — shift toward intermittent, punctuated
locomotion afterward (straighter bouts, pruned micro-bouts,
censoring-robust multi-second pauses), strongest in the first three minutes
and partially extinguishing within ten, with no change in twin-corrected
avoidance of the chaser at any point; meanwhile innate avoidance of the
(black) dot ramps up across the pre epoch. The aftereffect is a passive-
coping locomotor state, not learned spatial avoidance.

## Open items

- Merge order: escape-freeze provenance (305fb4b1) → twin-nulls (bbc10923)
  → role-contrasts (983d3771+b6e97ed3) → strategy-states (abba9d3f).
- Wall-conditioning of the straightening effect; residualized
  escape→post test (machinery in the module); arena-4 punctuated enrichment
  (p=0.01) wants a rig check; outlier recording wants an eyeball.
- A4 anticipation probe via the module's pluggable windows, blocked on the
  schedule importer.
- Pre-ramp follow-up: is the ramp sensitization (dot-locked) or
  exploration decay (dot-independent)? The twins answer it: an
  exploration-only account predicts no ramp in twin-corrected excess.
  It ramps — but a dedicated novel-object control would settle it.

## Addendum (2026-09-01): CRA quadrant endpoint is side-confounded in this layout

The legacy CRA quadrant group statistics cannot regenerate anywhere (the code
requires analytics-export contract v3; every existing export is v1/v2), but
its endpoint has a direct phase-B equivalent (`same_quadrant_fraction_valid`).
Run on goodbatbadbat it "works": specificity (agg−inert) post−pre = +0.122,
q=0.002. **This is an artifact.** Geometry check (park positions extracted
from `chaser_relative_samples`, fish fixed-quadrant occupancy from
`provider_motion_samples`, saved in the session scratchpad):

- Park layout is identical in all 80 recordings — pre: aggressive
  bottom-left, inert bottom-right; post: aggressive top-right, inert
  top-left. **Role↔arena-side is perfectly confounded cohort-wide**
  (aggressive = left in pre, right in post).
- Fish occupancy = (a) genuine dot-anchored avoidance of BOTH dots — the
  occupancy field flips halves with the dots (59%/60% in the far half;
  rankings fully reverse, so no fixed place preference) — which is
  role-symmetric and cancels in the specificity; plus (b) a **stable
  right-side bias** (right-half occupancy 0.538 pre, 0.542 post).
- Arithmetic: the side bias alone predicts a specificity diff-in-diff of
  +0.133 vs +0.122 observed. The endpoint measures the fish's lateral bias
  with the role labels swapped between sides, not avoidance.
- The training−pre specificity (+0.092, q=0.014) is the separately
  established pursuit effect (the aggressive dot visits the fish).

Consequences: quadrant-based role endpoints are unusable under this stimulus
layout; twin-corrected distance endpoints remain the valid avoidance
measures (the rotated null cancels side bias). The both-dot half-avoidance
in (a) is itself a real, previously unquantified result: fish re-anchor
their occupancy field to the dot pair's position.
