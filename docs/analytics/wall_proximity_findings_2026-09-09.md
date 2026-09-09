# Wall proximity across epochs — goodbatbadbat findings log

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-09
implementation: scratch-analysis-not-in-pipeline
-->

**Purpose.** Running record of what the time-resolved wall-proximity trace
shows on the goodbatbadbat cohort, and how it bears on the strategy-state
result (`docs/diagnostics/strategy_state_analysis_2026-09-01.md`) and the
cross-protocol feature contract
(`docs/analytics/cross_protocol_behavior_space_design.md`, phase 0). Append
dated entries; do not rewrite earlier ones.

**Inputs and method (all entries unless stated).** Dense
`provider_motion_samples` (keypoint provider, 100 fps) from the phase-B
validated-behavior export
(`/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_phase_b_20260901_b45aa6a5`).
Arena circle = the human-reviewed `visible_dish_top_rim_edge` fit from each
recording's `analysis/arena_geometry_selection/<run>` (radius 40.9 to
41.0 mm across the cohort); scale = the export's `row_axis_timing_and_scale`
binding (52.3 to 52.8 px/mm). Wall = valid in-arena sample with centre
distance >= radius minus 5 mm. Epochs from the export's `semantic_epochs`
table (chaser_pre 600 s, chaser_training 300 s, chaser_post 600 s). Binned at
5 s. Uniform-area expectation for a 5 mm band at this radius is 0.23. Four
members of the 84 (session 2026-08-12T21-14) carry empty binding tables and
are skipped; the 80 analysed match the strategy-state cohort exactly.
Outputs and scripts: `~/palette_figures/wall_fraction_goodbatbadbat_2026-09-09/`.
Nothing here is a pipeline product yet; see "what should become a product".

---

## 2026-09-09 — Epoch time course (80 fish)

| Epoch | Median wall fraction |
|---|---|
| Pre | 0.33 |
| Training | 0.66 |
| Post | 0.45 |

- **Training doubles wall time** within ~30 s of chase onset and holds
  ~0.6 for the full 5 min; 80% of fish increase pre to training (median
  +0.22). This is the "wall proximity is a chase mediator" result from the
  July synthesis seen as a time course.
- **Post does not return to pre.** Post opens at the training level, drops
  over ~2 min to ~0.45, then drifts up again through minute 10. 64% of fish
  are higher post than pre (median +0.09).
- **Pre ramps** from ~0.35 to ~0.45 across the 10 min with a dip near
  minute 5. Same shape as the twin-corrected avoidance ramp; the two are hard
  to separate when the parked dot is near the wall. Flag for the
  novel-object control already listed in the strategy-state open items.
- **Individual structure.** Sorted by pre mean, pre is a smooth gradient
  (continuum, no gap: agrees with "boundary affinity is an axis not a type").
  Post shows long solid horizontal runs in mid-pack fish.
- **Figure caveats.** First/last ~15 s of each smoothed curve are rolling-
  window edge artifacts. Band is 5 mm against the visible top rim, not the
  acquisition physical rim; absolute fractions move with that choice, epoch
  contrasts do not.

Figure: `wall_fraction_all_epochs.png`. Data: `wall_fraction_5s_all_epochs.parquet`,
`per_fish_epoch_wall_fraction.parquet`, `arena_geometry_used.parquet`.

## 2026-09-09 — Cross-reference with explorer→punctuated conversion

Transition labels from `strategy-states-v001/strategy_transitions.parquet`
(cluster 0 = explorer E, 1 = punctuated P, 2 = outlier O): E→E 52, E→P 16,
P→P 9, P→E 2, O→P 1. (The analysis doc's 17/69 count includes the outlier
recording's pre epoch; the same 80 fish.)

**Hypothesis tested:** the solid post-epoch wall runs are the converters
parked at the boundary, i.e. conversion is (partly) a wall-dwelling state.

**Result: rejected, and the sign is reversed.**

| Post-epoch metric (median) | E→E stayers (n=52) | E→P converters (n=16) | test |
|---|---|---|---|
| post − pre wall fraction | +0.13 | −0.07 | MWU p=0.013; arena-stratified permutation p=0.002 |
| post wall fraction | 0.50 | 0.40 | MWU p=0.11; within-arena rank p=0.07 |
| longest wall run (s) | 52 | 112 | MWU p=0.14 |
| longest *interior* run (s) | 45 | 118 | MWU p=0.0002 |
| longest run either side (s) | 80 | 162 | MWU p<0.0001 |
| wall/interior switches per 10 min | 24 | 15.5 | MWU p=0.0035 |

OLS among pre-explorers (n=68): post wall ~ pre wall + converter gives
converter b = −0.15, p = 0.015, with pre wall b = 0.68.

Reading:

1. **The post-training wall increase belongs to the stayers, not the
   converters.** Fish that kept escaping and stayed explorers patrol the
   wall more after training; converters' wall time goes slightly down.
   Escape fraction correlates positively with post switching (ρ=+0.33,
   p=0.003) and negatively with interior run length (ρ=−0.39, p=0.0004).
2. **The long runs in converters are not wall-specific.** Their interior
   runs are as long as their wall runs, and they switch sides half as
   often. The runs are the punctuated state itself (reduced locomotion,
   the fish stays wherever it is), not a boundary preference. A "wall run"
   metric is therefore a locomotion proxy and must not be read as
   thigmotaxis without the interior-run control beside it.
3. This is consistent with the strategy-state dissociation: conversion is
   locomotor, and now also not spatial with respect to the wall, not just
   with respect to the dot.
4. **Arena confound, handled but real.** Post wall fraction is ~0.5 in
   arenas 1 to 2 and ~0.34 in arenas 3 to 4; converters are enriched in
   arena 3 (6 of 16) and nearly absent from arena 4 (1). All headline
   comparisons above survive arena stratification, but any unstratified
   wall statistic on this cohort will be partly a rig statistic. This is
   the concrete instance of the design doc's warning that wall features
   are the most rig-sensitive features in the set.

Figure: `wall_runs_by_transition.png`. Data: `per_fish_wall_by_transition.parquet`,
`post_run_controls.parquet`.

**Open follow-ups.**
- Same analysis on the pre-epoch trace to ask whether pre wall-run length
  predicts conversion (a disposition test in the boundary family; pre wall
  fraction itself does not: p=0.52).
- Replace the 5 s-bin run definition with dense run lengths, and define
  runs on the smoothed-speed immobility criterion rather than position so
  the locomotion and boundary components are measured separately by
  construction.
- Rerun with the acquisition physical-rim circle to bound the band-choice
  sensitivity.
- Once GoodCopBadCop/batman/redscare are through the validated export,
  repeat as the first leave-protocol-out check of a boundary-family feature.

## What should become a pipeline product

- Per-sample `centre_distance_mm`, `distance_to_boundary_mm`, `wall` flag,
  `wall_band_mm`, and `boundary_method` on `provider_motion_samples`
  (mirrors the July `baseline_kinematic_samples` columns; the geometry is
  already bound by the bundle).
- A per-bin table keyed by `analysis_role` and bin index with wall fraction
  and centre-distance summaries, sharing the band constant.
- Band declared per export in mm and as a fraction of the reviewed radius,
  so the uniform-area expectation travels with it across rigs.
