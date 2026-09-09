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
- ~~Pre-epoch disposition test with run length~~ done, null (entry below).
- ~~Dense immobility-based run definition~~ done (entry below).
- Bench rig check for the arena 3 to 4 wall-peak shift: level, light
  gradient, visible edge features at each arena position.
- ~~Physical-rim sensitivity rerun~~ done, no conclusion moves (entry below).
- Once GoodCopBadCop/batman/redscare are through the validated export,
  repeat as the first leave-protocol-out check of a boundary-family feature.

## 2026-09-09 — Pre-epoch disposition test with run length (null)

**Question:** does pre-epoch run structure (wall runs, interior runs,
switching) predict which explorers convert, the way pre-epoch entropy and
bout rate do (pre→responder LORO AUC 0.64 to 0.69 in the strategy-state
doc)?

Pre-explorers n=68 (16 converters, 52 stayers). Same 5 s-bin run
definitions as the post entry. Data: `pre_disposition_runs.parquet`.

| Pre-epoch metric | Conv median | Stay median | MWU p | Arena-strat p | AUC | ρ vs esc_frac |
|---|---|---|---|---|---|---|
| wall fraction | 0.35 | 0.34 | 0.52 | 0.79 | 0.55 | −0.22 (p=0.08) |
| longest wall run (s) | 35 | 25 | 0.35 | 0.30 | 0.58 | −0.24 (p=0.048) |
| longest interior run (s) | 55 | 50 | 0.90 | 0.65 | 0.51 | −0.02 |
| longest run either side (s) | 65 | 60 | 0.67 | 0.69 | 0.54 | −0.08 |
| time in runs ≥60 s (s) | 65 | 62 | 0.53 | 0.94 | 0.55 | −0.09 |
| wall/interior switches | 30 | 28.5 | 0.66 | 0.76 | 0.54 | −0.06 |
| late − early wall fraction | +0.15 | +0.02 | 0.042 | 0.19 | 0.67 | −0.21 (p=0.08) |

Leave-one-out logistic on (longest run, switches, wall fraction):
converter AUC 0.40; escape-dominant responder AUC 0.51. Sanity: the same
run metrics separate pre-cluster punctuated from explorer at AUC 0.87
(longest run) and 0.79 (switches), so the metrics carry signal; they just
do not carry disposition.

**Result: null.** Boundary-family run structure in the pre epoch does not
predict conversion or escape responder class. The one nominal hit, a
steeper pre-epoch wall ramp in converters, drops to p=0.19 under arena
stratification and is one of seven tests. Treat it as noise unless it
reappears in another cohort.

Reading: the pre-disposition component of conversion identified earlier
lives on the locomotor axis (entropy, bout rate, IBI tail), not on the
boundary axis. Together with the post-epoch entry this closes the wall
question for goodbatbadbat in both directions: converters are not wall
fish before training and do not become wall fish after it.

## 2026-09-09 — Dense immobility-based runs (locomotion and boundary separated by construction)

**Definition.** Per frame from `provider_motion_samples` (keypoint, 100 fps):
immobile = valid and `speed_smoothed_mm_s` < 1.0 (the repo's
`DEFAULT_IMMOBILITY_SPEED_THRESHOLD_MM_S`); an immobile run is a contiguous
immobile stretch >= 2 s; a run is "at wall" if >= 50% of its frames are in
the 5 mm band. Wall fraction is then computed separately over mobile
frames and over immobile frames. Note that at this threshold larval fish
are immobile ~58% of the time (inter-bout pauses); the >= 2 s run filter
is what isolates the long pauses. Data: `dense_immobility_runs.parquet`,
`dense_immobility_deltas.parquet`. Figure: `dense_immobility_by_transition.png`.

**Post epoch, E→P converters (16) vs E→E stayers (52), arena-stratified permutation p:**

| Post metric (median) | Converters | Stayers | strat p |
|---|---|---|---|
| longest immobile run (s) | 20.2 | 2.3 | <0.0001 |
| time in immobile runs >= 2 s (s) | 148 | 3 | <0.0001 |
| immobile runs >= 5 s (count) | 6 | 0 | <0.0001 |
| immobile-run time at wall (s) | 26 | 0 | <0.0001 |
| immobile-run time in interior (s) | 53 | 0 | <0.0001 |
| wall fraction while mobile | 0.43 | 0.49 | 0.51 |
| wall fraction while immobile | 0.40 | 0.50 | 0.20 |

Post minus pre: converters gain +0.10 immobile fraction (stayers −0.00,
p<0.0001); converters lose −0.07 mobile-wall fraction while stayers gain
+0.13 (p=0.006); same sign and size for immobile-wall fraction (p=0.008).

Reading. With locomotion and boundary measured separately, the picture is
unambiguous: conversion is a ~50-fold increase in time spent in long
pauses, and those pauses fall in the interior twice as often as at the
wall. Neither the mobile nor the immobile wall fraction differs between
converters and stayers in post. The earlier 5 s-bin "wall run" result was
this immobility effect leaking through a position metric.

**Pre epoch, same groups.** Immobile fraction identical (0.58 vs 0.58);
long-run metrics show a weak converter lean (longest run 2.6 vs 0.0 s,
strat p=0.06; time in runs strat p=0.03; run count strat p=1.0) that is
inconsistent across metrics and far below the pre-cluster P vs E
separation (P→P pre: longest run 16 s, time in runs 100 s). Consistent
with the earlier null: the locomotor disposition is real but small, and
it is not a boundary disposition.

## 2026-09-09 — The arena difference: what it is and what it is not

Concern raised: post wall fraction ~0.5 in arenas 1 to 2 versus ~0.34 in
arenas 3 to 4. Diagnostics on the **pre epoch** (stimulus-free, so any
difference is rig or fish, not protocol). Data:
`pre_occupancy_anisotropy.parquet`, `pre_parked_chaser_positions.parquet`.
Figure: `arena_occupancy_diagnostics.png`.

What it is **not**:
- Not geometry. Reviewed rim radius 40.94 to 40.96 mm and scale 52.3 to
  52.8 px/mm across arenas; circle-fit spread ~1 px.
- Not tracking. Valid-position fraction 0.995 to 0.999 in every arena.
- Not the parked dot. Both chasers park 19.7 to 20.1 mm from the wall in
  every arena.
- Not a session or batch effect. Across the 20 sessions (4 arenas each),
  the mean of arenas 1 to 2 exceeds the mean of arenas 3 to 4 in 18 of 20.
  Kruskal across arenas: wall fraction p=0.0015, mobile-only wall fraction
  p=0.0025, median centre distance p=0.005.

What it **is**: a persistent, position-specific difference in how tightly
fish hug the wall. The per-fish radial occupancy density peaks at 2 to
3 mm from the wall in arenas 1 to 2 and at 4 to 7 mm, flatter and broader,
in arenas 3 to 4; the 5 mm band cuts straight through the arena 3 to 4
peak. Median centre distance is 35.2 mm in arenas 1 to 2 and 32.3 to
33.4 mm in arenas 3 to 4. So arena 3 to 4 fish are not less boundary-bound
in kind; they sit a few millimetres further in. Every arena also shows a
shared directional bias of occupancy toward the upper-right of the image
frame (circular mean −50 to −74 deg, Rayleigh p<0.01 in three of four),
i.e. a common cue across the rig; the wall-band angular profiles differ
per arena, which is what one expects if that cue is a fixed room or rig
feature seen from four positions.

Candidate causes, none testable from these exports: lighting or IR
illumination gradient across the four arena positions, a meniscus or
water-level difference in the dishes at positions 3 to 4, thermal
gradient, or a visible edge feature (adjacent arena, rig frame) that the
fish in positions 3 to 4 see differently. This needs a rig check at the
bench: level, light meter, and a look at what is visible through the
dish wall at each position.

**Consequences.**
1. Every wall statistic on this cohort must be arena-stratified or
   arena-residualized. All headline results in this log already are.
2. A fixed 5 mm band is the wrong feature contract across arenas. Two
   robust alternatives, both to go into the phase-0 contract: (a) the
   band expressed as a per-recording quantile of the radial occupancy
   (e.g. fraction of time within the fish's own 25th-percentile
   wall distance) and (b) the full radial density as the feature, with the
   band derived downstream. The design doc's open question on wall
   features in L0 is answered: include them only in residualized or
   self-normalized form.
3. This is precisely the effect the design doc's leakage test is meant to
   catch. A pre-epoch classifier would decode arena pair from wall
   fraction alone on this cohort.

## 2026-09-09 — Physical-rim sensitivity check (band choice does not move any conclusion)

Recomputed every wall fraction with the acquisition `physical_inner_rim`
candidate circle from `analysis/arena_geometry_runs/<acquisition run>`
instead of the human-reviewed visible-rim fit. Data:
`physical_rim_sensitivity.parquet`, `circle_differences.parquet`.

**How different the circles are.** Physical radius is 0.12 to 0.17 mm
smaller (range −0.37 to +0.26 mm across recordings); centres differ by
0.14 to 0.38 mm median, 0.85 mm max. Both circles are the same object to
within a fish body width.

**Headline numbers, reviewed versus physical:**

| | Reviewed | Physical |
|---|---|---|
| Median wall fraction pre / training / post | 0.33 / 0.66 / 0.45 | 0.33 / 0.66 / 0.46 |
| Pre, arenas 1 / 2 / 3 / 4 | 0.44 / 0.43 / 0.24 / 0.24 | 0.46 / 0.46 / 0.25 / 0.25 |
| Converter − stayer post−pre delta, strat p | −0.20, p=0.003 | −0.19, p=0.010 |
| Per recording-epoch Spearman | 0.973 | |
| Median absolute difference | 0.007 | |
| Recording-epochs with difference > 0.05 | 5 of 240 | |

Every conclusion in this log survives. The arena effect is if anything
slightly larger under the physical circle.

**Two instructive outliers.**
1. `2026-08-11T19-05-11Z_arena_3`, training epoch: reviewed 0.005 versus
   physical 0.970. The fish was immobile 86% of the epoch at a distance
   to the wall of 5.12 to 5.29 mm (5th to 95th percentile) under the
   reviewed circle: parked on the 5 mm knife edge, so a 0.12 mm radius
   change flips the whole epoch. Any hard band will do this to some fish;
   the per-fish radial quantile or the full radial density (already
   recommended above) is immune to it.
2. `2026-08-10T17-54-26Z_arena_1`, training epoch: 29% of valid samples
   fall outside the physical circle (0 outside the reviewed one). The
   fish sat at 0.5 to 1 mm from the reviewed rim for most of the epoch;
   a 0.23 mm smaller, 0.43 mm shifted circle pushes those samples out.
   The reviewed fit is the better boundary for a wall-hugging fish, which
   is the human reviewer's original decision ("acquisition fit close but
   slightly misregistered") confirmed in data. Under any circle, samples
   just outside the boundary should count as wall, not be dropped; the
   current exclusion is a small bias against the most thigmotactic fish.

**Decision recorded:** keep the reviewed visible-rim circle as the wall
authority for this cohort; treat out-of-circle valid samples within one
body width as wall rather than invalid in the future contract; retire the
"band sensitivity" concern for absolute conclusions, but not for
per-fish values at the band edge.

## What should become a pipeline product

- Per-sample `centre_distance_mm`, `distance_to_boundary_mm`, `wall` flag,
  `wall_band_mm`, and `boundary_method` on `provider_motion_samples`
  (mirrors the July `baseline_kinematic_samples` columns; the geometry is
  already bound by the bundle).
- A per-bin table keyed by `analysis_role` and bin index with wall fraction
  and centre-distance summaries, sharing the band constant.
- Band declared per export in mm and as a fraction of the reviewed radius,
  so the uniform-area expectation travels with it across rigs; and a
  per-fish radial-quantile form of the boundary feature alongside it.
- Valid samples just outside the circle (within ~1 body width) count as
  wall, not invalid.
