# GoodCopBadCop behavior analysis — synthesis & handoff (2026-07-17)

What was tested this session on the GoodCopBadCop chaser cohort, what survived
scrutiny, what was retracted, the methodological lessons, and where to pick up.
Companion to `docs/diagnostics/goodcopbadcop_detection_dropout_2026-07-16.md`
(the detection-quality half) and `docs/goodcopbadcop_avoidance_readout_survey.md`.

> **Cohort + radial update (2026-07-18):** the 11–12-recording limit was a
> STALE-REGISTRY bug — the scripts queried `/nvme1/palette_registry.sqlite`, which
> holds only the 12 June-14 recordings (plus retired May duplicate rows). The
> canonical live registry
> (`/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`) resolves
> **40** GoodCopBadCop analysis recordings, of which **32 are analyzable now**. The
> durable analyses have been **rerun on the 32**; the full-cohort numbers are folded
> in below, and a new **virtual-control radial analysis closes the wall-confound
> question** (see the Radial bout kinematics section). This supersedes the earlier
> "rerun on 36" framing.

## Cohort / data reality (read first)

- **Query the canonical registry on `/groups`, NOT `/nvme1`.** The `/nvme1` copy is
  stale (12 June-14 only + retired May duplicate rows); it was the real cause of the
  "selection bug". Canonical path:
  `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`.
- **40 GoodCopBadCop analysis recordings registered; 32 analyzable now:**
  June 14 (12 / 3 sessions) + June 21 (20 / 5 sessions) have the chaser components
  and are what every number below is computed on. **May 29 (4) and July 2 (4) are
  registered but their analysis Zarrs lack `chaser_distance_runs`/`track_kinematics`
  etc.** — they need the chaser analysis pipeline run (not just registration) before
  they join the cohort. So the ceiling is 40, the current analyzable set is 32.
- `resolve_cohort()` in `fisheye.analysis.goodcopbadcop_common` is the shared,
  registry-resolved, duplicate-deduped discovery (dedupe by `recording_id`; drops
  off-disk pointers). All durable scripts use it; it defaults to the canonical registry.
- Immobility/speed metrics: **use `speed_smoothed_mm`** (track_kinematics), NOT raw
  centroid diffs — see the artifact below. Bouts come from `chaser_bout_response`,
  detected on the `speed_exponential` signal (robust).

## TL;DR — what is real vs what died

All numbers below are the **full analyzable cohort, n=32** (June 14 + June 21) unless
noted; the earlier n=11–12 June-14 slice values are given in parentheses for contrast.

REAL (survived clean-signal + confound checks):
- **Acute escape response during the chase.** Escape rate (bout peak > 100 mm/s)
  pre 0.35 → chase 7.81 per validly-tracked minute, **22×, 32/32, p<0.0001** (on
  `speed_smoothed_mm`; ~10× on the bout-table peak). Strengthened from the slice
  (12×, p=0.0005). The core finding.
- **Bout-rate suppression during the chase.** ~100 → ~48 → ~84 bouts/valid-min
  (pre/chase/post), **pre→chase p<0.001**. Behavioural inhibition under threat.
- **Whole-range fleeing during the chase (radial).** Fish radial velocity toward the
  aggressive object is negative at **every** distance bin (object-vs-virtual p<0.001,
  6–50 mm), strongest 14–18 mm — active distance-opening, object-specific (see Radial
  section). This is the direct measure of "the fish stays off the chaser's region."
- **Flee→freeze habituation over chase trials — the one clearly LEARNED signal.**
  Freeze fraction early 0.41 → late 0.60 (**p<0.001, n=29**); escape rate early 0.62 →
  late 0.15 (**p<0.001, n=28**). On the full cohort this SURVIVES the wall over-control
  (partial r(freeze,trial|wall) = +0.23, **p=0.003**; the +0.11/p=0.47 at n=11 that
  looked null was underpowering). Promoted from "plausible/underpowered".

INNATE, not learned (present pre-training, retained, not strengthened):
- **Near-shell avoidance steering.** The fish steers to pass wider around the aggressive
  object's local region (8–22 mm): object-minus-virtual steering excess **+0.40 mm/bout
  pre (p<0.001)**, **+0.34 post (p=0.007)**, **paired post−pre Δ=−0.08, p=0.50 — not
  strengthened** (robust to an 8–16 mm shell). Innate and retained (see Radial section).
- **Near-object bout-vigor gradient.** Faster bouts near the aggressive object, but on
  the full cohort this WEAKENED to marginal (pre gradient +1.05 mm/s, **p=0.14**;
  diff-in-diff +1.24, **p=0.06**; the slice's +2.7/p=0.034 was optimistic). Report as
  suggestive; does not grow with training either way.

RETRACTED / DIED:
- **Mid-band immobility "avoidance" (Δ+0.177, p=0.002) — RAW-TRACKING-NOISE ARTIFACT.**
  Vanishes on the smoothed signal (Δ+0.004, p=0.85); flips sign above the noise floor.
  `chaser_response_regimes` was fixed (commit `7e931481`). See below.
- **Occupancy avoidance** — null (aggressive band occupancy Δ+0.026, diff-in-diff
  +0.032, p=0.63; prior full-cohort nearzone_occ p=0.20).
- **Approach-avoidance direction metrics** — null or geometry/occupancy-confounded
  (radial velocity no change; fraction-moving-away decreased via selection bias;
  min-distance +4.5 mm marginal but non-specific and censoring-suspect).
- **Learned static-epoch spatial avoidance in general** — every pre/post spatial
  metric died. This is an **acute-threat-response** dataset, not a spatial-learning one.

## Radial bout kinematics — and the virtual-control resolution of the wall confound (2026-07-18)

The wall confound that dogged the static and partial-correlation analyses is **resolved**
by the `chaser_bout_response` component's built-in **virtual controls**: each object's
position rotated about the arena centre (60–300°), giving a reference with *identical*
wall proximity but no object. Any signal present around the real object but not its
virtual twins is object-specific, not wall-following. This is a cleaner instrument than
partialling out wall-distance (which over-controls a mediator). Two durable analyses
profile bout kinematics across the **full 0–60 mm** range (not just the single near-band
"reactive ring"), aggregated **fish-level** (fish = the unit):

- **Frame-based** (`analyze_goodcopbadcop_radial_kinematics.py`): radial velocity, bout
  rate, tangential speed vs distance, object vs virtual, per epoch. Populated 32/32.
- **Per-bout directional** (`analyze_goodcopbadcop_radial_turn_direction.py`): `turn_toward`
  and `delta_predicted_miss` (avoidance steering) from the raw `bouts_per_reference`
  table, pooled across the cohort at fine 3 mm resolution with a **cluster bootstrap over
  (fish, visit)** — the component's own pseudoreplication note (effective n = visits, not
  bouts). Coarser bins are the WRONG fix for the per-bout sparsity: the signal is a
  localized shell that wide bins average away.

What the radial profile shows (n=32), sorting the responses into innate vs learned:

| Epoch | Readout | Result |
|---|---|---|
| **Chase** | radial velocity (frame) | Fish opens distance at *every* range vs virtual (p<0.001), strongest 14–18 mm — active fleeing. |
| **Pre** | avoidance steering (per-bout) | Steers wider in an 8–22 mm shell vs virtual, **+0.40 mm/bout, p<0.001** — present *before training* (innate). |
| **Post** | avoidance steering (per-bout + frame `steering_excess`) | Shell retained, 14–22 mm, **+0.34 mm/bout, p=0.007**. |
| **Pre→Post** | paired shell steering | **Δ=−0.08, p=0.50 — not strengthened by training** (robust to 8–16 mm shell). |

Caveats: `delta_predicted_miss` is **confounded during the chase** (the object-ahead gate
plus the chaser's own pursuit motion — matches `steering_excess_by_band` being undefined
in chase), so the steering story is pre/post and the chase story is the frame-based
radial-velocity one. `turn_toward` is a weak/mixed axis (turning *toward* during chase =
threat-orienting/escape, not avoidance). The sub-3 mm bin is partly the chaser's dot-clamp
radius, not free behaviour. Always read `*_excess_vs_virtual` / `_wall_excluded`, never the
raw object value (the component's `wall_confound_note`).

**Bottom line:** spatial avoidance of the aggressive object is **innate and retained, not
learned** (steering shell present pre-training, not strengthened); the only thing training
changes is the **flee→freeze habituation**. The wall confound no longer needs a mediation
argument — the virtual controls settle it.

## The immobility artifact (the cautionary tale)

Raw centroid speed has a **jitter noise floor of ~1.6 mm/s median** that straddles the
1 mm/s immobility threshold, so `immobile_fraction` on raw partly measured tracking
noise. A threshold sweep is the tell: aggressive mid-band Δ = +0.276 (0.5 mm/s), +0.177
(1 mm/s), then **−0.075 (p=0.002) by 3 mm/s** — a real "more still" effect stays positive
at every threshold; a sign flip means the noise *distribution* reshaped. On
`speed_smoothed_mm` (deadbanded between bouts, median 0) the effect is Δ+0.004, p=0.85.
Per-bout vigor and bout rate near the object are unchanged. **Fix landed:**
`chaser_response_regimes` now classifies immobile/moving on `speed_smoothed_mm` (falls
back to raw centroid with a `immobility_signal_fallback_raw_centroid` warning +
`diagnostics.immobility_signal_source`). Corrected raw-vs-smoothed figures exist.

## Methodological lessons (guardrails for the next analysis)

1. **Never threshold raw centroid speed near the noise floor.** Use `speed_smoothed_mm`
   or bout-level metrics. Any speed-threshold result should be checked with a threshold
   sweep (does it survive above ~2 mm/s?).
2. **Static-epoch near-vs-far metrics are a graveyard here** — the object *moves*
   between pre/post (position confound), near-object occupancy is sparse (noisy
   conditional metrics + selection bias when occupancy itself changes), and objects
   are wall-adjacent and uncounterbalanced. Prefer **bout-level, event-aligned, and
   during-the-chase** metrics.
3. **Wall-proximity is a MEDIATOR — and the virtual controls, not partial correlation,
   are how to handle it.** The chaser drives the fish to the wall (91% of chase time
   near-wall) and to a localized sector (angular R=0.55 chase vs 0.30 pre) — NOT
   uniform thigmotaxis. Partialling out wall-distance over-controls the mediator: on
   n=11 it looked like it killed the habituation (partial r=0.11, p=0.47), but on the
   full cohort the habituation **survives** the same partial (r=+0.23, p=0.003), and
   the object-vs-**virtual-control** radial analysis (identical wall proximity, no
   object) isolates object-specific avoidance directly. **Prefer the virtual-control
   contrast (`*_excess_vs_virtual`, `_wall_excluded`) over any wall partial correlation.**
4. **"Thigmotaxis" ≠ radial wall-proximity.** Check the angular distribution before
   calling wall-hugging thigmotaxis.
5. **Always query the canonical `/groups` registry.** The n=11–12 slice was a stale
   `/nvme1` registry, not missing data. The durable analyses now default to the
   canonical registry via `resolve_cohort()` and run on n=32.

## Code changes landed (branch `sun`)

- `7e931481` — `chaser_response_regimes` classifies immobility on `speed_smoothed_mm`
  (+ contract doc). 15/15 tests pass; backward-compatible fallback.
- `261119a6` — retraction across avoidance survey, dropout diagnostics, and the freeze
  + trajectory figure scripts.
- `737e80ab` / `c3291c43` / `52abac35` / `e2e148a2` — bout-rate + freeze + trajectory
  figure scripts, ring per-object videos, GLM-HMM prompt / survey / dropout diagnostics.
- `94ddcce1` — **promoted the scratch analyses to durable scripts** (escape, immobility
  artifact, bout kinematics, bout vigor, habituation, wall mediator, approach-avoidance)
  + `goodcopbadcop_common.py` (shared, deduped `resolve_cohort`).
- `c0083349` — **pointed `resolve_cohort` at the canonical `/groups` registry** (12→32
  fish); full-cohort numbers updated in the durable docstrings.
- `1170e4b4` — `analyze_goodcopbadcop_radial_kinematics.py` (frame-based radial profile,
  object vs virtual).
- `39c0332d` / `c7521b45` — `analyze_goodcopbadcop_radial_turn_direction.py` (per-bout
  radial turn direction, cluster-bootstrapped) + near-shell pre/post learned-vs-innate test.

Durable scripts (in repo, registry-resolved, write figures OUTSIDE it; `python -m
fisheye.analysis.<name>`):
- Stats/figures: `analyze_goodcopbadcop_escape`, `_bout_kinematics_distance`,
  `_bout_vigor_prepost`, `_habituation`, `_immobility_artifact`, `_wall_mediator`,
  `_approach_avoidance`, `_radial_kinematics`, `_radial_turn_direction`.
- Figures only: `plot_goodcopbadcop_freeze`, `_trajectory_prepost`, `_bout_rate`.
- Shared cohort/zarr helpers: `goodcopbadcop_common`.

## Figures (out-of-repo, `$PALETTE_RECORDINGS_ROOT/figures`)

Re-run any durable script to regenerate on the current n=32 cohort. Figures dated
`2026-07-17` are the n=11–12 slice (retained as historical); `2026-07-18` figures are
the full-cohort + radial outputs.

- `goodcopbadcop_freeze_curve` / `_summary` — raw-vs-smoothed contrast (the artifact).
- `goodcopbadcop_trajectory_prepost_*`, `goodcopbadcop_bout_rate_epochs`.
- `goodcopbadcop_bout_kinematics_distance`, `goodcopbadcop_bout_vigor_prepost`,
  `goodcopbadcop_habituation`.
- `goodcopbadcop_radial_kinematics_2026-07-18` — frame-based radial profile, object vs virtual.
- `goodcopbadcop_radial_turn_direction_2026-07-18` — per-bout radial turn direction, bootstrapped.

## Scratch analyses behind the findings — PROMOTED (2026-07-18)

The load-bearing scratch analyses are now durable `analyze_goodcopbadcop_*` scripts (see
Code changes landed). Each reproduces its recorded numbers; all use the shared, deduped,
canonical-registry `resolve_cohort`. The remaining scratch-only items are the detection-
dropout diagnostics (separate doc) and the single-fish illustrative trajectory.

## Open threads / next steps (the handoff)

1. **Reach n=40:** run the chaser analysis pipeline on May 29 (4) + July 2 (4) so their
   analysis Zarrs get `chaser_distance_runs`/`track_kinematics`/`swim_bout`; then
   `resolve_cohort` picks them up automatically (no code change). Optional — the n=32
   results are already decisive for the main claims.
2. **Session random effect.** The durable scripts use per-fish paired/one-sample tests.
   For final claims, add a session-level random effect via
   `src/fisheye/group_statistics/goodcopbadcop.py` (9 sessions across June 14 + June 21).
3. **Verify the freeze metric's signal source.** The habituation now leans on
   `freeze_low_speed_fraction`; confirm it is not computed on a raw-speed threshold (the
   immobility artifact) before it goes in a talk. Still the one open verification.
4. **Detection-quality fixes still pending** (separate doc): dish-mask +5% buffer +
   corner→center gate, jump-anchor cascade in `detect_quality.py`. These censor
   wall-proximal frames; they would tighten the near-shell / near-wall readouts.
5. **Tail/C-start kinematics** would give a shape axis for near-vs-far, but the
   registry currently reports `subject_shape`/`tail_kinematics` as largely unmaterialized.

## The honest framing for collaborators

This is an **acute-threat-response dataset with an innate spatial-avoidance component
and one learned change** (n=32). The defensible story:
- (a) **Reflexive flight when pursued:** the fish flees hard during the chase — escape
  22×, bout-rate suppression, and radial velocity away from the object at every distance
  (object-specific vs virtual controls).
- (b) **Innate, retained spatial avoidance:** it steers to pass wider around the
  aggressive object's local region — present *pre-training* and *not strengthened* by it
  (the near-object bout-vigor gradient is the weaker sibling of this).
- (c) **The one learned change:** flee→freeze **habituation** across chase trials
  (p<0.001, survives the wall control). Training shifts the *response to being chased*
  from fleeing toward freezing — it does not teach new spatial avoidance.

"Learned spatial avoidance of the red object" is NOT supported: the spatial avoidance is
innate (present before training), and the static occupancy / immobility / approach-
avoidance metrics all failed or were artifacts. The wall confound is settled by the
virtual controls, not by partial correlations. The single-fish trajectory is a nice
illustration but its void does not generalize (occupancy n.s. cohort-wide).
