# GoodCopBadCop behavior analysis — synthesis & handoff (2026-07-17)

What was tested this session on the GoodCopBadCop chaser cohort, what survived
scrutiny, what was retracted, the methodological lessons, and where to pick up.
Companion to `docs/diagnostics/goodcopbadcop_detection_dropout_2026-07-16.md`
(the detection-quality half) and `docs/goodcopbadcop_avoidance_readout_survey.md`.

## Cohort / data reality (read first)

- **12 reachable analysis zarrs** on `/groups/johnson/johnsonlab/jeremy/recordings`
  (3 sessions, all 2026-06-14, 4 arenas each). One fish per arena.
- **The 2026-05-29 session's 8 zarrs are registered but NOT on local disk** (their
  `/nvme1` paths don't resolve). So everything below is n=11–12, not the 32-fish /
  8-session cohort the earlier diagnostics used. **This underpowering is the single
  biggest limitation** — several marginal calls would likely resolve on the full
  cohort. Getting those 8 zarrs back is the top priority (see Open threads).
- Immobility/speed metrics: **use `speed_smoothed_mm`** (track_kinematics), NOT raw
  centroid diffs — see the artifact below. Bouts come from `chaser_bout_response`,
  detected on the `speed_exponential` signal (robust).

## TL;DR — what is real vs what died

REAL (survived clean-signal + confound checks):
- **Acute escape response during the chase.** Escape rate (bout peak > 100 mm/s)
  pre 0.66 → chase 8.01 per validly-tracked minute, **12.1×, 12/12, p=0.0005** (on
  `speed_smoothed_mm`; 7.8× on the bout-table peak). This is the core finding.
- **Bout-rate suppression during the chase.** ~100 → ~48 → ~84 bouts/valid-min
  (pre/chase/post), **pre→chase p<0.001, 12/12**. Behavioural inhibition under threat.
- **Innate near-object bout-vigor gradient.** Bouts are faster near the aggressive
  object than the inert one (peak near−far diff-in-diff **+2.7 mm/s, p=0.034**), but
  it is present **pre-training (+1.54 mm/s, p=0.042) and does NOT grow with training**
  (post −0.65, Δ p=0.105) → **innate, not learned** (matches known innate red-avoidance).

PLAUSIBLE BUT UNDERPOWERED / needs full cohort:
- **Flee→freeze habituation over chase trials.** Freeze fraction early 0.44 → late
  0.61 (p=0.005, n=11); escape rate 0.84(trial1)→~0.2, early-vs-late p=0.21 at n=11
  (full 32-fish diagnostic: 0.48→0.11, **p=0.0002**). This is the best *learning*
  candidate. See the wall caveat below — do NOT dismiss it via the naïve wall control.

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
3. **Wall-proximity during the chase is a MEDIATOR, not a clean confounder.** The
   chaser drives the fish to the wall (91% of chase time near-wall) and often to a
   localized sector (angular concentration R=0.55 during chase vs 0.30 pre) — so this
   is NOT uniform-perimeter thigmotaxis. Because the chase *causes* the wall-proximity,
   controlling for wall (a mediator) statistically removes part of the real chase
   effect. The naïve partial correlation freeze~trial|wall (r=0.11, p=0.47) was an
   **over-control** and should NOT be used to dismiss the habituation. There is also a
   session-long drift component (pre 57% vs post 72% near-wall), so wall is
   part-mediator, part-confounder — untangling it needs the full cohort or an explicit
   mediation model, not a partial correlation on n=11.
4. **"Thigmotaxis" ≠ radial wall-proximity.** Check the angular distribution before
   calling wall-hugging thigmotaxis.
5. **n=11 is underpowering the marginal calls.** Get the full cohort before final claims.

## Code changes landed (branch `sun`)

- `7e931481` — `chaser_response_regimes` classifies immobility on `speed_smoothed_mm`
  (+ contract doc). 15/15 tests pass; backward-compatible fallback.
- `261119a6` — retraction across avoidance survey, dropout diagnostics, and the freeze
  + trajectory figure scripts.
- `737e80ab` — `plot_goodcopbadcop_bout_rate.py` (cohort + single-fish bout rate/stats).
- `c3291c43` — `plot_goodcopbadcop_freeze.py`, `plot_goodcopbadcop_trajectory_prepost.py`.
- `52abac35` — `chaser_ring_traversal` per-object videos/PNGs.
- `e2e148a2` — GLM-HMM prompt, avoidance readout survey, detection dropout diagnostics.

Durable figure scripts (in repo, write figures OUTSIDE it): `plot_goodcopbadcop_freeze`,
`plot_goodcopbadcop_trajectory_prepost`, `plot_goodcopbadcop_bout_rate`
(all `src/fisheye/analysis/plot_*.py`, registry-resolved, `python -m ...`).

## Figures produced (in `/nvme1/recordings/figures/`, out-of-repo, dated 2026-07-17)

- `goodcopbadcop_freeze_curve` / `_summary` — raw-vs-smoothed contrast (the artifact).
- `goodcopbadcop_freeze_figures_2026-07-17_PROVENANCE.txt` — has a RETRACTED banner.
- `goodcopbadcop_trajectory_prepost_*` — arena_3, immobility on smoothed signal.
- `goodcopbadcop_bout_rate_epochs` (cohort) / `goodcopbadcop_bout_rate_*arena_3*` (single).
- `goodcopbadcop_bout_kinematics_distance` — kinematics vs onset distance (pooled).
- `goodcopbadcop_bout_vigor_prepost` — near-object peak gradient, pre vs post (innate).
- `goodcopbadcop_habituation` (cohort) / `goodcopbadcop_habituation_sheet_arena_3`.

## Scratch analyses behind the findings (NOT yet durable)

Ephemeral scripts in the session scratchpad computed the results above:
bout-rate/IBI, smoothed recompute + threshold sweep, escape re-confirm, approach-
avoidance, occupancy diff-in-diff, bout kinematics vs distance (+ pre/post split),
habituation cohort, freeze~wall partial, wall angular concentration. **If these
findings are kept, promote the load-bearing ones to durable `plot_*`/analysis scripts.**

## Open threads / next steps (the handoff)

1. **[TOP] Restore the 2026-05-29 session's 8 analysis zarrs to local disk** and re-run
   on the full 32-fish / 8-session cohort with a **session random effect**. This is the
   real fix for every marginal call (habituation especially) and lets the group-stats
   machinery (`src/fisheye/group_statistics/goodcopbadcop.py`) do proper mixed models.
2. **Habituation, done right.** With the full cohort: freeze/escape over trials as the
   learning signal, treating chaser-driven wall-proximity as a *mediator* (mediation
   analysis), not a nuisance to partial out. Also verify the freeze metric
   (`freeze_low_speed_fraction`) is not itself on a raw-speed threshold.
3. **Detection-quality fixes still pending** (separate doc): dish-mask +5% buffer +
   corner→center gate, jump-anchor cascade in `detect_quality.py`. These censor
   wall-proximal frames and would affect any wall/occupancy/near-object metric — worth
   landing before the wall-mediation analysis.
4. **Bout kinematics as the near-vs-far axis that works.** #1 in the survey doc's menu.
   Split by epoch and by escape-vs-ordinary; robust because bout-level.
5. **Tail/C-start kinematics** would give a shape axis for near-vs-far, but
   `subject_shape`/`tail_kinematics` is 0/12 materialized on this cohort.

## The honest framing for collaborators

This is an **acute-threat-response + innate-red-response** dataset. The defensible
story is: (a) the fish flees hard when actively pursued (escape 12×, bout-rate
suppression), (b) it has an innate faster-bouts-near-red response (not learned), and
(c) a plausible but not-yet-nailed flee→freeze habituation over trials (needs the full
cohort). "Learned spatial avoidance of the red object" is NOT supported — occupancy,
immobility, and approach-avoidance all failed or were artifacts. The single-fish
trajectory example is a nice illustration but its void does not generalize (occupancy
n.s. cohort-wide).
