# Avoidance readout survey — new measurement axes for GoodCopBadCop

Companion to `docs/glm_hmm_sample_avoid_state_analysis_prompt.md`. That doc
asks how to *model* a sample-vs-avoid state; this one asks the prior question —
**what should we measure at all?** It maps the avoidance-behavior literature
onto the components that already exist in this repo, and ranks the untapped
ones by how ready they are to feed a cohort comparison.

Written 2026-07-16 from a repo-component inventory + a PubMed/web literature
pass. Coverage/maturity claims reflect the state of the tree on that date;
re-check before relying on them.

## The reframe: change the axis, not the statistic

Every metric currently in the GoodCopBadCop cohort statistics lives on **one
axis** — locomotor-spatial (where the fish is, how it steers, how much it
moves). The cross-recording contrasts in `src/fisheye/group_statistics/
goodcopbadcop.py` (`DEFAULT_METRICS`) consume exactly six metric families and
nothing else:

1. `chaser_distance` — `chaser_epoch_distance_summary`
2. `spatial_occupancy` — `chaser_epoch_spatial_occupancy_zones`
3. `epoch_behavior` — speed, bout counts/rates, IBI
4. `cra_primary_endpoint` — `chaser_cra_primary_endpoint_summary`
5. `cra_near_field` — `chaser_cra_near_field_summary`
6. `egocentric_alignment` — `chaser_egocentric_epoch_summary`

That axis is precisely the one poisoned by the two dominant problems in the
2026-07-14 diagnostics: the **colour/position confound** (aggressive = red at
fixed positions in 32/32 recordings; see
`docs/diagnostics/goodcopbadcop_cohort_results_2026-07-14.md:135-142`) and
**tracking dropout** (the detector loses the fish exactly when it freezes
hardest, flipping escape-rate p-values and invalidating near-zone occupancy;
see `..._escape_pursuit_2026-07-14.md` and `..._chase_epoch_findings...`).

The literature on threat response in fish measures at least three *other*
axes — cardiac, escape-kinematic, and relational/coupling — and each sidesteps
a confound that currently sinks the locomotor-spatial work. The point is not a
cleverer test on the same axis; it is a different axis.

## Literature landscape

Sources via PubMed and web search (July 2026).

- **Cardiac / conditioned bradycardia.** According to PubMed, Matsuda et al.
  2017 (*Sci Rep*) showed late-stage zebrafish larvae develop a **conditioned
  bradycardia** response to a threat cue, and — critically — "a substantial
  population" displayed it while others did not, i.e. it naturally splits
  responders from non-responders on an autonomic axis
  ([DOI](https://doi.org/10.1038/s41598-017-10794-0)). The looming literature
  corroborates a bradycardia response during conditioned fear in larvae.
- **Calibrated threat assessment (escape latency / stereotypy / freeze
  balance).** Bhattacharyya, McLean & MacIver 2017 (*Curr Biol*): higher
  threat → shorter-latency, more stereotyped (Mauthner-type) escapes plus more
  freezing; lower threat → longer-latency, more kinematically variable escapes
  ([DOI](https://doi.org/10.1016/j.cub.2017.08.012)). So latency, escape
  stereotypy, and the freeze-vs-flee balance are all *graded* threat readouts.
- **Fast-start kinematics.** Nair & McHenry 2015 (*J Exp Biol*): the fast
  start's stage-1 bending angle encodes escape direction; a framework for
  turn-angle / directional-control metrics from tail geometry
  ([DOI](https://doi.org/10.1242/jeb.126292)).
- **Approach-avoidance conflict.** Maximino et al. 2011 (*Prog
  Neuropsychopharmacol Biol Psychiatry*) frames scototaxis (white avoidance)
  as an approach-avoidance *conflict* readout of anxiety
  ([DOI](https://doi.org/10.1016/j.pnpbp.2011.01.006)) — motivating discrete
  approach/abort decision events as a metric.
- **Directly comparable robot paradigms.** "Fish adapt and dynamically avoid
  an approaching robotic fish across repeated exposures"
  ([*Sci Rep* 2026](https://www.nature.com/articles/s41598-026-44115-1)) and
  "Zebrafish Adjust Their Behavior in Response to an Interactive Robotic
  Predator"
  ([*Front Robot AI* 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7806020/))
  quantify avoidance via extreme turns, accelerations, and freezing, and the
  latter uses **information-theoretic coupling** (transfer entropy) between
  robot and fish — motivating a relational metric that drops absolute position.
- **Bout-type repertoire.** Megabouts classifies swim bouts into ~13
  categories (C-start, J-turn, slow swim, routine/fast turn, scoot, struggle)
  and is built to be frame-rate/tracking robust
  ([bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.09.14.613078v3.full.pdf)) —
  motivating an avoidance readout as a *shift in bout-type composition* rather
  than in movement magnitude.

## Component inventory — what exists, and what feeds the stats

| Signal | Component / path | Output | Coverage / maturity | In cohort stats? |
|---|---|---|---|---|
| **Heart rate / bradycardia** | `playgrounds/heartrate_stabilization/`, `src/fisheye/analysis/heart_photometry_projection.py`, `local_rostral_heartrate.py` | Window-level cardiac rate (bpm) | **Embedded fixed-camera trials only** are trustworthy; freely-moving top-view explicitly not reportable as rate (`docs/heartrate_final_decision_2026-07-12.md`). Playground-grade. | No — and blocked on this rig (see below) |
| **Freeze curve / escape gain** | `src/fisheye/analysis/chaser_response_regimes.py` | `immobile_fraction(distance)` = P(frozen \| distance); `fish_radial_velocity_moving` = escape speed given swimming, by distance | Well-developed distance-conditioned policy curves; per-recording | **No** |
| **Tail / fast-start kinematics** | `subject_shape_runs.py`, `tail_kinematics_runs.py`, `bout_kinematics.py` | Per-frame midline/tail spline; per-bout peak speed, turn/escape angle, latency | subject_shape **hit production evidence 2026-07-15**; tail_kinematics active but canary-stage | **No** (`bout_kinematics_metrics` exists, absent from `DEFAULT_METRICS`) |
| **Egocentric bearing** | `chaser_egocentric_bearing.py` | Fish-centric chaser bearing + distance | Mature, exported | **Yes** — but signed turn-toward/away is not a separate metric (refinement opportunity) |
| **Escape rate / trigger / pursuit** | `chaser_escape_events.py` | Escape rate (per valid-tracked min), proximity trigger distance, pursuit recovery | Mature, cohort-aware | **No** in `DEFAULT_METRICS`, but already analyzed in the escape-pursuit diagnostic |
| **Bout-type composition** | `megabouts_classifier.py`, `bout_classification_runs.py` | Per-bout category label | **Draft** integration, external `megabouts` dependency | **No** (epoch summary counts bouts, does not type them) |
| **Eye / gaze** | `eye_angle_analysis.py`, `chaser_gaze_tracking.py` | Per-frame eye angles; gaze vs chaser bearing | Eye angle accepted; **vergence gaze sub-axis deprecated** (`docs/eye_angle_legacy_vergence_gaze_todo.md`) | **No**, descriptive-only |

## The cardiac axis: literature-ideal, rig-blocked

Bradycardia is the single most attractive idea in the literature for this
problem, for reasons that are worth recording because they define what we are
missing:

1. It is autonomic, not locomotor — the colour/position confound cannot touch
   it.
2. It does not need centroid tracking — immune to the dropout that corrupts
   escape counts and near-zone occupancy (the fish freezes hardest exactly
   when the tracker fails, but its heart is still resolvable).
3. It would define a learner/non-learner split on an axis *independent* of the
   locomotor avoidance being tested — the clean fix for the circularity
   prerequisite in the GLM-HMM doc.

**But it is not available on the GoodCopBadCop rig.** The repo's own final
decision (`docs/heartrate_final_decision_2026-07-12.md`) rules out a
trustworthy rate trajectory for freely-moving top-view recordings; heart rate
is validated only for the embedded fixed-camera preparation, which is a
different experiment. The only chaser tie-in is a lone exploratory script
(`playgrounds/heartrate_stabilization/align_hr_to_chase_trials.py`). Treat the
cardiac axis as a **design goal for a future rig**, not a current option.

## Ranked, actionable now

1. **Mid-band immobility curve (`chaser_response_regimes`) — RETRACTED 2026-07-17:
   the effect was a raw-tracking-noise artifact.** Tried 2026-07-16, it *looked*
   strong: distance-resolved `immobile_fraction` at 7–18 mm from the aggressive
   object rose 0.377 → 0.550 pre→post (Δ +0.177, 10/10 recordings, Wilcoxon
   p=0.002), aggressive- and distance-specific. **It does not survive.**
   Immobility was thresholded on RAW centroid speed, whose jitter noise floor
   (~1.6 mm/s median) straddles the 1 mm/s threshold — so the metric partly
   measured tracking noise. The tell is a threshold sweep: the raw effect is
   +0.276 at 0.5 mm/s, +0.177 at 1 mm/s, then FLIPS to −0.075 (p=0.002) by 3 mm/s.
   A real "more still" effect stays positive at every threshold; a sign flip means
   the noise *distribution* reshaped, not stillness. On the pipeline's
   `speed_smoothed_mm` (deadbanded between bouts) the effect is **Δ +0.004,
   p=0.85 — gone.** Bout rate near the object is flat (p=0.70), per-bout vigor
   (peak speed / path / duration) is unchanged, and there are no ≥2 s holds. So
   there is **no trustworthy mid-band immobility avoidance effect.**
   `chaser_response_regimes` was fixed to classify immobility on `speed_smoothed_mm`
   (commit 2026-07-17; see the contract doc). Corrected figures (raw-vs-smoothed
   contrast) are at
   `/nvme1/recordings/figures/goodcopbadcop_freeze_{curve,summary}_2026-07-17.png`.
   **Do NOT promote this as an avoidance readout.** Lesson: any raw-centroid
   speed-threshold metric near the ~1.6 mm/s noise floor is untrustworthy — use
   the smoothed / bout-level signals. What survived the clean-signal check is the
   escape result (12× during chase on `speed_smoothed_mm`, 12/12, p=0.0005).
2. **Tail / fast-start kinematics — NOT runnable on this cohort (checked
   2026-07-16).** Although `subject_shape_runs` hit production evidence
   2026-07-15, **none of the 12 GoodCopBadCop analysis zarrs have subject_shape
   or tail_kinematics runs materialized (0/12).** So this readout is
   production-*capable* but not *actionable* here until a materialization pass
   is run on the chaser cohort. Downgraded from "actionable now" to "needs a
   materialization job first." Once materialized it yields per-bout **escape
   latency, stage-1 turn angle, tail-beat** — the graded-threat substrate
   (shorter latency + more stereotyped escapes under higher threat), within-
   fish and per-event so it dodges the colour confound — and remains the best
   second axis.
3. **Escape directionality relative to threat bearing.** Buildable from two
   components already running (`chaser_egocentric_bearing` + escape heading
   change): the signed "turn-away-given-bearing" readout. The cheap, direct
   version of the GLM-HMM's input→action policy, no HMM required.
4. **Escape rate / trigger / pursuit (`chaser_escape_events`) into
   `DEFAULT_METRICS`.** Mature, but this is where the 8.5× escape result
   already came from — formalizing it is bookkeeping, not new signal.
5. **Bout-type composition (megabouts).** The "policy not magnitude" idea, but
   draft-stage and dependency-gated. Phase two.
6. **Gaze / eye angle.** Descriptive-only, vergence sub-axis deprecated. Skip
   unless the others dry up.

## The split-definition problem, without cardiac

Losing the cardiac axis costs the one clean, *independent* way to label
learners vs non-learners. All the runnable axes above are measured against the
same red aggressive object, so a split derived from them re-introduces the
circularity the GLM-HMM doc flags. Two candidate independent axes remain in
the existing data:

- **Habituation trajectory.** Define "learner" by the *rate of escape decline
  across trials* (a learning-dynamics label), not by steady-state avoidance
  magnitude. The escape-pursuit diagnostic already shows within-fish
  habituation (escapes/valid-s 0.48→0.11 across trials, p=0.0002).
- **Static-object control.** Define the split on response to the *stationary*
  dot, then test on the *moving* chaser. The data shows these are behaviorally
  distinct — static-dot escapes gain no distance (p=0.37) while moving-chaser
  escapes are directed — so a static-object label is not the moving-chaser
  behavior being tested.

## Recommendation

The 2026-07-16 mid-band immobility "avoidance" result was **retracted 2026-07-17
as a raw-tracking-noise artifact** (see item 1) — do not build on it. What
survived the clean-signal (smoothed) check is the **escape result** (12× during
the chase, 12/12 recordings, p=0.0005): that is the solid GoodCopBadCop
avoidance finding and it lives far above the noise floor. **Tail fast-start
kinematics** remains the best untested second axis but must be **materialized on
the chaser cohort first** (0/12 today; see item 2). General lesson from this
episode: prefer the pipeline's smoothed / bout-level signals over raw
centroid-speed thresholds, which are unreliable near the ~1.6 mm/s noise floor.
The GLM-HMM remains downstream of a defensible, non-circular split and
counterbalanced data (see the prerequisites section of the companion doc).

## References

PubMed (cite PubMed; DOIs linked inline above): Matsuda et al. 2017 *Sci Rep*
11865; Bhattacharyya, McLean & MacIver 2017 *Curr Biol* 27(18):2751; Nair,
Azatian & McHenry 2015 *J Exp Biol* 218:3996; Maximino et al. 2011 *Prog
Neuropsychopharmacol Biol Psychiatry* 35(2):624.

Web: Megabouts (bioRxiv 2024.09.14.613078); "Fish adapt and dynamically avoid
an approaching robotic fish" (*Sci Rep* 2026); "Zebrafish Adjust Their
Behavior in Response to an Interactive Robotic Predator" (*Front Robot AI*
2019, PMC7806020).

Internal: `docs/glm_hmm_sample_avoid_state_analysis_prompt.md`;
`docs/diagnostics/goodcopbadcop_cohort_results_2026-07-14.md`;
`docs/diagnostics/goodcopbadcop_escape_pursuit_2026-07-14.md`;
`docs/heartrate_final_decision_2026-07-12.md`.
