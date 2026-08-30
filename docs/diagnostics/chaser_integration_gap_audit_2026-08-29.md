# Chaser analytics integration gap audit — where the August wave dropped previous work

<!-- contract-meta
version: 1
status: active
last_verified: 2026-08-30
implementation: specified-only
-->

**Established:** 2026-08-29 (four parallel read-only Opus audits @ `4326af6d`)
**Purpose:** work queue for parallel fix worktrees. Each queue item below names the files it owns;
claim an item, do not touch another item's files, and record the landing commit in the status column.


Four read-only Opus audits @ 4326af6d: (1) provider path vs legacy chaser suite, (2) marimo surface vs
established findings, (3) roadmap A0–A7 + promoted analyses wiring, (4) guardrail regressions G1–G10.
Nothing written to the repo. Items cross-confirmed by ≥2 agents are marked [x2]/[x3].

## One-line diagnosis
Provenance is captured meticulously in the new producers and then discarded exactly where a scientist reads
it: the export layer, the statistics registry, and the marimo panels. The results that died are wired into
confirmatory inference and dashboards; the results that survived have no path anywhere.

## Ranked (most likely to produce/publish a false result first)

1. **Freeze metric behind the habituation result thresholds RAW centroid speed at 1.0 mm/s** [x2]
   `chaser_escape_freeze_summary.py:432-442` (np.diff(centroid)/ppm*fps), `:55` DEFAULT_LOW_SPEED_THRESHOLD_MM_S=1.0,
   `:889-897` → `freeze_low_speed_fraction:649` → `visualization/chaser_habituation_figures.py:145-148`, and
   `mean_freeze_low_speed_fraction` as a top-line stat in `apps/marimo/components/goodcopbadcop_chaser.py:3796-3799`.
   Noise floor is ~1.6 mm/s; `chaser_response_regimes.py:551-570` was fixed (7e931481), this sibling never was.
   Synthesis open-thread #3 asked for exactly this check. Fix: port `_load_smoothed_immobility_speed` + signal-source diagnostic.

2. **`fraction_of_epoch` (wall-clock denominator) is a PRIMARY confirmatory endpoint and the marimo default** [x2]
   `group_statistics/goodcopbadcop.py:190-195` primary=True (set by bcca19c7, the "hardening" commit); `time_s:182-189` too;
   denominator `export_cross_recording_analytics.py:4209-4213` = phase_len. Marimo `group_analytics.py:189`
   preferred_analysis="fraction_of_epoch". Both it and `fraction_of_detected` are primary → pick-the-significant-one hazard,
   on the family the synthesis retracted. Fix: demote to exploratory; default views to `fraction_of_detected`.

3. **The strongest result (escape 7.8–22×, gain_mm/recapture_mm) has no surface anywhere** [x3]
   `chaser_escape_events` (+ bout_response, response_regimes, gaze, escape_freeze_summary) has no export contract
   (`analytics_exports/*` — nothing), no MetricSpec (38 metrics / 6 families, zero escape/freeze/habituation), no marimo
   consumer. What the catalog calls "Escape outcomes" (`analysis_catalog.py:126-129`) renders the escape_freeze canary
   ("diagnostic only, thresholds not cohort-locked"). Every surviving GCBC result is `--exploratory-only` gated since
   43eacd11 and the migration table (`goodcopbadcop_standalone_inference_migration_2026-08-10.md:45-58`) is 100% open.
   Fix: escape_events export table + session-clustered MetricSpec family + one marimo panel.

4. **Provider path has ZERO null controls** [x2]
   No virtual twin / dish-center / annulus / shuffle in any of ~20 provider modules. `provider_chaser_distance_candidates.py:451-453`
   density has no annulus-area correction (same shape as the thigmotaxis inversion); `provider_occupancy_contrast.py:421-425`
   is a bare difference map; `:276-281` no coverage-parity check across arms. `chaser_behavior_full_v3.yaml:120` requires
   `virtual_reference_controls` on 1/14 modules and NO producer supplies it (unsatisfiable gate). Legacy twins live in
   `chaser_bout_response.py:373-422` (not the occupancy modules) and `chaser_escape_events.py:334-350` reconstructs them
   from stored angles — port that discipline, not defaults.

5. **Provider epoch summary silently drops every thigmotaxis/wall column, and a cohort export + plotter already run on it** [x2]
   `materializers/provider_epoch_behavior_summary.py:404-452` vs `chaser_epoch_behavior_summary.py:698-708`.
   `export_provider_epoch_behavior_cohort.py` emits Parquet with no statistics layer; `plot_provider_epoch_behavior_cohort.py:806,1198-1210`
   draws Mean±SEM on `mean_speed_mm_s` (the metric that died under clustering) with zero `acquisition_batch`/`exploratory`/`watermark`
   hits — a 2061-line figure path built around the 43eacd11 guard. Epoch summary now exists 3× (legacy, provider, shim).
   `SUPPORTED_SPEED_LEVELS` includes "raw" with no rejection (`:98-99,1266`); legacy authoritative mode requires explicit level.

6. **Live cohort marimo (`group_analytics_explorer.py`) shows retracted findings; canonical chaser dashboard is dark**
   - `group_analytics.py:885-888` prints unclustered paired-Wilcoxon `p=` in the quadrant-relocation chart title; caller passes
     bootstrap_iterations=0 (`explorer:944`); producer's `inference_note` (`query.py:3444`) never shown.
   - `n`/SEM in customdata with no hovertemplate (`group_analytics.py:508-532`); `source_speed_level` dropped at
     `query.py:1917-1930`; wall-excluded radial rows computed (`query.py:3794-3822`) and not plotted (`explorer:1273-1281`).
   - `goodcopbadcop_explorer.py` + 13/15 catalog entries fail closed since 2026-07-20 (`goodcopbadcop_interactive.py:2365-2374,2566,2638`);
     ~3.8k lines latent, incl. raw `immobile_fraction` bar with no speed_source (`goodcopbadcop_chaser.py:3440-3452`) and
     wall-clock bout-rate fallback (`:645,674`) — re-arm uncaveated the day the seal lands.
   - Provider views (`provider_chaser_candidate.py:1113-1183`) pool pre/chase/post although `epoch_label` is on every row,
     and rebuild the dead "bouts farther from aggressive object" read; plumbing there is the strictest in the repo.
   - Only the statistics panel (`query.py:863-950`) and gaze panel (`goodcopbadcop_chaser.py:415-438`) do it right — use as templates.

7. **"Session-clustered" inference: off by default, clusters by acquisition batch, singleton primary family** [x2]
   `goodcopbadcop.py:476` cluster="none" default; unit hard-coded `acquisition_batch` (`:511-527,578-596`) yet
   `group_analytics_viewer/query.py:965` labels it "Session-clustered"; subject_id never a grouping factor; `cra_primary_endpoint`
   family can be size 1 via `--contrast` (`:319-326,3041-3048`) and `singleton_test` is audited (`:601-607`) but not rejected.
   Credit: fails closed on fit failure, min 10 batches, unit never bout/frame.

8. **Nothing from the August wave is in any DAG, MetricSpec, or export contract** [x2]
   No provider/relative-frame module in `profiles/*.yaml`, `chaser_profiles.py`, `surface_classification_catalog.py`,
   `stage_catalog.py`, `analytics_exports/contracts.py`. ~30k lines reachable only from unregistered `utils/materialize_*` CLIs.
   Lifecycle recipe: steps 2-4,6 done to a high standard; steps 1 (science contract doc), 5 (nulls), 7 (YAML), 9 (MetricSpec/FDR) skipped everywhere.
   `provider_track_motion` writes into `analysis/track_kinematics_runs/` as an undeclared "successor" of the speed_smoothed_mm source.

9. **Relative frame (813012e9) is A0's chassis, not A0; schedule v2 importer never landed**
   Carries distance/trial_id/censoring/reason codes; lacks radial approach velocity, fish state label, escape onset/recapture flags,
   and the "track loss in immobility = censoring" policy (no state to exit). `body_bearing_deg` is all-NaN because the only
   producer passes `body_frame=None` (`chaser_proxy_relative_frame_adapter.py:557`). `brief_chaser_schedule_importer.md`
   unchanged since a460da72; repo-wide the only schedule-event string is CHASER_POSITIONING_START
   (`visualize_experiment_timeline_combined.py:78`) — the non-perceptual boundary. A1–A7: none fitted (recorded non-goal,
   checklist `:580-587`). A6 wired to INERT not RANDOM_NON_CHASING (`goodcopbadcop.py:868`).

10. **Definitions/authorities duplicated with no parity harness**
    `distance_mm` ×3 (`chaser_distance_runs.py:687`, `provider_chaser_distance_candidates.py:606`, `chaser_relative_frame.py:801`)
    on different row axes (dense frames vs unique stimulus_frame_num), scale authorities (projector vs camera mm/px) and transforms.
    Occupancy binning ×4. Two arena-geometry authorities (`CircularArenaGeometryAuthority` vs `require_dish_mask_arena_geometry`)
    never compared for the same recording — the exact failure mode of the 3 mm inversion. `visit_policy_version` (legacy epoch
    denominator) is stamped by the producer and dropped by the exporter (0 hits in `export_cross_recording_analytics.py`).

11. Smaller: bout detection zero-fills NaN speed before peak-finding (`detect_bouts_multi_level.py:2270,2497`); `per_second/speed_mm`
    = 0.0 for fully-censored seconds (`compute_speed.py:664-671`); detection-quality thread #4 (dish buffer, corner gate, jump cascade)
    dropped with no decision record; `plot_goodcopbadcop_occupancy_heatmaps/trajectory_prepost` missing from the standalone guard list;
    dead modules `chaser_phase_analysis.py`, `chaser_escape_freeze.py`, `goodcopbadcop_epoch_behavior_summary.py`,
    `analyze_goodcopbadcop_immobility_artifact.py`; provider marimo view hardcodes schema_version 5 and never shows the
    `exploratory_proxy` label the storage layer propagates.

## Explicitly fine (all four agents)
Producers: valid-tracked denominators (provider epoch summary better than legacy — unconditional), dish-mask geometry in legacy
components, near-field v2 contract carried into export (3357aecd model remediation), G5 net_mm never read, G8 canonical registry,
speed/bout/epoch-window code REUSED not reimplemented, provider provenance (digests, coverage chain, selector-ineligible),
statistics panel refuses naive p, gaze panel states design + unit.

## Addendum (agent 1 full resend)
- Legacy annulus null is explicitly invalid during pursuit (`chaser_radial_occupancy.py:723` closed_loop_null QC warning) —
  legacy has no valid null in the chase epoch either; porting it forward carries the weakness.
- `chaser_response_regimes.py:92,95,402-403`: 1.0–2.0 mm/s dead band (neither immobile nor moving), no hysteresis.
- Bearing convention is implemented twice inside legacy (`chaser_bout_response.py:236-248` vs `chaser_egocentric_bearing.py`).
- Egocentric summary metrics declared in DEFAULT_METRICS (`goodcopbadcop.py:429-459`) are not in the export table's column tuple
  (`analytics_exports/contracts.py:794-805`).

## Work queue for parallel worktrees

Items are disjoint by file so worktrees do not collide. Q1–Q3 are the ones that change what a talk shows.
Every item is subject to the standalone-guard rule from `43eacd11`: no new confirmatory-looking figure without
session-clustered inference or an EXPLORATORY ONLY watermark.

| id | item | owns (files) | acceptance | status |
|---|---|---|---|---|
| Q1 | Freeze metric on verified smoothed speed | `src/fisheye/analysis/chaser_escape_freeze_summary.py`, its tests, `docs/chaser_escape_freeze_summary*_contract.md` | `_compute_fish_speed_mm_s` reads `speed_smoothed_mm` via the same loader as `chaser_response_regimes.py:551`; emits `immobility_signal_source`; raw fallback warns; a test asserts raw is never silently used | open |
| Q2 | Demote wall-clock occupancy endpoints | `src/fisheye/group_statistics/goodcopbadcop.py` (MetricSpec block only), `apps/marimo/components/group_analytics.py:189` region, tests | `fraction_of_epoch` and `time_s` are `exploratory=True`; `fraction_of_detected` is the sole primary spatial endpoint; marimo default is `fraction_of_detected`; `_of_epoch` labeled "(uncorrected for dropout)" | open |
| Q3 | Escape-events publication path | new table in `src/fisheye/analytics_exports/contracts.py` (+ arrow contract), new `escape_events` MetricSpec family in `goodcopbadcop.py` (separate block from Q2 — coordinate), new marimo panel `apps/marimo/components/escape_events.py` | export reads `event_gain_mm`, `event_recapture_mm`, `escape_rate_per_valid_min`; never `net_mm`; family size > 1; session/batch-clustered; panel states unit + denominator | open |
| Q4 | Provider null controls | `src/fisheye/analysis/provider_spatial_trajectory.py`, `provider_occupancy_contrast.py`, `provider_chaser_distance_candidates.py`, their materializers/tests | rotated-twin + dish-center arms published beside every observed occupancy/density; twins reconstructed from stored angles; annulus-area correction on distance density; coverage-parity guard in contrast | open |
| Q5 | Provider epoch summary parity + cohort plot guard | `src/fisheye/analysis_workflows/materializers/provider_epoch_behavior_summary.py`, `src/fisheye/utils/export_provider_epoch_behavior_cohort.py`, `src/fisheye/utils/plot_provider_epoch_behavior_cohort.py` | thigmotaxis/wall columns present or explicit `spatial_metrics_omitted`; `speed_level` required, `raw` rejected for published summaries; plotter either uses `fit_acquisition_batch_random_intercept` or carries the `--exploratory-only` receipt + watermark | open |
| Q6 | Live cohort marimo honesty | `apps/marimo/group_analytics_explorer.py`, `apps/marimo/components/group_analytics.py` (except Q2 region), `src/fisheye/group_analytics_viewer/query.py` | no `p=` in chart titles outside the statistics-panel policy; hovertemplates render n/SEM; `source_speed_level` propagated and shown; wall-excluded radial rows plotted as primary; "Session-clustered" label reflects the actual cluster unit | open |
| Q7 | Statistics registry hardening | `src/fisheye/group_statistics/goodcopbadcop.py` (validation + defaults, not MetricSpec blocks), `acquisition_batch_cluster.py`, `utils/compute_group_statistics.py` | `singleton_test` rejected for primary tier; clustering default not `"none"`; unit declared once and, if fish repeat across sessions, `(1 \| subject_id)` added; A6 contrast uses `random_non_chasing` | open |
| Q8 | DAG + contract admission for provider modules | `src/fisheye/analysis/profiles/chaser_behavior_full_v3.yaml`, `chaser_profiles.py`, `registry/stage_catalog.py`, new `docs/chaser_relative_frame_contract.md` | provider/relative-frame modules declared `default_enabled: false` with real `depends_on`; `virtual_reference_controls` has a producer or is removed; `provider_track_motion` successor relationship declared; one science-contract doc | open |
| Q9 | Relative frame → real A0 | `src/fisheye/analysis_workflows/chaser_relative_frame.py`, `chaser_proxy_relative_frame_adapter.py`, `src/fisheye/shared/zarr/chaser_relative_frame_schema.py` | body-frame provider wired (bearing non-NaN); radial approach velocity, fish state label, escape onset/recapture flags materialized; censoring policy "track loss in immobility = censoring" implemented once | open |
| Q10 | Parity harness + geometry cross-check | new `tests/integration/...chaser_parity...`, `provider_spatial_grid_policy.py` (assertion only) | one canary recording through legacy + provider: per-epoch distance/speed/bout agreement within stated tolerance; `arena_geometry_selection` vs dish-mask circle asserted within tolerance; camera-frame distance renamed | open |
| Q11 | Export boundary carries policy | `src/fisheye/utils/export_cross_recording_analytics.py` | fails closed unless `visit_policy_version == valid_tracked_observed_transitions_v2`, or exports the policy column + denominator assertion | open |
| Q12 | Censoring in bout detection / per-second speed | `src/fisheye/analysis/detect_bouts_multi_level.py`, `compute_speed.py`, `provider_track_motion.py` | detection per contiguous valid segment (no zero-fill across gaps); `per_second/valid_transition_count` emitted and censored seconds NaN | open |
| Q13 | Subtraction | `chaser_phase_analysis.py`, `chaser_escape_freeze.py`, `goodcopbadcop_epoch_behavior_summary.py`, `analyze_goodcopbadcop_immobility_artifact.py`, `tests/unit/fisheye/test_goodcopbadcop_standalone_guard.py` | four dead modules deleted; `plot_goodcopbadcop_occupancy_heatmaps` + `_trajectory_prepost` added to the guard list; detection-quality thread #4 gets a one-line decision record | open |

Dependencies: Q3 depends on Q7's clustering defaults for its family to be meaningful (can land first as exploratory).
Q8 should land before Q4/Q9 are considered promotable. Q2 and Q3 both edit `goodcopbadcop.py` — separate blocks, rebase carefully.
