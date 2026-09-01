# Validated-behavior cohort export — independent read-only assessment

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-01
implementation: assessment-only
-->

**Assessed:** 2026-09-01, read-only, via the prescribed readers only
(`ValidatedBehaviorExportDataset.open(..., validate=True, full_part_hashes=False)`,
receipt mode, ~23 s; `ValidatedBehaviorStatisticsViewSource.open(...)`). No
globbing, no zarr opens, nothing mutated.

**Subject:**
- Cohort export `goodbatbadbat-validated-behavior-phase-b-20260901-b45aa6a5`
  at `/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_phase_b_20260901_b45aa6a5/publication`,
  manifest sha256 `230c30e0…791d05`; 84 recordings (80 admitted / 4 invalid
  retained), 30 tables, 140.5M rows.
- Derived grouped statistics `/tmp/goodbatbadbat-grouped-statistics-sandbox-v001`
  (manifest `7e75cf51…65cbd4`), 41 metrics / 144 planned contrasts / 141 computed.
- Sandbox worktree `/tmp/palette-validated-behavior-grouped-statistics-20260901`
  (stats layer uncommitted there at assessment time; export code merged to main
  at `bac5a52b`, PR #103/#104).

## Verdict

This is the export contract the 2026-08-29 integration gap audit
(`chaser_integration_gap_audit_2026-08-29.md`) said was missing, and it
substantially delivers. Gap #3 (escape results had no publication path) is
materially closed at the data layer. The identity layer is honest to a fault
and the exploratory posture is enforced in code, not prose.

### Confirmed honored
- **G2 denominators**: `epoch_behavior_summary` stamps
  `rate_denominator = valid_tracked_duration_s` on all 240 rows; near-field
  carries the full v2 censor-counter suite plus
  `near_zone_expected_fraction_geometric`.
- **G9 realized onsets**: `controller_trials` carries realized triggers,
  `trigger_source`, `fallback_used = 0` everywhere; `trial_ordinal` 1–4 with
  319/320 expected trials — the schedule-v2 fixed-n design is real in the data.
- **Escape evidence**: `trial_escape_freeze_summaries` has
  `mean_separation_gain_mm`, `recapture_fraction`,
  `escape_event_rate_per_min` over `valid_time_s`; threshold sweeps persisted
  per trial (10–30 mm/s).
- **Honesty**: every recording says
  `acquisition_batch_identity_status = missing_historical_not_inferred`; every
  stats row says `acquisition_batch_adjustment: not_performed`,
  `analysis_status: exploratory`; `validated_behavior_specs.py:99` refuses
  non-exploratory status. All multiplicity families > 1.

### Findings worth attention (from the exploratory contrasts, n=80 paired)
- Bout-rate suppression recovered: −19.2/min pre→training (d=−0.88),
  incomplete recovery training→post (+12.5), q=0.0006.
- **Candidate persistent aftereffect**: `mean_abs_bout_net_heading_change`
  rises in training AND stays elevated pre→post (−2.47° pre−post, d=−0.51,
  q=0.0006). Most interesting row in the file; unclustered, hold loosely.
- Trials: 208/319 speed-escape, 72 freeze-candidate, recapture ≈0.76 where
  traces valid.

## Ranked residual risks

1. **Near-field/same-quadrant epoch contrasts are mechanically confounded by
   chaser mobility.** Chaser is parked at corner presets pre/post
   (near-zone fraction ≈ 0 for everyone, expected 0.0149) and moves during
   training; "enrichment increases pre→training, q≈0.0008" is stimulus motion,
   not behavior. The meaningful axis is aggressive vs inert WITHIN training
   (aggressive median fraction 0.0063–0.0068, enrichment ~0.42–0.45 = below
   geometric → avoidance; inert 0.0) — absent from `paired_contrasts.parquet`.
2. **Zero virtual-twin controls** (audit gap #4 unchanged); only null is the
   geometric annulus, invalid during pursuit. Twins are computable as a pure
   derived table from `chaser_relative_samples` + the arena authority.
3. **Freeze semantics changed and signal provenance is not exported.**
   Successor (`escape_freeze_successor.py`): threshold 2.0 mm/s inside a
   1.0 s post-trigger window on caller-selected provider `speed_level`
   (publication default `filtered`; CLI accepts `raw`); neither the summaries
   table nor the export manifest records which level the published
   `…20260827_body_frame_projection_v4` runs used. Q1 residue: `filtered` is
   defensible, unlabeled is not.
4. **Trial-level rates on ~19.6 s denominators**: `escape_event_rate_per_min`
   median 12.4 ≈ 4 events/trial-window — fine for modeling, misleading as a
   headline. `recapture_fraction` is NaN (not null) for the 111 no-escape
   trials; engines differ on NaN-in-mean.
5. **`epoch_behavior_summary` still has no thigmotaxis/wall columns** (audit
   gap #5 persists); near-field partially compensates
   (`fish_arena_radius_*`, `fish_wall_distance_*` per epoch×chaser).
6. **`semantic_metadata` is empty on every table checked** — the interface's
   own discoverability channel is unpopulated while the manifest's arrow
   contracts are rich. The definitions that require reading source (freeze
   window, speed level, censor policies) belong there.
7. Trials exist only for the aggressive chaser (correct, undocumented);
   grain strings are excellent and should be the model for the metadata hook.

## Interface verdict

Discoverable and composable: receipt-mode open, grain/primary-key strings,
`collect_bounded`, lazy column-pruned scans over the 47.9M-row tables all
compose cleanly. The equal-recording-weight example (0.973 s vs 0.886 s naive)
is the right default, pedagogically exactly right. Sharp edges a new analyst
will hit: empty semantic_metadata; aggressive-only trials; NaN recapture; and
nothing warns that near-field epoch contrasts are chaser-mobility-confounded —
the explorer renders them as findings.

## Relation to the roadmap

`provider_motion_samples` + `chaser_relative_samples` + `body_relative_samples`
with `trial_ordinal` as the experience axis are sufficient substrate for
B1 (setpoint/gain), B2 (lagged responsiveness), B3 (bout hazard), and
B9 (axial bearing). This export is the input the B-series was waiting for.

## Follow-up implementation (in flight 2026-09-01, separate worktrees)

1. `agent/palette/validated-behavior-role-contrasts-20260901` — standalone
   aggressive-vs-inert within-training contrast module (risk 1).
2. `agent/palette/escape-freeze-signal-provenance-20260901` — speed_level /
   freeze_window / thresholds into the export contract + fail-closed raw
   guard (risk 3).
3. `agent/palette/chaser-relative-twin-nulls-20260901` — rotated-twin derived
   table with rotation-0 parity check against `radial_near_field_summary`
   (risk 2).

### Outcomes (landed 2026-09-01, all tested, none merged yet)

1. **Role contrasts** — commit `983d3771`, 10/10 tests, output
   `/tmp/role-contrasts-v001` (42 rows). Training, aggressive−inert, n=80,
   provider-concordant: near_zone_fraction +0.008 (q=0.0002), entry rate
   +0.4–0.6/min (q=0.0002), enrichment +0.55 (q=0.0002) — but distance_p50
   **+4.5–4.7 mm farther** from the aggressive chaser (q≈0.0004); quadrant
   occupancy no role difference. Reading: training near-zone occupancy is
   substantially **pursuit-driven** (the aggressive dot approaches the fish),
   so the near-zone role excess is not fish choice; the avoidance-consistent
   piece is the +4.5 mm median distance. Pre-epoch role asymmetries exist but
   carry `park_position_asymmetry=true` by construction.
2. **Signal provenance** — commit `305fb4b1`, 305 tests across affected
   suites. The successor manifest already sealed speed_level + all four
   thresholds; the fix threads them into `TRIAL_ESCAPE_FREEZE_SUMMARIES`
   (5 new columns incl. `signal_provenance_status` with sentinel
   `unavailable_pre_provenance_run` for legacy runs; partial/malformed
   provenance fails closed) and gates `speed_level="raw"` behind
   `--allow-raw-speed-level REASON` in both CLI and successor, reason sealed
   into the manifest. Contract payload_sha256 changes; the b45aa6a5
   publication predates the new roster — next export run picks it up.
3. **Twin nulls** — commit `bbc10923`, 21/21 tests, full 80-recording run in
   59 s, output `/tmp/twin-nulls-v001` (5,760 summary rows + 960 excess rows).
   **Rotation-0 parity: exact** on counts/fractions/entry counts (0 deviation)
   and 6.3e-6 mm on distance quantiles; the hysteresis entry policy
   reproduced exactly. Twin-excess, training/aggressive/keypoint:
   near-zone excess −0.0028 mean (57/80 negative, Wilcoxon p=3.1e-4),
   distance_p50 excess +1.48 mm (p=0.017) — a **small genuine
   object-avoidance component on top of a mostly-geometric enrichment**.
   Striking: object-specific avoidance is **strongest in the PRE epoch**
   (near excess −0.0087, p50 excess +5.1 mm, p≈1e-12), weaker during training
   (pursuit pulls observed distance toward the twins), intermediate post;
   inert chasers show similar pre/post excess to aggressive — object-specific
   but not role-specific outside training. Consistent with innate
   red-avoidance; the twins are load-bearing exactly as the sesh3 bearing
   analysis predicted.

### New residue found by the twin-null work
**The arena center is not published anywhere in the export** — the
`reviewed_arena_and_scale` binding carries digests and zarr refs but no
coordinates. The twin module recovers it per recording×provider by
constraint-fitting against the summary's own `fish_arena_radius_mean_mm`
(worst residual 2.4e-7 mm, gated at 0.01 mm, method recorded in its
manifest), but the export should publish center + radius explicitly.

### Revised synthesis after all three
The cohort's spatial story now reads: innate object avoidance present from
the pre epoch (twin-verified, both roles), largely geometric "enrichment"
during training with a small genuine avoidance component, pursuit dynamics
dominating proximity, and the best learned-change candidate remaining the
persistent post-training turning increase (`mean_abs_bout_net_heading_change`,
q=0.0006, unclustered). B-series items B1–B3 are the right next consumers.
