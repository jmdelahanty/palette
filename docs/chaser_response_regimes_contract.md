# Chaser response regimes contract

`palette.chaser_response_regimes.v1`

Freeze and escape-gain curves: the fish's **own policy** as a function of distance to the
chaser, written as a component under an existing chaser-distance run.

- Module: `src/fisheye/analysis/chaser_response_regimes.py`
- Batch runner: `src/fisheye/utils/run_goodcopbadcop_chaser_response_regimes.py`
- Tests: `tests/unit/fisheye/test_chaser_response_regimes.py`
- Zarr path: `analysis/chaser_distance_runs/<run>/chaser_response_regimes/<component>`

## Why this exists

Occupancy over distance cannot answer "what does the fish do when the chaser is close?".
Two reasons, and both are fatal:

**Distance is a joint quantity.** It changes when the fish moves *and* when the chaser
moves, and in a pursuit assay the chaser is actively driving it. A pile-up of frames at
some radius can be entirely the chaser's controller succeeding. You cannot read a fish
policy off it.

**The distance distribution has a hard floor that is not behavioral.** The chaser's
controller clamps the dot when its edge reaches the fish, so the minimum attainable
distance is the dot's own radius (`chasers[].radius_mm`). On GoodCopBadCop that is 2.0 mm,
and the observed minimum is 2.01 mm with zero frames below. The hole at the centre of the
radial profile looks exactly like a behavioral keep-out zone. It is geometry.

So this module measures the fish instead. It decomposes the rate of change of distance
onto the fish→chaser axis and conditions behavior on distance.

## Outputs

`regimes/`, axis order `[epoch, chaser, distance_bin]`:

| array | meaning |
|---|---|
| `immobile_fraction` | **the freeze curve** — P(speed < `immobility_speed_threshold_mm_s`) |
| `fish_radial_velocity_moving_mm_s` | **the escape gain** — E[v_r \| fish is swimming] |
| `fraction_moving_away` | share of swimming frames headed away from the dot |
| `fish_radial_velocity_mm_s` | fish velocity along the axis (+ = toward the chaser) |
| `chaser_radial_velocity_mm_s` | chaser velocity along it (+ = toward the fish) |
| `fish_separation_rate_mm_s` | the fish's contribution to d(distance)/dt (+ = opens the gap) |
| `chaser_separation_rate_mm_s` | the chaser's contribution |
| `net_separation_rate_mm_s` | their sum |
| `frame_count`, `moving_frame_count` | support, persisted raw |

The escape gain is conditioned on the fish actually swimming. An unconditioned mean radial
velocity is dominated by frozen frames sitting at ~0 and tells you nothing.

`per_epoch_chaser/` collapses these to scalars: `freeze_index` (= `immobile_fraction_near`
− `immobile_fraction_far`; positive means the fish freezes *more* when the chaser is close),
`escape_gain_near_mm_s` / `escape_gain_far_mm_s`, `approach_fraction` (a fish with no
approach routine sits well below 0.5), `min_distance_mm`, and `distance_floor_is_clamp`.

## Tracking QC is a first-class output, not a footnote

**A freezing fish is the hardest target a detector can be given.** The frames that matter
most are the frames most likely to be missing. On the one GoodCopBadCop recording that
exists today, the chase epoch has **44.8% fish dropout in a single contiguous 80.5-second
block** — the fish froze at the arena edge and the tracker lost it there. `pre_event` and
`post_event`, where the objects are static, have 0.08% and 0.48%.

So `tracking_qc/` records, per epoch × chaser: `dropout_fraction`, `gap_count`,
`longest_gap_frames`, `longest_gap_s`, `analyzable_pairs`. Two QC warnings fire:
`high_tracking_dropout:<epoch>:<chaser>` and `long_tracking_gap:<epoch>:<chaser>`.

**Wherever dropout is material, `immobile_fraction` is a LOWER BOUND on the true freeze
rate.** Read the freeze curve next to the tracking QC, never alone. The batch runner prints
the warning directly under the freeze index for exactly this reason.

## Undersampling guards

Two, because a mean over a handful of frames reads as a large effect while being noise, and
the fish is frequently almost never near a given object:

- `min_bin_frames` (default 20) — a per-bin curve value is `NaN` below this. `frame_count`
  is still persisted raw, so the support is visible.
- `min_band_frames` (default 50) — the near/far band scalars (`freeze_index`,
  `escape_gain_*`) are `NaN` below this.

Without them, `pre_event` reports a freeze index of −0.446 off a near-band holding a couple
of dozen frames. With them it correctly reports nothing.

## Velocity estimator

Adjacent-frame centroid difference, requiring **both** agents resolved on **both** sides of
the step. This matters: differencing across a tracking gap turns the fish's displacement
during the gap into one enormous fictitious escape bout. The pair mask drops those steps
(there is a test for it). This computed speed feeds the reported `mean_speed_mm_s` /
`median_speed_mm_s` and the radial-velocity components.

**Immobile/moving classification uses the smoothed signal, not this raw centroid speed**
(2026-07-17). Raw centroid speed has a jitter noise floor of ~1.6 mm/s that straddles the
1 mm/s immobility threshold, so `immobile_fraction` on raw partly measures tracking noise, not
stillness — it produced a spurious pre→post "avoidance" effect that flipped sign above the
noise floor and vanished on a clean signal (see
`docs/diagnostics/goodcopbadcop_detection_dropout_2026-07-16.md`). The component now thresholds
`speed_smoothed_mm` from the offline `track_kinematics` run (deadbanded between bouts, so
`immobile_fraction` reads as "fraction of time not in a bout"). `diagnostics.immobility_signal_source`
records which signal was used; if the track-kinematics run is absent it falls back to raw
centroid speed and emits an `immobility_signal_fallback_raw_centroid` warning.

## Parameters

| flag | default | meaning |
|---|---|---|
| `--immobility-speed-threshold-mm-s` | 1.0 | frozen below this (matches `cra_near_field`) |
| `--moving-speed-threshold-mm-s` | 2.0 | genuinely swimming above this |
| `--distance-bin-edges-mm` | 0,2,3,4,6,8,12,16,20,30,40,60 | finer near the clamp |
| `--near-distance-mm` / `--far-distance-mm` | 5.0 / 20.0 | the bands the scalars contrast |
| `--min-bin-frames` / `--min-band-frames` | 20 / 50 | undersampling guards |
| `--dropout-warn-fraction` | 0.10 | above this, warn |
| `--long-gap-warn-s` | 5.0 | above this, warn |
| `--settle-trim-s` | protocol | shared with `chaser_radial_occupancy` |

Epoch windows and the settle trim are imported from `chaser_radial_occupancy`, so the two
components are directly comparable on the same frames.

## Usage

```bash
python -m fisheye.analysis.chaser_response_regimes <analysis.zarr>            # dry run
python -m fisheye.utils.run_goodcopbadcop_chaser_response_regimes \
    --zarr <analysis.zarr> --apply --overwrite
python -m fisheye.utils.run_goodcopbadcop_chaser_response_regimes \
    --recording-like '%GoodCopBadCop%' --apply                                # across the registry
```

## What this does not do

It is frame-level. The natural next step is **bout-level**: bin swim bouts by
distance-to-chaser at bout onset and look at the bout-type distribution
(`swim_bouts`, `bout_kinematics`, and the megabouts classifier are all already in the repo).
A genuine regime switch shows as a change in *repertoire* below a critical radius, which is
a far stronger claim than a change in mean speed. `chaser_egocentric_bearing.py` supplies
the angular half.

And it is descriptive. One fish is not an experiment — see
`docs/diagnostics/goodcopbadcop_chase_epoch_findings_2026-07-14.md`.
