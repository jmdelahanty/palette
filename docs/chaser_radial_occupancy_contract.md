# Chaser radial occupancy contract

`palette.chaser_radial_occupancy.v1`

Area-normalized radial ("ring") occupancy around the **moving** chaser, written as a
component under an existing chaser-distance run.

- Module: `src/fisheye/analysis/chaser_radial_occupancy.py`
- Batch runner: `src/fisheye/utils/run_goodcopbadcop_chaser_radial_occupancy.py`
- Tests: `tests/unit/fisheye/test_chaser_radial_occupancy.py`
- Zarr path: `analysis/chaser_distance_runs/<run>/chaser_radial_occupancy/<component>`

## Why this exists

The chaser-distance run already stores a 1-D histogram of fish-to-chaser distance in
`epoch_distributions/`. That histogram is normalized by bin **width**
(`counts / (n_samples * bin_width)`, `chaser_distance_runs.py`), not by the **area** of
the annulus each bin represents.

An annulus at radius *r* has area ≈ 2π*r*·d*r*. So a fish moving completely at random
still produces a histogram that climbs with *r*, peaks somewhere in the middle of the
arena, and falls off once the ring starts getting clipped by the wall. On a 40 mm-radius
arena the null peaks around 45 mm. **Any peak read off the raw histogram near there is
arena geometry, not behavior.**

This component computes the corrected quantity. For every valid frame it takes the
annulus around the chaser's position *on that frame*, clips it to the arena, and
accumulates the available area.

## Relationship to `cra_near_field`

`cra_near_field` already does area-normalized rings — but only for the **static** objects
during the `pre_static` / `post_static` phases, using one frozen object position per
phase. It is the right tool there and this component does not replace it.

This component tracks the chaser frame by frame, so it is valid during the active chase
epoch, where the object actually moves. The two are complementary; both hang off the same
chaser-distance run.

## Outputs

`radial_occupancy/`, axis order `[epoch, chaser, radial_bin]`:

| array | meaning |
|---|---|
| `radial_count` | fish-frames observed in the ring |
| `radial_observed_fraction` | `radial_count / n_valid` |
| `radial_expected_area_mm2` | ring's available area, summed over valid frames |
| `radial_expected_fraction` | the geometric null — share of frame-weighted available area |
| `radial_occupancy_density_per_mm2` | `radial_count / radial_expected_area_mm2` |
| `radial_selection_index` | `observed / expected`; **1.0 == chance** |

Each has a `*_wall_excluded` twin that drops fish samples within `perimeter_band_mm` of
the wall *and* removes that band from the available area, so observed and expected still
refer to the same region.

`distance_cdf/` gives `cdf_observed_fraction`, `cdf_expected_fraction`, `cdf_enrichment`
over the threshold ladder. `per_epoch_chaser/` gives the near-zone scalars plus chaser
motion diagnostics. `control_reference/` gives the same rings around a fixed dish-centre
point, which isolates thigmotaxis from genuine chaser-relative structure.

## Epoch windows and the settle trim

At a phase boundary the static objects travel to their next position over
`position_transition_duration_s` (from the stimulus protocol). Those in-transit frames
belong to neither phase. `cra_primary_endpoint` already excludes them via its effective
phase windows, and this component keeps the same convention: `epochs/start_frame` is the
*effective* start, `epochs/source_start_frame` is the raw boundary from the distance run,
and `epochs/settle_excluded_frame_count` records the difference.

The trim is applied **only to epochs whose chasers are static once settled**
(`epochs/static_configuration`). An epoch where a chaser keeps moving past the settle
window is a dynamic-stimulus epoch — the chase itself — and there is nothing there to
settle; trimming it would silently discard real stimulus frames. "Static once settled" is
decided with the same `motion_spread_threshold_mm` used for `chaser_is_moving`.

On a typical GoodCopBadCop recording this trims 200 frames (2 s at 100 fps) off
`pre_event` and `post_event`, and leaves `training_event` untouched.

Without this trim the ~200 in-transit frames sit inside `post_event` and are enough on
their own to trip `chaser_is_moving` — a spurious "pursuing" flag on an epoch where the
objects are in fact motionless for 99.7% of the time. Override with `--settle-trim-s`;
if the protocol is unreadable the trim defaults to 0 and the raw windows are used.

## Two things that will burn you

**1. The null is geometric, not behavioral, when the chaser is pursuing.**

During the chase the chaser's controller drives it *toward the fish*, so its position is
closed-loop on the fish and is not independent of it. The area normalization is still the
correct geometric correction, but a high selection index there largely reflects the
controller succeeding — it is not evidence that the fish *chose* to be near the dot.

The component flags this: `per_epoch_chaser/chaser_is_moving` is true when the chaser's
within-epoch RMS position spread exceeds `motion_spread_threshold_mm`, and a
`closed_loop_null:<epoch>:<chaser>` QC warning is emitted. In the static phases the chaser
is independent of the fish and the null *is* behaviorally meaningful.

**2. Outer rings are unstable.**

Rings approaching the maximum attainable distance (≈ arena diameter) have almost no
available area, so `observed / expected` is a ratio of two near-zero numbers and explodes
on a handful of frames — selection indices of 10³–10⁴ are easy to produce and mean
nothing. `radial_selection_index` is therefore set to `NaN` in any ring where the null
expects fewer than `min_expected_count` (default 5) samples. Counts and areas are still
persisted for every ring; only the ratio is suppressed. A
`low_expected_count_rings:<epoch>:<chaser>` warning records it.

## Geometry

Read from the stimulus run's `calibration/arena_geometry` that the distance run points at
— not from the CRA endpoint — so this works on any recording with a materialized distance
run.

- **Circular arena** (`experimental_area_shape == CIRCLE`): the clipped annulus area has a
  closed form (circle–circle intersection, depending only on the chaser's distance from
  the arena centre). Exact and vectorised over frames.
- **Rectangular fallback**: grid quadrature, cached per chaser cell
  (`area_cache_step_mm`), reusing `cra_near_field._rectangle_annulus_area_mm2`. Sets
  `geometry_status = rectangular_approximation` and emits a QC warning.

Do **not** reuse the grid quadrature for the circular moving-chaser case: it meshes the
whole arena per call and would need one call per frame (>10⁵), where the closed form needs
one vectorised `arccos`.

## Parameters

| flag | default | meaning |
|---|---|---|
| `--radial-bin-width-mm` | 2.0 | ring width; bins span the arena |
| `--r-zone-mm` | 5.0 | near-zone radius (matches `cra_near_field`) |
| `--perimeter-band-mm` | 5.0 | wall band for the wall-excluded pass |
| `--motion-spread-threshold-mm` | 1.0 | above this RMS spread the chaser counts as pursuing |
| `--min-expected-count` | 5.0 | selection-index suppression floor |
| `--area-cache-step-mm` | 1.0 | rectangular-fallback area cache resolution |

## Usage

```bash
# single archive, dry run
python -m fisheye.analysis.chaser_radial_occupancy <analysis.zarr>

# write it
python -m fisheye.utils.run_goodcopbadcop_chaser_radial_occupancy \
    --zarr <analysis.zarr> --apply --overwrite

# across the registry
python -m fisheye.utils.run_goodcopbadcop_chaser_radial_occupancy \
    --recording-like '%GoodCopBadCop%' --apply
```
