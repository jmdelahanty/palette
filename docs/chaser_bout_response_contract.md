# Chaser bout response contract

`palette.chaser_bout_response.v1`

Bout-level object-relative kinematics, with wall-following and virtual-object controls.

- Module: `src/fisheye/analysis/chaser_bout_response.py`
- Tests: `tests/unit/fisheye/test_chaser_bout_response.py`
- Zarr path: `analysis/chaser_distance_runs/<run>/chaser_bout_response/<component>`
- Requires: an `egocentric_bearing` component on the same distance run (supplies fish heading
  on the camera-frame axis), and a `swim_bout_run`.

## Which bout level (READ THIS — it was a bug once)

The `swim_bout_run` table from `detect_bouts_multi_level` is **multi-level**: it concatenates
the bouts detected at all five speed levels — `speed_raw`, `speed_filtered`, `speed_smoothed`,
`speed_averaged`, `speed_exponential` — into one `tables/bouts`, tagged by a `signal_id` column.
The run marks `default_signal_id` (`speed_exponential`) as the level downstream consumers should
use.

This module **filters to `default_signal_id`**. Reading the whole table counts each physical
bout up to five times and mixes jittery raw peaks with smoothed ones; an early version did
exactly that and inflated every bout and escape count ~5× (all findings survived because the
inflation was near-uniform, but the magnitudes were wrong — see
`docs/archive/goodcopbadcop_escape_pursuit_2026-07-14.md`). A single-level table with no
`signal_id` column is read whole. The chosen level is recorded in the component's diagnostics as
`source_swim_bout_signal_id` / `source_swim_bout_level` / `bout_level_selection`, and pinned by
`test_multi_level_bout_table_is_filtered_to_the_default_level`.

`peak_speed_mm_s` in this component is `peak_physical_speed_mm_s` from the selected level. For
`speed_exponential` that physical speed is estimated from the `speed_filtered` trace. At 100 fps
it under-reads true C-start velocity (a C-start is 1.5–2 frames) — fine for the escape *contrast*,
not comparable to high-speed-imaging absolute values.

## Why the bout is the unit

What the fish does near a static object is an **event**, and events are rare: on a 600 s
epoch the fish comes within 12 mm of a given object only **4–10 times**. Any statistic built
on approach events is descriptive for a single fish.

Bouts are 30–50× denser — 100–400 of them start within 15 mm of an object per epoch — so
the bout is the unit with power. But see the pseudoreplication section: denser is not the
same as independent.

## Outputs

Per bout, per reference (`bouts_per_reference/`, axis `[bout, reference]`):

| array | meaning |
|---|---|
| `distance_at_onset_mm` | how far the reference was when the bout began |
| `bearing_at_onset_deg` | where it was relative to the fish's heading |
| `delta_distance_mm` | did the bout carry the fish away from it |
| `turn_toward` | did the bout rotate the fish toward its side |
| `visit_id` | which near-object excursion this bout belongs to |

Per bout (`bouts/`): `turn_deg`, `peak_speed_mm_s`, `path_length_mm`, `net_displacement_mm`,
`tortuosity`, `duration_s`, `wall_at_onset`.

Binned against distance (`binned/`, axis `[epoch, reference, distance_bin]`): `bout_rate_per_min`,
`median_peak_speed_mm_s`, `turn_toward_fraction`, **`turn_bias_r`**, **`circling_index`**,
`radial_velocity_mm_s`, `tangential_speed_mm_s`, `mean_delta_distance_mm`, plus
`*_wall_excluded` twins.

**`circling_index`** = mean `|v_tangential| / speed`. 1.0 is motion purely *around* the
reference, 0.0 is motion purely toward or away from it. This is the "meanders around it"
metric.

**`turn_bias_r`** = correlation between the reference's bearing at bout onset and the turn
that bout executes. Positive means the fish turns *toward* the reference. That is not
approach — **an arc around a point requires centripetal turning**, so a sustained positive
turn bias at close range is the signature of *orbiting*.

### `delta_predicted_miss_mm` — the steering metric

`predicted_miss = distance * |sin(bearing)|` is where the fish would pass the reference if it
kept going straight from here. Evaluated **before and after each bout**, its change says what
that bout did to the fish's aim:

- **positive** = the bout re-aimed the fish to pass *wider* — active avoidance steering
- **negative** = it re-aimed tighter
- restricted to bouts beginning with the object ahead (`|bearing| < 90 deg`), because with
  the object behind there is no miss distance to steer

Read `steering_excess_vs_virtual` (object minus its wall-matched twins), never the raw value.

**Why this has to be per bout.** The trajectory-level version of the same question is
unusable. `b <= r` by construction, so binning by range mechanically caps it; and
conditioning on approaches that *did* get close forces `b` to converge on the CPA — at the
closest point `b == r` identically. Per bout there is no such conditioning: every bout that
starts near the object is counted, whether or not the fish ever ends up close.

### Do not read the raw `circling_index` at close range

At the closest point of approach the radial velocity is **zero by definition** (the distance
is at a minimum), so the circling index there is 1.0 for *any* trajectory, including a
dead-straight one. The innermost distance bin the fish reaches is populated mostly by
near-CPA frames, so `circling_index` is inflated toward 1 in that bin as a geometric
artifact, not a behavior.

The virtual references absorb this identically — they are subject to the same tautology —
which is precisely why **`circling_excess_vs_virtual` is the quantity to read, and the raw
curve is a diagnostic only.** The same trap sinks two other tempting statistics: bearing at
CPA is always ~90° (same identity), and the angular sweep of a *straight* chord about the
object is already 100–140° for typical entry/exit radii. See
`docs/archive/goodcopbadcop_static_object_approach_findings_2026-07-14.md` §2.

## The confound this module exists to control

The fish wall-follows (~35% of frames in the perimeter band) and the objects sit close to
the wall. **A wall-following fish traces an arc that sweeps around any wall-adjacent object
for free**, manufacturing angular sweep, tangential velocity, and a turn-toward bias with no
object involvement at all. Every "orbiting" signature can be produced this way.

So each real object gets **virtual objects**: its own position *rotated about the arena
centre*. A virtual reference has, by construction, identical distance-from-centre and
identical wall proximity at every instant — it simply is not there. If a signature survives
around the virtual references, it is the wall.

`object_vs_virtual/` holds the object-specific quantities:

- `circling_excess_vs_virtual` = object's near-band circling − mean over its virtual twins
- `turn_bias_excess_vs_virtual` = same for the turn bias

**Read the excess, never the raw object value alone.** On the one recording we have, the
post-period turn bias is +0.090 raw and **+0.001 excess** — the wall null explains all of it.

A virtual twin that lands on top of another real object is dropped (with objects on opposite
corners, a 90° rotation does exactly that), and `virtual_reference_dropped_on_real_object`
is emitted. 90° is deliberately absent from the default rotation set. If *no* twin has
near-band support, the excess is `NaN` rather than a bare number.

A fixed `dish_center` reference is also included, as in `cra_near_field`.

## Pseudoreplication: `near_bout_count` is NOT the sample size

128 near-object bouts in the pre period come from **6 visits**. Bouts inside one visit are
one approach, subsampled — they are not independent draws of the fish's policy. A raw
p-value on n=128 would be wildly anticonservative.

So the component reports `near_visit_count` alongside `near_bout_count`, labels every bout
with its `visit_id`, and emits `pseudoreplicated:<epoch>:<ref>:<N>bouts_from_<M>visits` when
visits fall below `min_visits_for_inference` (default 10).

**The effective n is the visit count.** For inference, resample `visit_id` (a cluster
bootstrap), not bouts. Across a cohort the visit is also the unit that pools: 20 fish × ~6
visits ≈ 120 visits, which is a real n.

## The coordinate-frame trap

Arena positions are **y-down** (image style). Track-kinematics heading is **CCW-from-+x in a
y-up frame**. The object vector's y must be negated to bring it into the heading's frame:

```python
bearing = wrap(degrees(atan2(-dy, dx)) - heading)
```

Skipping that flip produces a bearing that looks entirely plausible and **silently inverts
the sign of the turn bias** — the whole scientific claim. `bearing_deg()` is pinned to
`chaser_egocentric_bearing`'s implementation by `test_bearing_matches_canonical_egocentric_bearing`,
and `test_bearing_without_the_y_flip_would_be_wrong` guards that the test has teeth.

## Parameters

| flag | default | meaning |
|---|---|---|
| `--distance-bin-edges-mm` | 0,4,8,12,16,20,25,30,40,60 | |
| `--near-distance-mm` | 15.0 | the band the scalars summarize |
| `--virtual-rotations-deg` | 60,120,180,240,300 | 90 omitted: it collides on a diagonal layout |
| `--min-virtual-separation-mm` | 8.0 | drop a twin landing this close to a real object |
| `--visit-enter-mm` / `--visit-exit-mm` | 15 / 20 | visit hysteresis |
| `--min-visits-for-inference` | 10 | below this, warn |
| `--min-bin-bouts` / `--min-bin-frames` | 20 / 20 | undersampling guards |
| `--perimeter-band-mm` | 5.0 | wall band |

## Usage

```bash
python -m fisheye.analysis.chaser_bout_response <analysis.zarr>                    # dry run
python -m fisheye.analysis.chaser_bout_response <analysis.zarr> --apply --overwrite
```

## What this does not do

No bout-type classification: `bout_kinematics_runs` and megabouts are not materialized on
the recordings we have. When they are, binning **bout type** by distance-at-onset is the
strongest available claim — a change in *repertoire* below a critical radius, rather than a
change in mean speed.
