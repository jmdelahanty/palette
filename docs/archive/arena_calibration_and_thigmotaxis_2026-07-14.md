# The analysis reads the wrong arena circle, and it hid the largest effect in the experiment

**Date:** 2026-07-14
**Recording:** `2026-06-14T21-12-08Z_arena_4_GoodCopBadCop`
**Status:** **FIXED.** `fisheye.shared.arena_geometry` now resolves from the dish mask; all
components re-run. The cohort is unblocked.

## The headline

**Training turns this fish thigmotactic, and it never comes off the wall again.**

Distance from the wall, measured against the **fitted dish mask** (the authoritative circle):

| epoch | median distance to wall | body touching wall (<3 mm) | within 5 mm |
|---|---|---|---|
| **pre** | **7.7 mm** | 15% | 37% |
| training | 1.4 mm | 87% | 92% |
| **post** | **1.7 mm** | **73%** | **87%** |

The fish is ~4 mm long. In pre it swims the interior. It enters the chase, goes to the wall,
and is still there ten minutes later.

`cra_near_field` reports this as `thigmotaxis_frac` **0.354 → 0.353**. No change.
Computed against the correct circle, with *the same function*: **0.367 → 0.873.**

A 2.4× increase, reported as zero.

## The data is fine. The dish is fine. We are reading the wrong circle.

There is **no calibration error and no bad detection data**. An earlier draft of this document
claimed otherwise; that was wrong.

- The dish is genuinely 80 mm (`tank_design_json`: `usable_area_diameter_mm: 80.0`), so
  `experimental_area_radius_mm = 40.0` is physically correct.
- A Hough-fitted **dish mask** already exists in the archive, at
  `analysis_metadata.attrs["dish_mask"]` — `{"method": "hough_circle", "detected_circle":
  {"center": [316, 318], "radius": 304}, "metrics": {"image_shape": [640, 640]}}`.
  `refine_detect` already reads it and gates detections with it.
- Checked in **image space**, where the mask lives and no homography is involved: the fish's
  maximum radial distance is 2127 px against a mask radius of 2143 px. **0.00% of frames are
  outside the dish mask.** At the mask's own scale the fish reaches 39.69 mm in a 40 mm dish.

The problem is that the analysis modules resolve arena geometry from
`calibration/arena_geometry.experimental_area_*` — the **projector's nominal model** of the
dish — and that circle does not coincide with where the dish actually is:

| circle | centre (arena-canvas mm) | radius |
|---|---|---|
| registered `experimental_area_*` (what the code uses) | (40.94, 40.94) | **40.00 mm** |
| **dish mask via the homography** (authoritative) | **(38.06, 41.75)** | **42.44 mm** |
| fish outer envelope (fitted from the data) | (38.05, 41.78) | 41.90 mm |

The dish mask and the fish's own envelope agree on the centre to **0.03 mm**; the 0.54 mm
radius gap is the fish's body half-width (a tracked centroid cannot reach the wall). The
nominal circle is **3.0 mm off-centre and 2.4 mm small**. The radius discrepancy is plausibly
the refraction stack — `tank_design_json` gives `z_eff_mm = 16.73` through acrylic and water —
which the flat `experimental_area` circle does not model.

## Why the wrong circle inverts the metric

Fraction of fish frames that fall *outside* each circle:

| epoch | outside the dish mask | outside the nominal circle |
|---|---|---|
| pre | 0.0% | 10.8% |
| training | 0.0% | 83.9% |
| post | **0.0%** | **56.4%** |

Now look at `cra_near_field._thigmotaxis_for_phase`:

```python
valid, in_bounds, wall = _in_bounds_and_wall_mask(...)   # in_bounds = radial <= arena_radius
count = np.count_nonzero(in_bounds & wall)
total = np.count_nonzero(valid)
fraction = count / total
```

Frames beyond the (too-small) radius get `in_bounds = False`. They are dropped from the
**numerator** but kept in the **denominator** — and they are precisely the frames where the
fish is hardest against the wall.

**The metric fails in proportion to the effect it is meant to measure.** The harder the fish
hugs the wall, the more of its wall-hugging is silently discarded. With the correct circle,
0% of frames are out of bounds and the asymmetry never bites.

## The fix (done)

**`src/fisheye/shared/arena_geometry.py`** is now the single canonical resolver. It reads
`analysis_metadata.attrs["dish_mask"]`, normalizes it (as `refine_detect` already does), scales
it to the detection frame (`source_video_width`, here 4512), maps it through
`analysis/calibration/homography_matrix` minus the arena origin — the same trip the fish
positions take — and fits a circle. It falls back to `experimental_area_*` **only** when no
mask exists, and emits `arena_geometry_fallback_to_nominal` when it does. A projective
homography that maps the mask to a non-circular conic is rejected
(`dish_mask_projection_not_circular`) rather than silently approximated.

Three duplicate `_resolve_arena_geometry` implementations (in `cra_near_field`,
`chaser_radial_occupancy`, `goodcopbadcop_epoch_behavior_summary`) now delegate to it. That
propagates to `chaser_response_regimes`, `chaser_bout_response`, and
`chaser_visit_trajectories`, which import the resolver from `chaser_radial_occupancy`.

**Out-of-bounds is now loud.** `out_of_bounds_notes()` emits
`arena_geometry_out_of_bounds:fish:<fraction>` above 2%. A fish outside the arena is a geometry
error, not a fish.

**The thigmotaxis asymmetry is fixed.** `_thigmotaxis_for_phase` now counts `wall` alone rather
than `in_bounds & wall`. `wall` is unbounded above (`radial >= radius - band`), so a frame past
the radius still counts as wall — which it physically is. A bad radius now degrades the metric
gracefully instead of inverting it.

**Everything is re-run.** On the recording:

| component | geometry |
|---|---|
| `cra_near_field` | `dish_mask` |
| `chaser_radial_occupancy` | `dish_mask` |
| `chaser_bout_response` | `dish_mask` |

- `thigmotaxis_frac`: **0.366 → 0.873** (was 0.354 → 0.353)
- fish out-of-bounds: **0.00%** (was 37%)
- `steering_excess_vs_virtual`, post/aggressive: **+0.92 mm** (was +1.10 mm before the fix),
  with the inert control at +0.19 mm — the contrast sharpened rather than dissolved.

Tests: `tests/unit/fisheye/test_arena_geometry.py` (12), including one asserting the mask wins
over a disagreeing nominal circle, and one asserting a too-small circle makes a wall-hugging
fish read as out-of-bounds and *warn*. 276 passing across the affected suites.

## What this reframes

The "object avoidance" signals in
`goodcopbadcop_static_object_approach_findings_2026-07-14.md` now have a simpler explanation:
the fish is pinned to the wall in post, and the objects sit ~10 mm off it. Not avoidance —
thigmotaxis. The trajectory panels
(`fisheye.visualization.chaser_visit_trajectories`) show it directly, and show the same
wall-hugging around the *virtual* control points, which is the tell that it is global rather
than object-driven.

The bout-level **steering excess** in post was the one result measured against controls at
matched distance-from-centre, and those controls had been rotated about a centre 3 mm off. It
has now been recomputed on the corrected geometry and **it survives**: **+0.92 mm** for the
aggressive object (was +1.10 mm), with the inert object at +0.19 mm. The aggressive/inert
contrast widened from 3.2× to 4.8×. It remains n = 9 visits, and remains a hypothesis for the
cohort rather than a result.

And: **"training induces thigmotaxis" may be the primary result of this assay.** It is large
(0.37 → 0.87), unambiguous, survives every control, and until today was computed as a QC
covariate that reported it as zero. It should be a first-class endpoint, and the cohort run
should be powered to test it.
