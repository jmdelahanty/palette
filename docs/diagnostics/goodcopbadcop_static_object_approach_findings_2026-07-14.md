# GoodCopBadCop static periods: the fly-by, the wall, and the fish that swims over the dot

**Date:** 2026-07-14
**Recording:** `2026-06-14T21-12-08Z_arena_4_GoodCopBadCop` (still the only recording with a
materialized `chaser_distance_run` — 1 of 57)
**Companion:** `goodcopbadcop_chase_epoch_findings_2026-07-14.md` (the chase epoch)

Unlike the chase epoch, the **pre/post static periods are analyzable**: the objects don't
move (no closed-loop confound, no controller clamp) and tracking is clean — 0.08% and 0.48%
dropout, versus 44.8% during the chase.

## Summary

1. **The 2 mm "keep-out zone" is not behavioral.** The fish demonstrably swims right over a
   *static* dot: 0.30 mm from the aggressive object in pre, 1.18 mm from the inert, with 100
   frames under 2 mm. The 2.01 mm floor during the chase is the chaser controller's clamp,
   now confirmed by contrast.
2. **The "fly-by with a veer" is mostly geometry.** The 90°-abeam-at-closest-approach is a
   mathematical identity (a straight line gives it exactly), and ~120° of the ~150° angular
   sweep is what a straight chord produces for free. The residual curvature (~+30°) is only
   modestly above what wall-following past an *empty* point produces (+20°). And it is
   **identical in pre and post** — training changed nothing about it.
3. **The post-period "standoff" is a sampling artifact**, not a learned avoidance.
4. **The post-period turn-toward bias is entirely wall-following.** The virtual-object
   controls kill it: raw +0.090, excess over the wall null **+0.001**.
5. **Two things survive the wall null.** (a) In **post**, bouts near the *aggressive* object
   re-aim the fish to pass **wider** (+0.55 mm/bout) while all five wall-matched control
   points re-aim it *narrower* — `steering_excess_vs_virtual` = **+1.10 mm**, absent in pre.
   This is the strongest signal in the recording and it points the way the experiment
   predicts. (b) A pre-period turn bias toward the aggressive object (excess +0.179).
   Both rest on 6–9 visits. Hypotheses, not results.

## 1. The fish swims over a static dot

Minimum fish-to-object distance (dot radius = 2.0 mm):

| epoch | object | min distance | frames < 2 mm | frames < 3 mm |
|---|---|---|---|---|
| pre | aggressive | **0.30 mm** | 5 | 9 |
| pre | inert | **1.18 mm** | **100** | 177 |
| training | aggressive | 2.01 mm | **0** | 579 |
| training | inert | 4.84 mm | 0 | 0 |
| post | aggressive | 4.62 mm | 0 | 0 |
| post | inert | 4.78 mm | 0 | 0 |

The fish will happily put its centroid inside a static 2 mm dot. During the chase it never
does — because the controller stops the dot when its edge reaches the fish. That closes the
question from the chase-epoch write-up: **the hole at the centre of the radial profile is
stimulus geometry.**

## 2. The fly-by: mostly geometry, and no pre/post difference

Segmenting approach events (enter within 12 mm, exit past 18 mm):

| epoch | object | \|bearing\| at entry | \|bearing\| at CPA | angular sweep | tortuosity | n events |
|---|---|---|---|---|---|---|
| pre | aggressive | 39° | 90° | 151° | 1.46 | 4 |
| pre | inert | 17° | 85° | 140° | 1.48 | 9 |
| post | aggressive | 52° | 101° | 156° | 1.28 | 6 |
| post | inert | 66° | 88° | 101° | 1.37 | 10 |

This *looks* like an approach, a veer, and a curved pass. Two of the three are artifacts of
how the numbers are defined. Both traps are easy to fall into and were fallen into here.

### The bearing at CPA is a mathematical identity

At the closest point of approach the distance is at a minimum, so its time derivative is
zero, so the radial velocity is zero — which means the velocity is **necessarily**
perpendicular to the line to the object. **A dead-straight trajectory yields exactly 90° at
CPA.** Bearing is measured against heading rather than velocity, and heading differs from
course by ~19° (median), which is exactly the observed spread (85°–101°).

The 90° is a tautology. It is not a veer and carries no behavioral information.

### Most of the angular sweep is forced too

A perfectly straight chord entering at 12 mm, passing at CPA *b*, and exiting at 18 mm
sweeps a fixed angle about the object for free. The only meaningful quantity is the
**excess** over that null, conditioned on the CPA the fish actually achieved:

| epoch | reference | n | CPA | sweep | straight-line null | **excess** |
|---|---|---|---|---|---|---|
| pre | aggressive **object** | 4 | 7.0 | 150.8° | 121.3° | **+29.5°** |
| post | aggressive **object** | 6 | 7.1 | 155.5° | 119.3° | **+36.2°** |
| post | agg *virtual_180* | 8 | 8.5 | 125.3° | 106.1° | +19.2° |
| post | agg *virtual_240* | 8 | 4.9 | 159.2° | 138.9° | +20.3° |
| post | inert **object** | 10 | 10.2 | 101.4° | 87.1° | +14.3° |
| pre | ine *virtual_240* | 5 | 7.0 | 133.3° | 121.1° | +12.3° |
| pre | agg *virtual_120* | 7 | 4.5 | 126.9° | 143.8° | −16.9° |

So ~120° of the ~150° sweep is what a straight line gives you. The residual is ~+30°.

**And the virtual references — empty points with no object — produce excess sweeps up to
+20°.** A curving, wall-following fish generates excess sweep around *anything*. The real
object's +30–36° sits above that spread, but not far above it, on 4 and 6 events.

### There is no pre/post difference

For the aggressive object: median CPA 7.0 → 7.1 mm; excess sweep +29.5° → +36.2°. Those are
the same number twice. Whatever the fly-by is, it is **not** something training changed.

### What survives

The fish passes on a curved path rather than a straight one (tortuosity 1.28–1.48, excess
sweep positive). That is all. It is not different pre versus post, and it is not cleanly
object-specific once straight-line geometry and wall-following curvature are subtracted.

**n = 4–10 events.** These are medians of a handful of excursions and should not be
reported as effects.

## 2b. It is not a collision course, and it does not dodge

The natural reading of §2 is "the fish swims at the dot, then swerves past." It does not.

Define the **predicted miss distance** `b = r · sin(bearing)` — where the fish would pass if
it kept going straight from here. Walking each approach backwards from its closest point:

| | 30 mm | 25 | 20 | 16 | 12 | 8 | CPA |
|---|---|---|---|---|---|---|---|
| post / **aggressive object** | 15.7 | 16.4 | 11.9 | 11.7 | 10.6 | 6.7 | 7.1 |
| post / *virtual_180 (empty point)* | 20.6 | 15.0 | 11.2 | 9.6 | 10.2 | 7.1 | 8.5 |

A collision-course-then-avoid starts near **0** and *rises*. This starts at ~16 mm and falls.
**At 30 mm out, the fish's course already predicts a ~16 mm miss.** It is never aimed at
the dot.

And the falling trend is not behavior either — the empty virtual point does the same thing.
It is forced twice over: `b ≤ r` (the ceiling shrinks with range), and conditioning on
approaches that *did* get close forces `b` to converge on the CPA (at the closest point
`b == r` identically — the same tautology as the 90° bearing).

Within-event, the difference between the miss predicted at entry and the miss actually
achieved is +1.5, −1.8, −2.5, −1.1 mm across the four object/epoch cells — inconsistent in
sign, with roughly half of events "veering" either way. Coin flips.

**Conclusion: the fish passes by on a path that was always going to pass by.** No collision
course, no dodge.

## 2c. But the *bouts* do steer — and only in post, and only around the aggressive object

The trajectory-level question is unanswerable for the reasons above. The **bout-level**
version is not, because it never conditions on the outcome: for each bout, compute the
predicted miss before it and after it. Positive `Δ` = the bout re-aimed the fish to pass
*wider*. That is active avoidance steering.

Δ predicted miss per bout (mm), object versus its five wall-matched virtual twins:

| epoch | reference | Δ miss / bout |
|---|---|---|
| pre | **aggressive object** | **−0.31** |
| pre | its 5 virtual twins | −0.13, −0.36, −0.19, −0.07, −0.40 |
| post | **aggressive object** | **+0.55** |
| post | its 5 virtual twins | −0.82, −0.57, −0.11, −0.58, −0.68 |
| post | **inert object** | +0.28 |
| post | its 5 virtual twins | +0.05, −0.28, +0.27, +0.02, −0.37 |

In **pre**, the aggressive object sits squarely inside its own control spread — no steering.

In **post**, **all five** wall-matched control points steer the fish *narrower*, and the
aggressive object is the only reference that steers it *wider*. `steering_excess_vs_virtual`
= **+1.10 mm**. The inert object shows +0.34 mm — same sign, a third the size.

This is the strongest, cleanest signal in the recording: object-specific, wall-corrected,
free of the CPA tautology, and pointing in the direction the experiment predicts (avoidance
steering appears *after* training, and more for the aggressive object than the inert one).

**It is still n = 9 visits** (229 bouts from 9 approaches; the component emits
`pseudoreplicated:post_event:chaser0:229bouts_from_9visits`). One fish. Treat it as the
hypothesis to take to the cohort, with a cluster bootstrap on `visit_id`, contrasting
aggressive vs inert and pre vs post.

## 3. The post-period standoff is sampling, not learning

It is tempting to read the table in §1 as "after training the fish stops touching the dots."
It rests on 6 approach events. In pre, 1 of 4 aggressive-object approaches went below 3 mm.
Getting 0 of 6 in post, if the underlying rate were unchanged at 1 in 4, has probability
0.75⁶ ≈ **0.18**.

That is chance. There is no standoff result.

This is a general trap with minimum-distance statistics: the minimum is an extreme-value
statistic whose expectation depends on how many times you looked. The fish visits the
objects less in post, so its minima rise for free.

## 4. The circling is the wall (mostly)

Bout-level turn bias — the correlation between an object's bearing at bout onset and the
turn that bout executes. Positive = the fish turns toward the object, which is what
maintains an arc *around* it.

Each object is compared against **virtual objects**: its own position rotated about the
arena centre. A virtual reference has identical distance-from-centre and wall proximity at
every instant, but no object. If the signature survives around virtual references, it is
wall-following.

| epoch | object | near bouts | **visits** | turn_bias_r (raw) | **excess vs virtual** |
|---|---|---|---|---|---|
| pre | aggressive | 128 | 6 | +0.201 | **+0.179** |
| pre | inert | 418 | 9 | −0.049 | −0.069 |
| post | aggressive | 229 | 9 | +0.090 | **+0.001** |
| post | inert | 423 | 11 | +0.094 | **−0.011** |

The post-period turn-toward bias — which looks like a clean object-directed effect at
+0.09 across *both* objects — is **entirely explained by wall-following**. The excess is
+0.001 and −0.011. The fish arcs around the objects because it arcs around everything at
that radius.

Same story for the circling index (`|v_tangential| / speed`): raw values 0.61–0.73 around
the objects, but `circling_excess_vs_virtual` is only +0.03 to +0.07. Most of the
"meandering around the object" is the fish meandering around the arena.

**This is why the virtual controls exist.** Without them, the post-period numbers would have
read as a beautiful object-directed circling response.

## 5. What survives, and why it is not yet a result

The pre-period turn bias toward the **aggressive** object survives the wall null:
excess **+0.179**, while the inert object in the same epoch shows −0.069 — a within-fish,
within-epoch control that points the other way.

That is the one object-specific, wall-corrected signal in this recording. It is also built
on **6 visits**.

`near_bout_count` is not the sample size. 128 bouts drawn from 6 approaches are one approach
subsampled 20-odd times each; they are not independent. A naive p-value on n=128 would be
wildly anticonservative. The component reports `near_visit_count`, tags every bout with a
`visit_id`, and emits `pseudoreplicated:pre_event:chaser0:128bouts_from_6visits`.

**Effective n = 6.** Treat +0.179 as a hypothesis to test on the cohort, not a finding.

## A bug this uncovered

The first implementation computed bearing as `atan2(+dy, dx) - heading`. Arena positions are
**y-down**; track-kinematics heading is **CCW-from-+x in a y-up frame**. Mixing them produces
a bearing that looks entirely plausible and **inverts the sign of the turn bias** — it
reported −0.15 where the truth is +0.20.

Fixed to `atan2(-dy, dx) - heading`, which reproduces `chaser_egocentric_bearing`'s
`bearing_deg` to 0.0000°, and pinned there by a test. Validated independently: with the flip,
the fish's heading agrees with its direction of travel (circular mean error −0.4°); without
it, 87°.

Anyone computing angles against these positions needs to know this. It is now documented in
`docs/chaser_bout_response_contract.md`.

## What to do

1. **Run the cohort.** Every number above is n=1, and the honest ones are n=6 *visits*. The
   visit is the unit that pools: 20 fish × ~6 visits ≈ 120, which is a real n. Cluster
   bootstrap on `visit_id`.
2. **Test the one live hypothesis**: is there a pre-period, wall-corrected turn bias toward
   the aggressive object, absent for the inert one? That is `turn_bias_excess_vs_virtual`,
   contrasted between objects.
3. **Materialize `bout_kinematics` / megabouts.** Binning bout *type* by distance-at-onset is
   a far stronger claim than binning mean speed — a change in repertoire below a critical
   radius. The bout runs exist; the classifiers do not.
4. **Fix detection on stationary fish** (see the chase-epoch write-up). That blocks the
   training epoch entirely.

## Components

- `chaser_bout_response` (`docs/chaser_bout_response_contract.md`) — everything in §4 and §5.
- `chaser_response_regimes` (`docs/chaser_response_regimes_contract.md`) — freeze/escape curves.
- `chaser_radial_occupancy` (`docs/chaser_radial_occupancy_contract.md`) — area-normalized rings.
