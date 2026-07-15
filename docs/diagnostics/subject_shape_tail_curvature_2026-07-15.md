# subject_shape tail curvature: a fit bug fixed, and what it exposed

**Date:** 2026-07-15
**Recording:** `2026-05-29T18-11-16Z_arena_1_GoodCopBadCop` — the only one of 40 GoodCopBadCop
archives with a materialized `subject_shape_run`, and **not** in the 32-fish chaser/escape cohort
(that batch has no chaser distance run). So none of this can yet be crossed with the escape work.

## Why we looked

The escape analysis classifies escapes by centroid peak speed alone. At 100 fps you can *detect*
a fast event but not *resolve* a C-start's kinematics (a ~10 ms C-bend is 1–1.5 frames). The
question was whether the fitted midline in `subject_shape_runs` could at least give the C-bend
*shape* (spatial curvature) that centroid speed cannot. It stores a per-frame 64-point B-spline
midline, tail geometry, and `tail_curvature_px_inv` — so in principle, yes.

## The bug: interpolating spline, differentiated twice

`tail_curvature_px_inv` was noise, not signal. Median max curvature ≈ **1 px radius on a 75 px
tail** — physically impossible. Root cause was the fit, not the masks:

- the pipeline fit an **interpolating** B-spline (`bspline_smoothing = 0`) that passes exactly
  through every mask-skeleton point,
- then took its analytic **second derivative** for curvature.

Skeleton points carry ±0.5–1 px pixel-quantization jitter; the second derivative amplifies it
into garbage. On the **same points**, a smoothing spline gives a ≈68 px radius. So the fault was
the choice to differentiate an interpolating spline.

## The fix (commit b9579617, method v8 → v9)

Positions and arc length keep the interpolating spline. The differentiated tail frame — tangent,
normal, **curvature** — now comes from a **separate smoothing spline**
(`s = n_points * (0.75 px)^2`), recorded in attrs `tail_curvature_method` /
`tail_curvature_smoothing_px`. Smoothing removes jitter but preserves coherent bends: a synthetic
20 px-radius arc reads back at ≈1/20 (test `test_smoothing_preserves_a_real_body_bend`). Falls
back to the interpolating spline if the smoothed fit fails, so it is never worse than before.

## Re-materialized and verified

Re-ran subject_shape on the recording as `subject_shape_goodcopbadcop_arena1_curvfix` (v9); the
old v8 run is kept for comparison.

| | v8 (old) | v9 (curvfix) |
|---|---|---|
| median max tail-curvature radius | **1.1 px** (noise) | **60 px** (sensible) |
| `tail_sample_valid` | 0.901 | 0.901 (unchanged — the fix changes values, not validity) |

Distribution of per-frame max curvature on v9 (129,048 valid-tail frames):

| bend radius | fraction | reading |
|---|---|---|
| > 20 px | **99.53%** | gentle / straight — normal swimming |
| 5–20 px | 0.43% (555 frames) | genuine tight bends the fix now recovers |
| < 5 px (impossible) | **0.04% (51 frames)** | tail-mask artifacts |

## Scope of the metric: tail-only, not whole-body

`tail_curvature_px_inv` covers the **tail segment only** — tail base (≈ caudal swim bladder,
roughly mid-body) to tail tip, i.e. the posterior ~half of the body. It does **not** include the
anterior trunk. Summarizing a frame by its max tail curvature is therefore effectively a
**tail-tip** measure. A C-start is a **whole-body** coil (the anterior trunk curls in stage 1), so
this metric structurally misses the defining feature. The whole-body midline (`centerline_xy`,
snout→tail, 64 pts) IS stored, but there is no clean stored whole-body-curvature metric: the run
outputs tail-only curvature, and an ad-hoc gross head→tail bend computed on the *raw* (unsmoothed)
centerline is itself jitter-corrupted (it returned impossible >360° values on straight fish). The
smoothing fix here is scoped to the **tail** spline; a trustworthy C-bend metric needs the same
smoothing applied to the **full centerline**, which is not yet done.

## What this recording does and does not contain

**It does not contain clean whole-body C-bends.** Sampling the 555 "tight-bend" frames (radius
12–18 px, during swim bouts): the body is a straight diagonal and the high curvature is almost
entirely at the **tail tip** — the last 1–2 points curl. Peak speeds are all low (10–27 mm/s);
these are not escapes. This is a freely-swimming, low-arousal recording with no strong startle,
so there are no C-starts to capture here regardless of tracking quality.

**Two problems, now cleanly separated** (they were both buried under the s=0 noise before):

1. **The fit (s=0)** — pervasive, ~99% of the corruption. **Fixed.**
2. **The masks** — the residual high-curvature outliers are the tail *tip* (both genuinely
   high-curvature during a beat *and* least reliable: thin mask, spline endpoint curl) and
   truncated/jagged tail masks (`tail_sample_valid` fails ~10% of frames). **Not fixed by the
   spline change** — a structurally wrong input is a coherent feature the smoothing spline must
   follow, so it survives.

## To get a trustworthy body-bend / C-bend metric

1. **Compute smoothed curvature over the FULL centerline** (snout→tail), not the tail-only array.
   A C-start coil is whole-body; the current `tail_curvature` (posterior half, tip-dominated)
   misses the anterior bend. Apply the same separate-smoothing-spline approach to the full
   centerline, or integrate the total head→tail turning of the *smoothed* midline (the raw one is
   too noisy). This is the main gap — it is not yet done.
2. Use the **v9** (smoothed) tail curvature if you do use the tail — the v8 array is unusable.
3. **Exclude the terminal tail points** (the tip is where mask/spline is least reliable), or use
   an integrated bend over the robust body region rather than pointwise tip curvature.
4. **QC-reject physically impossible radii** (< ~5 px) — these flag the bad-mask frames.
5. Materialize `subject_shape_runs` on **recordings that contain escapes** (i.e. the chaser
   cohort) — this recording has none.
6. Even then: this resolves C-bend *shape*, not *kinematics* (angular velocity, stage-1 latency),
   which stay behind the 100 fps wall. See `docs/chaser_escape_events_contract.md`
   "Can we see C-starts?".
