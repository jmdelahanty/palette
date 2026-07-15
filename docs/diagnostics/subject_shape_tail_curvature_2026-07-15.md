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

### Use the integrated angle, not the max

The right per-frame summary is the **integrated tail angle** — the sum of the between-point
tangent-angle changes along the tail (= ∫|κ|ds), a total bend in degrees. Max pointwise curvature
is an outlier statistic that lands on the noisiest/most-artifacted point (the tip); it is fine for
*showing the bug* (a 1 px radius is impossible) but wrong for *measuring bend*. This metric already
exists as `tail_kinematics_runs.integrated_abs_tail_curvature`.

Integrated tail bend, base→tip (deg), v8 vs v9:

| | v8 (s=0) median | v9 median | v9 p90 / p99 |
|---|---|---|---|
| total \|bend\| | **116°** (noise) | **5.7°** (near-straight) | 10° / 16° |
| net base→tip | 31° | 5.7° | — |

The sum of between-point angles is exactly where the s=0 jitter accumulates, so it is the metric
that most exposes the bug (116° of fake bend on a straight fish) and most benefits from the fix.
A consistency check: in v9 total\|bend\| ≈ net turn (5.7 ≈ 5.7), i.e. the smoothed tail is a clean
monotonic curve with no fake back-and-forth wiggle; in v8 they diverged (116 vs 31).

For reference, the max-curvature distribution on v9 (129,048 valid-tail frames) — a *diagnostic*
view, not the analysis metric:

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
this metric structurally misses the defining feature. **This was fixed in v10** — a whole-body
`centerline_curvature_px_inv` is now emitted (see the last section); before v10 the only options
were the tail-only array or an ad-hoc gross head→tail bend on the *raw* (unsmoothed) centerline,
which was jitter-corrupted (impossible >360° values on straight fish).

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

1. **Summarize by the integrated tail angle (Σ between-point angle change), not the max.** Max is
   an outlier that tracks the noisy tip; the integral averages over all points and is the actual
   tail-bend in degrees (`tail_kinematics_runs.integrated_abs_tail_curvature`).
2. For a whole-body C-coil, use **`centerline_curvature_px_inv`** (v10, snout→tail), not the
   tail-only array. Integrate it over a **trimmed body** (drop ~8 anterior + ~4 posterior points)
   to exclude the snout-join and tail-tip endpoint artifacts (see the last section).
3. Use the **v9+** (smoothed) curvature — the v8 arrays are unusable.
4. **QC-reject physically impossible radii** (< ~5 px) — these flag the bad-mask frames.
5. Materialize `subject_shape_runs` on **recordings that contain escapes** (i.e. the chaser
   cohort) — this recording has none.
6. Even then: this resolves C-bend *shape*, not *kinematics* (angular velocity, stage-1 latency),
   which stay behind the 100 fps wall. See `docs/chaser_escape_events_contract.md`
   "Can we see C-starts?".

## Whole-body curvature added (method v10, 2026-07-15)

`centerline_curvature_px_inv` (N, K_centerline) is now emitted alongside the tail-only array —
signed curvature over the full snout→tail midline, from the same smoothing spline, available even
without a valid tail base. A C-start is a whole-body coil, so this is the array for a whole-body
bend metric; `tail_curvature_px_inv` remains the tail-beat array.

Re-materialized as `subject_shape_goodcopbadcop_arena1_wholebody` (v10). Integrated bend (Σ|κ|·ds,
deg), whole-body vs tail:

| | median | p90 | p99 |
|---|---|---|---|
| whole-body \|bend\| | 31.7° (22.8° trimmed) | 43° | 61° |
| tail \|bend\| | 6.0° | 11° | 17° |

The whole-body curvature is **U-shaped along the body**: rigid straight trunk (~500 px radius),
curvature elevated at both ends. NOTE — an earlier draft called the anterior end a "snout-join
kink artifact"; that was wrong. The spline reaches the real snout tip (centerline point 0 = the
anatomical snout tip, distance 0), and the anterior curvature is a smooth ramp (only ~1.2x the
just-inside value at the endpoint), not a spike or corner. The rotation-invariance test (curvature
does not change as the fish turns, so a rigid part is temporally stable):

| region | mean signed curv | temporal std | reading |
|---|---|---|---|
| trunk | +0.0008 | 0.0035 | rigid, stable |
| head | +0.0019 | 0.022 | elevated, sign flips ~43% -> NOT a stable rigid shape |
| tail | +0.0003 | 0.016 | variable -> real flexing |

So: the trunk is truly rigid; the tail's variable curvature is real flexing (plus bad-mask outliers
at <5 px radius); the head's elevated-but-sign-flipping curvature is a mix of real head shape and
**instability in the head-region centerline construction** (the bridge/medial path near the head
blob and eyes wanders frame to frame), not a clean spline/join artifact. The endpoints contribute
~27% of the integrated whole-body bend. There is no one-line fix. For a robust bend/C-bend metric:
work in the trunk+tail and use curvature CHANGE over time (a C-start is a transient deviation from
resting posture), not absolute whole-body curvature; trimming the head helps pragmatically because
it is noisier. Actually cleaning the head would mean stabilizing the head-region centerline
construction -- a subject-shape-stage project, not a spline tweak.
