# Eye-axis half-plane margin: a geometric primer

This note explains the geometric quantity `|m̂ · f̂|` that gates the safety of
the eye-axis flip rule used in the v5 eye-angle pipeline. It's a teaching
artifact for understanding *why* Palette resolves the 180° axis ambiguity
on the major axis rather than the minor axis. The companion animation lives
at `apps/manim/gaze_flip_rule.py` (Scene 4 shows the live readout).

## The 180° ambiguity

OpenCV's `cv2.fitEllipse` (and any ellipse fitter) returns axis directions
modulo 180° — a fitted ellipse axis is a *line*, not a vector. The line
through the eye center has two endpoints; either could be "the" axis
direction, and the fitter arbitrarily picks one.

For downstream geometry (drawing a gaze ray, computing nasal angle,
computing vergence), the choice of endpoint matters: if we pick the wrong
end, the resulting vector points 180° from the truth.

## The forward half-plane rule

Palette resolves the ambiguity by keeping the endpoint that lies in the
**forward half-plane** of the body frame:

```
if dot(axis_raw, f̂) >= 0:   keep axis_raw
else:                        negate it
```

where `f̂` is the body-frame forward unit vector. The half-plane test is
robust *if* the axis is comfortably on one side of the boundary. It's
fragile if the axis is near the boundary, because small fit noise can flip
which side the axis happens to land on, and the rule will then negate the
"wrong" one.

For current canonical runs, `f̂` is recomputed from the exact base-keypoint
payload and success mask sealed by the selected subject-shape assignment proof.
A separately persisted upstream heading is not an input to this rule.

## The margin metric

The geometric quantity that captures this robustness is

```
|m̂ · f̂|
```

where `m̂` is the **resolved major-axis** unit vector (after the half-plane
flip rule has been applied). It's the magnitude of the dot product of
two unit vectors, i.e. `|cos(θ)|` where `θ` is the angle between the major
axis and the body forward axis.

| `|m̂ · f̂|` | Geometry | Interpretation |
|---|---|---|
| 1.00 | major axis ∥ body AP                | eye at rest — maximum margin |
| 0.71 | major axis 45° from body AP         | comfortable margin |
| 0.50 | major axis 60° from body AP         | still safe |
| 0.10 | major axis ~84° from body AP        | **marginal** — QA flag fires |
| 0.00 | major axis ⊥ body AP                | exactly on the half-plane boundary |

When the margin is large, fit noise of a few degrees can't push the axis
across the boundary. When the margin approaches zero, the resolution is
ambiguous: a 1° shift in the underlying `alpha_eye` can flip the rule's
output by 180°.

## Why the major axis has high margin

The TN (temporo-nasal) axis of a larval-zebrafish eye lies near the body
AP axis in any biologically plausible state:

- **At rest** (eyes lateral): TN axis exactly parallel to body AP.
  `|m̂ · f̂| = 1.00`.
- **Converged 30° per eye** (extreme prey-capture): TN axis tilted ~30°
  from body AP. `|m̂ · f̂| ≈ 0.87`.
- **Diverged ~35° past lateral** (rare): TN axis tilted ~35° the other way.
  `|m̂ · f̂| ≈ 0.82`.

Across the full biological range, `|m̂ · f̂|` stays well above 0.10. The
half-plane resolution is comfortably robust.

## Why the *minor* axis has low margin

The minor axis (gaze direction) is perpendicular to the major axis, so
`|ĝ · f̂| = √(1 − (m̂ · f̂)²)`. It tells the opposite story:

- **At rest**: gaze points anatomical lateral, perpendicular to forward.
  `|ĝ · f̂| = 0` exactly. **Born at the boundary.**
- **Converged 30°**: `|ĝ · f̂| ≈ 0.50`. Off the boundary, but only because
  the eye has rotated.
- **Diverged past lateral**: `|ĝ · f̂|` swings back toward zero, then
  inverts sign past 90° (the silent failure mode that motivated v5).

Resolving the 180° ambiguity directly on the minor axis is geometrically
fragile because the typical operating point is *at* the half-plane
boundary. The pre-v5 implementation did exactly this and exhibited two
failure modes:

1. **Boundary jitter**: small fit noise around the lateral position would
   flip the resolved minor by 180°, producing implausible frame-to-frame
   gaze velocities.
2. **Past-lateral silent flip**: an eye genuinely rotated past lateral
   (gaze in the backward half-plane) would be mapped to its diametric
   opposite, with `np.clip(signed_minor, -90, 90)` masking the violation.

The v5 fix resolves the major axis (where margin is high) and *derives*
the minor (gaze) direction as the perpendicular. The minor's role is now
output, not resolution input — its low margin no longer matters.

## The QA flag

Palette writes `major_axis_marginal` (per-eye and per-frame) when the
runtime margin falls below the threshold:

```
MAJOR_AXIS_MARGINAL_DOT_THRESHOLD = 0.1
```

In normal recordings this fires zero times. It's a defensive flag for
extremely contorted eyes or genuinely bad fits — the rare frame where the
flip rule's output is uncertain. The validation done during the v5
refactor showed a marginal count of 0 across ~19 000 left-eye and ~19 000
right-eye rows of real data.

## Where this lives in the code

- `MAJOR_AXIS_MARGINAL_DOT_THRESHOLD` in
  `src/fisheye/analysis/eye_angle_analysis.py`
- The half-plane test on the major axis: same file, `_process_chunk`,
  near the comment `axis_major_aligned = axis_major * sign_major[:, None]`
- Per-eye QA arrays: `qa/roi/{left,right}_major_axis_marginal`,
  per-frame: `qa/frame/major_axis_marginal`
- Schema: `analysis.eye_angle_runs` exact compact run schema v7
  (`schema_version = 7`; the closed v2-v6 layout allowlist remains explicit
  legacy compatibility),
  retaining scientific `method_version = "eye_angle_analysis.v5"`; output
  schema v9 carries explicit row identity and acquisition-frame support.

## Animation

The Manim companion animation (`apps/manim/gaze_flip_rule.py`,
`Scene4MajorAxis`) shows `|m̂ · f̂|` updating live as a representative left
eye sweeps from typical convergence through rest into past-lateral
divergence. Watching the readout confirms the geometry: the value peaks at
`1.00` at rest and decreases symmetrically as the eye rotates either
nasally or temporally, never approaching the marginal threshold.

The "correct flow" version of the animation (`MasterCorrectScene`) omits
this readout for narrative clarity; this doc is the corresponding teaching
content for anyone who wants the geometric intuition.
