# Manim animations roadmap

A plan for extending `apps/manim/` from the current single-file gaze-flip
animation (`gaze_flip_rule.py`) into a small library of teaching scenes
covering each derived metric Palette computes. Sibling doc:
[`README.md`](README.md) (developer handoff for what's already built).

This is a planning document, not a spec. Items here are proposed scenes,
priorities, and open questions; nothing should be built without a quick
sanity check against the user first.

## Goals

1. **Teach the algorithm, not just the output.** Every scene should make
   one transformation visible — the input, the math operation, the output,
   and *why* the operation is there (what artifact it removes, what shape
   it preserves).
2. **Stay faithful to the implementation.** Where feasible, the scene
   should run the actual repo function on a small fixture and animate the
   real output, not a hand-tuned cartoon. Cartoons are fine for didactic
   moments (boundary jitter, 180° flip) where stylization aids clarity.
3. **One concept per scene, composable into a master sequence.** Mirror
   the pattern in `gaze_flip_rule.py`: small numbered scenes plus a
   `MasterScene` that strings them together.

## Sample-data strategy

Decision needed before scenes 2+ are built. Two options:

**Option A — bundled synthetic trace.** Generate a ~5 s, 60 fps synthetic
track in `apps/manim/data/synthetic_trace.py` that exercises every
pipeline branch: a still period, a clean swim bout, a noisy near-threshold
period, a long bout with a deep mid-valley, and a gap. Pros: deterministic,
small, lives in repo, no privacy concerns. Cons: doesn't show real noise
structure; risk of the animation looking "too clean."

**Option B — carved slice of a real recording.** Pick one production zarr
run, extract ~5 s of one track plus its keypoints/eye ROIs, save as a
small zarr or parquet at `apps/manim/data/sample_run/`. Pros: real
detector noise is the whole point of the hysteresis-filter scene. Cons:
larger artifact, needs picking a representative slice, might want to keep
out of git.

**Recommendation:** start with **A** for the smoothing/hysteresis/exp-kernel
scenes (we want to *control* the noise to make the filter's effect
legible), and add **B** later as a "real-data validation" appendix scene
once the didactic ones are in place. A pure synthetic trace also lets us
parameterize noise amplitude with a `ValueTracker` and animate the filter
threshold sweeping through it — hard to do with a fixed real recording.

## Scene families

### Family 1 — Heading and body frame (foundation)

Already partially built in `gaze_flip_rule.py:Scene0Heading`. Worth
splitting out as the formal "before any metric is computed, this is the
body frame" scene.

- **1.1 Heading from swim-bladder→eye-midpoint.** Already exists;
  promote/re-record as a standalone teaching artifact.
- **1.2 Body-frame axes (f̂, l̂).** What positive/negative means in
  body-frame outputs (signed angles, angular velocity sign convention).
  Useful as a referent that later scenes can point back to.

### Family 2 — Track kinematics (speed pipeline)

The richest target. Maps directly to `compute_speed.py:440–587` and
`track_kinematics.py:687–754`.

- **2.1 Frame path distance (raw).** Two consecutive position dots,
  vector between, length readout. Show the >500 px guard kicking out a
  detection-blip frame.
- **2.2 Hysteresis filter.** Show the state machine as an inset diagram
  next to a noisy raw-distance trace. Animate the high/low thresholds as
  horizontal lines, the moving/still state as a colored band, and the
  `min_frames` exit-debounce as a small countdown. Output: `speed_filtered`.
- **2.3 Temporal smoothing (MA / Savitzky-Golay).** Window slides across
  the filtered trace; the kernel shape is shown as a small inset; the
  output curve is drawn behind the input. Make the difference legible by
  zooming in on a single bout edge — the smoothed curve clearly lags the
  raw onset.
- **2.4 Causal exponential kernel.** *Critical scene* — this is the one
  with a known gotcha (it's used for *segmentation* signal, not as a
  speed metric, because it leaks future motion backward across a bout
  edge if mis-applied). The scene should show the exponential rise/decay
  shape, the recurrence `y[i] = α x[i] + (1−α) y[i−1]`, and explicitly
  contrast it against a centered MA on the same bout to make the
  asymmetry visible. End-card: "this candidate is not a speed metric."
- **2.5 Acceleration.** Differentiate `speed_smoothed`, then show the
  centered MA pass that produces `smoothed_acceleration`. Highlight that
  acceleration sign distinguishes onset from offset.
- **2.6 Heading smoothing & angular velocity.** Circular mean over
  causal window (need to handle the wraparound visually — show a small
  circle inset with the running mean as a vector). Then `angular_velocity
  = wrap(Δheading) / Δt`, NaN'd across gaps.

### Family 3 — Bout detection

Maps to `detect_bouts_multi_level.py`. Probably the second-priority
family after speed, because bouts depend on speed.

- **3.1 Threshold crossing.** Speed trace with a horizontal threshold
  line. Animate sweep of threshold up/down to show how bout count and
  bout width depend on it. Highlight a single crossing event with the
  interpolated sub-frame crossing time.
- **3.2 Gap merging (`min_gap_frames`).** Two near-bouts separated by a
  few sub-threshold frames; animate the merge with the gap counter.
- **3.3 Boundary modes.** Side-by-side: `threshold` boundary,
  `local_minimum` boundary (expanded outward to the nearest valley),
  `peak_event` boundary (prominence/width box from `find_peaks`). Same
  underlying bout, three different boundary choices.
- **3.4 Peak-event refinement.** A single broad bout with a deep
  mid-valley, split into two peak events; show the prominence and width
  measurements as overlaid annotations.
- **3.5 Choice of detection signal.** The same speed level fed through
  three options (`speed_filtered`, `speed_smoothed`, `speed_exponential`)
  produces different bout boundaries. Teaching point: detection signal ≠
  speed metric.

### Family 4 — Tail kinematics

Maps to `tail_kinematics_runs.py`. These are more geometric than the
speed family, so visually closer to the gaze animation.

- **4.1 Sampled tail angles.** Tail outline with sample points, body
  frame overlaid, signed angle at each sample drawn as an arc back to
  the caudal axis. Animate a beat cycle.
- **4.2 Curvature.** Inverse-radius-of-curvature heat-mapped along the
  tail outline; an osculating circle drawn at the peak-curvature sample.
- **4.3 RMS amplitude / integrated absolutes.** Show how the per-frame
  scalar summaries collapse the per-sample arrays — animate a beat cycle
  with the RMS and integrated-abs values updating live, like the
  `|m̂·f̂|` readout in Scene 4 of the gaze animation.
- **4.4 Lateral deflection.** Project the tail tip onto the body-frame
  left axis; animate the projection during a bout.

### Family 5 — Eye angle / vergence (already done)

`gaze_flip_rule.py` already covers Scenes 1–6. Possible additions:

- **5.1 Mean eye vergence.** Bianco/Engert convention shown next to the
  raw vergence; demonstrate why averaging the two eye nasal rotations
  produces the BEAST-comparable scalar.
- **5.2 Eye angular velocity / saccade detection.** If/when added to the
  pipeline.
- **5.3 Chaser-tracking demo.** Mentioned as Scene 7 idea in the README.

### Family 6 — Cross-metric bout summaries

Maps to `bout_kinematics.py`. Lowest priority — these are derived
*from* bouts and tail/heading, so they only make sense after Families
2–4 are in place.

- **6.1 Pre/post heading mean.** Show the inter-bout windows from which
  the means are computed; net Δheading as the difference.
- **6.2 Within-bout heading dominant frequency.** Brief FFT scene on a
  single bout's within-bout heading trace.
- **6.3 Path length vs. net displacement.** Two scalars on the same
  bout — the path integral vs. the chord — with a worm-trace animation.

## Sequencing

A reasonable build order, optimizing for "each scene unlocks the next":

1. **2.1, 2.2, 2.3** — speed pipeline foundation (one synthetic-trace
   investment, three scenes off it).
2. **2.4** — causal exponential, the highest-value scene because of the
   "not a speed metric" gotcha.
3. **2.5, 2.6** — acceleration and angular velocity round out kinematics.
4. **3.1, 3.2, 3.3** — bouts, leveraging the speed traces from Family 2.
5. **3.4, 3.5** — bout refinements; lower priority.
6. **4.1, 4.2** — tail geometry; standalone, can be built any time after
   the synthetic-trace decision is made (needs a tail sample sequence).
7. **6.x** — bout summaries; only after 2–4 exist.

## Settled decisions

- **Sample data:** synthetic trace, controlled noise. Real-data appendix
  scenes deferred.
- **Pipeline functions:** scenes call the real repo functions on the
  synthetic trace; math overlays are hand-drawn but verified to match.
- **File layout:** one file per family
  (`track_kinematics_pipeline.py`, `bout_detection.py`,
  `tail_kinematics.py`), top-level `master.py` chains them. Existing
  `gaze_flip_rule.py` stays as the eye family.
- **Synthetic trace:** 60 fps, 8 s, deterministic seed, contains five
  named regions (still / clean bout / noisy near-threshold / multi-peak
  bout / gap). Generator at `apps/manim/data/synthetic_trace.py`,
  returns a dataclass with `t`, `position_xy`, `heading_deg`, `fps`,
  plus ground-truth bout intervals.
- **Units:** mm/s on speed axes, with a calibration constant baked into
  the synthetic trace.
- **Time axis:** seconds, with a "frame N" tick annotation when zooming
  into single-frame events (hysteresis debounce, threshold crossings).
- **Shared base:** `apps/manim/_common/timeseries.py` with a
  `TimeseriesScene` base for kinematics/bout families. `GazeBase`
  in `gaze_flip_rule.py` stays geometric and is not shared.
- **Caching:** the `manim` conda env doesn't have `zarr`/`fisheye`, and
  `palette-py311` has a broken networkx dataclass-import (the
  `nx-loopback` backend name's hyphen breaks `networkx.utils.configs`
  at import time). So scenes can't import the pipeline at render time.
  Workaround: a separate "cache builder" step runs in the `palette` env
  (`PYTHONPATH=src`), pipes the synthetic trace through the real
  pipeline functions, and writes a `.npz` to `apps/manim/data/cache/`.
  Scenes load the cache only — they never import `fisheye`. Re-run the
  cache builder when the synthetic trace or pipeline params change.

## Open questions

- **Synthetic vs. real data — A or B above?** Need a call before
  building 2.x scenes.
- **Run the real functions, or reimplement the math?** Three options
  per scene: (i) call `compute_speed` etc. and animate the output array,
  (ii) reimplement the math inline in the scene file (clearer code, risk
  of drift), (iii) a hybrid where the scene calls the function on a
  fixture and the math overlay is hand-drawn but verified to match.
  Default proposal: **(i) for the trace itself, (iii) for the math
  overlay.** Keeps the trace honest while letting the overlay be
  pedagogically clean.
- **One file per family, or one per scene?** `gaze_flip_rule.py` is
  already at ~6 scenes in one file. For Family 2 alone we'd add 6
  scenes; the file would balloon. Proposal: split into
  `track_kinematics_pipeline.py`, `bout_detection.py`, `tail_kinematics.py`,
  with each `MasterScene` per file plus a top-level `master.py` that
  imports and chains the family masters.
- **Sound / narration?** README open ideas list mentions audio. Out of
  scope until visual scenes settle.

## Useful repo references

- `src/fisheye/analysis/track_kinematics.py:626–754` — heading,
  angular velocity, acceleration glue
- `src/fisheye/analysis/compute_speed.py:440–590` — frame distance,
  hysteresis, smoothing
- `src/fisheye/analysis/detect_bouts_multi_level.py:104–493, 287–334` —
  bout detection, exponential kernel
- `src/fisheye/analysis/tail_kinematics_runs.py:245–391, 486–628` —
  tail angles, curvature, persistence
- `src/fisheye/analysis/eye_angle_analysis.py` — v5 eye/vergence
- `apps/marimo/track_kinematics_explorer.py` — interactive layout we
  can mirror in static form
- `docs/analytics_math_primer.md`,
  `docs/track_kinematics_bout_status.md`,
  `docs/derived_analysis_run_contract.md` — algorithm intent
