# Raw vs Smoothed Metrics in Behavioral Geometry Pipelines

## Overview

When computing behavioral geometry metrics such as **eye angle relative to fish heading**, it is generally best practice to separate:

1. **Raw geometric measurements**
2. **Post-processing operations (smoothing, filtering, interpolation)**
3. **Derived biological metrics**

This document explains why this separation is valuable and outlines a recommended structure for a reproducible analysis pipeline.

---

# Core Principle

> **Store raw measurements first, and treat smoothing as a derived analysis step.**

In practice this means:

```
raw geometry → quality checks → smoothing → biological metrics
```

Raw measurements should be preserved whenever feasible so that downstream analyses remain transparent and reproducible.

---

# Why This Design Is Good Practice

Separating measurement from smoothing provides several advantages.

## 1. Preserves Direct Observations

Raw measurements represent the **closest possible value to what was directly observed in the frame**.

Smoothing introduces assumptions about temporal continuity that may obscure frame-level behavior.

Keeping raw values ensures that:

* original geometry can always be inspected
* smoothing artifacts can be detected
* alternative filtering methods can be tested later

---

## 2. Improves Debuggability

When unexpected results appear, it is much easier to diagnose problems if raw data still exists.

Without raw values it can be difficult to determine whether anomalies originate from:

* segmentation errors
* ellipse fitting instability
* heading computation
* smoothing parameters
* temporal filtering assumptions

Maintaining raw measurements allows these sources to be isolated.

---

## 3. Enables Method Comparisons

Different smoothing strategies can produce different biological interpretations.

Examples include:

* moving average
* Gaussian smoothing
* Kalman filtering
* spline smoothing
* circular smoothing for angles

By storing raw measurements, multiple approaches can be compared without rerunning earlier pipeline stages.

---

# Why This Matters Especially for Angles

Angles behave differently from linear variables.

For example:

```
frame 1: 179°
frame 2: -179°
```

A naive average gives:

```
0°
```

which is incorrect.

Correct smoothing requires **circular statistics**, typically implemented by smoothing:

```
cos(θ)
sin(θ)
```

and reconstructing the angle with:

```
atan2(mean_sin, mean_cos)
```

For this reason, smoothing angles should always be treated carefully and ideally performed in a separate stage.

---

# Common Pipeline Structure

A robust behavioral geometry pipeline often contains several layers.

## Stage 1 — Raw Geometry

Direct measurements extracted from each frame.

Examples:

* `heading_raw`
* `eye_orientation_raw`
* `relative_eye_angle_raw`
* ellipse parameters `(x_c, y_c, a, b, θ)`
* confidence scores
* contour statistics

These values should reflect the **direct result of the measurement algorithm**.

---

## Stage 2 — Quality Control

Frame-level diagnostic information and validity checks.

Examples:

* ellipse fit success/failure
* segmentation confidence
* contour size thresholds
* left/right eye assignment validity
* occlusion flags
* missing data markers

This stage identifies frames that should be ignored or handled differently during smoothing.

---

## Stage 3 — Smoothing / Refinement

Temporal filters are applied to produce stable trajectories.

Examples:

* `heading_smoothed`
* `eye_orientation_smoothed`
* `relative_eye_angle_smoothed`

Typical smoothing methods:

* exponential filters
* Gaussian filters
* Kalman filters
* circular smoothing

Metadata should record:

* smoothing method
* window size or parameters
* whether smoothing was **causal**, **centered**, or **acausal**
* how missing frames were handled
* whether angular wrapping was considered

---

## Stage 4 — Biological Metrics

Higher-level behavioral quantities derived from smoothed measurements.

Examples:

* vergence angle
* conjugate eye movement
* saccade detection
* stimulus-aligned responses
* bout-triggered averages

At this stage the data represents **interpreted behavioral metrics** rather than raw geometry.

---

# When to Smooth: Two Common Approaches

There are two main strategies when computing derived angles.

## Option A — Smooth Inputs First

```
smooth heading
smooth eye orientation
compute relative eye angle
```

Advantages:

* reduces noise in both signals before comparison

---

## Option B — Smooth Derived Angle

```
compute relative eye angle
smooth relative eye angle
```

Advantages:

* preserves the raw relationship between signals

---

## Option C — Store Both (Recommended)

For maximum reproducibility:

* store smoothed inputs
* store raw derived angles
* optionally store smoothed derived angles

This provides flexibility for later analysis.

---

# Metadata and Provenance

Recording full provenance is essential for reproducibility.

Recommended metadata fields include:

* ellipse fitting algorithm
* contour extraction method
* heading computation method
* angle conventions (`[-π, π)` vs `[0, 2π)`)
* smoothing algorithm
* smoothing parameters
* frame exclusion rules
* circular statistics method

This allows results to be reproduced or reinterpreted later.

---

# Recommended Data Layout Example

Example dataset structure:

```
geometry/
    heading_raw
    heading_smoothed
    eye_orientation_left_raw
    eye_orientation_right_raw
    eye_orientation_left_smoothed
    eye_orientation_right_smoothed
    relative_eye_angle_left_raw
    relative_eye_angle_right_raw
    relative_eye_angle_left_smoothed
    relative_eye_angle_right_smoothed

qc/
    ellipse_fit_success
    contour_size
    segmentation_confidence
    frame_valid

analysis/
    vergence
    saccade_events
```

Each group should include metadata describing algorithms and parameters.

---

# Key Takeaway

A robust behavioral geometry pipeline should follow this principle:

```
measure → store raw → quality control → smoothing → biological analysis
```

Treat **smoothed values as derived products**, not replacements for the original measurements.

This structure:

* preserves reproducibility
* improves debugging
* allows future reanalysis
* avoids irreversible assumptions early in the pipeline

