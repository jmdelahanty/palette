# Stabilizing Ellipse Fits from Segmentation Masks

When fitting ellipses to segmentation masks (such as fish eyes from a U-Net), a single-frame ellipse fit can be noisy or occasionally unstable. Many tracking systems improve robustness by incorporating **temporal information**, **probability weighting**, or **alternative geometric cues**.

This document summarizes techniques commonly used in vision pipelines and explains why switching to `cv2.fitEllipse` significantly improved fitting success in our pipeline.

---

# 1. Temporal Smoothing of Ellipse Parameters

Instead of treating each frame independently, track ellipse parameters over time.

An ellipse can be parameterized as:

```
(x_c, y_c, a, b, θ)
```

Where:

* `x_c, y_c` = ellipse center
* `a, b` = major and minor axes
* `θ` = orientation angle

If these parameters are estimated every frame, they can be smoothed using a temporal filter.

Example pipeline:

```
raw ellipse fits (per frame)
        ↓
temporal filter
        ↓
stable ellipse trajectory
```

A simple exponential filter works surprisingly well:

```
p_t = α * p̂_t + (1 - α) * p_{t-1}
```

Where:

* `p̂_t` = newly estimated parameter
* `p_{t-1}` = previous filtered parameter
* `α` = smoothing factor (e.g. 0.2–0.5)

### Why this helps

Segmentation masks vary slightly frame-to-frame due to prediction noise.
Ellipse fitting amplifies these small boundary changes.

Temporal filtering suppresses high-frequency jitter.

Temporal smoothing is extremely common in:

* eye tracking systems
* animal tracking pipelines
* object tracking systems

---

# 2. Weighted Ellipse Fitting Using Probability Maps

If the segmentation model outputs **probability masks**, that information can be used directly.

Instead of fitting the ellipse to a binary contour, fit it to points weighted by probability.

Conceptually:

```
minimize  Σ w_i * d(point_i, ellipse)^2
```

Where:

```
w_i = P(x_i, y_i)
```

Pixels where the model is confident contribute more to the fit.

### Benefits

* low-confidence boundary pixels matter less
* interior pixels reinforce the shape
* segmentation noise has less influence

This idea appears in **probabilistic shape fitting** and **soft segmentation geometry**.

---

# 3. Using the Distance Transform

Another useful geometric cue is the **distance transform** of the mask.

Steps:

1. Compute the distance transform of the binary mask.
2. The maximum of the distance field approximates the object center.
3. Use the outer contour to estimate axes.

Pipeline example:

```
binary mask
      ↓
distance transform
      ↓
center estimate from ridge
      ↓
ellipse axes from contour
```

This reduces bias from jagged boundaries.

---

# 4. RANSAC Ellipse Fitting

If segmentation contours contain outliers (e.g., reflections or partial occlusions), **RANSAC** can improve robustness.

Conceptually:

```
repeat:
    sample subset of contour points
    fit ellipse
    measure inliers
choose ellipse with most support
```

Advantages:

* robust to outliers
* ignores spurious contour points

Tradeoff:

* slower than direct least-squares methods.

---

# 5. Combining Segmentation with Image Gradients

Some pipelines refine segmentation-based ellipses using image gradients.

Typical pipeline:

```
segmentation mask
      ↓
initial ellipse
      ↓
refine using intensity gradients
```

This works well when object boundaries have strong image edges.

Eyes often satisfy this condition.

---

# 6. Why Temporal Filtering Is Often the Biggest Win

In many pipelines, the most impactful improvement comes from recognizing:

```
ellipse geometry should not change drastically between frames
```

Instead of fitting each frame independently:

```
frame → fit ellipse
```

tracking systems often do:

```
previous ellipse
      ↓
predict next state
      ↓
refine with current frame
```

This dramatically reduces jitter.

---

# 7. Why `cv2.fitEllipse` Worked Better Than `skimage EllipseModel`

During refinement, the pipeline originally used:

```
skimage.measure.find_contours
→ skimage.measure.EllipseModel
```

This produced a very low success rate:

```
1908 / 23287 fits
```

After switching to:

```
cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)
→ cv2.fitEllipse
```

the success rate became:

```
23216 / 23287 fits
```

This large improvement suggests the **ellipse fitting algorithm**, rather than mask quality, was the main bottleneck.

---

## Differences Between the Two Ellipse Fitters

### `skimage.measure.EllipseModel`

EllipseModel uses a **total least squares algebraic fit**.

High-level behavior:

```
fit general conic
→ check if result is a valid ellipse
```

If the fitted conic is:

* degenerate
* numerically unstable
* close to a parabola or hyperbola

the estimation fails.

Typical failure causes include:

* flat contour segments
* small contours
* noisy masks
* near-collinear point sets

In these cases, `estimate()` returns `False`.

---

### `cv2.fitEllipse`

OpenCV uses a **direct least-squares ellipse fitting method** derived from:

Fitzgibbon, Pilu, Fisher (1999)
*Direct Least Squares Fitting of Ellipses*

Instead of fitting a general conic and checking validity, it solves a **constrained optimization problem that forces the result to be an ellipse**.

Conceptually:

```
fit ellipse directly
subject to ellipse constraints
```

As long as there are at least five contour points, the algorithm almost always returns an ellipse.

---

## Why This Matters for Segmentation Masks

Binary mask contours often contain:

* flat segments
* staircase edges
* uneven point spacing

These patterns can produce degenerate conic fits for strict solvers.

OpenCV's method is more tolerant of these geometries.

Even when contours look visually similar, the numerical structure of the point set can cause a solver like `EllipseModel` to reject a fit.

---

# Key Takeaway

For segmentation-based ellipse fitting pipelines:

```
segmentation → contour extraction → ellipse fitting → temporal smoothing
```

is typically the most robust architecture.

In this pipeline, the most important improvement came from switching to:

```
cv2.fitEllipse
```

which is significantly more tolerant of rasterized segmentation contours.

