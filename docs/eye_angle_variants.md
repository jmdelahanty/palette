# Eye Angle Variants
<!-- eye-angle-variants-meta
status: draft
last_verified: 2026-04-30
schema_context: analysis.eye_angle_runs v5, analysis.eye_angle_output_schema v6
-->

This document explains the eye-angle arrays currently written by
`analysis/eye_angle_runs/<run>`. It is a naming and interpretation guide for
users and downstream consumers. The storage contract remains
`src/fisheye/docs/eye_angle_conventions.md` and the run metadata stored in
`eye_angle_output_schema`.

## Short Answer

Use these first:

| Goal | Preferred fields |
| --- | --- |
| Canonical per-eye orientation | `left_major_signed_deg`, `right_major_signed_deg` |
| Bianco/Engert-style per-eye eye angle | `left_eye_angle_deg`, `right_eye_angle_deg` |
| Bianco/Engert-style signed vergence | `vergence_eye_angle_deg` |
| Draw gaze rays | `left_gaze_xy`, `right_gaze_xy` |
| Plot BEAST/Johnson-style mean eye vergence | `mean_eye_vergence_gaze_deg` |
| QA/debug gaze orientation in body frame | `left_gaze_signed_deg`, `right_gaze_signed_deg` |

For plots, prefer the `_smoothed` variant when available. For event-aligned or
bout-level analysis, use the exact eye-angle run as a source and write derived
summaries into `analysis/bout_kinematics_runs`, not back into the eye-angle run.

## Coordinate Frame

Eye-angle analysis uses the fish body frame:

```text
forward_axis = swim_bladder -> midpoint(eye_left, eye_right)
left_axis    = anatomical left from labeled eye identity
```

Body-frame signed angles use:

```text
angle = atan2(dot(vector, left_axis), dot(vector, forward_axis))
```

So:

| Value | Meaning |
| --- | --- |
| `0 deg` | aligned with fish forward |
| positive | rotated toward anatomical left |
| negative | rotated toward anatomical right |

## Canonical Major-Axis Orientation

Fields:

```text
left_major_signed_deg
right_major_signed_deg
vergence_major_signed_deg
version_major_deg
```

The ellipse major axis is the canonical stored eye-orientation axis. The writer
resolves the 180 degree line ambiguity by flipping the major axis into the fish
forward half-plane. This is stable because the major axis is normally near the
body anterior-posterior axis.

Interpretation:

| Field | Meaning |
| --- | --- |
| `left_major_signed_deg` | left eye temporo-nasal major-axis angle in body frame |
| `right_major_signed_deg` | right eye temporo-nasal major-axis angle in body frame |
| `vergence_major_signed_deg` | undirected separation between the two major-axis lines |
| `version_major_deg` | shared anatomical-left/right component, `0.5 * (left + right)` |

The major-axis fields are the best source for downstream code that wants the
most explicit geometric orientation.

## Bianco/Engert Eye-Frame Angles

Fields:

```text
left_eye_angle_deg
right_eye_angle_deg
vergence_eye_angle_deg
```

These are derived from the canonical major-axis fields but use per-eye signs:

```text
left_eye_angle_deg     = -left_major_signed_deg
right_eye_angle_deg    =  right_major_signed_deg
vergence_eye_angle_deg = left_eye_angle_deg + right_eye_angle_deg
```

Interpretation:

| Field/value | Meaning |
| --- | --- |
| positive `left_eye_angle_deg` | left eye is rotated nasally/inward |
| positive `right_eye_angle_deg` | right eye is rotated nasally/inward |
| positive `vergence_eye_angle_deg` | eyes are converged |
| negative `vergence_eye_angle_deg` | eyes are diverged |
| `vergence_eye_angle_deg == 0` | no convergence component |

This convention is useful because both eyes become positive during convergence.
For example:

| State | `left_major_signed_deg` | `right_major_signed_deg` | `left_eye_angle_deg` | `right_eye_angle_deg` | `vergence_eye_angle_deg` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rest | `0` | `0` | `0` | `0` | `0` |
| Converged 20 deg per eye | `-20` | `+20` | `+20` | `+20` | `+40` |
| Diverged 20 deg per eye | `+20` | `-20` | `-20` | `-20` | `-40` |
| Yoked left rotation | `+20` | `+20` | `-20` | `+20` | `0` |

Yoked rotation means both eyes rotate the same direction in the body frame. It
can change where the eyes point, but it is not convergence because the nasal
components cancel.

## Gaze And Minor-Axis Fields

Fields:

```text
left_gaze_signed_deg
right_gaze_signed_deg
left_gaze_deg
right_gaze_deg
left_gaze_xy
right_gaze_xy
vergence_gaze_deg
vergence_gaze_signed_deg
version_gaze_deg
```

The gaze axis is the ellipse minor axis, derived from the resolved major axis:

```text
left_gaze_signed_deg  = wrap(left_major_signed_deg + 90 deg)
right_gaze_signed_deg = wrap(right_major_signed_deg - 90 deg)
```

`left_gaze_xy` and `right_gaze_xy` are unit vectors in ROI/image coordinates for
drawing rays. They are the most direct fields for visualization.

Interpretation:

| Field | Meaning |
| --- | --- |
| `left_gaze_signed_deg`, `right_gaze_signed_deg` | directed gaze axis in body frame |
| `left_gaze_deg`, `right_gaze_deg` | unsigned magnitude of gaze angle |
| `left_gaze_xy`, `right_gaze_xy` | unit vectors for ray drawing |
| `vergence_gaze_deg` | nonnegative undirected separation between gaze-axis lines |
| `vergence_gaze_signed_deg` | compatibility name for the same nonnegative separation |
| `version_gaze_deg` | shared anatomical-left/right component of gaze axes |

Use signed gaze fields when you need geometry relative to body forward. Use
`*_gaze_xy` for overlays. Avoid interpreting `vergence_gaze_signed_deg` as a
directed convergence-vs-divergence sign; it is retained for compatibility with
older outputs.

## BEAST/Johnson-Comparable Nasal Gaze

Fields:

```text
left_nasal_gaze_deg
right_nasal_gaze_deg
mean_eye_vergence_gaze_deg
```

Current definition:

```text
left_nasal_gaze_deg          = 90 - abs(left_gaze_signed_deg)
right_nasal_gaze_deg         = 90 - abs(right_gaze_signed_deg)
mean_eye_vergence_gaze_deg   = 0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)
```

Interpretation:

| Field | Meaning |
| --- | --- |
| `left_nasal_gaze_deg` | left per-eye inward/nasal gaze estimate |
| `right_nasal_gaze_deg` | right per-eye inward/nasal gaze estimate |
| `mean_eye_vergence_gaze_deg` | mean per-eye convergence estimate |

Use `mean_eye_vergence_gaze_deg` for Johnson/BEAST-style plots where the
biological question is "how converged are the eyes on average?" This is distinct
from `vergence_gaze_deg`, which is a bilateral axis-separation total.

## Legacy Major/Minor Names

Fields retained for compatibility:

```text
left_deg
right_deg
left_signed_deg
right_signed_deg
vergence_deg
vergence_signed_deg
version_deg
left_minor_signed_deg
right_minor_signed_deg
vergence_minor_signed_deg
version_minor_deg
```

Current v5 meanings:

| Field | Current meaning |
| --- | --- |
| `left_signed_deg`, `right_signed_deg` | aliases of `left_major_signed_deg`, `right_major_signed_deg` |
| `left_deg`, `right_deg` | unsigned magnitudes of the major-axis signed angles |
| `vergence_deg`, `vergence_signed_deg` | undirected separation between legacy major-axis lines |
| `version_deg` | legacy major-axis version |
| `left_minor_signed_deg`, `right_minor_signed_deg` | aliases of `left_gaze_signed_deg`, `right_gaze_signed_deg` |
| `vergence_minor_signed_deg` | alias of gaze-axis undirected separation |
| `version_minor_deg` | alias of `version_gaze_deg` |

New readers should prefer explicit `*_major_*`, `*_eye_angle_*`, and
`*_gaze_*` names instead of these compatibility names.

## Centroid-Position Angles

Fields:

```text
left_centroid_deg
right_centroid_deg
vergence_centroid_deg
```

These use eye center positions, not eye ellipse orientation. They are computed
from vectors between a head-center estimate and each eye centroid.

Use centroid fields as pose diagnostics or covariates. Do not use them as the
primary eye-gaze or eye-vergence signal when ellipse geometry is available.

## Suffixes And Row Axes

Most scalar angle families live under both:

```text
angles/roi/<field>
angles/frame/<field>
```

`angles/roi` is row-aligned to keypoint/eye-geometry detections.
`angles/frame` is frame-aligned and may contain NaNs where there is no valid
single detection for that frame.

Common suffixes:

| Suffix | Meaning |
| --- | --- |
| none | base measurement |
| `_smoothed` | NaN-aware moving-average smoothed measurement |
| `_delta_deg` | absolute row-to-row or frame-to-frame change |
| `_delta_deg_smoothed` | delta computed from the smoothed measurement |

Some older/gaze families also expose speed and acceleration derivatives such as
`left_gaze_speed_deg_s` and `left_gaze_accel_deg_s2`. The eye-frame v6 fields
currently expose base, smoothed, delta, and frame variants.

## QA Fields

Important QA arrays:

```text
qa/roi/valid_left
qa/roi/valid_right
qa/roi/valid_frame
qa/roi/reason_codes
qa/roi/left_major_axis_marginal
qa/roi/right_major_axis_marginal
qa/roi/major_axis_marginal
```

`major_axis_marginal` is a warning, not a hard invalidation. It means the
major axis was close to the half-plane boundary used to resolve 180 degree
ambiguity:

```text
abs(dot(resolved_major_axis_xy, forward_axis_xy)) < 0.1
```

This should be rare in normal data. Consumers should show it as a QA flag and
avoid silently dropping those rows unless the analysis explicitly requires it.

## Practical Selection Guide

| If you are doing this | Use |
| --- | --- |
| Drawing the eye rays on an ROI | `left_gaze_xy`, `right_gaze_xy` |
| Plotting per-eye Bianco-style traces | `left_eye_angle_deg_smoothed`, `right_eye_angle_deg_smoothed` |
| Plotting signed convergence/divergence | `vergence_eye_angle_deg_smoothed` |
| Recreating Johnson/BEAST-style mean eye vergence | `mean_eye_vergence_gaze_deg_smoothed` |
| Debugging orientation math | `left_major_signed_deg`, `right_major_signed_deg`, `left_gaze_signed_deg`, `right_gaze_signed_deg` |
| Maintaining old readers | legacy `left_signed_deg`, `right_signed_deg`, `left_minor_signed_deg`, `right_minor_signed_deg` |

## Related Docs

- `src/fisheye/docs/eye_angle_conventions.md`
- `docs/analytics_math_primer.md`
- `docs/body_frame_contract.md`
- `docs/derived_analysis_run_contract.md`
