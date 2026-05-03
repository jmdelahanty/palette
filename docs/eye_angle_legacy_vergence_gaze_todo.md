# Eye-Angle Legacy Gaze Vergence Deprecation TODO
<!-- todo-meta
status: draft
created: 2026-05-01
-->

Purpose: remove or clearly quarantine the legacy `vergence_gaze_deg` /
`vergence_minor_signed_deg` output family from biological eye-convergence
workflows.

## Problem

`vergence_gaze_deg` currently stores a nonnegative geometric separation between
the two derived gaze-axis lines:

```text
undirected_axis_separation(left_gaze_signed_deg, right_gaze_signed_deg)
```

This is not a signed biological vergence measure. It answers:

```text
How far apart are the two gaze-axis lines as directionless geometry?
```

It does not answer:

```text
Are the eyes converging or diverging?
```

Because ellipse axes are directionless, this value intentionally collapses
orientation by taking the smaller separation between two axis lines. As a
result, it can lose the sign/direction needed for biological interpretation.

Example:

```text
left_gaze_signed_deg  = +70
right_gaze_signed_deg = -70
raw difference        = 140
axis separation       = min(140, 180 - 140) = 40
```

The stored `40 deg` is a geometric line separation. It is not directly a
converged-vs-diverged signal. Similar axis separations can arise from
biologically different configurations, so this field is unsafe as the primary
output for convergence analyses.

The legacy alias `vergence_minor_signed_deg` is especially misleading because
the suffix says `signed`, but the binocular aggregate is not signed.

## Preferred Outputs

Use `vergence_eye_angle_deg` for signed Bianco/Engert-style convergence:

```text
left_eye_angle_deg  = -left_major_signed_deg
right_eye_angle_deg =  right_major_signed_deg
vergence_eye_angle_deg = left_eye_angle_deg + right_eye_angle_deg
```

Interpretation:

```text
positive  = convergence
negative  = divergence
near zero = no vergence component, including rest or yoked shifts
```

Use `mean_eye_vergence_gaze_deg` for BEAST/Johnson-style mean per-eye
convergence:

```text
left_nasal_gaze_deg  = 90 - abs(left_gaze_signed_deg)
right_nasal_gaze_deg = 90 - abs(right_gaze_signed_deg)
mean_eye_vergence_gaze_deg =
    0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)
```

Interpretation:

```text
larger = more inward/nasal eye rotation on average
```

## Desired End State

- New biological analyses do not use `vergence_gaze_deg` or
  `vergence_minor_signed_deg`.
- Visualizers do not present `vergence_gaze_deg` as the default convergence
  trace.
- Documentation labels this field as legacy geometric axis separation, not
  biological vergence.
- Exported analytics prefer:
  - `vergence_eye_angle_deg` for signed convergence/divergence
  - `mean_eye_vergence_gaze_deg` for BEAST/Johnson-style convergence summaries
- Legacy fields remain readable only for backward compatibility until an
  explicit schema-breaking cleanup is acceptable.

## Implementation Tasks

- [ ] Audit all readers, plots, dashboards, exports, and tests for
      `vergence_gaze_deg` and `vergence_minor_signed_deg`.
- [ ] Change default eye-angle visualizations to prefer
      `vergence_eye_angle_deg` or `mean_eye_vergence_gaze_deg`, depending on
      the display purpose.
- [ ] Update `eye_angle_variant_schema` so `vergence_gaze_deg` is marked as
      legacy geometric axis separation.
- [ ] Update docs that call `vergence_gaze_deg` "vergence" without the
      geometric/legacy qualifier.
- [ ] Add tests that assert biological convergence code does not select
      `vergence_gaze_deg` by default.
- [ ] Decide whether the next schema version should stop writing
      `vergence_gaze_deg` entirely, or keep it under a clearer name such as
      `gaze_axis_separation_deg`.
- [ ] If retained, rename or alias the field with explicit semantics:
      `gaze_axis_separation_deg`.
- [ ] Deprecate `vergence_minor_signed_deg`; if retained for compatibility,
      document it as an alias of legacy geometric axis separation and not as a
      signed value.

## Migration Notes

Short-term compatibility:

- Continue reading existing `vergence_gaze_deg` arrays from old runs.
- Do not delete old arrays in-place.
- Prefer adding clearer new fields in a new run/schema version.

Recommended replacement mapping:

```text
Old use case: "signed convergence/divergence"
Replacement: vergence_eye_angle_deg

Old use case: "BEAST/Johnson mean convergence"
Replacement: mean_eye_vergence_gaze_deg

Old use case: "legacy total axis geometry"
Replacement name if retained: gaze_axis_separation_deg
```

## Rationale

The current field name makes a geometry quantity look like a biological
vergence quantity. That is risky because downstream analyses may threshold or
interpret it as convergence. Removing it from default workflows reduces the
chance of mixing up:

- axis-line separation,
- signed convergence/divergence,
- mean per-eye nasal convergence,
- yoked/conjugate eye movement.

The code can still support legacy archives, but new analyses should use the
fields whose signs and biological meanings match the intended question.
