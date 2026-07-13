# Heart-Rate Analysis Final Decision - 2026-07-12
<!-- contract-meta
status: decision
last_updated: 2026-07-12
-->

## Purpose

This document records the final interpretation of the embedded-fish positive
control and the freely moving-fish investigation. It supersedes earlier
operational suggestions that the moving-fish window-frequency trajectory might
be reportable. The earlier status documents remain the audit trail for methods,
failed analyses, and intermediate decisions.

## Final Decision

**Embedded Trial 1 supports the feasibility of compact-mask, top-camera,
window-level cardiac-rate estimation under fixed imaging. The freely moving
recording does not support a trustworthy time-resolved heart-rate or oscillator-
rate trajectory.**

For the freely moving recording, retain only this descriptive result:

> A recurring, spatially localized `2..4 Hz` image oscillator was detected in
> an expert-supported anatomical region, with a full-recording peak near
> `3.10 Hz`. Its biological identity and frequency over time are not reliably
> established.

Do not report `3.10 Hz`, `186 cycles/min`, or the window median near `192/min`
as the fish's heart rate. Do not interpret individual `4 s` or `8 s` changes as
tachycardia, bradycardia, or other physiology.

## Comparison

| Question | Embedded Trial 1 | Freely moving recording |
|---|---|---|
| Imaging condition | Fish fixed; heart structures visible in the side view | Fish translating and rotating; anatomy inferred from tracked keypoints |
| Measurement support | Cross-fit compact top-view masks, `85` and `93` pixels | Frozen lower local-coordinate mask, `20` pixels |
| External cardiac reference | Supplied side-camera-derived rate trajectory and bradycardia bouts | None |
| Simple spatial projection | Compact-mask equal mean | Literal lower-mask Mono8 mean |
| Rate agreement | `8 s`: `r=0.921`, MAE `3.49 bpm`; `4 s`: `r=0.959`, MAE `2.84 bpm` | Cannot be measured without a reference |
| Brief response | Top peak intervals recover `76.6%` of the supplied bradycardia drop depth with close timing | Apparent excursions have no physiological reference |
| Spatial controls | Four offset boxes perform poorly: `r=-0.41..0.13`, MAE `26.6..39.2 bpm` | Compact support improves localization, but motion and adjacent anatomy remain credible |
| Temporal coverage | Short, fixed trial with held-out cross-fit blocks | Only `45,941/140,035` frames survive common support and filter-edge requirements |
| Window trajectory | Coherent with the side-derived reference | Fragmented and visibly unstable, including abrupt `8 s` changes |
| Beat validation | Not established one-to-one | Not available |
| Final interpretation | Positive feasibility result for window-level top-view rate | Localized oscillator only; no reliable rate trajectory |

## Why The Embedded Result Is Different

The embedded result has an outcome against which the top-camera measurement can
be evaluated. The supplied side-derived trajectory was not used to learn the
cross-fit masks or spatial weights. On held-out temporal halves, restricting
the top view to a compact mask improved the `8 s` rate comparison from broad-
ROI `r=0.403`, MAE `12.11 bpm` to compact-mask `r=0.921`, MAE `3.49 bpm`.
Equal-mean and PCA projections were nearly identical, showing that the main gain
came from anatomical localization rather than flexible spatial fitting.

The shorter-window sweep also behaved like a rate estimator rather than merely
becoming smoother with longer windows:

| Window | Correlation with side-derived reference | MAE |
|---:|---:|---:|
| `2 s` | `0.963` | `2.88 bpm` |
| `4 s` | `0.959` | `2.84 bpm` |
| `8 s` | `0.921` | `3.49 bpm` |

The `2 s` and `4 s` estimates followed the brief supplied rate change better
than the `8 s` estimate. This is evidence that short-window variation can be
physiological under fixed imaging when an external reference confirms it.

The embedded evidence is still limited. It covers one trial, the supplied
reference is a smooth dense rate trace rather than manual contraction times,
and the colleague's good-pixel selection, beat events, interpolation, and
smoothing code have not yet been reproduced. Therefore the embedded top-camera
peaks are not validated individual heartbeats, and generalization to other
trials remains untested.

## Why The Moving Result Is Not Reportable As Rate

The moving analysis did identify a useful measurement surface. The literal raw
mean of the frozen lower `20` pixels is essentially identical to the
per-pixel-normalized lower mean (`r=0.9997`), and both retain the aggregate
`3.10 Hz` peak. This shows that per-pixel normalization did not manufacture the
oscillator. It does not show that the oscillator is cardiac or temporally
stable.

The time-resolved output fails the practical trust test:

- only `109/351` non-overlapping `4 s` windows and `95/176` `8 s` windows are
  scorable;
- `4 s` values span `120..219 cycles/min`;
- `8 s` values span `138..222 cycles/min`;
- the final unsmoothed trajectory contains abrupt changes and long gaps;
- smoothing would conceal instability rather than supply missing information;
- no synchronized cardiac reference can distinguish physiology from tracking,
  esophageal motion, other body motion, or spectral peak switching.

The excursion diagnostic compared `15` large `4 s` deviations with `45` stable
windows. Excursions had much weaker spectral identification: median peak-to-
band contrast `3.36` versus `7.10`. Tracking proxies were also modestly worse,
including larger source-coordinate steps and gradient-displacement exposure,
higher transformation uncertainty, and lower minimum detection confidence.
However, direct motion-trace amplitude and observed/motion correlation were
nearly unchanged. Some excursions retained strong spectral peaks despite
acceptable tracking.

The evidence therefore does not support one simple correction. Short-window
frequency ambiguity is demonstrated, tracking degradation is a contributor,
and smoothly incorrect anatomical placement may evade the available metrics.
Selecting only visually stable intervals or tuning a smoother on this same
recording would introduce additional outcome-guided selection.

## Reporting Boundary

Supported language:

> In embedded Trial 1, a cross-fit compact top-camera mask and simple masked
> mean followed a supplied side-camera-derived rate reference, supporting the
> feasibility of window-level top-view cardiac-rate estimation under fixed
> imaging. In the freely moving recording, a recurring localized oscillator
> near `3.10 Hz` was observed, but its fragmented and unstable window-frequency
> trajectory did not support heart-rate estimation.

Not supported:

- the freely moving fish maintained a heart rate near `186` or `192 bpm`;
- freely moving `4 s` or `8 s` values are cardiac-rate measurements;
- apparent moving-fish excursions are bradycardia or tachycardia;
- either top-camera analysis provides validated beat-to-beat heart rate or HRV;
- the compact moving mask is an anatomically validated heart segmentation.

## Operational Disposition

1. Stop tuning the freely moving recording for rate recovery.
2. Preserve the lower-mask raw trace, masks, validity rows, controls, and plots
   as exploratory artifacts.
3. Use the final time-frequency plot as a negative quality-control result, not
   as a biological trajectory.
4. Continue embedded work only with frozen methods across additional trials.
5. Require visible contraction events or a synchronized cardiac reference
   before reopening freely moving rate or beat-level claims.

Meaningful future improvements require new information: better anatomical
tracking, higher-resolution acquisition, synchronized side imaging, blinded
contraction labels, or a new freely moving recording analyzed with frozen
parameters. Additional smoothing or mask adjustment on the current recording
does not meet that standard.

## Primary Artifacts

Embedded Trial 1:

```text
playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/
  top_camera_crossfit_mask_projection_comparison.summary.json
  top_camera_masked_equal_mean_window_sweep_step1s.summary.json
  top_camera_masked_equal_mean_stimulus_bradycardia.summary.json
```

Freely moving recording:

```text
playgrounds/heartrate_stabilization/outputs/
  reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_with_raw_lower_comparison.summary.json
  reliable_local_rostral_start_0_140035f_lower_raw_mean_4s_excursion_tracking_diagnostic.summary.json
  reliable_local_rostral_start_0_140035f_lower_raw_mean_candidate_oscillator_frequency_over_time.png
  reliable_local_rostral_start_0_140035f_lower_raw_mean_candidate_oscillator_frequency_over_time.pdf
```

Detailed history:

- [`heartrate_analysis_status_2026-07-11.md`](heartrate_analysis_status_2026-07-11.md)
- [`heartrate_top_view_mask_decision_2026-07-11.md`](heartrate_top_view_mask_decision_2026-07-11.md)
- [`heartrate_frozen_mask_validation_status_2026-07-10.md`](heartrate_frozen_mask_validation_status_2026-07-10.md)
- [`heartrate_local_rostral_roi_status_2026-07-09.md`](heartrate_local_rostral_roi_status_2026-07-09.md)

Cross-trial intern handoff:

- [`embedded_fish_heartrate_intern_handoff_2026-07-13.md`](embedded_fish_heartrate_intern_handoff_2026-07-13.md)
