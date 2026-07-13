# Top-View Heartrate Mask Decision - 2026-07-11
<!-- contract-meta
status: draft
last_updated: 2026-07-12
-->

## Purpose

This document records the current decision about using a learned compact
top-camera support instead of averaging or fitting the complete local ROI. It
covers both the stationary embedded positive control and the freely moving
fish. Detailed analysis history and withdrawn results remain in
[`heartrate_analysis_status_2026-07-11.md`](heartrate_analysis_status_2026-07-11.md).

Final interpretation: see
[`heartrate_final_decision_2026-07-12.md`](heartrate_final_decision_2026-07-12.md).
That document supersedes this note where it suggests the moving-fish window-
frequency trajectory may be reportable.

## Current Status

The investigation has progressed from broad-ROI frequency exploration to a
specific, testable top-view measurement strategy.

| Question | Current answer | Evidence level |
|---|---|---|
| Is a compact top-view mask better than the broad ROI? | Yes for the tested embedded and moving examples. | Strong descriptive evidence. |
| Is PCA required inside the embedded compact mask? | No. The equal-mask mean and PCA projections are nearly identical. | Held-out within Trial 1. |
| Can the embedded top view follow the supplied side-derived rate? | Yes. The equal-mask mean reaches ridge correlation `0.921` and MAE `3.49 bpm` at `8 s`. | Descriptive positive control. |
| Can a shorter window follow faster changes? | Yes in embedded Trial 1. A `4 s` ridge reaches `0.959` correlation and `2.84 bpm` MAE; a `2 s` ridge reaches `0.963` and `2.88 bpm`. | Held-out within Trial 1; window selection remains exploratory. |
| Can the top view capture the fearful-stimulus bradycardia? | Substantially. Raw top peak intervals recover `76.6%` of the reference drop depth with close timing. | Stimulus-locked descriptive positive result. |
| Are top-camera peaks validated individual heartbeats? | No. They have not been checked one-to-one against visible side contractions. | Not established. |
| Can the moving-fish compact mask measure an oscillator? | Plausibly. The localized candidate recurs and is more stable than the broad ROI. | Exploratory candidate; cardiac identity remains unvalidated. |
| Which moving-fish projection should be primary now? | The frozen lower 20-pixel equal mean. PCA, upper mean, and full mean remain controls. | Operational freeze based on same-cache evidence; requires new-data confirmation. |
| Has the colleague's smooth side reference been reproduced exactly? | No. Their good-pixel mask, beat events, interpolation, and smoothing procedure are missing. | Open reproducibility gap. |

The practical measurement model is now:

1. Use a frozen compact mask rather than the complete administrative ROI.
2. Use an equal-mask mean as the simplest primary embedded projection, with
   masked PCA retained as a challenger.
3. Use a short spectral ridge for robust background rate.
4. Use raw peak intervals for brief response depth and timing.
5. Smooth peak intervals only for display, never silently for detection or
   response metrics.
6. Emit no estimate through guards, invalid gaps, and failed quality windows.

This is sufficient to motivate a window-level top-camera cardiac-rate method
in the embedded positive-control dataset. It is not sufficient to promote
beat-to-beat reporting or to call the freely moving oscillator cardiac.

## Decision

**Use a frozen compact mask as the primary top-view measurement support, with
the broader ROI retained as a required control.**

For the freely moving analysis, use the **frozen lower 20-pixel equal mean** as
the primary candidate projection from this point forward. Retain masked PCA,
the upper 18-pixel mean, and the full 38-pixel mean as declared comparison and
failure-mode controls. Do not choose among these projections separately by
window.

The tested compact supports are better measurement surfaces than the broader
ROIs. They reduce dilution by irrelevant pixels, avoid allowing a dominant
unrelated component elsewhere in the box to win, and improve transfer to data
that did not fit the support. The mask is therefore useful for measuring the
candidate oscillator.

This decision does not mean that the mask is:

- a validated anatomical heart segmentation;
- proof that the moving-fish oscillator is cardiac;
- permission to relearn or adjust the support in every scored window;
- a substitute for beat-level side-camera contraction annotations.

The mask must be frozen before evaluating a target interval. A failed quality
gate must produce no estimate rather than trigger mask expansion, movement, or
frequency retuning.

## Intended Measurement Target

The plausible near-term moving-fish output is a **window-level candidate
oscillator rate**, not a beat-to-beat series. For example, a predeclared
multi-second window could summarize the dominant frequency in cycles per
minute while reporting coverage and no-estimate intervals. Such a measurement
can follow gradual rate changes even when individual contraction times cannot
be recovered reliably.

This is a difference in temporal resolution, not a distinction between a
"real" and "non-real" heart rate. After independent cardiac validation, a
window-smoothed rate would be a legitimate average heart-rate measurement,
analogous to reporting beats per minute over a time window rather than every
inter-beat interval. Before that validation, it must retain the qualifier
`candidate oscillator rate` because image motion or adjacent anatomy could
still generate the observed periodicity.

The current evidence therefore supports this staged claim:

1. **Supported now:** the frozen compact mask can capture a recurring,
   spatially localized oscillator and may provide stable window-level rate
   summaries in scorable intervals.
2. **Plausible after positive-control generalization and stronger controls:**
   the oscillator rate is a smoothed cardiac-rate measurement.
3. **Not supported without event annotations:** individual peaks are
   one-to-one heartbeats or provide beat-to-beat heart rate and variability.

### Shorter temporal-window check

Earlier exploratory pipelines used `5 s` and `10 s` windows, but the primary
full-recording frozen-mask summary used non-overlapping `60 s` bins. A new
held-out sweep tested the embedded top-camera equal-mask mean at `2, 3, 4, 6,
8, 10 s` using a common `1 s` step. Windows remained inside their finite
cross-fit confirmation blocks and never crossed the midpoint guard.

| Window | Reference correlation | Ridge MAE | Ridge samples |
|---:|---:|---:|---:|
| `2 s` | `0.963` | `2.88 bpm` | `36` |
| `3 s` | `0.919` | `4.69 bpm` | `34` |
| `4 s` | `0.959` | `2.84 bpm` | `32` |
| `6 s` | `0.938` | `3.55 bpm` | `28` |
| `8 s` | `0.921` | `3.49 bpm` | `24` |
| `10 s` | `0.884` | `4.16 bpm` | `20` |

The exact historical `8 s` result is reproduced. The embedded control shows
that `2..4 s` estimation can follow rapid reference changes rather than only a
long average. The non-monotonic `3 s` result and sensitivity to evaluation-step
alignment warn against selecting the single best-looking window post hoc.
`4 s`, containing roughly 12 cycles near `3 Hz`, is the conservative
high-resolution candidate; `2 s` is a lower-bound diagnostic rather than the
default.

The frozen 38-pixel moving-fish support was then evaluated in matched `4 s` and
`8 s` non-overlapping bins:

| Window | Scorable bins | Median candidate | IQR | In `180..216/min` | Median peak/band |
|---:|---:|---:|---:|---:|---:|
| `4 s` | `109/351` | `189/min` | `177..198/min` | `67%` | `5.06` |
| `8 s` | `95/176` | `189/min` | `177..201/min` | `71%` | `4.81` |

The `4 s` bins contain a median `3.10 s` of usable samples. They provide finer
time localization, but the strict validity rules leave only `31%` of bins
scorable and produce more opportunity for short excursions. The `8 s` bins
are scorable more often because they can accumulate the same two-second
minimum within a wider interval; that does not mean all eight seconds are
valid. Neither resolution can be ranked by accuracy in the moving fish because
there is no cardiac reference.

Operationally, retain both views: use `4 s` as an exploratory responsive trace
and a longer predeclared window as the stability/context trace. A short-window
value should be emitted only when its valid-sample, spectral-contrast, and
motion-control gates pass. Do not fill rejected bins by smoothing across them.

The complete embedded `2 s` inspection video is:

```text
embedded_fish_positive_control/inspection_videos/
  Trial1_top_camera_masked_equal_mean_2s_window_overlay_slow4x.mp4
```

It retains all `8001` source frames and plays at `50 fps`, or `4x` slow motion.
The left panel is the raw top view, the center panel is the raw `22 x 28` ROI,
and the right panel gives every active mask pixel one uniform color from the
held-out equal-mask mean. The white curve is the side-derived reference and the
orange curve is the `2 s` spectral ridge. The midpoint guard is intentionally
blank. This is a window-rate visualization and does not display detected
beats.

### Fear-evoked bradycardia endpoint

The biological target for the embedded dataset is not only whole-clip rate
agreement. It includes a momentary bradycardia during the highlighted fearful-
stimulus epoch. Trial 1's workbook supplies side-camera bradycardia bouts at
`3.01..3.53 s`, `8.39..8.96 s`, and `33.25..33.94 s`; only the last overlaps
the supplied plot's highlighted `32..36 s` stimulus interval.

A stimulus-locked comparison used a fixed `28..32 s` baseline. Results for the
principal response are:

| Measurement | Baseline | Nadir | Drop | Nadir time | Reference depth recovered |
|---|---:|---:|---:|---:|---:|
| Side-derived reference | `151.2 bpm` | `106.0 bpm` | `45.2 bpm` | `33.57 s` | reference |
| Top equal-mask peak interval | `150.0 bpm` | `115.4 bpm` | `34.6 bpm` | `33.165 s` | `76.6%` |
| Top equal-mask `2 s` ridge | `149.4 bpm` | `128.9 bpm` | `20.5 bpm` | `32.998 s` | `45.4%` |

Using the predeclared `30 bpm` drop threshold, the top peak-interval response
is `33.165..33.685 s`. It begins `85 ms` before the supplied side bout and has
interval IoU `0.56`. This is a descriptive positive result: the frozen top
masked mean captures a substantial, correctly timed stimulus-associated
slowing, and peak intervals preserve its depth better than a two-second
spectral window.

The top peak times are still automatically detected candidates, not manual
contraction annotations. The `32..36 s` stimulus interval currently comes from
the supplied Trial 1 plot and declared CLI parameters rather than a separate
machine-readable protocol field; that timing should be confirmed before a
cross-trial endpoint is frozen.

The focused 4x-slow inspection video is:

```text
embedded_fish_positive_control/inspection_videos/
  Trial1_top_camera_stimulus_bradycardia_response_overlay_slow4x.mp4
  Trial1_top_camera_stimulus_bradycardia_response_overlay_smooth100ms_slow4x.mp4
  Trial1_top_camera_full_masked_mean_bradycardia_overlay_smooth100ms_slow4x.mp4
```

It covers source time `28..40 s`. White is the side reference, orange is the
`2 s` ridge, cyan is the top masked-mean peak-interval rate, purple is the
declared stimulus epoch, red is the supplied side bradycardia bout, and blue is
the independently thresholded top response.

The `smooth100ms` variant applies a `100 ms` Gaussian kernel only to the cyan
display curve, separately within finite blocks. Detection, response depth,
timing metrics, and the numeric peak-interval status remain based on the raw
blockwise intervals. The unsmoothed video is retained as the literal interval
representation.

The `full_masked_mean` variant retains all `8001` top-camera source frames and
plays at `50 fps`, yielding `160.02 s` at 4x slow motion. It provides the two
earlier reference-rate dips and the cross-fit midpoint guard as context around
the focused fearful-stimulus response. Only the stimulus-overlapping supplied
bout is shaded as the declared response endpoint.

### Side-reference reproducibility gap

Our side-camera reproduction and the supplied reference begin with the same
general idea: measure intensity in a fixed heart ROI and convert its periodic
variation to rate. They are not the same processing pipeline.

Our current side method averages every pixel in the declared ROI or chamber
polygon, detrends the trace, applies a zero-phase `1.5..4 Hz` Butterworth
bandpass, detects peaks using fixed distance and prominence rules, and converts
successive intervals to BPM. The resulting raw interval rate is necessarily
blockwise.

The supplied `heart_rate_trace` contains a smooth value at every `100 fps`
frame. The workbook reports `1688` good pixels for Trial 1 but does not include
their coordinates. It also does not contain the original beat timestamps,
interpolation rule, smoothing kernel, filter settings, or analysis source code.
The smooth reference therefore appears to include post-event interpolation and
temporal smoothing, but the exact operations are unknown and must not be
reverse-engineered by choosing whichever smoothing best matches the outcome.

The `100 ms` Gaussian smoothing used in our newest cyan video curve is
explicitly visualization-only. It operates separately within finite blocks.
The reported `34.6 bpm` top response depth, `33.165..33.685 s` response span,
and all other metrics continue to use raw peak intervals.

The requested colleague notebook should resolve:

- how the good pixels were selected and whether their coordinates were saved;
- whether the source signal was a mean, median, weighted projection, or another
  statistic;
- detrending, nuisance correction, and temporal filter parameters;
- beat polarity, peak criteria, refractory period, and rejected-event rules;
- how interval rates were assigned or interpolated onto individual frames;
- the smoothing method, kernel width, and boundary handling;
- whether `Brady_bouts` were detected from the smoothed trace or from events;
- the authoritative stimulus onset and offset for each trial;
- the exact top/side timebase alignment and any frame offset.

Once that notebook is available, first reproduce the supplied side trace and
Trial 1 bradycardia bouts exactly. Then freeze the recovered event and smoothing
contract, compare top-mask events one-to-one with side events, and evaluate the
same declared response endpoints across Trials 2--48. Do not tune the top mask,
filter, or response threshold separately for each trial.

## Why The Mask Helps

A broad ROI is an extraction envelope, not a promise that every pixel measures
the same structure. It may contain heart tissue, adjacent anatomy, edges,
background, stabilization interpolation, and pixels whose intensity changes
have different signs or phases. Averaging all of those pixels can cancel a
localized signal. Fitting PCA over all of them can instead select the strongest
unrelated oscillation in the box.

A compact support changes the question from:

> What is the strongest or average temporal behavior anywhere in this box?

to:

> What temporal behavior is present on this predeclared local support?

That restriction is beneficial only when the support was discovered without
using the values on which it will be evaluated, or when its same-data status is
reported explicitly.

## Embedded Top-Camera Evidence

The embedded Trial 1 top-camera ROI is `22 x 28`, or `616` pixels. The two
cross-fit compact masks contain `85` and `93` pixels. Each mask was learned on
one temporal half and applied to the opposite half after a midpoint guard.
The supplied side-camera-derived rate reference was not used to fit the masks
or their spatial weights.

### Broad ROI Versus Compact Mask

| Measurement | Ridge correlation | Ridge MAE | Event-rate MAE | Interpretation |
|---|---:|---:|---:|---|
| Broad top-camera ROI mean | `0.403` | `12.11 bpm` | `28.76 bpm` | Too much irrelevant or cancelling content. |
| Compact-mask discovery loading | `0.921` | `3.49 bpm` | `4.61 bpm` | Strong held-out rate tracking. |
| Compact-mask PCA refit | `0.921` | `3.49 bpm` | `4.49 bpm` | No material ridge advantage over the original loading. |
| Compact-mask equal mean | `0.921` | `3.49 bpm` | `4.84 bpm` | Nearly the same result with simpler spatial weighting. |

The three compact-mask waveforms correlate at `0.9968..0.9998`. Therefore,
the important gain in this trial comes from **where the signal is measured**,
not from delicate PCA weights inside the compact support. An equal mean of
the discovery-normalized, bandpassed mask pixels is a reasonable primary
measurement for this stationary positive control. PCA should remain a
predeclared challenger rather than being preferred solely for complexity.

Four independently fitted, same-size offset boxes have ridge correlations of
`-0.41..0.13` and ridge errors of `26.6..39.2 bpm`. This supports spatial
specificity, although it is not a calibrated search-wide null test.

### What This Establishes

The compact top-view support contains a signal that follows the supplied
side-camera-derived rate trajectory well in embedded Trial 1. It also shows
that a simple masked mean can preserve nearly all of the useful top-view rate
information once the measurement location is restricted appropriately.

It does not establish one-to-one top-camera beats. The supplied reference is a
dense rate trajectory rather than visible contraction timestamps, and Trial 1
is one fish and one recording.

## Freely Moving Evidence

The moving-fish administrative ROI contains `226` anatomically eligible
pixels. The main frozen whole-heart candidate contains `38` pixels, and a
recurrent-core intersection contains `8` pixels.

On the five-minute frame-0 cache:

| Support | Aggregate peak | Peak/band median | PCA variance | Independence note |
|---|---:|---:|---:|---|
| Broad 226-pixel ROI | `1.75 Hz` | diffuse | not promoted | Selects a different low-frequency component. |
| Frozen 38-pixel support | `3.10 Hz` | `1.94` | `0.375` | Learned from later chase frames; temporally frozen here. |
| Recurrent 8-pixel core | `3.10 Hz` | `2.23` | `0.770` | Partly learned from this five-minute cache; descriptive here. |

Across the complete recording, the 38-pixel and 8-pixel supports agree within
`0.10 Hz` in all `22` scorable one-minute windows. Both place `19/22` windows
in `3.0..3.6 Hz` and select an aggregate `3.10 Hz` candidate. The broad support
is less stable and can select unrelated lower-frequency behavior.

This is strong evidence that the compact support is a better surface for
tracking the **localized candidate oscillator**. It is not evidence that the
candidate is definitively cardiac. The motion-prediction control also has an
aggregate `3.10 Hz` peak, despite differing in waveform and window trajectory,
and there is no synchronized beat reference for the moving fish.

## Mean Or PCA Inside The Mask

The embedded result supports a compact-mask mean because the mean and PCA
traces are almost identical there. The moving fish is less settled: some
mask pixels have appeared to change with opposing polarity or phase, so an
equal mean could cancel real spatial structure.

For the next frozen evaluation, retain both methods:

1. **Masked equal mean:** average discovery-normalized, bandpassed pixel
   residuals with equal positive weights. This is the simplest interpretable
   photometry measurement.
2. **Masked PCA projection:** fit weights only on declared discovery data and
   apply them unchanged to confirmation data. This can preserve opposed pixel
   responses but has more fitting freedom.

The method, polarity, normalization, band, quality gates, and mask must be
frozen before confirmation. Prefer the mean if it transfers as well as PCA.
Prefer neither if both fail the declared gates. Do not select whichever method
looks cleaner separately in each target window.

### Moving-fish frozen upper/lower mean comparison

The full-recording moving cache was reanalyzed without changing the frozen
38-pixel support or its previously declared geometric split. The split contains
18 upper pixels and 20 lower pixels. All projections use the same 147
common-valid segments, robust per-pixel scaling, `2..4 Hz` bandpass, bounded
gap handling, and `0.75 s` segment-edge rejection.

| Projection | Aggregate peak | Peak/band | Interval CV | `4 s` median | `4 s` in `180..216/min` | `8 s` median | `8 s` in band |
|---|---:|---:|---:|---:|---:|---:|---:|
| Masked PCA | `3.10 Hz` | `2.20` | `0.152` | `189/min` | `67%` | `189/min` | `71%` |
| Full 38-pixel mean | `3.10 Hz` | `1.68` | `0.172` | `177/min` | `43%` | `183/min` | `48%` |
| Upper 18-pixel mean | `3.15 Hz` | `2.05` | `0.152` | `189/min` | `66%` | `189/min` | `64%` |
| Lower 20-pixel mean | `3.10 Hz` | `2.56` | `0.136` | `192/min` | `73%` | `192/min` | `85%` |

The lower mean is almost the same waveform as PCA (`r=0.981`). Their window
frequencies have zero median difference and agree within `0.10 Hz` in `90%` of
scorable `4 s` windows and `81%` of `8 s` windows. The lower observed mean has
an aggregate `3.10 Hz` peak while its matched motion-prediction mean peaks at
`2.50 Hz`; their absolute waveform correlation is `0.034`.

The upper and lower means correlate only `0.225`. Their full-recording circular
mean phase is approximately `101 degrees` lower-minus-upper with phase-locking
value `0.312`. Thus the two regions are neither consistently in phase nor
consistently opposite. Equal averaging over all 38 pixels partially dilutes or
cancels their structured differences, explaining why the whole-mask mean is
less stable than either PCA or the lower mean.

This is promising for simplification: a frozen lower-half equal mean may
capture most of the moving PCA oscillator without fitted spatial weights. It
is not an independent validation. The split was formulated after earlier
visual inspection, normalization and event polarity use the same full cache,
and no moving-fish cardiac reference exists. Treat the lower mean as a frozen
primary candidate for new-recording transfer, not as a retrospectively
validated winner.

### Frozen projection decision

The lower 20-pixel equal mean is now the primary moving-fish candidate trace
for subsequent visualization, rate-window, quality-gate, motion-control, and
new-recording transfer work. This decision is based on four observations:

1. It preserves the recurring `3.10 Hz` aggregate candidate.
2. It has higher peak-to-band contrast and lower interval CV than masked PCA.
3. It matches PCA's window frequency closely without fitted spatial weights.
4. Its matched motion-prediction mean selects `2.50 Hz` rather than the
   observed `3.10 Hz` aggregate peak.

The upper mean remains biologically and diagnostically relevant. It expresses
a nearby `3.15 Hz` oscillator, but it is not a stable opposite-phase copy of
the lower mean. The circular mean phase offset is about `101 degrees`, not
`180 degrees`, and phase locking is weak (`0.312`). Its phase, amplitude, and
selected frequency vary enough that combining both halves weakens the simple
mean.

Operationally, “primary” means:

- freeze the existing lower mask coordinates and equal-weight projection;
- calculate it on every interval that passes the common validity contract;
- emit no estimate when its predeclared gates fail;
- report PCA, upper, full-mask, and motion results beside it as controls;
- never call its cycles heartbeats or heart rate without cardiac validation;
- test the unchanged lower projection on a new recording before promotion.

It does not mean that the upper region is noncardiac, that the lower region is
an anatomically confirmed chamber, or that the lower projection may be tuned
again on this recording.

Artifacts:

```text
reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_comparison.summary.json
reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_comparison.windows.csv
reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_comparison.arrays.npz
reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_comparison.png
```

### Full-recording lower-mask inspection overlay

The selected lower projection now has a dedicated full-recording overlay. It
loads the saved `trace_lower_equal_mean` rather than refitting PCA or deriving
a new trace during rendering. Every one of the 20 lower-mask pixels receives
the same instantaneous color. Therefore the display answers whether the
regional mean oscillates while remaining aligned to the intended anatomy; it
does not depict pixel-level phase, spatial propagation, or a traveling wave.

The timeline shows the lower-mean `4 s` candidate in white, the `8 s`
candidate in green, and raw within-segment event intervals in orange. Invalid
segments and the first/last `0.75 s` of valid segments have no oscillator
color or waveform. The raw interval series is explicitly not validated beat
timing.

```text
reliable_local_rostral_start_0_140035f_lower20_equal_mean_4s_8s_full_overlay_stride3_25fps.mp4
reliable_local_rostral_start_0_140035f_lower20_equal_mean_4s_8s_full_overlay_stride3_25fps.json
```

The video contains `46,679` H.264 frames at `1152x648`, `25 fps`, and
`1867.16 s` playback duration. Source frames are sampled at stride 3, so
playback is `0.75x` real time. A transfer copy is in the shared recording's
`inspection_videos/` directory.

### Literal lower-mask Mono8 mean

The lower 20-pixel support was also evaluated using the simplest photometry
projection: average the original Mono8 samples within each frame first, then
detrend and `2..4 Hz` bandpass that single trace separately inside each common-
valid segment. This differs from `lower_equal_mean`, which robustly scales and
bandpasses every pixel before averaging.

| Lower projection | Waveform correlation | Aggregate peak | Peak/band | Interval CV | `4 s` in `180..216/min` | `8 s` in band |
|---|---:|---:|---:|---:|---:|---:|
| Per-pixel normalized mean | reference | `3.10 Hz` | `2.560` | `0.136` | `73.4%` | `85.3%` |
| Literal raw Mono8 mean | `0.9997` | `3.10 Hz` | `2.578` | `0.133` | `74.3%` | `85.3%` |

The two methods selected exactly the same frequency in all `109` paired
scorable `4 s` windows and all `95` paired scorable `8 s` windows. The literal
raw mean also retained the low absolute matched-motion waveform correlation
(`0.034`) and selected `2.50 Hz` for its motion prediction rather than the
observed `3.10 Hz` peak.

Thus per-pixel normalization is not responsible for the recurring lower-mask
oscillator in this recording. The literal raw mean is the preferred simplest
projection for new-recording transfer, with the normalized lower mean retained
as a processing-sensitivity control. This remains a same-cache comparison and
does not establish cardiac identity.

Artifacts:

```text
reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_with_raw_lower_comparison.summary.json
reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_with_raw_lower_comparison.windows.csv
reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_with_raw_lower_comparison.arrays.npz
reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_with_raw_lower_comparison.png
```

### Four-second excursion diagnostic

The `109` scorable literal-mean `4 s` windows were divided descriptively around
their `192/min` median: `45` stable windows within `6/min`, `15` excursions at
least `24/min` away, and `49` intermediate windows. This grouping diagnoses the
already observed output; it is not an independently calibrated acceptance
rule.

The clearest difference was spectral ambiguity. Excursion windows had median
peak-to-band contrast `3.36`, versus `7.10` in stable windows (Cliff's delta
`-0.67`). Tracking was modestly worse in excursions:

| Diagnostic | Stable median | Excursion median | Cliff's delta |
|---|---:|---:|---:|
| Source-coordinate step p95 | `3.20 px` | `3.52 px` | `0.27` |
| Gradient-displacement p95 | `13.24` | `17.37` | `0.30` |
| Transform uncertainty p95 | `0.789` | `0.822` | `0.28` |
| Minimum detection confidence | `0.787` | `0.727` | `-0.29` |
| Local translation p95 | `19.85 px` | `21.38 px` | `0.27` |

However, common-valid sample count differed little (`315` stable versus `294`
excursion), motion-trace RMS was essentially unchanged (`0.657` versus
`0.649`), and absolute observed/motion correlation was similar (`0.219` versus
`0.249`; Cliff's delta `0.05`). Some excursions also retained strong spectral
contrast and acceptable tracking, including `32..36 s` (`150/min`, contrast
`11.53`) and `804..808 s` (`219/min`, contrast `13.53`).

Therefore degraded tracking is associated with the unstable `4 s` estimates
and is likely a contributor, but it does not primarily or completely explain
them. Short-window peak ambiguity is the strongest demonstrated explanation.
Smooth but anatomically incorrect mask placement remains invisible to these
geometric diagnostics, and genuine or noncardiac biological rate changes also
remain possible.

```text
reliable_local_rostral_start_0_140035f_lower_raw_mean_4s_excursion_tracking_diagnostic.summary.json
reliable_local_rostral_start_0_140035f_lower_raw_mean_4s_excursion_tracking_diagnostic.windows.csv
reliable_local_rostral_start_0_140035f_lower_raw_mean_4s_excursion_tracking_diagnostic.png
```

The final time-versus-frequency summary uses the literal lower-mask mean. It
shows unsmoothed `8 s` estimates as the primary series, faint `4 s` estimates
as context, and the `3.10 Hz` full-recording peak. Lines are broken across
unscorable gaps rather than interpolated. The right axis reports candidate
cycles per minute without relabeling them as heart rate.

```text
reliable_local_rostral_start_0_140035f_lower_raw_mean_candidate_oscillator_frequency_over_time.png
reliable_local_rostral_start_0_140035f_lower_raw_mean_candidate_oscillator_frequency_over_time.pdf
```

## Operational Contract

Use the broad ROI for extraction, bookkeeping, and controls. Use the compact
mask for the primary candidate trace only under these rules. For freely moving
data, `compact mask` now means the frozen lower 20-pixel support unless an
explicit comparison output is being produced:

1. Map the frozen mask through the declared stabilized local-coordinate
   contract; do not redraw it on scored frames.
2. Require the declared pixel-validity, tracking, and filter-edge coverage.
3. Keep invalid gaps segmented; never compress time or filter across long
   gaps.
4. Emit no estimate when support or quality gates fail.
5. Report broad-ROI, boundary, eye, esophagus, and motion measurements as
   controls rather than silently discarding them.
6. Report moving-fish results as candidate cycles per minute until cardiac
   attribution is independently established.
7. Report beat-to-beat values only after one-to-one comparison with visible
   contraction annotations.

## Recommended Next Test

Freeze the 38-pixel moving-fish support as the primary spatial hypothesis and
compare its equal-mean and discovery-fitted PCA traces on untouched intervals.
Score both against the same predeclared coverage and motion-control gates.
Separately, apply the embedded cross-fit procedure to additional positive-
control trials so mask discovery, equal-mean transfer, rate error, and
no-estimate coverage are evaluated across recordings rather than only Trial 1.

For beat-level validation, annotate visible side-camera contractions and
unscorable spans without viewing the algorithm trace. Then measure one-to-one
precision, recall, missed and doubled events, and timing error for the frozen
top-camera detector.

## Reporting Language

Supported now:

> A frozen compact top-camera support is a substantially better measurement
> surface than the broader ROI in the embedded positive control and is more
> stable for the recurring moving-fish oscillator. In embedded Trial 1, a
> simple masked mean follows the side-camera-derived rate reference nearly as
> well as masked PCA. In the freely moving example, the frozen compact support
> provides a plausible route to window-level candidate oscillator rates in
> scorable intervals.

Not supported now:

> The learned moving-fish mask is a validated heart segmentation, or its
> oscillation provides validated beat-to-beat heart rate.
