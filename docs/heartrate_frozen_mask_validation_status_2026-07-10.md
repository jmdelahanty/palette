# Heartrate Frozen-Mask Validation Status - 2026-07-10
<!-- contract-meta
status: draft
last_updated: 2026-07-11
-->

Purpose: record the current evidence for the local-rostral frozen whole-mask
heart candidate after temporal cross-fitting, a second 30-second interval, a
five-minute frame-0 cache, and full-recording photometry diagnostics.

The combined moving-fish and embedded positive-control status, including an
explicit failed/inconclusive/withdrawn attempt ledger, is in
[`heartrate_analysis_status_2026-07-11.md`](heartrate_analysis_status_2026-07-11.md).

The detailed implementation and earlier investigation history remain in
[`heartrate_local_rostral_roi_status_2026-07-09.md`](heartrate_local_rostral_roi_status_2026-07-09.md).

## Bottom Line

The current analysis supports **a repeatable localized periodic image pattern
near `3.2..3.35 Hz` within the frozen local-rostral region**. A compact recurrent
core appears more coherent than the full 38-pixel mask, but its surrogate
calibration must be rerun after the missingness correction below.

It does **not establish a validated heart rate or reliable beat times** because:

- no visible biological contraction landmark has been defined;
- no blinded beat timestamps have been compared with inferred events;
- no explicit esophagus mask/control has been measured;
- the regional phase null is conditional on the selected frequency and filter;
- regional propagation direction is not invariant across valid blocks;
- event-like derivative traces are often similar to measured-motion controls;
- transform and injection diagnostics remain conditional on previously selected
  masks or frequencies;
- all intervals come from one fish and one recording.

The appropriate current report is:

> A frozen expert-supported rostral region contains a reproducible periodic
> image pattern near 3.2-3.35 Hz across temporally separate intervals from one
> recording. The pattern is a cardiac hypothesis, but cardiac identity, heart
> rate, and beat timing remain unvalidated.

Do not report a beat-to-beat series or label `192..201 bpm` as measured heart
rate yet.

## Surrogate Calibration Correction - 2026-07-11

Code review found that the shared autocorrelation-preserving surrogate rolled
pixel traces but left `pixel_valid` at its original time positions. It then
intersected original validity with finiteness of the shifted trace. In the full
cache, `18,935/88,935` otherwise valid frames have partial pixel validity, and
per-pixel validity within valid frames ranges from `85.4%` to `99.99%`.
Therefore this mismatch can reduce usable null samples and make null scores too
small.

The affected helper was used by earlier discovery, dynamic-support, consensus,
and global-null analyses. All p-values produced through that helper, including
the `p=0.005` compact-mask and five-minute results and the longitudinal
`p=0.025` recurrence result, are **withdrawn pending recalibration**. Their
observed scores, selected frequencies, masks, plots, and held-out descriptive
comparisons remain useful diagnostics; they are not calibrated detections.
Regional phase-randomization tests use a different null mechanism, but remain
conditional on already selected masks, frequencies, and filters.

## Frozen Hypothesis

The hypothesis has four fixed parts:

```text
canonical local ROI:       235 pixels
frozen heart support:       38 pixels
frequency search:          3.0..3.5 Hz in 0.05 Hz steps
frozen regional split:      18 upper pixels / 20 lower pixels
```

The 38-pixel support was created once from the union of the two fold-specific
clusters discovered in frames `30000..32999`. It was not recomputed for the
other intervals.

The upper/lower masks use only frozen-mask geometry. A horizontal division at
canonical `y=121.5` balances the support into 18 upper and 20 lower pixels.

## Independence Contract

The word "independent" has a limited and precise meaning here.

| Interval | Mask status | Frequency status | Regional status | Interpretation |
|---|---|---|---|---|
| `30000..32999` | derived from same interval | post-hoc centered search | split formulated after viewing | exploratory discovery |
| `33000..35999` | frozen from prior interval | `3.0..3.5 Hz` fixed before analysis | split formulated after viewing | temporal mask confirmation; regional result exploratory |
| `0..29999` | same frozen mask | same explicit bounds | frozen upper/lower masks | retrospective held-out test relative to algorithmic selection |

The last two analyses test one unchanged spatial hypothesis on pixels that did
not create it. The frame-0 interval occurs earlier chronologically, but it was
not used to select the mask, frequency range, or regional split.

This is not full biological independence. All intervals share:

- one fish and recording;
- one acquisition and crop pipeline;
- the same 235-pixel canonical ROI;
- fixed reference anchors derived from the frame-30000 reference geometry;
- the same nuisance variables and analysis implementation.

No claim has been tested across fish, sessions, temperatures, or cameras.

## Pixel And Timebase Contract

All quantitative traces are bilinear samples from original acquisition
crop-video pixels. Stabilized/local coordinates define where to sample; warped
display pixels are not the photometry source.

The frame-0 cache contract is:

```text
frame range:             0..29999, step 1
elapsed time:            299.99 s
effective rate:          100.0013 fps from acquisition timestamps
local valid frames:      26959 / 30000 (89.86%)
canonical ROI pixels:    235
```

Invalid frames and long gaps remain invalid. Filtering and spectral estimation
operate on contiguous valid blocks rather than compressing or bridging the time
axis.

## Long-Cache Extraction

The dense subject masks have physical Zarr chunks `(256, 1, 512, 512)`. The old
one-row read loop repeatedly decompressed the same 256-row component chunk.

The optimized extraction uses:

1. a 256-row aligned cache per mask component;
2. four concurrent, resumable range workers;
3. ten logical chunks of 3000 frames;
4. one-frame overlaps for exact motion and duplicate-frame handling;
5. fixed checked anchors `anterior=(128,113)` and
   `posterior=(127,143)`;
6. a fail-closed merge that validates pixel grids, masks, schemas, anchors, and
   static metadata before writing the authoritative cache.

A cached/uncached 300-frame comparison was identical for every persisted
numeric array. Runtime changed from `73.5 s` to `7.6 s`, a `9.7x` single-worker
speedup. Each component recorded 29872 cache hits and 127 misses over the merged
five-minute cache.

The authoritative cache is:

```text
playgrounds/heartrate_stabilization/outputs/
  reliable_local_rostral_start_0_30000f.local_pixel_matrix.npz
```

## Whole-Mask Results (Historical; P-Values Withdrawn)

The dynamic model permits fixed phase and contrast-polarity differences between
pixels. Pixel loadings and nuisance models are learned on discovery blocks and
applied only to opposite-fold held-out blocks. Every full-pipeline surrogate
reruns nuisance fitting, frequency search, and loading estimation.

| Interval | Duration | Selected frequency | Support p | Shared-phase p | Latent p | Control ratio | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `30000..32999` | 30 s | `3.25 Hz` | `0.575` | `0.450` | `0.010` | `4.59` | exploratory mask discovery |
| `33000..35999` | 30 s | `3.35 Hz` | `0.815` | `0.940` | `0.010` | `3.51` | frozen-mask latent confirmation |
| `0..29999` | 300 s | `3.20 Hz` | `0.005` | `0.005` | `0.005` | `4.61` | full frozen-hypothesis confirmation |

The frame-0 full test used 199 unique full-pipeline surrogates. No surrogate
equaled or exceeded the observed support, shared-phase, or latent statistic.

```text
observed support score:       0.1233
maximum null support score:  0.1114

observed shared score:        0.0857
maximum null shared score:   0.0365

observed latent score:        0.6338
maximum null latent score:   0.1645
```

### Exploratory 2--4 Hz sensitivity scan

After inspecting the prespecified `3.0..3.5 Hz` result, the untouched
`36000..65999` interval was searched over `2.0..4.0 Hz` in `0.05 Hz` steps.
This is a post hoc sensitivity analysis, not a replacement confirmatory test.
The four anatomical masks remained frozen, and the familywise null took the
maximum across every searched frequency and all four masks.

All four masks still selected `3.25 Hz`. The compact consensus-9 and
intersection-8 masks exceeded the familywise support, shared-phase, and latent
nulls at `p=0.005`; original-38 and union-39 did not. Their familywise latent
`p` values were `0.430` and `0.375`, respectively.

The cross-fit latent curves also favor the original frequency neighborhood.
For the four masks, the strongest latent alternatives below `3.0 Hz` were only
`47.5%`, `51.8%`, `47.6%`, and `45.4%` of each mask's `3.25 Hz` peak. The
strongest alternatives above `3.5 Hz` were `39.8%`, `46.3%`, `45.3%`, and
`39.0%`. Full support, shared-phase, and latent frequency curves are saved in:

```text
reliable_local_rostral_start_36000_30000f_exploratory_four_mask_2p0_4p0.mask_comparison.arrays.npz
```

The wider band changes the sideband set used by the shared-phase statistic, so
its raw score should not be compared directly with the narrow-band raw score.
The selected frequency, support score, and latent score remain unchanged.

### Complete-recording longitudinal analysis

The complete recording contains `140035` frames over `1400.34 s` (`23.34
min`). A resumable 47-part extraction produced one validated local-coordinate
cache with `88935/140035` locally valid frames (`63.51%`):

```text
reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz
```

The frozen original-38, consensus-9, intersection-8, and union-39 masks were
then evaluated in 23 non-overlapping one-minute windows plus the final
20.34-second window. Each window independently cross-fit nuisance coefficients
and pixel phase/polarity loadings while searching the fixed `2.0..4.0 Hz` grid.
This produces a descriptive candidate-frequency trajectory, not validated
per-minute heart rates.

Twenty-one of 24 windows supported the dynamic analysis. Windows covering
minutes 10--13 failed because locally valid frame fractions were only `0.141`,
`0.036`, and `0.256`. The trajectory plot leaves an explicit gap rather than
connecting across those windows.

For the two compact frozen masks:

| Mask | Scorable windows | Median | Range | Frequency range |
|---|---:|---:|---:|---:|
| consensus-9 | 21 | `195 cycles/min` | `183..219` | `3.05..3.65 Hz` |
| intersection-8 | 21 | `195 cycles/min` | `183..219` | `3.05..3.65 Hz` |

The first ten minutes generally select `183..195 cycles/min`. Later scorable
windows more often select `201..219 cycles/min`, and the final 20.34 seconds
select `195 cycles/min`. Thus the descriptive trajectory does not support a
constant approximately 180-BPM source throughout the recording.

The broad original-38 and union-39 masks select `2.00 Hz` and `2.15 Hz` in
windows 18 and 21 while both compact masks remain near `3.5 Hz`. Those lower
edge broad-mask windows are associated with worse tracking diagnostics than
ordinary broad-mask windows:

```text
mask sample validity:        0.575 vs 0.751
detection confidence:        0.697 vs 0.826
local translation:           7.28 vs 6.27 px
source-coordinate step:      1.29 vs 1.12 px
transform uncertainty:       0.284 vs 0.247
```

This supports treating the broad-mask lower-frequency estimates as tracking or
peripheral-mask failures. It is an association diagnostic, not a causal proof.

A 39-surrogate longitudinal null reran the complete `2.0..4.0 Hz` frequency
search for consensus-9 and intersection-8 in every scorable window. Two tests
answer different questions:

1. The maximum-window test corrects across 21 windows, two masks, and every
   frequency. No individual one-minute window passes. The observed maximum
   latent score is `2.192`, below the global `3.261` threshold; the smallest
   familywise window `p` is `0.275`.
2. An exploratory sustained-recurrence test uses the median latent score across
   all 21 scorable windows and corrects across the two masks. Observed medians
   are `1.319` and `1.408`; the strongest null median is `0.909`; both masks
   have conditional familywise `p=0.025`. This aggregation statistic was chosen
   after reviewing the maximum-window result, so it is post hoc and cannot be
   promoted as a confirmatory p-value on this recording.

The 39-surrogate test has coarse Monte Carlo resolution (`1/40 = 0.025`) and
can be extended to 199 using the saved deterministic batches. The current
result provides exploratory support for a sustained recurrent compact-mask
oscillator across the recording, but not confirmatory evidence for the chosen
aggregate, inferential claims about any particular minute, or cardiac identity.

Key artifacts:

```text
reliable_local_rostral_start_0_140035f_frozen_masks_60s_2p0_4p0.longitudinal.csv
reliable_local_rostral_start_0_140035f_frozen_masks_60s_2p0_4p0.longitudinal.png
reliable_local_rostral_start_0_140035f_frozen_masks_60s_2p0_4p0_global_null39.global_null.summary.json
reliable_local_rostral_start_0_140035f_frozen_masks_60s_2p0_4p0.tracking_diagnostics.summary.json
```

Two frame-locked full-recording videos use stride 3 at 30 fps (`0.9x` source
speed). Both contain `46679` frames and last `1555.97 s` in playback:

```text
reliable_local_rostral_start_0_140035f_frozen_masks_full_length_stride3.mask_overlay.mp4
reliable_local_rostral_start_0_140035f_frozen_masks_full_length_stride3.data_overlay.mp4
```

The clean video shows thin frozen-mask contours on the local stabilized ROI.
The data video keeps encodings separate:

- intersection-8 fill hue uses the cyclic `twilight_shifted` phase map;
- phase opacity uses held-out loading weight and relative analytic amplitude;
- intersection outline and timeline use a fixed `viridis` `2..4 Hz` scale;
- text reports the descriptive window frequency, cycles/min, latent score, and
  validity;
- unscorable windows are gray, say `no reliable window estimate`, and preserve
  a timeline gap without interpolation.

Phase is reconstructed separately inside each scorable window using opposite-
fold loadings and that window's selected frequency. It remains a visualization
of the candidate oscillator, not biological contraction phase.

A second full-recording spatial-phase diagnostic expands the colored support
from intersection-8 to all 38 pixels in the original frozen anatomical mask:

```text
reliable_local_rostral_start_0_140035f_frozen_masks_60s_2p0_4p0_original38.longitudinal_phase.arrays.npz
reliable_local_rostral_start_0_140035f_frozen_masks_full_length_stride3_original38_phase.data_overlay.mp4
```

This diagnostic still freezes each window to the frequency selected by
intersection-8. The outer pixels therefore cannot move the frequency estimate;
their colors show only their held-out phase at the compact core's candidate
frequency. All 38 mask pixels receive phase estimates when supported, opacity
continues to encode held-out loading and relative analytic amplitude, and the
intersection remains outlined as a spatial reference. This makes persistent
top-versus-bottom agreement, opposition, or phase delay visible without
claiming that every original-mask pixel is part of the oscillator. Unscorable
windows remain phase-free. The video has the same `46679` frames, `1555.97 s`
playback duration, and `0.9x` source speed as the compact-only render.

The broader rendering is a visual diagnostic, not a completed measurement of
top/bottom coupling. Quantifying that relationship still requires prespecified
upper/lower regions and held-out circular phase-difference statistics across
valid blocks.

That regional measurement was then run across the full recording using the
previously frozen `y=121.5` split: 18 upper and 20 lower original-mask pixels.
It uses the same per-window intersection-8 frequency as the broad phase render.
Across 21 scorable windows it contains 155 held-out valid blocks and 705 paired
analytic-phase crossings. The window-level circular mean is `+99.2 deg`
(lower-minus-upper phase), with across-window PLV `0.644`. A conditional
one-minute-window random-rotation null gives `p=0.001`.

This is evidence that the regional offsets are not distributed uniformly
around the cycle, but it is not evidence for a rigid traveling wave. The
window means range from about `-99 deg` to `+158 deg`; 18/21 have the lower
region leading by sign, but several late windows are nearly synchronous and
one clear window reverses. Across paired crossings, the lower region leads in
`69.1%`; median lag is `-0.167` local cycles with MAD `0.175` cycles. The
within-window PLV also varies substantially (`0.23..0.94`). Thus the observer's
impression of inconsistent direction is compatible with the quantitative
result: there is a broad directional bias, not a uniform propagation delay.
The crossings are narrow-band phase landmarks, not independently detected raw
brightness peaks or validated cardiac events.

```text
reliable_local_rostral_start_0_140035f_original38_frozen_upper_lower.regional_longitudinal.png
reliable_local_rostral_start_0_140035f_original38_frozen_upper_lower.regional_longitudinal.windows.csv
reliable_local_rostral_start_0_140035f_original38_frozen_upper_lower.regional_longitudinal.summary.json
```

## Mono8 Photometry Transform Challengers

A descriptive outer-window comparison now evaluates 14 Mono8 trace
constructions at each window's already frozen intersection-8 frequency. Within
each window, nuisance fits and the matched spatial projection are learned on
alternating discovery blocks and applied only to the opposite blocks. Even
one-minute windows select a descriptive challenger; odd windows are retained
for display only. No transform-family surrogate null has been run, so the
selection and ratios below are not detection p-values.

The candidates include mean and Huber intensity, local-reference log and
fractional normalization, mask-normalized Gaussian smoothing, regional spatial
standard deviation, three centered Savitzky-Golay derivatives, a
Gaussian-plus-derivative trace, three signed normalized lag differences, and a
cross-fitted inverse-noise matched spatial projection. All temporal transforms
preserve invalid gaps; spatial smoothing cannot cross a frozen mask boundary.

Key discovery/outer-confirmation medians are:

| candidate | discovery target/sideband | confirmation target/sideband | confirmation target/control | confirmation upper/lower PLV |
| --- | ---: | ---: | ---: | ---: |
| mean intensity | 1.01 | 1.37 | 1.42 | 0.75 |
| regional spatial SD | 2.49 | 2.42 | 3.23 | 0.38 |
| matched spatial projection | 1.71 | 2.08 | 2.08 | n/a |
| SG derivative, 11 frames | 1.56 | 1.75 | 1.49 | 0.86 |
| signed lag, 12 frames | 1.51 | 1.69 | 1.36 | 0.89 |
| signed lag, 16 frames | 1.45 | 1.73 | 1.87 | 0.94 |

Regional spatial SD is the strongest descriptive frequency-presence
challenger, but its poor upper/lower phase locking means it should not be used
to infer directional propagation or event timing. It likely measures periodic
texture/deformation rather than a signed photometric waveform. The matched
projection best recovers a target-specific signed latent trace when pixel
polarities cancel. The 11-frame derivative and 12/16-frame differences give
the cleanest upper/lower phase relationship and are the appropriate event-trace
challengers. Log reference normalization and spatial smoothing alone do not
materially improve the ordinary intensity trace.

The full run produced 293 scorable candidate-window rows; the three known
unscorable minutes remain absent for all candidates, and only the 16-frame lag
loses one additional low-validity window. These results justify carrying
spatial SD, matched projection, SG-11, and signed-lag challengers into a
transform-family surrogate and injection-recovery study. They do not validate
cardiac identity or events.

```text
reliable_local_rostral_start_0_140035f_original38_photometry_transform_comparison.photometry_transforms.png
reliable_local_rostral_start_0_140035f_original38_photometry_transform_comparison.photometry_transforms.windows.csv
reliable_local_rostral_start_0_140035f_original38_photometry_transform_comparison.photometry_transforms.summary.json
```

## Photometry Validation Audit - 2026-07-11

Three follow-up checks were implemented to test the descriptive transform
results: measured-motion reconstructions, additive Mono8 injection/recovery,
and a transform-family surrogate calibration. These checks narrow the claim;
they do not promote the candidate to a heart-rate measurement.

### Representative Motion Controls

Five prespecified minutes (`0`, `7`, `14`, `18`, and `22`) were compared with
two motion-only reconstructions:

1. integrated cached image-gradient dot measured coordinate displacement;
2. a static cached-frame surface resampled at the measured coordinates.

For the full 38-pixel mask, median target/sideband ratios were:

| transform | observed | gradient/displacement | static reference |
| --- | ---: | ---: | ---: |
| regional spatial SD | `1.54` | `0.81` | `1.00` |
| matched spatial projection | `1.59` | `1.03` | `0.78` |
| SG derivative, 11 frames | `1.48` | `1.27` | `1.35` |
| signed lag, 12 frames | `1.38` | `1.21` | `1.29` |
| signed lag, 16 frames | `1.31` | `0.94` | `1.31` |

Spatial SD and matched projection are not reproduced by these two limited
motion controls. The SG and lag traces are much closer to at least one control,
so they cannot yet support reliable event timing. The eroded 10-pixel interior
also contains periodic structure, which argues against a purely administrative
mask-boundary effect.

These comparisons are descriptive. The first control uses gradients from the
observed dynamic frames, the second reconstructs only the cached local surface,
and neither is optical flow or a calibrated motion null. The original run also
evaluated every control only at the observed compact-mask frequency. A revised
comparison therefore needs each control's own best frequency and identical
temporal support before addressing an adaptively selected peak.

The corrected runner independently searched the same `2.0..4.0 Hz` grid and
rescored observed and control signals on identical valid rows and logical
blocks. Across the same five representative minutes, eligible full-mask median
observed/control searched-maximum ratios were:

| transform | gradient/displacement | eligible windows | static reference | eligible windows |
| --- | ---: | ---: | ---: | ---: |
| regional spatial SD | `1.22` | `3/5` | `1.01` | `4/5` |
| SG derivative, 11 frames | `1.13` | `3/5` | `1.14` | `2/5` |
| signed lag, 12 frames | `0.94` | `3/5` | `0.80` | `1/5` |
| signed lag, 16 frames | `0.99` | `3/5` | n/a | `0/5` |

Matched projection is excluded from this adaptive comparison because its
spatial weights were learned at the frozen observed frequency rather than
refit across the grid. Search-boundary maxima are also excluded from claim
eligibility. The eroded-interior spatial-SD ratio against the gradient control
was `0.91` across only `2/5` eligible windows, while the boundary ratio was
`1.31` across `4/5`. These small or sparsely supported full-mask ratios and the
stronger boundary result do not provide convincing separation from measured
motion. No calibrated motion null was run.

```text
reliable_local_rostral_start_0_140035f_original38_photometry_motion_controls_representative5.motion_controls.png
reliable_local_rostral_start_0_140035f_original38_photometry_motion_controls_representative5.motion_controls.comparisons.csv
reliable_local_rostral_start_0_140035f_original38_photometry_motion_controls_representative5.motion_controls.summary.json

reliable_local_rostral_start_0_140035f_original38_photometry_motion_controls_paired_representative5_v3.motion_controls.paired_support.csv
reliable_local_rostral_start_0_140035f_original38_photometry_motion_controls_paired_representative5_v3.motion_controls.summary.json
```

### Conditional Injection/Recovery Pilot

The pilot adds known signals to real cached Mono8 photometry while preserving
the recording's validity, nuisance, and tracking arrays. It freezes the existing
38-pixel mask and tests five selected transforms over six 16-second windows and
five frequencies. It therefore measures conditional cached-photometry
recoverability, not end-to-end localization, false-positive rate, cardiac
identity, or event detection.

The corrected schema requires minimum confirmation coverage, phase coherence,
and per-region/per-band target-to-sideband support before reporting angles,
timing, or direction. Continuous and intermittent `1.5 DN`, `3.25 Hz`,
three-band waves were both recovered at the exact injected frequency. The
continuous family selected SG-11 and the intermittent family selected spatial
SD. However, **all phase, timing, and direction outputs were withheld** because
the individual regions or traveling bands did not pass the signal-support gate
in enough windows. A `0.5 DN` pilot did not confirm. The stricter six-window `0
DN` sanity job also did not confirm.

Earlier schemas reported approximately `26..42 ms` timing errors for some `1.5
DN` jobs and allowed one unchanged background to pass a heuristic gate. Those
legacy phase/timing values are withdrawn because they could assign stable phase
to weak or effectively zero-amplitude subregions. They remain evidence that
coverage, coherence, and regional signal-support gates materially affect the
conclusion, not a false-positive-rate estimate. One fixed background trial
cannot estimate false-positive rate.

```text
reliable_local_rostral_original38_photometry_injection_smoke_final.injection_recovery.png
reliable_local_rostral_original38_photometry_injection_smoke_final.injection_recovery.csv
reliable_local_rostral_original38_photometry_injection_smoke_final.injection_recovery.summary.json

reliable_local_rostral_original38_photometry_conditional_smoke_v4_final.injection_recovery.png
reliable_local_rostral_original38_photometry_conditional_smoke_v4_final.injection_recovery.csv
reliable_local_rostral_original38_photometry_conditional_smoke_v4_final.injection_recovery.summary.json
```

### Transform-Family Null Status

The first attempted 39-surrogate transform-family run was stopped after code
review. It shifted intensity traces without shifting per-pixel missingness,
which reduced usable null samples and could make the test anti-conservative.
Its partial batches are invalid and must not be cited.

The corrected calibration preserves traces and per-pixel validity together,
requires a minimum amount of scorable confirmation data, and fails closed when
implementation dependencies change. Even after correction, it is explicitly a
**conditional** test: the one-minute frequencies come from an earlier adaptive
search on overlapping compact-mask pixels. It does not rerun mask discovery or
frequency selection and cannot establish full-pipeline oscillator significance.

### Decision

The data justify retaining a candidate localized oscillator and the code needed
to test it. They do not justify reporting heart rate or beat events from this
recording. The embedded positive-control Trial 1 analysis now provides a strong
descriptive rate recovery, as recorded in
[`heartrate_analysis_status_2026-07-11.md`](heartrate_analysis_status_2026-07-11.md).
It does not replace frozen evaluation on Trials 2--48, a no-heart/no-signal
control, or blinded visible beat annotations or an independent physiological
measurement for the moving fish.

The standard compact adaptive-cluster analysis still returns no estimate on the
five-minute cache. Only one discovery fold passes, and cross-fold dilated
overlap is `0.367`. This does not negate the separately prespecified whole-mask
test; it shows that a compact same-location cluster is not a stable model of the
observed spatial pattern.

## Phase Visualization

The five-minute phase reconstruction has:

```text
selected frequency:             3.20 Hz
phase-valid frames:             10510 / 30000 (35.03%)
median spatial alignment:       0.702
video stride / playback:        10 frames / 30 fps
display speed:                  3x source time
```

Uncolored spans are expected. They represent invalid frames, cross-fit guards,
long gaps, or the 0.75-second edge removed from each filtered block.

Smooth hue cycling is not detection evidence by itself. Narrow-band analytic
filtering necessarily produces phase cycles. The evidential quantities are the
held-out spatial agreement and the full-pipeline surrogate comparison.

Key artifacts:

```text
reliable_local_rostral_start_0_30000f_frozen_hypothesis_confirmatory199.dynamic_phase.mp4
reliable_local_rostral_start_0_30000f_frozen_hypothesis_confirmatory199.dynamic_phase.strip.png
reliable_local_rostral_start_0_30000f_frozen_hypothesis_confirmatory199.dynamic_support.diagnostic.png
```

## Regional Delay Results

Positive lag means the lower region reaches the same phase after the upper
region. Negative lag means the lower region leads.

| Interval | Blocks | Paired cycles | Within-block PLV | Across-block PLV | Conditional p | Direction |
|---|---:|---:|---:|---:|---:|---|
| `30000..32999` | 4 | 21 | `0.989` median | `0.720` | `0.14` | same sign, variable magnitude |
| `33000..35999` | 4 | 22 | `0.990` median | `0.180` | `0.86` | reverses between blocks |
| `0..29999` | 55 | 306 | `0.979` median | `0.592` | `0.001` | lower leads in 47/55 blocks |

For the five-minute interval:

```text
circular mean lower lag:       -103 ms
median paired-cycle lag:       -88.6 ms
paired-cycle lag MAD:           34.2 ms
```

The regional `p=0.001` is conditional. Its null independently randomizes the
lower-region phase of each held-out block while holding the selected band,
filter, support, and regional masks fixed. It tests repeatability of the delay,
not whether the oscillator is cardiac.

The contrast with the chase-associated intervals is informative. Regional
delay is more consistent in the higher-coverage early cache, which supports the
hypothesis that chase motion, pose, or tracking reliability contributed to the
later direction changes.

## What The Evidence Supports

The current data support:

- a repeatable periodic image pattern within the frozen local-rostral region;
- recurrence of a compact spatial core in temporally separated intervals from
  one recording;
- descriptive peaks near `3.2..3.35 Hz` across three intervals;
- spatial-SD and matched-projection peaks that are not fully reproduced by two
  limited measured-motion reconstructions at the frozen frequency;
- a conditional, reproducible upper/lower delay over many early-video blocks;
- treating chase-associated instability as a possible state/tracking effect
  rather than immediate disproof of the source.

## What The Evidence Does Not Support

The current data do not support:

- calling the source definitively cardiac;
- assigning a biological contraction phase to the latent oscillator;
- reporting beat timestamps or heart-rate variability;
- claiming every colored cycle is a visible contraction;
- claiming the source is absent from the suspected esophagus;
- generalization across animals or recordings;
- production thresholds for sensitivity or false-positive rate.

## Next Promotion Gates

1. Rerun every shared-surrogate calibration after the validity-alignment fix.
2. Define one visible contraction landmark with the expert.
3. Collect blinded timestamps and unscorable intervals without algorithm
   overlays.
4. Repeat annotation or use a second viewer to estimate human timing error.
5. Freeze event polarity, phase convention, thresholds, masks, and matching
   tolerance before scoring.
6. Report one-to-one event precision, recall, F1, timing error, interval error,
   and validated coverage.
7. Add a separately drawn esophagus mask and run it as a control.
8. Calibrate the complete adaptive mask/frequency/event pipeline with real-noise
   injection recovery and multiple independent null backgrounds.
9. Apply the frozen embedded-fish method to positive-control Trials 2--48
   without outcome-guided ROI or parameter tuning.
10. Repeat the moving-fish analysis on additional fish and recordings.

Until those gates pass, retain the current status:

```text
repeatable periodic image-pattern candidate
not yet validated heart rate
```

## Five-Minute Mask Relearning Experiment (Historical; P-Values Withdrawn)

The original 38-pixel support was the union of two compact clusters learned
from frames `30000..32999`. To test whether that short chase-associated interval
was a poor mask-discovery source, the frame-0 five-minute cache was divided into
five contiguous one-minute outer folds.

For each outer fold:

1. the other four minutes learned nuisance coefficients, frequency, compact
   support, and per-pixel complex loadings;
2. a one-second boundary guard separated discovery from the excluded minute;
3. the learned loadings phase-aligned only the excluded minute;
4. 199 autocorrelation-preserving held-out surrogates calibrated the temporal
   score; and
5. matched interior, boundary, global, body, external, and predicted-motion
   controls were measured.

Across the five discovery masks, spatial recurrence was calibrated with 999
independent shape-preserving translations within the physically eligible ROI.
This spatial null is conditional on the discovered cluster shapes. Temporal
circular shifts were not used for mask-location recurrence because they
preserve which pixels contain narrow-band power.

Nine pixels were selected in all `5/5` outer discoveries:

```text
canonical coordinates:
(128,123) (129,123) (130,123)
(128,124) (129,124) (130,124)
          (129,125)
(128,126) (129,126)

spatial maximum-null p:       0.019
pixels inside original mask:  8 / 9
Jaccard with original mask:   0.205
```

Only `2/5` excluded minutes passed the phase-aligned temporal/control gate.
Therefore the prespecified internal promotion result is:

```text
detected: false
reason: too_few_outer_folds_confirmed
```

The nine-pixel support was retained only as an exploratory frozen challenger.
It was not substituted for the original mask before evaluating the next
interval.

## Untouched Four-Mask Comparison (Historical; P-Values Withdrawn)

Frames `36000..65999` were predeclared before extraction as the next temporal
test. The chunk-safe extraction produced:

```text
frame count:          30000
locally valid:        19775 / 30000 (65.92%)
```

Four masks were frozen before reading this cache:

```text
original support:       38 pixels
five-minute consensus:   9 pixels
intersection:             8 pixels
union:                   39 pixels
```

Each mask searched the same explicit `3.0..3.5 Hz` grid using 199 identically
seeded dynamic-support surrogates. For each surrogate index, the familywise null
took the maximum score across all four masks; each mask's own null already took
the maximum across searched frequencies.

| Mask | Frequency | Support FWER p | Shared-phase FWER p | Latent FWER p | Latent/control |
|---|---:|---:|---:|---:|---:|
| original 38 | `3.25 Hz` | `1.000` | `0.485` | `0.480` | `5.02` |
| consensus 9 | `3.25 Hz` | `0.005` | `0.005` | `0.005` | `6.86` |
| intersection 8 | `3.25 Hz` | `0.005` | `0.005` | `0.005` | `7.91` |
| union 39 | `3.25 Hz` | `1.000` | `0.375` | `0.415` | `4.96` |

The strongest matched control for every mask was the predicted heart-motion
trace. The standard compact adaptive-cluster analysis still returned no
estimate (`discovery_not_significant_in_both_folds`).

This result supports a compact recurrent periodic core inside the original
region more strongly than the broad 38-pixel union on this interval. It does
not show that the remaining pixels are biologically irrelevant: the broad-mask
statistics use a familywise raw maximum whose null is dominated by the
strongest small-mask statistic, and the comparison remains within one fish and
recording. Most importantly, none of these tests identifies the oscillator as
cardiac or validates event times.

Matched diagnostic-only `0.9x` phase videos were then rendered for the
consensus-9 and union-39 masks on the untouched interval without rerunning any
surrogate decision. Both have `20.55%` phase-valid coverage. Median spatial
alignment is `0.908` for the compact core and `0.682` for the broad union. This
visual comparison is consistent with, but not independent of, the completed
four-mask statistics.

The previously frozen 18-pixel upper and 20-pixel lower regions were then
applied unchanged to the union-39 phase reconstruction. The additional
consensus pixel outside the original mask was not assigned to either region.

```text
valid regional blocks:           34
paired cycles:                   174
lower-leading blocks:            27 / 34
lower-leading paired cycles:     141 / 174
circular mean lower lag:         -123.6 ms
mean phase lower minus upper:     144.6 deg
median paired-cycle lag:         -111.2 ms
paired-cycle lag MAD:              21.6 ms
across-block PLV:                  0.739
median within-block PLV:           0.994
conditional block-phase p:         0.001
```

This objectively supports the visual impression that the lower/core region is
usually near the opposite phase from the upper region. Direction reverses in
7/34 blocks, so the relationship is dominant rather than invariant. The test
is conditional on the already frozen support, `3.25 Hz` frequency/filter, and
regional masks. It does not rescue the broad union's four-mask detection result
or identify the oscillator as cardiac.

## Validation

The current implementation passed:

```text
87 focused photometry and shared-surrogate unit tests
scripts/py -m py_compile for the analysis, extraction, and rendering modules
git diff --check
cached-versus-uncached exact numeric comparison
video metadata and representative valid/invalid frame inspection
```

The historical dynamic runs produced 199/199 distinct surrogate values, but
distinctness does not correct the validity-alignment defect described above.
Those p-values remain withdrawn until regenerated with the corrected helper.
