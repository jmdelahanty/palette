# Heartrate Frozen-Mask Validation Status - 2026-07-10
<!-- contract-meta
status: draft
last_updated: 2026-07-10
-->

Purpose: record the current evidence for the local-rostral frozen whole-mask
heart candidate after temporal cross-fitting, a second 30-second interval, and a
five-minute frame-0 cache.

The detailed implementation and earlier investigation history remain in
[`heartrate_local_rostral_roi_status_2026-07-09.md`](heartrate_local_rostral_roi_status_2026-07-09.md).

## Bottom Line

The current analysis provides **strong evidence for a reproducible periodic
anatomical source near `3.2..3.35 Hz` inside a frozen 38-pixel local-rostral
support**.

It does **not yet establish a validated heart rate** because:

- no visible biological contraction landmark has been defined;
- no blinded beat timestamps have been compared with inferred events;
- no explicit esophagus mask/control has been measured;
- the regional phase null is conditional on the selected frequency and filter;
- all intervals come from one fish and one recording.

The appropriate current report is:

> A frozen expert-supported rostral mask contains a reproducible periodic
> spatial pattern near 3.2–3.35 Hz across temporally separate intervals from one
> recording. The source is a strong cardiac candidate, but beat identity and
> beat timing remain unvalidated.

Do not report a beat-to-beat series or label `192..201 bpm` as measured heart
rate yet.

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

## Whole-Mask Results

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

- a periodic spatial source inside the frozen 38-pixel support;
- temporal generalization within one recording;
- a selected band near `3.2..3.35 Hz` across three intervals;
- separation from measured global, body-control, external-control, and
  gradient-predicted motion traces;
- reproducible upper/lower delay over many early-video held-out blocks;
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

1. Define one visible contraction landmark with the expert.
2. Collect blinded timestamps and unscorable intervals without algorithm
   overlays.
3. Repeat annotation or use a second viewer to estimate human timing error.
4. Freeze event polarity, phase convention, thresholds, masks, and matching
   tolerance before scoring.
5. Report one-to-one event precision, recall, F1, timing error, interval error,
   and validated coverage.
6. Add a separately drawn esophagus mask and run it as a control.
7. Calibrate the latent whole-mask detector with real-noise injection recovery.
8. Repeat the frozen analysis on additional fish and recordings.

Until those gates pass, retain the current status:

```text
strong reproducible periodic anatomical candidate
not yet validated heart rate
```

## Validation

The current implementation passed:

```text
16 focused unit tests
scripts/py -m py_compile for the analysis, extraction, and rendering modules
git diff --check
cached-versus-uncached exact numeric comparison
199/199 unique full-pipeline null values for each dynamic statistic
video metadata and representative valid/invalid frame inspection
```
