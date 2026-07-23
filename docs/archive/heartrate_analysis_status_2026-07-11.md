# Heartrate Analysis Status - 2026-07-11
<!-- contract-meta
status: draft
last_updated: 2026-07-12
-->

Purpose: summarize the current heartrate evidence across the freely moving fish
investigation and the new embedded-fish positive control, including negative,
inconclusive, and withdrawn analyses.

Final decision: see
[`heartrate_final_decision_2026-07-12.md`](heartrate_final_decision_2026-07-12.md).
That document supersedes earlier operational suggestions in this history that
the moving-fish window-frequency trajectory might be reportable.

Detailed method and artifact history remains in:

- [`heartrate_top_view_mask_decision_2026-07-11.md`](heartrate_top_view_mask_decision_2026-07-11.md)
- [`heartrate_local_rostral_roi_status_2026-07-09.md`](heartrate_local_rostral_roi_status_2026-07-09.md)
- [`heartrate_frozen_mask_validation_status_2026-07-10.md`](heartrate_frozen_mask_validation_status_2026-07-10.md)
- [`heartrate_stabilized_roi_case.md`](heartrate_stabilized_roi_case.md)

## Executive Status

There are now two materially different results.

### Freely moving fish

The moving-fish recording contains a repeatable localized periodic image
pattern, usually near `3.2..3.35 Hz`, in an expert-identified local-rostral
region. That pattern remains a cardiac hypothesis, not a validated heart-rate
measurement.

The simple no-fold full-recording scan strengthens the recurrence claim: 22/24
one-minute windows are scorable, 19/22 select `3.0..3.6 Hz`, no window selects a
search boundary, and the 38-pixel and 8-pixel supports agree within `0.10 Hz`
in all scorable windows. The aggregate candidate is `3.10 Hz`, with later
windows more often near `3.35..3.45 Hz`. Only 459.4 seconds of the 1400.34-second
recording survive common-pixel and filter-edge requirements.

The standard compact adaptive-cluster method returns no estimate. Several
whole-mask and compact-core diagnostics look promising, but the main shared
surrogate p-values are withdrawn because the surrogate shifted intensity
without shifting per-pixel validity. Motion-control comparisons also do not yet
show convincing separation for the event-like derivative and lag traces. The
simple motion-prediction PCA shares the aggregate `3.10 Hz` peak, although its
window trajectory and waveform differ from the observed PCA. No blinded visible
beat timestamps or independent physiological events exist for this recording.

Repeated visual inspection adds a qualified observation: some valid intervals
show clear, repeatable spatial organization as colors change within the frozen
mask, while other intervals are weak, noisy, or spatially incoherent. The
organized intervals are encouraging but observer-reported. They do not provide
a predeclared detection rule, and they cannot be selected post hoc as the only
intervals used for inference.

Current allowed description:

> Repeatable localized periodic image-pattern candidate in one moving fish;
> cardiac identity and beat timing are unvalidated.

The embedded positive control makes a window-level measurement target more
plausible: a frozen compact top-view support may capture a smoothed candidate
oscillator rate even when individual beats cannot be resolved. This would
become a legitimate average heart-rate measurement if the oscillator is
independently validated as cardiac. Until then, report the output as a
candidate oscillator rate rather than heart rate, and do not infer beat-to-beat
timing or variability from the smoothed trajectory.

### Embedded-fish positive control

Trial 1 of the embedded-fish dataset provides a clear positive descriptive
result. A reference-blind, band-limited spatial component from a fixed `22 x
28` top-camera ROI tracks the supplied per-frame rate trace with:

```text
8 s spectral-ridge correlation:     0.883
8 s spectral-ridge MAE:              4.30 bpm
event-interval MAE:                  3.92 bpm
detected peaks:                     88
expected beats from reference:      88.53
target peak / band median power:    47.85
```

Four independently fitted same-size offset boxes have ridge correlations from
`-0.41` to `0.13` and ridge errors from `26.6` to `39.2 bpm`. This argues
against a spatially global video oscillation.

The supplied rate appears to have been derived from the side-camera heart ROI,
so reproducing it from those same side pixels is partly circular. Recovering
the trajectory independently from the anatomically expected top-camera ROI is
the more informative cross-view result. It establishes that spatial
photometry in the stationary top camera can recover a signal consistent with
the side-camera-derived cardiac rate. It does not validate exact contraction
times or the moving-fish pipeline, and Trial 1 is not held-out validation
because the spatial component and event polarity were fitted on the same clip
that was scored.

## Evidence Labels

This report uses four labels deliberately:

| Label | Meaning |
|---|---|
| **negative / no estimate** | A declared gate failed and the pipeline correctly withheld an estimate. |
| **inconclusive** | The result is informative but cannot distinguish cardiac signal from an alternative such as motion. |
| **withdrawn / invalid** | An implementation defect makes the numerical inference unusable; it must not be cited. |
| **descriptive positive** | The observed signal agrees strongly with a reference, but selection and evaluation are not yet fully independent. |

Failure to validate cardiac identity is not evidence that no cardiac signal is
present. It means the current measurement does not support that stronger claim.

## Attempt Ledger

| Attempt | Result | Current interpretation |
|---|---|---|
| Initial keypoint join using `camera_frame_id` | failed overlay | Bookkeeping failure. Corrected by using crop-video row/frame index. |
| Fixed stabilized whole-ROI mean | broad, inconsistent peaks | Inconclusive. Whole means can cancel spatially opposed pixels and remain sensitive to geometry. |
| Local rostral anchor correction | improved worst-frame geometry | Geometric success, signal-validation failure. On the same six windows, local peak-IQR was `10.95 bpm` versus `4.28 bpm` for the fixed ROI; local rhythm score did not improve. |
| Compact cross-fit cluster, original chase interval | no estimate | Negative. One fold failed discovery; dilated overlap was `0.485`, frequency disagreement was `0.15 Hz`, and no events were emitted. |
| Expert-informed 38-pixel whole-mask model | repeated latent oscillation | Inconclusive for cardiac identity. It permits opposite pixel polarities and phases, which fits the visual pattern better than a same-sign mean. |
| Dynamic phase videos | intermittently clear repeatable spatial organization, with other weak or incoherent intervals | Diagnostic only. The organized intervals are encouraging, but any sufficiently strong narrow-band pixel cycles through phase colors; the videos do not independently detect beats, and visually favorable intervals cannot be selected post hoc. |
| Upper/lower delay in two chase intervals | direction unstable between blocks | Negative for the fixed-propagation hypothesis. Conditional delay p-values were `0.14` and `0.86`; the second interval reversed direction across blocks. |
| Frozen mask on the early five-minute interval | more stable regional delay | Promising but conditional. The early interval had better tracking/coverage and a dominant lower-leading relationship, but it still lacked biological beat labels and used an already selected band and mask. |
| Five-minute mask relearning across five outer folds | only `2/5` excluded minutes passed | Negative for internal promotion. The consensus support was retained only as an exploratory challenger. Associated shared-surrogate p-values are withdrawn. |
| Four frozen masks on a later untouched interval | compact 8/9-pixel masks looked stronger than broad masks | Descriptively promising; affected shared-surrogate p-values are withdrawn. It does not validate events or cardiac identity. |
| Full-recording one-minute trajectory | `21/24` windows scorable; three tracking gaps | Inconclusive. Compact masks usually selected `3.05..3.65 Hz`; broad masks sometimes jumped to the low search boundary during poorer tracking. No individual window passed the historical global maximum test, whose shared-surrogate calibration is now withdrawn. |
| Whole-recording regional delay | dominant but non-invariant direction | Inconclusive. Most windows had the same directional sign, but lag and coherence varied and some windows reversed. Crossings are narrow-band phase landmarks, not visible cardiac events. |
| Fourteen Mono8 photometry transformations | spatial SD and matched projection strongest descriptively | Candidate generation only. Derivative and signed-lag traces looked event-like but were often close to measured-motion controls. |
| Corrected adaptive motion-control comparison | small or sparse observed/control ratios | Inconclusive to negative. It did not convincingly separate full-mask transforms from the limited gradient/displacement and static-reference motion reconstructions. No calibrated optical-flow null was run. |
| End-to-end real-noise injection coverage | `0/3` null and `9/27` positive detections | Pipeline check, not an operating curve. No amplitude reached 80% detection; drift, intermittent, and low-frequency cases were weak. |
| Conditional transform injection pilot | `1.5 DN` continuous/intermittent signals recovered; `0.5 DN` did not confirm | Amplitude sensitivity is plausible, but phase, timing, and direction were withheld because regional support gates failed. Earlier `26..42 ms` timing claims are withdrawn. |
| Embedded Trial 1 fixed-ROI spatial PCA | strong agreement with supplied reference and poor offset controls | Descriptive positive control. This is the first result with an external per-frame rate reference, but it is same-trial fitting/evaluation. |
| Simple no-fold PCA transferred back to moving-fish caches | broad ROI selects `1.75 Hz`; frozen 38/core masks select `3.10..3.35 Hz` | Exploratory corroboration of the localized candidate, not a validated rate. All qualifying segments are used, but PCA and polarity are fitted on the same caches. |

## Withdrawn Or Superseded Results

The following results must not be reused in conclusions or presentations.

### Fixed half-shift short-segment surrogates

An early surrogate implementation shifted every short valid segment by exactly
half its length. The nominal 199-entry arrays therefore repeated one
transformation rather than sampling the intended null. Earlier `p=0.005`
claims and the associated `12/27` injection sensitivity count were superseded.
After repair, the compact analysis returned no estimate and the coverage study
detected `9/27` positives.

### Per-pixel validity not shifted with trace

The shared autocorrelation-preserving surrogate later proved to roll intensity
traces while leaving `pixel_valid` in its original time positions. This can
reduce scorable null support and make p-values anti-conservative. All p-values
from that helper are withdrawn pending complete recalibration, including:

- dynamic-support and five-minute `p=0.005` results;
- consensus and four-mask familywise p-values;
- the longitudinal sustained-recurrence `p=0.025` result.

Observed frequencies, masks, phase videos, raw scores, and descriptive held-out
comparisons remain usable as diagnostics. They are not calibrated detections.

### Partial transform-family null

The first 39-surrogate transform-family job was stopped after review because it
also failed to carry missingness with shifted intensity. Its partial batches
are invalid. The corrected implementation preserves trace and validity
together, but any future result remains conditional unless the complete mask,
frequency, transform, and event-selection pipeline is rerun inside every null.

### Legacy injection timing

Earlier injection schemas allowed timing or direction to be reported when some
regions had weak or effectively zero signal. Those `26..42 ms` timing values
are withdrawn. The corrected schema withholds phase and timing unless coverage,
coherence, and per-region signal-support gates all pass.

## Embedded Positive-Control Contract

Dataset:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/
  embedded_fish_positive_control/
```

Trial 1 inputs:

```text
top video:
  Top_Camera/20250109_F1_10_30_Trial1.mp4
  544 x 500, 200 fps, 8001 frames, 40.005 s

supplied rate workbook:
  Bradyinfo_20250109_F1_10_30.xlsx
  heart_rate_trace row 2, 4000 samples at 100 Hz

fixed source-pixel ROI:
  top_camera_roi_trial1.json
  x=261, y=77, width=22, height=28
```

The duration mapping is explicit: one supplied rate sample spans two
top-camera frames. The supplied trace is a dense smoothed rate estimate, not a
set of independently blinded beat timestamps, so this comparison validates
rate tracking more directly than biological event timing.

The Trial 1 analysis predeclares a `1.5..4.0 Hz` band and reports three signals
without selecting a winner from the supplied reference:

1. mean ROI intensity;
2. ROI-to-surround log intensity ratio;
3. leading PCA component of the bandpassed, robustly standardized ROI pixels.

The PCA component is reference-blind: the workbook is not used to fit its
spatial loadings. However, the PCA loading, event polarity, and event trace are
all fitted and assessed within Trial 1. This is why the result is descriptive
rather than confirmatory.

### Trial 1 results

| Signal | Whole-clip peak | Peak/median | Ridge correlation | Ridge MAE | Event MAE |
|---|---:|---:|---:|---:|---:|
| ROI mean | `120.0 bpm` | `6.06` | `0.403` | `12.11 bpm` | `28.76 bpm` |
| ROI/ring log ratio | `142.5 bpm` | `4.32` | `0.584` | `7.66 bpm` | `34.36 bpm` |
| Band PCA 1 | `120.0 bpm` | `47.85` | `0.883` | `4.30 bpm` | `3.92 bpm` |

The whole-clip `120 bpm` spectral maximum is not an estimate of a constant
Trial 1 rate. The supplied rate changes from approximately `95.3` to `153.5
bpm`, and the time-resolved ridge and peak intervals are the relevant
comparisons.

The PCA result indicates that cardiac information is expressed primarily as a
spatial redistribution or deformation pattern rather than uniform brightening
of the complete box. This is conceptually consistent with the moving-fish
observation that different portions of the candidate region can have opposite
contrast polarity. It does not prove the two recordings contain the same
source.

The defensible top-camera claim is therefore:

> A spatially localized top-camera photometric component tracks the
> side-camera-derived cardiac-rate trajectory in stationary embedded Trial 1.

It is deliberately a rate-tracking claim, not a claim that each PCA peak is a
verified contraction.

### Reference-blind embedded mask discovery

A variable-rate version of the moving-fish mask idea was applied to embedded
top-camera Trial 1. A literal fixed-frequency transfer would be inappropriate
because the supplied rate changes from approximately `1.6` to `2.6 Hz`.
Instead, the `22 x 28` ROI was split into guarded temporal halves. Within each
discovery half, four-second blocks independently fitted a `1.5..4.0 Hz`
band-PCA loading. Pixels were scored by median absolute loading across blocks,
thresholded at a predeclared robust spatial `z >= 1.5`, and reduced to the
strongest 8-connected component of at least three pixels. Empty selection was
allowed. Each frozen mask and signed loading was then evaluated only on the
opposite temporal half.

The side-camera workbook was not loaded until both top-camera fold masks and
loadings were frozen. The result was:

```text
fold selected pixels:                 93 and 85 / 616
raw mask Jaccard:                         0.798
one-pixel-dilated overlap:                0.925
cross-fit ridge correlation:              0.921
cross-fit ridge MAE:                       3.49 bpm
early-half confirmation:        r=0.671, 4.92 bpm MAE
late-half confirmation:         r=0.957, 2.06 bpm MAE
held-out detected events:                         78
valid within-half event intervals:                76
event-interval rate MAE:                    4.61 bpm
event-interval CV:                            0.115
```

This is strong exploratory evidence that independently sampled portions of the
embedded top-camera recording identify nearly the same compact responsive
region and that a loading learned on one half tracks the side-derived rate on
the other. It strengthens the rationale for spatial mask discovery in the
moving fish. It is not a calibrated detection: the method was designed after
Trial 1 had already been examined, no complete mask-selection surrogate null
or familywise spatial correction was run, and only one positive-control trial
was tested. The selected pixels also remain a photometric support rather than
an anatomical chamber segmentation.

The exploratory compact-mask event detector also preserves the cross-fit
contract. Each fold learns event polarity and robust amplitude scale on its
discovery half, then applies fixed `0.5 MAD` prominence and band-derived spacing
to the opposite half. It rejects `0.75 s` at each confirmation edge, permits no
interval across the midpoint guard, and omits intervals outside `1.5..4.0 Hz`.
Both folds independently selected the same polarity. The resulting interval
rate is close to the supplied smoothed rate, but this remains rate-surrogate
agreement rather than one-to-one visible-contraction validation. The `78`
events cover less than the full 40-second clip because the midpoint guard and
four filter edges are intentionally excluded; they must not be compared as a
raw count with the full-clip `88.53` expected cycles.

### Frozen-mask projection comparison

The compact masks were then held fixed while three opposite-half projections
were compared:

1. the existing discovery-block PCA loading, zeroed outside the mask;
2. a new PCA refit using only frozen-mask pixels in the discovery half;
3. an equal-weight mean of the frozen-mask pixels after discovery scaling.

All weights were frozen before loading the side-derived reference. Held-out
results were:

| Projection | Ridge correlation | Ridge MAE | Events / valid intervals | Event-rate MAE | Interval CV |
|---|---:|---:|---:|---:|---:|
| Discovery loading | `0.921` | `3.49 bpm` | `78 / 76` | `4.61 bpm` | `0.1150` |
| Within-mask PCA refit | `0.921` | `3.49 bpm` | `78 / 76` | `4.49 bpm` | `0.1146` |
| Within-mask equal mean | `0.921` | `3.49 bpm` | `78 / 76` | `4.84 bpm` | `0.1150` |

Pairwise held-out waveform correlations are `0.9968..0.9998`. Thus, within
embedded Trial 1, hard spatial support accounts for nearly all of the useful
difference; precise weighting inside that support contributes little. The
within-mask PCA refit has a small descriptive event-rate advantage, but the
difference is too small and too post hoc to select it as a winner. The simple
equal mean is an important challenger because it is easier to interpret and
freeze. This conclusion is specific to this positive-control trial and does
not establish that a mean will work in the moving fish, where opposite pixel
polarities have been observed.

Artifacts:

```text
top_camera_crossfit_mask_projection_comparison.summary.json
top_camera_crossfit_mask_projection_comparison.arrays.npz
top_camera_crossfit_mask_projection_comparison.diagnostic.png
```

Artifacts:

```text
playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/
  top_camera_roi_crossfit_mask.summary.json
  top_camera_roi_crossfit_mask.arrays.npz
  top_camera_roi_crossfit_mask.diagnostic.png
  Trial1_top_camera_crossfit_mask_overlay_slow4x.mp4
  Trial1_top_camera_full_roi_pca_vs_crossfit_mask_side_by_side_slow4x.mp4
```

The overlay retains all `8001` source frames at `50 fps` (`4x` slow motion).
Early frames use fold 1, whose mask and signed loading were learned from the
late half; late frames use fold 0, learned from the early half. The one-second
guard on each side of the midpoint is intentionally blank. Blue/red mask
outlines identify fold ownership, while blue/red pixels in the dynamic panel
represent signed bandpassed change. The white rate curve is the side-derived
reference and the blue ridge is the held-out top-camera window estimate; these
rate curves evaluate the frozen masks and did not select them.
The orange step trace and bottom status value are the held-out compact-mask
event-interval rate. They are blank through the guard and whenever an interval
is missing or outside the predeclared band.

The synchronized side-by-side video compares the original same-clip PCA1 fit
over the complete `22 x 28` ROI with the compact opposite-half cross-fit mask.
The left dynamic panel displays all `616` ROI pixels, although PCA internally
uses continuous spatial weights and may downweight many of them. The right
panel imposes a hard support of `85` or `93` pixels, approximately `14--15%` of
the original box, and blanks the guarded midpoint. Therefore the comparison
tests whether compact support removes visually irrelevant spatial change; it
must not be interpreted as equal weighting on the left or as an independent
second biological measurement.

### Side-camera ROI reproduction

The colleague's fixed Trial 1 side-camera ROI `(307, 351, 127, 68)` was also
analyzed directly. The simple all-pixel ROI mean tracks the supplied rate trace
with ridge correlation `0.891`, ridge MAE `4.27 bpm`, and interval-rate MAE
`4.15 bpm` across `88` detected events. The ROI-to-surround ratio is similar
(`0.889`, `4.26 bpm`, and `4.63 bpm`). In contrast, side-camera band PCA 1 is
poor (`0.248` correlation and `24.64 bpm` ridge MAE).

This reproduces the supplied side-camera rate descriptively, but it is partly
circular because the reference appears to have been derived from the same
video and ROI. The workbook records `1688` "good pixels" for Trial 1 but does
not retain their coordinates or mask. Our reproduction and visualization use
all ROI pixels and must not be described as reconstructing that missing
selection or validating individual contractions.

Direct inspection of the raw side-camera video provides stronger biological
visibility than either top-camera view: two chamber-like structures visibly
contract and expand, and blood cells can be seen moving through the region.
This makes the side camera a credible source from which to create beat-level
cardiac reference annotations. It does not make the existing workbook trace or
our automatically detected peaks beat annotations retroactively. A defensible
beat-to-beat side-camera result still requires explicit contraction timestamps
or chamber-state labels, an unscorable-frame policy, and verification that one
reported event corresponds to one visible contraction.

The complete `3998`-frame inspection video is rendered at `25 fps` for `4x`
slow motion:

```text
embedded_positive_control/inspection_videos/
  Trial1_side_camera_colleague_roi_mean_overlay_slow4x.mp4
```

### Side-camera chamber polygons

Two visually distinct contracting structures were independently outlined on
side-camera frame 2000. Because the annotator is not an anatomical expert,
the saved source-pixel polygons remain neutrally labeled `chamber_a` and
`chamber_b`. They contain `2696` and `1298` pixels, do not overlap, and are
explicitly fixed ROIs rather than per-frame segmentations.

Mean-intensity analysis gives:

| Region | Ridge correlation | Ridge MAE | Events | Event-rate MAE | Interval CV | Peak/median |
|---|---:|---:|---:|---:|---:|---:|
| Chamber A | `0.893` | `4.28 bpm` | `89` | `5.20 bpm` | `0.126` | `33.5` |
| Chamber B | `0.895` | `3.99 bpm` | `88` | `2.78 bpm` | `0.117` | `96.0` |
| Chamber union | `0.893` | `4.19 bpm` | `88` | `3.23 bpm` | `0.115` | `69.2` |

Chamber B is the cleaner event surface. Its result is stable after a two-pixel
erosion or dilation: ridge MAE remains `3.94..3.95 bpm`, event-rate MAE remains
`2.91..3.04 bpm`, and all variants emit `88` events. This boundary sensitivity
reduces concern about the non-expert polygon placement.

Across overlapping eight-second windows, Chamber B follows Chamber A by a
median `80 ms` with an `80..90 ms` interquartile range. Median signed
correlation is `0.956`; `15/16` windows have positive lag and positive
correlation. One window near 24 seconds reverses, so a globally invariant
direction is not established. The stable dominant relationship is consistent
with coupled chambers, but chamber identity must not be assigned until an
expert reviews the anatomy.

These same-clip fixed-ROI means and automatically chosen event polarities are
still exploratory. Visible contraction annotation remains necessary before
calling individual events ground-truth beats.

Artifacts:

```text
side_camera_chamber_comparison.summary.json
side_camera_chamber_comparison.arrays.npz
side_camera_chamber_comparison.diagnostic.png
embedded_positive_control/inspection_videos/
  Trial1_side_camera_chambers_overlay_slow4x.mp4
```

The complete chamber overlay retains all `3998` source frames and plays them
at `25 fps`, or `4x` slow motion. It shows the native raw view and polygons,
the raw chamber crop, one uniform signed bandpassed-mean color per polygon,
the two mean waveforms, and the reference plus automatically detected
event-interval rates. Uniform polygon color is intentional: this visualization
tests chamber-level fixed-ROI photometry and does not encode per-pixel motion
or a traveling wave.

For this side-camera target, the polygon mean is the primary candidate
measurement. It is directly interpretable, remains stable under the
two-pixel erosion/dilation sensitivity check, and substantially outperforms
the side-camera band-PCA result reported above. PCA remains a useful challenger
for heterogeneous or oppositely signed pixel responses, but should not replace
the mean solely because it is more complex.

An oscillator trace and a beat series are related but not equivalent. The
bandpassed means demonstrate continuous periodic chamber-level intensity
variation, while beat-to-beat reporting requires one discrete detection per
visible contraction. Filtering can yield a plausible oscillation and average
frequency even when individual peaks are shifted, missed, or doubled. The
overlay is therefore an annotation aid and candidate-event diagnostic, not the
missing event-level reference.

### Reporting boundary

For the freely moving top-camera recording, do not target or report
beat-to-beat heart rate from the current analysis. Report window-level
candidate oscillator frequency, in cycles per minute, with validity coverage
and motion-control results. Do not relabel those values as heart rate until
cardiac attribution is independently established.

For the embedded side camera, beat-to-beat heart rate is a reasonable future
target because individual contractions and associated blood-cell motion are
visually observable in the raw video. Report beat-to-beat values only after
the visible events have been explicitly annotated or a chamber
segmentation/flow detector has been checked event-by-event against the raw
frames. The current dense smoothed rate trace supports rate comparison but is
not a substitute for that event reference.

### Stimulus-associated bradycardia recovery

The embedded positive-control endpoint includes a brief fearful-stimulus-
associated bradycardia, so whole-clip correlation and long-window rate error
are insufficient by themselves. Trial 1's supplied `Bradyinfo` row identifies
the stimulus-overlapping side bout as frames `3325..3394` at `100 fps`, or
`33.25..33.94 s`.

Against a declared `28..32 s` baseline and the plot-highlighted `32..36 s`
stimulus epoch, the side reference drops `45.2 bpm` from `151.2` to `106.0
bpm`. The frozen top equal-mask peak intervals drop `34.6 bpm`, from `150.0`
to `115.4 bpm`, recovering `76.6%` of the reference depth. The top response
span is `33.165..33.685 s`, begins `85 ms` before the supplied bout, and has
interval IoU `0.56`. The `2 s` spectral ridge recovers only `20.5 bpm`, or
`45.4%` of the reference depth.

This supports peak-interval response depth and timing as the relevant
high-resolution embedded endpoint, with a spectral ridge retained for robust
background rate. It remains descriptive until top peaks are checked against
visible side contractions and stimulus timing is confirmed from an
authoritative protocol source.

### Spatial controls

Each control is the same `22 x 28` size, shifted 40 pixels, and independently
fits the same reference-blind band PCA.

| Control | Peak | Peak/median | Ridge correlation | Ridge MAE |
|---|---:|---:|---:|---:|
| left | `180.0 bpm` | `2.20` | `0.127` | `39.18 bpm` |
| right | `172.5 bpm` | `1.87` | `-0.030` | `37.28 bpm` |
| anterior | `150.0 bpm` | `2.80` | `-0.408` | `33.75 bpm` |
| posterior | `112.5 bpm` | `3.61` | `-0.027` | `26.63 bpm` |

These controls support spatial specificity. They are not a calibrated maximum
statistic across every possible box location, size, transform, and band.

## Simple No-Fold Transfer To Moving-Fish Caches

The embedded analysis idea was transferred back to the cached moving-fish
photometry without the earlier discovery/confirmation machinery. This is an
exploratory descriptive analysis with:

```text
frequency band:                    1.5..4.0 Hz
spatial model:                     one robust bandpassed PCA component
folds:                             none
discovery/confirmation partitions: none
minimum valid segment:             2.0 s
maximum interpolated gap:          0.02 s
filter-edge rejection:             0.75 s at each segment end
```

Every qualifying segment contributes to one PCA fit. Long invalid gaps are
never bridged, time is never compressed, and event intervals never cross
segment boundaries. The `0.02 s` rule fills only bounded one- or two-frame
holes using the same cache policy as the earlier analysis.

### Five-minute frame-0 cache

Applying PCA to all 226 anatomically eligible administrative-ROI pixels does
not recover the earlier candidate. It selects a diffuse `1.75 Hz` (`105
cycles/min`) component, and the ten 30-second candidates range from `93` to
`171 cycles/min`. Only `26.8%` of its loading energy lies in the frozen
38-pixel mask. This is a negative result for treating the full administrative
ROI as equivalent to the embedded heart box.

Restricting the same method to the frozen heart supports changes the result:

| Support | Pixels | Common-valid duration | Post-edge samples | Aggregate peak | Peak/band median | PCA variance |
|---|---:|---:|---:|---:|---:|---:|
| frozen whole-heart | 38 | `239.07 s` | `165.07 s` | `3.10 Hz` (`186/min`) | `1.94` | `0.375` |
| recurrent core | 8 | `239.71 s` | `168.73 s` | `3.10 Hz` (`186/min`) | `2.23` | `0.770` |

The independence status differs between these rows. The 38-pixel mask was
defined from later chase frames, so it is temporally frozen relative to the
frame-0 cache. The 8-pixel intersection includes the consensus mask learned
from this same five-minute cache; its stronger PCA variance is therefore a
same-cache descriptive result and should not be treated as independent
confirmation.

At 60-second resolution:

```text
38-pixel mask:  99, 186, 186, 189, 183 cycles/min
8-pixel core:  99, 186, 183, 189, 195 cycles/min
```

Thus minutes 1--4 recur near `3.05..3.25 Hz`, while minute 0 instead selects
`1.65 Hz`. Because no fold is withheld, this disagreement is a real temporal
result rather than cross-validation coverage. The aggregate `3.10 Hz` peak
should not be described as a constant five-minute heart rate.

The independently fitted motion-prediction PCA peaks at `2.20 Hz` for the
38-pixel support and `3.35 Hz` for the 8-pixel support. Its power ratio at the
observed `3.10 Hz` candidate is `1.40` and `1.08`, respectively. Absolute
observed/motion-PCA correlations are `0.003` and `0.080`. These simple controls
do not reproduce the observed PCA waveform, but they are not a calibrated
motion null. Eye-mask area also has a weak `3.10..3.20 Hz` peak and remains a
possible shared-state or tracking covariate.

### Two chase-associated caches

Using the same frozen 38-pixel support:

| Source frames | Segments | Common-valid duration | Post-edge duration | Peak | Peak/band median | Median event-interval rate | Interval CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| `30000..32999` | 3 | `15.24 s` | `10.72 s` | `3.15 Hz` (`189/min`) | `5.61` | `187.5/min` | `0.256` |
| `33000..35999` | 3 | `16.46 s` | `11.94 s` | `3.35 Hz` (`201/min`) | `11.78` | `200.0/min` | `0.161` |

These no-fold frequencies closely match the earlier dynamic-support candidates
of `3.25` and `3.35 Hz`. The second interval is cleaner by peak contrast and
interval variability. The first interval retains meaningful nuisance power at
the candidate frequency, including body-control and axis-geometry traces, so
its separation from tracking remains weaker.

The first chase interval also created the 38-pixel support and is post hoc with
respect to that mask. The second interval received the support frozen from the
first interval and is the cleaner temporal transfer, although its PCA loading
and event polarity are still fitted within the second cache.

### Full-recording 2--4 Hz scan

The same no-fold method was applied to all `140035` frames (`23.34 min`) using
non-overlapping 60-second windows and the complete `2.0..4.0 Hz` grid in `0.05
Hz` steps. The 38-pixel support result is:

```text
qualifying common-valid segments:       147
common-valid duration:                681.41 s
post-filter-edge analyzed duration:   459.41 s
scorable windows:                       22 / 24
aggregate peak:                       3.10 Hz (186/min)
aggregate peak / band median:         2.20
PCA explained band variance:          0.315
median one-minute candidate:           195/min
one-minute candidate range:            150..219/min
windows in 3.0..3.6 Hz:                 19 / 22
search-boundary selections:              0 / 22
```

Minutes 10--11 are explicitly unscorable because they contain only 77 and 0
post-edge common-valid samples. The three scorable windows outside `3.0..3.6
Hz` select `2.50`, `2.75`, and `3.65 Hz`. Early windows generally select
`3.10..3.25 Hz`; later windows more often select `3.25..3.45 Hz`. This supports
a recurring, time-varying frequency neighborhood rather than one constant rate.

The 8-pixel intersection also selects an aggregate `3.10 Hz` peak and a median
`195/min`. Its per-window candidates agree with the 38-pixel support within
`0.10 Hz` in all 22 scorable windows and within `0.05 Hz` in 20. Both supports
place 19/22 windows in `3.0..3.6 Hz`. This is useful spatial sensitivity, but
not independent confirmation: the 8 pixels are a subset of the 38 and include
a consensus support learned from the same recording.

The motion comparison prevents a stronger conclusion. Independently fitting
PCA to the cached motion-prediction matrix also gives an aggregate `3.10 Hz`
peak. The observed and motion PCA waveforms have low absolute correlation
(`0.017`), and their one-minute frequencies differ by a median `0.325 Hz`;
only `4/22` windows agree within `0.10 Hz` and `8/22` within `0.20 Hz`.
Therefore motion does not reproduce the observed trajectory, but the shared
aggregate peak means frequency recurrence alone cannot establish cardiac
origin. Eye-mask area also has a weak `3.10 Hz` peak (`1.29x` band median).

The event detector finds a median within-segment interval rate of `193.5/min`
with interval CV `0.152`. This is not validated timing. It differs from the
aggregate spectral `186/min`, is fitted on the same cache, and has no visible
beat reference. No event count, interval rate, or apparent smoothness should be
reported as heart rate from this scan.

### Four- and eight-second candidate windows

A shorter-window check retained the same full-recording PCA fit, frozen
38-pixel support, `2..4 Hz` band, frequency grid, gap handling, and filter-edge
policy. Only the non-overlapping reporting window changed.

The `4 s` run yields `109/351` scorable bins, a median `189/min`, an
`177..198/min` interquartile range, and `67%` of scorable bins in
`180..216/min`. The `8 s` run yields `95/176` scorable bins, the same
`189/min` median, a `177..201/min` interquartile range, and `71%` in
`180..216/min`. Median peak-to-band contrast is `5.06` and `4.81`,
respectively. Both retain a median `0.40 Hz` separation from the independently
fitted motion-PCA window candidate.

The embedded positive control supports the feasibility of `2..4 s` rate
windows: its frozen equal-mask mean reaches correlation `0.959` and MAE `2.84
bpm` at `4 s`, compared with `0.921` and `3.49 bpm` at the historical `8 s`
setting. In the moving fish, the `4 s` trajectory is a plausible responsive
candidate but is not demonstrably more accurate. Only `31%` of its bins are
scorable, there is no cardiac reference, and short excursions remain
ambiguous. Retain a longer stability view and do not interpret short-window
variation as heart-rate variability.

### Equal means over the frozen full, upper, and lower supports

The previously declared 18-pixel upper and 20-pixel lower split was applied to
the full moving-fish cache using equal means of the same robustly normalized,
bandpassed pixels used by the 38-pixel PCA. Common-valid segments, gap policy,
filter edges, frequency grid, and `4 s`/`8 s` windows were identical.

The full 38-pixel mean retains the aggregate `3.10 Hz` peak but is weaker than
PCA: peak-to-band contrast is `1.68` versus `2.20`, interval CV is `0.172`
versus `0.152`, and only `43%` of scorable `4 s` windows fall in
`180..216/min`, versus `67%` for PCA. This supports the earlier concern that
averaging the complete support can cancel regional structure.

The lower 20-pixel mean is the strongest simple projection. It peaks at `3.10
Hz` with contrast `2.56`, interval CV `0.136`, and median `192/min`; `73%` of
its `4 s` and `85%` of its `8 s` candidates lie in `180..216/min`. Its waveform
correlation with PCA is `0.981`, and their candidate frequencies agree within
`0.10 Hz` in `90%` of `4 s` and `81%` of `8 s` windows. The matched lower
motion mean instead peaks at `2.50 Hz` and has absolute waveform correlation
`0.034` with the observed lower mean.

The upper and lower means correlate only `0.225`, with approximately `101
degrees` circular mean phase difference and low phase-locking value `0.312`.
This does not support a fixed propagation direction, but it explains why the
full equal mean is inferior. Because the regional split was formulated after
earlier inspection and there is no moving cardiac reference, the lower mean is
an exploratory frozen primary candidate rather than a validated heart-rate
trace.

Decision for subsequent moving-fish work: use the frozen lower 20-pixel equal
mean as the primary candidate projection. Keep masked PCA, the upper 18-pixel
mean, the full 38-pixel mean, and matched motion projections as required
controls. Apply one fixed method to every qualifying interval and emit no
estimate when lower-support gates fail; do not fall back to whichever control
looks cleaner in that window.

The upper region is not simply a polarity-inverted copy. It has a nearby
`3.15 Hz` aggregate oscillator, but a stable opposite-phase relationship would
require a phase near `180 degrees` and high locking. The observed `101 degrees`
and PLV `0.312` instead indicate a variable regional offset. This is consistent
with intermittent red/blue opposition in the videos but does not establish a
fixed traveling wave, chamber identity, or cardiac propagation direction.

Full-video artifacts:

```text
reliable_local_rostral_start_0_140035f_simple_segmented_pca_original38_60s_2p0_4p0.*
reliable_local_rostral_start_0_140035f_simple_segmented_pca_intersection8_60s_2p0_4p0.*
```

The full-recording inspection overlay is:

```text
reliable_local_rostral_start_0_140035f_simple_segmented_pca_original38_60s_2p0_4p0_full_overlay_stride3_25fps.mp4
```

It covers source frames `0..140034` by encoding every third source frame at
`25 fps`. Thus all `23.34 min` of source time are represented, playback is
`1.333x` slower than source time (`31.12 min` total), and this is not a
frame-complete export. The white timeline is the independently reported
one-minute candidate frequency; the orange timeline and waveform are the
within-segment exploratory event output. The local mask colors and waveform
are intentionally blank during invalid post-edge samples and during windows
that lack a reliable candidate. These displays remain diagnostic and do not
turn the candidate oscillator into a validated heart rate.

### Visual organization record

The full overlay does not look uniformly random or uniformly coherent. During
some intervals, pixels within the mask change with a visually clear and
repeatable spatial organization over successive cycles. During other
intervals, the same organization is difficult to see or appears absent. This
intermittent visual quality is compatible with several explanations that the
current display cannot distinguish:

- changing cardiac contrast or viewing geometry;
- tracking and stabilization quality;
- motion from adjacent anatomy;
- ordinary Mono8 noise and weak-signal filtering;
- a mixture of physiological signal and nuisance motion.

This does not contradict the negative result for a fixed traveling-wave
direction. A local spatial arrangement can repeat within an interval without
maintaining one propagation direction, phase delay, or coherence value across
the entire recording. Conversely, every narrow-band pixel can cycle through
red and blue, so color cycling without repeatable organization is not positive
evidence by itself.

Visual organization may be used to formulate a future quality metric, but not
to choose favorable windows retrospectively. Any scorable-window rule must be
frozen before rate inspection and should use independently measurable factors
such as valid-pixel coverage, keypoint confidence, stabilization residuals,
filter-edge distance, and predeclared spatial-coherence support. Results must
include all windows passing that rule and explicitly report no-estimate
coverage.

### Interpretation

This transfer is encouraging because a much simpler spatial method recovers
the same frequency neighborhood when restricted to the expert whole-heart mask
or recurrent core. It also demonstrates why the anatomical restriction
matters: the full 226-pixel ROI selects a different low-frequency component.

It remains an exploratory same-data fit. There is no cardiac reference, no
surrogate detection threshold, and no held-out test of PCA loadings or event
polarity. The reported peaks and intervals are candidate oscillator summaries,
not validated heart rates or beats.

To establish window-level rate trust rather than recurrence alone:

1. Freeze the PCA normalization, band, spatial support, window estimator, and
   quality/no-estimate rules before further outcome inspection.
2. Apply the frozen embedded method across positive-control Trials 2--48 to
   measure rate generalization and coverage when a cardiac rate is supplied.
3. Create blinded visible-contraction timestamps and unscorable intervals from
   the embedded side camera if beat-level validation is pursued there.
4. Learn moving-fish spatial weights on declared discovery segments and score
   window-level frequency on disjoint held-out segments. The visualization can
   still use all segments, but confirmation cannot.
5. Add frozen esophagus, eye, boundary, and offset-region controls plus a
   stronger optical-flow or static-resampling motion null.
6. Rerun the complete validity-aligned adaptive null before restoring any
   detection p-value.

The full-video scan satisfies a recurrence/stability diagnostic. Steps 2--6 are
what would support cardiac identity and trustworthy window-level rate
tracking. Beat timing remains a separate embedded-side-camera validation
target.

Artifacts use these prefixes:

```text
reliable_local_rostral_start_0_30000f_simple_segmented_pca
reliable_local_rostral_start_0_30000f_simple_segmented_pca_original38_60s
reliable_local_rostral_start_0_30000f_simple_segmented_pca_intersection8_60s
reliable_local_rostral_chase_start_30000_3000f_simple_segmented_pca_original38
reliable_local_rostral_chase_start_33000_3000f_simple_segmented_pca_original38
```

Each prefix has a JSON summary, window CSV, NPZ arrays, and diagnostic PNG.

The cleaner second chase interval also has a 4x slow-motion inspection video:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/
  inspection_videos/
  freely_moving_frames33000_35999_simple_pca_original38_slow4x.mp4
```

It contains all 3000 source frames at 25 fps. The source crop is enlarged 2x
with nearest-neighbor sampling, mapped analysis pixels are shown at their saved
per-frame source coordinates, and the raw canonical ROI remains unenhanced. The
red/blue panel uses one fixed signed band-change scale over all three valid
segments. The white timeline is the descriptive `3.35 Hz` candidate; orange is
the within-segment peak-interval rate. Invalid rows, long gaps, and filter edges
retain the source video but deliberately suppress analysis colors and the PCA
waveform.

## Inspection Video Contract

The Trial 1 inspection video is:

```text
embedded_fish_positive_control/inspection_videos/
  Trial1_top_camera_heart_overlay_slow4x.mp4
```

It contains all 8001 source frames encoded at 50 fps, giving 4x slow motion.
Its panels mean:

- full frame: original top-camera pixels with the fixed ROI outline;
- raw ROI: nearest-neighbor 10x enlargement, with no enhancement;
- red/blue ROI: fixed-scale signed bandpassed pixel changes, not Hilbert phase;
- white rate trace: supplied workbook reference;
- orange rate trace: intervals between detected PCA peaks;
- orange waveform: the band-limited PCA score, with polarity chosen internally.

The overlay is offline and noncausal. Zero-phase filtering uses past and future
frames, so it must not be interpreted as a real-time detector.

## What Is Established Now

The evidence supports all of the following statements:

- the stationary top-camera Trial 1 ROI contains a spatially structured
  oscillation that closely tracks the side-camera-derived rate reference;
- the result is strongly localized relative to four simple offset controls;
- uniform top-camera ROI averaging is inferior to a spatial component, while
  side-camera ROI averaging closely reproduces its same-source rate trace;
- the moving-fish recording contains a recurring local-rostral oscillator worth
  testing further;
- the moving-fish data do not currently support reported beat times or heart
  rate.

The evidence does not yet support:

- claiming the moving-fish oscillator is cardiac;
- transferring the embedded PCA accuracy to the moving-fish recording;
- treating every red/blue cycle in either visualization as a contraction;
- reporting Trial 1 as held-out validation;
- estimating sensitivity, specificity, or false-positive rate from one trial
  and four offset boxes;
- claiming the supplied dense rate trace is independent manual event ground
  truth.

## Decision After The Full Scan

The full scan changes the status from "promising selected clips" to "recurring
localized oscillator across much of one recording." It does not change the
biological claim: cardiac identity and beat timing remain unvalidated.

The embedded result does change the feasibility assessment. It shows that a
top-down monochrome camera can encode cardiac-rate information as a localized
spatial photometry pattern even when individual beats are not cleanly visible
or validated. Therefore the freely moving analysis does not need perfect
beat-to-beat peaks to be scientifically useful: stable held-out window-level
rate tracking is a plausible target. However, the stationary result does not
identify the moving-fish oscillator. Stabilization error, tracking loss,
adjacent esophageal motion, and other body motion remain credible explanations
for the moving result, especially where the motion control shares aggregate
frequency power or event intervals vary wildly.

For the freely moving example, the current positive interpretation is limited
to a candidate: the oscillator is in an expert-supported anatomical location,
recurs near `3.0..3.6 Hz` across many windows, and has a spatial polarity pattern
that is compatible with the stationary top-camera observation. The missing
evidence is independent cardiac attribution. A reliable moving-fish result
would require frozen discovery parameters, held-out window-level rate
stability, stronger noncardiac motion controls, and preferably synchronized or
blinded visible-contraction labels. Until then, report candidate cycles per
minute rather than heart rate.

Recommended work order:

| Priority | Work | Decision it enables |
|---:|---|---|
| 1 | Freeze the current code revision, masks, bands, PCA normalization, polarity rule, peak rule, and metrics. | Prevent additional outcome-guided tuning. |
| 2 | Annotate visible side-camera contractions and unscorable spans in Trial 1 without viewing algorithm events. | Create an actual beat-level cardiac reference rather than a smoothed rate surrogate. |
| 3 | Compare side-camera and synchronized top-camera events with the frozen detector. | Measure one-to-one event precision, recall, and timing error in the stationary control. |
| 4 | Run the frozen embedded rate method on positive-control Trials 2--48. | Measure rate generalization, coverage, and offset-control separation; add beat labels only for a declared subset if beat-level claims are needed. |
| 5 | Fit moving-fish spatial weights on declared discovery segments, score untouched windows, and add esophagus, eye, boundary, optical-flow, and static-resampling controls. | Test window-level stability while distinguishing cardiac deformation from tracking or adjacent anatomy. |
| 6 | Rerun the complete validity-aligned surrogate/null pipeline. | Restore calibrated detection statistics only after the implementation defect is corrected. |

Lower-value next actions are narrowing the band around `3.1 Hz`, selecting a
new mask because it looks cleaner, choosing only high-coverage minutes, or
adding more display enhancement. Those can improve appearance while increasing
selection bias; they do not answer whether the oscillator is cardiac.

Promotion of a freely moving rate measure requires both:

1. positive-control generalization across trials; and
2. independent cardiac attribution and held-out window-level stability in the
   moving recording, including explicit no-estimate coverage.

Promotion of beat-to-beat reporting additionally requires one-to-one agreement
with blinded visible side-camera contractions. That requirement currently
applies to the embedded side-camera target, not to the freely moving analysis.

## Next Frozen Evaluation

Before inspecting rate agreement in Trials 2--48:

1. Freeze the `1.5..4.0 Hz` band, pixel normalization, PCA construction,
   polarity rule, peak detector, and comparison metrics.
2. Decide how a Trial 1 box maps to the remaining top-camera videos without
   using their supplied rate traces. If placement varies, use image-only rigid
   registration or boxes drawn without outcome overlays.
3. Validate every video timebase, decoded frame count, duplicate fraction, and
   the two-top-frames-per-reference-sample mapping.
4. Apply the frozen algorithm to every trial, including trials that yield no
   estimate.
5. Report per-trial correlation, rate MAE, event-interval MAE, coverage, and
   offset-control separation, plus across-trial distributions.
6. Only after that evaluation decide whether the positive-control method should
   inform a new moving-fish challenger.

Separately, the moving-fish shared-surrogate analyses require full
validity-aligned recalibration before any p-value can be restored.

## Reproducible Artifacts

Implementation:

```text
playgrounds/heartrate_stabilization/select_fixed_video_roi_web.py
playgrounds/heartrate_stabilization/analyze_embedded_positive_control.py
playgrounds/heartrate_stabilization/render_embedded_positive_control_overlay.py
playgrounds/heartrate_stabilization/analyze_segmented_cache_pca.py
playgrounds/heartrate_stabilization/render_segmented_cache_pca_overlay.py
```

Trial 1 outputs:

```text
playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/
  top_camera_roi.summary.json
  top_camera_roi.summary.csv
  top_camera_roi.diagnostic.png
  Trial1_top_camera_heart_overlay_slow4x.mp4
  Trial1_top_camera_heart_overlay_slow4x.json
```

The playground output directory is ignored by git. The transfer copy of the
inspection video and its metadata live under the shared dataset's
`inspection_videos/` directory.

Full moving-fish outputs:

```text
playgrounds/heartrate_stabilization/outputs/
  reliable_local_rostral_start_0_140035f_simple_segmented_pca_original38_60s_2p0_4p0.summary.json
  reliable_local_rostral_start_0_140035f_simple_segmented_pca_original38_60s_2p0_4p0.windows.csv
  reliable_local_rostral_start_0_140035f_simple_segmented_pca_original38_60s_2p0_4p0.arrays.npz
  reliable_local_rostral_start_0_140035f_simple_segmented_pca_original38_60s_2p0_4p0.diagnostic.png
  reliable_local_rostral_start_0_140035f_simple_segmented_pca_intersection8_60s_2p0_4p0.summary.json
  reliable_local_rostral_start_0_140035f_simple_segmented_pca_intersection8_60s_2p0_4p0.windows.csv
  reliable_local_rostral_start_0_140035f_simple_segmented_pca_intersection8_60s_2p0_4p0.arrays.npz
  reliable_local_rostral_start_0_140035f_simple_segmented_pca_intersection8_60s_2p0_4p0.diagnostic.png
  reliable_local_rostral_start_0_140035f_lower20_equal_mean_4s_8s_full_overlay_stride3_25fps.mp4
  reliable_local_rostral_start_0_140035f_lower20_equal_mean_4s_8s_full_overlay_stride3_25fps.json
```

The lower-mask overlay uses the saved 20-pixel equal-mean projection and
colors the entire lower mask uniformly. It is the inspection surface for the
current primary moving-fish candidate. It must not be interpreted as showing
within-mask propagation, because no per-pixel phase is encoded. Its `4 s` and
`8 s` curves are window-level candidate-cycle summaries; the orange event
interval curve remains unvalidated as heartbeat timing.

An additional full-cache comparison formed the lower-mask trace by averaging
the literal Mono8 values before detrending and bandpass. That raw lower mean
was essentially identical to the per-pixel normalized lower mean (`r=0.9997`):
both selected `3.10 Hz` in aggregate and exactly the same frequency in every
paired scorable `4 s` and `8 s` window. Its peak-to-band ratio was `2.578`,
interval CV was `0.133`, and absolute matched-motion correlation was `0.034`.
This supports the simpler raw lower mean as the preferred transfer projection,
but it does not add an independent cardiac reference.

The raw lower mean's `4 s` excursions were compared with cached tracking
diagnostics. Fifteen of `109` scorable windows were at least `24/min` from the
`192/min` median. Their spectral contrast was substantially weaker than in 45
stable windows (`3.36` versus `7.10`). They also showed modestly larger source-
coordinate steps, gradient-displacement exposure, translation, and transform
uncertainty, plus lower minimum detection confidence. Direct motion-trace RMS
and observed/motion waveform correlation did not differ meaningfully. Tracking
degradation is therefore a plausible contributor, but short-window spectral
ambiguity is the strongest demonstrated association and neither explanation
accounts for every excursion.

The final moving-fish summary plot reports literal lower-mask candidate
frequency over recording time. It emphasizes unsmoothed `8 s` estimates,
retains `4 s` estimates as faint context, and never connects across unscorable
gaps. Both PNG and vector PDF copies are stored locally and in the shared
recording's `inspection_videos/` directory.
