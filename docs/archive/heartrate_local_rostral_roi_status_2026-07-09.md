# Heartrate Local Rostral ROI Status - 2026-07-09
<!-- contract-meta
status: draft
last_updated: 2026-07-11
-->

Purpose: record the current state of the heartrate stabilization playground
after adding subject-mask-relative ROI generation, live-vs-fixed mask
diagnostics, local rostral-segment alignment, and first-pass local ROI signal
comparison.

The current combined moving-fish and embedded positive-control summary is in
[`heartrate_analysis_status_2026-07-11.md`](heartrate_analysis_status_2026-07-11.md).

Current frozen-mask validation results are summarized in
[`heartrate_frozen_mask_validation_status_2026-07-10.md`](heartrate_frozen_mask_validation_status_2026-07-10.md).

Important: the 2026-07-11 audit withdrew p-values based on the shared
autocorrelation-preserving surrogate because trace shifts did not carry
per-pixel validity with them. The observed masks, frequencies, and plots remain
descriptive; they are not calibrated detections.

## Bottom Line

The local rostral coordinate model supports a reproducible periodic spatial
pattern in an expert-identified 38-pixel whole-heart candidate, but it does
**not yet support a validated beat-to-beat heart-rate series**.

While implementing the whole-mask analysis, a surrogate-generation defect was
found: short valid segments were always shifted by exactly half their length,
so the earlier 199-entry null arrays contained one repeated transformation.
After randomizing every short-segment shift and rerunning the full calibration,
the original compact-cluster result is:

```text
fold 0:
  selected frequency: 3.10 Hz
  discovery p:         0.760
  selected pixels:     33
  discovery detected:  no

fold 1:
  selected frequency: 2.95 Hz
  discovery p:         0.050
  confirmation p:      0.490
  selected pixels:     13
  control ratio:       2.58

cross-fit one-pixel-dilated spatial overlap: 0.485
cross-fit frequency difference:              0.150 Hz
final compact-cluster result:                no estimate
reason: discovery_not_significant_in_both_folds
event count:                                 0
coverage:                                    0
```

The corrected discovery and confirmation null arrays each contain 199 distinct
statistics. The earlier `p=0.005` values and `12/27` injection sensitivity count
are invalid and are superseded by the results in this document.

The expert-informed analysis treats the complete red/blue/magenta union as a
single anatomical support while allowing each pixel to have a fixed phase and
contrast polarity. On the original interval this post-hoc 38-pixel support has
a latent-pattern `p=0.010` at `3.25 Hz`, which is exploratory because the mask
and candidate-centered bounds came from the same interval.

The mask was then frozen and applied unchanged to frames `33000..35999`, with
the `3.0..3.5 Hz` range declared before inspecting that interval. Despite only
`2010/3000` geometrically valid frames, four held-out blocks survived. The
cross-fitted whole-mask pattern reports:

```text
selected frequency:              3.35 Hz (201 bpm candidate)
latent-pattern p:                0.010 (199 varied surrogates)
latent-pattern score:            0.650
held-out spatial alignment:      0.40, 0.52, 0.67, 0.86
strongest nuisance control:      body_control_mean
latent/control ratio:            3.51
same-sign whole-mask p:          0.815
confirmatory status:             periodic latent spatial pattern supported
validated cardiac event series: no
```

This is promising independent evidence for a periodic process in the proposed
heart region. It is not yet a reported heart rate because the esophagus has not
been separately annotated, the phase of a biological contraction has not been
matched to manual or external cardiac events, and beat-level extraction from
the latent trace has not been validated.

The earlier whole-ROI comparison remains a geometry/signal diagnostic only. Its
`36.0 bpm` fixed-ROI IQR used all 11 windows while the `11.0 bpm` local-ROI IQR
used only the six windows that survived anchor gating. On those same six paired
windows, the fixed ROI IQR is `4.28 bpm` and the local ROI IQR is `10.95 bpm`.
The earlier statement that local correction narrowed the estimate spread was
therefore not supported by a paired comparison.

## Pixel Contract

All quantitative measurements in this status report use the original acquisition
crop-video pixels as the intensity source.

The stabilized and locally corrected views are diagnostic and coordinate-mapping
surfaces. They are not the photometry pixel source.

The current comparison uses:

```text
current fixed ROI:
  fixed canonical ROI
  -> inverse keypoint body transform into source crop frame
  -> intersect source-frame subject_body
  -> exclude source-frame eye_left/eye_right dilation
  -> average original crop-video pixels

gated local rostral ROI:
  fixed canonical ROI
  -> inverse local rostral correction into the frame's current stable coordinates
  -> inverse keypoint body transform into source crop frame
  -> intersect source-frame subject_body
  -> exclude source-frame eye_left/eye_right dilation
  -> average original crop-video pixels
```

The local correction changes which source pixels are sampled. It does not sample
from a warped rendered video.

## Inputs

The checked recording is:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording
```

Key configured inputs:

```text
crop video:
  derived/external_crop_recorder/Cam2010096_2026-06-14T21-12-08Z_arena_4_crop_external.mp4

crop metadata:
  derived/external_crop_recorder/Cam2010096_2026-06-14T21-12-08Z_arena_4_crop_meta.csv

analysis zarr:
  zarr/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop_analysis.zarr

keypoints:
  refined_keypoints_runs/refined_keypoints_2026-06-18_15-00-03
```

The frame-domain join for this example is:

```toml
frame_id_column = "crop_video_frame_index"
```

The keypoint `frame_indices` align to crop-video row/frame index, not to
`camera_frame_id`.

## Current Candidate ROI

The current candidate ROI was generated from projected subject masks in canonical
stabilized coordinates:

```text
playgrounds/heartrate_stabilization/outputs/
  mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.roi.json
  mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.npz
```

Current geometry:

```text
ROI bbox xyxy: [118, 116, 138, 128]
ROI mask pixels: 235
center mode: eye-mask midpoint
width x height: 20 x 12 px
```

This ROI is intentionally rostral and narrow. It is anchored below the eyes and
above/around the rostral swim-bladder region rather than using the tail or whole
body centroid.

## Why Whole-Fish Alignment Was Not Enough

The worst fixed-mask alignment frames were unsurprisingly associated with body
bending and turning. Whole-body IoU is therefore a poor sole discriminator for
this measurement. The analysis region is only the segment from the lower eye
boundary toward the rostral swim-bladder area. Tail bend should not pull that
ROI around.

The live-vs-fixed diagnostic clip around the worst stability windows showed:

```text
selected frames: 363
live failures: 0
ROI live usable fraction:
  min: 0.515
  median: 0.987
  max: 1.000
body centroid shift:
  min: 1.05 px
  median: 3.14 px
  max: 19.39 px
eye union centroid shift:
  min: 0.16 px
  median: 1.69 px
  max: 7.73 px
```

Important artifact:

```text
playgrounds/heartrate_stabilization/outputs/
  live_vs_fixed_mask_overlay_worst_frames_eye_mid_w20_h12.mp4
```

That video showed that the fixed ROI/mask geometry was usually stable, but bad
frames coincided with live-vs-fixed mask/anchor divergence.

## Local Rostral Correction

The local correction is a second rigid transform in stabilized coordinates.

It uses only local rostral anchors:

```text
anterior anchor:
  midpoint of the two individual eye-component bottom anchors

posterior anchor:
  rostral/top swim-bladder anchor

source axis:
  framewise live posterior anchor -> framewise live anterior anchor

target axis:
  fixed/reference posterior anchor -> fixed/reference anterior anchor
```

The transform is translate plus rotate only by default. Scale is disabled unless
explicitly requested. Nonrigid warping is not used.

The diagnostic draws two local axes:

```text
white axis:
  fixed/reference local axis

green axis:
  live frame's local axis
```

In the locally corrected panel, the green axis should lie close to the white
axis. Frames requiring an extreme correction are rejected rather than forced
into place.

Current correction gates:

```text
max local rotation: 50 deg
max local translation: 150 px
```

These gates are deliberately conservative. A frame that needs a larger
correction is treated as a tracking/mask-quality failure, not a frame to warp
into analysis.

## Local Alignment Diagnostic Result

The key diagnostic artifact is:

```text
playgrounds/heartrate_stabilization/outputs/
  local_rostral_alignment_comparison_component_eye_anchor_gated_worst_frames_eye_mid_w20_h12.mp4
  local_rostral_alignment_comparison_component_eye_anchor_gated_worst_frames_eye_mid_w20_h12.key_frames.png
  local_rostral_alignment_comparison_component_eye_anchor_gated_worst_frames_eye_mid_w20_h12.summary.json
  local_rostral_alignment_comparison_component_eye_anchor_gated_worst_frames_eye_mid_w20_h12.csv
```

The gated component-eye-anchor comparison rendered:

```text
selected frames: 363
rendered frames: 363
live failures: 3
correction failures: 3
```

The accepted local corrections improved the worst ROI geometry substantially:

```text
frame 30700:
  current usable ROI: 0.515
  local usable ROI:   0.983

frame 30701:
  current usable ROI: 0.549
  local usable ROI:   0.970

frame 31135:
  current usable ROI: 0.660
  local usable ROI:   0.996

frame 31440:
  current usable ROI: 0.791
  local usable ROI:   1.000
```

Three frames were rejected by the rotation gate:

```text
31400
31402
31425
```

Those frames required local rotations around `54..91 deg` and translations over
`180 px`. They are better interpreted as mask/anchor failures than as frames to
force into alignment.

## Signal Comparison Result

The current signal comparison artifact set is:

```text
playgrounds/heartrate_stabilization/outputs/
  local_roi_signal_compare_chase_start_30000_3000f_1p5_3p5hz_5s_windows.png
  local_roi_signal_compare_chase_start_30000_3000f_1p5_3p5hz_5s_windows.samples.csv
  local_roi_signal_compare_chase_start_30000_3000f_1p5_3p5hz_5s_windows.summary.csv
  local_roi_signal_compare_chase_start_30000_3000f_1p5_3p5hz_5s_windows.summary.json
```

Run parameters:

```text
frame_start: 30000
frame_count: 3000
fps: 100
band: 1.5..3.5 Hz
window_seconds: 5.0
window_step_seconds: 2.5
primary estimator: autocorr
```

Frame validity:

```text
current fixed ROI:
  valid frames: 3000 / 3000

gated local ROI:
  valid frames: 2672 / 3000
  valid fraction: 0.891
```

Local correction status:

```text
ok: 2672
swim:projected_mask_outside_crop: 222
rejected_rotation_limit: 88
missing_live_anchor: 18
```

The local correction diagnostics over frames with finite local details were:

```text
local rotation:
  min: -45.58 deg
  median: 0.50 deg
  p95: 14.13 deg
  max: 136.75 deg

local translation:
  min: 0.0 px
  median: 5.92 px
  p95: 63.34 px
  max: 306.96 px
```

The extreme maxima are rejected by the local gates and are not used in the local
signal.

Windowed autocorrelation estimates:

```text
current fixed ROI:
  windows: 11
  ok windows: 11
  median peak bpm: 144.9
  IQR peak bpm: 36.0
  min..max peak bpm: 90.3..164.6
  median peak score: 0.397

gated local ROI:
  windows: 11
  ok windows: 6
  median peak bpm: 137.9
  IQR peak bpm: 11.0
  min..max peak bpm: 120.2..148.3
  median peak score: 0.319
```

Interpretation:

- The unpaired aggregate makes the local spread appear smaller, but the paired
  six-window comparison does not show an improvement (`4.28 bpm` fixed versus
  `10.95 bpm` local IQR).
- The local ROI has fewer valid windows because it fails closed when the
  swim-bladder anchor is unavailable or implausible.
- The local ROI does not yet increase the rhythm score.
- Autocorrelation and spectral estimators disagree substantially in several
  local windows; a finite in-band peak is not a detection decision.
- The fixed ROI produces more continuous estimates, but its late-window values
  move down toward `90..118 bpm`, which may reflect motion/geometry artifacts or
  broad intensity structure rather than a stable beat signal.

## What This Does Not Prove

The current local ROI result does not prove that the extracted signal is a true
heartbeat series.

Reasons:

- Whole-ROI means can dilute a small localized signal.
- Broad intensity sweeps remain visible in the raw traces.
- The local method has anchor-dependent gaps.
- The local autocorrelation score is lower than the fixed ROI score in the
  current run.
- A stable ROI is necessary for trustworthy photometry, but it is not sufficient
  to prove the sampled pixels contain a coherent rhythm.

The correct next question is not "what rate does the whole local ROI produce?"
It is:

```text
Within the locally stabilized ROI, is there a compact group of pixels whose
1.5..3.5 Hz evidence is stable across chunks and not explained by eye edges,
mask boundaries, global brightness, or local correction failures?
```

## Reliable Analysis Implementation

The recommended local-coordinate analysis is now implemented in:

```text
src/fisheye/analysis/local_rostral_heartrate.py
playgrounds/heartrate_stabilization/extract_reliable_local_rostral_heartrate.py
```

Checked extraction summary:

```text
timestamp source: timestamp (nanoseconds)
effective fps: 99.99872
timestamp jitter p99: 0.000000256 s
canonical pixels: 235
valid frames: 2564 / 3000 (0.855)
reliable-extractor gates: 25 deg, 40 px
failure counts:
  swim mask outside crop: 222
  rejected rotation:      126
  rejected translation:    70
  missing live anchor:      18
physically eligible pixels: 228 / 235
bounded invalid rows bridged for analysis: 100
usable disjoint blocks: 4 (2 per fold)
```

The stricter `25 deg` / `40 px` limits apply to the reliable extractor. The
earlier `50 deg` / `150 px` diagnostic limits remain recorded above for
historical comparison and visualization only.

The implementation follows these decision stages:

1. Validate acquisition timestamps, monotonic frame order, decoded duplicates,
   blank frames, and Nyquist support.
2. Map every canonical ROI pixel through the local and body inverse transforms,
   then bilinearly sample the original crop frame.
3. Persist the full time grid, invalid-row mask, source `xy`, all four bilinear
   weights, body/eye occupancy, gradient magnitude, transform uncertainty, and
   named nuisance covariates.
4. Keep administrative ROI distance as a reported diagnostic only. Eligibility
   and penalties use physical body/eye boundaries, warp validity, image
   gradients, and transform sensitivity.
5. Interpolate only bounded gaps of at most `0.02 s`; long gaps remain separate
   segments. Four usable contiguous blocks are alternated between two folds,
   with guard intervals and two blocks required per fold.
6. Fit nuisance coefficients on discovery blocks only. Frequency, pixel scores,
   connected component, weights, and polarity are also frozen from discovery.
7. Calibrate maximum cluster mass by rerunning nuisance fitting, frequency
   search, pixel scoring, and cluster selection on spatial-block circular-shift
   surrogates that preserve real per-pixel autocorrelation.
8. Apply the frozen source to held-out blocks and require it to beat its matched
   surrogate null plus interior, physical-boundary, body, global, and external
   controls.
9. Reverse discovery/confirmation roles. Both folds must confirm, their selected
   frequencies must agree within `0.10 Hz`, and their one-pixel-dilated cluster
   support must overlap by at least `0.50`.
10. Extract events only inside held-out blocks that survive a maximum-statistic
    correction across confirmation blocks. Filtering is segment-local,
    filter-edge events are rejected, and all other intervals are emitted as
    explicit no-estimate coverage.

The selection threshold may produce an exploratory component, but it cannot
produce a reported signal unless the complete discovery, held-out, control, and
cross-fit gates pass. Empty/no-estimate output is an expected result.

## Dynamic Whole-Mask Analysis

The expert interpretation is implemented separately in:

```text
src/fisheye/analysis/dynamic_heart_support.py
```

It does not relax the compact-cluster gate. Instead it:

1. Treats the full fold-union or an externally supplied anatomical mask as one
   fixed support.
2. Fits nuisance regression and per-pixel complex phase/polarity loadings on
   discovery blocks only.
3. Freezes those loadings and measures how well the same spatial contraction
   pattern aligns on the opposite held-out blocks.
4. Reverses the folds and searches only the declared frequency range.
5. Reruns nuisance fitting, pixel-loading estimation, and the frequency search
   for every spatial-block circular-shift surrogate.
6. Keeps core, fold-exclusive, anatomical-only, and optional esophagus pixels
   separate in the diagnostic and block table.

This model permits one portion of the visible structure to brighten while
another darkens or lags in phase. It therefore tests the proposed whole-heart
motion pattern without relying on a same-sign mean, which strongly attenuates
the checked signal. A mask derived from the current clusters is always labeled
post hoc; confirmatory status requires an externally supplied mask declared
independent plus a preconfigured full band or explicit frequency bounds.

## Frame-Resolved Phase Diagnostic

The held-out phase visualization is implemented in:

```text
playgrounds/heartrate_stabilization/render_dynamic_heart_phase.py
```

Add these options to the frozen-mask confirmation command:

```text
--render-dynamic-phase
--dynamic-phase-frame-stride 3
--dynamic-phase-playback-fps 30
```

The renderer writes:

```text
*.dynamic_phase.mp4
*.dynamic_phase.strip.png
*.dynamic_phase.arrays.npz
*.dynamic_phase.summary.json
```

The video has three synchronized views: fixed-scale local stabilized intensity
with frozen support contours, held-out band-limited activation, and observed
per-pixel analytic phase. The trace below them aligns each pixel by the complex
loading learned on the opposite cross-fit partition. The static strip shows
both observed phase and phase after subtracting that frozen loading.

This is not an event detector. The selected `3.35 Hz` frequency came from the
dynamic-support search, the visualization does not identify which phase is a
biological contraction, and it does not fill partition guards, long invalid
gaps, or the `0.75 s` edge at each filtered block. For the checked next interval:

```text
source frames:                 3000
phase-valid frames:             751 (25.0%)
median spatial alignment:     0.660
rendered frames:               1000 at stride 3
output duration:              33.33 s at 30 fps
```

The original exploratory interval was also rendered with its post-hoc fold
union. It has `766/3000` phase-valid frames (`25.5%`) and median spatial
alignment `0.758` at the selected `3.25 Hz`. Those numbers are descriptive,
because both its support and frequency were selected from that interval.

The gaps are part of the result. They prevent the narrow-band display from
implying continuous evidence during rows that were not evaluated out of sample.
Within valid blocks, smooth phase stripes are expected from any narrow-band
analytic signal. Their visual presence is not additional detection evidence;
the relevant questions are whether the spatial relationship is stable out of
sample and whether future inferred events agree with blinded biological marks.

## Upper-to-Lower Delay Measurement

The apparent flow was quantified separately in:

```text
src/fisheye/analysis/regional_phase_delay.py
playgrounds/heartrate_stabilization/render_regional_phase_delay.py
```

The default split is independent of pixel intensity: it chooses the horizontal
boundary that balances the frozen support. Canonical `y=121.5` produces an
18-pixel upper region and 20-pixel lower region. Per-pixel amplitude weights are
learned on the opposite cross-fit partition, but their phase is not removed;
the regional phase difference therefore remains measurable. The sign contract
is explicit: positive lag means the lower region reaches the same phase after
the upper region.

For each valid block the analysis reports:

1. upper and lower complex analytic traces;
2. instantaneous phase offset and lower-region lag in milliseconds;
3. within-block phase-locking value and spatial coherence;
4. paired upper/lower zero-phase crossings for cycle-level lag;
5. block mean lag and circular dispersion.

Across blocks it tests the product of median within-block phase locking and
across-block phase locking against independent random phase rotations of the
lower-region blocks. This is a conditional null: it holds the mask, selected
frequency, filtering, and split fixed. It is not a new end-to-end detection
test.

The exploratory source interval result at `3.25 Hz` is:

```text
block mean lower lags:          -99, -38, -138, -60 ms
within-block PLV:             0.935 to 1.000
across-block PLV:             0.720
stable-delay conditional p:   0.14
paired cycles:                21
```

All four source blocks have the same sign, matching the visible directional
flow, but its magnitude changes substantially and the across-block test is not
significant.

The next frozen-mask interval at `3.35 Hz` is:

```text
block mean lower lags:         -119, +43, +148, -39 ms
within-block PLV:             0.976 to 0.997
across-block PLV:             0.180
stable-delay conditional p:   0.86
paired cycles:                22
```

This explains why the animation can look like a clean flow over several cycles:
the delay is very stable within each short block. It does not preserve direction
or magnitude between blocks, so it cannot currently serve as a heartbeat-event
signature. This does not disprove a cardiac source; it rejects the stronger
claim that these intervals contain one fixed upper-to-lower propagation delay.

Run it by adding to either dynamic-support command:

```text
--analyze-regional-phase-delay
--regional-phase-surrogate-count 199
```

The output arrays include canonical `upper_mask` and `lower_mask`. Because this
hypothesis was formulated after viewing the current intervals, both current
delay analyses remain exploratory. Freeze those masks before examining a third
interval and supply them with:

```text
--regional-phase-regions-npz <prior.regional_phase_delay.arrays.npz>
--regional-phase-upper-key upper_mask
--regional-phase-lower-key lower_mask
--regional-phase-regions-independent
```

## Five-Minute Frame-0 Cache

The recording has `140035` frames at `100 fps` (`1400.35 s`). Frames
`0..29999` form a continuous five-minute interval immediately before the
original frame-30000 analysis. The stabilization status is valid for
`29990/30000` of those frames.

The original per-frame extraction was I/O-bound. Each component mask lives in a
dense Zarr array with physical chunks `(256, 1, 512, 512)`, but the old loop
requested one row at a time. It therefore repeatedly decompressed the same
64 MiB component chunk. The optimized path adds:

1. an aligned 256-row mask-store cache per component;
2. fixed checked reference anchors (`anterior=(128,113)`,
   `posterior=(127,143)`), avoiding repeated 300-frame anchor scans;
3. ten resumable 3000-frame parts with four concurrent readers;
4. one-frame overlaps so cross-frame motion and duplicate checks remain exact;
5. a fail-closed merge that validates static pixel, mask, schema, anchor, and
   metadata contracts and rebuilds timestamps from crop metadata.

A 300-frame cached/uncached comparison was numerically identical for every
persisted array. Runtime decreased from `73.5 s` to `7.6 s` (`9.7x`). Across
the merged cache, each mask component recorded 29872 cache hits and 127 misses.

The authoritative cache is:

```text
playgrounds/heartrate_stabilization/outputs/
  reliable_local_rostral_start_0_30000f.local_pixel_matrix.npz
```

Its contract is:

```text
frame range:             0..29999, step 1
elapsed time:            299.99 s
timestamp source:        acquisition nanosecond timestamps
local valid frames:      26959 / 30000 (89.86%)
canonical ROI pixels:    235
```

The frozen 38-pixel support, explicit `3.0..3.5 Hz` search, and frozen 18/20
upper/lower masks were all defined from later frames and applied without
retuning. With 199 full-pipeline surrogates, the early interval result is:

```text
selected frequency:                 3.20 Hz (192 bpm candidate)
support p:                          0.005
shared-phase p:                     0.005
latent-pattern p:                   0.005
latent/control ratio:               4.61
dynamic held-out blocks:            59
phase-valid frames:                 10510 / 30000 (35.03%)
median framewise spatial alignment: 0.702
```

The compact adaptive-cluster method still returns no estimate because only one
discovery fold passes its calibrated gate and cross-fold overlap is `0.367`.
That does not negate the separately prespecified whole-mask test.

The frozen regional-delay result is:

```text
valid regional blocks:              55
paired cycles:                      306
median within-block PLV:            0.979
across-block PLV:                   0.592
conditional block-phase p:          0.001
mean lower lag:                    -103 ms (lower leads upper)
same leading direction:             47 / 55 blocks
median paired-cycle lag:            -88.6 ms
paired-cycle lag MAD:               34.2 ms
```

This early, higher-coverage interval is substantially more stable than the two
chase-associated 30-second intervals. That supports the hypothesis that chase
motion, pose, or tracking reliability contributed to their changing regional
delay. More importantly, the independent early interval provides strong
evidence for a reproducible periodic anatomical source near `3.2 Hz`.

It still does not prove heart rate. The analysis has no separately drawn
esophagus control and no blinded visible-contraction timestamps, and the
narrow-band phase display cannot identify biological contraction phase. The
next promotion gate remains event-level agreement against blinded annotations.

## Injection Recovery

The injection harness creates a no-signal background by spatial-block shifting
the real source-pixel matrix, splats a synthetic compact oscillator onto each
saved source crop raster, and samples it back through the exact stored bilinear
weights. Tracking and masks remain frozen so injected luminance cannot alter the
geometry estimator.

The coverage design spans:

```text
amplitude:      0, 1, 2, 4 robust-noise sigma
frequency:      1.8, 2.5, 3.2 Hz
source radius:  1, 3 px
phase drift:    0, 0.02 Hz/s
activity:       full interval, first half only
location:       ROI center, physical-mask edge
replicates:     3 independently shifted real-noise backgrounds
```

After repairing short-segment surrogate randomization, the final 30-case
coverage run had `0/3` null detections and `9/27` positive detections. Injection
sigma is based on the median robust frame-difference noise, which is `3.14`
intensity units in this matrix. Baseline compact `1 sigma` sources were rejected
`0/3`; baseline `2 sigma` sources (`6.29` intensity units) were detected `2/3`.
The `4 sigma` center baseline was detected `2/3`, the physical-edge case `3/3`,
the `3.2 Hz` case `1/3`, and the broad `3 px` case `1/3`. Low-frequency,
half-interval, and `0.02 Hz/s` drift cases were rejected `0/3`.

Across the nine confirmed positive cases, median frequency bias was `0.025 Hz`,
median localization error was `0.66 px`, median matched-event timing RMSE was
`0.015 s`, median precision was `1.000`, median recall was `0.947`, and median
coverage was `0.181`. No tested amplitude reached 80% detection after the null
repair. These are useful pipeline checks, not a production
operating curve: there are only three null replicates, factor coverage is
intentionally unequal across amplitudes, the masks/transforms are frozen, and
source-detection sensitivity must remain separate from event accuracy.

## Implementation Notes

Scripts added in the playground:

```text
playgrounds/heartrate_stabilization/render_live_vs_fixed_mask_overlay_video.py
playgrounds/heartrate_stabilization/render_local_rostral_alignment_comparison.py
playgrounds/heartrate_stabilization/measure_local_roi_signal_compare.py
playgrounds/heartrate_stabilization/extract_reliable_local_rostral_heartrate.py
```

Importable inference/statistics core and deterministic tests:

```text
src/fisheye/analysis/local_rostral_heartrate.py
src/fisheye/analysis/dynamic_heart_support.py
tests/unit/fisheye/test_local_rostral_heartrate.py
```

The current full 3000-frame signal comparison is slow because it projects
multiple zarr-backed masks per frame. The next engineering improvement should be
to cache or batch-read the projected masks for the target interval before doing
per-pixel local-coordinate sampling.

The reliable extractor writes progress every 250 frames and persists a compressed
local pixel matrix. Reuse that matrix with `--dataset-npz` for all surrogate,
parameter-audit, and injection runs. Do not repeat mask projection for those
analyses or scale the extraction path to the whole recording without batched
mask reads or a chunked on-disk matrix.

## Reproduction Commands

Render the local rostral alignment comparison:

```bash
scripts/py playgrounds/heartrate_stabilization/render_local_rostral_alignment_comparison.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --roi-json playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.roi.json \
  --mask-npz playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.npz \
  --center-frames 30700,31190,31440 \
  --context-frames 60 \
  --stride 1 \
  --playback-fps 30 \
  --panel-size 384 \
  --output playgrounds/heartrate_stabilization/outputs/local_rostral_alignment_comparison_component_eye_anchor_gated_worst_frames_eye_mid_w20_h12.mp4
```

Run the corrected 5-second-window signal comparison:

```bash
scripts/py playgrounds/heartrate_stabilization/measure_local_roi_signal_compare.py \
  --roi-json playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.roi.json \
  --mask-npz playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.npz \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --frame-start 30000 \
  --frame-count 3000 \
  --window-seconds 5 \
  --window-step-seconds 2.5 \
  --primary-estimator autocorr \
  --output-prefix playgrounds/heartrate_stabilization/outputs/local_roi_signal_compare_chase_start_30000_3000f_1p5_3p5hz_5s_windows
```

Extract the timestamped local-coordinate source-pixel matrix once:

```bash
scripts/py playgrounds/heartrate_stabilization/extract_reliable_local_rostral_heartrate.py \
  --roi-json playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.roi.json \
  --mask-npz playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.npz \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --frame-start 30000 \
  --frame-count 3000 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f \
  --extract-only
```

Run the confirmatory analysis from the cached matrix:

```bash
scripts/py playgrounds/heartrate_stabilization/extract_reliable_local_rostral_heartrate.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f.local_pixel_matrix.npz \
  --surrogate-count 199 \
  --alpha 0.05 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_confirmatory
```

Run the exploratory whole-fold-union analysis on the source interval:

```bash
scripts/py playgrounds/heartrate_stabilization/extract_reliable_local_rostral_heartrate.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f.local_pixel_matrix.npz \
  --surrogate-count 199 \
  --alpha 0.05 \
  --analyze-dynamic-support \
  --dynamic-support-surrogate-count 199 \
  --dynamic-support-frequency-margin-hz 0.25 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support
```

Extract the next interval with the same ROI/mask/status inputs shown above,
changing `--frame-start` to `33000` and the output prefix to
`reliable_local_rostral_chase_start_33000_3000f`, then confirm the frozen
38-pixel mask:

```bash
scripts/py playgrounds/heartrate_stabilization/extract_reliable_local_rostral_heartrate.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_33000_3000f.local_pixel_matrix.npz \
  --surrogate-count 199 \
  --alpha 0.05 \
  --analyze-dynamic-support \
  --dynamic-heart-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.dynamic_support.arrays.npz \
  --dynamic-heart-mask-key heart_support_mask \
  --dynamic-support-mask-independent \
  --dynamic-support-frequency-min-hz 3.0 \
  --dynamic-support-frequency-max-hz 3.5 \
  --dynamic-support-surrogate-count 199 \
  --render-dynamic-phase \
  --dynamic-phase-frame-stride 3 \
  --dynamic-phase-playback-fps 30 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_33000_3000f_frozen_union_confirmatory
```

Run the real-noise injection coverage design:

```bash
scripts/py playgrounds/heartrate_stabilization/extract_reliable_local_rostral_heartrate.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f.local_pixel_matrix.npz \
  --surrogate-count 39 \
  --alpha 0.05 \
  --run-injection-study \
  --injection-design coverage \
  --injection-amplitudes 0,1,2,4 \
  --injection-frequencies-hz 1.8,2.5,3.2 \
  --injection-radii-px 1,3 \
  --injection-phase-drifts-hz-per-s 0,0.02 \
  --injection-active-fractions 1,0.5 \
  --injection-locations both \
  --injection-replicates 3 \
  --injection-surrogate-count 39 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_injection_coverage
```

## Current Recommendation

Keep the local rostral coordinate sampler and frozen 38-pixel whole-mask model as
the leading candidate. The same-sign support statistic is significant in the
five-minute early interval but not in the later confirmation interval, so do not
replace the spatial latent model with a simple mean. Do not promote either
fold-specific component or a beat-to-beat event series.

The independent 30-second interval supports a latent pattern at `3.35 Hz`, and
the retrospective five-minute interval independently supports the complete
frozen hypothesis at `3.20 Hz` with `p=0.005`. Report this as strong evidence for
a reproducible periodic anatomical source near `3.2..3.35 Hz`, not yet as
validated heart rate. The next required step is event-level validation: define a
visible contraction landmark, collect blinded manual timestamps, and score
one-to-one event timing against the persisted latent trace and valid-block mask.
Add an explicit esophagus mask/control and calibrate the latent detector through
real-noise injection before defining production operating points.
