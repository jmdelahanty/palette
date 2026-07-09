# Heartrate Stabilization Playground

Purpose: prototype fish-attached ROI measurements for the heartrate example
recording without turning the first experiments into production Palette
analysis surfaces.

Top-level case note: [`docs/heartrate_stabilized_roi_case.md`](../../docs/heartrate_stabilized_roi_case.md).

The target problem is that the online cropped video is centered from a
framewise detector bbox. When the fish rotates, the bbox center can move even
if the anatomical heart region has not moved in fish coordinates. A fixed ROI
drawn on the crop video can therefore measure translation/rotation artifacts
instead of heartbeat intensity.

This playground treats the lossless cropped video as the first pixel source.
Refined keypoints from the Palette zarr are used only to derive a fish-attached
body transform. The measurement path samples pixels from the cropped video
frame, not from a pre-rendered stabilized movie.

## Example Source

The checked-in example config points at:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording
```

Relevant inputs:

```text
derived/external_crop_recorder/*_crop_external.mp4
derived/external_crop_recorder/*_crop_meta.csv
zarr/*_analysis.zarr/refined_keypoints_runs/refined_keypoints_2026-06-18_15-00-03
```

The example crop video is 256 x 256 at 100 fps. The raw camera video is 4512 x
4512 at the same frame count, but this playground starts with cropped-video
pixels to minimize bookkeeping.

## Coordinate Idea

For each crop-video frame:

1. Find the crop metadata row and resolve its configured frame domain.
2. Find the matching refined-keypoint row in the zarr.
3. Convert `keypoints_img` from source-frame pixels to crop-video pixels by
   subtracting `crop_x,crop_y` and applying any crop-to-video scale.
4. Build a rigid fish transform from the traditional v2 pose:
   - `swim_bladder`
   - `eye_left`
   - `eye_right`
   - `snout_tip`
   - `tail_tip`
5. Use the vector from swim bladder to eye midpoint as the forward axis.
6. Rotate around the configured origin. The default is `eye_swim_midpoint`,
   halfway between the eye midpoint and swim bladder keypoint, so the stabilized
   crop is less head-biased than rotating around the eye midpoint.
7. Render a stabilized view with fish-forward pointing up.
8. Store/draw the heart ROI in the stabilized view's coordinates.
9. For measurement, inverse-transform the stabilized ROI into each crop-video
   frame and average original crop-frame pixels under that transformed polygon.

This is a first-pass rigid transform. Later probes can add scale normalization,
mask-constrained sampling, or raw-camera-pixel sampling.

## Commands

Inspect paths and metadata without opening the zarr through Python:

```bash
scripts/py playgrounds/heartrate_stabilization/inspect_sources.py \
  --config playgrounds/heartrate_stabilization/config.example.toml
```

Render a small stabilization contact sheet:

```bash
scripts/py playgrounds/heartrate_stabilization/render_stabilization_probe.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --frame-count 80 \
  --stride 10
```

Render a short side-by-side diagnostic clip:

```bash
scripts/py playgrounds/heartrate_stabilization/make_subset_clip.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --frame-count 300 \
  --output playgrounds/heartrate_stabilization/outputs/stabilization_subset_clip.mp4
```

Render a clean stabilized-only video for downstream processing:

```bash
scripts/py playgrounds/heartrate_stabilization/render_clean_stabilized_video.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --frame-start 0 \
  --frame-count 1000 \
  --output playgrounds/heartrate_stabilization/outputs/stabilized_clean_10s_lossless.mkv
```

The clean video has no keypoints, ROI outlines, labels, or mask overlays. A
sidecar CSV with the same basename records which output frames had valid
stabilization transforms. Invalid frames are written as black frames to keep
output timing aligned with the source crop video.

Clean stabilized videos default to FFV1 in Matroska (`.mkv`), which is lossless
for the generated stabilized frames. Rotation itself still resamples pixels; use
`--interpolation nearest` only when preserving exact source pixel values is more
important than visual quality.

Draw a heart ROI rectangle on a stabilized reference frame:

```bash
scripts/py playgrounds/heartrate_stabilization/draw_roi.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --frame-start 0 \
  --output playgrounds/heartrate_stabilization/outputs/heart_roi.json
```

For headless smoke tests, pass `--roi x,y,width,height` to write the same JSON
without opening the selector.

Build a fixed mask-relative ROI in canonical stabilized-video coordinates:

```bash
scripts/py playgrounds/heartrate_stabilization/build_mask_relative_roi.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --frame-start 0 \
  --frame-count 6000 \
  --mask-projection-stride 10 \
  --body-component subject_body \
  --eye-components eye_left,eye_right \
  --center-mode eye_midpoint \
  --width-px 20 \
  --height-px 12 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/mask_relative_roi
```

This projects the subject body and eye masks into the canonical rotated frame,
uses the lower eye-mask boundary as the top anchor and the midpoint of the
projected eye-mask centroids as the lateral anchor, then writes a full-frame
`roi_mask` NPZ plus a matching ROI JSON. Downstream probes can use the pair as
`--roi-json ...mask_relative_roi.roi.json --mask-npz ...mask_relative_roi.npz`.

Measure a placeholder rectangular ROI in stabilized coordinates:

```bash
scripts/py playgrounds/heartrate_stabilization/measure_roi_signal_probe.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi.json \
  --frame-count 1000 \
  --output playgrounds/heartrate_stabilization/outputs/roi_signal.csv
```

The drawn ROI is saved in stabilized fish/body-frame pixels. The measurement
script maps it back to each crop frame.

Analyze rhythmic intensity changes from a chosen ROI:

```bash
scripts/py playgrounds/heartrate_stabilization/analyze_roi_rhythm.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --frame-count 6000 \
  --fps 100 \
  --band-min-hz 1.5 \
  --band-max-hz 10 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/roi_rhythm_stabilized_mkv_60s_1p5hz_min
```

The rhythm analyzer writes a sampled ROI CSV, a JSON summary, and a plot with the
raw trace, filtered trace, and Welch power spectrum. The `--band-min-hz` value is
intentionally explicit because slow drift can dominate a fixed ROI in the
stabilized video.

Map which pixels contributed to a signal sampled directly from the stabilized
video:

```bash
scripts/py playgrounds/heartrate_stabilization/map_roi_contributors.py \
  --stabilized-video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --frame-count 6000 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/roi_contributors_stabilized_mkv_60s
```

For the source-crop measurement path, the same script can map the original crop
pixels sampled after inverse ROI projection and optional subject-mask
intersection:

```bash
scripts/py playgrounds/heartrate_stabilization/map_roi_contributors.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --mask \
  --frame-count 120 \
  --debug-frames 3 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/roi_contributors_source_crop_masked_120f
```

The map script writes stable-frame overlays, crop-frame overlays when using the
source-crop path, and a compressed `.maps.npz` with raw contribution counts and
mean-intensity maps. Use `--debug-frames` to write per-frame overlays: yellow
pixels are the exact pixels averaged for that frame, magenta is the projected
subject mask, and white is the ROI outline.

Map which ROI pixels fluctuate in the heartbeat-frequency band:

```bash
scripts/py playgrounds/heartrate_stabilization/map_pixel_band_contributions.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --frame-count 6000 \
  --fps 100 \
  --band-min-hz 1.5 \
  --band-max-hz 2.0 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/roi_pixel_band_contrib_mkv_60s_1p5_2hz
```

This is different from the sampled-pixel map. It computes a time series for each
pixel in the stabilized ROI, band-passes each pixel trace, then reports per-pixel
band power, correlation with the band-passed ROI mean signal, and signed
covariance with that aggregate rhythm.

To exclude eye-mask pixels before computing those per-pixel spectra:

```bash
scripts/py playgrounds/heartrate_stabilization/map_pixel_band_contributions.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --frame-count 6000 \
  --fps 100 \
  --band-min-hz 1.5 \
  --band-max-hz 2.0 \
  --include-mask-component subject_body \
  --exclude-mask-components eye_left,eye_right \
  --exclude-mask-dilate-px 2 \
  --exclude-mask-occupancy-threshold 0.05 \
  --mask-projection-stride 20 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/roi_pixel_band_contrib_mkv_60s_1p5_2hz_exclude_eyes
```

This projects the refined subject-mask components into stabilized coordinates,
keeps pixels that are consistently inside `subject_body`, and excludes pixels
that overlap the eye components in at least 5% of projected mask frames. The
mask projection stride only affects the stable validity-mask estimate, not the
video frames used for spectral analysis.

Analysis interpretation notes:

- The ROI mean uses all valid pixels selected by the current ROI/mask geometry;
  there is no fixed "50 pixels" sampling rule.
- Framerate controls temporal sampling, Nyquist limits, and spectral windowing,
  not how many spatial pixels should be averaged.
- For this example, the 60-second run found a `1.75 Hz` / `105 bpm` peak. Treat
  biological search bands as stage-, temperature-, and anesthesia-dependent.
- Pixel contribution maps are evidence relative to the extracted band-passed ROI
  rhythm. They are not independent proof of heart rate, and eye edges can
  dominate unless eye-mask components are excluded.

See the top-level case note for the full photometry pixel contract, framerate
contract, and literature anchors.

Compare whether the extracted peak survives different pixel-selection
strategies across time windows:

```bash
scripts/py playgrounds/heartrate_stabilization/compare_roi_pixel_strategies.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --frame-count 6000 \
  --fps 100 \
  --band-min-hz 1.5 \
  --band-max-hz 2.0 \
  --window-seconds 10 \
  --mask-npz playgrounds/heartrate_stabilization/outputs/roi_pixel_band_contrib_mkv_60s_1p5_2hz_exclude_eyes.pixel_band_maps.npz \
  --mask-label eye_excluded \
  --top-k-values 25,50,100 \
  --top-score-modes covariance,correlation \
  --output-prefix playgrounds/heartrate_stabilization/outputs/roi_pixel_strategy_compare_60s_1p5_2hz
```

This writes a summary CSV, a windowed QC plot, and a selected-pixel frequency
map. Top-K strategies are selected by leave-one-window-out scoring so a window
is not used to choose the pixels evaluated on that same window.

For a lag-based full-window estimate, use autocorrelation as the primary
estimator. A denser window step is useful when aligning to short stimulus trials:

```bash
scripts/py playgrounds/heartrate_stabilization/compare_roi_pixel_strategies.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --frame-count 140035 \
  --fps 100 \
  --band-min-hz 1.5 \
  --band-max-hz 3.0 \
  --min-roi-mean-intensity 1 \
  --primary-estimator autocorr \
  --window-seconds 10 \
  --window-step-seconds 2.5 \
  --mask-npz playgrounds/heartrate_stabilization/outputs/roi_pixel_band_contrib_mkv_60s_1p5_2hz_exclude_eyes.pixel_band_maps.npz \
  --mask-label eye_excluded \
  --top-k-values 50 \
  --top-score-modes covariance \
  --output-prefix playgrounds/heartrate_stabilization/outputs/roi_pixel_strategy_compare_full_autocorr_1p5_3hz_step2p5s
```

Visualize the underlying masked-ROI intensity structure behind that HR trace:

```bash
scripts/py playgrounds/heartrate_stabilization/visualize_roi_intensity_diagnostics.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --mask-npz playgrounds/heartrate_stabilization/outputs/roi_pixel_band_contrib_mkv_60s_1p5_2hz_exclude_eyes.pixel_band_maps.npz \
  --hr-csv playgrounds/heartrate_stabilization/outputs/hr_timeseries_eye_excluded_autocorr_1p5_3hz_step2p5s.csv \
  --fps 100 \
  --band-min-hz 1.5 \
  --band-max-hz 3.0 \
  --min-roi-mean-intensity 1 \
  --window-seconds 10 \
  --window-step-seconds 2.5 \
  --darkening-phase-z-threshold 0.75 \
  --top-pixels 120 \
  --sort-by covariance \
  --save-top-matrix \
  --output-prefix playgrounds/heartrate_stabilization/outputs/roi_intensity_diagnostics_eye_excluded_full_1p5_3hz
```

This writes a masked-pixel overlay, a full-recording pixel-correlation map, raw
and band-passed ROI mean traces, a top-pixel raster, and sliding-window
correlation summaries. It also writes a directional darkening-support map: ROI
band-passed z-scores below `-0.75` define the dark phase, values above `0.75`
define the bright phase, and positive darkening support means a pixel is darker
during the ROI dark phase than during the bright phase. A single frame cannot
have a Pearson correlation by itself; the per-frame support trace is the signed
agreement/covariance contribution of top band-passed pixels at that frame. Long
invalid status gaps are preserved as gaps rather than filtered across.

Use `--min-roi-mean-intensity 1` for acquisition-time crop dropouts. In these
recordings, if the real-time detector fails to localize the fish, the crop
recorder can save an all-black frame. Those frames must be treated as invalid
photometry samples, not as real intensity changes.

Render a video of the heartbeat-band fluctuations:

```bash
scripts/py playgrounds/heartrate_stabilization/render_roi_fluctuation_video.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --mask-npz playgrounds/heartrate_stabilization/outputs/roi_pixel_band_contrib_mkv_60s_1p5_2hz_exclude_eyes.pixel_band_maps.npz \
  --frame-start 30000 \
  --frame-count 3000 \
  --fps 100 \
  --playback-fps 30 \
  --band-min-hz 1.5 \
  --band-max-hz 3.0 \
  --min-roi-mean-intensity 1 \
  --trace-seconds 8 \
  --output playgrounds/heartrate_stabilization/outputs/roi_fluctuation_chase_start_30000_3000f_1p5_3hz.mp4
```

The video overlay is temporally filtered before display: each sampled ROI pixel
is passed through the same third-order zero-phase Butterworth band-pass used by
the diagnostics. With the example band, the display rejects slow drift below
`1.5 Hz` and fast noise above `3.0 Hz`. Red means the pixel is darker than its
local baseline in that heartbeat band; blue means brighter. Noise inside the
selected band can still pass through, so spatial coherence, mask exclusion, and
directional darkening support remain important QC signals.

Derive a candidate heart mask from per-pixel spectral power, then use that mask
only for time-domain luminance and beat timing:

```bash
scripts/py playgrounds/heartrate_stabilization/derive_heart_pixel_mask.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi_eye_swim_origin_smoke.json \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --mask-npz playgrounds/heartrate_stabilization/outputs/roi_pixel_band_contrib_mkv_60s_1p5_2hz_exclude_eyes.pixel_band_maps.npz \
  --frame-count 140035 \
  --fps 100 \
  --band-min-hz 1.5 \
  --band-max-hz 3.5 \
  --chunk-seconds 30 \
  --chunk-step-seconds 30 \
  --top-pixels 50 \
  --chunk-top-fraction 0.20 \
  --min-top-chunk-fraction 0.10 \
  --boundary-penalty-width-px 5 \
  --boundary-penalty-weight 5 \
  --min-roi-mean-intensity 1 \
  --beat-prominence-z 0.75 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/heart_pixel_mask_eye_excluded_30s_chunks_1p5_3p5hz_k50_boundary5w5
```

This uses Welch spectra per pixel and per chunk only to select pixels. It then
averages the selected pixels in the time domain, band-passes that luminance
trace, flips the sign so darkening is positive, and detects darkening peaks in
time. Inspect the heart-mask image before trusting the beat table. In the first
example run, the strongest pixels are concentrated along high-contrast ROI/body
edges rather than forming a compact central heart-shaped region, so the result
should be treated as evidence of residual motion/edge artifact, not a validated
heart-rate extraction.

Boundary penalization can be enabled with `--boundary-penalty-width-px` and
`--boundary-penalty-weight`. The penalty subtracts from the per-pixel spectral
selection score for pixels close to the edge of the usable candidate mask. In
the checked example, both a moderate penalty (`5 px`, weight `5`) and a stronger
penalty (`7 px`, weight `10`) still selected pixels with median distance only
`2 px` from the usable-mask boundary, so this soft prior did not rescue the ROI.
That suggests the strongest heartbeat-band evidence in this crop is still
edge-dominated.

Align those HR windows to GoodCopBadCop chase trials:

```bash
scripts/py playgrounds/heartrate_stabilization/align_hr_to_chase_trials.py \
  --hr-csv playgrounds/heartrate_stabilization/outputs/roi_pixel_strategy_compare_full_autocorr_1p5_3hz_step2p5s.summary.csv \
  --strategy eye_excluded \
  --crop-meta-csv /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/derived/external_crop_recorder/Cam2010096_2026-06-14T21-12-08Z_arena_4_crop_meta.csv \
  --stimulus-h5 /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/raw/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop.h5 \
  --fps 100 \
  --pre-seconds 20 \
  --post-seconds 30 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/hr_chase_alignment_eye_excluded_autocorr_1p5_3hz_step2p5s_unwrapped
```

The alignment script unwraps the crop metadata's `camera_frame_id` counter before
matching to trial camera-frame IDs. In this example the counter wraps from
`65535` to `1`.

After a compatible subject-mask job has written masks into the analysis zarr,
enable mask intersection on the same measurement path:

```bash
scripts/py playgrounds/heartrate_stabilization/measure_roi_signal_probe.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi.json \
  --mask \
  --mask-component subject_body \
  --frame-count 1000 \
  --output playgrounds/heartrate_stabilization/outputs/roi_signal_masked.csv
```

With `--mask`, `mean_intensity` is measured from `ROI polygon AND projected
subject mask` when the mask is available for that frame. The CSV also includes
`roi_unmasked_mean_intensity` and mask diagnostics so the masked and unmasked
signals can be compared. Add `--require-mask` when missing masks should make a
frame invalid instead of falling back to ROI-only sampling.

Preferred input is dense binary `masks_roi` from `refined_subject_masks_runs`.
For raw `subject_mask_runs`, the playground can also threshold `mask_probs_roi`
with the run's `thresholds_by_label` or `mask_probability_threshold` attrs,
defaulting to `0.5`.

To visually check the mask projection before trusting the trace:

```bash
scripts/py playgrounds/heartrate_stabilization/make_subset_clip.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --roi-json playgrounds/heartrate_stabilization/outputs/heart_roi.json \
  --mask \
  --frame-count 300 \
  --output playgrounds/heartrate_stabilization/outputs/stabilization_subset_clip_masked.mp4
```

The mask overlay is magenta; the ROI is yellow. The checked example currently
has no `subject_mask_runs` or `refined_subject_masks_runs`, so `--mask` reports
that masks are unavailable until a mask job has populated the zarr.

The subset clip is for visual QA only. Do not measure heartbeat intensity from
the stabilized clip; measure from the source crop video through
`measure_roi_signal_probe.py`.

The stabilized panel applies a circular mask by default so the user can inspect
fish/body-frame stability without watching the source crop corners rotate
through the view. Disable `[alignment].stable_circular_mask` in a local
`config.toml` when debugging warp boundaries.

## Frame-Domain Guard

This example's refined-keypoint `frame_indices` start at `0` and align to the
crop-video frame index. They do not align to the crop metadata's
`camera_frame_id` column. That is why the config uses:

```toml
frame_id_column = "crop_video_frame_index"
```

Treat this as a required join contract for every recording. Before trusting a
stabilized overlay, verify that keypoints land on the fish in the original crop
panel.

## Sandbox Note

The `/groups` zarr can hang when opened from the Codex sandbox. These scripts
are written for normal workstation execution with `scripts/py`. Static metadata
inspection does not require zarr reads; rendering and ROI measurement do.

## Promotion Criteria

Only promote this into `src/fisheye/analysis` after the playground can show:

- low residual motion of the swim bladder and eye midpoint in stabilized view;
- visually stable heart ROI over a representative rotation interval;
- reduced motion artifacts in the intensity trace compared with fixed crop
  pixels;
- explicit invalid-frame handling when keypoints or crop metadata are missing.
