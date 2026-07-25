# Heartrate Stabilization Playground

Purpose: prototype fish-attached ROI measurements for the heartrate example
recording without turning the first experiments into production Palette
analysis surfaces.

Top-level case note: [`docs/archive/heartrate_stabilized_roi_case.md`](../../docs/archive/heartrate_stabilized_roi_case.md).
Current moving-fish and embedded positive-control status:
[`docs/archive/heartrate_analysis_status_2026-07-11.md`](../../docs/archive/heartrate_analysis_status_2026-07-11.md).
Final interpretation and the frozen cross-trial intern handoff:
[`docs/heartrate_final_decision_2026-07-12.md`](../../docs/heartrate_final_decision_2026-07-12.md) and
[`docs/embedded_fish_heartrate_intern_handoff_2026-07-13.md`](../../docs/embedded_fish_heartrate_intern_handoff_2026-07-13.md).

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

### Embedded-Fish Top-Camera Box Over SSH

For the embedded positive control, run the browser selector in a remote tmux
window. It reads a frame directly from the top-camera video and stores the box
in source-video pixel coordinates:

```bash
scripts/py playgrounds/heartrate_stabilization/select_fixed_video_roi_web.py \
  --video /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Top_Camera/20250109_F1_10_30_Trial1.mp4 \
  --frame-index 0 \
  --output /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/top_camera_roi_trial1.json \
  --port 8765
```

Leave that process running. In a separate terminal on the laptop, forward the
loopback port through the same SSH destination normally used for the
workstation:

```bash
ssh -N -L 8765:127.0.0.1:8765 <same-workstation-ssh-target>
```

Open `http://127.0.0.1:8765` on the laptop, drag a rectangle around the fish,
and choose **Save ROI**. The selector also accepts exact `x`, `y`, `width`, and
`height` values. Saving writes the JSON plus an annotated preview PNG beside
it. Rerunning the command reloads a saved selection only when its video path
and frame dimensions still match.

The server binds to remote loopback by default; do not change `--host` when
using the SSH tunnel. The saved box is fixed in top-camera frame coordinates,
which is appropriate for this embedded fish but is not a fish-attached ROI for
a freely moving recording.

Analyze Trial 1 against the supplied 100 Hz per-frame rate trace:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/analyze_embedded_positive_control.py \
  --roi-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/top_camera_roi_trial1.json \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_roi
```

The runner reports the ROI mean, ROI-to-surround log ratio, and a
reference-blind band-limited spatial component. It also independently fits the
same spatial component in four same-size offset boxes. Outputs include a JSON
summary, compact CSV, and diagnostic plot. These are descriptive
positive-control results; fitting and assessing the spatial component on the
same trial is not a substitute for frozen-method evaluation on the remaining
trials.

Render every Trial 1 source frame as a 4x slow-motion inspection video:

```bash
scripts/py playgrounds/heartrate_stabilization/render_embedded_positive_control_overlay.py \
  --roi-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/top_camera_roi_trial1.json \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --output-fps 50 \
  --output playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/Trial1_top_camera_heart_overlay_slow4x.mp4
```

The H.264 overlay retains the native full frame, enlarges the raw ROI with
nearest-neighbor sampling, displays fixed-scale signed bandpassed pixel
changes, and shows both the supplied reference and extracted event-interval
rate. The 200 fps source plays at 50 fps, so all frames are retained at 4x slow
motion.

Render the colleague's fixed side-camera ROI using the all-pixel ROI mean that
best reproduces the supplied side-camera rate trace:

```bash
scripts/py playgrounds/heartrate_stabilization/render_embedded_positive_control_overlay.py \
  --roi-json playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/side_camera_colleague_roi.json \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --signal roi_mean \
  --output-fps 25 \
  --output playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/Trial1_side_camera_colleague_roi_mean_overlay_slow4x.mp4
```

The `100 fps` side video plays at `25 fps`, retaining all `3998` frames at 4x
slow motion. Pixel colors include the complete fixed ROI. The colleague's
workbook reports a per-trial good-pixel count but does not include the selected
pixel mask, so this visualization does not reconstruct that selection.

Select two neutral side-camera chamber polygons over SSH:

```bash
scripts/py playgrounds/heartrate_stabilization/select_fixed_video_chambers_web.py \
  --video /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Side_Camera/20250109_F1_10_30_heart_1.avi \
  --frame-index 2000 \
  --output /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/side_camera_chambers_trial1.json \
  --port 8766
```

Analyze fixed chamber means, their union, two-pixel boundary sensitivity, and
windowed chamber delay:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/analyze_embedded_side_chambers.py \
  --chambers-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/side_camera_chambers_trial1.json \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --band-hz 1.5 4.0 \
  --sensitivity-radius-px 2 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/side_camera_chamber_comparison
```

The saved regions remain `chamber_a` and `chamber_b` until anatomical identity
is confirmed. Fixed-polygon photometry is not per-frame chamber segmentation.

Render the raw side view, both fixed polygons, their bandpassed mean states,
candidate event-interval rates, and the supplied rate reference at 4x slow
motion:

```bash
scripts/py playgrounds/heartrate_stabilization/render_embedded_side_chambers_overlay.py \
  --chambers-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/side_camera_chambers_trial1.json \
  --analysis-arrays playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/side_camera_chamber_comparison.arrays.npz \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --band-hz 1.5 4.0 \
  --output-fps 25 \
  --ffmpeg-preset veryfast \
  --output playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/Trial1_side_camera_chambers_overlay_slow4x.mp4
```

Each colored polygon is spatially uniform: its color is the signed bandpassed
mean of all pixels in that fixed polygon. The display therefore compares two
interpretable photometry traces and does not claim pixel-resolved propagation.
The plotted intervals are automatically detected candidates, not validated
one-to-one contraction annotations.

Discover a variable-rate spatial support independently in the two temporal
halves of embedded top-camera Trial 1, then use the side-derived workbook only
for held-out evaluation:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/analyze_embedded_crossfit_mask.py \
  --roi-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/top_camera_roi_trial1.json \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --band-hz 1.5 4.0 \
  --guard-seconds 1.0 \
  --block-seconds 4.0 \
  --score-threshold-z 1.5 \
  --min-cluster-pixels 3 \
  --event-prominence-mad 0.5 \
  --event-filter-edge-seconds 0.75 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_roi_crossfit_mask
```

This is a reference-blind mask-selection diagnostic, not a calibrated
detection. It permits an empty mask and never compresses the guarded midpoint
gap during spectral display. Event polarity and amplitude scale are learned on
the discovery half; interval detection runs only on the opposite half and
never crosses the guard. Event-rate agreement with the supplied trace does not
replace visible contraction timestamps.

Render every Trial 1 top-camera frame with its opposite-half cross-fit mask:

```bash
scripts/py playgrounds/heartrate_stabilization/render_embedded_crossfit_mask_overlay.py \
  --roi-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/top_camera_roi_trial1.json \
  --analysis-summary playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_roi_crossfit_mask.summary.json \
  --analysis-arrays playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_roi_crossfit_mask.arrays.npz \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --output-fps 50 \
  --output playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/Trial1_top_camera_crossfit_mask_overlay_slow4x.mp4
```

Fold outlines indicate which opposite-half model owns the displayed frame;
dynamic red/blue pixels indicate signed bandpassed change instead. The guarded
midpoint intentionally has no mask colors or held-out waveform.

Compare the existing masked loading with a discovery-half PCA refit and an
equal-weight frozen-mask mean:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/compare_embedded_crossfit_mask_projections.py \
  --roi-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/top_camera_roi_trial1.json \
  --mask-summary playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_roi_crossfit_mask.summary.json \
  --mask-arrays playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_roi_crossfit_mask.arrays.npz \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --event-prominence-mad 0.5 \
  --event-filter-edge-seconds 0.75 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_crossfit_mask_projection_comparison
```

All three variants use the same frozen fold masks and opposite-half ownership.
The side-derived reference is loaded only after every projection weight is
frozen.

Evaluate how the held-out embedded equal-mask mean tracks the reference as the
spectral-ridge window is shortened:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/evaluate_embedded_rate_window_sweep.py \
  --analysis-arrays playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_crossfit_mask_projection_comparison.arrays.npz \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --method masked_equal_mean \
  --fps 200 \
  --reference-fps 100 \
  --band-hz 1.5 4.0 \
  --window-seconds 2 3 4 6 8 10 \
  --step-seconds 1 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_masked_equal_mean_window_sweep_step1s
```

The sweep operates separately inside each finite held-out block. It cannot
bridge the guarded midpoint. Shorter windows trade temporal responsiveness
against spectral variance and should be selected on positive-control error,
not on which moving-fish trajectory looks most convincing.

Render the complete embedded top-camera `2 s` held-out masked-mean diagnostic
at 4x slow motion:

```bash
scripts/py playgrounds/heartrate_stabilization/render_embedded_masked_mean_window_overlay.py \
  --roi-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/top_camera_roi_trial1.json \
  --mask-summary playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_roi_crossfit_mask.summary.json \
  --projection-arrays playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_crossfit_mask_projection_comparison.arrays.npz \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --method masked_equal_mean \
  --window-seconds 2 \
  --step-seconds 1 \
  --output-fps 50 \
  --ffmpeg-preset veryfast \
  --output playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/Trial1_top_camera_masked_equal_mean_2s_window_overlay_slow4x.mp4
```

The colored support is spatially uniform and encodes the signed equal-mask
mean, not per-pixel phase. The orange trajectory is a held-out `2 s` spectral
ridge with a `1 s` step, not an event-interval trace. The mask, waveform, and
current estimate are blank through the cross-fit guard.

Quantify the fearful-stimulus-associated Trial 1 bradycardia using the supplied
side bout for evaluation and a fixed prestimulus baseline:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/analyze_embedded_bradycardia_response.py \
  --projection-arrays playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_crossfit_mask_projection_comparison.arrays.npz \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --method masked_equal_mean \
  --fps 200 \
  --reference-fps 100 \
  --baseline-s 28 32 \
  --stimulus-s 32 36 \
  --drop-threshold-bpm 30 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_masked_equal_mean_stimulus_bradycardia
```

Render the baseline, stimulus, supplied side bout, independently thresholded
top response, `2 s` ridge, and top peak-interval rate as a focused 4x-slow
clip:

```bash
scripts/py playgrounds/heartrate_stabilization/render_embedded_masked_mean_window_overlay.py \
  --roi-json /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/top_camera_roi_trial1.json \
  --mask-summary playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_roi_crossfit_mask.summary.json \
  --projection-arrays playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_crossfit_mask_projection_comparison.arrays.npz \
  --workbook /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/embedded_fish_positive_control/Bradyinfo_20250109_F1_10_30.xlsx \
  --trial-number 1 \
  --method masked_equal_mean \
  --window-seconds 2 \
  --step-seconds 1 \
  --brady-summary playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/top_camera_masked_equal_mean_stimulus_bradycardia.summary.json \
  --source-start-s 28 \
  --source-stop-s 40 \
  --event-display-smoothing-s 0.1 \
  --output-fps 50 \
  --output playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/Trial1_top_camera_stimulus_bradycardia_response_overlay_smooth100ms_slow4x.mp4
```

The workbook provides the side bradycardia bouts but not a separate stimulus-
timing field. The declared `32..36 s` interval matches the highlighted Trial 1
plot and must be confirmed from protocol metadata before cross-trial use.
The `100 ms` Gaussian smoothing affects only the displayed cyan curve, operates
separately inside finite blocks, and does not change response detection,
metrics, or the raw peak-interval value printed below the video.

Omit `--source-stop-s` and use `--source-start-s 0` to retain all `8001`
top-camera frames. The complete 4x-slow output used here is:

```text
playgrounds/heartrate_stabilization/outputs/embedded_positive_control_trial1/
  Trial1_top_camera_full_masked_mean_bradycardia_overlay_smooth100ms_slow4x.mp4
```

Apply the same simple band-PCA idea to every qualifying segment of a moving-fish
cache, without folds or discovery/confirmation guards:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/analyze_segmented_cache_pca.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f.local_pixel_matrix.npz \
  --band-hz 1.5 4.0 \
  --window-seconds 60 \
  --analysis-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_36000_30000f_exploratory_four_mask_2p0_4p0.mask_comparison.arrays.npz \
  --analysis-mask-key intersection_8_mask \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f_simple_segmented_pca_intersection8_60s
```

The runner uses all common-valid segments of at least two seconds, bridges only
bounded gaps of at most `0.02 s`, and rejects `0.75 s` at each filtered segment
edge. It does not compress gaps or calculate intervals across them. Because the
PCA loading and event polarity are fitted on the same cache, outputs are
exploratory candidate summaries rather than validated heart-rate estimates.

Run the frozen 38-pixel support across the complete recording and search
`2..4 Hz` in one-minute windows:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/analyze_segmented_cache_pca.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz \
  --band-hz 2.0 4.0 \
  --frequency-step-hz 0.05 \
  --window-seconds 60 \
  --analysis-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_36000_30000f_exploratory_four_mask_2p0_4p0.mask_comparison.arrays.npz \
  --analysis-mask-key original_38_mask \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_simple_segmented_pca_original38_60s_2p0_4p0
```

The full run writes observed and independently fitted motion-PCA frequency for
each exact same window. Matching aggregate peaks are not treated as
confirmation; compare their complete window trajectories and retain every
unscorable window.

Compare that PCA projection with equal means over the frozen full mask and its
previously declared 18-pixel upper / 20-pixel lower partition:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/compare_moving_frozen_mask_means.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz \
  --mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_36000_30000f_exploratory_four_mask_2p0_4p0.mask_comparison.arrays.npz \
  --mask-key original_38_mask \
  --regions-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.regional_phase_delay.arrays.npz \
  --upper-key upper_mask \
  --lower-key lower_mask \
  --band-hz 2 4 \
  --frequency-step-hz 0.05 \
  --window-seconds 4 8 \
  --filter-edge-seconds 0.75 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_comparison
```

All projections share the exact same segment and preprocessing contract.
The regional split and mask are reused rather than relearned. The comparison
is same-cache and exploratory; its lower-half result must transfer to new data
before being promoted.

The comparison also emits `lower_raw_mean`: the literal Mono8 lower-mask mean
is formed before segment detrending and bandpass. Unlike `lower_equal_mean`,
it does not robustly scale pixels individually. On the complete example cache
the two lower traces correlate `0.9997` and select the same frequency in every
paired scorable `4 s` and `8 s` window. Prefer the raw lower mean as the
simplest transfer candidate and keep the normalized version as a sensitivity
control.

For subsequent moving-fish work, the literal mean over the frozen 20-pixel
`lower_mask` is the primary candidate projection. Keep the normalized lower
mean, 38-pixel PCA, 18-pixel upper mean, 38-pixel full mean, and motion
projections as controls. Do not switch the primary projection per window,
expand the lower mask after inspection, or call its oscillations heartbeats
without independent cardiac validation.

Render the cleaner second chase cache using its no-fold `3.35 Hz` candidate:

```bash
scripts/py playgrounds/heartrate_stabilization/render_segmented_cache_pca_overlay.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_33000_3000f.local_pixel_matrix.npz \
  --analysis-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.dynamic_support.arrays.npz \
  --analysis-mask-key heart_support_mask \
  --band-hz 1.5 4.0 \
  --candidate-frequency-hz 3.35 \
  --output-fps 25 \
  --output playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_33000_3000f_simple_segmented_pca_original38_overlay_slow4x.mp4
```

The video retains every source frame at 4x slow motion. Quantitative colors and
waveforms appear only inside the three post-edge valid segments; source video
continues through all invalid spans.

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

To compare whole-frame keypoint stabilization against a local rostral-segment
correction, render the eye-to-swim-bladder diagnostic:

```bash
scripts/py playgrounds/heartrate_stabilization/render_local_rostral_alignment_comparison.py \
  --video playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.mkv \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --roi-json playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.roi.json \
  --mask-npz playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.npz \
  --center-frames 30700,31190,31440 \
  --context-frames 60 \
  --stride 1 \
  --output playgrounds/heartrate_stabilization/outputs/local_rostral_alignment_comparison_component_eye_anchor_gated_worst_frames_eye_mid_w20_h12.mp4
```

The local correction is a rigid translate/rotate only. It uses the midpoint of
the two individual eye-component bottom anchors and the rostral swim-bladder
anchor. Corrections above `50 deg` or `150 px` are rejected and shown as local
alignment failures; they should be treated as tracking/mask-quality flags, not
as frames to warp into place.

To compare source-pixel intensity traces from the fixed ROI and the gated local
ROI:

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

This measurement still samples original crop-video pixels. The local correction
only maps the fixed ROI back into each frame's source-pixel coordinates and
gates frames with missing or implausible local anchors.

For calibrated per-pixel discovery, held-out confirmation, confirmed-only event
extraction, and source-raster injection recovery, use:

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

The matrix persists the original timestamp grid, invalid rows, canonical and
source coordinates, bilinear weights, mask occupancy, risk surfaces, transform
diagnostics, and nuisance covariates. Reuse it without Zarr reads:

```bash
scripts/py playgrounds/heartrate_stabilization/extract_reliable_local_rostral_heartrate.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f.local_pixel_matrix.npz \
  --surrogate-count 199 \
  --alpha 0.05 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_confirmatory
```

To test an extended anatomical support whose pixels may have different phase or
contrast polarity, use `--analyze-dynamic-support`. A cluster-union mask from
the same interval is always exploratory. For confirmation, supply a mask frozen
from an earlier interval and predeclare the frequency search:

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

The checked frozen-mask interval supports a cross-fitted latent spatial pattern
at `3.35 Hz` with `p=0.010`, but it does not yet provide validated cardiac event
times. Same-sign whole-mask averaging remains non-significant.

`--render-dynamic-phase` writes a three-panel diagnostic video, a static phase
strip, the frame-resolved arrays, and a JSON summary. The panels show fixed-scale
local stabilized samples with the frozen support contours, held-out band-limited
activation, and unaligned pixel phase with confidence encoded by opacity. A
synchronized trace shows the opposite-fold phase-aligned latent signal. Invalid
frames, partition guards, long gaps, and filter edges remain uncolored. In the
checked interval, phase is available on `25.0%` of source frames and median
within-frame spatial alignment is `0.660`. These are visualization diagnostics,
not cardiac-event annotations or validated beat coverage. The presence of
smooth color cycles is expected after narrow-band analytic filtering; the color
cycles themselves are not additional evidence of a heartbeat.

To measure whether an apparent upper-to-lower flow repeats, add:

```text
--analyze-regional-phase-delay
--regional-phase-surrogate-count 199
```

The default split uses only frozen-mask geometry: it chooses the horizontal
boundary that most evenly divides support pixels. For this mask, canonical
`y=121.5` produces 18 upper and 20 lower pixels. Positive reported lag means
the lower region reaches the same phase after the upper region. Outputs include
frame-level lag arrays, per-block and per-cycle CSVs, a summary, and a diagnostic
figure.

The checked source interval has highly repeatable delay within each valid block,
but the block means range from `-38` to `-138 ms`; across-block stability is not
above the conditional block-phase null (`PLV=0.720`, `p=0.14`). The independent
video interval reverses direction between blocks (`-119`, `+43`, `+148`, and
`-39 ms`) and is clearly unstable across blocks (`PLV=0.180`, `p=0.86`). Thus
the visible flow is measurable within short blocks, but it is not currently a
stable propagation signature.

This regional hypothesis was formulated after viewing these intervals and is
exploratory for both. The generated `upper_mask` and `lower_mask` can be frozen
for a future interval using `--regional-phase-regions-npz` with
`--regional-phase-regions-independent`.

### Long Frame-0 Cache

The first five minutes can be extracted as ten resumable parts and merged into
one validated cache:

```bash
scripts/py playgrounds/heartrate_stabilization/extract_local_rostral_cache_chunks.py \
  --roi-json playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.roi.json \
  --mask-npz playgrounds/heartrate_stabilization/outputs/mask_relative_roi_chase_start_30000_3000f_eye_mid_w20_h12.mask_relative_roi.npz \
  --status-csv playgrounds/heartrate_stabilization/outputs/stabilized_clean_full_eye_swim_origin_lossless.csv \
  --frame-start 0 \
  --frame-count 30000 \
  --chunk-frames 3000 \
  --workers 4 \
  --mask-read-cache-rows 256 \
  --reference-anterior-xy 128,113 \
  --reference-posterior-xy 127,143 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f
```

The dense subject-mask arrays use physical chunks of `(256, 1, 512, 512)`.
Without a row-block cache, reading one mask row repeatedly decompresses the same
256-row component chunk. The aligned cache reduced a checked 300-frame run from
`73.5 s` to `7.6 s` (`9.7x`) with identical numeric arrays. One-frame overlaps
preserve motion prediction and duplicate-frame detection at part boundaries.
The merge fails unless all parts have identical pixel grids, schemas, masks,
anchors, and static metadata, then rebuilds timestamps from acquisition metadata.

The merged cache contains frames `0..29999`, spans `299.99 s`, and has
`26959/30000` valid local-coordinate frames (`89.86%`). Apply the later frozen
hypothesis to this earlier interval with:

```bash
scripts/py playgrounds/heartrate_stabilization/extract_reliable_local_rostral_heartrate.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f.local_pixel_matrix.npz \
  --surrogate-count 39 \
  --alpha 0.05 \
  --analyze-dynamic-support \
  --dynamic-heart-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.dynamic_support.arrays.npz \
  --dynamic-heart-mask-key heart_support_mask \
  --dynamic-support-mask-independent \
  --dynamic-support-frequency-min-hz 3.0 \
  --dynamic-support-frequency-max-hz 3.5 \
  --dynamic-support-surrogate-count 199 \
  --render-dynamic-phase \
  --dynamic-phase-frame-stride 10 \
  --dynamic-phase-playback-fps 30 \
  --analyze-regional-phase-delay \
  --regional-phase-regions-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.regional_phase_delay.arrays.npz \
  --regional-phase-regions-independent \
  --regional-phase-surrogate-count 999 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f_frozen_hypothesis_confirmatory199
```

The later frozen support independently identifies `3.20 Hz` in the frame-0
cache. Support, shared-phase, and latent-pattern statistics are each `p=0.005`
with 199 full-pipeline surrogates, and the latent score is `4.61x` the strongest
motion/control score. The standard compact-cluster method still emits no
estimate. The frozen upper/lower split has 55 valid blocks, 306 paired cycles,
median within-block PLV `0.979`, across-block PLV `0.592`, and conditional
`p=0.001`; the lower region leads in 47/55 blocks. This is strong evidence for a
reproducible periodic anatomical source near `3.2 Hz`, not yet validated cardiac
event timing.

The command is expected to emit no estimate when either held-out fold, matched
controls, cross-fit frequency agreement, or cluster reproducibility fails. It
never emits events from an unconfirmed source or a held-out block that fails
the maximum-statistic correction across confirmation blocks. See
`docs/archive/heartrate_local_rostral_roi_status_2026-07-09.md` for the checked result
and injection-recovery command.

### Five-Minute Consensus-Mask Discovery

To ask whether the original 30-second union was too dependent on that interval,
learn compact candidates on five contiguous outer folds. Each candidate is
learned on four minutes and phase-aligned on the excluded minute. Spatial
recurrence is calibrated by independently translating each discovered cluster,
with its shape and size preserved, within the physically eligible ROI:

```bash
scripts/py playgrounds/heartrate_stabilization/learn_consensus_heart_mask.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f.local_pixel_matrix.npz \
  --band-min-hz 3.0 \
  --band-max-hz 3.5 \
  --frequency-step-hz 0.05 \
  --outer-fold-count 5 \
  --outer-guard-seconds 1.0 \
  --min-selection-folds 3 \
  --min-confirmed-outer-folds 3 \
  --consensus-surrogate-count 999 \
  --heldout-surrogate-count 199 \
  --alpha 0.05 \
  --seed 101 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f_consensus5x60s_phase_aligned_cached_discovery
```

The checked run found nine pixels selected in all five outer discoveries. Their
spatial maximum-null `p` value is `0.019`; pixels selected in four folds do not
pass (`p=0.201`). Only two of five excluded minutes pass the phase-aligned
temporal/control gate, so the formal discovery result remains
`too_few_outer_folds_confirmed`. The nine pixels are an exploratory challenger,
not an automatically promoted replacement mask.

The predeclared untouched interval is frames `36000..65999`. Extract it using
the same anchors and chunk-safe cache workflow, then compare the original mask,
the nine-pixel challenger, their intersection, and their union:

```bash
scripts/py playgrounds/heartrate_stabilization/compare_frozen_heart_masks.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_36000_30000f.local_pixel_matrix.npz \
  --original-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.dynamic_support.arrays.npz \
  --original-mask-key heart_support_mask \
  --consensus-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f_consensus5x60s_phase_aligned_cached_discovery.consensus_mask.arrays.npz \
  --consensus-mask-key consensus_mask \
  --frequency-min-hz 3.0 \
  --frequency-max-hz 3.5 \
  --base-surrogate-count 39 \
  --dynamic-surrogate-count 199 \
  --surrogate-batch-size 25 \
  --surrogate-workers 4 \
  --alpha 0.05 \
  --seed 211 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_36000_30000f_prespecified_four_mask_comparison
```

The four masks share each surrogate's nuisance fits and frequency-coefficient
tensors whenever their exact valid-block layouts match. Surrogate random streams
are keyed by the analysis seed and global surrogate index, so changing the batch
size or worker count does not change any null sample. Each batch is written under
`<output-prefix>.surrogate_batches/` and reused after an interrupted run only
when its dataset, masks, configuration, frequency bounds, and seed identity all
match. Use `--recompute-surrogate-batches` to intentionally replace those
cached batches. Keep `--surrogate-workers 1` when CPU or memory contention is
more important than elapsed time.

The 30,000-frame untouched cache has `19775/30000` locally valid frames. All
four masks select `3.25 Hz`, but only the nine-pixel consensus and eight-pixel
intersection exceed the maximum across all four masks and all searched
frequencies for support, shared-phase, and latent statistics (`p=0.005` for
each statistic). The original 38-pixel mask has familywise latent `p=0.480`; the
39-pixel union has `p=0.415`. This supports a compact recurrent core within the
original region, still without validating biological beat identity.

Matched diagnostic-only phase videos can be rendered from the frozen comparison
masks with `--dynamic-support-surrogate-count 0`,
`--dynamic-phase-frame-stride 3`, and `--dynamic-phase-playback-fps 30`. The
checked consensus-9 and union-39 videos each contain 10,000 frames, last
333.33 seconds, and play at `0.900x` source time. Phase-valid coverage is
`20.55%` for both; median spatial alignment is `0.908` for consensus-9 and
`0.682` for union-39. Their local dynamic summaries report `p=1` because no new
null was requested for visualization. Use the completed four-mask summary for
all inferential p-values.

A frame-locked side-by-side render places consensus-9 on the left and union-39
on the right. It preserves the same 10,000 frames and 333.33-second timeline so
a single video player can pause and seek both diagnostics without drift.

The previously frozen 18-pixel upper and 20-pixel lower regions can also be
applied unchanged to the union-39 phase reconstruction on the untouched
interval. The checked 999-surrogate conditional test uses 34 valid blocks and
174 paired cycles. The lower region leads in 27/34 blocks and 141/174 cycles;
the circular mean lag is `-123.6 ms` (`144.6 deg` lower-minus-upper phase), the
median paired-cycle lag is `-111.2 ms` with MAD `21.6 ms`, across-block PLV is
`0.739`, median within-block PLV is `0.994`, and conditional `p=0.001`. This
tests repeatability of the frozen regional delay, not cardiac identity.

### Surrogate Missingness Audit

As of 2026-07-11, p-values from analyses that call
`autocorrelation_preserving_surrogate` are withdrawn pending recalibration. The
shared helper rolled traces but did not roll per-pixel validity with them,
which could reduce usable null samples and make the null anti-conservative.
This affects earlier discovery, dynamic-support, consensus, and global-null
results, including the compact-mask p-values described above. Observed scores,
frequencies, masks, and visualizations remain descriptive artifacts only.

The regional phase-randomization result uses a different null mechanism and is
not affected by that implementation defect, but it remains conditional on the
already selected support, frequency, filter, and regional split. None of these
results currently validates cardiac identity or beat timing.

### Mono8 Photometry Transform Comparison

Compare low-SNR Mono8 trace constructions across the full cache while keeping
each one-minute frequency frozen to the compact intersection estimate:

```bash
scripts/py playgrounds/heartrate_stabilization/compare_heart_photometry_transforms.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz \
  --longitudinal-csv playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_frozen_masks_60s_2p0_4p0.longitudinal.csv \
  --original-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.dynamic_support.arrays.npz \
  --consensus-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f_consensus5x60s_phase_aligned_cached_discovery.consensus_mask.arrays.npz \
  --regions-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.regional_phase_delay.arrays.npz \
  --sg-windows 5,7,11 \
  --lag-frames 8,12,16 \
  --gaussian-sigma-px 0.8 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_original38_photometry_transform_comparison
```

This is a descriptive challenger comparison. It uses even windows for
selection and shows odd-window results separately, permits no selected
challenger, and explicitly reports that no transform-family null was run.
Regional spatial standard deviation is strongest for frequency presence;
matched projection and signed derivative/difference traces remain separate
challengers for polarity-preserving event work.

### Conditional Transform-Family Calibration

The transform-family calibration reruns all 14 trace constructions on
validity-aware spatial-block surrogates. It is conditional on the previously
learned masks and compact-mask frequencies; it is not a full-pipeline null and
cannot establish cardiac identity or event timing.

```bash
scripts/py playgrounds/heartrate_stabilization/calibrate_heart_photometry_transform_family.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz \
  --longitudinal-csv playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_frozen_masks_60s_2p0_4p0.longitudinal.csv \
  --original-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.dynamic_support.arrays.npz \
  --consensus-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_30000f_consensus5x60s_phase_aligned_cached_discovery.consensus_mask.arrays.npz \
  --regions-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.regional_phase_delay.arrays.npz \
  --surrogate-count 39 \
  --surrogate-batch-size 4 \
  --surrogate-workers 4 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_original38_photometry_family_conditional_null39_v2
```

Each batch is identity checked and resumable. The full-recording matched
projection makes this expensive; use the saved batches rather than restarting
after interruption. The earlier partial `photometry_family_null39` batches
predate the validity fix and must not be reused or cited.

### Full-Recording Simple PCA Overlay

Render the exploratory no-fold 38-pixel PCA result across the complete moving
recording using the saved one-minute candidate-frequency table:

```bash
scripts/py playgrounds/heartrate_stabilization/render_segmented_cache_pca_overlay.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz \
  --analysis-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.dynamic_support.arrays.npz \
  --analysis-mask-key original_38_mask \
  --band-hz 2 4 \
  --window-csv playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_simple_segmented_pca_original38_60s_2p0_4p0.windows.csv \
  --output-fps 25 \
  --frame-stride 3 \
  --crf 20 \
  --output playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_simple_segmented_pca_original38_60s_2p0_4p0_full_overlay_stride3_25fps.mp4
```

This covers the entire source time range while encoding every third frame. At
`25 fps`, it plays `1.333x` slower than source time. Invalid post-edge samples
and unscorable one-minute windows intentionally have no analysis colors. The
sidecar JSON records the stride, playback timing, mask, band, and window table.

### Photometry Motion Controls

This runner preserves the frozen-frequency diagnostic but also lets each source
find its own maximum over the same frequency grid. Observed and control traces
are rescored on identical valid rows and logical blocks; only comparisons that
pass the paired-support gate are eligible to address whether a motion control
reproduces an adaptive peak. Matched projection remains ineligible until its
spatial weights are refit separately at every searched frequency.

```bash
scripts/py playgrounds/heartrate_stabilization/evaluate_heart_photometry_motion_controls.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz \
  --longitudinal-csv playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_frozen_masks_60s_2p0_4p0.longitudinal.csv \
  --original-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.dynamic_support.arrays.npz \
  --frequency-min-hz 2.0 \
  --frequency-max-hz 4.0 \
  --frequency-step-hz 0.05 \
  --window-indices 0,7,14,18,22 \
  --minimum-paired-block-count 4 \
  --minimum-paired-block-fraction 0.5 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_original38_photometry_motion_controls_paired_representative5_v3
```

These are limited first-order coordinate-motion and local static-template
controls, not optical flow and not a calibrated null distribution. In the
checked five-minute subset, the eligible full-mask median searched-maximum
observed/control ratios against gradient/displacement motion were `1.22`
(spatial SD), `1.13` (SG-11), `0.94` (lag-12), and `0.99` (lag-16), each across
only `3/5` eligible windows after excluding support failures and search-boundary
maxima. The observed signal is therefore not convincingly separated from
measured motion.

### Conditional Injection/Recovery

The injection runner adds known Mono8-DN oscillations after cached bilinear
sampling. It tests transform/frequency recovery with frozen masks; it does not
rerun localization, estimate a false-positive rate, or validate events.

```bash
scripts/py playgrounds/heartrate_stabilization/inject_recover_heart_photometry.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz \
  --longitudinal-csv playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_frozen_masks_60s_2p0_4p0.longitudinal.csv \
  --original-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.dynamic_support.arrays.npz \
  --regions-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.regional_phase_delay.arrays.npz \
  --patterns traveling_wave \
  --injection-frequencies-hz 3.25 \
  --amplitudes-dn 0,1.5 \
  --activity-modes continuous,intermittent \
  --traveling-band-counts 3 \
  --traveling-directions 1 \
  --traveling-phase-span-deg 120 \
  --frequency-min-hz 2.75 \
  --frequency-max-hz 3.75 \
  --frequency-step-hz 0.25 \
  --window-indices 0,1,2,3,4,5 \
  --frame-count 1600 \
  --workers 1 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_original38_photometry_conditional_smoke_v4_final
```

The `0 DN` job is one fixed-background sanity control. It cannot estimate a
false-positive rate. Broad sensitivity studies should use deterministic batches
and independent background recordings. In this smoke, continuous and
intermittent `1.5 DN` waves were recovered at `3.25 Hz`, but every phase,
timing, and direction result was withheld because individual regions/bands did
not pass the target-to-sideband support gate in enough windows. The `0 DN` job
did not confirm.

### Moving lower-mask equal-mean overlay

Render the saved lower 20-pixel equal-mean trace over the full moving-fish
recording without refitting the projection:

```bash
scripts/py playgrounds/heartrate_stabilization/render_moving_lower_mean_overlay.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz \
  --full-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_36000_30000f_exploratory_four_mask_2p0_4p0.mask_comparison.arrays.npz \
  --lower-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.regional_phase_delay.arrays.npz \
  --comparison-arrays playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_comparison.arrays.npz \
  --comparison-summary playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_comparison.summary.json \
  --window-csv playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_comparison.windows.csv \
  --frame-stride 3 \
  --output-fps 25 \
  --ffmpeg-preset veryfast \
  --output playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_lower20_equal_mean_4s_8s_full_overlay_stride3_25fps.mp4
```

The renderer verifies the lower membership and pixel count against the saved
comparison. It colors all lower-mask pixels with one regional mean, preserves
the 38-pixel common-valid segment contract, and hides color across invalid
rows and filter edges. The output is a candidate-oscillator inspection video,
not validated heartbeat timing.

Diagnose whether literal-mean `4 s` rate excursions coincide with cached
tracking degradation:

```bash
MPLCONFIGDIR=/tmp/palette-mpl \
scripts/py playgrounds/heartrate_stabilization/analyze_moving_lower_window_excursions.py \
  --dataset-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f.local_pixel_matrix.npz \
  --full-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_36000_30000f_exploratory_four_mask_2p0_4p0.mask_comparison.arrays.npz \
  --lower-mask-npz playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_chase_start_30000_3000f_dynamic_support.regional_phase_delay.arrays.npz \
  --comparison-arrays playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_with_raw_lower_comparison.arrays.npz \
  --window-csv playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_original38_equal_mean_upper_lower_with_raw_lower_comparison.windows.csv \
  --method lower_raw_mean \
  --window-seconds 4 \
  --stable-threshold-bpm 6 \
  --excursion-threshold-bpm 24 \
  --output-prefix playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_start_0_140035f_lower_raw_mean_4s_excursion_tracking_diagnostic
```

The classification is a same-cache descriptive comparison, not a calibrated
quality gate. The tracking metrics can detect jitter and transformation stress
but cannot prove that a smoothly moving mask remains on the correct anatomy.

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

The reusable inference/statistics core lives in
`src/fisheye/analysis/local_rostral_heartrate.py` and
`src/fisheye/analysis/dynamic_heart_support.py`. Do not promote their outputs to
a production metric until the workflow can show:

- low residual motion of the swim bladder and eye midpoint in stabilized view;
- visually stable heart ROI over a representative rotation interval;
- reduced motion artifacts in the intensity trace compared with fixed crop
  pixels;
- explicit invalid-frame handling when keypoints or crop metadata are missing.
- calibrated false-positive and sensitivity estimates across multiple real
  null recordings;
- reproducible cross-fit clusters and independent biological event timing.
