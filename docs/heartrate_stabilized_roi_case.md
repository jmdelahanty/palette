# Heartrate Stabilized ROI Case
<!-- contract-meta
status: draft
last_updated: 2026-07-10
-->

Purpose: document the concrete heartrate use case that needs a fish-attached
ROI workflow over acquisition crop-video pixels.

Current local-rostral ROI status and results are summarized in
[`heartrate_frozen_mask_validation_status_2026-07-10.md`](heartrate_frozen_mask_validation_status_2026-07-10.md).
The detailed implementation history remains in
[`heartrate_local_rostral_roi_status_2026-07-09.md`](heartrate_local_rostral_roi_status_2026-07-09.md).

## Case

The example recording lives at:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording
```

It contains:

- a lossless acquisition crop video under `derived/external_crop_recorder/`;
- matching crop metadata in `*_crop_meta.csv`;
- an analysis zarr with refined `traditional_v2` keypoints.

The user-facing problem is that the online crop video is centered from a
framewise detector box. When the fish rotates, the detector box center can move
relative to the fish anatomy. A static heart ROI drawn in crop-video pixels can
therefore drift off the heart even when the crop video itself was recorded
losslessly.

The measurement need is:

```text
draw heart ROI in a stabilized fish/body-frame view
-> inverse-transform that ROI into each crop-video frame
-> sample original crop-video pixels for intensity
```

The stabilized view is a QC/drawing surface. It should not become the pixel
surface used for heartbeat photometry.

When subject masks are available, the stronger measurement need is:

```text
draw heart ROI in a stabilized fish/body-frame view
-> inverse-transform that ROI into each crop-video frame
-> project the subject mask into that same crop-video frame
-> sample original crop-video pixels under ROI AND mask
```

This handles the case where `refined_subject_masks_runs/<run>/masks_roi` is
larger than the online crop video. The mask remains ROI-local to its source
`crop_runs/<source_crop_run>` row; `source_crop_row_ids` and
`roi_coordinates_full` place that mask in full-frame pixels, and the online
crop metadata then maps the overlapping portion into crop-video pixels.

## Photometry Pixel Contract

The photometry signal should be computed from all valid pixels selected by the
current measurement surface:

```text
ROI-only signal = mean(original crop-video pixels under inverse-projected ROI)
masked signal   = mean(original crop-video pixels under inverse-projected ROI AND projected mask)
```

Do not treat the number of averaged pixels as a fixed constant or a direct
function of framerate. Framerate controls temporal sampling; the number of
spatial pixels controls signal-to-noise and anatomical specificity.

For the current playground outputs:

- the direct stabilized-video ROI map used `441` ROI pixels;
- the eye-excluded spectral map used `249` ROI pixels after keeping
  `subject_body` and excluding `eye_left,eye_right`;
- there is no hard-coded "use 50 mean pixels" rule in the playground.

The "50" value found while checking related literature is not a Palette
photometry pixel-count rule. In the ZACAF embryo paper, one "50" refers to the
number of videos used to train the model; another refers to pixel-scale
variation in manual ventricle segmentation. Neither is a rule for how many
pixels to average in this ROI photometry path.

Spatial pixel selection should instead be justified by anatomy and validation:

- start with the user-drawn stabilized ROI;
- intersect with `subject_body` when masks are available;
- exclude eye components when they overlap the ROI or dominate the spectral map;
- if selecting a top-K or thresholded subset of pixels, choose K by held-out
  stability and reproducibility across time windows, not by video fps.

## Framerate and Rhythm Contract

Framerate affects the time-series analysis, not the spatial pixel count.

Relevant rules:

- Nyquist max frequency is `fps / 2`; requested band maxima must be clamped
  below that limit.
- Basic spectral resolution is approximately `1 / window_seconds`.
- A 60-second window gives roughly `0.017 Hz`, or `1 bpm`, frequency spacing.
- A 10-second window gives roughly `0.1 Hz`, or `6 bpm`, frequency spacing.
- Lower fps can still support frequency-based HR detection, but it reduces the
  chance of sampling exact end-systolic/end-diastolic frames and weakens
  morphology-derived estimates.

The playground rhythm analyzer currently chooses the Welch PSD segment length
from time, not from a fixed sample count:

```text
nperseg = min(number_of_samples, max(64, round(fps * 8 seconds)))
```

At `100 fps`, this is an 800-sample segment. At `20 fps`, it is 160 samples.
At very low fps, the 64-sample floor can make the segment longer than 8 seconds;
that behavior should be made explicit if this path becomes production code.

For the current 60-second playground run, the signal summary was:

```text
fps: 100
sample_count: 6000
search_band_hz: 1.5..10.0
peak_frequency_hz: 1.75
peak_bpm: 105.0
peak_to_median_band_power: ~7.08
```

At `105 bpm`, a 10-second clip contains roughly 17.5 beats, and a 60-second
clip contains roughly 105 beats. The longer window is therefore much better for
checking whether a narrow peak is stable.

## Biological HR Prior

Do not hard-code a single zebrafish heart-rate band without recording stage,
temperature, anesthesia, and imaging conditions. A reasonable first-pass search
prior for this playground is broad:

```text
1.0..4.0 Hz = 60..240 bpm
```

Use a narrower exploratory band only when the trace or experimental context
justifies it, for example `1.5..2.5 Hz` around the current `1.75 Hz` peak.

Literature anchors:

- ZACAF 2021, 3-dpf embryonic zebrafish at `28.5 C`, anesthetized with
  tricaine, records from a 60-fps camera but evaluates stored `5`, `10`, and
  `20 fps` videos. It defines HR from the interval between identical successive
  ED or ES points and notes that higher fps improves the chance of capturing
  exact ED/ES frames.
  <https://arxiv.org/pdf/2102.12173>
- The later ZACAF transfer-learning paper records 2-dpf zebrafish at `250 fps`
  for `4.35 s` and says this roughly captured 8 cardiac cycles, implying about
  `1.84 Hz` or `110 bpm` for that setup.
  <https://arxiv.org/pdf/2402.09658>
- A MaPS embryonic zebrafish heart-imaging paper reports heartbeat frequency
  around `2.5..3 Hz`, or `150..180 bpm`, for `4.5..5.5 dpf` embryos.
  <https://arxiv.org/pdf/2011.01688>

The current example's `105 bpm` peak is therefore plausible, especially if the
fish is anesthetized or otherwise slower than the 4.5-5.5 dpf MaPS examples.

## Pixel Contribution Maps

There are two different pixel-map concepts in the playground:

- `map_roi_contributors.py` shows which pixels were sampleable and actually
  averaged under the ROI/mask geometry. It does not say which pixels support a
  heart-rate peak.
- `map_pixel_band_contributions.py` computes one trace per stabilized ROI pixel,
  band-passes each trace, and maps band power, correlation with the band-passed
  ROI mean, and signed covariance with that aggregate signal.

Red or high positive values in the correlation/signed-covariance maps mean that
the pixel fluctuated in phase with the aggregate ROI rhythm in the requested
band. They are evidence for the extracted rhythm under the current ROI and mask
choices, not independent proof of the biological HR. Because the aggregate
reference is computed from the same ROI pixels, eye edges or other moving
high-contrast structures can create strong apparent support. This is why the
current recommended spectral-map path excludes `eye_left` and `eye_right` before
interpreting pixel-level support.

## Related Existing Docs

This is not a new frame-domain problem. It is a concrete consumer of several
existing contracts and diagnostics:

- `docs/diagnostics/realtime_offline_detection_comparison_design_2026-06-17.md`
  records that `derived/external_crop_recorder/*_crop_meta.csv` is the primary
  realtime crop-recorder source, and that acquisition `recording_frame_id` is
  1-based while Palette offline `frame_indices` are 0-based.
- `docs/acquisition_crop_pose_training_workflow.md` records the acquisition
  crop-video row surfaces, including `source_crop_video_frame_indices`,
  `source_crop_meta_row_indices`, `source_recording_frame_ids`, and
  `source_crop_local_frame_ids`.
- `docs/notes/orange_crop_video_frame_contract_2026-06-25.md` records the
  frame-domain ambiguity: current crop MP4 frames align to CSV row order,
  `local_frame_id` is not the crop MP4 frame index, and Orange should emit an
  explicit `crop_video_frame_index`.
- `docs/diagnostics/acquisition_crop_video_integration_2026-06-17.md` records
  that acquisition crop videos are first-class recording media, but downstream
  runs must still declare their actual pixel source.
- `docs/body_frame_contract.md` defines the shared fish anatomical body-frame
  convention. The current keypoint-only playground uses the documented
  `swim_bladder -> midpoint(eye_left, eye_right)` forward anchor, and centers
  the stabilized view on the midpoint between that eye midpoint and the swim
  bladder keypoint.

## Frame-Domain Requirement

Do not join crop-video frames to keypoints by guessing from column names.

For the current heartrate example:

```text
crop MP4 frame index = crop_meta CSV row index = recording_frame_id - 1
refined_keypoints_runs/<run>/frame_indices = zero-based crop/source video frame index
camera_frame_id != keypoint frame_indices
local_frame_id != crop MP4 frame index
```

This was confirmed in the playground by first trying `camera_frame_id`, which
produced keypoints that did not overlay the fish, then switching the join to
`crop_video_frame_index`.

Consumers should expose the selected frame domain in config/provenance and fail
closed or visibly warn when keypoint overlays do not land on the fish.

## Playground

The prototype lives in:

```text
playgrounds/heartrate_stabilization/
```

The first scripts are intentionally diagnostic:

- `inspect_sources.py`: metadata and frame-domain inspection;
- `make_subset_clip.py`: short side-by-side crop/stabilized diagnostic clip;
- `render_stabilization_probe.py`: still-frame/contact-sheet probe;
- `measure_roi_signal_probe.py`: ROI intensity sampling from original crop
  video pixels.

The playground output directory is ignored by git. A representative short clip
can be generated without processing the full recording:

```bash
scripts/py playgrounds/heartrate_stabilization/make_subset_clip.py \
  --config playgrounds/heartrate_stabilization/config.example.toml \
  --frame-count 120 \
  --stride 2 \
  --output playgrounds/heartrate_stabilization/outputs/stabilization_subset_clip.mp4
```

## Promotion Criteria

Before promoting this from playground to a production analysis surface:

- persist the frame-domain mapping used to align crop video, crop metadata, and
  keypoints;
- store the body-frame estimator and semantic anchors following
  `docs/body_frame_contract.md`;
- preserve the distinction between drawing/QC pixels and measurement pixels;
- report invalid frames with explicit reasons, including missing keypoints,
  blank crop frames, missing crop metadata, and out-of-crop ROI polygons;
- validate the ROI intensity trace against a fixed crop-pixel baseline on a
  short subset before full-recording execution.
- record the temporal search band, fps, analysis window duration, and biological
  HR prior used for peak selection.
- record whether spectral pixel maps used all ROI pixels, mask-filtered pixels,
  or a top-K/thresholded subset, and how that subset was validated.
- preserve acquisition timestamps and invalid rows instead of compressing the
  time axis; interpolate only explicitly bounded short gaps.
- choose frequency, pixels, cluster, weights, polarity, and event threshold on
  discovery blocks, then apply them frozen to disjoint held-out blocks.
- calibrate the complete adaptive discovery search with a maximum cluster-mass
  surrogate null that permits an empty/no-estimate result.
- require cross-fit spatial reproducibility and matched physical-boundary,
  interior, body, global, and external controls before event extraction.
- report event coverage and explicit no-estimate intervals, and validate
  detection/localization/timing through source-raster injection recovery on real
  null backgrounds.
- for extended anatomical supports, learn per-pixel phase/polarity only on
  discovery blocks and confirm the frozen mask over predeclared frequency bounds
  on an independent interval.
- validate latent-trace event times against blinded manual contraction landmarks
  or a synchronized cardiac reference before reporting heart rate.

For mask-constrained photometry, additionally validate that:

- the selected mask run has `frame_indices`, `source_crop_run`,
  `source_crop_row_ids`, and semantic `mask_labels`;
- the `subject_body` projection lands on the fish in the online crop panel;
- `ROI AND mask` leaves enough pixels for a stable intensity estimate;
- missing mask rows are explicit in output diagnostics instead of silently
  becoming empty measurements.
