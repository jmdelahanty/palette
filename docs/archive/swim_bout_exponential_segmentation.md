# Swim Bout Segmentation From Exponential Response

This document describes how Palette segments swim bouts from the
`speed_exponential` detector signal written by
`fisheye.analysis.detect_bouts_multi_level`.

## Inputs

Bout detection starts from one track in an `analysis/track_kinematics_runs`
offline run. The detector reads the frame index vector and the stored speed
levels:

- `speed_raw_mm`
- `speed_filtered_mm`
- `speed_smoothed_mm`
- `speed_averaged_mm`

It then derives an additional detector candidate:

- `speed_exponential_mm`

The exponential candidate is not measured physical speed. It is a transformed
detection signal derived from a configured source speed level, usually
`speed_filtered_mm`.

## Exponential Response

For each valid consecutive frame, the exponential response is updated as:

```latex
\[
\alpha_i = 1 - \exp\left(-\frac{\Delta t_i}{\tau}\right)
\]

\[
y_i = \alpha_i x_i + (1 - \alpha_i)y_{i-1}
\]
```

where:

- \(x_i\) is the source speed at frame \(i\)
- \(y_i\) is the exponential response at frame \(i\)
- \(\tau\) is `exponential_tau_s`
- \(\Delta t_i = (f_i - f_{i-1})/\mathrm{fps}\)

Invalid samples or invalid transitions reset the response state. This prevents
motion from smearing across track gaps.

The recurrence is equivalent to convolution with a one-sided causal exponential
kernel:

```latex
\[
k(t) = \frac{1}{\tau}\exp\left(-\frac{t}{\tau}\right)H(t)
\]
```

Longer `tau` values broaden the response and extend its decay tail. Shorter
`tau` values track the source speed more closely.

## Detection Levels

`detect_bouts_multi_level` runs the selected detection method over every speed
level in the same run:

- `speed_raw`
- `speed_filtered`
- `speed_smoothed`
- `speed_averaged`
- `speed_exponential`

For `speed_exponential`, the detector signal is the transformed response trace.
Physical bout metrics still use the underlying physical movement source chosen
by `exponential_source_level`. For the default source, this means the
exponential subgroup uses filtered path-distance and filtered physical speed for
movement metrics, while using `speed_exponential_mm` to define or score the
detected event.

Detector-window duration is still owned by the swim-bout candidate. Stricter
physical active-motion duration, path length, mean speed, and peak speed for a
selected bout candidate should be read from
`analysis/bout_kinematics_runs/<run>/movement/per_bout_metrics/`, which records
the physical source signal and boundary constraint used for measurement.

## Threshold Segmentation

With `--method threshold`, bouts are threshold-connected components in the
selected detector signal.

The algorithm is:

1. Mark samples where the detector signal is greater than `--threshold-mm`.
2. Treat NaN samples as below threshold.
3. Find contiguous above-threshold regions.
4. Merge neighboring regions if the gap is shorter than the configured
   minimum gap.
5. Drop regions whose core duration is shorter than `--min-bout-duration`.
6. Write the remaining regions as bouts.

For threshold mode, `core_start_frame` and `core_end_frame` describe the
above-threshold core. If `--boundary-mode threshold` is used, the stored
`start_frame` and `end_frame` match that core. If `--boundary-mode local_minimum`
is used, the stored outer boundary can expand to nearby local minima while the
core fields preserve the threshold crossing region.

Threshold mode is simple and stable, but exponential tails can bridge visually
separate movement pulses. If the response does not fall below threshold between
two pulses, threshold mode will keep them in one connected bout.

## Peak-Event Segmentation

With `--method peak_event`, bouts are defined as one event per accepted peak in
the detector signal. This is usually the more useful way to segment from
`speed_exponential` when the goal is to separate discrete movement pulses.

The algorithm is:

1. Replace invalid detector-signal samples with zero for peak finding.
2. Convert `--min-peak-distance-s` into frames using
   `ceil(seconds * fps)`.
3. Run `scipy.signal.find_peaks` with:
   - `height = --min-peak-height-mm-s`, if provided
   - `prominence = --min-peak-prominence-mm-s`
   - `distance = resolved min peak distance frames`
4. For each accepted peak, compute a peak-width envelope with
   `scipy.signal.peak_widths` using `--peak-width-rel-height`.
5. Convert the interpolated left and right width positions into integer sample
   boundaries.
6. If neighboring peak-width envelopes overlap, split them at the local minimum
   between the two peak centers.
7. Drop events shorter than `--min-bout-duration`.
8. Write one row to `bouts` and one aligned row to `peak_events` for each
   accepted event.

The current boundary mode for peak events is
`relative_prominence_width`. Larger `peak_width_rel_height` values measure
farther down the peak prominence and therefore produce wider event envelopes.
For example, `0.98` generally captures more of the decay tail than `0.90`.

Peak-event mode does not use the threshold gap-merge policy. Separation is
driven by peak acceptance, peak distance, prominence, and the valley split when
width envelopes overlap.

## Stored Outputs

Each run is written under:

```text
analysis/swim_bout_runs/<run_name>/<speed_level>/
```

For `speed_exponential`, the subgroup includes:

- `detection_signal_mm_s`: the exponential response trace
- `bouts`: the segmented bout table
- `peak_events`: peak-event metadata, populated for `--method peak_event`
- `inter_bout_intervals`
- `inter_bout_interval_histogram`
- `global_metrics`
- `bout_points`

Important `bouts` fields include:

- `start_frame`, `end_frame`: stored event boundary
- `core_start_frame`, `core_end_frame`: core detector boundary
- `duration_s`, `core_duration_s`
- `path_length_mm`, `path_length_px`
- `mean_speed_mm_s`
- `peak_detection_signal_mm_s`
- `peak_physical_speed_mm_s`
- `valid_transition_fraction`
- `gap_censored`

For `speed_exponential`, `peak_detection_signal_mm_s` is the maximum value of
the exponential detector signal inside the event. `peak_physical_speed_mm_s` is
the maximum value of the underlying physical speed source inside the event.
These can differ because the exponential response is a transformed detector
signal, not the primary speed measurement.

`duration_s` and `observed_duration_s` in this table are detector-boundary
durations. They should not be interpreted as physical active-motion duration
when the detector signal is transformed or broadened.

Important `peak_events` fields include:

- `peak_frame`
- `peak_time_s`
- `peak_signal_value_mm_s`
- `peak_prominence_mm_s`
- `peak_width_samples`
- `peak_width_s`
- `peak_width_height_mm_s`
- `left_width_frame_interpolated`
- `right_width_frame_interpolated`
- `left_base_frame`
- `right_base_frame`

## Practical Interpretation

Use the exponential response as a bout-detection signal, not as a replacement
for measured fish speed.

Threshold segmentation asks:

```text
When is the detector signal above baseline?
```

Peak-event segmentation asks:

```text
Which prominent movement pulses are present?
```

For exponential traces, threshold segmentation can merge separate pulses through
a low tail. Peak-event segmentation can separate those pulses, but it can also
over-segment small ripples on the tail of a large bout if peak prominence,
height, or distance are too permissive.

Useful tuning knobs for `speed_exponential` peak-event runs are:

- `--exponential-tau-s`: controls the response tail
- `--exponential-source-level`: chooses the source speed level
- `--min-peak-height-mm-s`: absolute minimum peak height
- `--min-peak-prominence-mm-s`: how strongly a peak must stand out
- `--min-peak-distance-s`: minimum spacing between accepted peaks
- `--peak-width-rel-height`: controls event envelope width
- `--min-bout-duration`: drops very short envelopes

## Example Command Shape

```bash
scripts/py -m fisheye.analysis.detect_bouts_multi_level "$ZARR" \
  --track-kinematics-run latest \
  --method peak_event \
  --default-level exponential \
  --exponential-source-level filtered \
  --exponential-tau-s 0.025 \
  --min-peak-height-mm-s 1.0 \
  --min-peak-prominence-mm-s 5.0 \
  --min-peak-distance-s 0.10 \
  --peak-width-rel-height 0.98 \
  --min-bout-duration 0.05
```

This produces a `speed_exponential` subgroup where the event boundaries come
from accepted peaks in the causal exponential response, while physical bout
metrics remain tied back to the source movement arrays.
