# Swim Bout Peak-Event Detector Design

Date anchored: 2026-04-27

Status: first implementation slice available; valley-depth splitting remains future work

Purpose: define an additive peak/event swim-bout detector that complements the
current threshold-connected-component detector without replacing it.

## Motivation

The current `detect_bouts_multi_level` threshold detector asks:

```text
When is the selected speed signal above a threshold?
```

That is useful and defensible for measuring periods of movement above baseline,
but recent canary review exposed standard threshold-detector failure modes:

- a low threshold can make exponential tails or low-amplitude noise bridge two
  visually distinct movement pulses
- a minimum-gap rule can merge two peaks if the below-threshold valley is brief
- two peaks can remain one threshold-connected component when the valley stays
  above the absolute threshold
- changing `min_gap_duration_s` affects only threshold-separated regions; it
  cannot split two peaks that never fall below threshold

The additional detector should ask a different question:

```text
Where are the discrete movement pulses in the speed signal?
```

That is a peak/event detector. It should live beside the threshold detector as a
new `detection_method`, not as hidden behavior inside the existing threshold
path.

## Vocabulary

- `signal`: the one-dimensional speed-like trace used for event detection,
  such as `speed_filtered` or `speed_exponential`.
- `peak`: a local maximum in the signal.
- `peak_height`: the absolute signal value at a peak.
- `prominence`: how much a peak stands out above nearby valleys. This is often
  more useful than raw height for noisy traces.
- `valley`: a local minimum between two neighboring peaks.
- `valley_depth`: how far the signal drops between peaks.
- `minimum peak distance`: the minimum allowed time between candidate peaks.
- `event boundary`: the start/end frame assigned to the movement pulse around a
  peak.
- `refractory period`: a short interval after an event during which nearby
  peaks may be merged or ignored.
- `threshold-connected component`: a contiguous region where the signal is
  above an absolute threshold.

These are signal-processing terms, not biological conclusions. The biological
interpretation comes later in `bout_kinematics_runs`.

## Peak Detection Signal Vs Physical Speed

Peak detection is not the same operation as computing physical movement metrics
inside an already detected bout.

`peak_detection_signal_mm_s` is a detector-signal summary. It assumes a bout
interval already exists and answers:

```text
What was the maximum value of the signal that defined this bout?
```

`peak_physical_speed_mm_s` answers the corresponding physical-speed question
using the declared movement source inside the same boundaries.

`scipy.signal.find_peaks` is a detector primitive. It runs before event
intervals are finalized and answers:

```text
Where are the local maxima in this one-dimensional signal?
```

Then it can filter those local maxima using signal-shape criteria:

- `height`: the absolute signal value at the peak
- `prominence`: how much the peak stands above nearby valleys
- `distance`: the minimum sample spacing between accepted peaks
- `width`: how broad the peak is at a chosen relative height or prominence

For example:

```text
0, 1, 10, 2, 9, 1, 0
```

A low threshold detector may treat this as one threshold-connected bout because
the signal never returns to baseline between `10` and `9`. A peak detector can
still identify two local maxima because it is looking for movement pulses and
their surrounding valleys, not only contiguous time above an absolute threshold.

This is why `peak_event` is useful as an additional detector family. It can
separate visually distinct movement pulses inside one long above-threshold
region, while `peak_detection_signal_mm_s` only summarizes a bout after the bout
boundary has already been chosen.

## Detector Families

Palette should treat these as distinct detector families:

| Detector | Question | Strength | Weakness |
| --- | --- | --- | --- |
| `threshold` | When is movement above baseline? | Simple, stable, interpretable active intervals | Can merge distinct pulses through low tails/noise |
| `peak` | Which peaks are prominent enough? | Already available as a rough peak-width path | Current implementation is not tuned as the main review path |
| `peak_event` | Which discrete movement pulses are present, and what envelope belongs to each? | Matches visual "separate bout" intuition better | More parameters; needs careful provenance and validation |
| future template/matched filter | Which events match a learned bout waveform? | Could model bout kinetics directly | Premature before simpler peak-event behavior is understood |

The first implementation slice is `peak_event`.

## Proposed Storage Surface

Peak-event detection should write to the existing canonical segmentation
surface:

```text
analysis/swim_bout_runs/<run_name>/<speed_level>/
```

It sets:

```text
detection_method = "peak_event"
method_version = "detect_bouts_multi_level.v7"
swim_bout_run_schema_version = 6
peak_event_schema_version = 1
```

It should not write heading, pre/post position, or Johnson-style metrics. Those
remain linked downstream outputs in:

```text
analysis/bout_kinematics_runs/<run_name>/
```

## Source Signal Policy

The detector may run on any stored speed level:

- `speed_filtered`: closest to the hysteresis-filtered measured movement trace
- `speed_exponential`: causal response trace that can better resemble bout
  rise/decay shape
- `speed_smoothed` / `speed_averaged`: useful for comparison, but less
  attractive for onset-sensitive detection because smoothing can broaden events

The detector signal is for segmentation only. Biological movement metrics such
as path length, net displacement, and mean speed should continue to be computed
from `track_kinematics` path-distance and position arrays, not from the
transformed detector signal.

Each subgroup persists this distinction in metadata:

```text
detection_signal_transform_type = identity|convolution
detection_signal_source_path = ...
movement_metric_source_level = raw|filtered|smoothed
peak_detection_signal_field = peak_detection_signal_mm_s
peak_physical_speed_field = peak_physical_speed_mm_s
```

For transformed signals, the derived trace is stored as
`detection_signal_mm_s`. The transform-specific attrs, such as kernel family and
tau, define how that signal was produced.

## Proposed Algorithm

Initial `peak_event` is deliberately simple:

1. Select one speed-level signal.
2. Replace invalid samples with a detection-safe baseline for peak finding, but
   preserve validity masks for downstream bout quality fields.
3. Find candidate peaks with `scipy.signal.find_peaks`.
4. Filter peaks by explicit criteria:
   - `min_peak_height_mm_s`
   - `min_peak_prominence_mm_s`
   - `min_peak_distance_s`
5. For each accepted peak, define a core event around the peak using
   `peak_event_boundary_mode="relative_prominence_width"`.
6. Resolve overlapping peak-width envelopes by splitting adjacent event
   envelopes at the local minimum between peak centers. This avoids overlapping
   rows while preserving one event per accepted peak.
7. Write one row per accepted event to the existing `bouts` table and one
   aligned row to `peak_events`.

Future boundary policies may include:

- `low_threshold_crossing`: walk left/right until the signal crosses a low
  movement threshold
- `local_minimum`: expand to nearest local minima within a bounded window

This detector should not silently reuse `gap_merge_policy`. Gap merging is for
threshold-separated regions. Peak-event splitting is about waveform shape and
should have separate parameter names.

## Valley Split Policy

The phrase "visible valley" should become explicit parameters. A first policy
could be:

```text
shape_split_policy = "valley_depth"
```

Potential parameters:

- `min_valley_drop_mm_s`: absolute drop required between the lower neighboring
  peak and the intervening valley
- `min_valley_drop_fraction`: fractional drop required relative to the lower
  neighboring peak's prominence or height
- `min_valley_duration_s`: minimum time spent near the valley or below a valley
  threshold
- `valley_boundary_assignment`: where to split events, e.g. at the valley frame
  or at interpolated crossings around the valley

The first implementation should probably support either an absolute drop or a
fractional drop, not both as required criteria. Running both criteria at once
can become hard to reason about during tuning.

Recommended first-pass semantics:

```text
Split two neighboring peaks if:
  peak distance >= min_peak_distance_s
  and valley drop fraction >= min_valley_drop_fraction
```

Then use the valley frame as the boundary between the two events unless a
separate boundary mode gives a more specific start/end.

## Suggested Parameters

Initial CLI / provenance parameters:

```text
--method peak_event
--default-level exponential|filtered|smoothed|raw|averaged
--min-peak-height-mm-s <float>
--min-peak-prominence-mm-s <float>
--min-peak-distance-s <float>
--peak-event-boundary-mode relative_prominence_width
--peak-width-rel-height <float>
--shape-split-policy none
```

The important design rule is that every active decision must be represented in
provenance. `shape_split_policy="valley_depth"` and low-threshold boundary
parameters are intentionally not part of the first implementation slice.

## Proposed Bout Columns

The existing bout columns should remain:

- `start_frame`, `end_frame`
- `core_start_frame`, `core_end_frame`
- duration fields
- path length / net displacement fields
- validity and gap-censoring fields
- interpolated core threshold timing when applicable

Peak-event runs write method-specific fields in an aligned `peak_events` table
beside `bouts`:

- `peak_frame`
- `peak_time_s`
- `peak_signal_value_mm_s`
- `peak_prominence_mm_s`
- `peak_width_samples`
- `peak_width_s`
- `peak_width_height_mm_s`
- `left_ips` / `right_ips`
- `left_width_frame_interpolated` / `right_width_frame_interpolated`
- `left_base_frame` / `right_base_frame`
- `left_base_signal_value_mm_s` / `right_base_signal_value_mm_s`
- `boundary_mode`
- `shape_split_policy`

The first implementation stores `shape_split_policy`, not per-row
`shape_split_reason`, because no valley-depth split policy is active yet. The
key requirement is that peak provenance is inspectable without recomputing from
the source signal.

## Provenance Requirements

Peak-event runs must record:

- source `track_kinematics` run and `track_id`
- source speed level used for peak finding
- detector method and method version
- all peak finding parameters
- all boundary assignment parameters
- all split/merge parameters, including `shape_split_policy`
- whether event metrics were computed from transformed detector signal or from
  canonical path-distance arrays
- invalid sample handling policy

The run name should encode the most important tuning parameters, but provenance
is the source of truth.

## Marimo Review Surface

The track kinematics explorer now supports persisted peak-event boundary review:

- aligned `peak_events` columns are exposed as `peak_event_*` fields in the
  swim-bout dataframe
- `interpolated_peak_width` overlays render persisted
  `peak_event_left_width_time_s` and `peak_event_right_width_time_s`
- boundary review is no longer limited to sampled
  `bouts.start_time_s/end_time_s` rows

Remaining review improvements:

- render detected peaks as markers
- render valleys used for splitting as markers or vertical guide lines
- show peak height and prominence in hover data
- show event boundary mode and shape split policy in the candidate table
- allow comparing threshold and peak-event candidates for the same
  `track_kinematics` source

This should remain a review surface over persisted arrays. The notebook should
not recompute peak detection interactively except as a future tuning sandbox.

## Validation Plan

Unit tests should use synthetic signals where the expected behavior is obvious:

- one clean isolated peak produces one event
- two peaks separated by below-threshold baseline produce two events
- two peaks connected above threshold but separated by a deep valley split under
  `shape_split_policy="valley_depth"`
- two peaks connected by a shallow valley merge or remain one event
- close peaks below `min_peak_distance_s` merge or choose one peak according to
  policy
- NaN or invalid track gaps do not create artificial peaks or bridge events
- event boundaries do not overlap and remain ordered

Canary validation should compare at least:

- current threshold sampled-frame candidate
- threshold interpolated-gap candidate
- peak-event candidate on `speed_filtered`
- peak-event candidate on `speed_exponential`

The comparison should focus on concrete review examples, not just total bout
counts.

## Remaining Decisions

- Is `speed_exponential` with `tau=0.025`, prominence `5.0`, distance `0.10`,
  and `peak_width_rel_height=0.98` stable enough beyond the current canary?
- Should a later boundary mode use low-threshold crossing or local-minimum
  expansion?
- Should valley splitting use absolute drop, fractional drop, or a single
  prominence-based criterion?
- Should tail-fragment cleanup be handled by stricter peak acceptance, a
  refractory/merge policy, or a future valley-depth policy?

## Implemented First Slice

The first slice implements:

- `--method peak_event`
- `find_peaks` with `min_peak_height_mm_s`, `min_peak_prominence_mm_s`, and
  `min_peak_distance_s`
- boundaries from relative prominence width
- `shape_split_policy="none"` only
- method-specific peak metadata persisted

Open follow-up items after the first implementation slice:

- add Marimo overlay markers for persisted peak-events
- add `shape_split_policy="valley_depth"` only if peak-only behavior still
  leaves visually obvious merged pulses

## Canary Findings: 2026-04-27

The first real canary run was written on the 2026-01-28 arena 2 analysis Zarr:

```text
bouts_tk_hyst4_low2_s005_peak_event_exp_tau025_prom2_dist050
```

Key parameters:

```text
track_kinematics_run = tk_hyst4_low2_s005
method = peak_event
default_level = exponential
exponential_tau_s = 0.025
exponential_source_level = filtered
min_peak_height_mm_s = 1.0
min_peak_prominence_mm_s = 2.0
min_peak_distance_s = 0.05
peak_width_rel_height = 0.9
shape_split_policy = none
```

Persisted candidate counts from that run:

| Speed level | Bout count |
| --- | ---: |
| `speed_raw` | 1664 |
| `speed_filtered` | 974 |
| `speed_smoothed` | 585 |
| `speed_averaged` | 550 |
| `speed_exponential` | 604 |

The useful finding is that `peak_event` fixes some large threshold-bridging
failures. In examples where threshold segmentation produced one long event
because a low exponential tail stayed above threshold, peak-event detection
could split the region into separate movement pulses.

The new failure mode is over-segmentation of large bout tails. Small local
maxima on the tail of a large pulse can pass the current permissive peak
criteria and become separate events. This is expected for a plain
`find_peaks`-based detector with low prominence, low height, and short peak
distance. It means the first slice is functional but not tuned enough to become
the default review path.

Interpretation:

- `min_peak_distance_s` is a first-pass refractory-like filter, but it only
  constrains spacing between accepted peaks.
- A true refractory or merge policy would answer a different question: whether
  small peaks after a major event should be absorbed into the same event until
  the signal returns near baseline or another explicit reset condition.
- `shape_split_policy="valley_depth"` is not the first next step. Before adding
  another split mechanism, tune peak acceptance so obvious tail ripples do not
  become events.

Focused parameter sweep that was run:

```text
speed_level = speed_exponential
exponential_tau_s = 0.025
min_peak_height_mm_s = 1.0 or 2.0
min_peak_prominence_mm_s = 5.0, 8.0, 10.0
min_peak_distance_s = 0.10, 0.15, 0.20
```

The sweep was reviewed against the same concrete windows:

- the around-47.7 s threshold-bridging case where peak-event helped
- the around-51-53 s large-pulse/tail case where peak-event over-segmented

Focused sweep result:

| Run suffix | Total `speed_exponential` bouts | 47.7 s bridge window | 51-53 s tail window |
| --- | ---: | ---: | ---: |
| `prom2_dist050` | 604 | 3 events | 11 events |
| `prom5_dist010` | 500 | 2 events | 6 events |
| `prom8_dist015` | 460 | 2 events | 5 events |
| `prom10_dist020` | 434 | 2 events | 5 events |

This suggests the first-order tuning knobs are doing useful work. Increasing
prominence and peak distance suppresses many tail fragments while preserving the
large 47.7 s bridge split. The current review question is now biological rather
than purely technical: whether the remaining 51-53 s events are true separate
pulses or still excessive tail fragmentation.

For `relative_prominence_width`, `peak_width_rel_height` follows SciPy
`peak_widths` semantics: larger values measure farther down the peak prominence
and therefore usually produce wider event boundaries. A follow-up boundary
sweep kept the useful `prom5_dist010` peak acceptance settings and compared:

| Run suffix | `peak_width_rel_height` | 52.9 s large-peak event boundary |
| --- | ---: | --- |
| `prom5_dist010` | 0.90 | about 52.87-53.13 s |
| `prom5_dist010_w098` | 0.98 | about 52.80-53.37 s |
| `prom5_dist010_w100` | 1.00 | about 52.62-54.18 s |

This confirms that width contour mostly affects event envelope boundaries, not
accepted peak counts. `w098` may be a useful review candidate. `w100` can capture
long tails, but may over-extend events because it reaches the full prominence
base.

Historical canary choice from the first non-latch sweep: use
`bouts_tk_hyst4_low2_s005_peak_event_exp_tau025_prom5_dist010_w098` /
`speed_exponential` as a review candidate for downstream bout kinematics on
the 2026-01-28 arena 2 recording. This was a canary-level decision, not a
repository-wide default. A linked full bout-kinematics run was generated as:

```text
bk_tk_hyst4_low2_s005_peak_event_prom5_w098_interbout
```

Current feeding canary choice after adopting explicit latch hysteresis is:

```text
bouts_tk_hyst4_low2_latch_s005_peak_event_exp_tau025_prom4_dist010_w098_compact_v2_fresh_20260509
bk_tk_hyst4_low2_latch_s005_peak_event_prom4_w098_compact_v2_canary_20260510
```

That pair uses `min_peak_prominence_mm_s=4.0`, `min_peak_distance_s=0.10`,
`peak_width_rel_height=0.98`, compact swim-bout layout, and compact
bout-kinematics layout.

That downstream run uses the sampled frame boundaries stored in the source
`bouts` table for frame-indexed heading and position windows. The Marimo overlay
can display interpolated peak-width boundaries for visual review, but biological
metrics still operate on concrete frames unless a future analysis explicitly
interpolates heading or position at fractional boundary times.

Current bout-kinematics schema `7` preserves both layers: integer frame
boundaries remain the slicing contract, and aligned peak-event boundary context
is copied into downstream `source_peak_*` fields for provenance and review.

Near-zero speed display policy: preserve tiny nonzero values in smoothed or
response traces. They are properties of the transformation and should not be
rewritten to zero in stored arrays. Segmentation should control baseline
sensitivity through explicit detector parameters such as prominence, peak
distance, threshold, or a future review-only display deadband.

Next implementation steps, in order:

1. Add Marimo markers for persisted `peak_events` so accepted peaks, peak
   boundaries, and peak prominence can be inspected directly.
2. Review the selected `w098` candidate on additional recordings before treating
   it as more than a canary default.
3. If tail fragmentation remains a problem, add an explicit
   `peak_merge_policy`, such as `refractory_until_baseline` or
   `merge_tail_peaks`, with separate provenance.
4. Only after merge/refractory behavior is understood, consider
   `shape_split_policy="valley_depth"` for cases where one accepted envelope
   still contains multiple biologically plausible pulses.

Current status: keep `peak_event` as an exploratory candidate, not the default
operator path.
