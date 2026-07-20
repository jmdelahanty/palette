# Crimson Track Motion Read Contract

Date anchored: 2026-05-02

Purpose: define the Palette-side read contract Crimson should use for current
motion traces, swim-bout overlays, and per-bout metrics. This is the current
consumer-facing replacement for legacy Crimson `analysis/movement_runs` loading.

## Source Map

Use these run families for new Crimson readers:

| Need | Preferred Palette source |
| --- | --- |
| Per-track positions with coordinate/identity authority | `analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/` through the typed position-binding loader |
| Heading, speed, path distance, acceleration, and time-series validity | same physical run, but not a canonical read surface until its all-array payload/derivation seal is implemented and validated |
| Swim-bout event windows and detector traces | `analysis/swim_bout_runs/<run>/` through the logical candidate/signal resolver |
| Per-bout physical movement, heading, eye-gaze summaries | `analysis/bout_kinematics_runs/<run>/` |
| Body frame, snout/tail landmarks, B-spline, subject-shape QC | `analysis/subject_shape_runs/<run>/` |
| Protocol step timing and stimulus geometry | `analysis/stimulus_runs/<run>/steps/step_<i>/` |

Legacy `analysis/movement_runs` may remain a compatibility path for old
archives, but it is not the current Palette motion source.

Stimulus step details are specified in
[`crimson_stimulus_step_read_contract.md`](./crimson_stimulus_step_read_contract.md).

## Track Kinematics

`analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/` stores
positions plus gap-aware speed, distance, heading, and validity values. The
current typed canonical boundary authorizes only the row-bound
`positions_px`/optional `positions_mm` surfaces. The other persisted values are
numerically useful but are not yet a canonical read surface because the run
does not persist and freshly validate one exact all-array payload/derivation
seal.

For canonical future runs, Crimson must freshly validate all three gates:

```text
run.attrs["coordinate_binding_status"] == "bound_canonical_v2"
run.attrs["palette_run_completion_status"] == "complete"
run.attrs["stage_selector_eligible"] is True
```

An incomplete, staged, publishing, selector-ineligible, unsupported, or
ambiguously labelled run is not selectable. Convenience `latest*` attrs are
discovery hints only and do not override these gates. After resolving a pointer,
the reader must reopen the exact child and revalidate its live position
payloads, descriptors, identity/time records, and digests before reading
positions. These gates alone do not authorize derived motion arrays.

Required canonical identity and position fields:

```text
track_sample_key                       # int64 [track_id, acquisition_frame]
source_acquisition_frame_index         # authoritative camera-frame mapping
source_frame_interpolation             # exact acquisition-time lineage
source_instance_key                    # nullable observation lineage
source_row_index                       # exact selected immediate-source row
positions_px
```

`frame_indices` remains a compatibility alias of
`source_acquisition_frame_index`. Canonical consumers use `track_sample_key`
and `source_acquisition_frame_index`; row offset and `frame_indices` alone are
not identity.

`positions_mm` and the `*_mm` motion fields are optional. They are present only
when Palette bound an exact compatible typed physical-frame calibration. Their
absence means physical output is unavailable, not that Crimson should infer a
scale. Crimson must never apply a run-level `pixel_to_mm` scalar or a resolution
ratio to create missing physical coordinates.

### Position coordinate contract

Every canonical `positions_px` or `positions_mm` array owns a
`coordinate_descriptor` and digest. Crimson must validate the array-owned
descriptor, its row identity, exact reference-extent/frame records, transform
chain, and derivation records before use. See
[`coordinate_metadata_framework.md`](./coordinate_metadata_framework.md).

For `positions_px`:

- `space_id == "source_camera_image_px"` with
  `source_camera_overlay.status == "direct"` may be drawn directly on the exact
  referenced source-camera frame;
- `source_camera_overlay.status == "requires_transform"` requires applying the
  persisted ordered direction-labelled chain;
- `not_suitable`, an unsupported profile, a stale digest, or missing evidence
  fails closed.

Canonical offline track publication currently requires an exact dtype-preserving
subset/reorder of the selected crop producer's persisted source-camera
`centers_img_xy`. It does not reconstruct positions from normalized detection
centres, root dimensions, ROI dimensions, or simple scale factors.

Historical `coordinate_space = "camera"` or `"texture"` attrs are not
coordinate authority. A legacy reader may resolve them only through an explicit
compatibility mode with exact dimensions and lineage. New recordings and normal
Crimson paths do not use that adapter.

Renderer viewport/display coordinates remain ephemeral Crimson state. They are
not written back as Palette coordinate metadata.

The existing grouped speed/distance layout is:

```text
movement/speed/<raw|filtered|smoothed|averaged>/
  px
  mm
  frame_path_distance_px
  frame_path_distance_mm
  acceleration_px
  acceleration_mm
  smoothed_acceleration_px
  smoothed_acceleration_mm
```

This list describes physical discovery only; it is not coordinate or
derivation authority. Normal Crimson readers must currently fail closed for
these derived arrays. Once Palette publishes an exact all-array
payload/derivation manifest and the public reader freshly validates it, the
physical-layout resolution order will be:

1. Prefer `movement/speed/<level>/...`.
2. Fall back to flat `speed_<level>_px`, `speed_<level>_mm`,
   `frame_path_distance_<level>_px`, and `frame_path_distance_<level>_mm`.
3. For derivatives, fall back to `speed_derivatives/speed_<level>/...`.
4. Historical flat `acceleration_*` arrays are available only through an
   explicitly requested historical compatibility path; they are not a normal
   future-recording fallback.

`source_acquisition_frame_index` is the authoritative sparse row-to-camera-frame
lineage. Crimson must not assume track row index equals video frame index. Build
a frame-to-row lookup for interactive seeking and cross-check it against column
1 of `track_sample_key`. `frame_indices` is accepted only as the declared
compatibility alias and must be numerically identical.

Missing frames are gaps. Displaying them as zero speed is a UI choice, not the
stored analysis semantics.

## Swim-Bout Runs

`analysis/swim_bout_runs/<run>/` is the canonical bout segmentation candidate
surface. It answers: what events did this detector or speed level find?

Crimson should treat swim-bout data as a logical run/candidate/signal surface,
not as a fixed physical path. Palette currently has two physical layouts:

- hierarchical v1: `analysis/swim_bout_runs/<run>/<speed_level>/...`
- compact v2: `analysis/swim_bout_runs/<run>/indexes/*`,
  `tables/*`, and `signals/*`, with `attrs["layout"] ==
  "compact_tabular_v2"`

The detailed compact-v2 handoff is in
[`crimson_swim_bout_compact_v2_read_contract.md`](./crimson_swim_bout_compact_v2_read_contract.md).

Important fields:

```text
bouts
bout_points
inter_bout_intervals
global_metrics
detection_signal_mm_s        # present for transformed detector responses
speed_exponential_mm         # compatibility/plotting mirror when present
```

In compact v2, these logical fields are materialized differently:

```text
indexes/candidates
indexes/signal_variants
tables/bouts
tables/peak_events
tables/inter_bout_intervals
signals/detector_signal_mm_s
signals/detector_signal_signal_ids
signals/frame_indices              # historical or declared portable fallback
```

Crimson should create one selectable UI candidate per selected candidate row
and signal row, using `candidate_id` and `signal_id` as stable identity inside
the run. Do not require a physical `<speed_level>` subgroup when
`layout == "compact_tabular_v2"`.

For detector traces, new run-schema-8 outputs resolve the frame axis from the
versioned run-level `frame_axis_contract`. Existing schema-8 outputs may point
to the exact source track-kinematics compatibility `frame_indices` array
relative to the same Zarr root; that source array must itself declare
`source_acquisition_frame_index` as authority and match it exactly. Schema-7
runs continue to use embedded `signals/frame_indices`, and schema-8 runs may
declare that same path as an embedded portability fallback.
Use the resolution and fail-closed checks in the detailed compact-v2 handoff;
do not substitute a `latest` track pointer or reconstruct an `arange` axis.

The `bouts` table includes frame and timing boundaries such as:

```text
start_frame
end_frame
core_start_frame
core_end_frame
start_time_s
end_time_s
duration_s
observed_duration_s
path_length_mm
path_length_px
net_displacement_mm
net_displacement_px
peak_detection_signal_mm_s
peak_physical_speed_mm_s
```

Frame boundaries are authoritative for overlay rectangles and slicing. Optional
interpolated threshold times are annotations, not replacements for frame
boundaries.

### Detector vs Physical Metrics

`speed_exponential` is a detector response, not an independent physical speed
measurement. It can be useful for bout segmentation and visualization, but
biological speed, path length, and active-duration metrics should be read from
declared physical movement sources such as `speed_filtered` or from linked
`analysis/bout_kinematics_runs`.

Crimson should label detector traces as detector responses when plotted.

## Matching Track Kinematics To Swim Bouts

Given a selected track-kinematics run, track ID, and speed level, Crimson should
discover compatible swim-bout candidates by lineage attrs and/or compact signal
columns:

```text
source_track_kinematics_run
track_id
detection_signal_source_level
detection_signal_source_path
movement_metric_source_level
source_level
path_distance_source_level
signal_id
candidate_id
```

Direct matches:

- selected `filtered` speed maps to subgroup `speed_filtered`
- selected `smoothed` speed maps to subgroup `speed_smoothed`

Transformed matches:

- `speed_exponential` is compatible only when its attrs point back to the
  selected source speed, for example
  `detection_signal_source_level = "filtered"`.

Do not auto-select a bout subgroup whose detector source points at a different
speed trace than the selected track-kinematics speed.

## Bout Kinematics

`analysis/bout_kinematics_runs/<run>/` is downstream of one exact
track-kinematics run and one exact swim-bout candidate. It owns physical
per-bout measurement policy.

Use it for:

- physical active duration
- physical active path length
- physical active mean and peak speed
- pre/post position and heading windows
- within-bout heading changes
- optional eye-gaze summaries

Primary group:

```text
movement/per_bout_metrics/
```

Key fields include:

```text
source_start_frame
source_end_frame
source_core_start_frame
source_core_end_frame
physical_active_start_frame
physical_active_end_frame
physical_active_duration_s
physical_active_duration_s_interpolated
physical_active_path_length_mm
physical_active_path_length_px
physical_active_mean_speed_mm_s
physical_active_peak_speed_mm_s
physical_active_valid
failure_reason_bytes
```

Crimson should not mutate swim-bout runs when displaying these metrics. Bout
kinematics is a linked measurement layer, not a replacement segmentation layer.

## Subject Shape Boundary

`analysis/subject_shape_runs/<run>` is the geometry/QC surface for body axes,
B-splines, snout/tail landmarks, and body-frame overlays. It is not the default
source for speed or path distance and must not be used to bypass a missing
track-motion payload seal.

Subject-shape fallback motion labels are acceptable only as a preview/debug path
for archives missing `analysis/track_kinematics_runs`. If used, label them as
derived preview values.

## Recommended Crimson UI Slice

1. Discover `analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/`
   and validate its typed position binding.
2. Let the user select a track. Expose speed-level selection only after the
   exact derived-motion payload/derivation seal is supported and validates.
3. Build a sparse frame-to-track-row lookup from
   `source_acquisition_frame_index`, cross-checked against `track_sample_key`.
4. Draw the validated position in its declared overlay frame. After the
   derived-motion seal exists and validates, draw per-frame motion labels:
   - heading from `heading_degrees` or `smoothed_heading_degrees`
   - speed from selected `movement/speed/<level>/mm` or fallback flat arrays
   - px/s fallback only when mm/s is unavailable, with honest units
5. Discover matching `analysis/swim_bout_runs` candidates and overlay bout
   windows from logical `bouts/start_frame` and `bouts/end_frame`, regardless
   of whether the run is hierarchical v1 or compact v2.
6. Load linked `analysis/bout_kinematics_runs` for per-bout measurement tables
   and histograms when present.
7. Keep subject-shape overlays independent from motion traces.

## Canary

Current feeding canary:

```text
/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr
```

Useful current sources:

```text
analysis/track_kinematics_runs/offline/tk_hyst4_low2_s005/tracks/id_0
analysis/swim_bout_runs/<candidate>/<speed_level>
analysis/bout_kinematics_runs/<candidate>
analysis/subject_shape_runs/<candidate>
```

The archive has no useful `analysis/movement_runs` path for current Crimson
motion display.
