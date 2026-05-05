# Crimson Stimulus Step Read Contract

Purpose: define the Palette-side read surface Crimson should use for protocol
step timing and stimulus geometry. This replaces ad hoc parsing of
`protocol_json` or event rows in downstream viewers.

## Canonical Path

Read canonical step metadata from:

```text
analysis/stimulus_runs/<stimulus_run>/steps/step_<step_index>/
```

Resolve `<stimulus_run>` by:

1. User-selected run, if provided.
2. `analysis/stimulus_runs.attrs["latest"]`, if present.
3. Otherwise, enumerate available child groups and present them to the user.

Palette Zarrs may have stale consolidated metadata. Crimson should merge normal
Zarr group enumeration with local filesystem checks for
`analysis/stimulus_runs/*/zarr.json` and
`analysis/stimulus_runs/<run>/steps/step_*/zarr.json` when reading local
archives.

## Step Group Attrs

Each `step_<i>` group stores source-derived attrs:

```text
metadata_schema_version
step_index
step_name
stimulus_mode_id
stimulus_mode
start_camera_frame
end_camera_frame
duration_s
raw_protocol_params_json
```

`start_camera_frame` and `end_camera_frame` are camera-frame indices. Crimson can
use these to draw protocol-step spans on timelines and to seek to step starts.

`raw_protocol_params_json` is provenance/debug context. Crimson should prefer
the normalized mode-specific attrs below for display and behavior.

## Moving Grating Subgroup

Moving grating steps add:

```text
analysis/stimulus_runs/<run>/steps/step_<i>/moving_grating/
```

Important attrs:

```text
orientation_degrees_authored
grating_direction_camera_deg
camera_to_projector_offset_deg
direction_mapping_status
direction_mapping_validated
speed_mm_s
speed_pps
spatial_freq_cycles_per_mm
spatial_freq_rpp
temporal_frequency_hz
duty_cycle
```

For camera overlays and OMR interpretation, Crimson should use
`grating_direction_camera_deg`. This is the Palette-normalized camera-space
best estimate. Do not reinterpret `orientation_degrees_authored` as camera-space
motion direction.

If `direction_mapping_status == "configured_camera_offset"`, Palette applied the
recorded `camera_to_projector_offset_deg` correction when materializing the
step. If `direction_mapping_status == "unvalidated_default_zero_offset"`, the
field is only the Citrus-authored orientation with no explicit projector/camera
offset. Crimson can still display it, but should label it as unvalidated or
prefer a response run whose provenance records the offset.

## Concentric Grating Subgroup

Concentric grating steps add:

```text
analysis/stimulus_runs/<run>/steps/step_<i>/concentric_grating/
```

Important attrs:

```text
stimulus_role
radial_polarity_authored
radial_sign_authored
radial_polarity_validated
center_x_px
center_y_px
center_coordinate_frame
center_source
center_x_mm
center_y_mm
speed_mm_s
speed_pps
spatial_freq_cycles_per_mm
spatial_freq_rpp
temporal_frequency_hz
target_radius_min_mm
target_radius_max_mm
centering_success_fraction_threshold
```

`radial_polarity_authored` is the Citrus-authored intent
(`expanding` or `contracting`). `radial_polarity_validated=false` means Palette
has not independently verified shader/rendered polarity. Crimson should label it
as authored intent unless validation metadata is added later.

## Relationship To Stimulus Response Runs

`analysis/stimulus_runs/<run>/steps` is the source stimulus metadata.

`analysis/stimulus_response_runs/<run>/steps` contains derived fish response
metrics. Crimson should use stimulus-run steps for protocol timing/geometry even
when no stimulus-response run exists.

When both are loaded:

```text
stimulus_response_runs/<response_run>.attrs["source_stimulus_run"]
```

links the response run back to the source stimulus run. Crimson can join by
`step_index`.

For derived moving-grating OMR metrics, use:

```text
analysis/stimulus_response_runs/<response_run>/steps/step_<i>/grating/omr/
```

The current implementation writes:

```text
omr.attrs["method_version"] = "stimulus_response_omr_v3"
omr.attrs["stimulus_direction_deg"]          # camera-space direction used for projections
omr.attrs["grating_direction_camera_deg"]    # copied from canonical stimulus step when available
omr.attrs["orientation_degrees_authored"]    # Citrus/projector-authored direction
omr.attrs["camera_to_projector_offset_deg"]
omr.attrs["direction_mapping_source"]
omr.attrs["direction_mapping_status"]
omr.attrs["direction_mapping_validated"]
omr.attrs["detector_estimator_policy"]
omr.attrs["window_lengths_s"]
omr.attrs["early_response_window_lengths_s"]
omr/per_fish/omr_path_index
omr/per_fish/bout_path_index
omr/per_fish/bout_fraction_correct_weighted_by_path
omr/per_fish/time_choice_index
omr/per_fish/first_aligned_bout_latency_s
omr/per_bout/per_bout_omr_score
omr/per_bout/correct_label
omr/windows/omr_path_index
omr/early_windows/omr_path_index
```

Crimson should treat `stimulus_direction_deg` as the exact camera-space vector
used by Palette for OMR projections. If `direction_mapping_validated=false`,
display the response metrics but surface the direction as unvalidated. The
per-bout `correct_label` convention is `+1` aligned, `-1` opposing, and `0`
ambiguous.

For derived concentric-grating radial response metrics, use:

```text
analysis/stimulus_response_runs/<response_run>/steps/step_<i>/concentric_grating/radial_omr/
```

The first numeric implementation writes:

```text
radial_omr.attrs["method_version"] = "stimulus_response_concentric_radial_omr_v1"
radial_omr.attrs["stimulus_radial_polarity"] = "expanding" | "contracting"
radial_omr.attrs["stimulus_radial_sign"] = +1 | -1
radial_omr.attrs["stimulus_radial_polarity_validated"] = false for current backfilled Citrus recordings
radial_omr/per_fish/omr_path_index
radial_omr/per_fish/tangential_bias_index
radial_omr/per_bout/radial_omr_score
radial_omr/per_bout/omr_label
```

The response metrics use outward-positive physical radial components, with
`stimulus_aligned = stimulus_radial_sign * outward_radial`. This is separate
from the older centering-style `concentric_grating/per_frame/radial_speed_mm_s`,
where positive means approaching the center.

## Recommended Crimson UI Slice

1. Add a stimulus-step layer independent of OMR metrics.
2. Draw current protocol step spans on speed/behavior timelines using
   `start_camera_frame` and `end_camera_frame`.
3. For moving gratings, draw the camera-space grating direction from
   `moving_grating.attrs["grating_direction_camera_deg"]`.
4. For concentric gratings, draw the center point and radial polarity from the
   `concentric_grating` subgroup.
5. If an OMR/stimulus-response run is selected, join response metrics to these
   source steps by `step_index`; do not require response metrics to display
   protocol geometry.

## Current Canary

The moving-grating canary used during development is:

```text
/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr
```

Default stimulus run:

```text
analysis/stimulus_runs/stimulus_20260209_084518
```

Default OMR response run:

```text
analysis/stimulus_response_runs/stimulus_response_tk_hyst4_low2_latch_s005_omr_canary
```
