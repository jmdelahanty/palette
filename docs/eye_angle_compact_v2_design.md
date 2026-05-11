# Eye-Angle Compact-Dense-V2 Design

<!-- design-meta
status: proposed
last_updated: 2026-05-11
-->

## Purpose

`analysis/eye_angle_runs` stores dense ROI- and frame-aligned eye-angle time
series. The current hierarchical layout is readable, but it materializes many
related scalar outputs as one array per name under `angles/roi` and
`angles/frame`, plus QA arrays and vector outputs in separate groups. This
creates a large Zarr metadata fanout for data that is naturally dense.

Compact-dense-v2 keeps the dense time-series surfaces, but stacks related
channels into a small number of arrays with explicit channel-index metadata.
The goal is object-count reduction without changing eye-angle semantics.

This is a resolver-first design. The writer is not changed yet. Readers should
be able to consume both current hierarchical-v1 and future compact-dense-v2
runs through `fisheye.analysis.eye_angle_io`.

## Physical Layout

Proposed future layout:

```text
analysis/eye_angle_runs/<run>/
  attrs:
    layout = "compact_dense_v2"
    schema_id = "analysis.eye_angle_runs"
    schema_version = 5
    eye_angle_output_schema.schema_version = 7

  angle_channel_index/
    name                         # string channel names
    representation               # major, gaze, eye_frame, nasal_gaze, centroid, legacy
    eye                          # left, right, binocular, none
    value_kind                   # angle, version, vergence, speed, acceleration, delta
    units                        # deg, deg/s, deg/s2
    source_channel               # optional canonical/parent channel
    formula                      # optional short formula for derived channels
    compatibility_alias_of       # optional target for legacy aliases

  roi_angles                     # float32, shape (n_roi_rows, n_angle_channels)
  frame_angles                   # float32, shape (n_frames, n_angle_channels)

  vector_channel_index/
    name                         # left_gaze_xy, right_gaze_xy, ...
    representation
    value_kind
    units

  roi_vectors                    # float32, shape (n_roi_rows, n_vector_channels, 2)
  frame_vectors                  # optional float32, shape (n_frames, n_vector_channels, 2)

  qa_channel_index/
    name                         # valid_left, valid_right, valid_frame, ...
    value_kind                   # bool, reason_code, warning_flag

  roi_qa                         # bool/int, shape (n_roi_rows, n_qa_channels)
  frame_qa                       # bool/int, shape (n_frames, n_qa_channels)

  support/
    frame_indices
    time_seconds
    frame_time_seconds
    ellipse_major
    ellipse_minor
    ellipse_ratio
    body_frame/...               # can remain grouped because it is semantic support
```

The channel index is the contract. Consumers must resolve by channel name, not
hard-code column positions.

## Logical API Contract

`load_eye_angle_run_tables(root, run_name=...)` should return the same logical
maps for both layouts:

```python
tables.roi["left_eye_angle_deg"]
tables.frame["left_gaze_deg"]
tables.qa_frame["valid_frame"]
tables.support["frame_indices"]
tables.source_paths["analysis/eye_angle_runs/<run>/angles/frame/left_gaze_deg"]
```

For compact-dense-v2, `source_paths` should point to the dense backing channel,
for example:

```text
analysis/eye_angle_runs/<run>/angles/frame/left_gaze_deg
  -> analysis/eye_angle_runs/<run>/frame_angles[:,12]
```

This lets downstream provenance remain layout-aware while preserving old
logical names.

## Current Output Inventory

### Canonical Orientation Channels

Persist these as first-class dense channels:

| Channel family | Examples | Notes |
| --- | --- | --- |
| Major-axis body-frame angles | `left_major_signed_deg`, `right_major_signed_deg` | Canonical resolved ellipse orientation. |
| Eye-frame Bianco/Engert angles | `left_eye_angle_deg`, `right_eye_angle_deg`, `vergence_eye_angle_deg` | Nasal-positive per eye; commonly consumed. |
| Gaze/body-frame angles | `left_gaze_signed_deg`, `right_gaze_signed_deg`, `vergence_gaze_signed_deg` | Derived from resolved major axis; important for visualization/debug. |
| Gaze vectors | `left_gaze_xy`, `right_gaze_xy` | Store in vector channel arrays, not scalar angle arrays. |

### Derived Compatibility Channels

Persist for v2 compatibility unless a later migration proves all consumers can
compute them from canonical channels:

| Channel family | Examples | Reason to keep initially |
| --- | --- | --- |
| BEAST/Johnson-style nasal gaze | `left_nasal_gaze_deg`, `right_nasal_gaze_deg`, `mean_eye_vergence_gaze_deg` | Existing analysis and docs use these names. |
| Version/vergence summaries | `version_major_deg`, `version_gaze_deg`, `vergence_gaze_deg` | Common plotting/query fields. |
| Smoothed variants | `*_smoothed` | Existing UI expects them directly. |
| Delta/speed/acceleration variants | `*_delta_deg`, `*_speed_deg_s`, `*_accel_deg_s2` | Existing schema exposes them; reader migration should not force recompute yet. |

### Compatibility Aliases

Represent these in `angle_channel_index.compatibility_alias_of` when they are
duplicate values:

| Alias family | Examples | Alias target |
| --- | --- | --- |
| Legacy major names | `left_deg`, `right_deg`, `left_signed_deg`, `right_signed_deg` | Major-axis signed/unsigned channels. |
| Legacy minor names | `left_minor_signed_deg`, `right_minor_signed_deg`, `vergence_minor_signed_deg` | Gaze signed/separation channels. |
| Legacy vergence/version names | `vergence_deg`, `vergence_signed_deg`, `version_deg` | Current major/gaze aggregate fields as documented. |

The first writer implementation may materialize aliases as physical channels
to keep migration low-risk. A later version can remove duplicate alias channels
only after resolver-backed consumers are established.

### QA and Support Channels

Keep QA dense and explicit:

- `valid_left`, `valid_right`, `valid_frame`
- `reason_codes`
- `left_major_axis_marginal`, `right_major_axis_marginal`,
  `major_axis_marginal`

Support arrays remain semantically grouped because they are not just angle
variants:

- ROI/frame mapping: `frame_indices`, `time_seconds`, `frame_time_seconds`
- Ellipse geometry: `ellipse_major`, `ellipse_minor`, `ellipse_ratio`
- Body-frame support: `body_frame/origin_xy`, `body_frame/forward_axis_xy`,
  `body_frame/left_axis_xy`, `body_frame/heading_deg`, `body_frame/valid`,
  `body_frame/failure_reason_bytes`

## Reader Migration Status

Implemented in this slice:

- `eye_angle_io` keeps hierarchical-v1 support.
- `eye_angle_io` can read future compact-dense-v2 runs with:
  `roi_angles`, `frame_angles`, `roi_vectors`, `frame_vectors`, `roi_qa`,
  `frame_qa`, and channel-index groups.
- `load_eye_gaze_frame_series(...)` reports compact backing channels in
  `source_eye_angle_arrays`.
- In-memory tests prove compact channels roundtrip to existing logical names.

Not implemented yet:

- `eye_angle_analysis.py` writer support for compact-dense-v2.
- Real canary compact eye-angle run.
- Crimson compact eye-angle reader validation.

## Writer Migration Plan

1. Keep hierarchical writer as the default until consumers are verified.
2. Add `--layout compact_dense_v2` to `eye_angle_analysis.py`.
3. Write compact arrays by stacking the existing arrays in schema order.
4. Populate `angle_channel_index`, `vector_channel_index`, and
   `qa_channel_index` from the existing output schema metadata.
5. Validate resolver parity against a hierarchical canary: same logical field
   names, shapes, values, and strict JSON attrs.
6. Only then decide whether compact-dense-v2 should become the default.

## Risks

- Channel-index mistakes can silently reinterpret correct numeric values as the
  wrong angle. Tests must verify names, formulas, alias targets, and source
  paths.
- Chunking must support common reads: one channel across all frames and many
  channels for one frame window.
- External consumers that hard-code hierarchical paths need a resolver or
  explicit compatibility period.
