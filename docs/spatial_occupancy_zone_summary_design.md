# Spatial Occupancy Zone Summary Design
<!-- contract-meta
status: draft
last_updated: 2026-06-21
-->

Purpose: define how Palette should store coarse spatial occupancy summaries
derived from detections, with quadrants as the first predefined zone set and a
path toward richer spatial occupancy maps from experimental metadata.

## Decision

Spatial occupancy zone summaries should be a module inside the existing
`analysis/detection_occupancy_runs/<run>` surface, not a new visualization run
family and not a separate run family for each zone layout.

Detection occupancy already owns the required source dependencies:

- the refined detection source used to locate the fish,
- the stimulus epoch run used to define time windows,
- the coordinate frame used for image-space occupancy, and
- coverage information needed to interpret missing detections.

The first concrete zone set is image quadrants. Future zone sets may come from
experimental metadata, protocol metadata, registered polygons, or predefined
specs. They should all resolve to the same run-local zone-spec shape before
summary arrays are computed.

## Boundary

`analysis/stimulus_epoch_runs/<run>` remains the authority for event windows.
It defines windows such as `pre_event`, `training_event`, and `post_event`.

`analysis/detection_occupancy_runs/<run>` owns detection-derived spatial
summaries within those windows:

- dense heatmaps under `heatmaps/`,
- coarse zone summaries under `spatial_occupancy/`, and
- visualization specs or PNG review artifacts under `visualizations/`.

Visualization components can point to the stored zone summaries, but the
summaries themselves are analysis arrays. They are not visualization artifacts.

If future experimental metadata becomes the canonical authority for zone
geometry, the detection occupancy run should store the exact metadata source
reference plus a resolved copy of the zone geometry used for the computation.
Consumers should not have to re-run metadata resolution to understand an
already-written analysis run.

## Recommended Storage

Recommended location:

```text
analysis/detection_occupancy_runs/<run>/
  windows/
    window_id
    label_bytes
    start_frame
    end_frame
    duration_s
    source_stimulus_epoch_window_id
  coverage/
    detection_count
    covered_frame_count
    coverage_pct
  heatmaps/
    counts
    normalized
    x_edges
    y_edges
  spatial_occupancy/
    <zone_set_id>/
      zarr.json
      zone_spec/
        zone_id
        label_bytes
        display_order
        geometry_type
        bounds_xyxy
        polygon_offsets          # optional for polygon zone sets
        polygon_xy               # optional for polygon zone sets
        mask_ref_path_bytes      # optional for mask-backed zone sets
      summary/
        frame_count
        time_s
        fraction_of_epoch
        fraction_of_detected
        detected_frame_count
        missing_frame_count
        total_span_frames
        coverage_pct
```

The `spatial_occupancy/<zone_set_id>` group should carry attrs such as:

```text
schema_id = "palette.spatial_occupancy_zones.v1"
zone_set_id = <zone_set_id>
zone_set_source = "predefined_spec:quadrants.v1" | "experimental_metadata:<id>" | ...
zone_set_source_ref = <metadata path, protocol field, or spec identifier>
coordinate_frame = "source_image_px" | "arena_relative_canvas_px" | ...
coordinate_origin = "top_left"
x_axis_direction = "right"
y_axis_direction = "down"
source_detection_path = refined_detect_runs/<run>/instances
source_stimulus_epoch_path = analysis/stimulus_epoch_runs/<run>
detection_selection_policy = <policy string>
zone_overlap_policy = <policy string>
time_basis = "frame_count / frame_rate_hz" | <explicit time source>
```

The `summary/*` arrays use window rows from `windows/` and zone columns from
`zone_spec/`. For example:

```text
frame_count[window, zone]
time_s[window, zone]
fraction_of_epoch[window, zone]
fraction_of_detected[window, zone]
```

`fraction_of_epoch` uses the full frame span of the epoch as its denominator.
`fraction_of_detected` uses only frames with a selected valid detection as its
denominator. Both are useful: the first preserves missing-detection penalties,
while the second describes spatial preference among successfully detected
frames.

## Quadrants V1

The predefined quadrant set should use:

```text
zone_set_id = "image_quadrants_v1"
zone_set_source = "predefined_spec:quadrants.v1"
coordinate_frame = "source_image_px"
coordinate_origin = "top_left"
x_axis_direction = "right"
y_axis_direction = "down"
```

Given source image dimensions `width_px` and `height_px`:

```text
mid_x = width_px / 2
mid_y = height_px / 2
```

Quadrant membership is:

| display_order | zone_id | Label | Bounds |
| ---: | --- | --- | --- |
| 0 | `top_left` | Top left | `0 <= x < mid_x`, `0 <= y < mid_y` |
| 1 | `top_right` | Top right | `mid_x <= x < width_px`, `0 <= y < mid_y` |
| 2 | `bottom_left` | Bottom left | `0 <= x < mid_x`, `mid_y <= y < height_px` |
| 3 | `bottom_right` | Bottom right | `mid_x <= x < width_px`, `mid_y <= y < height_px` |

Midline points belong to the right or bottom quadrant. This makes the zones
non-overlapping and deterministic. `bounds_xyxy` should use `[x_min, y_min,
x_max, y_max]` with the max bounds treated as exclusive for membership.

Plotting code may reverse the rendered y-axis to make image-space plots look
like camera images, but the stored coordinate contract remains top-left origin
with positive y downward.

## Counting Policy

The default counting unit is one selected detection per frame.

For current single-fish refined detection workflows, there should usually be at
most one valid detection row per frame. If multiple candidate detections exist,
the writer should use a deterministic selection policy and store that policy in
group attrs. The recommended default is:

1. choose detections with finite centroid coordinates inside the source image,
2. prefer the highest confidence or score when that field exists,
3. break ties by stable source row order, and
4. count no zone for frames without a selected valid detection.

No interpolation should be applied for missing frames in the initial
implementation. Missing detections should affect `coverage_pct`,
`missing_frame_count`, and `fraction_of_epoch`.

`time_s` should be derived from frame counts and the frame-rate source already
used by the detection occupancy run unless an explicit per-frame timestamp
source is available. The selected time basis should be stored in attrs.

## Metadata-Derived Zone Sets

Future experimental metadata may define zones at several levels:

- physical arena or chamber regions,
- camera-view polygons,
- protocol-specific regions such as danger zones or target areas, or
- registered masks in image, arena-canvas, or physical coordinates.

The detection occupancy writer should resolve those definitions into the same
`zone_spec/` arrays before counting. If a zone definition is not already in the
detection coordinate frame, the run should store the transform lineage used to
compare detections and zones. Acceptable strategies are:

- transform detections into the zone coordinate frame, or
- transform zones into the detection coordinate frame.

Either strategy is acceptable if the stored attrs make the coordinate frame,
origin, axis directions, and source transform explicit.

These occupancy zones are not necessarily the same concept as the long-term
Citrus topology `zone_id`, which represents an independently analyzed spatial
unit. A spatial occupancy zone can be a subregion inside one topology zone, such
as an image quadrant. When both concepts are present, use attrs or columns such
as `topology_zone_id` and `occupancy_zone_id` to avoid ambiguity.

## Multiple Zone Sets

A single detection occupancy run may contain multiple zone sets when they share
the same detection source, epoch source, and core counting policy:

```text
spatial_occupancy/
  image_quadrants_v1/
  goodcopbadcop_protocol_regions_v1/
  metadata_zone_set_<id>/
```

Create a new `analysis/detection_occupancy_runs/<run>` only when the scientific
lineage changes, for example:

- different refined detection source,
- different stimulus epoch source,
- different coordinate calibration or transform source,
- different frame inclusion policy, or
- different detection selection policy.

This keeps the archive modular without turning every visualization or zone
layout into a separate top-level run.

## Visualization Use

Marimo components should discover `spatial_occupancy/*` zone sets and let the
user choose among them. A component can render bar charts, tables, or spatial
maps from the stored `zone_spec/` and `summary/` arrays.

Interactive specs may point to these arrays from
`visualizations/<artifact>/spec_json`, but they should not duplicate the
summary values inside the spec. PNG review snapshots may be written when useful,
but they should be treated as renderings of the stored arrays, not as
authorities.
