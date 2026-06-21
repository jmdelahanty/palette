# Detection Analysis Run Surfaces
<!-- contract-meta
status: draft
last_updated: 2026-06-21
-->

Purpose: document how detection-related per-recording analysis products should
fit into Palette's existing `analysis/<analysis_type>_runs/<run>` model, and
name the detection-specific run families without creating duplicate surfaces.

## Current Per-Recording Model

Each recording's analysis zarr is the canonical archive for that recording.
Root-level stage families hold raw or refined authorities; `analysis/` holds
deterministic derived products that can be regenerated from those authorities.

Use root-level stage families for detection authorities:

- `detect_runs/<run>`: raw/model detection output.
- `refined_detect_runs/<run>`: curated/refined detection authority.
- `refined_detect_runs/<run>/instances`: current instance-table surface for
  downstream consumers.

Use `analysis/<analysis_type>_runs/<run>` for derived analysis products:

```text
<recording>_analysis.zarr/
  analysis/
    <analysis_type>_runs/
      zarr.json                       # parent attrs, normally latest=<run>
      <run>/
        zarr.json                     # schema/method/source/provenance attrs
        row_index/                    # optional source-row/frame mapping
        <semantic output groups>/     # analyzer-specific arrays/tables
        visualizations/               # optional PNG/spec artifacts
```

The inspected example
`2026-01-28T21-47-47Z_arena_1_DefaultScreen_analysis.zarr` follows this model
with:

- `analysis/eye_angle_runs/<run>`
- `analysis/subject_shape_runs/<run>`
- `analysis/swim_bout_runs/<run>`
- `analysis/bout_kinematics_runs/<run>`
- `analysis/stimulus_response_runs/<run>`
- `analysis/tail_posture_view_runs/<run>`
- `analysis/bout_classification_runs/<run>`
- `analysis/keypoint_profile_runs/<run>`
- `analysis/track_kinematics_runs/offline/<run>`

Those run families use parent `latest` attrs, run-level schema/method/source
attrs, and semantic child groups such as `row_index`, `tables`, `signals`,
`frames`, `tracks`, and `visualizations`.

## Detection-Specific Names

Do not introduce a generic `detection_summary_runs` family. Palette already has
a more precise implemented surface:

### `analysis/detection_profile_runs/<run>`

Use this for scalar/profile summaries of a detection source. This is the
implemented profile surface and is projected into the registry through
`detection_data_profile`.

Good contents:

- coverage/distribution summaries
- source detection path and content hash
- training-selection/profile metrics
- profile JSON suitable for registry projection

Avoid putting epoch-specific heatmaps or realtime-vs-offline comparison arrays
here. Those are different analyses with different source dependencies and row
axes.

### `analysis/stimulus_epoch_runs/<run>` (shared event-window source)

Use this shared stimulus analysis family to define reusable event-aligned
windows such as GoodCopBadCop `pre_event`, `training_event`, and `post_event`.
Detection, keypoint, mask, tracking, bout, and response analyses should consume
these windows instead of redefining them locally.

See [`stimulus_epoch_run_contract.md`](stimulus_epoch_run_contract.md).

### `analysis/detection_occupancy_runs/<run>` (implemented family)

Use this for detection-derived spatial occupancy and epoch-window summaries,
including dense heatmaps, coarse zone summaries, and the GoodCopBadCop
advisor-style plots.

This name is better than `detection_summary_runs` because it states the
measured quantity: where refined detections occurred over time or within
stimulus epochs.

Expected source refs:

- `source_detection_path`: usually `refined_detect_runs/<run>/instances`
- `source_stimulus_epoch_run`: required when windows are event-aligned
- `source_stimulus_epoch_path`: usually `analysis/stimulus_epoch_runs/<run>`
- optional `source_dish_mask` / calibration refs when masking or coordinate
  transforms are applied

The concrete writer is `fisheye.analysis.detection_occupancy_runs`. The
GoodCopBadCop registry wrapper is
`fisheye.utils.run_goodcopbadcop_detection_analysis`; it selects recordings by
registry detect coverage, writes `analysis/stimulus_epoch_runs/<run>`, then
writes `analysis/detection_occupancy_runs/<run>`.

Recommended child groups:

```text
analysis/detection_occupancy_runs/<run>/
  windows/
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
    counts                      # e.g. window x y_bin x x_bin
    normalized                  # optional display-normalized heatmaps
    x_edges
    y_edges
  spatial_occupancy/
    <zone_set_id>/
      zone_spec/                # resolved zone geometry and labels
      summary/                  # window x zone frame/time/fraction arrays
  visualizations/
    detection_occupancy_overview_png
```

Recommended `row_axis`: `stimulus_epoch_windows` for window summaries, or
`frames` if a writer stores frame-level occupancy inputs.

The `windows/` group may copy the resolved labels and frame bounds used for the
measurement, but the authoritative event-window definition should remain in the
referenced `analysis/stimulus_epoch_runs/<run>`.

Coarse spatial occupancy maps, such as quadrant summaries, should live under
`spatial_occupancy/<zone_set_id>` inside this same run. Zone sets may come from
predefined specs, such as image quadrants, or from future experimental metadata.
They are analysis arrays, not visualization runs. See
[`spatial_occupancy_zone_summary_design.md`](spatial_occupancy_zone_summary_design.md).

### `analysis/detection_comparison_runs/<run>` (current compatibility family)

This is the current implemented path written by
`fisheye.diagnostics.compare_realtime_offline_detections`. Its schema is more
specific than the parent name:

```text
schema_id = "palette.detection_realtime_offline_comparison.v1"
```

Use it for comparing acquisition-time/realtime boxes against offline/refined
detections and for evaluating whether acquisition crop videos are sufficient for
downstream crop-based models.

Current outputs include:

- offline and realtime presence masks
- offline and realtime bbox/center arrays
- centroid deltas and bbox IoU
- epoch labels
- crop sufficiency arrays when crop metadata is available
- a run-local PNG summary in `visualizations/`

The current parent name is broad. If we decide to migrate the path before
additional consumers depend on it, the better long-term name is:

```text
analysis/realtime_offline_detection_comparison_runs/<run>
```

That name is long, but it accurately distinguishes this analysis from generic
model-vs-model or raw-vs-refined comparisons. A shorter acceptable alternative
for acquisition-specific recordings is:

```text
analysis/acquisition_detection_comparison_runs/<run>
```

Do not silently split these names. If the path changes, provide a compatibility
reader for `analysis/detection_comparison_runs` and make the new writer emit
only the new canonical path.

## Recommended Naming Rules

- Use `detection_*` for analysis run families, matching existing
  `detection_profile_runs` and `detection_data_profile`.
- Use `detect_*` for root-level stage names and CLI internals that already use
  the pipeline stage vocabulary (`detect`, `detect_quality`, `refined_detect`).
- Prefer names that state the scientific/operational question:
  `detection_occupancy_runs` is clearer than `detection_summary_runs`.
- Avoid catch-all families such as `detection_analysis_runs`; they become
  monolithic and make source lineage harder to reason about.
- Keep separate analyses in separate run families. A later keypoint, mask, or
  track-derived result should reference detection runs through `source_refs`
  rather than being appended into a detection run.

## Incremental Analysis Strategy

For GoodCopBadCop, the staged per-recording approach should be:

1. Keep raw/offline detections as root-level `detect_runs` and
   `refined_detect_runs`.
2. Use or refresh `analysis/detection_profile_runs` for scalar/profile metrics.
3. Add `analysis/stimulus_epoch_runs` for reusable pre/training/post event
   windows.
4. Add `analysis/detection_occupancy_runs` for per-recording event-aligned
   heatmaps, spatial zone summaries, and coverage summaries that reference the
   epoch run.
5. Use `analysis/detection_comparison_runs` for realtime-vs-offline/crop
   sufficiency diagnostics until a deliberate path migration is made.
6. Use `analysis/chaser_distance_runs` for stimulus-specific offline
   fish-to-chaser distance measurements that consume refined detections,
   stimulus state, and shared stimulus epochs. See
   [`chaser_distance_run_contract.md`](chaser_distance_run_contract.md).
7. Add later keypoint, segmentation, tracking, and stimulus-response products as
   their own `analysis/*_runs` families with explicit `source_refs`.

This keeps each recording self-contained while still allowing future
cross-recording Parquet exports to select exact source run IDs from each zarr.

For the current GoodCopBadCop workflow and review commands, see
[`goodcopbadcop_analysis_surfaces.md`](goodcopbadcop_analysis_surfaces.md).
