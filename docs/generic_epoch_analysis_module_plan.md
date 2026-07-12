# Generic Epoch Analysis Module Plan
<!-- contract-meta
status: draft
last_updated: 2026-07-01
-->

Purpose: plan a reusable epoch/segment analysis layer that can support
GoodCopBadCop and future protocols without forcing protocol-specific biology
into shared helper code.

## Motivation

GoodCopBadCop now has several event-aligned analysis surfaces:

- `analysis/stimulus_epoch_runs/<run>` defines `pre_event`,
  `training_event`, and `post_event` windows.
- `analysis/detection_occupancy_runs/<run>` measures detection occupancy in
  those windows.
- `analysis/chaser_distance_runs/<run>` measures fish-to-chaser distances and
  carries protocol-specific components such as CRA metrics, near-field metrics,
  egocentric bearing, escape/freeze canaries, and epoch behavior summaries.

The current design already has the right storage boundary: stimulus windows
are derived once and downstream analyses consume them. The implementation is
still more duplicated than it needs to be. Several modules define their own
window dataclasses, window readers, frame-assignment helpers, and per-window
summary loops.

The immediate issue is the per-recording swim-bout distribution display:
GoodCopBadCop persists raw `per_epoch_bouts` rows, then the marimo viewer
computes histograms with viewer-side binning. The durable fix is an
analysis-owned `per_epoch_bout_histograms` table. That table should be built
with generic epoch histogram helpers, not a one-off GoodCopBadCop-only
histogram loop.

## Design Boundary

Shared epoch code should own:

- reading and validating segment/window definitions,
- assigning framewise samples, point events, and interval events to segments,
- computing standard per-segment summaries,
- computing persisted histogram tables with explicit bin contracts,
- storing provenance about segment sources, assignment policies, and bin specs.

Protocol modules should own:

- which event policy defines the segment set,
- which source tables and signals are biologically meaningful,
- protocol labels such as aggressive/inert or chaser identity,
- protocol-specific metrics and statistical interpretation,
- visual layout and captions.

This keeps the shared layer mechanical and auditable. It should not know what
"aggressive", "inert", "escape", "freeze", or "conditioned response" means.

## Current Surfaces To Reuse

The existing authority for reusable windows is documented in
[`stimulus_epoch_run_contract.md`](stimulus_epoch_run_contract.md):

```text
analysis/stimulus_epoch_runs/<run>/windows/
  window_id
  label_bytes
  start_frame
  end_frame
  start_time_s
  end_time_s
  duration_s
  source_start_event_name_bytes
  source_end_event_name_bytes
  source_start_event_frame
  source_end_event_frame
  source_policy_bytes
```

The current writer lives in `fisheye.analysis.stimulus_epoch_runs`. It is
currently GoodCopBadCop-specific at the policy layer, but the written window
schema is already generic enough for other protocols.

The following modules currently duplicate parts of this logic and are good
initial migration targets:

- `fisheye.analysis.detection_occupancy_runs`
- `fisheye.analysis.chaser_distance_runs`
- `fisheye.analysis.goodcopbadcop_epoch_behavior_summary`
- `fisheye.analysis.cra_primary_endpoint`
- `fisheye.analysis.chaser_egocentric_bearing`
- `fisheye.utils.export_cross_recording_analytics`

## Proposed Shared Module

Add a module such as:

```text
src/fisheye/analysis/epoch_segments.py
```

Initial public objects:

```python
@dataclass(frozen=True)
class EpochSegment:
    segment_id: int
    label: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    source_start_event_name: str | None = None
    source_end_event_name: str | None = None
    source_policy: str | None = None

@dataclass(frozen=True)
class AssignmentPolicy:
    kind: Literal["frame", "point_event", "interval"]
    frame_field: str | None = None
    start_frame_field: str | None = None
    end_frame_field: str | None = None
    interval_rule: Literal["start", "end", "midpoint", "peak", "overlap"] | None = None
```

Initial helper families:

- `resolve_stimulus_epoch_run(root, run_name=None)`
- `read_epoch_segments(epoch_group)`
- `write_epoch_segments(group, segments, attrs=...)`
- `assign_frames_to_segments(total_frames, segments)`
- `assign_point_events_to_segments(frames, segments)`
- `assign_intervals_to_segments(start_frames, end_frames, segments, policy)`
- `finite_summary(values, percentiles=(5, 50, 95))`
- `rate_per_minute(count, duration_s)`
- `histogram_table(values_by_segment, metric_spec, segments)`

The helper should use neutral naming internally: `segment` or `epoch_segment`.
Existing zarr tables may continue to use `window_id` and `window_label` for
schema compatibility.

## Histogram Contract

Persisted histograms should be tables, not viewer-side `nbins` choices.

Recommended table columns:

```text
metric_name
units
window_id
window_index
window_label
bin_index
bin_left
bin_right
bin_center
bin_width
hist_count
hist_fraction
hist_density
source_sample_count
finite_sample_count
bin_policy
```

Recommended metric spec:

```python
@dataclass(frozen=True)
class HistogramMetricSpec:
    metric_name: str
    units: str
    bin_policy: str
    bin_width: float | None = None
    bin_edges: tuple[float, ...] | None = None
    range_min: float | None = None
    range_max: float | None = None
    include_overflow_bins: bool = False
```

Rules:

- bins must be shared across all segments being compared in one component,
- bin edges must be persisted or reconstructible from persisted table rows,
- plots should render from `hist_count`, `hist_fraction`, or `hist_density`,
  not recompute bins from raw rows,
- custom viewer re-binning is allowed only as an explicit exploratory mode.

## GoodCopBadCop First Consumer

The first implementation target should be
`goodcopbadcop_epoch_behavior_summary`.

Current component:

```text
analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>/
  per_epoch_fish
  per_epoch_chaser
  per_epoch_bouts
  center_distance_histogram
```

Add:

```text
  per_epoch_bout_histograms
```

Initial metrics:

- `bout_duration_s`
- `bout_path_length_mm`
- `bout_net_heading_change_deg`
- `abs_bout_net_heading_change_deg`
- `bout_heading_path_deg`

Suggested first-pass bins:

| Metric | Units | Policy |
| --- | --- | --- |
| `bout_duration_s` | s | fixed width, component config |
| `bout_path_length_mm` | mm | fixed width, component config |
| `bout_net_heading_change_deg` | deg | fixed edges from `-180` to `180` |
| `abs_bout_net_heading_change_deg` | deg | fixed edges from `0` to `180` |
| `bout_heading_path_deg` | deg | fixed width, component config |

The source `per_epoch_bouts` table should remain. It is the auditable raw layer
for per-bout values. The histogram table is the persisted visualization and
distribution layer.

## Future Consumers

Once the helpers exist, they should be reused by:

- per-epoch inter-bout interval distribution tables,
- per-epoch speed distribution tables,
- protocol trial windows such as chase bouts or OMR steps,
- cross-recording export of binned distributions,
- group viewers that need guaranteed comparable bin contracts.

Trial-like segments should use the same core representation. A chase trial is
just a segment set with different source policy and row axis. The biological
classifier remains protocol-specific.

## Non-Goals

- Do not replace `analysis/stimulus_epoch_runs`.
- Do not move GoodCopBadCop CRA, near-field, or escape/freeze semantics into a
  generic module.
- Do not make viewer-side custom windows impossible.
- Do not rewrite every existing epoch consumer in one pass.
- Do not change old persisted component schemas without versioning or
  compatibility handling.

## Implementation Checklist

1. Add `fisheye.analysis.epoch_segments` with `EpochSegment`, window reader,
   frame/point/interval assignment helpers, finite summaries, rate helpers, and
   histogram table builder.
2. Add in-memory unit tests for the helper module:
   - window read validation,
   - inclusive frame assignment,
   - point-event assignment,
   - interval assignment policies,
   - fixed-edge histogram output,
   - empty/NaN sample handling.
3. Refactor one low-risk reader to use `read_epoch_segments` while preserving
   output schema. Good candidates are `chaser_distance_runs` or
   `goodcopbadcop_epoch_behavior_summary`.
4. Add `per_epoch_bout_histograms` to
   `goodcopbadcop_epoch_behavior_summary` using the shared histogram builder.
5. Update the marimo GoodCopBadCop component to prefer
   `per_epoch_bout_histograms` and only fall back to raw-row viewer histograms
   for older recordings.
6. Update cross-recording export to include the persisted bout histogram table,
   with source component/run provenance.
7. Backfill the GoodCopBadCop `/groups` recordings after the schema is stable.
8. Audit remaining viewer-side histogram paths and classify each as either:
   persisted-analysis view, exploratory custom view, or deprecated legacy view.

## Acceptance Criteria

- A generic helper can read `analysis/stimulus_epoch_runs/<run>/windows`
  without importing a GoodCopBadCop module.
- GoodCopBadCop bout distribution plots no longer depend on Plotly automatic
  binning for current components.
- Persisted bout histogram rows use identical bins for all epoch labels within
  a component.
- The raw `per_epoch_bouts` table remains available for audit and future
  derived metrics.
- Exports and viewers can tell whether a distribution came from persisted
  analysis bins or from exploratory viewer-side re-binning.
