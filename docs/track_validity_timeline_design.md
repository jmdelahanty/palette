# Track Validity Timeline Design

Last reviewed: 2026-04-26

Purpose: define how Palette should unite detection, crop, keypoint, tracking,
and movement validity into one track-level timeline that can be consumed by
swim-bout metrics, plotting, review tools, and future exports.

## Summary

Palette already stores useful validity and coverage metadata, but it is split
across the stages that produce it:

- detection/refined detection knows frame and row coverage
- crops know detection lineage and source type
- keypoints know model/refinement success and heading usability
- tracking knows whether a row is assigned to a real track
- track kinematics knows frame order, positions, speed, and path distance

The missing layer is a unified track-centric validity timeline.

That layer should be written by `track_kinematics`, because that is where row
lineage becomes a biological/temporal track and where movement transitions are
computed. Upstream stages should continue to own their stage-specific validity
signals. `track_kinematics` should project those signals into a per-track
contract that downstream consumers can use without re-resolving all upstream
lineage.

## Implementation Status

Implemented:

- `track_kinematics` now persists the transition validity already used by
  `compute_track_speed(...)`.
- Per-track groups include `delta_frames`, `delta_seconds`,
  `transition_valid`, and `transition_reason_code`.
- Track groups attach `transition_validity_schema_id` and
  `transition_reason_codes` attrs.
- The legacy standalone `analysis/speed_runs` writer also receives the same
  transition arrays for consistency with the shared speed helper.
- `track_kinematics` now projects the upstream validity it already has into
  `sample_observed`, `sample_valid`, `source_observed`, `keypoint_usable`,
  `position_finite`, `heading_usable`, and `sample_reason_code`.
- Offline keypoint usability is resolved from `heading_usable`,
  `refined_success`, `detection_success`, or `source_success`, in that order,
  rather than silently assuming all rows are usable when `detection_success` is
  absent.
- `detect_bouts_multi_level` consumes track-level transition/sample validity
  when available and writes explicit bout metrics: `observed_duration_s`,
  `path_length_mm`, `path_length_px`, `net_displacement_mm`,
  `net_displacement_px`, `mean_speed_mm_s`,
  `peak_detection_signal_mm_s`, `peak_physical_speed_mm_s`,
  `n_invalid_transitions`, `valid_transition_fraction`, and `gap_censored`.
- New swim-bout outputs no longer write the ambiguous first-class `distance`
  field; path length is grounded in track-kinematics frame path-distance arrays.
- Interactive plot specs expose validity source paths when those arrays are
  present, and the marimo track-kinematics explorer can overlay invalid sample
  or transition intervals on the speed plot.

Not implemented yet:

- projection of richer refined-detection source-kind labels into
  `source_kind_code`
- projection of richer keypoint/refinement reason labels into
  `keypoint_reason_code`
- static plot overlays for invalid or uncertain intervals
- dense viewer-cache timelines

## Design Philosophy

The core rule is:

```text
Stage-local validity remains local.
Track-level validity is the consumer-facing timeline.
```

This avoids two bad outcomes:

- duplicating every upstream validity field into every downstream analysis run
- forcing every viewer, bout detector, and exporter to know how to join detect,
  crop, keypoint, and tracking internals

The track-level timeline should be enough to answer:

- is this track row observed, interpolated, manually refined, or missing?
- is the keypoint or heading usable at this row?
- is the movement transition from the previous row valid?
- if not valid, why not?
- how much of a bout interval is observed versus gap-censored?
- where should a viewer draw invalid or uncertain timeline bands?

## Current Upstream Signals

### Detection And Refined Detection

Current useful fields include:

- `frame_indices`
- `frame_counts`
- `refined_row_ids`
- `source_detect_row_index`
- `source_kind_codes`
- `manual_edit_flags`
- `reason` / `reason_bytes`

These fields define row coverage and whether a refined row was copied, manually
created, interpolated, or otherwise altered.

### Crop

Current useful fields include:

- `frame_indices`
- `frame_counts`
- `detection_source`
- `source_detect_row_index`
- `source_refined_row_ids`

These fields preserve the row lineage between crops and the detection rowset
they came from.

### Keypoints And Refined Keypoints

Current useful fields include:

- `source_success`
- `refined_success`
- `detection_success` in raw keypoint runs
- `heading_finite`
- `heading_usable`
- `geometry_valid`
- `confidence_valid`
- `usable_keypoints`
- `n_rois`
- `frame_counts`
- `detection_source`
- `reason` / `reason_bytes`

These fields are the best current source for whether a row has usable pose and
heading data.

### Tracking

Current useful fields include:

- `track_ids`
- `arena_ids`
- `source_row_indices`
- `track_ids_present`
- `track_arena_ids`

Tracking defines which rows belong to real tracks. Rows with `track_id == -1`
are valid upstream rows but are not valid track samples unless a consumer
explicitly opts into unassigned data.

### Track Kinematics

Current useful fields include:

- `frame_indices`
- `time_seconds`
- `detection_indices`
- `positions_px`
- `positions_mm`
- `heading_degrees`
- `heading_radians`
- `keypoint_success`
- `detection_source`
- `speed_*`
- `frame_path_distance_*`
- `cumulative_path_distance_*`

`track_kinematics` already carries some upstream validity into the track group,
but it does not yet persist explicit movement-transition validity.

`compute_track_speed(...)` currently computes transition validity internally
from rules equivalent to:

- elapsed time must be positive
- displacement must be finite
- frames must be consecutive
- displacement must be below the configured sanity threshold

Invalid transitions contribute zero path distance, which avoids inventing
movement across gaps. The missing piece is that the validity mask and reason are
not written for downstream consumers.

## Canary Observation

On the current 2026-01-28 arena 2 canary:

- refined detection has one row per frame
- crop has one row per frame
- refined keypoints have one row per frame
- tracking assigns every row to track `0`
- `detection_source == 0` for every track row
- `heading_usable == True` for every refined keypoint row
- track frames are dense from frame `0` through `19234`

So for this canary, zero values in `frame_path_distance_filtered_mm` mostly mean
hysteresis-filtered no-motion, not invalid detections or missing rows.

This distinction is exactly why the track-level validity timeline matters:

```text
zero movement != missing movement
```

## Proposed Track-Level Arrays

Target location:

```text
analysis/track_kinematics_runs/<mode>/<run>/tracks/<track_id>/
```

Required per-row arrays:

| Array | Shape | DType | Meaning |
| --- | --- | --- | --- |
| `sample_observed` | `(n_rows,)` | `bool` | This row belongs to a real track sample. |
| `sample_valid` | `(n_rows,)` | `bool` | This row is usable for generic track-level analysis. |
| `keypoint_usable` | `(n_rows,)` | `bool` | Pose/heading source is usable at this row. |
| `source_observed` | `(n_rows,)` | `bool` | Detection/crop source is observed rather than interpolated or synthetic. |
| `delta_frames` | `(n_rows,)` | `int32` | Frame delta from the previous row. First row uses `0`. |
| `delta_seconds` | `(n_rows,)` | `float32` | Time delta from the previous row. First row uses `0`. |
| `transition_valid` | `(n_rows,)` | `bool` | Movement transition from previous row to this row is valid. First row is false. |
| `transition_reason_code` | `(n_rows,)` | `int16` | Reason transition is valid or invalid. |

Recommended per-row arrays:

| Array | Shape | DType | Meaning |
| --- | --- | --- | --- |
| `sample_reason_code` | `(n_rows,)` | `int16` | Reason the sample is valid or invalid. |
| `source_kind_code` | `(n_rows,)` | `int16` | Projected source kind from refined detect/crop lineage when available. |
| `keypoint_reason_code` | `(n_rows,)` | `int16` | Projected keypoint/refinement reason when available. |
| `position_finite` | `(n_rows,)` | `bool` | Position values are finite. |
| `heading_usable` | `(n_rows,)` | `bool` | Heading is finite and accepted by keypoint/refinement rules. |

These arrays should be sparse-track arrays aligned to the existing per-track
`frame_indices`, not dense arrays over every recording frame.

## Reason Code Vocabulary

Use integer codes for compact storage and attach a JSON mapping in attrs:

```text
attrs["sample_reason_codes"]
attrs["transition_reason_codes"]
```

Suggested sample reason codes:

| Code | Name | Meaning |
| --- | --- | --- |
| `0` | `ok` | Sample is usable. |
| `1` | `unassigned` | Source row had no real `track_id`. |
| `2` | `source_interpolated` | Source row is interpolated or synthetic. |
| `3` | `source_missing` | Expected source row is missing. |
| `4` | `keypoint_failed` | Keypoint/refined keypoint failed. |
| `5` | `heading_unusable` | Position may exist, but heading is not usable. |
| `6` | `position_nan` | Position is not finite. |
| `7` | `manual_reject` | Manual or review state rejects the row. |

Suggested transition reason codes:

| Code | Name | Meaning |
| --- | --- | --- |
| `0` | `ok` | Transition is usable. |
| `1` | `first_sample` | No previous row exists. |
| `2` | `frame_gap` | Previous and current rows are not consecutive frames. |
| `3` | `nonpositive_dt` | Elapsed time is zero or negative. |
| `4` | `position_nan` | One or both endpoint positions are not finite. |
| `5` | `teleport` | Displacement exceeds the sanity threshold. |
| `6` | `source_not_observed` | One or both endpoint sources are interpolated/synthetic. |
| `7` | `keypoint_unusable` | One or both endpoint keypoint states are unusable. |

The first implementation does not need every code if the upstream information
is not available yet. It should still write the mapping for the codes it does
emit.

## Bout Metric Implications

Swim-bout metrics should use the track-level transition validity rather than
inferring validity from zeros or NaNs.

For each bout interval, persist:

| Field | Meaning |
| --- | --- |
| `elapsed_duration_s` | Wall-clock duration from bout start to end. |
| `observed_duration_s` | Sum of valid transition durations inside the bout. |
| `path_length_mm` | Sum of valid `frame_path_distance_*_mm` inside the bout. |
| `path_length_px` | Sum of valid `frame_path_distance_*_px` inside the bout. |
| `net_displacement_mm` | Euclidean displacement between valid endpoint positions. |
| `net_displacement_px` | Pixel-space endpoint displacement. |
| `mean_speed_mm_s` | `path_length_mm / observed_duration_s` when observed duration is positive. |
| `peak_detection_signal_mm_s` | Maximum detector signal inside the bout. For transformed levels, this is the transformed response value. |
| `peak_physical_speed_mm_s` | Maximum declared physical speed source inside the same bout boundaries. |
| `n_invalid_transitions` | Count of invalid transitions inside the bout. |
| `valid_transition_fraction` | Valid transition count divided by possible transition count. |
| `gap_censored` | True when the interval contains invalid or missing transitions. |

Do not treat invalid transitions as ordinary zero-speed observations for bout
statistics. Zero can mean true no-motion; invalid means unknown movement.

It is still defensible for `track_kinematics` to write zero
`frame_path_distance_*` on invalid transitions so cumulative path distance does
not invent motion across gaps. The important rule is that downstream consumers
must also see the validity mask that explains those zeros.

## Plotting And Marimo Implications

The marimo track-kinematics explorer overlays invalid timeline regions on speed
plots when track validity arrays are present.

Display policy:

- swim-bout intervals remain translucent warm bands
- invalid or missing movement intervals use a separate red band
- interpolated/synthetic source intervals use a different hatch/color if the
  renderer supports it
- keypoint/heading failures can be separated from detection gaps by reason label

The viewer builds overlay bands from sparse track arrays at load time. A dense
persisted timeline is not required initially.

If performance becomes an issue, add a cache group under the visualization
artifact rather than treating dense overlay arrays as canonical science data.

## Dense Timeline Option

Sparse per-track arrays should be the canonical storage in the first
implementation.

A later dense convenience group may be useful:

```text
analysis/track_kinematics_runs/<mode>/<run>/tracks/<track_id>/timeline/
```

Possible arrays:

- `frame_valid`
- `frame_observed`
- `frame_keypoint_usable`
- `transition_valid`
- `reason_code`

This dense group should be considered a cache or viewer acceleration layer. The
canonical contract remains the sparse per-track row arrays plus `frame_indices`.

## Implementation Plan

### Phase 1: Persist Existing Transition Semantics

Add outputs from the current `compute_track_speed(...)` logic:

- `delta_frames`
- `delta_seconds`
- `transition_valid`
- `transition_reason_code`
- reason-code attrs

This phase should not change numeric speed or path-distance values. It only
materializes the validity already implied by the current computation.

### Phase 2: Project Upstream Sample Validity

Carry upstream row validity into track groups:

- `source_observed`
- `sample_observed`
- `sample_valid`
- `keypoint_usable`
- `sample_reason_code`
- optional `source_kind_code`
- optional `keypoint_reason_code`

This phase may need small helpers to normalize detection/crop/keypoint validity
without making every consumer know every upstream schema.

### Phase 3: Update Bout Metrics

Update `detect_bouts_multi_level` so bout tables contain:

- explicit path length
- explicit net displacement
- observed versus elapsed duration
- invalid transition counts
- valid transition fraction
- gap-censored flag

Remove ambiguous first-class `distance` fields from new outputs. If migration
requires compatibility, use an explicit legacy name such as
`path_length_from_mean_speed_mm`.

### Phase 4: Update Plot Overlays

Update `apps/marimo/track_kinematics_explorer.py` and static plot artifacts to
draw invalid or uncertain timeline bands from `sample_valid`,
`transition_valid`, and reason codes.

Plot specs should reference the canonical track arrays rather than embedding
full overlay data.

Status: marimo overlays are implemented; static PNG overlays are still open.

### Phase 5: Export Contract

When Palette adds clean exportable analysis datasets, export the track-level
validity timeline alongside motion and bout metrics. Exporters should not
reconstruct validity by joining raw detect/crop/keypoint groups unless the
track-level arrays are missing and the export explicitly requests a repair mode.

## Open Questions

- Should interpolated refined detections be valid for some analyses but invalid
  for movement distance?
- Should keypoint failure invalidate only heading metrics, or also position and
  speed metrics when the position source comes from keypoints?
- Should `source_observed` distinguish manual observed rows from model observed
  rows, or is that only provenance?
- Should invalid transition reason priority be fixed globally, for example
  `frame_gap` before `teleport`, or should multiple bit flags be stored?

The first implementation can use one reason code per sample/transition. If
multiple simultaneous reasons become common, add bitmask fields later.

## Related Docs

- [`coverage_unification_todo.md`](archive/coverage_unification_todo.md)
- [`single_subject_per_arena_tracking_contract.md`](./single_subject_per_arena_tracking_contract.md)
- [`kinematics_zarr_access_guide.md`](./kinematics_zarr_access_guide.md)
- [`plot_visualization_artifact_contract.md`](./plot_visualization_artifact_contract.md)
- [`tracking_unassigned_row_policy.md`](./tracking_unassigned_row_policy.md)
