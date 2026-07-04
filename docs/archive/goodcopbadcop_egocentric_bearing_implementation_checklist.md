<!-- ARCHIVED 2026-07-04: dated point-in-time snapshot / spent work ticket, retained for history only. -->

# GoodCopBadCop Egocentric Chaser Bearing Implementation Checklist

<!-- contract-meta
status: implemented
last_updated: 2026-06-21
-->

Purpose: implement fish-centric chaser bearing analysis for GoodCopBadCop
recordings using track kinematics as the required heading authority.

This analysis asks where each chaser is in the fish's body-centered coordinate
system over time and across stimulus epochs. The primary visualization is a
polar plot where angle is chaser bearing relative to fish heading and radius is
fish-to-chaser distance.

## Decision

Require an offline track-kinematics run before computing egocentric chaser
bearing.

Do not silently fall back to raw or refined keypoint heading arrays for the
GoodCopBadCop egocentric analysis. Raw/refined keypoint heading may remain a
future explicit diagnostic mode, but the default scientific output should use
the same kinematics surface that downstream movement, heading-change, bout, and
PSTH analyses will use.

Rationale:

- `analysis/track_kinematics_runs/offline/<run>` provides a coherent heading
  surface with frame indices, validity, smoothing level, and provenance.
- The analysis is about body heading relative to an object, not merely whether a
  finite heading scalar exists.
- Requiring kinematics keeps this component compatible with later trajectory,
  movement, heading-change, and chaser-event analyses.
- Missing kinematics is a pipeline prerequisite to backfill, not a visualization
  fallback condition.

## Coordinate And Angle Contract

- Fish and chaser positions come from
  `analysis/chaser_distance_runs/<run>/positions/*`.
- Fish heading comes from
  `analysis/track_kinematics_runs/offline/<run>/tracks/id_<track_id>/`.
- Default heading array: `smoothed_heading_degrees`.
- Optional explicit raw heading array: `heading_degrees`.
- Heading convention follows the keypoint/track-kinematics contract:
  `0 deg` points image +x/right, positive rotation is counter-clockwise in math
  coordinates, and image y is converted by negating `dy`.
- Egocentric bearing convention:
  `bearing_deg = wrap(object_bearing_world_deg - fish_heading_deg)`.
- Interpretations:
  - `0 deg`: chaser directly in front of fish.
  - `+90 deg`: chaser on anatomical-left side.
  - `-90 deg`: chaser on anatomical-right side.
  - `+/-180 deg`: chaser behind fish.

## Storage Target

Store this as a modular component under the existing chaser-distance run:

```text
analysis/chaser_distance_runs/<run>/
  egocentric_bearing/
    <component_name>/
      frames/
        camera_frame_id
        stimulus_epoch_window_id
        fish_heading_deg
        fish_heading_valid
      per_chaser/
        chaser_index
        object_vector_arena_xy
        bearing_deg
        alignment_cos
        lateral_sin
        valid
      epoch_summary/
        window_id
        label_bytes
        start_frame
        end_frame
        valid_frame_count
        circular_mean_bearing_deg
        circular_resultant_length
        mean_alignment_cos
        mean_lateral_sin
        fraction_front_45
        fraction_lateral_45
        fraction_behind_45
      distance_bearing_histogram/
        window_id
        chaser_index
        distance_bin_edges_mm
        distance_bin_centers_mm
        bearing_bin_edges_deg
        bearing_bin_centers_deg
        hist_counts
        hist_probability
```

Parent attrs:

```text
analysis/chaser_distance_runs/<run>/egocentric_bearing.attrs["latest"] = <component_name>
analysis/chaser_distance_runs/<run>/egocentric_bearing.attrs["latest_complete"] = <component_name>
```

The component attrs must include:

- `schema_id = "palette.chaser_egocentric_bearing.v1"`
- source chaser-distance run path
- source track-kinematics run path
- source track id
- source heading array
- heading level
- angle convention
- binning parameters
- summary counts
- provenance block

## Implementation Checklist

- [x] Finish the analysis module using the kinematics-required policy.
- [x] Resolve `analysis/chaser_distance_runs/<run>` by name or latest complete.
- [x] Resolve `analysis/track_kinematics_runs/offline/<run>` by name or latest.
- [x] Fail clearly if no offline track-kinematics run exists.
- [x] Fail clearly if requested `track_id` does not exist.
- [x] Load `frame_indices`, `sample_valid`, and the selected heading array from
      the track group.
- [x] Densify track heading onto the chaser-distance camera-frame axis.
- [x] Gate valid frames with finite heading, `sample_valid`, fish position
      validity, chaser position validity, and finite distance.
- [x] Compute object vector in arena coordinates.
- [x] Convert object vector to world bearing using image-y-down to math-y-up
      conversion.
- [x] Compute signed egocentric bearing.
- [x] Compute `alignment_cos = cos(bearing)`.
- [x] Compute `lateral_sin = sin(bearing)`.
- [x] Compute epoch-level circular summaries.
- [x] Compute distance-by-bearing histograms per epoch and chaser.
- [x] Write the component under the selected chaser-distance run.
- [x] Update the GoodCopBadCop interactive spec to include egocentric source
      paths when the component exists.
- [x] Keep the chaser-distance run itself as the visualization/spec anchor.

## Marimo Checklist

- [x] Extend `GoodCopBadCopInteractiveData` with optional egocentric-bearing
      arrays and histogram arrays.
- [x] Add loader support for the component selected by
      `egocentric_bearing.attrs["latest_complete"]`.
- [x] Add a polar plot panel:
      angle = egocentric bearing, radius = distance, color = chaser id.
- [x] Allow epoch selection to reuse the existing epoch/custom time-window
      controls.
- [x] Add a point-cloud mode with downsampling when raw points are too dense.
- [x] Add a polar heatmap/density view from the selected epoch or custom
      time-window samples.
- [x] Add a fish-heading plot from the kinematics-derived heading stored in the
      egocentric component.
- [x] Add an alignment-vs-distance plot using binned summaries.
- [x] Show a clear message when the chaser-distance run has no egocentric
      component yet.

## Backfill Checklist

- [ ] Identify GoodCopBadCop zarrs under `/groups` that have chaser-distance
      runs but no offline track-kinematics run.
- [ ] Run or backfill offline track kinematics for those zarrs first.
- [ ] Run egocentric-bearing component generation after kinematics exists.
- [ ] Refresh each GoodCopBadCop interactive spec after the component is
      written.
- [ ] Verify a small subset in marimo before batch-applying all recordings.

Expected order:

```text
refined keypoints/refined detections
  -> offline track kinematics
  -> chaser distance
  -> egocentric chaser bearing component
  -> GoodCopBadCop marimo spec refresh
```

## Test Checklist

- [x] Unit-test angle convention with synthetic frames:
      heading right + object right = `0 deg`.
- [x] Unit-test image-y-down conversion:
      heading right + object above = `+90 deg`.
- [x] Unit-test object behind fish = `-180` or wrapped equivalent.
- [x] Unit-test dense heading alignment from sparse track `frame_indices`.
- [x] Unit-test invalid frames from missing heading, failed `sample_valid`,
      invalid fish position, and invalid chaser position.
- [x] Unit-test zarr writer output shape and attrs.
- [x] Unit-test interactive spec source paths when the component exists.
- [x] Unit-test marimo plotting helpers with fixture data.
- [x] Run `py_compile` in sandbox.
- [x] Run focused pytest outside the sandbox if tests touch zarr.
- [x] Run `marimo check` outside the sandbox for the notebook.

## Open Questions

- Which track id should be default for GoodCopBadCop recordings if multiple
  tracks exist? Current likely default is `track_id=0`, but batch tooling should
  report available ids and fail if ambiguous.
- Should the first production component use `smoothed_heading_degrees` only, or
  write both smoothed and raw components?
- What distance bin width should be default for pooled group analysis?
  Proposed initial default: `2 mm`.
- What bearing bin width should be default for pooled polar density?
  Proposed initial default: `15 deg`.

## Agent Execution Prompt

Implement GoodCopBadCop egocentric chaser bearing using the
kinematics-required policy in this document. Use
`analysis/track_kinematics_runs/offline/<run>` as the heading authority,
default to `smoothed_heading_degrees`, densify by `frame_indices` onto the
chaser-distance camera-frame axis, write the derived component under
`analysis/chaser_distance_runs/<run>/egocentric_bearing/<component>`, update the
GoodCopBadCop interactive spec/loader/marimo component to visualize polar
point clouds and distance-binned alignment, and add focused tests plus
validation commands.
