# Stimulus Response Implementation Plan

Companion to `stimulus_response_run_design.md` (storage layout and metric
definitions). This document covers implementation sequencing, design decisions,
and the dependency contracts needed to build the framework.

## Status

All three passes are implemented and tested (56 unit tests).

| Pass | Status | Module |
|------|--------|--------|
| 1. Base framework | Done | `src/fisheye/analysis/stimulus_response.py` |
| 2. Bout integration | Done | Same module, optional `--bout-run` |
| 3. Grating metrics | Done | Same module, MOVING_GRATING dispatch |

Related files:

- `src/fisheye/shared/zarr/analysis_stage_arrays.py` — input/output ArraySpecs
- `tests/unit/fisheye/test_stimulus_response.py` — 46 tests
- `tests/unit/fisheye/test_analysis_stage_arrays.py` — 10 tests
- `src/fisheye/docs/zarr_structure.md` — updated with `stimulus_response_runs/` layout

Canonical moving-grating downstream runner:

```bash
scripts/run_moving_grating_downstream_pipeline.sh --apply
```

The runner chains `arena_assignment` / `tracking_runs`, `track_kinematics`,
optional `detect_bouts_multi_level`, and `stimulus_response` for the current
moving-grating canary. It defaults to a dry run unless `--apply` is provided,
and exposes the source Zarr, crop run, stimulus run, keypoint run, and output
run names as flags. See
`docs/moving_grating_downstream_prerequisites.md` for the procedural blockers
and required rowset alignment.

The current canary uses crop-row routing as a historical rescue because its
refined keypoints align to a crop rowset with more rows than the curated
refined-detect `instances` table. That is not the desired steady-state design.
Clean modern datasets should be rebuilt as coherent run generations from one
canonical refined instance rowset, with crops, keypoints, tracking, kinematics,
bouts, and stimulus response all recording and validating the same
`source_rowset_path`.

This runner is canonical operational glue, not yet a `fisheye.core.pipeline`
stage. Formal pipeline integration is tracked in Deferred Work below.

Direct stimulus-response usage:

```bash
scripts/py -m fisheye.analysis.stimulus_response <zarr_path> \
    --track-kinematics-type offline \
    --moving-threshold-mm-s 2.0 \
    --camera-to-projector-offset-deg 0.0 \
    --bin-size-s 1.0
```

## Decisions

### Bout producer: `detect_bouts_multi_level`

Two bout detection paths exist:

- `detect_bouts_multi_level.py`: threshold-based, reads from track_kinematics,
  internal code, 4 speed levels, well-understood output.
- `swim_bout_statistics.py`: external `chaser_analysis` library,
  re-detects from scratch, opaque output schema.

Canonical producer for stimulus_response: **`detect_bouts_multi_level`**.
Uses the `default_level` attribute (speed_smoothed) for downstream consumption.

### Dense frame representation at consumer

Track kinematics stores **sparse arrays** — only frames with valid detections.
A track detected at frames `[0, 1, 2, 5, 6, 7]` has arrays of length 6.
Gaps are implicit (absent from array), not explicit (NaN or zero).

stimulus_response expands sparse to dense on load:

```python
speed = np.zeros(n_frames, dtype=np.float32)
valid = np.zeros(n_frames, dtype=bool)
speed[frame_indices] = track_group["speed_smoothed_mm"][:]
valid[frame_indices] = True
```

This makes step slicing positional (`array[start:end]`), gaps are zeros,
and coverage is `valid[start:end].mean()`.

This expansion is local to stimulus_response. A future improvement
(see `docs/analysis_dense_array_migration_todo.md`) would move dense
production upstream to track_kinematics so all consumers benefit.

### Specs: lightweight input contract, proper output spec

No full ArraySpec retrofit of existing analysis modules. Instead:

- **Input read contract**: tuple of 6 ArraySpecs validated per track at load
  time. Fail fast with a clear message if track_kinematics output doesn't match.
- **Output ArraySpec**: full spec for stimulus_response output arrays, validated
  after write.

### Provenance from day one

Use `build_stage_provenance()` / `write_stage_provenance()` for all
stimulus_response output. Record source runs, parameters, git info.

## Implementation Passes

### Pass 1: Base framework

Per-step movement summaries for all stimulus types.

Files:

- `src/fisheye/shared/zarr/analysis_stage_arrays.py` (new)
- `src/fisheye/analysis/stimulus_response.py` (new)
- `tests/unit/fisheye/test_stimulus_response.py` (new)
- `tests/unit/fisheye/test_analysis_stage_arrays.py` (new)

Input read contract (6 arrays from track_kinematics per track):

- `frame_indices` (int64)
- `time_seconds` (float32)
- `positions_mm` (float32, 2D)
- `heading_degrees` (float32)
- `speed_smoothed_mm` (float32)
- `angular_velocity_deg_s` (float32)

Output:

- `global/`: recording-wide per-fish (distance, speed, active time, fraction moving)
- `steps/step_{i}/per_fish/`: per-step per-fish base metrics + **coverage**
  (fraction of step frames with valid detection data)

Biological value: "During SOLID_BLACK steps, fish swam X mm at Y mm/s.
During MOVING_GRATING steps, fish swam Z mm at W mm/s."

### Pass 2: Bout integration

Per-step bout metrics from `detect_bouts_multi_level`.

Extends stimulus_response.py with:

- Bout loading from `swim_bout_runs/` (default_level)
- Per-step bout filtering by frame range
- Per-fish: num_bouts, mean_bout_duration, mean_interbout_interval
- Per-bout: fish_id, bout boundaries, speed

Biological value: "Fish initiated more swim bouts during grating
presentation vs baseline."

### Pass 3: Grating metrics

Heading alignment, optomotor gain, temporal dynamics for MOVING_GRATING steps.

Angular accuracy depends on acquisition calibration
(`docs/grating_analysis_acquisition_questions.md`). Implementation uses
a configurable `camera_to_projector_offset_deg` (default 0.0), flagged
in provenance. Results are valid relative to the offset; absolute accuracy
requires calibration answers.

Adds:

- Per-frame: alignment_angle_deg, alignment_cos, speed_along_grating
- Per-fish: mean alignment, optomotor gain, drift, latency to follow
- Time series: binned alignment, speed, gain (default 1s bins)
- Per-bout: alignment during bouts

Biological value: "Fish aligned heading with grating direction
(mean cos = 0.7), with optomotor gain of 0.4, onset latency of 2.1s."

## Deferred Work

- Promote the moving-grating downstream chain into the formal
  `fisheye.core.pipeline` stage system instead of relying only on the shell
  runner. This likely means adding derived analysis stages for
  `track_kinematics`, `swim_bout_runs`, optional `bout_kinematics_runs`, and
  `stimulus_response_runs`, with explicit dependencies, stale-state semantics,
  and registry/status reporting.
- Concentric grating metrics (Layer 4 in comprehensive map)
- Eye angle integration (Layer 5)
- Full ArraySpec coverage for existing analysis modules (Layer 6)
- Dense array production at track_kinematics source
  (see `docs/analysis_dense_array_migration_todo.md`)
- Retire `compute_speed.py` as standalone stage
- Consolidate bout producers (deprecate `swim_bout_statistics.py`)
- Provenance retrofit for existing analysis modules

## References

- `docs/stimulus_response_run_design.md` — storage layout and metric definitions
- `docs/grating_analysis_acquisition_questions.md` — calibration blockers
- `docs/track_kinematics_bout_status.md` — known issues with bout mirroring
- `docs/analysis_dense_array_migration_todo.md` — future dense array work
