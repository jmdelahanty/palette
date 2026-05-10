# Stimulus Response Run Design

Supersedes the storage layout and identity model from the archived
`docs/archive/base_analysis_moving_grating_design.md`. Metric definitions from
that doc are preserved and expanded here.

## Goal

Define a general-purpose analysis run that answers stimulus-specific behavioral
questions for every protocol step in a recording. The implemented surface now
covers base step summaries, moving-grating alignment, moving-grating OMR, and
first-slice concentric radial OMR. The structure accommodates additional
stimulus types (looming dots, coherent dots, chaser, etc.) without changing
the base step-first layout.

Each run produces:

1. **Global** recording-wide movement summary per fish
2. **Per-step base metrics** (movement, bout counts) for every protocol step
3. **Stimulus-specific metrics** (e.g., grating alignment) only for steps whose
   stimulus type warrants them

---

## Identity Model

### Principle: pristine inputs

Stimulus response analysis is a **pure consumer** of identity-resolved data.
It does not perform identity resolution and has no knowledge of the
`arena_assignment` pipeline.

Track-kinematics runs already consume `arena_assignment_runs` and store consolidated,
per-fish tracks under `tracks/id_<fish_id>/`. By the time data reaches this
run, `fish_id` is a settled biological identity — not a tracker label.

### Why this matters

If a recording has multiple animals and an identity switch occurs mid-step
(e.g., two fish cross paths and the tracker swaps IDs), the resulting metrics
for that step are meaningless for both fish. No amount of downstream
bookkeeping fixes this — the correction must happen upstream in identity
resolution, which then cascade-invalidates everything below it.

### Provenance chain

```
arena_assignment ──► track_kinematics ──► bouts ─┐
                                      ├──► stimulus_response
              stimulus ──────────────┘
              eye_angles ────────────┘
```

This run stores references to its direct inputs:

| Attribute | Points to |
|-----------|-----------|
| `source_track_kinematics_run` | `analysis/track_kinematics_runs/<type>/<run>/` |
| `source_stimulus_run` | `analysis/stimulus_runs/<run>/` |
| `source_bout_run` | `analysis/swim_bout_runs/<run>/` (optional) |
| `source_eye_angle_run` | `analysis/eye_angle_runs/<run>/` (optional) |

No `source_id_assignment_run` is needed — identity resolution is the movement
run's responsibility. Eye angle data is optional — if unavailable, eye-related
grating metrics are omitted.

### Cascade invalidation

The stimulus response step is a dependent of `tracks` in the step dependency
graph (`step_cascade.py`). When any upstream step changes:

```
arena_assignment changes
  → movement marked "missing", recomputed
    → bouts marked "missing", recomputed
      → stimulus_response marked "missing", recomputed
```

---

## Storage Layout

The layout below is the current default hierarchical-v1 storage contract.
Compact-tabular-v2 is available as an explicit opt-in writer layout via
`--layout compact_tabular_v2`, and current reader migrations use
`fisheye.analysis.stimulus_response_io.resolve_stimulus_response_tables(...)`.
The compact layout is not the default yet. See
`docs/stimulus_response_compact_v2_design.md`.

```
analysis/stimulus_response_runs/<run_name>/
│
├── attrs:
│   ├── source_track_kinematics_run       str
│   ├── source_stimulus_run       str
│   ├── source_bout_run           str (optional)
│   ├── source_eye_angle_run      str (optional)
│   ├── parameters                dict {bin_size_s, moving_threshold_mm_s,
│   │                                    follow_latency_window_s, follow_threshold, ...}
│   ├── created_utc               str (ISO 8601)
│   ├── git_info                  dict {commit, branch, is_dirty}
│   ├── n_steps                   int
│   ├── n_fish                    int
│   └── fish_ids                  list[int]
│
├── global/                              # Recording-wide, per-fish
│   ├── fish_id               int32[n_fish]
│   ├── total_distance_mm     float32[n_fish]
│   ├── mean_speed_mm_s       float32[n_fish]
│   ├── total_bouts           int32[n_fish]
│   ├── total_active_s        float32[n_fish]
│   └── fraction_moving       float32[n_fish]
│
└── steps/
    └── step_{i}/                        # One group per protocol step
        │
        ├── attrs:                       # Step identity & stimulus params
        │   ├── step_index          int
        │   ├── step_name           str
        │   ├── stimulus_mode       str    # "MOVING_GRATING", "SOLID_BLACK", ...
        │   ├── start_frame         int
        │   ├── end_frame           int
        │   ├── duration_s          float
        │   └── stimulus_params     dict   # stimulus-specific, see below
        │
        ├── per_fish/                    # Base metrics (ALL stimulus types)
        │   ├── fish_id             int32[n_fish]
        │   ├── total_distance_mm   float32[n_fish]
        │   ├── mean_speed_mm_s     float32[n_fish]
        │   ├── median_speed_mm_s   float32[n_fish]
        │   ├── max_speed_mm_s      float32[n_fish]
        │   ├── num_bouts           int32[n_fish]
        │   ├── fraction_moving     float32[n_fish]
        │   ├── mean_bout_duration_s    float32[n_fish]
        │   └── mean_interbout_interval_s float32[n_fish]
        │
        ├── per_bout/                    # Bout-level (ALL stimulus types)
        │   ├── fish_id             int32[n_bouts_in_step]
        │   ├── bout_id             int32[...]
        │   ├── start_frame         int64[...]
        │   ├── end_frame           int64[...]
        │   ├── duration_s          float32[...]
        │   ├── mean_speed_mm_s     float32[...]
        │   └── peak_physical_speed_mm_s float32[...]
        │
        ├── grating/                     # MOVING_GRATING steps only
        │   │                            # This group does NOT exist for non-grating steps
        │   │
        │   ├── per_frame/               # Full-resolution per-frame data
        │   │   ├── frame_indices             int64[n_frames_in_step]
        │   │   ├── alignment_angle_deg       float32[n_fish x n_frames]  # 0=same dir, ±180=opposite
        │   │   ├── alignment_cos             float32[n_fish x n_frames]
        │   │   ├── speed_along_grating_mm_s  float32[n_fish x n_frames]  # projected onto grating dir
        │   │   ├── angular_velocity_deg_s    float32[n_fish x n_frames]  # heading change rate
        │   │   │   # --- eye alignment (present only if source_eye_angle_run provided) ---
        │   │   ├── left_eye_alignment_deg    float32[n_fish x n_frames]
        │   │   └── right_eye_alignment_deg   float32[n_fish x n_frames]
        │   │
        │   ├── per_fish/                # Indexed by parent per_fish/ fish_id ordering
        │   │   │ # --- heading alignment ---
        │   │   ├── mean_alignment_cos             float32[n_fish]
        │   │   ├── mean_alignment_angle_deg       float32[n_fish]
        │   │   ├── resultant_vector_length         float32[n_fish]
        │   │   ├── fraction_following              float32[n_fish]
        │   │   ├── fraction_opposing               float32[n_fish]
        │   │   ├── fraction_perpendicular          float32[n_fish]
        │   │   ├── speed_weighted_alignment        float32[n_fish]
        │   │   ├── following_speed_mm_s            float32[n_fish]
        │   │   ├── opposing_speed_mm_s             float32[n_fish]
        │   │   │ # --- optomotor gain ---
        │   │   ├── optomotor_gain                  float32[n_fish]
        │   │   │ # --- starting position & drift ---
        │   │   ├── initial_pos_along_grating_mm    float32[n_fish]  # starting position projected onto grating dir
        │   │   ├── initial_pos_perp_grating_mm     float32[n_fish]  # starting position perpendicular to grating dir
        │   │   ├── drift_along_grating_mm          float32[n_fish]
        │   │   ├── drift_perp_grating_mm           float32[n_fish]
        │   │   │ # --- latency & angular velocity ---
        │   │   ├── latency_to_follow_s             float32[n_fish]  # time to first sustained following (NaN if never)
        │   │   ├── mean_angular_velocity_deg_s     float32[n_fish]
        │   │   ├── onset_angular_velocity_deg_s    float32[n_fish]
        │   │   │ # --- eye alignment (optional) ---
        │   │   ├── mean_left_eye_alignment_deg     float32[n_fish]
        │   │   ├── mean_right_eye_alignment_deg    float32[n_fish]
        │   │   ├── mean_vergence_deg               float32[n_fish]
        │   │   └── mean_version_deg                float32[n_fish]
        │   │
        │   ├── time_series/             # Binned temporal dynamics
        │   │   ├── bin_center_s              float32[n_bins]
        │   │   ├── alignment_angle_deg       float32[n_fish x n_bins]
        │   │   ├── alignment_cos             float32[n_fish x n_bins]
        │   │   ├── speed_mm_s                float32[n_fish x n_bins]
        │   │   ├── fraction_following        float32[n_fish x n_bins]
        │   │   ├── optomotor_gain            float32[n_fish x n_bins]
        │   │   └── angular_velocity_deg_s    float32[n_fish x n_bins]
        │   │
        │   └── per_bout/                # Indexed by parent per_bout/ row ordering
        │       ├── mean_alignment_cos       float32[n_bouts_in_step]
        │       ├── mean_alignment_angle_deg float32[...]
        │       ├── initial_alignment_cos    float32[...]
        │       ├── is_following             bool[...]
        │       └── is_opposing              bool[...]
        │
        └── concentric_grating/          # CONCENTRIC_GRATING steps only
            │                            # This group does NOT exist for non-concentric steps
            │
            ├── per_frame/               # Full-resolution per-frame data
            │   ├── frame_indices             int64[n_frames_in_step]
            │   ├── distance_to_center_mm     float32[n_fish x n_frames]
            │   ├── radial_heading_angle_deg  float32[n_fish x n_frames]  # 0=toward center, ±180=away
            │   ├── radial_speed_mm_s         float32[n_fish x n_frames]  # positive=approaching center
            │   └── tangential_speed_mm_s     float32[n_fish x n_frames]  # speed perpendicular to radius
            │
            ├── per_fish/                # Indexed by parent per_fish/ fish_id ordering
            │   │ # --- radial position ---
            │   ├── mean_distance_to_center_mm      float32[n_fish]
            │   ├── initial_distance_to_center_mm   float32[n_fish]  # at step start
            │   ├── final_distance_to_center_mm     float32[n_fish]  # at step end
            │   ├── min_distance_to_center_mm       float32[n_fish]
            │   │ # --- centering behavior ---
            │   ├── net_radial_displacement_mm       float32[n_fish]  # negative = moved toward center
            │   ├── fraction_approaching             float32[n_fish]  # fraction of frames with radial_speed > 0
            │   ├── mean_radial_heading_cos          float32[n_fish]  # cos(radial_heading_angle); +1=toward center
            │   ├── time_to_center_s                 float32[n_fish]  # time to first reach center threshold (NaN if never)
            │   ├── fraction_near_center             float32[n_fish]  # fraction of time within threshold of center
            │   │ # --- speed decomposition ---
            │   ├── mean_radial_speed_mm_s           float32[n_fish]  # mean speed toward/away from center
            │   └── mean_tangential_speed_mm_s       float32[n_fish]  # mean speed orbiting around center
            │
            ├── time_series/             # Binned temporal dynamics
            │   ├── bin_center_s                float32[n_bins]
            │   ├── distance_to_center_mm       float32[n_fish x n_bins]
            │   ├── radial_speed_mm_s           float32[n_fish x n_bins]
            │   ├── radial_heading_cos          float32[n_fish x n_bins]
            │   └── fraction_approaching        float32[n_fish x n_bins]
            │
            └── radial_omr/              # Optional stimulus-aligned radial OMR outputs
                ├── per_frame/
                ├── per_bout/
                ├── per_fish/
                ├── windows/
                └── early_windows/
```

### Design rationale

**Hierarchical steps instead of flat tables.** The previous design used a
single `per_fish_per_step/` table with compound `(fish_id, step_index)` keys.
The hierarchical approach:

- Matches existing codebase patterns (movement runs use `tracks/id_N/`,
  bout runs use `speed_raw/`, `speed_filtered/` subgroups)
- Makes each step self-contained — load a single step with
  `root["steps/step_3/per_fish"]`, no filtering required
- Eliminates compound indexing
- Step metadata lives naturally as group attrs rather than in a separate table

The trade-off is that cross-step queries (e.g., "alignment for all fish across
all steps") require iterating over step groups. This is a simple loop and
keeps the data model cleaner.

**Global is recording-wide, per-step is independently computed.** The
`global/` group summarizes each fish across the entire recording regardless
of stimulus type. It establishes the canonical `fish_id` list and
recording-wide movement statistics. Per-step `per_fish/` contains movement
metrics computed independently over that step's frame range — these are not
copies or subsets of global arrays. The `global/` and per-step metrics answer
different questions: "how did this fish behave overall?" vs "how did this
fish behave during this specific stimulus period?"

**Stimulus-specific subgroups inherit parent indexing.** The `grating/`
subgroup does not duplicate `fish_id` or `bout_id` arrays. Its `per_fish/`
arrays are ordered identically to the parent `step_{i}/per_fish/fish_id`,
and its `per_bout/` arrays are ordered identically to the parent
`step_{i}/per_bout/`. To read grating alignment for a fish, read `fish_id`
from `per_fish/` and alignment metrics from `grating/per_fish/` — they're
aligned by position.

For future exported/merged analysis datasets (cross-recording queries like
"give me all MOVING_GRATING trials"), self-contained step data with redundant
`fish_id` arrays may be needed. That is an export concern, not a storage
concern — the canonical on-disk format avoids the duplication.

**Stimulus-specific subgroups are optional.** The `grating/` subgroup only
exists on steps where `stimulus_mode == "MOVING_GRATING"`. Non-grating steps
(SOLID_BLACK, SOLID_WHITE, etc.) have `per_fish/` and `per_bout/` but no
stimulus-specific subgroup. This keeps the schema honest — no NaN-filled
arrays for metrics that don't apply.

**Per-bout at two levels.** Base `per_bout/` (under the step) carries movement
metrics for all stimulus types. Grating-specific `grating/per_bout/` adds
alignment metrics that only make sense in the context of a grating stimulus.
The grating `per_bout/` inherits row ordering from the base `per_bout/` —
row N in `grating/per_bout/mean_alignment_cos` corresponds to row N in
`per_bout/bout_id`. This avoids mixing base and stimulus-specific bout data
in one table while keeping them joinable by position.

### Step attrs: `stimulus_params`

The `stimulus_params` dict in step attrs contains parameters specific to the
stimulus type. Its contents vary by `stimulus_mode`:

**MOVING_GRATING:**
```python
{
    "grating_direction_deg": 270.0,     # drift direction in camera space
    "grating_speed_mm_s": 10.0,
    "spatial_freq_cycles_per_mm": 0.5,
    "duty_cycle": 0.5,
    "camera_to_projector_offset_deg": 0.0,  # applied calibration correction
}
```

**SOLID_BLACK / SOLID_WHITE:**
```python
{}  # no stimulus-specific parameters
```

**CONCENTRIC_GRATING:**
```python
{
    "center_x_mm": 10.0,              # grating center in camera/mm space
    "center_y_mm": 10.0,
    "center_source": "stimulus_coordinates",
    "radial_polarity_authored": "expanding",  # Citrus-authored intent
    "radial_sign_authored": +1,       # +1 expanding/outward, -1 contracting/inward
    "radial_polarity_validated": False,
    "grating_speed_mm_s": 10.0,       # radial drift speed magnitude
    "spatial_freq_cycles_per_mm": 0.5,
    "center_threshold_mm": 2.0,       # radius defining "near center" for fraction_near_center
    "stimulus_role": "unknown",       # primary_stimulus, centering_utility, or unknown
    "target_radius_min_mm": None,     # optional centering utility metadata
    "target_radius_max_mm": None,
}
```

**LOOMING_DOT (future):**
```python
{
    "loom_center_x_mm": 5.0,
    "loom_center_y_mm": 5.0,
    "expansion_rate_deg_s": 20.0,
}
```

---

## Proposed Metrics

### Base metrics (all stimulus types)

These are computed for every protocol step regardless of stimulus type.

#### Per-fish per-step movement summary

| Metric | Description |
|--------|-------------|
| `total_distance_mm` | Total distance traveled during this step |
| `mean_speed_mm_s` | Mean swimming speed |
| `median_speed_mm_s` | Median swimming speed |
| `max_speed_mm_s` | Peak swimming speed |
| `fraction_moving` | Fraction of frames where speed > threshold |
| `num_bouts` | Number of swim bouts |
| `mean_bout_duration_s` | Mean swim bout duration |
| `mean_interbout_interval_s` | Mean time between bouts |

#### Per-bout movement

| Metric | Description |
|--------|-------------|
| `start_frame` / `end_frame` | Bout boundaries |
| `duration_s` | Bout duration |
| `mean_speed_mm_s` | Mean speed during bout |
| `peak_physical_speed_mm_s` | Peak physical speed during bout |

### Grating-specific metrics (MOVING_GRATING steps only)

#### 1. Per-frame heading–grating difference

For each fish at each frame, compute the angular difference between the
fish's heading and the grating drift direction:

```
alignment_angle_deg = fish_heading - grating_direction   (mod 360, centered to [-180, +180])
```

- **0°** = fish heading matches grating direction exactly (following)
- **±180°** = fish heading is opposite to grating direction (opposing)
- **±90°** = fish heading is perpendicular to grating direction

This per-frame value is stored in `grating/per_frame/alignment_angle_deg`
and is the foundation for all derived alignment metrics. `alignment_cos`
(cosine of this angle) is stored alongside it for convenience.

#### 2. Heading alignment summary (per fish, per step)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `alignment_cos` | cos(alignment_angle) | +1 = following, -1 = opposing, 0 = perpendicular |

| Metric | Description |
|--------|-------------|
| `mean_alignment_cos` | Mean cosine across all frames. Positive = net following. |
| `mean_alignment_angle_deg` | Circular mean of alignment angle |
| `resultant_vector_length` | Circular mean resultant length (0 = random, 1 = perfectly consistent) |
| `fraction_following` | Fraction of frames where `alignment_cos > 0` |
| `fraction_opposing` | Fraction of frames where `alignment_cos < 0` |
| `fraction_perpendicular` | Fraction of frames where `|alignment_cos| < 0.25` |
| `speed_weighted_alignment` | `mean(speed * alignment_cos) / mean(speed)` |
| `following_speed_mm_s` | Mean speed when `alignment_cos > 0` |
| `opposing_speed_mm_s` | Mean speed when `alignment_cos < 0` |

#### 3. Optomotor gain

The classic optomotor response metric: how well does the fish's swimming
speed match the grating speed?

```
speed_along_grating = fish_speed * cos(alignment_angle)   # per frame
optomotor_gain      = mean(speed_along_grating) / grating_speed
```

| Gain | Interpretation |
|------|----------------|
| 1.0 | Perfect following — fish matches grating speed |
| 0.0 | Stationary or perpendicular swimming |
| < 0 | Actively opposing the grating |
| > 1 | Swimming faster than the grating in its direction |

`speed_along_grating_mm_s` is stored per-frame. `optomotor_gain` is stored
per-fish and in the binned time series.

#### 4. Starting position & positional drift

A fish that starts near the wall the grating is moving toward has little
room to follow — it will hit the boundary quickly even if it's actively
tracking the stimulus. Recording starting position allows downstream
analysis to control for this confound (e.g., by normalizing drift by
available distance, or by stratifying fish by initial position).

| Metric | Description |
|--------|-------------|
| `initial_pos_along_grating_mm` | Fish position at step start, projected onto the grating drift direction (relative to arena center) |
| `initial_pos_perp_grating_mm` | Fish position at step start, perpendicular to grating direction |
| `drift_along_grating_mm` | Net displacement projected onto the grating drift direction (positive = displaced in grating direction) |
| `drift_perp_grating_mm` | Net displacement perpendicular to grating direction |

Drift is computed as the vector from position at step start to step end,
decomposed into components parallel and perpendicular to the grating
direction. Starting position is stored so that drift can be interpreted
in context — a 2mm drift means different things depending on whether the
fish started at the center or 1mm from the wall.

#### 5. Latency to follow

How quickly does the fish begin swimming in the grating direction after
step onset?

| Metric | Description |
|--------|-------------|
| `latency_to_follow_s` | Time from step onset to first sustained following episode (NaN if fish never follows) |

"Sustained following" is defined as a sliding window (configurable, default
1 second) where `mean(alignment_cos) > follow_threshold` (configurable,
default 0.5). This avoids counting a single frame of incidental alignment
as the onset of following behavior.

A fish with low latency orients quickly; a fish with high latency may be
slow to detect the stimulus or initially indifferent. `NaN` means the fish
never met the following criterion during the step.

#### 6. Angular velocity / turning

How rapidly does the fish change heading, and does turning behavior differ
at grating onset vs later?

| Metric | Description |
|--------|-------------|
| `angular_velocity_deg_s` | Per-frame heading change rate (°/s), stored in per_frame and time_series |
| `mean_angular_velocity_deg_s` | Mean absolute turning rate across the step |
| `onset_angular_velocity_deg_s` | Mean absolute turning rate in the first N seconds of the step (configurable, default 2s) |

Comparing onset vs sustained angular velocity reveals whether the fish
makes a quick orienting turn at grating onset and then holds course, or
turns continuously.

#### 7. Eye alignment (optional, requires `source_eye_angle_run`)

Per-frame eye orientation relative to the grating direction, computed from
`eye_angle_runs` ellipse-based angles and the fish heading.

**Per-frame:**

| Metric | Description |
|--------|-------------|
| `left_eye_alignment_deg` | Left eye orientation relative to grating direction |
| `right_eye_alignment_deg` | Right eye orientation relative to grating direction |

**Per-fish summary:**

| Metric | Description |
|--------|-------------|
| `mean_left_eye_alignment_deg` | Mean left eye angle relative to grating |
| `mean_right_eye_alignment_deg` | Mean right eye angle relative to grating |
| `mean_vergence_deg` | Mean binocular convergence angle during the step |
| `mean_version_deg` | Mean conjugate eye position during the step |

These metrics answer: are the eyes tracking the grating independently of
body heading? Does vergence change during grating presentation vs baseline
(SOLID_BLACK/WHITE) steps?

#### 8. Temporal dynamics (binned time series)

All key metrics binned into time windows (default 1 second) to capture
how behavior evolves over the step:

| Metric | Description |
|--------|-------------|
| `alignment_angle_deg` | Mean heading–grating difference per bin |
| `alignment_cos` | Mean alignment cosine per bin |
| `speed_mm_s` | Mean speed per bin |
| `fraction_following` | Fraction following per bin |
| `optomotor_gain` | Gain per bin |
| `angular_velocity_deg_s` | Mean turning rate per bin |

Enables plotting gradual onset, habituation, or oscillatory alignment
patterns across a stimulus step.

#### 9. Per-bout alignment

| Metric | Description |
|--------|-------------|
| `mean_alignment_cos` | Mean alignment during the bout |
| `mean_alignment_angle_deg` | Mean heading–grating difference during the bout |
| `initial_alignment_cos` | Alignment at bout onset (first 100ms) |
| `is_following` | `mean_alignment_cos > 0.5` |
| `is_opposing` | `mean_alignment_cos < -0.5` |

Answers: "When the fish decides to swim, does it choose to go with or against
the grating?"

#### 10. OMR responsiveness

OMR responsiveness metrics live under the moving-grating step as:

```text
steps/step_{i}/grating/omr/
  per_fish/
  per_bout/
  windows/
  early_windows/
```

These metrics are stricter response indices built from the same grating
direction and dense track data used by the existing grating metrics:

| Metric | Description |
|--------|-------------|
| `omr_path_index` | `sum(displacement · stimulus_direction) / sum(path_length)`, range `[-1, +1]` |
| `omr_net_direction_index` | Direction of net displacement relative to stimulus direction |
| `bout_fraction_correct_classified` | Fraction of classifiable bouts displaced along the stimulus direction |
| `bout_choice_index` | `(N_correct - N_opposing) / (N_correct + N_opposing)` |
| `bout_path_index` | Signed per-bout physical displacement sum normalized by per-bout path length sum |
| `bout_fraction_correct_weighted_by_path` | Fraction of classifiable bout path length contributed by aligned bouts |
| `bout_fraction_correct_weighted_by_displacement` | Fraction of classifiable bout displacement magnitude contributed by aligned bouts |
| `time_choice_index` | Time-weighted correct-vs-opposing movement index |
| `mean_position_axis_norm` | Mean arena-normalized occupancy on the stimulus axis, when arena geometry is available |
| `fraction_time_correct_side` | Fraction of valid frames spent on the stimulus-forward side of the arena |
| `opportunity_normalized_parallel_displacement` | Stimulus-axis displacement normalized by available arena space in the direction traveled |
| `first_aligned_bout_latency_s` | Latency from step onset to the first classifiable aligned bout |

`windows/` stores non-overlapping window summaries for distribution analyses.
`early_windows/` stores onset-anchored first-response windows, defaulting to
`5 s` and `10 s`, with the same physical path/time metrics plus weighted bout
summaries.

Detector-vs-estimator semantics are explicit: swim-bout runs provide event
boundaries only, while OMR displacement/path metrics are measured from
physical position arrays. See `docs/omr_stimulus_response_design.md` for the
full contract and open questions.

Missing metric values are stored as `NaN` in numeric arrays. Missing optional
metadata in attrs/provenance uses JSON `null`; writers must not emit JSON
`NaN` or `Infinity` in Zarr metadata because strict consumers reject those
files.

OMR review visualizations are persisted under the run-local visualization
contract when `stimulus_response` is run with `--write-zarr-artifacts`, or by
running:

```bash
scripts/py -m fisheye.analysis.plot_stimulus_response_omr <analysis.zarr> \
  --run <stimulus_response_run>
```

The canonical artifacts are:

```text
analysis/stimulus_response_runs/<run>/visualizations/
  stimulus_response_omr_summary_png
  stimulus_response_omr_summary_interactive/
  stimulus_response_omr_bout_trajectory_png
  stimulus_response_omr_bout_trajectory_interactive/
```

The PNG is a review snapshot showing step-level OMR direction indices,
arena-axis occupancy/opportunity, first classified/aligned/opposing bout
latencies, and windowed path-index traces. The interactive artifact is a small
spec that points viewers back to `steps/step_<i>/grating/omr/`; it does not
duplicate the numeric OMR arrays.

The bout-trajectory PNG is a spatial review snapshot inspired by Megabouts'
trajectory panels. It overlays the source track-kinematics `positions_mm` path
with OMR-classified bout segments from `grating/omr/per_bout`, heading arrows,
stimulus direction, and the arena outline when calibration is available. It is
intentionally spatial-only for now; yaw/heading time traces remain available
from track-kinematics arrays and should be added later only if they answer a
distinct QC question.

### Concentric grating metrics (CONCENTRIC_GRATING steps only)

Concentric gratings produce radially symmetric flow designed to drive fish
toward (or away from) a center point. The relevant axes are radial
(toward/away from center) and tangential (orbiting around center), rather
than a single linear direction.

This section describes the centering/polar-decomposition outputs. Optional
stimulus-aligned radial OMR metrics with explicit expanding/contracting
polarity are written under `concentric_grating/radial_omr/` and specified in
`docs/concentric_omr_stimulus_response_design.md`. That design explicitly
supports both primary radial-flow stimulus use and centering-utility use.

#### 1. Per-frame radial decomposition

For each fish at each frame, compute position and velocity relative to the
grating center:

```
distance_to_center = ||fish_position - grating_center||

radial_heading_angle_deg = angle between fish heading and the vector
                           pointing from fish toward center
                           (0° = heading toward center, ±180° = heading away)

radial_speed    = fish_speed * cos(radial_heading_angle)   # positive = approaching
tangential_speed = fish_speed * sin(radial_heading_angle)   # orbiting component
```

#### 2. Centering behavior (per fish, per step)

| Metric | Description |
|--------|-------------|
| `mean_distance_to_center_mm` | Mean distance to grating center across the step |
| `initial_distance_to_center_mm` | Distance at step start |
| `final_distance_to_center_mm` | Distance at step end |
| `min_distance_to_center_mm` | Closest approach to center |
| `net_radial_displacement_mm` | Final minus initial distance (negative = moved toward center) |
| `fraction_approaching` | Fraction of frames where radial speed > 0 (moving inward) |
| `mean_radial_heading_cos` | cos(radial_heading_angle); +1 = heading toward center, -1 = away |
| `time_to_center_s` | Time to first reach within `center_threshold_mm` of center (NaN if never) |
| `fraction_near_center` | Fraction of step time spent within `center_threshold_mm` |

#### 3. Speed decomposition (per fish, per step)

| Metric | Description |
|--------|-------------|
| `mean_radial_speed_mm_s` | Mean speed component toward/away from center (positive = inward) |
| `mean_tangential_speed_mm_s` | Mean speed component perpendicular to radius (orbiting) |

A fish that is effectively centered by the grating will show positive
`mean_radial_speed_mm_s` and decreasing `distance_to_center_mm` in the
time series. A fish that ignores the stimulus will show near-zero radial
bias and no distance trend.

#### 4. Temporal dynamics (binned time series)

| Metric | Description |
|--------|-------------|
| `distance_to_center_mm` | Mean distance to center per bin |
| `radial_speed_mm_s` | Mean radial speed per bin |
| `radial_heading_cos` | Mean radial heading cosine per bin |
| `fraction_approaching` | Fraction approaching per bin |

Captures whether centering is immediate or gradual, and whether the fish
overshoots and oscillates.

#### 5. Per-bout centering

| Metric | Description |
|--------|-------------|
| `mean_radial_heading_cos` | Mean radial heading cosine during the bout |
| `is_centering` | Bout directed toward center (`mean_radial_heading_cos > 0.5`) |
| `radial_displacement_mm` | Net radial movement during the bout (negative = inward) |

---

## Computation Pipeline

```
Inputs:
  movement_run    (identity-resolved positions, headings, speeds per fish per frame)
  stimulus_run    (protocol_json, events for step boundaries, frame alignment)
  bout_run        (optional: swim bout segments per fish)
  eye_angle_run   (optional: per-frame eye angles, vergence, version per fish)

Step 1  Load canonical stimulus steps
        Prefer analysis/stimulus_runs/<run>/steps/step_<i>, which are
        materialized from STEP_START / STEP_END events plus protocol JSON.
        For each step, record: step_index, step_name, stimulus_mode,
        start_frame, end_frame, duration_s, and stimulus-specific params.

Step 2  Compute global metrics
        For each fish, aggregate movement across the full recording:
        total distance, mean speed, total bouts, fraction moving.
        Write to global/ group.

Step 3  For each protocol step:
    3a  Slice each fish's movement time series to the step's frame range.
    3b  Compute base per-fish metrics (distance, speed, bouts, fraction moving).
    3c  Write step_{i}/per_fish/ and step_{i}/per_bout/.

    3d  If stimulus_mode == "MOVING_GRATING":
        - Look up grating_direction_deg from protocol params.
        - Apply coordinate transform (camera_to_projector_offset_deg).
        - Compute per-frame:
            alignment_angle_deg (heading - grating direction, centered [-180, +180])
            alignment_cos
            speed_along_grating_mm_s (speed projected onto grating direction)
            angular_velocity_deg_s (heading change rate)
        - If eye_angle_run available:
            Compute per-frame left/right eye alignment relative to grating.
        - Write step_{i}/grating/per_frame/.
        - Aggregate to per-fish summary:
            heading alignment, optomotor gain, positional drift,
            angular velocity, and (if available) eye alignment metrics.
        - Bin into time windows for temporal dynamics.
        - Compute per-bout alignment metrics.
        - Write step_{i}/grating/per_fish/, time_series/, per_bout/.
        - If OMR is enabled, compute path, bout, time, windowed,
          early-window, occupancy/opportunity, and first directed-bout metrics
          from physical estimator arrays, using swim-bout runs only as event
          boundary detectors.
        - Write step_{i}/grating/omr/.

    3e  If stimulus_mode == "CONCENTRIC_GRATING":
        - Resolve grating center position from canonical step attrs,
          stimulus-coordinate metadata, or calibration-backed fallbacks.
        - Compute per-frame:
            distance_to_center_mm
            radial_heading_angle_deg (0° = toward center, ±180° = away)
            radial_speed_mm_s, tangential_speed_mm_s
        - Write step_{i}/concentric_grating/per_frame/.
        - Aggregate to per-fish summary:
            centering metrics, speed decomposition, time_to_center.
        - Bin into time windows for temporal dynamics.
        - Compute per-bout centering metrics.
        - Write step_{i}/concentric_grating/per_fish/, time_series/, per_bout/.
        - If OMR is enabled, compute radial/tangential OMR metrics using the
          best available expanding/contracting polarity metadata. Preserve
          outward-positive physical components separately from
          stimulus-aligned radial components.
        - Write step_{i}/concentric_grating/radial_omr/.

    3f  [Future] If stimulus_mode == "LOOMING_DOT":
        - Compute escape metrics.
        - Write step_{i}/looming/ subgroup.

Step 4  Write run-level attrs (provenance, parameters, git_info).
        Update latest pointer on parent group.

Step 5  Register in recording_step_status as "stimulus_response" step.
```

---

## Extensibility: Adding a New Stimulus Type

To add support for a new stimulus type (e.g., LOOMING_DOT):

1. **Define stimulus-specific metrics** (escape latency, escape angle, etc.)

2. **Add a subgroup** under the step, named for the stimulus type:
   ```
   step_{i}/looming/
       ├── per_fish/
       │   ├── fish_id, escape_latency_s, escape_speed_mm_s,
       │   │   escape_angle_deg, ...
       │
       └── time_series/    (if applicable)
           └── ...
   ```

3. **Add a dispatch branch** in Step 3 of the computation pipeline that
   checks `stimulus_mode` and calls the stimulus-specific computation.

4. **Populate `stimulus_params`** in the step attrs with the relevant
   parameters for the new stimulus type.

No schema changes are needed to the base structure — `global/`, `per_fish/`,
`per_bout/` remain unchanged. The new stimulus type only adds its own
subgroup alongside the existing ones.

| Stimulus Type | Subgroup | Key Metrics |
|---------------|----------|-------------|
| MOVING_GRATING | `grating/` | Heading alignment, optomotor gain, positional drift, angular velocity, eye alignment |
| CONCENTRIC_GRATING | `concentric_grating/` | Distance to center, radial heading, centering fraction, radial/tangential speed, optional `radial_omr/` |
| COHERENT_DOTS | `coherent_dots/` | Heading vs dot motion direction, pursuit duration |
| LOOMING_DOT | `looming/` | Escape latency, escape angle, distance to loom center |
| CHASER | `chaser/` | Distance to chaser, escape angle, response latency |
| SOLID_BLACK / SOLID_WHITE | *(none)* | Base movement metrics only (baseline) |

---

## Coordinate System Note

The grating `orientation_degrees` defines the direction of drift in
projector/texture space. Fish `heading_degrees` is computed from keypoints
in camera space. A per-rig angular offset must be applied to align them.

This alignment is not optional for OMR. Directional projections compare the
fish's camera-space trajectory with the stimulus direction, so the stored
stimulus angle must first be corrected into camera coordinates. The current
CLI/attribute name, `camera_to_projector_offset_deg`, is legacy wording; in the
stimulus-response implementation it is the angular correction applied to the
stored stimulus direction before computing camera-space alignment metrics. The
corrected angle is wrapped into `[0, 360)`.

The current moving-grating canary uses an inverted projector orientation: the
recorded Citrus `0 deg` grating moved left in the camera view. That recording
therefore uses `camera_to_projector_offset_deg = 180.0`. Rigs where Citrus
directions are already expressed in camera-space should use `0.0`.

This offset is:

- Determined by calibration (see `grating_analysis_acquisition_questions.md`)
- Stored in `stimulus_params.camera_to_projector_offset_deg` per step
- Applied during computation (Step 3d) before computing alignment angles

See `grating_analysis_acquisition_questions.md` for the full list of
calibration questions that must be resolved before implementation.

---

## Reactive Grating Steps

Some protocol steps use a `reactive_logic_module_name` (e.g.,
`OrientationMirrorsXPosition`) that dynamically adjusts grating parameters
based on fish position or behavior. For these steps, `orientation_degrees`
in the protocol is only the initial value — the actual direction changes
during the step.

### Acquisition requirement: per-frame parameter logging

Reactive steps require **per-frame grating direction** logged by the
acquisition system. Per-frame logging is strongly preferred over event-based
(logging only on change) for the following reasons:

| | Per-frame | Event-based (changes only) |
|-|-----------|---------------------------|
| **Downstream consumption** | Index by frame, zero interpolation | Must reconstruct via step-function interpolation |
| **Continuous updates** | Captures every intermediate state | Misses values between events if module updates frequently |
| **Alignment** | Guaranteed frame-aligned via existing `camera_to_metadata_index` | Requires `camera_frame_id` on each event |
| **Storage cost** | ~240 KB/hour at 120Hz for one float32 — trivial | Minimal |
| **Trust** | No data loss possible | Silent gaps if event logger skips updates |

**Recommendation:** Log per-frame as the default. Events remain useful as
semantic markers (threshold crossings, mode switches) but should not be
the primary source of truth for continuous parameters.

Per-frame reactive parameters would be stored in the stimulus run:

```
stimulus_runs/<run>/reactive_params/
    step_index          int32[n_reactive_frames]     # which step this frame belongs to
    stimulus_frame_num  int64[n_reactive_frames]
    orientation_deg     float32[n_reactive_frames]   # actual grating direction at this frame
    # (additional reactive params as needed)
```

### How reactive steps fit the analysis schema

Reactive grating steps use the same `grating/` subgroup as static grating
steps. The only difference is how `grating_direction_deg` is resolved
during computation:

| Step type | `grating_direction_deg` source |
|-----------|-------------------------------|
| Static grating | Scalar from `stimulus_params` (constant for entire step) |
| Reactive grating | Per-frame array from `stimulus_runs/<run>/reactive_params/` |

The per-frame computation is identical:

```
alignment_angle_deg[frame] = fish_heading[frame] - grating_direction[frame]
```

For static steps, `grating_direction[frame]` is a broadcast scalar.
For reactive steps, it varies per frame.

Step attrs indicate whether the step is reactive:

```python
stimulus_params = {
    "grating_direction_deg": 270.0,              # initial value
    "is_reactive": True,
    "reactive_module_name": "OrientationMirrorsXPosition",
    # grating_direction_deg is overridden per-frame from reactive_params
}
```

All derived metrics (alignment summary, optomotor gain, drift, latency,
etc.) work identically — they aggregate from the per-frame values
regardless of whether the direction was constant or varying.

---

## Open Questions

1. **Coordinate transform validation**: The angular relationship between
   camera heading and projector grating direction must be verified per rig.
   Blocking for grating-specific metrics. See companion doc.

2. **Moving vs stationary threshold**: What speed separates "actively
   swimming" from "drifting/stationary"? Current hysteresis filter uses
   2.0 px / 1.0 px. Should this be a mm-based threshold (e.g., 2 mm/s)?

3. **Bin size for temporal dynamics**: 1 second default for 30-120s steps.
   Should this be configurable per run or fixed?

4. **Reactive grating logging**: Per-frame parameter logging is the
   recommended approach (see Reactive Grating Steps section above), but
   this requires acquisition-side changes. Until per-frame logging is
   available, reactive steps should be excluded from grating analysis or
   processed with the static initial `orientation_degrees` value (with a
   warning flag in the output).

5. **Registry integration**: Should stimulus response metrics feed a
   registry table (e.g., `stimulus_response_quality`) for cross-recording
   dashboards? Or is zarr-level storage sufficient for now?

6. **Multi-dish grating orientation**: In multi-dish setups, does each
   sub-arena receive the same grating orientation per step? Blocking for
   multi-dish recordings only.

---

## Implementation

The base framework, bout integration, and grating metrics are implemented in:

- `src/fisheye/analysis/stimulus_response.py`
- `src/fisheye/shared/zarr/analysis_stage_arrays.py`

See `docs/stimulus_response_implementation_plan.md` for design decisions,
sequencing, and deferred work. See
`docs/stimulus_response_compact_v2_design.md` for the compact-layout migration
plan. See `src/fisheye/docs/zarr_structure.md` for the authoritative zarr
layout reference.
