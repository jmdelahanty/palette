# Base Analysis Design: Moving Grating Response

## Goal

Compute a standard set of per-recording, per-step, per-fish metrics that answer:
**Is the fish following the grating, swimming against it, or ignoring it — and how
does this change over time?**

These metrics should be computed automatically after tracking completes and stored
in the zarr for quick retrieval, dashboard display, and cross-recording aggregation.

## Available Data

### Tracking data (per fish, per frame)

From `analysis/movement_runs/<run>/tracks/id_<track_id>/`:

| Array | Description |
|-------|-------------|
| `frame_indices` | Frame number for each sample |
| `time_seconds` | Time in seconds |
| `positions_px` / `positions_mm` | (x, y) position |
| `heading_degrees` / `heading_radians` | Fish body orientation (swim_bladder → midpoint of eyes) |
| `smoothed_heading_degrees` | Temporally smoothed heading |
| `speed_smoothed_mm` | Smoothed swimming speed (mm/s) |
| `speed_filtered_mm` | Hysteresis-filtered speed (mm/s) |
| `displacement_smoothed_mm` | Frame-to-frame displacement |
| `cumulative_distance_mm` | Total distance traveled |

### Swim bout data (per fish)

From `analysis/swim_bout_runs/<run>/`:

| Field | Description |
|-------|-------------|
| `start_frame` / `end_frame` | Bout boundaries |
| `duration_s` | Bout duration |
| `mean_speed` / `peak_speed` | Bout speed metrics |
| `distance` | Distance per bout |

### Stimulus data

From `analysis/stimulus_runs/<run>/`:

**Protocol parameters** (constant per step):
- `orientation_degrees` — grating drift direction (0-360)
- `speed_mm_per_sec` — grating speed
- `spatial_freq_cycles_per_mm` — spatial frequency
- `duty_cycle` — light/dark ratio

**Event timing**:
- `events/event_type` = 11 (STEP_START) / 12 (STEP_END)
- `events/step_index` — which protocol step
- `events/camera_frame_id` — camera frame at event
- `events/relative_timestamp_ns` — nanosecond timestamp

**Frame alignment**:
- `frame_alignment/camera_to_metadata_index` — camera frame → stimulus metadata row

## Coordinate System Note

The grating `orientation_degrees` defines the **direction of drift** in projector/texture
space. Fish `heading_degrees` is computed from keypoints in **camera space**. A
coordinate transform is needed to align them — the `coordinate_transform` attribute
on the stimulus run provides the mapping between texture space (358x358) and camera
space (4512x4512). For grating analysis, since we only need the angular relationship
(not position), the key factor is whether the camera view is flipped/rotated relative
to the projector. This should be validated per-rig and stored as a constant offset.

## Proposed Metrics

### 1. Heading alignment (the core metric)

For each fish at each frame, compute the **alignment angle** between the fish's
swimming direction and the grating drift direction:

```
alignment_angle = fish_heading - grating_direction   (mod 360, centered to [-180, +180])
```

From this, derive:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `alignment_cos` | cos(alignment_angle) | +1 = following grating, -1 = opposing, 0 = perpendicular |
| `alignment_sin` | sin(alignment_angle) | positive = turning left relative to grating, negative = right |

**Time series** (per fish, per frame):
- `alignment_angle_deg` — raw angle difference (-180 to +180)
- `alignment_cos` — cosine of alignment (scalar -1 to +1)

**Per-step summary** (per fish, per protocol step):
- `mean_alignment_cos` — mean cosine across all frames in the step. Positive = net following.
- `resultant_vector_length` — circular mean resultant (0 = random, 1 = perfectly consistent direction)
- `mean_alignment_angle_deg` — circular mean of alignment angle
- `fraction_following` — fraction of frames where `alignment_cos > 0` (fish heading within ±90° of grating)
- `fraction_opposing` — fraction of frames where `alignment_cos < 0`
- `fraction_perpendicular` — fraction where `|alignment_cos| < 0.25` (within ~75-105° of grating)

### 2. Speed-weighted alignment

A fish sitting still and "aligned" with the grating is not meaningfully following it.
Weight alignment by swimming speed to capture active following:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `speed_weighted_alignment` | mean(speed × alignment_cos) / mean(speed) | Following strength weighted by how much the fish is actually swimming |
| `following_speed_mm_s` | mean(speed where alignment_cos > 0) | Speed when swimming with grating |
| `opposing_speed_mm_s` | mean(speed where alignment_cos < 0) | Speed when swimming against grating |

### 3. Movement summary (per step)

| Metric | Description |
|--------|-------------|
| `total_distance_mm` | Total distance traveled during this step |
| `mean_speed_mm_s` | Mean swimming speed |
| `median_speed_mm_s` | Median swimming speed |
| `max_speed_mm_s` | Peak swimming speed |
| `fraction_moving` | Fraction of frames where speed > threshold (~2 mm/s) |
| `num_swim_bouts` | Number of swim bouts |
| `mean_bout_duration_s` | Mean swim bout duration |
| `mean_bout_speed_mm_s` | Mean speed during bouts |
| `mean_interbout_interval_s` | Mean time between bouts |

### 4. Temporal dynamics (time series, binned)

Bin the step into time windows (e.g., 1-second bins) and compute per-bin:

| Metric | Description |
|--------|-------------|
| `binned_alignment_cos` | Mean alignment cosine per time bin |
| `binned_speed_mm_s` | Mean speed per time bin |
| `binned_fraction_following` | Fraction of frames following grating per bin |

This enables plotting how alignment evolves over the course of a stimulus step
(e.g., does the fish start following gradually? does it habituate?).

### 5. Bout-level alignment

For each swim bout, compute:

| Metric | Description |
|--------|-------------|
| `bout_mean_alignment_cos` | Mean alignment during the bout |
| `bout_initial_heading_alignment` | Alignment at bout onset (first 100ms) |
| `bout_is_following` | Boolean: bout_mean_alignment_cos > 0.5 |
| `bout_is_opposing` | Boolean: bout_mean_alignment_cos < -0.5 |

This answers: "When the fish decides to swim, does it choose to go with or against
the grating?"

## Storage Layout

```
analysis/grating_response_runs/<run_name>/
├── attrs:
│   ├── source_movement_run: str
│   ├── source_stimulus_run: str
│   ├── source_bout_run: str (optional)
│   ├── created_utc: str
│   ├── parameters: {bin_size_s, moving_threshold_mm_s, camera_to_projector_angle_offset}
│   └── summary: {per-recording aggregate stats}
│
├── per_step/
│   ├── step_index:          int32[n_steps]
│   ├── step_name:           UTF-8[n_steps]
│   ├── stimulus_mode:       UTF-8[n_steps]
│   ├── grating_direction_deg: float32[n_steps]
│   ├── grating_speed_mm_s:   float32[n_steps]
│   ├── grating_spatial_freq:  float32[n_steps]
│   ├── start_frame:         int64[n_steps]
│   ├── end_frame:           int64[n_steps]
│   ├── duration_s:          float32[n_steps]
│   └── (step-level metadata for joining)
│
├── per_fish_per_step/
│   ├── fish_id:               int32[n_fish × n_steps]
│   ├── step_index:            int32[n_fish × n_steps]
│   ├── mean_alignment_cos:    float32[...]
│   ├── resultant_vector_length: float32[...]
│   ├── mean_alignment_angle_deg: float32[...]
│   ├── fraction_following:    float32[...]
│   ├── fraction_opposing:     float32[...]
│   ├── speed_weighted_alignment: float32[...]
│   ├── following_speed_mm_s:  float32[...]
│   ├── opposing_speed_mm_s:   float32[...]
│   ├── total_distance_mm:     float32[...]
│   ├── mean_speed_mm_s:       float32[...]
│   ├── max_speed_mm_s:        float32[...]
│   ├── fraction_moving:       float32[...]
│   ├── num_swim_bouts:        int32[...]
│   ├── mean_bout_duration_s:  float32[...]
│   └── mean_interbout_interval_s: float32[...]
│
├── time_series/
│   ├── fish_id:             int32[n_fish × n_steps × n_bins]
│   ├── step_index:          int32[...]
│   ├── bin_center_s:        float32[...]
│   ├── alignment_cos:       float32[...]
│   ├── speed_mm_s:          float32[...]
│   └── fraction_following:  float32[...]
│
└── per_bout/  (optional, if swim bout data available)
    ├── fish_id:             int32[n_bouts]
    ├── step_index:          int32[n_bouts]
    ├── bout_id:             int32[n_bouts]
    ├── mean_alignment_cos:  float32[n_bouts]
    ├── initial_alignment:   float32[n_bouts]
    ├── is_following:        bool[n_bouts]
    └── is_opposing:         bool[n_bouts]
```

## Computation Pipeline

```
Input:
  movement_run (positions, headings, speeds per fish per frame)
  stimulus_run (protocol_json, events for step boundaries)
  bout_run     (optional, swim bout segments)

Step 1: Parse protocol → extract grating steps (stimulus_mode = MOVING_GRATING)
Step 2: Extract step boundaries from events (STEP_START/STEP_END → camera frame ranges)
Step 3: For each fish, for each grating step:
   a. Slice heading + speed time series to the step's frame range
   b. Compute alignment_angle = heading - grating_direction (with coordinate transform)
   c. Compute frame-level alignment_cos, alignment_sin
   d. Aggregate to per-step summary metrics
   e. Bin into time windows for temporal dynamics
Step 4: If bout data available, join bouts to steps and compute bout-level alignment
Step 5: Write results to zarr
Step 6: Register in recording_step_status as a new step (e.g., "grating_analysis")
```

## Open Questions

1. **Coordinate transform validation**: Need to verify the angular relationship between
   camera heading and projector grating direction for each rig. Is there a fixed offset?
   A mirror flip? This needs a one-time calibration check per rig setup.

2. **Moving vs stationary threshold**: What speed threshold separates "actively swimming"
   from "drifting/stationary"? The existing hysteresis filter uses 2.0 px high / 1.0 px
   low. Should we use the same, or a mm-based threshold (e.g., 2 mm/s)?

3. **Bin size for temporal dynamics**: 1 second seems reasonable for typical 30-120s steps.
   Should this be configurable?

4. **Multi-dish handling**: In multi-dish setups, each dish may have a different grating
   orientation (if the projector covers multiple arenas). Need to confirm whether
   `orientation_degrees` applies per-arena or globally.

5. **Reactive grating modules**: Some gratings use `reactive_logic_module_name` (e.g.,
   "OrientationMirrorsXPosition") which dynamically adjusts the grating based on fish
   position. For these, `orientation_degrees` is not constant — we'd need frame-level
   grating direction, which may not be logged. Should reactive gratings be handled in
   a separate analysis module?

6. **Registry integration**: Should grating response metrics be stored in a new registry
   table (like `grating_response_quality`) for cross-recording comparison? Or is the
   zarr-level storage sufficient for now?

## Future Extension: Other Stimulus Types

This design is grating-specific but the pattern generalizes:

| Stimulus | Key alignment metric |
|----------|---------------------|
| MOVING_GRATING | Heading vs drift direction (this doc) |
| COHERENT_DOTS | Heading vs dot motion direction |
| LOOMING_DOT | Distance/speed relative to loom center, escape latency |
| CHASER | Distance to chaser, escape angle, response latency (existing chaser metrics) |
| MOVING_DOTS | Heading vs dot direction, pursuit behavior |
| SOLID_BLACK/WHITE | Baseline movement (no stimulus alignment) |

A shared "base analysis" framework would:
1. Parse step boundaries from events (shared across all types)
2. Compute generic movement metrics per step (shared)
3. Dispatch to stimulus-specific alignment computation
4. Store in a consistent zarr layout with stimulus-type-specific sub-groups
