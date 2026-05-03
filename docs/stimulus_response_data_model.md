# Stimulus Response Data Model

Why the data is organized step-first with per-fish arrays embedded within
each step, rather than fish-first with per-step arrays.

## Layout

```
stimulus_response_runs/<run>/
├── frames/                        ← recording-wide annotation
│   ├── step_index     [n_frames]
│   └── stimulus_mode_id [n_frames]
├── global/                        ← recording-wide per-fish summary
│   ├── fish_id        [n_fish]
│   └── ...
└── steps/                         ← one group per protocol step
    ├── step_0/
    │   ├── attrs: stimulus_mode, start_frame, end_frame, stimulus_params
    │   ├── per_fish/              ← base metrics, indexed by fish position
    │   ├── per_bout/              ← optional, variable length per step
    │   └── grating/               ← only on MOVING_GRATING steps
    ├── step_1/
    │   ├── per_fish/
    │   └── concentric_grating/    ← only on CONCENTRIC_GRATING steps
    └── step_N/
        └── per_fish/
```

## Why step-first, not fish-first

### Steps have heterogeneous structure

A SOLID_BLACK step has only `per_fish/`. A MOVING_GRATING step adds
`grating/` with per-frame alignment, per-fish summary, and time series.
A CONCENTRIC_GRATING step adds `concentric_grating/` with radial
decomposition. In a fish-first layout, each fish group would need to carry
stimulus-specific subgroups that only apply to some steps — lots of empty
or absent groups.

### Steps own their metadata

Each step has `stimulus_mode`, `start_frame`, `end_frame`, `duration_s`,
`stimulus_params` (orientation, speed, center point, etc.). These are
properties of the experimental condition, not the fish. In step-first,
they're attrs on the step group — natural and self-contained. In
fish-first, you'd need a separate step metadata table that every fish
group references.

### The biological question is usually step-scoped

"How did fish respond to this grating presentation?" → read one step group.
"Compare baseline vs grating?" → read step_0 and step_1 side by side.
"Average across all 90-degree grating repetitions?" → filter steps by
`stimulus_params.orientation_degrees == 90` and aggregate. All natural
in step-first.

### Bout arrays are variable length per step

A fish might have 3 bouts during a 30-second grating step and 0 bouts
during a 10-second baseline step. The `per_bout/` group has a different
number of rows per step. In step-first, each step's `per_bout/` is
self-contained. In fish-first, you'd need variable-length bout arrays
per fish per step — much harder to manage.

### Fish indexing is consistent without duplication

The `fish_id` array in every step's `per_fish/` uses the same ordering
as `global/fish_id`. Position `i` in `per_fish/speed_mm_s` always refers
to the same fish across all steps. No per-fish group needed for indexing.

Stimulus-specific subgroups (e.g., `grating/per_fish/`) inherit the parent
`per_fish/` ordering — they don't duplicate `fish_id`. Row N in
`grating/per_fish/mean_alignment_cos` corresponds to row N in
`per_fish/fish_id`.

## Accessing data

### All fish in one step

```python
step = sr["steps/step_3"]
fish_ids = step["per_fish/fish_id"][:]
speeds = step["per_fish/mean_speed_mm_s"][:]
alignment = step["grating/per_fish/mean_alignment_cos"][:]  # if grating step
```

### One fish across all steps

```python
fish_idx = 0  # position in fish_id array (same in every step)
for i in range(n_steps):
    step = sr[f"steps/step_{i}"]
    speed = step["per_fish/mean_speed_mm_s"][fish_idx]
    mode = step.attrs["stimulus_mode"]
```

### Group steps by stimulus type or parameters

```python
grating_90_steps = [
    sr[f"steps/step_{i}"]
    for i in range(n_steps)
    if sr[f"steps/step_{i}"].attrs["stimulus_mode"] == "MOVING_GRATING"
    and sr[f"steps/step_{i}"].attrs["stimulus_params"].get("orientation_degrees") == 90.0
]
```

### Continuous recording trace

```python
# Speed from track_kinematics (the data lives there, not duplicated).
speed = track["speed_smoothed_mm"][:]

# Stimulus context from stimulus_response (annotation only).
step_idx = sr["frames/step_index"][:]
mode_id = sr["frames/stimulus_mode_id"][:]

# Color-coded plot.
plt.scatter(range(len(speed)), speed, c=mode_id, cmap="tab10", s=1)
```

## Cross-step aggregation

The per-step structure does not include cross-step summaries (e.g.,
"mean alignment across all 90-degree grating repetitions"). This is a
consumer concern:

- **Notebooks**: iterate steps, filter by params, aggregate with numpy.
- **Export artifact**: a future cross-recording analysis tool could
  produce summary tables grouped by stimulus type and parameters.

The per-step data is the foundation. Cross-step aggregation builds on it
without requiring a different storage layout.

## Related documents

- `docs/stimulus_response_analysis_flow.md` — data flow and provenance
- `docs/stimulus_response_run_design.md` — full metric definitions
- `docs/stimulus_response_implementation_plan.md` — design decisions
- `docs/concentric_omr_stimulus_response_design.md` — planned radial OMR
  metrics for concentric grating steps
- `src/fisheye/docs/zarr_structure.md` — authoritative zarr layout
