<!-- ARCHIVED 2026-07-17: superseded by sparse logical readers and authoritative frame-axis references. -->

# Analysis Dense Array Migration TODO

Last reviewed: 2026-04-26

## Problem

Track kinematics (`src/fisheye/analysis/track_kinematics.py`) stores
**sparse arrays** per track — only frames with valid detections are present.
A track detected at frames `[0, 1, 2, 5, 6, 7]` stores arrays of length 6
with a separate `frame_indices` array to map positions back to absolute
frame numbers.

This pushes gap-handling onto every downstream consumer:

- Consumers cannot use positional slicing (`array[start:end]`) for time-range
  queries. They must first find which array elements fall in the range via
  `frame_indices`.
- Frame gaps are implicit (absent), not explicit (NaN or zero). Consumers
  must infer gap locations from discontinuities in `frame_indices`.
- Metrics like "fraction moving" have ambiguous denominators — detected
  frames or total frames?
- Each consumer re-implements sparse-to-dense expansion independently.

## Current Mitigation

`stimulus_response.py` expands sparse to dense on load:

```python
speed = np.zeros(n_frames, dtype=np.float32)
valid = np.zeros(n_frames, dtype=bool)
speed[frame_indices] = track_group["speed_smoothed_mm"][:]
valid[frame_indices] = True
```

This is local to stimulus_response and works, but if additional consumers
need dense frame-aligned data, the expansion will be duplicated.

Important caveat:

- dense expansion must preserve the gap-aware distance semantics produced by
  `track_kinematics`
- consumers should not compute distance by taking `np.diff(...)` across only
  valid positions inside a window, because that can count movement across
  missing-frame gaps
- downstream distance summaries should consume dense versions of
  `frame_path_distance_*` or `cumulative_path_distance_*`, or apply the same
  consecutive frame rule as the source producer

## Proposed Change

Modify `track_kinematics.py` to produce **dense arrays** per track:

- Array length = total frames in the recording (or a configurable range)
- Gaps filled with NaN (for float arrays) or sentinel values (for int arrays)
- A `valid` boolean mask array per track indicating frames with real data
- `frame_indices` retained for backward compatibility and efficient
  sparse access

### Storage impact

At 120 fps for 30 minutes: ~216K frames. Per track:
- 37 float32 arrays x 216K = ~32 MB (uncompressed)
- With blosc/lz4 compression, NaN runs compress well — expect ~5-10 MB

This is acceptable for the recording sizes in this pipeline.

### Consumer impact

- `detect_bouts_multi_level`: currently reads speed arrays and applies
  thresholds. Dense arrays with NaN at gaps would require NaN-aware
  threshold detection. The existing speed arrays already contain NaN at
  gap boundaries in sparse form, so the change is minimal.
- `stimulus_response`: drops the 5-line sparse-to-dense expansion. Reads
  arrays directly with positional slicing.
- `swim_bout_statistics`: if retained, would need similar NaN handling.

### Backward compatibility

- Keep `frame_indices` array (useful for sparse access, audit)
- Add `valid` mask as a new array
- Existing consumers that index via `frame_indices` continue to work
- New consumers can use positional indexing directly

## When to Do This

After stimulus_response Pass 1 is working and producing results. This is
primarily a source-of-truth cleanup rather than a blocker, but it becomes a
correctness issue if more consumers independently expand sparse tracks and
recompute distance.

Trigger: when a second consumer needs dense frame-aligned data from
track_kinematics, the duplication justifies fixing the source.

## Related

- `docs/archive/stimulus_response_implementation_plan.md` — current consumer approach
- `docs/track_kinematics_bout_status.md` — known issues with track_kinematics
