# Moving-Grating Downstream Prerequisites

Purpose: short operational checklist for running the moving-grating downstream
analysis chain.

Canonical runner:

```bash
scripts/run_moving_grating_downstream_pipeline.sh --apply
```

## Required Sequence

| Step | Requires | Produces | Why it matters |
|------|----------|----------|----------------|
| Dish mask tuning | Import/raw video access and working `cv2` in `palette-py311` | `analysis_metadata.attrs["dish_mask"]` | Defines the spatial dish ROI used by single-dish arena assignment. |
| Arena assignment | Dish mask plus the exact source rowset to label | `arena_assignment_runs/<run>` | Labels each row with an arena/dish ID. |
| Tracking | Arena assignment on the same rowset | `tracking_runs/<run>` | Converts arena IDs into stable run-local `track_id` values. |
| Track kinematics | Refined keypoints plus matching `tracking_runs` | `analysis/track_kinematics_runs/offline/<run>` | Computes speed, heading, distance, derivatives, and per-track movement traces. |
| Swim-bout candidates | Track kinematics | `analysis/swim_bout_runs/<run>` | Segments bouts from selected speed traces. |
| Stimulus response | Stimulus alignment plus track kinematics, optional swim bouts | `analysis/stimulus_response_runs/<run>` | Computes grating/step movement-response summaries. |

## Rowset Rule

The rowset used by `arena_assignment` must match the rowset consumed by
`track_kinematics`.

Modern datasets should use a single canonical instance rowset for the whole
post-detection generation. The intended steady-state source is the curated
refined instance surface, for example:

```text
refined_detect_runs/<run>/instances
```

From that canonical rowset, rerun the downstream generation coherently:

1. crop from the canonical instance rowset
2. keypoints from those crop rows
3. masks/shape from those crop rows, when needed
4. arena assignment and tracking against the same source rowset
5. track kinematics, swim bouts, and stimulus response from the same rowset

Each downstream run must record the exact `source_rowset_path` it consumes. A
new run generation should not be promoted to `latest` until row counts,
`frame_indices`, and source-row lineage agree across crop, keypoint, tracking,
and analysis outputs.

Movement-generation parameters are also part of the run generation identity.
For track kinematics, record and keep fixed:

- `hysteresis_high_px`
- `hysteresis_low_px`
- `hysteresis_min_frames`
- `hysteresis_band_policy` (`reset` for historical Palette behavior, `latch`
  for Schmitt-style dead-band behavior)
- smoothing method, smoothing alignment, and smoothing window

Do not partially rerun only one stage when row counts disagree. A partial rerun
can create a more dangerous archive: arrays may have the same names but no
longer refer to the same physical detections.

## Historical Canary Rescue

For the current moving-grating canary, refined keypoints are aligned to:

```text
crop_runs/crop_2026-02-10_21-05-18
```

Therefore arena assignment must run against that exact rowset:

```bash
scripts/py -m fisheye.tracking.arena_assignment \
  /path/to/analysis.zarr \
  --source-rowset crop_runs/crop_2026-02-10_21-05-18
```

Do not let arena assignment silently default to
`refined_detect_runs/<run>/instances` when downstream keypoints were generated
from a crop run with additional interpolated rows. The tracking array will have
the wrong row count.

This crop-row routing is a transitional rescue path for historical data, not
the preferred future design. If the goal is a clean modern dataset, rebuild a
coherent generation from the canonical refined instance rowset instead of
continuing to add new analysis runs on top of a mismatched historical rowset.

## Environment Blocker

Dish mask tuning uses OpenCV. Verify:

```bash
scripts/py -c 'import cv2; print(cv2.__version__)'
```

If this fails, install OpenCV into `palette-py311` before tuning masks.
