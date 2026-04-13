Refinement Overview

As of 2026-04-07, detect refinement in Palette uses a sparse-first two-surface model:

- `detect_runs/<run>` stores raw detector output.
- `detect_runs/<run>/quality_reports/<quality_run>` stores raw detect artifact labels.
- `refined_detect_runs/<run>` stores the canonical curated detect surface.

Primary writes land on the sparse curated surfaces under the refined run:

- `refined_detect_runs/<run>/instances` for active curated bbox reads
- `refined_detect_runs/<run>/source_detections` for raw-candidate audit

Legacy `filtered/` and `interpolated/` groups remain compatibility surfaces for
older archives only.

## Detect Refinement

`fisheye.refinement.refine_detect` initializes `refined_detect_runs/<run>` by
consuming raw detect quality labels and writing sparse `instances/` plus
`source_detections/`.

Core semantics:

- `status_codes` is the machine-readable row state:
  - `present`
  - `missing`
  - `filtered_out`
  - `ambiguous`
- `source_kind_codes` records where the current curated row came from:
  - `none`
  - `raw_detect`
  - `interpolated` (legacy compatibility only)
  - `manual`
- `reason` remains explanatory only. It is not the primary machine-readable
  state.
- current sparse-first runs should normally resolve review status to
  `resolved_group="refined"`, not to legacy subgroup names

`fisheye.tune.detect_review` now supports:

- one slot per frame for the legacy single-subject workflow
- one slot per `(frame, arena_id)` when fixed sub-arena ROI definitions are
  available from subdish masks or arena-assignment metadata

It still does not support unconstrained multiple curated detections within the
same arena/ROI. The canonical sparse surfaces are:

- `instances/` for curated accepted rows
- `source_detections/` for raw candidate decisions

Legacy sparse manual subgroups may still exist in older archives, and readers
may still support them as fallback, but they are no longer the primary detect
contract.

For the practical workflow, see:

- `docs/detection_refinement_workflow.md`
- `docs/refined_detect_collapse_v2.md`
- `src/fisheye/docs/zarr_structure.md`

## Eye Mask Refinement

Eye-mask refinement is documented separately and is unaffected by the detect
collapse above.

See:

- `src/fisheye/docs/eye_mask_tuning_workflow.md`
- `src/fisheye/docs/zarr_structure.md`
