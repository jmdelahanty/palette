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

- Raw `detect_runs/<run>` outputs are immutable model/blob outputs. Spatial
  cleanup belongs in the refined run, not by rewriting raw predictions.
- If `analysis_metadata.attrs["dish_mask"]` exists, refinement applies it as a
  bbox-center gate after raw quality labels are resolved and before curated
  `instances/` are selected. Outside-dish candidates remain auditable in
  `source_detections` with reason `outside_dish_mask`.
- `--per-frame-top-k` is a curated-surface selection policy. Non-top clean
  candidates remain in `source_detections` as `duplicate` with reason
  `per_frame_top_k_excluded`.
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

`ambiguous` means the dense/single-slot compatibility view cannot represent a
frame as one obvious detection, usually because there are multiple source
candidates or multiple curated instances for that frame. It is a review/UI
state, not a biological label and not automatically a failed detection.

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
