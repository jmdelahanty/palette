# Provenance TODO

## Goals
- Make every derived dataset (detect/refine/crop/keypoints/eye masks) auditable and reproducible.
- Capture enough context to re-run or explain any result without relying on external notes.

## High-priority fixes
- Add a `provenance` block to each run group with:
  - `git_commit`, `pipeline_version` (if exists), `config_path`, `config_hash`
  - `command`, `user`, `host`, `timestamp`
  - `environment` summary (python version + key package versions)
  - Reuse `get_environment_info()` (src/fisheye/utils/system.py) and align fields across run types
- Record model provenance where applicable:
  - `model_path`, `model_hash`, `model_version/tag`
- Persist the resolved source group used for downstream steps:
  - e.g., `resolved_detect_group` for crop/keypoints (manual/interpolated/filtered/raw)
- Link manual review artifacts:
  - Store `frame_flag_file` path or hash in review attrs
  - Store `retune_frame_flags` / `retune_flags` path or hash when used

## Medium-priority improvements
- Make `consolidated_metadata` the explicit source of truth, but optionally
  support a “materialize child attrs” helper for inspection/debug.
- Add provenance validation utility:
  - Scan recordings and report missing provenance fields per run type
- Store explicit run lineage per step:
  - `source_*_run` (detect → refine → crop → keypoints) on each run

## Open decisions
- Should we mirror consolidated metadata into child `zarr.json` files, or keep
  only consolidated metadata and require readers to merge?
- Standardize `provenance` schema across all run types in `src/fisheye/shared/zarr/schema.py`.
- Decide which environment details are required vs optional.

## Related docs
- `src/fisheye/docs/provenance_workflow.md`
- `docs/keypoint_review_policy.md`
- `docs/keypoint_review_status_notes.md`
