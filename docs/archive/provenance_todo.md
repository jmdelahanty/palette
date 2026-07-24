<!-- ARCHIVED 2026-07-17: superseded provenance TODO retained for history. -->

# Provenance TODO

## Goals
- Make every derived dataset (detect/refine/crop/keypoints/eye masks) auditable and reproducible.
- Capture enough context to re-run or explain any result without relying on external notes.

## Current status (offline dataset focus)
- Migrate offline non-refinement stage writers to `palette_stage_provenance` contract:
  - [x] `eye_masks_runs`
  - [x] `keypoints_runs`
  - [x] `detect_runs`
  - [x] `crop_runs`
  - [x] `arena_assignment_runs`
- Add a generic provenance backfill tool for legacy offline runs:
  - [x] inject `provenance.contract` when missing
  - [x] normalize git payload into canonical `provenance.git.commit` shape
  - [x] keep dry-run and non-destructive defaults
- Expand diagnostics gating for offline stages after backfill:
  - [x] add stricter contract checks stage-by-stage in `check_provenance_capture`
  - [x] preserve compatibility reads during migration window
- Diagnostics/filtering/lineage follow-ups:
  - [x] add `--zarr-use` filtering to `check_provenance_capture`
  - [x] add stale refined-detect lineage repair utility (`fix_refined_detect_lineage`)
  - [x] add JSON audit reporting for lineage repairs (`--json-report`)
  - [x] clear known training consistency mismatch via lineage repair + revalidation

## High-priority fixes

- [ ] Add full `provenance` block to each run group with:
  - `git_commit`, `pipeline_version` (if exists), `config_path`, `config_hash`
  - `command`, `user`, `host`, `timestamp`
  - `environment` summary (python version + key package versions)
  - Reuse `get_environment_info()` (src/fisheye/utils/system.py) and align fields across run types
- [ ] Record model provenance where applicable:
  - `model_path`, `model_hash`, `model_version/tag`
- [ ] Persist the resolved source group used for downstream steps:
  - e.g., `resolved_detect_group` for crop/keypoints (manual/interpolated/filtered/raw)
- [ ] Link manual review artifacts:
  - Store `frame_flag_file` path or hash in review attrs
  - Store `retune_frame_flags` / `retune_flags` path or hash when used

## Medium-priority improvements

- [ ] Record consolidated-metadata freshness as finalization provenance, while
  keeping direct child metadata as the correctness source of truth for mutable
  Palette stores.
  - Future writers should refresh consolidated metadata after successful
    mutations, but readers must still tolerate stale or absent consolidated
    metadata on local working archives.
- [ ] Add provenance validation utility:
  - Scan recordings and report missing provenance fields per run type
- [ ] Store explicit run lineage per step:
  - `source_*_run` (detect → refine → crop → keypoints) on each run
- [ ] Add merged-training row provenance for label origin and supervision mode:
  - distinguish `auto`, `manual_review`, `manual_training`,
    `interpolated/synthetic`, and task-specific supervision semantics where
    available
  - see [training_label_origin_provenance_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/training_label_origin_provenance_todo.md)

## Open decisions

- [ ] Define the exact finalization helper API for refreshing consolidated
  metadata after writer mutations.
  - Direct child `zarr.json` metadata remains required for mutable stores.
  - Consolidated metadata is a freshness/performance surface for finalized and
    transferred stores, not the only reader contract.
- [ ] Standardize `provenance` schema across all run types in `src/fisheye/shared/zarr/schema.py`.
- [ ] Decide which environment details are required vs optional.

## Deferred (online stage)

- [ ] Migrate `refine_online_detect` to shared stage helpers.
  - `src/fisheye/refinement/refine_online_detect.py` currently writes an ad-hoc
    provenance payload (no stage contract block).
  - This migration is intentionally deferred until offline dataset provenance
    standardization is complete.
- [ ] Add/extend tests in `tests/unit/fisheye/test_check_provenance_capture.py`
  for `stage=refine_online_detect` contract coverage.

## Related docs
- `src/fisheye/docs/provenance_workflow.md`
- `docs/keypoint_review_policy.md`
- `docs/archive/keypoint_review_status_notes.md`
- `docs/provenance_contract_draft.md`
- `docs/pipeline_metadata_boundaries.md`
- `docs/provenance_multi_agent_handoff.md`
- `docs/training_label_origin_provenance_todo.md`
