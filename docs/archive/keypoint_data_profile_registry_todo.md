# Keypoint Data Profile Registry TODO

Purpose: add keypoint profile-registry parity with the detect profile-registry
surface so keypoint dataset/profile queries and training data-card aggregation
can use canonical SQL projections instead of rescanning Zarr stores.

## Problem Statement

Detect has a complete profile-registry path:
- profile payloads in Zarr profile runs,
- registry table + latest views,
- sync utility,
- `registry_query` surface.

Keypoint currently lacks the equivalent profile-registry projection layer.

## Policy Decisions

- [x] Keep profile rows separate from quality-gate rows.
  - `keypoint_quality` remains the gating table.
  - `keypoint_data_profile` stores distribution/profile summaries.

- [x] Keep fail-closed freshness semantics aligned with detect.
  - Store `zarr_mtime_ns` on profile rows.
  - Reject stale selected rows at build time when profile-based checks are required.
  - Status (2026-02-24): keypoint data-card aggregation now fails closed on stale
    `keypoint_data_profile_latest` rows unless explicitly overridden.

- [x] Use canonical `dataset_id` linkage everywhere.
  - Resolve to canonical registry IDs in profile rows/views.
  - Avoid recording-id-only linkage in profile projections.

- [x] Partition latest views by keypoint method where needed.
  - Keypoint can have multiple methods per dataset (`traditional_pose`, `yolo_pose`).
  - Latest views should not collapse method-distinct profiles.

## Priority 0 (Schema + Read Path)

- [x] Define keypoint profile schema contract (`v1`) for:
  - on-disk profile payload (`analysis/keypoint_profile_runs/<run>/attrs["profile_summary"]`)
  - registry projection columns
  - training data-card aggregate linkage fields
  - relation to existing card contract:
    `docs/keypoint_training_data_card_contract.md`
  - contract doc: `docs/keypoint_data_profile_schema_contract.md`

- [x] Add registry table: `keypoint_data_profile`.
  - Proposed key:
    - `PRIMARY KEY (dataset_id, profile_run)`
  - Proposed identity/context columns:
    - `dataset_id`, `profile_run`, `recording_id`, `zarr_use`
    - `keypoint_method`, `source_keypoint_path`, `source_keypoint_run`
    - `skeleton_id`, `kpt_shape`, `profile_created_utc`, `updated_utc`, `zarr_mtime_ns`
  - Proposed quality/coverage summary columns:
    - `rows_total`, `rows_usable`, `usable_keypoints_total`
    - `usable_rate`, `confidence_valid_rate`, `geometry_valid_rate`
  - Proposed geometry summary columns:
    - `triangle_area_p10/p50/p90`
    - `min_angle_p10/p50/p90`
    - `heading_p10/p50/p90`
  - Proposed composition/lineage columns:
    - `rig_id`, `camera_id`, `arena_id`, `dish_design`, `canvas_name`, `protocol_name`
    - `genotype`, `dpf_at_acquisition`
  - Opaque payload:
    - `profile_json`

- [x] Add latest views:
  - `keypoint_data_profile_latest` (latest per `dataset_id` + `keypoint_method`)
  - `recording_keypoint_data_profile_latest` (latest per `recording_id` + `keypoint_method`)

- [x] Add indexes for query/filter ergonomics:
  - `dataset_id`
  - `(keypoint_method, usable_rate)`
  - `(review_state, review_intended_use)` if projected into this table
  - `(genotype, dpf_at_acquisition)` for lineage filtering

## Priority 1 (Write Path + Backfill)

- [x] Add keypoint profile writer/backfill utility.
  - Suggested module:
    `scripts/py -m fisheye.utils.backfill_keypoint_profiles`
  - Write canonical profile payloads into `analysis/keypoint_profile_runs`.
  - Status (2026-02-24): implemented with dry-run/apply and registry identity enrichment.

- [x] Add registry sync utility.
  - Suggested module:
    `scripts/py -m fisheye.utils.sync_keypoint_profile_registry`
  - Read latest profile run per dataset/method and upsert into registry projection.
  - Support `--dry-run` and `--apply`.
  - Status (2026-02-24): implemented with robust group lookup fallback and canonical row upsert.

- [x] Add maintenance integration for parity with detect quality/profile workflows.
  - Suggested maintenance flags:
    - `--backfill-keypoint-profiles`
    - `--refresh-keypoint-profiles`
  - Print deterministic inserted/updated/unchanged/missing/error counts.
  - Status (2026-02-24): implemented in `fisheye.registry.maintenance`.

- [x] Run one-time production backfill + sync.
  - Capture observed counts and exceptions in this doc once executed.
  - Status (2026-02-24):
    - Backfill dry-run:
      - `zarr_scanned=105`, `would_write=104`, `missing_source=1`, `errors=0`
    - Backfill apply:
      - `zarr_scanned=105`, `updated=104`, `missing_source=1`, `errors=0`
    - Sync dry-run:
      - `datasets=54`, `would_upsert=52`, `missing_profile=2`, `errors=0`
    - Sync apply:
      - `datasets=54`, `updated=52`, `missing_profile=2`, `errors=0`
    - Registry verification:
      - `registry_query --keypoint-data-profile-latest | jq 'length'` -> `52`
    - Maintenance refresh check:
      - `--refresh-keypoint-profiles` -> `scanned=52`, `no_profile=1`, `deleted=1`, `unchanged=51`
    - Notes:
      - `missing_profile=2` corresponded to detect merged training artifacts without keypoint profile groups.
      - `missing_source=1` was a non-standard camera zarr lacking `keypoints_runs`.

## Priority 2 (Query + Registry UX)

- [x] Extend `registry_query` with keypoint profile surfaces.
  - Add:
    - `--keypoint-data-profile-latest`
    - `--recording-keypoint-data-profile-latest`
  - Add profile filters analogous to detect profile query mode:
    - method, coverage/usable thresholds, skeleton filters, lineage filters.

- [x] Extend `check_training_registry` profile visibility.
  - Add keypoint profile view/report:
    - total rows
    - stale rows
    - missing profile rows for selected datasets
    - method/skeleton composition summary

## Priority 3 (Build/Data-Card Integration)

- [x] Use keypoint profile-registry rows as first-class input for keypoint data-card aggregation.
  - Prefer registry projection rows over direct Zarr scanning when available.
  - Keep explicit fallback behavior documented.
  - Status (2026-02-24): aggregator now selects `keypoint_data_profile_latest`
    rows first; direct Zarr scan fallback requires explicit
    `--allow-profile-fallback-scan`.

- [x] Enforce fail-closed profile freshness in keypoint build pipeline where profile rows are required.
  - stale `zarr_mtime_ns` -> hard failure with actionable remediation message.
  - Status (2026-02-24): stale profile rows hard-fail aggregation by default;
    override only via `--allow-profile-mtime-mismatch`.

- [x] Add explicit operator remediation flow:
  - refresh/sync profile rows,
  - rerun validation/check view,
  - rerun pipeline.
  - Status (2026-02-24): remediation commands are embedded in fail-closed
    error messages and documented in workflow.

## Validation Checklist

- [x] Schema migration creates table/views/indexes successfully.
- [x] Backfill/sync commands are idempotent and deterministic.
- [x] `registry_query` keypoint-profile modes return expected row counts.
- [x] `check_training_registry` keypoint profile view explains stale/missing rows.
- [x] Keypoint data-card aggregation can run using registry profile rows end-to-end.
- [x] Build/pipeline fails closed on stale profile rows with clear recovery guidance.
