# Eye-Mask Parity TODO (Post Detect/Keypoint Parity)
_As of 2026-02-26_

## Purpose and Scope
- Align eye-mask training-data surfaces with the parity standard now used by detection and keypoint flows: profile registry, quality/performance summaries, data-card JSON, plot bundle, pipeline orchestration, maintenance, and query/check UX.
- This TODO now tracks active implementation progress; checkboxes are updated only from verified code/test evidence.
- Scope includes registry/profile/card/plot/pipeline/maintenance/query/check surfaces for eye-mask training datasets and merged exports.
- Scope excludes detect/pose parity items already completed in prior parity waves.

## Current State Snapshot (Reality Check)
- [x] Eye-mask performance registry tables/views exist (`eye_mask_performance`, `eye_mask_performance_latest`, recording-step status integration).
- [x] Maintenance CLI already supports eye-mask performance backfill/refresh flags.
- [x] Eye-mask batch/run and manual QA surfaces exist (`run_eye_masks_batch`, `inspect_refined_eye_masks`, `show_eye_mask_runs`, eye-mask viewers).
- [x] Dedicated eye-mask data-profile registry table/views now exist (`eye_mask_data_profile`, `eye_mask_data_profile_latest`, `recording_eye_mask_data_profile_latest`).
- [x] `scripts/py -m fisheye.utils.sync_eye_mask_profile_registry` command exists.
- [x] Eye-mask training data-card aggregation/plot modules now exist.
- [x] Eye-mask merged-export orchestration now wires profile sync + card/plot generation surfaces.
- [x] `check_training_registry` now has dedicated `eye-mask-performance` and `eye-mask-profile` views.

## Stale Items in Existing Eye-Mask TODO
- [x] Historical completion notes (2026-02-22/23) predate newer parity surfaces now required for detect/keypoint parity (data-profile registry parity, data-card schema parity, plot-bundle parity, query/check UX parity).
- [x] Historical evidence from that wave is treated as context only, not acceptance for the expanded parity scope in this document.
- [x] This document is the canonical tracker for current eye-mask parity scope.

## Prioritized Phases (P0-P6)
1. **P0 - Lock parity target + contracts**
- [x] Gather detect/keypoint parity reference docs and contracts.
- [x] Publish eye-mask parity target table mapping each required surface to an owner/module/CLI.
- [x] Define explicit out-of-scope items and defer list (if any).

P0 implementation map (owner/module/CLI):
- `EM-A` registry profile contract/storage/sync:
  - `docs/eye_mask_data_profile_schema_contract.md`
  - `src/fisheye/registry/db.py`
  - `scripts/py -m fisheye.utils.sync_eye_mask_profile_registry`
- `EM-B` data-card + plotting:
  - `docs/eye_mask_training_data_card_contract.md`
  - `scripts/py -m fisheye.utils.aggregate_eye_mask_training_data_card`
  - `scripts/py -m fisheye.utils.plot_eye_mask_training_data_card`
- `EM-C` export/pipeline orchestration:
  - `src/fisheye/utils/export_eye_mask_training_zarr.py`
  - `src/fisheye/core/pipeline.py`
- `EM-D` maintenance/backfill/refresh + runbook:
  - `src/fisheye/registry/maintenance.py`
  - `docs/eye_mask_profile_registry_ops_runbook.md`
- `EM-E` query/check UX:
  - `src/fisheye/utils/check_training_registry.py`
  - `src/fisheye/utils/registry_query.py`

Out of scope/deferred for this wave:
- broad non-eye-mask registry normalization work.
- historical detect/pose parity task rewrites from prior waves.

2. **P1 - Registry profile/quality/performance parity**
- [x] Add an eye-mask profile registry contract (schema name/version + canonical run attrs + profile JSON payload contract).
- [x] Add eye-mask profile registry storage (table + latest view + indexes) analogous to detect/keypoint profile registries.
- [x] Define required profile metrics parity set:
  - [x] Selection/source identity, lineage/provenance, and freshness fields.
  - [x] Quality/coverage metrics (usable/reviewed/excluded rates + reasons).
  - [x] Performance/geometry summaries (eye-mask-specific area/shape/coverage distributions and percentiles).
  - [x] Composition facets (genotype/age/dpf/context) needed for training-card aggregation.
- [x] Add fail-closed freshness semantics for missing/stale eye-mask profiles in training-card/pipeline consumers.

3. **P2 - Data-card schema + plot bundle parity**
- [x] Add `eye_mask_training_data_card` schema contract with required sections:
  - [x] `selection`
  - [x] `quality`
  - [x] `geometry` (eye-mask-specific metrics)
  - [x] `spatial`
  - [x] `composition`
  - [x] `subject_coverage`
  - [x] `parity` (train/val deltas)
  - [x] `audit_freshness`
- [x] Implement eye-mask data-card aggregation requirements (registry-first read path; explicit fallback policy if needed).
- [x] Define required default plot bundle parity (JSON + static plots saved together), including:
  - [x] Quality/usable-rate distributions.
  - [x] Eye-mask geometry distributions.
  - [x] Spatial coverage visualization.
  - [x] Composition summaries (e.g., genotype/DPF/context distributions).
  - [x] Train/val parity comparison visuals.

4. **P3 - Pipeline/orchestration/merged-export parity**
- [x] Wire eye-mask merged export flow to registry registration parity (dataset/run provenance recorded at export time).
- [x] Add pipeline flags matching detect/keypoint ergonomics for data-card generation and opt-out behavior.
- [x] Ensure merged-export workflows trigger eye-mask profile sync before aggregation (or fail with clear remediation).
- [x] Ensure produced card + plots are emitted to deterministic paths and referenced in registry metadata.

5. **P4 - Maintenance/backfill/refresh command parity**
- [x] Existing commands available today:
  - [x] `scripts/py -m fisheye.registry.maintenance --backfill-eye-mask-performance ...`
  - [x] `scripts/py -m fisheye.registry.maintenance --refresh-eye-mask-performance --eye-mask-performance-all-datasets ...`
- [x] Commands to add for parity:
  - [x] `scripts/py -m fisheye.utils.sync_eye_mask_profile_registry --registry ... --apply`
  - [x] `scripts/py -m fisheye.registry.maintenance --backfill-eye-mask-profiles ...`
  - [x] `scripts/py -m fisheye.registry.maintenance --refresh-eye-mask-profiles ...`
  - [x] `scripts/py -m fisheye.utils.aggregate_eye_mask_training_data_card --registry ... --training-set ...`
  - [x] `scripts/py -m fisheye.utils.plot_eye_mask_training_data_card --card-json ... --outdir ...`
- [x] Document operator runbooks for one-time backfill, routine refresh, and dry-run/apply expectations.

6. **P5 - Query/check registry UX parity**
- [x] Extend `check_training_registry --view` with dedicated eye-mask surfaces (quality/profile/performance summary parity).
- [x] Add detail-table options consistent with detect/keypoint quality/profile views (staleness, exclusion reasons, review-state rollups).
- [x] Ensure `registry_query` can expose eye-mask profile-linked outputs with parity filters and output formats.
- [x] Add actionable remediation messages for missing/stale eye-mask profile rows (pointing to sync/refresh commands).

7. **P6 - Validation checklist and acceptance criteria**
- [x] Unit/integration coverage proves schema contracts and registry projections are stable.
- [x] Backfill/sync/refresh commands are idempotent and report inserted/updated/unchanged counts.
- [x] At least one representative eye-mask training set passes end-to-end:
  - [x] merged export
  - [x] profile sync
  - [x] card aggregation
  - [x] plot generation
  - [x] check/query inspection
- [x] Failure-path validation confirms fail-closed behavior on stale/missing profiles with clear operator remediation.
- [x] Acceptance gate: eye-mask operator UX reaches detect/keypoint parity for registry, card, plot, pipeline, and maintenance surfaces.

Validation evidence (2026-02-25):
- `scripts/py -m py_compile $(git diff --name-only -- '*.py')` passed.
- `scripts/py -m pytest tests/unit/fisheye/test_aggregate_eye_mask_training_data_card.py tests/unit/fisheye/test_plot_eye_mask_training_data_card.py tests/unit/fisheye/test_check_training_registry.py tests/unit/fisheye/test_core_pipeline_refined_eye_masks_stage.py tests/unit/fisheye/test_registry_detection_data_profile.py tests/unit/fisheye/test_sync_eye_mask_profile_registry.py tests/unit/fisheye/test_registry_query.py -q` passed (`104 passed`).
- `scripts/py -m pytest tests/unit/fisheye/test_registry_maintenance.py -k "backfill_eye_mask_profiles or refresh_eye_mask_profiles or main_no_action_message_includes_profile_flags or main_backfill_eye_mask_profiles_wiring_and_summary or main_backfill_keypoint_profiles_wiring_and_summary or refresh_keypoint_profiles_deletes_stale_rows_and_is_deterministic" -q` passed (`6 passed`).
- Local validation completed (user-run) for zarr-heavy path:
  - `scripts/py -m pytest tests/unit/fisheye/test_validate_eye_mask_training_zarr.py -q` passed.
- Encoding contract parity addendum:
  - Eye merged export now writes reason labels via canonical reason codec (`reason_bytes` primary + `reason` mirror + fallback attrs).
  - Validation now checks reason-label decode via fallback order and asserts `reason_bytes` metadata consistency when present.
  - Cross-repo contract comparison and rationale recorded in `docs/zarr_string_encoding_todo.md` (2026-02-25 addendum).

Validation evidence (2026-02-26):
- Trainer lifecycle parity added for eye-mask U-Net:
  - `src/fisheye/segmentation/train_unet_eye_masks.py` now records registry run states (`in_progress`, `failed`, `success`) with invocation metadata and set linkage.
  - `src/fisheye/utils/run_eye_mask_training_pipeline.py` now passes `--manifest`, `--set-id`, and `--registry` to U-Net training stage for parity with detect/keypoint pipelines.
- Focused parity tests passed:
  - `scripts/py -m pytest tests/unit/fisheye/test_run_eye_mask_training_pipeline.py -q`
  - `scripts/py -m pytest tests/unit/fisheye/test_train_unet_eye_masks_registry.py -q`
- Representative operator workflow run completed (user-run):
  - merged export with row gating
  - eye-mask profile backfill/sync
  - data-card aggregation + plot generation/view
  - U-Net training runs (LR and union experiments)
  - registry training run registration and cleanup validation
- Workflow documentation now available in `docs/eye_mask_training_workflow.md`.

## Acceptance Definition for This TODO
- [x] All P1-P6 checkboxes are complete and validated.
- [x] Old eye-mask parity doc is either archived or clearly labeled as superseded for these surfaces.
- [x] Operators can run the full eye-mask parity workflow without bespoke/manual one-off scripts.
