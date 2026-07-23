# Eye Masks Detect/Pose Parity TODO (Ordered by Impact)

Status (2026-02-25): completed historical plan for detect/pose parity surfaces.
For current eye-mask parity backlog covering profile/data-card/plot/pipeline/check surfaces, see `docs/eye_mask_parity_todo.md`.

Purpose: track the work that brought eye-mask workflow parity up to the detect/pose workflow standard at the time.

Scope: prioritize correctness and auditability first, then orchestration and registry ergonomics.

## P0: Correctness Blockers (Do First)

- [x] Write and adopt the eye-mask row-mapping contract.
  - Canonical source for `frame_indices`/`detection_indices`/`frame_counts` is
    the source crop run; keypoint arrays are validation/fallback only.
  - Covers both `eye_masks_runs` and `refined_eye_masks_runs`.
  - Contract doc: `docs/archive/eye_mask_row_mapping_contract.md`.

- [x] Canonicalize eye-mask provenance attribute names across all producers/consumers.
  - Today we mix `source_keypoint_run` and `source_keypoints_run` across traditional/YOLO/U-Net/refine paths.
  - Target: canonicalize to `source_keypoints_run`, dual-read during migration, and backfill legacy runs.
  - Scope: applies to both `_analysis.zarr` and curated/versioned `_training.zarr` archives.
  - Primary files:
    - `src/fisheye/segmentation/eye_segmentation.py`
    - `src/fisheye/segmentation/eye_segmentation_yolo.py`
    - `src/fisheye/segmentation/infer_unet_eye_masks.py`
    - `src/fisheye/refinement/refine_eye_masks.py`
    - `src/fisheye/diagnostics/check_eye_masks.py`
    - `src/fisheye/diagnostics/check_full_provenance.py`

- [x] Remove ambiguous keypoint source fallback in eye-mask refinement.
  - `refine_eye_masks` can fall back to latest keypoints if source attrs are missing/misaligned.
  - Target: fail fast unless refinement resolves an explicit, valid source keypoint run/group (or user passes override).
  - Primary file: `src/fisheye/refinement/refine_eye_masks.py`.

- [x] Enforce strict input alignment checks for eye-mask stages.
  - Require row-count compatibility for `roi_images`, `keypoints_roi`, `detection_source`, `masks_roi`, and optional `mask_probs_roi`.
  - Target: hard errors before write for mismatched inputs, not silent partial output.

## P1: Pipeline and Orchestration Parity

- [x] Add first-class `refined_eye_masks` stage wiring in pipeline orchestration.
  - Detect/pose have explicit refine stages in pipeline flow; eye-mask refine should be similarly discoverable and optional.
  - Add stage dependencies, stage completion checks, and configurable enable/disable flags.
  - Primary file: `src/fisheye/core/pipeline.py`.

- [x] Add eye-mask batch runner parity with detect/pose wrappers.
  - Standardize per-zarr result payloads (run name, method, source runs, duration, failure reason).
  - Emit JSONL suitable for reconciliation/reporting like detect/pose batch utilities.
  - Primary files:
    - `src/fisheye/utils/run_eye_masks_batch.py`
    - `tests/unit/fisheye/test_run_eye_masks_batch.py`

- [x] Add method-aware prerequisites for eye-mask stages.
  - Validate method-specific requirements up front (e.g., model path required for YOLO/U-Net).
  - Keep traditional path behavior unchanged.
  - Primary files:
    - `src/fisheye/utils/run_eye_masks_batch.py`
    - `tests/unit/fisheye/test_run_eye_masks_batch.py`

## P2: Registry and Query Parity

- [x] Add eye-mask performance registry table + latest views.
  - Track runtime/throughput and quality summary fields from `eye_masks_runs`/`refined_eye_masks_runs`.
  - Add rescan/backfill path to populate historical runs.
  - Primary files:
    - `src/fisheye/registry/db.py`
    - `src/fisheye/registry/maintenance.py`
    - `tests/unit/fisheye/test_registry_maintenance.py`

- [x] Add eye-mask review status registry fields and query filters.
  - Mirror detect/pose review ergonomics (`state`, `intended_use`, timestamps, reviewer, source linkage).
  - Extend `registry_query.py` with eye-mask quality/performance slices.
  - [x] Zarr-level review status contract is in place:
    - `refined_eye_masks_runs/<run>.attrs["eye_mask_review_status"]`
    - `refined_eye_masks_runs.attrs["eye_mask_review_status_latest"]`

- [x] Add stale/in-progress reconciliation for eye-mask review/refinement runs.
  - Align maintenance behavior with detect/pose failure recovery patterns.

## P3: Model Resolution and Provenance Parity

- [x] Add registry-resolved eye-mask inference wrapper(s) for model-based methods.
  - Detect/pose have registry model resolution wrappers; eye-mask YOLO/U-Net should have equivalent entrypoints.
  - Write `model_resolution_*` attrs and `provenance.model_resolution` on resulting runs.
  - Added: `src/fisheye/utils/run_eye_masks_with_registry_model.py`.

- [x] Standardize provenance payload shape with detect/pose conventions.
  - Align command, git, environment, platform, parameters, inputs, artifacts sections and key naming.
  - Draft contract: `docs/provenance_contract_draft.md`.
  - Added shared builder: `src/fisheye/utils/model_resolution_provenance.py`.
  - Adopted in wrappers:
    - `src/fisheye/utils/run_detect_with_registry_model.py`
    - `src/fisheye/utils/run_keypoints_with_registry_model.py`
    - `src/fisheye/utils/run_eye_masks_with_registry_model.py`

## P4: Tests, Docs, and Migration Hygiene

- [x] Add unit tests for provenance attr compatibility and strict source resolution.
  - Cover traditional, YOLO, U-Net, and refine flows.
  - Cover both legacy and canonical attr names during migration window.

- [x] Add diagnostics tests for `check_eye_masks` and full-provenance checks.
  - Ensure both tools agree on required attrs and resolution rules.

- [x] Update docs to match implemented canonical behavior.
  - `src/fisheye/docs/provenance_workflow.md`
  - `src/fisheye/docs/zarr_structure.md`
  - `src/fisheye/docs/refinement.md`

- [x] Add a backfill utility for legacy eye-mask attrs with dry-run support.
  - Default scope should include both `_analysis.zarr` and `_training.zarr` archives.
  - Include an opt-out flag when operators intentionally want analysis-only runs.

## Acceptance Criteria

- [x] Eye-mask runs and refined-eye-mask runs use a consistent, queryable provenance contract.
- [x] Eye-mask lineage attrs are present and queryable in both `_analysis.zarr` and `_training.zarr` archives.
- [x] Refinement never silently binds to the wrong keypoint run.
- [x] Pipeline/orchestration can run eye-mask segmentation and refinement with detect/pose-like ergonomics.
- [x] Registry query surfaces eye-mask quality/performance status alongside detect/pose.
- [x] Unit tests cover cross-method provenance and migration compatibility.

Validation evidence (2026-02-22/23, historical snapshot):
- `scripts/validate_recording_step_status_registry.sh` passed for a scoped real recording (`status=PASS`).
  - Artifact: `/tmp/recording_step_status_validate_20260222_214840`
- `scripts/validate_recording_step_status_registry_batch.sh --skip-backfill` passed across training recordings.
  - Artifact: `/tmp/recording_step_status_batch_20260222_215442`
  - Summary: `passed=52 failed=0 skipped_missing_zarr=1`
- `scripts/py -m fisheye.utils.backfill_eye_mask_lineage_attrs /nvme1/recordings --recursive --apply`
  confirmed canonical lineage attrs already present (`updated=0 already_canonical=102`).
