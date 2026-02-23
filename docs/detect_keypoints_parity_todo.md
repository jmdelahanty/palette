# Detect-Keypoints Parity TODO (Ordered by Impact)

Purpose: track the remaining work to bring keypoints/pose workflow parity up to the current detection workflow standard.

Scope: prioritize correctness and auditability first, then UX/orchestration, then export/deployment parity.

## P0: Correctness Blockers (Do First)

- [x] Remove ambiguous keypoint source fallback.
  - Today, keypoint prepare can fall back to `crop_runs/latest` when `source_crop_run` is missing.
  - Target: fail fast unless an explicit, valid `source_crop_run` is resolved.
  - Primary file: `src/fisheye/utils/prepare_keypoint_training_from_registry.py`.
  - Status: implemented; preflight now errors when `source_crop_run` is missing.

- [x] Enforce strict row-alignment invariants for pose inputs.
  - Require consistent row counts across `roi_images`, `keypoints_roi`, and `detection_success` (when present).
  - Target: hard error on mismatch in preflight/export paths, not runtime surprises.
  - Status: implemented with hard errors in keypoint preflight.

- [x] Add/confirm pre-train validation gate for keypoints merged data.
  - Detection has `validate_detect_training_zarr`; keypoints needs an equivalent hard gate before training.

## P1: Dataset Build Parity (Highest Product Impact)

- [x] Implement `fisheye.utils.export_keypoint_training_zarr` (merged mode first).
  - Include split arrays (`train/val/test`).
  - Include source traceability arrays (`source_dataset_idx`, `source_frame_idx`, source path map).
  - Emit summary JSON next to merged artifact.

- [x] Implement `fisheye.utils.validate_keypoint_training_zarr`.
  - Validate shapes, split coverage, index bounds, and source-index consistency.
  - Return non-zero on invalid dataset.

- [x] Ensure keypoint merged export supports registry linkage like detect merged export.
  - Register merged dataset row.
  - Link merged dataset to training set dataset_ids where applicable.

## P2: Orchestration Parity (One-Command UX)

- [x] Add `fisheye.utils.run_keypoint_training_pipeline`.
  - Query registry -> preflight -> optional merged export -> optional train.
  - Match detect wrapper behavior for `--dry-run`, `--set-name`, `--set-version`, output path defaults.

- [x] Require `set_id` for `--train` path in keypoint orchestration.
  - Prevent unlinked training runs.
  - Use manifest `set_id` contract consistently.

- [x] Align keypoint wrapper flags with detect wrapper where meaningful.
  - Common: `--registry`, `--set-name`, `--set-version`, `--run-name`, `--project`, `--train`, `--register`.
  - Keypoint-specific selectors remain task-specific (for example `--keypoint-run`).
  - Status: implemented; wrappers share core orchestration surface, with task-specific flags retained intentionally.

## Current Command-Surface Differences (Detect vs Keypoint)

- Shared core surface:
  - `--registry`, dataset filters (`--dish-design`, camera/codec/exposure filters), `--source-type`, `--input-format`, `--model-input`
  - `--set-name`, `--set-version`, `--out-config`, `--out-manifest`, `--project`, `--run-name`
  - `--register`, `--register-registry`, `--export-merged`, `--merge-split`, `--merge-seed`, `--merge-overwrite`
  - `--train`, `--dry-run`

- Detect-specific surface:
  - provenance and source controls: `--provenance-policy`, `--metadata-json`, `--allow-source-mismatch`, `--allow-unapproved`, `--no-prefer-manual`
  - export/deploy controls: `--export-onnx`, `--export-trt`, `--onnx-opset`, `--onnx-simplify`, `--onnx-path`, `--nms-conf`, `--nms-iou`, `--nms-topk`, `--trt-precision`, `--trtexec`, `--trt-cuda-graph`, `--trt-profiling`, `--trt-verbose`
  - training profiling: `--profile`

- Keypoint-specific surface:
  - keypoint selector and quality/review gates: `--keypoint-run`, `--min-usable-keypoints-rate`, `--require-review-state`, `--require-review-intended-use`
  - optional relaxation: `--allow-cross-method-review-fallback`

- Intended parity target:
  - Keep task-specific quality controls distinct.
  - Add run-aware keypoint ONNX/TRT export and matching registry writes.
  - Keep shared orchestration flags semantically aligned across detect and keypoint.

## P3: Export and Deployment Parity

- [x] Decide and implement keypoint export pathway (ONNX/TRT) with same audit semantics as detect.
  - Run-aware export entrypoint.
  - Registry writes with build metadata and artifact hashes.
  - Manifested model I/O contract and plugin requirements (if any).
  - Status: implemented in `train_pose` + shared export registry writers; wrapper supports `--export-onnx`/`--export-trt`.

- [x] Define pose-specific model table strategy in registry.
  - Reuse existing model tables if schema is task-agnostic enough, or add pose-specific model tables.
  - Keep one clear canonical read path for registry views.
  - Status: resolved by reusing shared tables (`training_runs`/`training_models`/`onnx_models`/`tensorrt_models`) with pose skeleton/task metadata.

## P4: Registry and Monitoring Parity

- [x] Implement SQL-level keypoint quality gating in registry and use it in keypoint selection.
  - Track reviewed refined quality (`review_state`, `review_intended_use`, `usable_keypoints_rate`) as first-class registry fields.
  - Keep build-time fail-closed checks for stale/divergent records.
  - Detailed plan: `docs/keypoint_quality_registry_todo.md`.
  - Status: implemented and completed per `docs/keypoint_quality_registry_todo.md`.

- [x] Extend `check_training_registry` views to cleanly surface pose set/run/export status.
  - Keep set view focused on set artifacts and latest run summary.
  - Keep run/model/export details in run/model-specific views.
  - Status: models/onnx/tensorrt/keypoint-quality views surface pose run/export artifacts and statuses.

- [x] Add stale `in_progress` reconciliation for pose runs (same policy as detect).
  - Implemented maintenance command support in `fisheye.registry.maintenance`:
    `--reconcile-in-progress-runs` with `--in-progress-task` and `--in-progress-max-age-hours`.

## P5: Docs and Migration Hygiene

- [x] Update stale status in `docs/detect_keypoints_parity_contract.md`.
  - The keypoint preflight wrapper already exists; document should reflect current reality.

- [x] Add a keypoint workflow doc mirroring detect workflow structure.
  - Registry query examples.
  - Prepare-only flow.
  - One-command pipeline flow once implemented.
  - Status: documented in `docs/keypoint_training_workflow.md`.

- [x] Add backfill guidance for any new registry tables/fields.
  - Dry-run first.
  - Integrity-check command sequence.
  - Status: documented in keypoint quality registry workflow and maintenance runbooks.

## Acceptance Criteria

- [x] Keypoint training cannot silently mix keypoints with ROI images from a different crop source.
- [x] A keypoint merged dataset can be exported, validated, and trained from a single wrapper command.
- [x] Registry clearly shows pose set status, run status, and model/export artifacts.
- [x] Failure states are terminal and auditable (`failed` with stage/error metadata).
- [x] Docs reflect real command behavior and defaults.
  - Status: workflow/default behavior captured in `docs/keypoint_training_workflow.md` and `docs/training_data_workflow.md`.
