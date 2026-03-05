# Detect-Keypoints Training Parity Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
-->

Purpose: define the minimum and target parity between detection and keypoints (pose) training workflows so curation, export, training, and auditability behave consistently.

Related contracts:
- keypoint training data-card schema/metrics:
  `docs/keypoint_training_data_card_contract.md`

## Decision

- Enforce strict parity for user-facing interfaces:
  - command naming,
  - core flags,
  - artifact naming/layout,
  - registry lifecycle semantics.
- Keep internals task-specific unless code paths are truly identical.

## Scope

- In scope:
  - dataset selection/preflight from registry,
  - merged training dataset export,
  - merged dataset validation,
  - training run lifecycle logging,
  - default model artifact output paths.
- Out of scope:
  - forcing identical label schemas between detect and pose,
  - over-abstracting loaders that have task-specific invariants.

## Canonical Command Parity

| Workflow step | Detect (current) | Keypoints/Pose (target) | Status |
| --- | --- | --- | --- |
| Registry preflight wrapper | `fisheye.utils.prepare_detect_training_from_registry` | `fisheye.utils.prepare_keypoint_training_from_registry` | present |
| One-command pipeline wrapper | `fisheye.utils.run_detect_training_pipeline` | `fisheye.utils.run_keypoint_training_pipeline` | present |
| Dataset exporter | `fisheye.utils.export_detect_training_zarr` | `fisheye.utils.export_keypoint_training_zarr` | present |
| Export validator | `fisheye.utils.validate_detect_training_zarr` | `fisheye.utils.validate_keypoint_training_zarr` | present |
| Trainer | `fisheye.training.train_detection` | `fisheye.training.train_keypoints` | present (partial parity) |

## Shared CLI Contract

All task prepare/export/train CLIs should support the same core auditability flags where applicable:

- `--registry`
- `--manifest`
- `--set-id`
- `--log-registry` / `--no-log-registry`
- `--project`
- `--run-name`

Prepare/export parity flags:

- `--set-name`, `--set-version`
- `--input-format`
- `--model-input` (must match training input modality)
- `--export-merged`
- `--merge-out-zarr`
- `--merge-out-dir`
- `--merge-split`
- `--merge-seed`
- `--merge-overwrite`

Task-specific extension flags remain allowed, for example:

- detect: `--source-type` (`manual|filtered|detect|interpolated`)
- pose: `--keypoint-run` selector semantics

## Artifact Contract

For both tasks:

- Generated training config:
  - `<dataset_root>/<set_id>.yaml`
- Generated manifest:
  - `<dataset_root>/<set_id>.manifest.json`
- Merged dataset:
  - `<dataset_root>/<set_id>_merged.zarr`
- Export summary:
  - `<dataset_root>/<set_id>_merged.summary.json`

Default model output roots:

- detect: `/nvme1/models/detect/<set_id>/<run_id>/...`
- pose: `/nvme1/models/pose/<set_id>/<run_id>/...`

## Run ID Naming Contract

Default `run_id` generation must be shared between detect and pose (implemented in `fisheye.training.training_naming_shared`).

Canonical default pattern:

- `<rig>_<dish>_<canvas>_<version>_<task>_<timestamp>_<hash8>`

Field rules:

- `rig`: manifest/query hint (`rig_id`) when available, else `unknown_rig`.
- `dish`: manifest/query hint (`dish_design`) when available, else `unknown_dish`.
- `canvas`: manifest/provenance hint when available, else `unknown_canvas`.
- `version`: derived from `set_id`/set slug suffix (`_v###`), else `v001`.
- `task`: `detect` or `pose`.
- `timestamp`: `YYYYMMDD-HHMMSS`.
- `hash8`: first 8 chars of `manifest_sha256` when available; otherwise deterministic fallback hash.

Examples:

- detect: `omnifin0_cedar_shadow_defaultscreen_v007_detect_20260206-235656_25f3fbcb`
- pose: `omnifin0_cedar_shadow_defaultscreen_v007_pose_20260208-030800_505915ab`

Notes:

- `--run-name` still overrides default generation.
- Existing historical run IDs remain valid; this contract applies to new default-generated runs.

## Registry Lifecycle Contract

`training_runs.status` semantics must match between tasks:

- `in_progress`: training launched and not yet terminal.
- `success`: trainer exited successfully and final artifacts/metrics were written.
- `failed`: trainer terminated with error; include failure stage and error metadata.

Required behavior:

- write `in_progress` before long-running training work,
- write terminal status (`success` or `failed`) exactly once per `run_id`,
- include `manifest_path`, `config_path`, and hashes when available.

## Pose Data-Linkage Contract

For pose training correctness:

- Selected keypoint run must resolve to a concrete `source_crop_run`.
- Pose ROI image source must come from that exact `crop_runs/<source_crop_run>/roi_images`.
- Never silently fallback to `crop_runs/latest` when a specific keypoint run is selected.
- If linkage is missing or inconsistent, fail fast with explicit error.

## Query Filter Contract

Persist task-specific query context in `training_sets.query_filter`:

- common:
  - `tool`, `task`, `input_format`, `model_input`,
  - selection filters and selected dataset paths.
- detect:
  - `source_type`, provenance policy fields.
- pose:
  - `keypoint_run` resolution strategy,
  - keypoint method/provider constraints (if specified).

## Implementation Phases

### Phase 1: Interface and Logging Parity

- [x] Keypoints trainer supports registry lifecycle status writes.
- [x] Keypoints trainer supports default `/nvme1/models/pose/<set_slug>` project root.
- [x] `keypoint_run` passthrough in pose config plumbing.
- [x] Add keypoints registry preflight wrapper (`prepare_keypoint_training_from_registry`).
- [x] Add keypoints one-command pipeline wrapper (`run_keypoint_training_pipeline`).

### Phase 2: Data Build Parity

- [x] Add keypoints merged exporter (`export_keypoint_training_zarr`).
- [x] Add keypoints merged validator (`validate_keypoint_training_zarr`).
- [x] Ensure pose exporter writes split arrays and source traceability similar to detect.

### Phase 3: Loader Correctness and Guardrails

- [ ] Enforce keypoint run -> source crop run linkage in pose loader.
- [ ] Remove/forbid ambiguous `crop_runs/latest` fallback for pose when a run is selected.
- [ ] Add drift checks between persisted summary metadata and derived counts.

## Acceptance Checklist

- [ ] A new user can run detect and pose workflows using parallel command patterns.
- [ ] Both tasks emit auditable config/manifest/export artifacts with consistent naming.
- [ ] Registry status can be used to monitor long runs without log tailing.
- [ ] Pose training cannot mix keypoints from one run with ROI images from another crop run.
- [ ] Validation CLIs fail on malformed merged datasets before training starts.
