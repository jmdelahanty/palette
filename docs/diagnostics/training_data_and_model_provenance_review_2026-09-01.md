# Training Data and Model Provenance Review — 2026-09-01

> Disposition added 2026-09-06: historical review; checklist proposals require
> reconciliation with the [second opinion](review_wave_second_opinion_2026-09-04.md). Existing reviewed-keypoint compaction,
> persisted splits, pose seed forwarding, and copied-export verification must
> be preserved. Recording/model overlap is not exact-example overlap or proof
> of downstream evaluation leakage; unknown membership is not a clean result.
> See the [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation); no training or live-store measurements were rerun here.

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-01
implementation: specified-only
-->

**Date:** 2026-09-01
**Snapshot:** `dcdcc081` on branch `agent/palette/refined-assignment-rebinding-gaze-20260831`
**Method:** two parallel read-only review agents (training dataset structures and exports; model training workloads and model provenance) plus a synthesis pass. No tests run, no zarrs opened; disk listing only where paths were obvious from config. Findings marked **[VERIFIED]** were re-read by the synthesizer at the cited line. All other line numbers are agent code reading at this snapshot and **must be re-verified before implementation**.
**Question asked:** can an auditor go from a model used in inference back to the exact human-accepted labeled frames it was trained on, and is training reproducible and contamination-free?
**Companion evidence:** [`provenance_chain_review_2026-09-01.md`](provenance_chain_review_2026-09-01.md) (the inference-side chain this review joins at the model sha256); [`architecture_review_five_lens_2026-09-01.md`](architecture_review_five_lens_2026-09-01.md); `docs/training_quality_gate_contract.md`; `docs/analysis_to_training_promotion_contract.md`; `docs/training_dataset_versioning_todo.md`.

**Queue disposition:** audit evidence and checklist source only. Items here should be adopted into the authority consolidation queue or the wave-1 brief's successor. This document does not track status.

---

## 1. Verdict

**Inference-side model pinning is strict and correct. Training-side provenance is not, and nothing prevents evaluating a model on the data it was trained on.**

The strong parts:

- Every modality verifies `model_sha256` before load and refuses on mismatch. The UNet path checks before and after load, comparing mtime and size as well (`inference/infer_unet_subject_masks.py:3446-3475`). This is the best-practice pattern.
- Pose training writes an `effective_arguments` receipt verified against the trainer's own args (`training/train_pose.py:2054-2060`).
- Keypoint v3 exports use leakage-grouped splits (subject → acquisition cohort → recording), validate group disjointness across splits, and publish with a hash-content inventory under atomic rename.
- Detect exports bind a review decision digest and re-verify the plan before copying.
- TensorRT engine manifests record source ONNX sha, GPU name, UUID, compute capability, and TRT version.

The defects, in order of consequence:

1. **Train/eval contamination is unguarded and the selector favors it.** Model resolution scores candidates by feature similarity between the target recording and each training set's source rows (`registry/model_resolution.py:15-25, 366-376`) **[VERIFIED]**. It never checks whether the target recording is *in* the set. A model trained on recording X scores maximally for inference on X. No `used_in_training` flag exists on recordings, and evaluation and analytics exporters do not exclude training recordings.
2. **Human acceptance is not bound into most exports.** Keypoint exporter never reads review state (row gate counts are quality gates, not human decisions). Subject-mask export can run with `--allow-unapproved-refined` and leaves no attr saying so (`utils/export_subject_mask_training_zarr.py:416-446, 1528`). Only detect (decision digest) and the compose-v3 keypoint path (`reviewed_artifact_receipt_digest`) bind acceptance.
3. **Training registry write is fail-open.** `training/training_run_shared.py:121-122` catches every exception and prints "Registry update skipped" **[VERIFIED]**. A training run can exist on disk with no row, no model sha, and no error. Same pattern as `stage_complete.py:488`.
4. **Base weights are never hashed and lineage does not exist.** `YOLO("yolov8n.pt")` is a string (`train_detection.py:1831`, `train_pose.py:1972`). No `base_weights_sha256`, no `parent_run_id`. A from-scratch run and a fine-tune from a prior `best.pt` are indistinguishable in the registry.
5. **Realized splits are not persisted for detect and pose.** The 80/20 seeded split happens at load time (`training/config.py:20-31`); train and val index sets are never written. Ultralytics `seed` and `deterministic` are not passed or recorded; cudnn flags are not captured. A retrain is comparable only when the zarr already ships `splits/`.
6. **The "training quality gate" is a data-selection gate, not a model-quality gate.** No mAP, PCK, or IoU threshold blocks registration or promotion. Metrics are coupled to the dataset only by sharing a registry row.
7. **Merged training zarrs have no content digest.** `training_runs.manifest_sha256` hashes the manifest, not the rows or pixels. If a set is re-exported with the same manifest, the trained-on bytes are unprovable. Keypoint v3 is the exception.
8. **Training launches are manual.** No `scripts/submit_*train*_bsub.sh` exists for any modality **[VERIFIED: `ls scripts | grep -i train` shows import, review-bootstrap, and analytics only]**. Training runs `subprocess.run` inline from `utils/run_*_training_pipeline.py`. Nothing records the LSF job, host GPU, or commit pin the way inference jobs do.

---

## 2. Dataset pipeline: label → export → trainer

| Artifact | Schema / version | Source labels identified by | Review state bound? | Splits | Own digest | Leakage guard | Commit + params |
|---|---|---|---|---|---|---|---|
| Detect frame decisions (review run) | `palette.detect_frame_decisions` v1 (`shared/zarr/detect_frame_decisions.py:32-40`) | decision codes per acquisition frame; positives implicit via bbox rows | yes: `decision_digest` over codes+reasons+frame index (`detection_frame_supervision.py:118-140`) | n/a | yes | n/a | no |
| Keypoint review artifact | `palette.training_keypoint_review_artifact` receipt; instance-key-bound delta generation (`training/training_review_artifact_publication.py:81-90, 497-512`) | `instance_key_digest`, `review_qc_policy_digest`, skeleton digest | at artifact level via receipt `payload_digest`; per-instance state lives in delta generation | n/a | yes | n/a | no |
| Promotion analysis → training zarr | contract v1 draft (`docs/analysis_to_training_promotion_contract.md:3-9`) | parent-frame index | **no**; upsert of boxes, review state of the edit not carried | n/a | `write_promotion_result` only | n/a | no |
| Prepare-from-registry manifests (3 tools) | ad-hoc JSON, no `schema_id` | registry `datasets` rows + run names; gate on `review_status.state=="approved"` and `intended_use=="training"` (`prepare_detect_training_from_registry.py:570-572`; `prepare_keypoint_training_from_registry.py:1015-1075`), both with `--allow-unapproved` | as strings (`:1503-1504`), not receipt digest, except compose-v3 (`compose_task_keypoint_training_manifest.py:161, 445`) | not here | manifest sha taken by exporter | keypoint only: `leakage_group` from `shared/training_leakage_groups.py` | no |
| Detect merged zarr | `training_export.schema_version="2.0.0"` (`export_detect_training_zarr.py:2195-2260`) | `source_index/{source_dataset_idx, source_frame_idx, source_refined_row_ids, source_detect_row_index}`; `source_frame_decisions[].digest` | yes via decision digest; plan re-verified before copy (`:1870-1876`); refuses interpolated rows (`:1055-1075`) | `global_random`, **row-level**, seed 42 (`:2251-2258`); `splits/{train,val,test}_indices` | manifest sha only; no zarr content digest | **none**: frames of one recording land in train and val | invocation only, no commit |
| Keypoint merged zarr (v2 legacy / v3 immutable) | `"2.0.0"` (`export_keypoint_training_zarr.py:2657`); v3 adds `palette.immutable_merged_keypoint_training_publication` v2 (`:2955`) | v3: `source_sample_row_index`, `source_acquisition_frame_index`, `source_refined_row_ids` (`:3560-3575`) | indirect only; exporter never reads review state | `split_unit` default `"row"` (`:1906`); v3 requires `leakage_group`, validator rejects shared groups (`:3676-3691`) | v3: `tree_inventory(hash_content=True)` (`:2985`) | **v3 only** | no commit |
| Subject-mask merged zarr | `training_export` with **no schema_version** (`export_subject_mask_training_zarr.py:1071-1104`); dense uint8 `masks_roi` enforced per AGENTS.md | `source_index/{source_frame_idx, source_roi_idx, source_run_name, source_crop_run}` (`:879-887`) | component `approved` gate, bypassable with `--allow-unapproved-refined`; **bypass not recorded** (`:416-446, 1528`) | `global_random`, seed 123 (`:705`) | none | **none** | no commit, no invocation |
| Acquisition-crop pose export | `palette.acquisition_crop_pose_training_export.v1` | crop_meta + video frames | none (raw crops) | none | schema only | none | no |
| Registry | `training_sets(set_id, query_filter, dataset_ids_json, invocation_json)`, `training_runs(manifest_sha256, config_sha256, model_sha256, final_metrics_json, ...)` (`registry/migration_bodies.py:537-585`) | dataset ids, not rows | no | not stored | manifest sha | no | trainers record git via `get_git_info` |
| Trainers | YOLO loader reads `splits/` only when `zarr_purpose=="training"`, else seeded 80/20 at load (`zarr_yolo_dataset_loader.py:446-455`); UNet requires `splits/` and validates disjointness (`zarr_subject_mask_dataset.py:74-104`) | `training_export.source_zarr_paths` | no | consumes stored splits; `test_indices` ignored by YOLO path | manifest sha verified vs registry (`training_run_shared.py:72-78`) | no | yes |

**Twins and superseded paths:** detect per-dataset copy vs `--merge`; keypoint v2 vs v3 both emitted from one 4,974-line module; `train_keypoints.py` is a star-import alias of `train_pose.py`; `run_pose_training_pipeline.py` aliases `run_keypoint_training_pipeline.py`; `prepare_pose_training_from_registry` and `prepare_keypoint_training_from_registry` both exist; acquisition-crop pose is a fourth pose export; eye-mask training sets still on disk.

**On disk (listing only):** `/nvme1/training/datasets/` holds ~25 sets of 1.2–3.1 GB from Feb–Aug 2026, including detect v001–v004 as near-duplicates and three eye-mask sets. Three keypoint v3 sets of 1.6 GB under `/groups/.../.palette_benchmarks/training/keypoint_merged_v3/`. Nothing indexes which are superseded; `training_dataset_versioning_todo.md` P0 lineage fields are unchecked.

---

## 3. Model training and provenance

| | Detect (YOLO) | Pose (YOLO-pose) | Subject-mask UNet |
|---|---|---|---|
| Launch | manual, inline `subprocess.run` (`utils/run_detect_training_pipeline.py:163`); no bsub submitter | same via `run_keypoint_training_pipeline.py` | `run_subject_mask_training_pipeline.py`, manual |
| Manifest written | `<ts>_detection_training_report.yaml` in ultralytics run dir (`train_detection.py:2021-2046`); `inputs/{config,manifest,train_invocation.json}` (`training_run_shared.py:29-54`); registry `training_runs` + `training_models` | `<ts>_pose_training_report.yaml` + `pose_runtime_receipt` (`train_pose.py:2160-2200`) | `training_summary.json`, `training_history.json`, `dataset_metadata.json` (`train_unet_subject_masks.py:1155-1160`) |
| Dataset digest | manifest sha only | manifest sha; compose-v3 adds `composition_digest`, `reviewed_artifact_receipt_digest` | path only; no digests in `dataset_metadata.json` |
| Hyperparameters | `effective_training_params` = what palette passed, not ultralytics-merged defaults (`:2027`); ultralytics `args.yaml` is the de-facto record | `effective_arguments` verified vs `trainer.args` (`:2054-2060`) | full config dump |
| Seed / determinism | `random_seed` for sampler and val workers (`:1721-1727`); **not passed as ultralytics `seed=`**; `deterministic` popped (`train_pose.py:1673`); cudnn not recorded | same | `torch.manual_seed` (`:808-813`); `cudnn.benchmark=True` |
| Base weights sha | **no** (`:1831`) | **no** (`:1972`) | n/a, from scratch |
| Commit + dirty | commit + branch in report; dirty only in `invocation_json.git` | same | `git_commit` in summary; dirty in invocation |
| Environment | torch / ultralytics / python; cuda only as `cuda_available` | + torch/cuda in export `build_env` | `get_environment_info()` |
| Hardware | hostname only | same | hostname only |
| Checkpoint sha + registry row | yes, `model_sha256` → `training_models` (`db.py:6901`) | yes | yes |
| Pinning at inference | registry → `model_sha256` verified pre-load, mismatch raises (`detect_yolo.py:216-247`); batch refuses missing identity (`run_detections_batch.py:836-842`) | + pose-schema binding requires digest (`detect_keypoints_yolo.py:1940-1948`) | verified before and after load, mtime/size compared (`:3446-3475`) |
| Model-quality gate | none; contract is data selection | none | none |
| TRT engine binding | ONNX manifest `weights.sha256`; engine manifest `onnx.sha256` + `build_env` (`export_shared.py:150-170`); `tensorrt_models` has gpu_name/uuid/CC | same | no export path |
| Metrics ↔ dataset | `final_metrics_json` in same row as `manifest_sha256`; val identity = seeded split unless zarr has `splits/` | same | `splits/` required; best of the three |

**Schema count:** six registry tables (`training_runs`, `training_models`, `onnx_models`, `tensorrt_models`, `model_deployment_artifacts`, legacy `model_exports`) plus five JSON/YAML shapes plus the `model_resolution` payload. Git field names drift (`git_commit`, `git_commit_hash`, `git.commit`); `_normalize_git_payload` exists to paper over it (`model_resolution_provenance.py:28-40`).

**Audit chain today:** detect run attrs → `model_sha256` + `model_resolution_selected_run_id` → `training_runs.run_id` → `manifest_sha256` → manifest `dataset_ids` → source zarrs. Works to the manifest. Breaks at: no content digest of the merged zarr; registry row may be missing (fail-open write); `register_training_run.py` backfill accepts any set id and manifest on faith; review acceptance unbound except detect and compose-v3.

---

## 4. Checklists

### 4.1 Contamination and evaluation identity

- [ ] **T-1** Add a registry view `recording_training_membership(recording_id, set_id, run_id)` derived from `training_sets.dataset_ids_json` → `datasets` → recordings. One query, no schema change.
- [ ] **T-2** `model_resolution` excludes, or at minimum flags with `contaminated=true` in the resolution payload, any candidate whose training set contains the target recording. Record the decision in `detection_model_provenance` attrs.
- [ ] **T-3** Analytics and evaluation exporters emit `training_membership` per source recording and refuse, or flag, recordings that trained the model used to produce their detections.
- [ ] **T-4** Persist realized `splits/{train,val,test}_indices` for detect and pose training when the zarr lacks them, plus a `splits_sha256` into `final_metrics_json`, mirroring the UNet contract.
- [ ] **T-5** Make leakage-grouped splitting the only split unit for detect and subject-mask exports; record `split_unit` and group ids in `source_index`. The helper exists in `shared/training_leakage_groups.py`.
- [ ] **T-6** Define a model-quality gate: minimum val metric on a leakage-clean held-out set, recorded with the split digest, required before a model becomes selector-eligible. Reconcile with `docs/training_quality_gate_contract.md`, which currently names a data-selection gate.

### 4.2 Review acceptance binding

- [ ] **R-1** Every exporter writes `source_review_receipts[]` = `{dataset_id, run, receipt_schema_id, payload_digest, state, intended_use, bypass_flag}` into `training_export` attrs and the manifest. Detect already has the digest; keypoint reads what prepare read; subject-mask records `--allow-unapproved-refined` when used.
- [ ] **R-2** `validate_*_training_zarr` fails when `source_review_receipts` is absent for post-cutover exports.
- [ ] **R-3** Promotion (`analysis_to_training_promotion_contract.md`) carries the review state of the source edit into the promoted rows.
- [ ] **R-4** `prepare_*_from_registry` manifests gain a `schema_id` and record receipt digests, not state strings.

### 4.3 Training run provenance

- [ ] **M-1** `training_run_shared.py:121`: raise unless `--allow-registry-skip`; when skipped, write a `registry_skip_receipt.json` next to the run with the reason. One hour.
- [ ] **M-2** Hash base weights (`base_weights_sha256`, path, source URL if downloaded) in both YOLO trainers; add `base_weights_sha256` and `parent_run_id` columns to `training_runs` via a migration.
- [ ] **M-3** Pass and record ultralytics `seed` and `deterministic`; record `torch.backends.cudnn.{deterministic,benchmark}` and the realized device name and driver in the report.
- [ ] **M-4** Detect training report records `effective_training_params` as the ultralytics-merged args (the pattern pose already uses), not the palette-supplied subset.
- [ ] **M-5** Merged training zarr exports write a rowset content digest (grammar #8 from the provenance review) into attrs and the manifest; `training_runs` gains `dataset_content_sha256`; trainer verifies before training.
- [ ] **M-6** One `training_export` `schema_id` + version shared by detect, keypoint, subject-mask exporters; subject-mask gets a version at all. Add `git_commit` and `git_dirty` via the existing `get_git_info`.
- [ ] **M-7** Collapse the five report shapes onto one `training_run_receipt` schema with the same git-identity block the provenance review specifies (`palette_code_identity`); delete `_normalize_git_payload` once writers agree.
- [ ] **M-8** A `cluster/lsf` planner for training so training jobs get commit pinning, status JSON, and the runtime commit assertion from wave-1 Package B. Retire the inline `subprocess.run` pipelines once parity is tested.
- [ ] **M-9** `register_training_run.py` backfill verifies the manifest sha and model sha it is handed against the files on disk before writing.

### 4.4 Subtraction

- [ ] **X-1** Delete `train_keypoints.py` and `run_pose_training_pipeline.py` aliases; pick one name.
- [ ] **X-2** Retire keypoint v2 legacy export once every consumer reads v3; split the 4,974-line module by path.
- [ ] **X-3** Decide whether `prepare_pose_training_from_registry` or `prepare_keypoint_training_from_registry` survives.
- [ ] **X-4** Index training sets on disk: `training_sets` gains `superseded_by`; write a one-shot census of `/nvme1/training/datasets/` and the v3 benchmark root and mark detect v001–v003 and eye-mask sets.

---

## 5. Sequencing

Independent of wave 1 and safe now: M-1 (one hour, same shape as Package A but a separate file), M-2, M-3, M-4, T-4, X-1, X-3. These are all local to `training/` and `utils/export_*`.

After wave 1 Package A and the resolver work: T-2, T-3, T-6 (they touch model resolution and selector eligibility), R-2 (a fail-closed validator), M-8 (depends on Package B's runtime assertion).

Recommended first training package: **M-1 + M-2 + M-3 + T-4 + T-1**. It makes the next training run reproducible and makes contamination visible in one query, without changing any selector.

---

## 6. What was not assessed

- No training zarr or registry row was opened; disk sizes are from `ls` only.
- Label quality, annotator agreement, and the labeling web app's own receipts were out of scope.
- TensorRT engines were reviewed for binding only, not for numerical parity with the source checkpoint.
- Whether any current production model was in fact trained on recordings it was later run on was not checked. T-1 is the query that answers it and should be run first.
