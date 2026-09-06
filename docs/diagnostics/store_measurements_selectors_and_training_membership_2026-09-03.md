# Store Measurements: Selector Disagreement and Train/Eval Membership — 2026-09-03

> Correction added 2026-09-06; original measurements and mutation history remain
> below. The [second opinion](review_wave_second_opinion_2026-09-04.md) establishes 32 selector-pair mismatches as 28 RedScare
> plus four GoodCopBadCop, four registry backups rather than five, and remaining
> `/nvme1` references. Five recording/model overlaps across four cameras do not
> establish exact training frames, downstream evaluation leakage, or that all
> other recordings are clean. S-5's blanket `latest` backfill is rejected:
> missing selectors can be deliberate ineligibility, and names do not prove
> review. These are corrections to historical evidence, not new measurements.
> See the [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation) before any remediation.

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-03
implementation: measured
-->

**Date:** 2026-09-03
**Snapshot:** `c051443c`; registry `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite` (mtime 2026-09-02 19:05); stores under `/groups/.../recordings/` and `/nvme1/recordings/`.
**Method:** two read-only passes. (1) A sweep opening every active `source_recording` dataset's archive with `use_consolidated=False`, reading each run family's selector attrs (`latest`, `latest_complete`, `latest_pending`, `latest_any`, `latest_materialized`, `authoritative_run`), the sorted-last child, the registry `run_name` and `latest_selector` for the matching step, and model-identity attrs on every candidate run. (2) A registry expansion of `training_sets.dataset_ids_json` through `dataset_lineage` (`training_merge_source`) to leaf recording ids per `training_models` row, joined to the inference model found on each recording's selected run. Scripts: `scripts/sweep_run_selectors.py` (sweep + per-family table) and `scripts/check_training_membership.py` (membership join), both read-only; the numbers below were reproduced with the landed versions. No writes.
**Questions:** (a) how many live runs were located through a fallback, and where do the resolvers disagree (P-10 in the architecture review; prerequisite for the flag flips in the propagation trace); (b) was any production model run on recordings it was trained on (T-1 / L-2).
**Companion evidence:** [`canonical_propagation_trace_2026-09-03.md`](canonical_propagation_trace_2026-09-03.md), [`training_data_and_model_provenance_review_2026-09-01.md`](training_data_and_model_provenance_review_2026-09-01.md), [`architecture_review_five_lens_2026-09-01.md`](architecture_review_five_lens_2026-09-01.md) P-10.

---

## 1. Headline

| Question | Answer |
|---|---|
| Archives registered / opened | 408 / 296. **112 registered archives do not exist on disk** (56 recordings, all `/nvme1` paths, both training and analysis, parent directories gone). |
| Families with multiple runs and **no `latest`** | crops 191 of 248; refined detect 124 of 225; refined subject masks 103 of 176; keypoints 85 of 205; tracking 84 of 153. These families are resolved by sorted order or source-match today. |
| `latest` disagrees with sorted-last | detect **120 of 122** multi-run families; keypoints 28; refined keypoints 29. A sorted-last fallback on detection picks the wrong run almost every time. |
| Registry run selected by a sorted fallback | 46 `ok` rows (40 crop, 4 stimulus, 1 detect, 1 refined detect). Plus 63 crop rows whose registry run equals sorted-last with no `latest` present. |
| Registry disagrees with zarr `latest` | 11 rows total (refined detect 3, crop 3, refined keypoints 4, subject masks 1). Small, but nonzero, and reconcile trusts the zarr side. |
| `latest` ≠ `latest_complete` | 32 refined-keypoint families, all RedScare training-review runs. `latest` points at the seed, `latest_complete` at the review. |
| `latest_pending` set | 16 refined-subject-mask families. |
| Production models in use | 5. One pose model (`pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2`) on every keypoint run in the store. Detect: `cedar_shadow_v007` (113), `all_available_v004` (127), `v002` (1). Subject masks: `union_all_components_v001` (166). |
| **Recordings inferred by a model trained on them** | **5 run-level hits on 4 recordings**, all `sleepyfish_2026_05_05_17_45_30_cam20100{93,94,95,96}`: 4 keypoint runs (pose model) + 1 detect run (`v002`). All four are the registry-selected `ok` run and all four have `tracks`, `track_kinematics`, and `swim_bouts` = `ok` downstream. |
| Training recordings that no longer exist | 51 of the pose model's 59 training recordings are the `2026-01-28T19-22-28Z_arena_*` set, whose archives are the missing `/nvme1` paths. They cannot be re-inferred today, so they cannot currently contaminate. They also cannot be re-exported. |

**Reading.** The contamination answer is narrow and real: one four-camera session is in both the pose and detect training sets and is analyzed with those models through to swim bouts. Every other production recording is clean against its model. The selector answer is broader than expected: `latest` is absent on the majority of crop, refined-detect, and refined-mask families, so those families already live on the fallback paths the propagation trace proposes to delete.

---

## 2. Selector disagreement per family

Counts are archives where the family exists with at least one run.

| family | present | multi-run | `latest` set | `latest` unset | `latest`≠sorted-last | `latest`≠`latest_complete` | `latest_pending` | registry `ok` | registry≠`latest` | registry = sorted-last, no `latest` | registry selected by sorted fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|
| detect_runs | 246 | 122 | 245 | 1 | **120** | 0 | 0 | 214 | 0 | 1 | 1 |
| refined_detect_runs | 225 | 97 | 101 | **124** | 1 | 0 | 0 | 97 | 3 | 1 | 1 |
| crop_runs | 248 | 113 | 57 | **191** | 0 | 0 | 0 | 100 | 3 | **63** | **40** |
| keypoints_runs | 205 | 129 | 120 | 85 | 28 | 0 | 0 | 116 | 0 | 0 | 0 |
| refined_keypoints_runs | 193 | 70 | 104 | 89 | 29 | **32** | 0 | 102 | 4 | 0 | 0 |
| subject_mask_runs | 173 | 96 | 85 | 88 | 4 | 0 | 0 | 85 | 1 | 0 | 0 |
| refined_subject_masks_runs | 176 | 93 | 73 | 103 | 5 | 0 | **16** | 71 | 0 | 0 | 0 |
| tracking_runs | 153 | 40 | 69 | 84 | 4 | 0 | 0 | 65 | 0 | 0 | 0 |

Registry `latest_selector` distribution over all `ok` rows (from `details_json`): `latest_attr` dominates; `sorted_fallback` 45, `source_match_sorted_fallback` 1, `source_match_latest_attr` 296 (refined detect 93, refined keypoints 141, refined masks 62), `runtime_*_write` 136 (masks), `keypoint_failure_review_manual` 21, `nested/root_latest_attr` 122 (detect quality), `<none>` on every calibration, geometry, gate, and dish-mask row.

**What this means for the propagation-trace flag flips.**

- **D-1 (refine-detect `latest` fallback).** Only 1 of 246 detect families lacks `latest`, and registry never disagreed. Deleting the fallback is safe on the live store. But 120 of 122 multi-run detect families would resolve to a *different* run under sorted-last, so any code path that still sorts (the cross-recording export, subject-mask export) must be closed at the same time.
- **K-1/K-5 (tracking resolver).** 84 of 153 tracking families have no `latest`; `resolve_tracking_run`'s attr-scan is what picks them. Replacing it with the manifest-verified handle needs a per-archive migration that writes `latest` from the registry row first (I-3 in the pipeline-model doc), or the swap will fail on 84 archives.
- **M-2 (mask refinement `latest`).** 88 of 173 raw mask families have no `latest`. Same migration dependency.
- **X-5 / X-8 (exports' sorted-last).** 191 crop families and 124 refined-detect families have no `latest`; the exporters' sorted-last is the live path for them. 40 registry crop rows were themselves chosen by sorted fallback. Crop is the family where the fallback is load-bearing.
- **`latest` ≠ `latest_complete` on 32 RedScare refined-keypoint families.** Two selectors name different runs. Any consumer reading `latest` (track kinematics heading, keypoint export) gets the seed, not the reviewed run. This is a live instance of the dual-representation finding F-2.

---

## 3. Train/eval membership

### 3.1 Models and training sets

| model run | task | training recordings | inferred recordings (store) | overlap |
|---|---|---|---|---|
| `pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2` | pose | 59 (51 × `2026-01-28T19-22-28Z_arena_*`, 4 sickyfish, 4 sleepyfish) | 205 (every keypoint run) | **4** |
| `detect_all_available_detect_training_v004_yolo11n_trt_20260520` | detect | 60 (incl. the 4 sleepyfish) | 127 | 0 |
| `omnifin0_cedar_shadow_v007_detect_20260206-235656_25f3fbcb` | detect | 52 | 113 | 0 |
| `detect_all_available_detect_training_v002_yolo11n_trt_20260513_tmux` | detect | 60 (incl. the 4 sleepyfish) | 1 | **1** |
| `subject_masks_union_all_components_v001` | subject masks | 15 | 166 | 0 |

Two `training_models` rows have no `set_id` (`unknown_rig_..._detect_20260305`, `subject_masks_union_canary_v001`); their membership is unknowable from the registry. Two pose rows are `failed`.

### 3.2 The five hits

| recording | family | selected run | model | downstream `ok` |
|---|---|---|---|---|
| `sleepyfish_2026_05_05_17_45_30_cam2010093` | keypoints | `keypoints_registry_sleepyfish_cam2010093_full_20260714_v001_…` | pose retry2 | refined_keypoints, tracks, track_kinematics, swim_bouts |
| `…cam2010094` | keypoints | `keypoints_registry_sleepyfish_cam2010094_full_encoded_20260715_v002_…` | pose retry2 | same |
| `…cam2010095` | keypoints | `keypoints_sleepyfish_kp_snapshot_20260718_01` | pose retry2 | tracks, track_kinematics, swim_bouts, subject_masks |
| `…cam2010095` | detect | `detect_2026-05-14_15-39-11` | detect v002 | same |
| `…cam2010096` | keypoints | `keypoints_registry_sleepyfish_cam2010096_full_encoded_20260717_v001_…` | pose retry2 | refined_keypoints, tracks, track_kinematics, swim_bouts |

All four recordings are direct members of both training sets (`dataset_ids_json`), not merged-in via lineage. The detect run on cam2010095 is the registry-selected run and `latest`; the other detect runs on that session use a model not trained on them.

**Consequence.** Any kinematics, bout, or export result that includes the `sleepyfish_2026_05_05` session was computed on model output for frames the model saw as labels. Whether that matters depends on how those recordings are used: as an evaluation of the model, it is invalid; as a behavioral measurement, the effect is an optimistic bias in keypoint accuracy on those four recordings relative to the rest of the cohort. Nothing in the registry or the run attrs flags it.

### 3.3 The 51 vanished training recordings

The pose and detect training sets are dominated by `2026-01-28T19-22-28Z_arena_*` recordings registered only under `/nvme1/recordings/`. The sweep found none of those 112 archive paths on disk, with parent directories absent. The registry rows are `status='active'`. Consequences:

- Training-set provenance for the production pose model points at archives that no longer exist. The merged training zarr at `/nvme1/training/datasets/pose_all_registry_reviewed_v2_keypoints_20260520_v001/zarr/…_merged.zarr` (present, 29 GB for all merged sets) is now the only copy of those labels; its `training_export.source_zarr_paths` lists the 59 vanished `/nvme1/recordings/...` archives, and the training review already found it carries no persisted content inventory. Follow-up on 2026-09-03: `/nvme1/recordings/` holds only a canary, `figures`, `logs`, and `smoke`; the 52 `2026-01-28` and 4 sickyfish recording directories are absent from `/nvme1`, `/groups/.../recordings`, `palette_backups`, `old_videos`, and `staging`. The 2026-07-06 detect-review pointer census read them at their registered paths, so they were removed between then and now.
- `model_resolution` feature-similarity scoring still counts those rows.
- Any "re-export the training set" or "re-review the labels" action is impossible for 51 of 59 recordings.

This was not a question the sweep set out to answer. It is the most consequential thing it found.

---

## 4. Checklist

The checklist below is preserved as a historical record. S-5 is rejected, not
pending execution. S-6 requires validated review/use evidence and the corrected
28-plus-four cohort accounting, not promotion inferred from run names. S-3/S-4
need use-scoped membership and evaluation decisions; the scan did not establish
exact-example leakage or a mandatory cohort exclusion. Completed mutation rows
describe past operations, not permission to repeat them. See the correction at
the top and the second opinion before adopting any item into the owning queue.

- [x] **S-1** Done 2026-09-03: 112 `/nvme1/recordings/...` `source_recording` rows set `status='missing'` in one transaction after a validated backup (`registries/backups/palette_registry_20260903_024848.sqlite`); 296 active recording datasets remain. Rows kept because `training_sets.dataset_ids_json` and `dataset_lineage` reference them. The 17 `derived_training_merge` rows are handled by S-2b.
- [x] **S-2** Decision (operator, 2026-09-03): the `2026-01-28` and sickyfish per-recording archives were deleted deliberately to reclaim `/nvme1` space; they were old, migrated-from data not needed going forward. `/groups/.../recordings` is the only recording store from now on, backed up and managed by HPC/IT. The registry was not updated at deletion time (S-1).
- [x] **S-2b** Done 2026-09-03. `rsync -a /nvme1/training/datasets/ /groups/johnson/johnsonlab/jeremy/training/datasets/` (30.16 GB, 11,771 files). All 19 merged zarrs verified by per-tree content digest (sorted relpath+size+sha256 of every file), 0 mismatches; digests persisted at `/groups/.../training/datasets/_index/copy_verification_from_nvme1_2026-09-03.json`. After a second validated backup (`registries/backups/palette_registry_20260903_025814.sqlite`), one transaction re-pointed the 17 `derived_training_merge` rows (`zarr_path` + recomputed `path_hash`, `dataset_id` unchanged) and 13 `training_runs.manifest_path` values to the `/groups` copy. Still open: (i) one training run + model row (`eye_mask_..._unet_20260226-001529`) has `model_path` under `/nvme1/training/models/`, which was not copied; (ii) the three `path-<hash>` dataset ids were derived from the old path hash, so a future `register_from_root` over the new location would mint duplicate rows for them unless it matches on `dataset_id` first; (iii) the merged zarrs still carry no in-archive content digest (B-7 in the census); the `_index` JSON is the interim receipt. `/nvme1/training/datasets` may now be treated as scratch.
- [ ] **S-3** Flag the four `sleepyfish_2026_05_05` recordings: add a `training_membership` column or view to the registry (T-1 in the training review) and have `model_resolution` and the analytics exporters read it. Any cohort or export that includes them declares it in the manifest.
- [ ] **S-4** Decide whether the `sleepyfish_2026_05_05` session stays in the analysis cohort. Options: exclude; keep with a flag; re-infer with a model that excludes the session. Record the decision.
- [ ] **S-5** Before D-1/K-1/M-2/X-5/X-8, run a one-time migration that writes `latest` on every family where it is unset, taking the value from the registry `run_name` when the row is `ok` and refusing when the registry disagrees with `latest_complete`. Re-run this sweep afterwards; the "latest unset" column must be zero for the families being flipped.
- [ ] **S-6** Resolve the 32 RedScare refined-keypoint families where `latest` ≠ `latest_complete`. Decide which is authoritative (the reviewed run, presumably) and set both.
- [x] **S-7** Scripts landed as `scripts/sweep_run_selectors.py` and `scripts/check_training_membership.py`. Still open: schedule them (the architecture review's P-10) with the mismatch counts as the error-budget signal.
- [x] **S-8** Scanner reads `registry_run_id` inside schema bindings; all 205 keypoint runs resolve directly. Still unresolved: 5 detect runs and 7 subject-mask runs whose attrs carry no model identity at all.

- [x] **S-9** Model-store coherence (2026-09-03). Survey: `jeremy/models/<task>/<set>/<run>/` is the live tree; every `training_models`, `onnx_models`, `tensorrt_models` path resolves there and exists on disk. `jeremy/palette_models/` is a 2026-05-14 rsync snapshot of the old `/nvme1/models` (679 files, own sha manifest), a strict subset of `models/`, byte-identical where shared, referenced by nothing; **operator to delete** (keep its `MANIFEST.sha256` under `operations/` first). After a third validated backup (`registries/backups/palette_registry_20260903_030504.sqlite`) one transaction: re-pointed the eye-mask UNet `training_runs`/`training_models` `model_path` from `/nvme1/training/models/...` to the identical file (same sha) under `models/eye_masks/`; re-pointed 13 `training_runs.config_path` values into the copied `/groups/.../training/datasets` tree; deleted the two `failed` pose runs (`..._20260520_01`, `..._retry1`) from `training_runs` and `training_models` after confirming no ONNX/TensorRT/deployment/export row referenced them. Result: 14 `training_runs`, all `success`; 14 `training_models`; zero `/nvme1` references in either table. Follow-ups the same day: `palette_models/` deleted (manifest, metadata, README kept at `operations/palette_models_snapshot_20260514/`; three docs' example commands re-pointed to `models/`); the two set-less runs (`unknown_rig_..._detect_20260305`, `subject_masks_union_canary_v001`) deleted from `training_runs`/`training_models` after a fourth backup (`palette_registry_20260903_031808.sqlite`) and a scan of every `*run_id` column plus JSON blobs found only their own model rows. Their weight directories `models/detect/detect_config_dan_talk_manual_rect/` (31 MB) and `models/subject_masks/subject_mask_union_canary_20260406/` (96 MB) were then deleted at operator request after confirming zero registry, ONNX, TensorRT, or model-path references; `models/` is 4.7 GB. Registry now: 12 `training_runs`, all `success`, all with a `set_id`; 12 `training_models`. `docs/artifact_storage_map.md` now has a Storage Roots section documenting `recordings/`, `training/datasets/`, `models/<task>/<set>/<run>/`, `registries/`, and `operations/`, with the models-tree layout and the one-tree rule.

## 5. Caveats

- The membership join used model path substrings, model file sha, and engine sha. TensorRT engines carry a different sha from the `.pt`, so a run that recorded only an engine sha not present in `tensorrt_models` would be unresolved, not miscounted.
- `training_sets.dataset_ids_json` was taken as the membership truth. If a merged training zarr was built from more sources than its set row lists, membership is undercounted. The training review's R-items address the missing binding.
- The first pass left 84 goodbatbadbat keypoint runs unresolved because their model identity sits under `pose_model_schema_binding.model.registry_run_id`; the landed scanner reads that key and the re-run resolved all 205 keypoint runs with the same five hits.
- No eye-mask or subject-shape model membership was checked (no `set_id` linkage for the eye-mask model; subject shape has no model).
