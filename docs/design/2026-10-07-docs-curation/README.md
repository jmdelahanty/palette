# Palette docs curation: triage manifest

- **Status:** proposal for review. Nothing has moved yet.
- **Owner:** Jeremy Delahanty.
- **Snapshot:** Palette `main` at `d7797af1` (2026-10-07).
- **Builds on:** the shared docs contract in agent-contracts PR #62 (`docs-contract/`), which is not merged yet. Frontmatter waits for it.
- **Manifest:** [`manifest.csv`](manifest.csv). One row per doc, sorted needs_decision → merge → archive → keep. GitHub renders it as a searchable table.

## Goal

Palette's docs are mostly old, confusing or stale. The plan is to trim them and build a small curated core that gives anyone working with Palette, human or agent, a real chance. Agreed sequence:
1. This manifest.
2. Your review.
3. One reversible commit that moves every approved `archive` row into `docs/archive/`.
4. A curated core of 10-20 docs, each with frontmatter and `verified_against`.
5. Only the core is validated, with the shared `docs_index.py`.

## How the manifest was made

- **Scope:** five read-only agents read all 608 markdown files outside `docs/archive/`. That is 387 top-level `docs/` files, 144 in `docs/diagnostics/`, and 77 elsewhere: root files, `docs/design/`, `docs/operator_guide/`, `agents_todo/`, `src/fisheye/docs/`, and READMEs next to code.
- **Decision:** for each doc they proposed `keep`, `merge` (with a target), `archive` or `needs_decision`, with a one-line reason.
- **Spot checks:** for keeps, they grepped 1-3 concrete claims (module, function, flag, array name) against the code. The `checked` column records each result.
- **Inbound references:** these were computed mechanically before triage.
  - `load_bearing_refs` lists files in `AGENTS.md`, `src/`, `scripts/`, `.github/` or `tests/` that mention the doc.
  - `refs_need_update = yes` marks an archive or merge row that something still references.
  - README rows were recounted by full path, because bare `README.md` matches were false positives.
- **Validation of the merged output:**
  - every doc appears exactly once;
  - every non-`NEW:` merge target exists;
  - one merge chain (swim_bladder → mask_review_save_approval → refined_subject_masks_runs_contract) was resolved to its final owner.

The agents judged docs; they did not rewrite them. A `keep` means "still true and worth linking". It is not a promise that the doc is correct line by line.

## Numbers

| Proposal | Docs | Lines |
|---|---:|---:|
| archive | 279 | 84,025 |
| merge | 102 | 27,267 |
| keep | 195 | 80,402 |
| needs_decision | 32 | 17,492 |

Approving archive and merge as proposed leaves about 195-227 docs outside the archive, down from 608. The curated core sits on top of these: about 10-20 pages that link into the keeps rather than replace them. Frontmatter would go on the core first. Keeps would be adopted gradually and shown as "not yet adopted" in the meantime.

**Biggest merge owners:**

| Owner | Docs merged in |
|---|---:|
| `cross_recording_analytics_export_design.md` (per-table Arrow schema docs; field lists already live in `analytics_exports/arrow_contracts.py`) | 7 |
| `refined_subject_masks_runs_contract.md` | 6 |
| `stimulus_response_run_design.md` | 4 |
| `validated_behavior_cohort_export_implementation_design_2026-08-31.md` | 4 |
| `training_data_workflow.md` | 3 |
| `keypoint_training_workflow.md` | 3 |
| `subject_shape_runs_contract.md` | 3 |

19 merge rows target a new core page (`NEW:<topic>`). Nine of those target `data_model_store_layout`.

## Decisions for you (32)

The exact question for each is in the manifest's `reason` column (filter `proposal = needs_decision`). They group into five kinds:

- **Is this plan still live?**
  - `chaser_analytics_roadmap_2026-08-10` (A0-A7)
  - `chaser_exact_full_gap_closure_implementation_checklist_2026-08-30` (324 unchecked)
  - `composable_chaser_analytics_implementation_checklist_2026-08-20`
  - `derived_analytics_storage_contract_audit_and_checklist_2026-08-03`
  - `canonical_detection_storage_implementation_checklist` (92 open)
  - `clipped_whole_detection_convergence_implementation_checklist`
  - `core_behavior_paradigm_composition_authority_design_2026-09-03`
  - `source_of_truth_consolidation_plan_2026-08-25` (§4.7 vs the recording-identity design)

  Several of these are large "active" checklists that compete with the work queue AGENTS.md names.
- **Is this design still planned?**
  - `bout_morphology_collection_design_decision`
  - `container_packaging_and_distribution_design`
  - `crop_only_recording_storage_profile`
  - `track_validity_timeline_design`
  - `keypoint_multi_skeleton_todo`
  - `multicamera_3d_analysis_todo`
  - `native_tensorrt_inference_todo`: the TensorRT *export* path is load-bearing for realtime and stays either way. The question is only the native runtime.
  - `eye_angle_legacy_vergence_gaze_todo`
  - `workload_aware_analysis_scheduling`
  - `brief_chaser_schedule_importer`
- **Superseded or still governing?**
  - `behavior_event_analysis_design_decision`
  - `composable_stimulus_analysis_and_plot_recipes_design`
  - `stable_identity_incremental_materialization_decision` (vs row-scoped staleness)
  - `detection_storage_production_closure_checklist` (canonical v1, vs v3 in `detection_publication_contract`)
  - `chaser_stimulus_camera_temporal_projection_audit_2026-08-20` (its camera-projection policy may still govern)
- **Adopted policy or aspiration?**
  - `error_budget_policy_2026-08-11`
  - `raw_video_storage_tiering_proposal`
  - `workflow_provenance_prior_art` (background reference or archive)
- **Code-coupled choices:**
  - `web_labeling_implementation_status`: `web_handoff_files.py` prints its path into launch bundles. Keep and refresh it, or retarget the code to the runbook?
  - `web_labeling_admin_correction_review_design`: implemented; update and keep, or archive?
  - `group_analytics_viewer_design`: is the stdlib viewer still maintained next to Marimo?
  - `operator_guide/test_data.md`: `/nvme1/palette_test_data` is gone, but `scripts/run_shared_diagnostics_smoke.sh` still uses it.
  - `detect_decode_backend_benchmark_todo`: live performance item, or move to the performance queue?
  - `deferred_keypoint_publication_validation_optimization_2026-08-29`: tracked nowhere else.

## References that must change with the archive commit (8)

| Doc | Proposal | Referenced by |
|---|---|---|
| `agents_todo/registry_query_since_filter.md` | archive | `tests/unit/fisheye/test_registry_query_cli.py` |
| `docs/diagnostics/goodcopbadcop_behavior_synthesis_handoff_2026-07-17.md` | archive | two `analyze_goodcopbadcop_*` docstrings |
| `docs/diagnostics/store_measurements_selectors_and_training_membership_2026-09-03.md` | archive | `scripts/check_training_membership.py`, `scripts/sweep_run_selectors.py` |
| `docs/diagnostics/subject_mask_profile_design_2026-06-18.md` | archive | `src/fisheye/shared/subject_mask_profile.py` |
| `docs/diagnostics/training_data_and_model_provenance_review_2026-09-01.md` | archive | `scripts/check_training_membership.py` |
| `docs/experiment_types_reference.md` | merge → NEW:external_contracts | `src/fisheye/shared/citrus_enums.py` |
| `docs/web_labeling_deployment_examples.md` | merge → web_labeling_deployment_runbook | `docs/web_labeling_implementation_manifest.json` |
| `docs/web_labeling_implementation_checklist_clean.md` | archive | `docs/web_labeling_implementation_manifest.json` |

Each needs one of two fixes in the same commit: re-point the reference to `docs/archive/...`, or drop it if it was only provenance. The two docs AGENTS.md names as live coordination docs are both proposed `keep`: `authority_consolidation_work_queue_2026-08-25.md` and `authority_acceptance_implementation_checklist_2026-08-27.md`.

## Proposed curated core

Each page is new and short. It links to the kept docs that own the details instead of copying them. Each one is checked against the code before it gets `verified_against`.

| Core page | Best source docs found |
|---|---|
| Start here | `README.md`, `current_pipeline_contract.md`, `analytics_math_primer.md` |
| How to run | `operator_guide/pipeline_workflow.md` (needs a rewrite: /nvme1 paths, a dead link), `cluster_batching_guide.md`, `recording_analysis_pipeline_contract.md`, `environment_setup.md` |
| Architecture and data flow | `current_pipeline_contract.md`, `analysis_workflow_dag.md`, `production_dag_recording_layout_design.md`; the Mermaid diagram in `stimulus_response_analysis_flow.md` |
| Data model and store layout | `src/fisheye/docs/zarr_structure.md` (mark the eye-mask sections legacy), `zarr_storage_lifecycle_policy.md`, `analytics_storage_schema_matrix.md`, the generated Zarr census pair, `docs/design/2026-09-23-recording-identity/` |
| Contracts with Citrus, Orange, MetaZebrobot | `unified_h5_native_import.md`, `src/fisheye/shared/unified_h5/contracts/README.md`, `stimulus_coordinate_contract.md`, `recording_manifest_contract.md`, `orange_rolling_clip_recording_contract.md`, `zebrobot_snapshot.md` (shows `schema_version: 1`; the code has the v3 schema), `docs/design/2026-10-06-canonical-subject-fields/`. Link MetaZebrobot's own `docs/zebrobot_snapshot.md` and agent-contracts `metazebrobot-consumers/` rather than copying them. |
| Authority and acceptance | `zarr_run_completion_contract.md`, `run_resolution_semantics.md`, `mutable_review_runs_contract.md`, `publication_receipt_hashing_lifecycle`, the AGENTS-named checklist |
| Stage reference (index page) | one line per family, linking to the owning contract: detection, refined detect, keypoints, subject masks, subject shape, tracking, eye angles, chaser components |
| Operator guide | `web_labeling_deployment_runbook.md`, `geometry_review_web_operations.md`, `cohort_release_workflow.md`, `registry_repair_playbook.md` (backup/prune parts), `operator_guide/training_data.md` |
| Agent coordination | `AGENTS.md`, `docs/design/README.md`, the work queue, `git_worktrees_guide.md` (fix the `sun` branch name), `dask_zarr_write_safety.md` |

## Cross-cutting findings

These came up across batches and should be fixed while the core is built.

- **Status headers are unreliable in both directions.** Some docs say "proposed", "draft", "specified-only" or "not started" for things that exist on main:
  - `run_resolution_semantics`
  - `shared_zarr_storage_policy_design` (StoragePlan is used in 62 files)
  - `chaser_distance_run_contract`
  - `analysis_to_training_promotion_contract`
  - `orange_rolling_clip`
  - the unified-H5 reference-storage, canonical-subject-fields and legacy-adapter designs

  Others call themselves "current" or "active" but are stale snapshots. Don't filter on status lines; the core's `verified_against` replaces them.
- **Deleted store and branch.**
  - `/nvme1` appears 789 times across 123 non-archive docs, 85 of them in `docs/operator_guide/`, although `/groups` is the only live store. Some mentions are legitimate history; the operator guides and how-to docs are the ones that send people to a deleted path.
  - `git_worktrees_guide`, `README.md`'s CI badge and an agents_todo brief still treat `sun` as the main branch.
- **Docs describing deleted code.** Examples:
  - eye-mask tuner and review tools (`eye_mask_tuning_workflow.md`, `mask_review_save_approval_policy`, `keypoint_late_correction_contract`)
  - all 8 scripts cited by `alignment.md`
  - `scripts/submit_eye_masks_batches_bsub.sh` (cluster batching guides)
  - `export_pose_coco`, `draft_video_only_organizer_manifest`, `pose_kinematics_runs`, `check_import_profile`
  - `_choose_profile_candidate` (`training_quality_gate_contract`)
- **Broken links.**
  - AGENTS.md points to `docs/sandbox_zarr_fallback.md`, which only exists in `docs/archive/` now.
  - `pipeline_workflow.md` links the archived `organize_recordings.md`.
  - `web_labeling_implementation_status`, `review_status_schema_unification` and the repo-wide-staleness docs link missing files.
- **Possible policy conflict:** `crimson_stimulus_step_read_contract.md` advises merging directory scans with possibly stale consolidated metadata, which may clash with AGENTS.md's Consolidated Metadata Read Policy.
- **Junk:**
  - `src/best_possible.md` is a pasted terminal log.
  - `smoothing_strats.md` is a generic list.
  - `matlab/matlab.md` is empty.
  - `docs/diagnostics/storage_and_rig_conversation_2026-07-24.md` is a 7,548-line chat transcript.
- **Code defect seen during triage, already tracked:** `registry/stage_complete.py` `emit_stage_completion` still catches every exception, including failed contract validation, and returns `False` with a warning. Most callers ignore the return value, and the run is already `latest` by then. This is item O-1 in `docs/diagnostics/architecture_review_five_lens_2026-09-01.md`; it is reported here, not fixed.

## After your review

1. Edit `manifest.csv` directly, or tell an agent which rows change and what each `needs_decision` row resolves to.
2. **Archive commit:** `git mv` each approved archive row into `docs/archive/`, fix the 8 references above and the AGENTS.md `sandbox_zarr_fallback` link in the same commit, and run the link and census checks. One commit, reversible with `git revert`.
3. **Merge commits:** one per owner doc, so each merge can be reviewed against what it absorbs.
4. **Core pages:** written after the agent-contracts docs contract merges, with frontmatter from that schema.
