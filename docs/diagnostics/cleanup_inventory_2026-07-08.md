# Cleanup inventory — docs/ and utils/ (2026-07-08)

Evidence-based deletion/archive plan. Follows from the accretion findings in
`codebase_review_2026-07-01.md` (utils growth, doc corpus ~28% the size of src).
This doc is itself subject to the rule it proposes: **archive it to
`docs/archive/` in the same commit that completes the last checkbox.**

## Method

Staleness here is *evidence*, not vibes. Three mechanical signals per file:

1. **Last git-touched date** (`git log -1 --format=%as -- <file>`); "pre-May"
   (>60 days) is the staleness threshold.
2. **Inbound references** — for docs: basename grepped across living docs
   (top-level + diagnostics, not archive/), all of `src/`, `scripts/`,
   `CLAUDE.md`, `README.md`. For utils modules: `fisheye.utils.<mod>`,
   `from fisheye.utils import … <mod>`, and `utils/<mod>.py` grepped across
   src-minus-utils, tests, scripts/, pyproject, living docs, and other utils
   modules.
3. **Checkbox state** for todo/plan/checklist docs (`- [ ]` vs `- [x]`).

Caveats: ad-hoc invocation (`python -m fisheye.utils.foo` from shell history)
is invisible to this census — hence the "mark keepers" pass before any code
deletion. Reference detection is basename/module-name matching; a rename since
the referrer was written would show as unreferenced.

---

## Part 1 — docs/ (273 top-level files, ~139k lines; archive/ already holds 83)

Age profile (last touched): 2026-01: 3, 02: 9, 03: 26, 04: 41, 05: 70, 06: 77,
07: 47. **79 docs pre-May.** 63 docs are named `*_todo|plan|checklist`.

### Tranche 1 — archive now, mechanically safe (35 docs)

No inbound references from living docs, code, or scripts (29), or checklist
fully completed (5), or every referenced code path deleted (1).

- [x] Move to `docs/archive/` as a single dedicated commit (do not mix with
      working changes; time it when no parallel agent is mid-flight).
      Done 2026-07-08. `camera_metadata.md` skimmed: prescriptive Jan schema
      note, unreferenced — archived rather than promoted to src/fisheye/docs/.

Unreferenced pre-May (29):

```
docs/camera_metadata.md                             # skim first: pure reference; if still true, belongs in src/fisheye/docs/
docs/citrus_arena_topology_design.md
docs/coverage_unification_todo.md
docs/crop_distributed_tradeoffs.md
docs/detect_keypoints_parity_contract.md
docs/ellipse_fitting_notes.md
docs/int8_trt_todo.md                               # TRT export itself is load-bearing (realtime path); only the stale TODO is archived
docs/keypoint_auto_approval_todo.md
docs/keypoint_row_gating_workflow.md
docs/pose_detect_parity_parallel_agents_contract.md
docs/protocol_hash_stability_todo.md
docs/recording_status_page_design.md
docs/recording_status_page_todo.md
docs/refined_subject_mask_staleness_todo.md
docs/registry_metadata_ownership_refactor_design.md
docs/registry_metadata_ownership_refactor_todo.md
docs/registry_multi_source_provenance_design.md
docs/registry_multi_source_provenance_todo.md
docs/registry_tui_todo.md
docs/repo_wide_staleness_implementation_todo.md
docs/repo_wide_staleness_workflow_edge_checklist.md
docs/sandbox_zarr_fallback.md
docs/subject_mask_keypoint_coverage_runbook.md
docs/swim_bladder_tuning_metadata_audit.md
docs/swim_bout_exponential_segmentation.md
docs/testing_todo.md
docs/training_performance_todo.md
docs/zarr_sharding_design_note.md
docs/zarr_split_policy.md
```

Fully-completed checklists (5):

```
docs/swim_bout_runs_v2_compact_layout.md
docs/training_crop_representation_migration.md
docs/detection_review_web_todo.md
docs/subject_mask_stage_unification_todo.md
docs/goodcopbadcop_cra_primary_endpoint_design.md
```

All referenced code paths gone (1):

```
docs/cluster_workflow_orchestration.md              # 6/6 referenced scripts deleted
```

Note: archiving an open plan is not abandoning it — it stays in git and in
`docs/archive/`. Anything still wanted should be re-scoped fresh anyway.

### Tranche 2 — pre-May but still referenced (50 docs)

Referenced almost exclusively by *other docs* (one code referrer:
`src/fisheye/utils/serve_recording_status_page.py` →
`recording_status_page_deployment.md` — update that pointer when archiving).
Archive in a second sitting; either fix inbound links or accept `archive/`
paths.

- [ ] Second-pass sweep of the list below (skim each; genuinely-current
      contracts get a touch/update instead of archive)

```
docs/analysis_dense_array_migration_todo.md
docs/body_frame_contract.md
docs/crimson_refined_detect_manual_contract.md
docs/derived_metrics_schema_contract.md
docs/detect_batch_analysis_zarr_parallel_agents_contract.md
docs/detection_chunking_findings.md
docs/eye_mask_row_mapping_contract.md
docs/eye_subject_mask_unification_design.md
docs/geometry_live_gpu_design_note.md
docs/grating_analysis_acquisition_questions.md
docs/keypoint_heading_computation_contract.md
docs/keypoint_heading_validity_todo.md
docs/keypoint_late_correction_contract.md
docs/keypoint_merged_row_gate_contract.md
docs/keypoint_quality_registry_workflow.md
docs/keypoint_refined_coordinate_space_incident_2026-03-04.md
docs/keypoint_review_policy.md
docs/keypoint_review_status_notes.md
docs/keypoint_training_data_card_contract.md
docs/kvikio_gds_subject_mask_experiment.md
docs/legacy_archive_migration_policy.md
docs/multicamera_3d_analysis_todo.md
docs/paintera_palette_subject_mask_workflow.md
docs/pipeline_metadata_boundaries.md
docs/pose_heuristic_profile_contract.md
docs/pose_kinematics_run_design.md
docs/pose_schema_heuristics_split_proposal.md
docs/protocol_parameter_registry_todo.md
docs/raw_vs_smoothed_metrics_behavioral_geometry.md
docs/realtime_sparse_row_index_contract.md
docs/recording_status_page_deployment.md
docs/refined_detect_downstream_adoption_checklist.md
docs/refined_detect_multisubject_goal.md
docs/refined_detect_row_identity_contract.md
docs/repo_wide_staleness_checklist.md
docs/repo_wide_staleness_gap_matrix.md
docs/repo_wide_staleness_policy.md
docs/review_status_schema_unification_contract.md
docs/segmentation_stage_split_review.md
docs/subject_mask_tuning_workflow.md
docs/swim_bladder_polar_boundary_design.md
docs/swim_bladder_review_policy.md
docs/track_validity_timeline_design.md
docs/tracking_unassigned_row_policy.md
docs/traditional_subject_segmentation_scaling_todo.md
docs/training_dataset_versioning_todo.md
docs/training_label_origin_phase1_audit.md
docs/zarr_spec_runtime_drift_todo.md
docs/zarr_transfer_benchmark_plan.md
docs/zebrobot_snapshot.md
```

### Tranche 3 — stale open plans: want-it-or-not decisions (17 docs)

Unchecked boxes, untouched since before May. Each needs one 30-second call:
still wanted → re-scope and touch it; not wanted → archive. (Some overlap
tranches 1/2; the decision overlay still applies.)

- [ ] Decide each:

```
docs/multicamera_3d_analysis_todo.md                (36 open)
docs/zarr_spec_runtime_drift_todo.md                (16 open)
docs/keypoint_auto_approval_todo.md                 (1 open / 5 done)
docs/protocol_hash_stability_todo.md                (19 open)
docs/protocol_parameter_registry_todo.md            (28 open / 4 done)
docs/registry_multi_source_provenance_todo.md       (43 open / 4 done)
docs/review_status_schema_unification_contract.md   (5 open)
docs/training_dataset_versioning_todo.md            (11 open / 3 done)
docs/recording_status_page_todo.md                  (15 open / 42 done)
docs/registry_metadata_ownership_refactor_todo.md   (25 open / 42 done)
docs/training_label_origin_phase1_audit.md          (20 open)
docs/traditional_subject_segmentation_scaling_todo.md (6 open)
docs/keypoint_heading_validity_todo.md              (2 open / 4 done)
docs/registry_tui_todo.md                           (64 open / 6 done)
docs/refined_detect_downstream_adoption_checklist.md (6 open / 13 done)
docs/detect_keypoints_parity_contract.md            (8 open / 8 done)
docs/realtime_sparse_row_index_contract.md          (6 open / 0 done)
```

### Forcing functions (docs)

- When a doc's work lands, archive the doc **in the landing commit**.
- No new `*_todo.md` while an older todo on the same subsystem is open —
  extend or archive the old one first.
- Monthly: rerun the census (method above); anything pre-threshold and
  unreferenced auto-moves to archive.

---

## Part 2 — src/fisheye/utils/ (274 modules, ~150k LOC)

Age profile: 2025-10: 2, 2025-11: 1, 2026-02: 20, 03: 12, 04: 14, 05: 33,
06: 30, **07: 163**. The graveyard framing is wrong: utils is a
*fast-churning workshop* — 59% of files touched in the last 8 days, only 49
files (~11.5k LOC) pre-May. The problem is placement and growth rate, not an
ancient stale tail.

### Census by referrer class

| Class | Files | LOC | pre-May files | pre-May LOC |
|---|---|---|---|---|
| 1. Imported by library code (src outside utils) | 14 | 9,838 | 1 | 455 |
| 2. Referenced only by other utils modules | 60 | 52,321 | 8 | 5,055 |
| 3. Referenced only by its own tests | 130 | 61,358 | 9 | 2,433 |
| 4. Referenced only by a scripts/*.sh wrapper | 26 | 12,910 | 7 | 1,500 |
| 5. Referenced only by living docs | 27 | 10,541 | 13 | 5,006 |
| 6. Unreferenced anywhere | 17 | 3,421 | 11 | 2,553 |

Full per-file classification: regenerate via the method above (census script
pattern preserved there; original JSON was session-scratch).

### Structural findings (bigger than the stale tail)

1. **Class 1 is mislabeled library code.** 14 modules in `utils/` are imported
   by real `src/` packages. They should graduate out of utils into the module
   that imports them (or `shared/`), enforced by an import-linter contract
   forbidding `!utils -> utils` imports.
2. **Class 2 is a 52k-LOC shadow library.** 60 modules alive only via other
   utils modules. Needs a reachability pass: build the intra-utils import
   graph, roots = classes 1/3/4/5 keepers; unreachable subgraphs are dead in
   clusters, not single files.
3. **Class 3 (61k LOC, 130 files) is scripts kept alive by their own tests.**
   Mostly recent/active, but the pattern means deleting a script must delete
   its test file too, or the suite pins the corpse in place.

### Tranche U1 — pre-May deletion candidates (~40 files, ~11.5k LOC)

For code, **delete, don't archive** — git history is the archive; archived
code still greps, still imports, still rots. This doc records the list for
recoverability.

- [ ] Mark keepers below (ad-hoc tools invisible to the census), then delete
      the rest + their tests + their scripts/*.sh wrappers in one commit.

Class 6, unreferenced (11):

```
src/fisheye/utils/fix_stimulus_mode_mappings.py     (2025-10-30, 281)
src/fisheye/utils/inspect_zarr_events.py            (2025-11-02, 239)
src/fisheye/utils/check_h5_subject_metadata.py      (2026-02-04, 132)
src/fisheye/utils/check_h5_tracking_data.py         (2026-02-04, 144)
src/fisheye/utils/clear_detection_flags.py          (2026-02-06, 328)
src/fisheye/utils/inspect_keypoint_review_linkage.py (2026-02-08, 301)
src/fisheye/utils/audit_registry_dataset_paths.py   (2026-02-09, 251)
src/fisheye/utils/compare_analysis_training_zarr.py (2026-02-09, 192)
src/fisheye/utils/setup_experiment_metadata.py      (2026-03-29, 279)
src/fisheye/utils/set_crop_review_status.py         (2026-04-14, 100)
src/fisheye/utils/export_protocol_mermaid.py        (2026-04-28, 306)
```

Class 5, docs-only (13) — archive the referring doc first or together:

```
src/fisheye/utils/list_training_versions.py         (2026-02-04, 151)
src/fisheye/utils/read_h5_data.py                   (2026-02-04, 475)
src/fisheye/utils/report_zarr_storage.py            (2026-02-04, 309)
src/fisheye/utils/backfill_pose_onnx_registry_metadata.py (2026-02-08, 138)
src/fisheye/utils/check_training_sample_accounting.py (2026-02-08, 217)
src/fisheye/utils/run_pose_training_pipeline.py     (2026-02-08, 10)
src/fisheye/utils/rename_recording_zarrs_to_training.py (2026-02-09, 177)
src/fisheye/utils/repair_keypoint_offset_corruption.py (2026-03-05, 447)
src/fisheye/utils/compute_backgrounds_batch.py      (2026-03-29, 637)
src/fisheye/utils/repair_keypoint_training_refined_run_ties.py (2026-03-30, 156)
src/fisheye/utils/zarr_inspector.py                 (2026-04-12, 1778)  # likely KEEPER: interactive tool?
src/fisheye/utils/view_merged_pose_training_zarr.py (2026-04-24, 398)
src/fisheye/utils/backfill_refined_subject_mask_metrics.py (2026-04-29, 113)
```

Class 4, shell-wrapper-only (7) — delete script + wrapper as a pair:

```
src/fisheye/utils/validate_detect_training_zarr.py  (2026-02-06, 42)
src/fisheye/utils/prepare_pose_training_from_registry.py (2026-02-08, 10)
src/fisheye/utils/validate_keypoint_training_zarr.py (2026-02-08, 42)
src/fisheye/utils/index_source_recording_profiles.py (2026-03-05, 867)
src/fisheye/utils/index_training_data_cards.py      (2026-03-05, 338)
src/fisheye/utils/sync_refined_subject_mask_metadata.py (2026-04-15, 95)
src/fisheye/utils/write_refined_subject_mask_edit.py (2026-04-29, 106)  # KEEPER? exposed as pyproject console script palette-write-refined-subject-mask-edit
```

Class 3, tests-only (9) — delete test file with the script:

```
src/fisheye/utils/check_detect_training_config.py   (2026-02-06, 132)
src/fisheye/utils/validate_recording_manifest.py    (2026-02-09, 397)
src/fisheye/utils/list_detect_group_fallbacks.py    (2026-02-10, 133)
src/fisheye/utils/serve_recording_status_page.py    (2026-03-28, 96)   # verify vs live status_page/ tooling before deleting
src/fisheye/utils/inspect_roi_cache.py              (2026-04-04, 375)
src/fisheye/utils/backfill_hevc_keyframe_flags.py   (2026-04-17, 257)
src/fisheye/utils/inspect_refined_detect_run.py     (2026-04-23, 331)
src/fisheye/utils/patch_crops_from_refined.py       (2026-04-24, 663)
src/fisheye/utils/validate_subject_mask_training_zarr.py (2026-04-24, 49)
```

Standing exclusion: anything on the TensorRT export path is load-bearing for
realtime acquisition — never delete via this process.

### Tranche U2 — structural follow-ups

- [ ] Graduate the 14 class-1 modules out of utils; add import-linter contract.
- [ ] Intra-utils reachability pass over the 60 class-2 modules; delete dead
      clusters.
- [ ] Growth rule going forward: a new utils script is born with an expiry —
      either it gains a shell wrapper / doc runbook (operational) or it is
      deleted when its investigation lands. Monthly census sweep enforces.

---

## Status log

- 2026-07-08: inventory created.
- 2026-07-08: docs tranche 1 executed — 35 docs moved to `docs/archive/`
  (top-level docs/ 273 → 238). Tranches 2/3 and utils U1/U2 still open.
