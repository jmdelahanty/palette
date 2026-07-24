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
docs/archive/analysis_dense_array_migration_todo.md
docs/body_frame_contract.md
docs/archive/crimson_refined_detect_manual_contract.md
docs/derived_metrics_schema_contract.md
docs/detect_batch_analysis_zarr_parallel_agents_contract.md
docs/detection_chunking_findings.md
docs/archive/eye_mask_row_mapping_contract.md
docs/archive/eye_subject_mask_unification_design.md
docs/geometry_live_gpu_design_note.md
docs/grating_analysis_acquisition_questions.md
docs/keypoint_heading_computation_contract.md
docs/keypoint_heading_validity_todo.md
docs/keypoint_late_correction_contract.md
docs/keypoint_merged_row_gate_contract.md
docs/keypoint_quality_registry_workflow.md
docs/archive/keypoint_refined_coordinate_space_incident_2026-03-04.md
docs/keypoint_review_policy.md
docs/archive/keypoint_review_status_notes.md
docs/keypoint_training_data_card_contract.md
docs/kvikio_gds_subject_mask_experiment.md
docs/legacy_archive_migration_policy.md
docs/multicamera_3d_analysis_todo.md
docs/paintera_palette_subject_mask_workflow.md
docs/pipeline_metadata_boundaries.md
docs/pose_heuristic_profile_contract.md
docs/pose_kinematics_run_design.md
docs/archive/pose_schema_heuristics_split_proposal.md
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
docs/archive/zarr_transfer_benchmark_plan.md
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

## Part 2 — src/fisheye/utils/ — DEFER to the existing strategy

**Do not run a separate utils deletion plan from this doc.** An authoritative,
deeper plan already exists: **`docs/utils_reorganization_strategy.md`**
(2026-07-04, synthesis of four opus agents). This section is demoted to a
pointer plus the small set of facts this session independently verified.

Why the deferral: the reorg strategy reframes the problem correctly and this
session's census confirmed the reframe the hard way.

- **The dead tail is small and is not the prize.** Strategy sizes confirmed-dead
  at ~1–1.5k LOC; the real win is Layer 2 — ~120 copy-pasted helpers with **3
  live correctness drifts** (`_iter_zarr` recording-discovery divergence,
  `_utc_now` 4-format provenance timestamps, `_write_json` non-atomic writers).
  That is Phase 1 of the strategy and the highest-value next work — a careful
  per-call-site consolidation, **not** a blind sed, and worth its own focused
  effort, not a tail-of-session cleanup.
- **"Orphan-in-code ≠ dead" — verified twice this session.** The strategy's
  central guardrail held: my census's reference detection had false negatives.
  `setup_experiment_metadata.py` (census said unreferenced) is invoked as a
  subprocess script by `cli/interactive_launcher.py` — a near-miss deletion of
  live code. `patch_legacy_h5.py` (strategy said zero-ref) is referenced by
  the now-retired `visualization/visualize_experiment_timeline_combined_h5.py`.
  That visualizer was deleted after operator confirmation that it is no longer
  used; `patch_legacy_h5.py` was then deleted after operator confirmation that
  enum patching now happens at acquisition. **Gate deletion on code+test+script
  greps of the bare module name plus operator sign-off, not import-graph class.**
- **H5 is not fully legacy.** `analysis/import_stimulus_to_zarr.py` and
  `analysis/calibration_manager.py` still ingest H5.

### What this session actually executed for utils — and why

The strategy's "7 high-confidence deletes (read-and-confirmed, zero refs)" list
did **not** survive per-file inspection. Under operator review (Jeremy flagged
the `check_h5_*` scripts as import-adjacent), the H5 group turned out to be the
**live or operator-gated H5→zarr stimulus-import toolkit**, not spent debris.
Six files remain gated; one file, `patch_legacy_h5.py`, was deleted only after
the separate confirmation that enum patching now happens at acquisition:

| File | Reality |
|---|---|
| `backfill_h5_metadata.py` | imports `analysis.import_stimulus_to_zarr` — wired into the live import module |
| `fix_stimulus_mode_mappings.py` | mutates `/enums/stimulus_modes` in Citrus H5 — stimulus repair tool |
| `inspect_zarr_events.py` | verifies `import_stimulus_to_zarr` captured the event stream |
| `read_h5_data.py`, `check_h5_tracking_data.py`, `check_h5_subject_metadata.py` | operator eyeball scanners for the same H5 import path |
| `patch_legacy_h5.py` | deleted after confirming enum patching happens at acquisition and the H5 timeline visualizer is retired |

**Lesson (reinforces the strategy's own guardrail, harder than it stated it):**
"H5 + old" read as "dead" to both my census *and* four opus agents; it was
actually the operator's import toolkit. The confirmed-dead surface is smaller
than the strategy's ~1–1.5k LOC estimate — possibly near zero without live
operator sign-off per file. Deletion of utils must be **operator-gated per
file**, full stop; static analysis (import graph *or* read-and-confirm by an
agent) is insufficient. This makes Phase 1 (drift consolidation) even more
clearly the only high-value utils work available without the operator in the
loop.

### Everything else about utils

Lives in `docs/utils_reorganization_strategy.md`. Do not duplicate its tranches
here. The census table this session produced (referrer-class breakdown) is a
weaker view of the same ground the strategy's four-axis analysis already covers;
regenerate on demand rather than maintaining a second copy.

---

## Status log

- 2026-07-08: inventory created.
- 2026-07-08: docs tranche 1 executed — 35 docs moved to `docs/archive/`
  (top-level docs/ 273 → 238). Tranches 2/3 still open.
- 2026-07-08: utils section demoted to a pointer at
  `utils_reorganization_strategy.md` (2026-07-04) after discovering it as the
  authoritative plan. Attempted the strategy's 7-file "high-confidence delete";
  reverted all of it — operator review found the H5 cluster is the live
  stimulus-import toolkit. At that point, **net utils deletions: 0**. No
  competing utils plan is maintained in this doc.
- 2026-07-08: deleted obsolete
  `src/fisheye/visualization/visualize_experiment_timeline_combined_h5.py` after
  operator confirmation that the H5 timeline visualizer is unused. This removes
  the live import of `patch_legacy_h5.py`.
- 2026-07-08: deleted `src/fisheye/utils/patch_legacy_h5.py` after operator
  confirmation that legacy enum patching now happens at acquisition. The
  remaining H5 import, scanner, repair, and backfill tools are still
  operator-gated per file.
