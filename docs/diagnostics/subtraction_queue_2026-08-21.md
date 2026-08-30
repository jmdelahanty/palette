# Subtraction Queue: Deletion & Extraction Candidates — 2026-08-21

**Date:** 2026-08-21 (four-agent parallel census at HEAD ~6e7caa07, branch sun)
**Lenses:** dead code, superseded twins, extraction targets, non-code cruft.
**Companion:** `docs/diagnostics/contract_enforcement_divergence_review_2026-08-21.md` (Waves A–D). Several items below are the same work seen from the subtraction side; cross-references marked [=Bn].

**Plan disposition (2026-08-25):** scoped deletion census. Overlapping
authority, resolver, selector, and adapter-retirement status is absorbed into
[`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md).
Unrelated deletion candidates remain evidence here and require current
call-site verification before removal.

**Headline totals:** ~27,500 LOC of high-confidence dead code; ~2,000 more LOC deletable after repointing ≤2 call sites each; ~1,400 LOC of backfill loops deletable once the dry-run fix lands; 62 doc/script/config/brief files DELETE or ARCHIVE; and ~10 extraction moves that collapse 300+ duplicated sites (including two live bug classes: 35 CLIs accepting `--apply --dry-run` together, and 21 read-intent tools opening the registry writable).

---

## Part 1 — Delete now (zero live references, verified)

### 1.1 `src/chaser_analysis/` — 30 of 48 modules, ~12,700 LOC
Only one live import into the package exists: `fisheye/analysis/swim_bout_statistics.py:38` → `chaser_analysis.swimming_bout_analysis`. The 30 zero-ref modules (list in census) go. Side effect: regenerate `docs/diagnostics/zarr_*_census.json` artifacts that enumerate some of them.

### 1.2 Top-level `src/*.py` — ~14,200 LOC
Provably uninstallable (`pyproject.toml:73-77` packages directories only) and zero imports repo-wide.
- **Tier A (zero refs of any kind, ~10,400 LOC):** 27 files from `detection_inspector.py` (740) down to `debug_tensorrt_model.py` (81), incl. `yolo_visualizer_vispy.py`, `export_detections_to_csv.py`, `test_kvikio_gds.py`.
- **Tier B (closed dead clusters):** `multi_roi_grating_analyzer.py`+`plot_csv_proportions.py` (3,068); `subdish_heatmap_analyzer.py`+`subdish_roi_mask_creator.py`+`roi_mask_creator.py` (742).
- **Tier C (docs-only refs, delete + one doc edit each):** 9 files, ~3,500 LOC.
- **Tier D (needs a companion edit):** `src/zarr_inspector.py` (+ dead fallback at `interactive_launcher.py:1826,1832`); `src/roi_heatmap_generator.py` (+ status string at `core/pipeline.py:1096`); `src/visualizer.py` (refs are false positives); `src/test_fisheye.py` (check pytest collection first).
- Also: `src/models/common.py` (178), `src/trt_cpp_predictor/extract_zarr_batch.py` (59), `src/best_possible.md`.
**Do NOT delete:** anything TensorRT-export related in `fisheye/` proper — TRT is load-bearing for realtime acquisition; the dead files above are standalone debug scripts only.

### 1.3 `scripts/` one-offs — ~1,000 LOC high-confidence
DELETE: `tmp_compare_keypoint_quality_metrics.py`, `tmp_consolidate_zarr_metadata.py`, `tmp_inspect_keypoint_review_attrs.py`, `check_mask_values.py` + `check_mask_probabilities.py` (Oct-2025, retired eye-mask stage). ARCHIVE: `eye_mask_severance_census.py` (severance complete), `run_moving_grating_downstream_pipeline.sh` (hardcoded January recording default). The other zero-ref scripts in the census are KEEP-pending (operator tools, live wrappers).

### 1.4 `src/fisheye/utils/` dead modules — ~450 LOC
`compare_analysis_training_zarr` (192), `export_protocol_mermaid` (306), `inspect_keypoint_review_linkage` (301), `clear_detection_flags` (328), `clean_refined_keypoint_runs` (175), `set_zarr_purpose_batch` (183), `register_training_run` (109), `set_crop_review_status` (100), `publish_accept_all_refined_detection` (92), `publish_crimson_canonical_v3_companion` (140, thin wrapper — verify), plus the two 6-line goodcopbadcop re-export shims.
**Trap:** `cleanup_external_ipc_rolling_recordings.py` looks zero-ref but was committed at HEAD 2026-08-20 — brand new, keep.
**Not dead:** the ~15 test-only operator CLIs (`apply_source_video_metadata_backfill`, `backfill_clipped_analysis_metadata`, …) — test-only is the normal state for `python -m` tools here; needs operator confirmation per file.

### 1.5 Eye-mask surface — policy correction + one deletion
**Correction to prior guidance:** `AGENTS.md:110-125` says eye-mask data is *legacy-compatibility* — read/migrate/validate is permitted; it is NOT delete-on-sight. The 11 raise-stubs in `db.py`/`maintenance.py` are intentional tripwires (tests assert they raise) — keep them [supersedes-twins item 1 is thereby downgraded]. The one genuine deletion: `diagnostics/preview_eye_mask_background_subtraction.py` (311) + its test — already on the severance plan's delete list (8 of 11 listed files already gone). `check_eye_mask_lineage._analyze_run_group` is live (imported by `prune_zarr_runs.py:9`) — keep.

### 1.6 `agents_todo/` — 14 of 20 briefs are spent
DELETE (each verified against a landed module/verb): both frame_domains design+impl briefs and their two successors, `curation_instance_keys`, three detect-pointer-retirement briefs, `registry_stale_rows_cleanup`, `registry_temp_store_guard`, `registry_reconcile_root_sweep`, `registry_reconcile`, `inventory_accessor_verb_port`, `registry_query_since_filter`, `promotion_authority`. KEEP: `utils_phase2`, `provenance_content_hashes`, `chaser_schedule_importer` (blocked on Citrus fixture), the two 07-05 wave briefs (pending audit).

### 1.7 Docs, configs, root strays — 62 files total (full tables in census)
- DELETE: `docs/diagnostics/storage_and_rig_conversation_2026-07-24.md` (raw chat transcript; both derived products exist as proper docs), `docs/diagnostics/git_history_cleanup_paths_2026-05-28.txt`, root `test.yaml`, `test_video_integrity.sh`, `pipeline_flowchart.html`, `src/best_possible.md`.
- ARCHIVE: 27 docs (9 spent `*_todo.md` checklists incl. `zarr_spec_runtime_drift_todo`, 5 self-declared-superseded contracts, `design_review_findings_2026-08-09.md` [self-marked implemented/superseded], 5 registry-cleanup JSON dumps, 3 census JSONs, misc notes); 4 configs (`detect_config_dan_talk_manual_rect`, `import_batman_training_canary`, `eye_segmentation_config`, `subject_mask_union_canary_20260406`); `check_frame_gaps.py`, `conda-packages-explicit.txt`+`pip-packages-exact.txt` (superseded env snapshots), `meeting_artifacts/chaser_analytics_20260805/`.
- Tests: only 3 files die with the eye-mask surface, and policy currently keeps that surface. Suite hygiene is good (4 xfails all gated on open work; 1 stale skip: `test_frame_domains.py:375`, /nvme1-dependent since 2026-07-05).

---

## Part 2 — Delete after repointing (superseded twins)

Ranked by LOC-deletable ÷ migration cost. Full table in census; the actionable core:

| Old | Survivor | Callers to move | Cost | LOC |
|---|---|---|---|---|
| `track_kinematics_io` legacy flat-speed readers (`:328-387`, `:708-862`) | grouped v2 path, same module | 0 (test-only) | S | ~215 |
| `inline_refresh.sync_latest_detection_profile_for_zarr` + `db.upsert_detection_data_profile` | `reconcile_dataset_from_root` [=B4] | 2 (`accept_detect_review.py:346`, `migrate_legacy_detect_labels.py:543`) | S/M | ~292 + test |
| 8 per-table `--backfill-*` loops (`maintenance.py:2486-3779`) | `reconcile_dataset_from_root` + `reconcile_sweep` | CLI flags only | L — **blocked on A4** (dry-run fix) | ~1,400 |
| `refresh_*_from_root` pair (`db.py:5344,5952`) | `_for_dataset` twins / `register_from_root` | 3 | M — after B2 | ~68 |
| 8 "resolve dataset row by zarr path" copies (2 byte-identical, 2 with *different* bare-vs-suffixed tie-breaks) | promote one into `registry/db.py` | ~8 | M (tie-break is a behavior choice) | ~130 |
| `_apply_recording_step_status_rows` | `status_ledger.upsert_recording_step_status` [=B1] | 2 | M — blocked on `"stale"` enum fix [=A3] | ~105 |
| `backfill_detection_profiles` + `backfill_keypoint_profiles` (zarr-attr writers, ~90% identical to each other, NOT registry duplicates) | merged `--kind` CLI (per 2026-06-20 utils review) | n/a | M | ~300 |

**Legacy reader-compat: what's still load-bearing (do not delete yet):** completion-epoch `legacy_default` (gated on C3/C4 — 26 of 28 callers already strict); sparse refined-detect subgroups (`migrate_refined_detect_sparse` must report zero remaining first); `refined_runs` fallback (6 constants, 18 sites — needs a store census). Deletable opt-ins: the `--legacy-*-compatibility` flag plumbing (~45 sites, explicit opt-in escape hatches), `reason`→`reason_bytes` ladder (self-liquidating), `full_image_xyxy` alias (2 sites).

---

## Part 3 — Extract once, collapse N sites

Ordered by bug-fix value first, then dedup value:

1. **`registry/connection.py`** — move `connect_read_only`/`connect_writable` out of `prune_stale_datasets` (a CLI module nobody imports). Collapses 5 private defs + ~13 inline `mode=ro` connects, and fixes **21 read-intent tools that currently connect writable** (dedupe plan mode, repair, goodcopbadcop runners, mask_tuner, task_generation…). [=A4/dry-run fix #2]
2. **`cli/shared_args` adoption** — 9 adopters today vs 187 hand-rolled `--apply`, 69 `--dry-run`, 113 `--registry`, 192 `--json`. **35 files accept `--apply --dry-run` together** with no mutual exclusion — migrate those 35 first (correctness), then the rest (tidiness).
3. **One zarr opener** — 57 private `_open_root`/`_open_group` wrappers (19 byte-identical to `open_zarr_group_direct`; `db.py:199` and `maintenance.py:68` duplicate *each other*; `zarr_helpers.py:540` duplicates its own module's public helper 131 lines down). Pick `zarr_io.open_zarr_root` as the single door; alias the other. Start with the 9 identical `_open_root` siblings in `analysis/chaser_*`. [supports C3]
4. **`Registry.mint_and_upsert_dataset`** — collapses the 5 mint-then-upsert prologues and the `_resolve_effective_dataset_id` vs public fork; prerequisite shape for closing the bare-vs-suffixed duplicate-row source. [=B2]
5. **`selected_calibration.resolve_camera_scale`** — the two full `pixel_to_mm` ladders are **inverses with different precedence** (`maintenance.py:3944` yields px/mm; `plot_sampled_component_contours.py:272` yields mm/px) and bypass the reciprocity guard that already exists at `selected_calibration.py:2946`. 4 sites → 1. [=D2]
6. **`registry/publish_verify.refresh_and_assert_identity`** — the three subject-metadata CLIs' shared tail, with the strictest identity assertion; also lift their triplicated `_open_root`/`_validate_published`/`_connect_read_only`. [=B6]
7. **`registry/write_gate.py`** — `PALETTE_DISABLE_REGISTRY_WRITES` is declared as a string in 6 consumers + 4 producers; one constant + `registry_writes_disabled()` makes the env contract greppable. [supports C1]
8. **`RegistryPaths` normalization** — `from_env` doesn't normalize, so 92 sites append `.expanduser().resolve()` (three different normalizations in `labeling/task_generation.py` alone). Normalize inside; delete the chains.
9. **`tests/conftest.py` fixtures** — `file_digest` + `registry_snapshot` (the volatile-filtered `_dump_rows`) + an `assert_unchanged` context manager. 5 named defs + ~23 inline idioms; also what the dry-run test wave needs.
10. **`scripts/lib/bsub_common.sh`** — 62 `submit_*_bsub.sh` scripts × ~40 lines of identical preamble; or route through the existing `fisheye.cluster` submitter.

**Module splits (defer until the above land; seam maps in census):** `maintenance.py` (9,489 — dominated by one 1,381-line function, `_build_recording_step_rows_from_root`) → 12 modules along its own section boundaries; `db.py` (8,582 — one 7,265-line class) → 6 mixins mirroring the existing `RegistryMigrationMixin` pattern; `migration_bodies.py` → per-version files. The other giants (`audit_coordinate_contracts` 16,856; `track_kinematics` 14,305; `labeling/web` 12,756) have seam markers recorded but lower priority.

---

## Suggested sequencing

1. **Pure deletions first** (Part 1) — no behavior change, ~27.5k LOC and 62 files, one review pass. Split into: (a) `chaser_analysis` + top-level `src/*.py`, (b) scripts/utils/briefs/docs. Gate each batch on the full test suite + import-linter.
2. **Extractions 1–2** (connection + shared_args) — these fix live bug classes and are prerequisites for the dry-run test wave.
3. **Twins with ≤2 callers** (Part 2 rows 1–2) — ~500 LOC for 2 edits.
4. **The big blocked items** ride their Wave-A dependencies: backfill-loop deletion after A4; step-status primitive merge after A3.
5. Module splits last — they're churn without the consolidations, and mechanical after them.

**Standing rule (proposed):** no new contract doc, registry table, or `--backfill-*` flag lands until this queue's Part 2 counter goes down by one. The specified-to-enforced ratio is the disease; this queue is the diet.
