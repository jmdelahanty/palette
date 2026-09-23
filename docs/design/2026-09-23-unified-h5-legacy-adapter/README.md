# Unified H5 → legacy stimulus-run adapter: field mapping

- **Status:** draft. Specification for an adapter that is not yet built.
  Drafted by an agent from code and fixtures; the blockers were spot-checked
  against the code (step-status gate, experiment-setup subject count) but the
  table has not been reviewed row by row.
- **Owner:** Jeremy Delahanty.
- **Last reviewed:** 2026-09-23.
- **Builds on:** [native unified H5 import](../../unified_h5_native_import.md)
  (PR 169), the registry native-run guard (PR 181), and the
  [recording identity design](../2026-09-23-recording-identity/README.md).
- **Targets:** unified core schema `development_core_definitions_not_production_admitted`;
  every row must be re-checked when Citrus freezes it.

Code base: main at `a43f21a6` (same code as `d0282143` for these files). Paths below are relative to `src/fisheye/` unless they start with `tests/`.

Evidence markers: **[V]** means verified by reading code, reading a fixture with h5py, or reading a real store artifact. **[I]** means inferred and not verified.

Real-data witnesses (all read-only):
- legacy raw H5 `recordings/2026-08-12T21-59-55Z_arena_2_goodbatbadbat/raw/*.h5` (Citrus chaser v4, 232 B rows)
- its Palette migration derivative `derived/stimulus_coordinate_migration/*.canonical_stimulus_v1.h5` (chaser v5 canonical). This is what the legacy importer actually consumes.
- the published run `analysis/stimulus_runs/stimulus_canonical_v1_a1e09d9d2d87e228` in that recording's analysis zarr.

Unified witnesses: the `base` and `appearance` fixtures in `tests/fixtures/unified_h5_v1`, decoded with `emit_fixture`.

---

## Summary

Read this first: **today the legacy "input" is not the raw Citrus H5.** It is a Palette-built canonical v5 derivative. `utils/migrate_legacy_goodbatbadbat_stimulus_h5.py` and `utils/migrate_legacy_batman_stimulus_h5.py` build it by adding:
- `stimulus_state_key`
- `source_acquisition_frame_index`
- the target-source acquisition arrays
- the row-identity contract
- the coordinate descriptor
- the surface manifest

The legacy importer (`analysis/import_stimulus_to_zarr.py` plus `shared/stimulus_coordinate_contract.py`, `selected_calibration.py`, `stimulus_physical_coordinate.py`, about 11k LOC) refuses anything less. The unified adapter's real job is therefore to replace that migration step. The recommended shape is "unified → canonical-v5-shaped derivative H5 → unchanged importer" (see design decision D1).

Main-table counts (53 legacy output rows: 52 classified + 1 composite chaser row):

| class | rows | notes |
|---|---|---|
| EXACT | 17 | path move or rename only (e.g. protocol snapshot, execution index, calibration mirror, arena geometry, events, enums, frame columns) |
| DERIVED | 28 | deterministic rule, mostly re-running existing legacy code on unified inputs; includes the whole canonical-coordinate layer and the camera-alignment helpers |
| MISSING | 5 | subject metadata (A1), experiment setup (A2), run-local subject refs (C1), display selected-output block (D18), stimulus video path (B4) |
| N/A | 2 | `tracking_data/bounding_boxes` (the legacy canonical path refuses non-empty boxes) and `stimulus_coordinates` (a historical path only) |
| chaser_states (composite row D7) | 56 legacy columns | see the per-column breakdown below |

chaser_states per column (legacy 56 columns / 232 B; unified 63 fields / 280 B = the same 56 plus 7 `*_valid` flags):
- 41 EXACT: same dtype.
- 8 EXACT with narrowing: unified u8 → legacy u1 (7 columns) or u4 (1 column). Needs a range check.
- 7 DERIVED: a validity flag collapses back to a legacy in-band sentinel. Several sentinels are unconfirmed, so these are Citrus questions.

NO-LEGACY-HOME: 30 unified nodes or families stay only in `native_h5` (list below).

**Verdict (blunt):** unified H5 cannot fully replace legacy H5 today. There are three hard blockers. A fourth becomes a blocker for any non-trivial session.
1. Subject metadata. `/metadata/subject` carries only `subject_id`. There is no `subject_count`, so `build_experiment_setup_record` raises, and every zebrobot field is gone.
2. Display evidence. There is no `/display_snapshot/selected_output_block` equivalent, so `_preflight_selected_calibration` fails.
3. Protocol execution status. Both fixtures are `status: "interrupted"`, and `_materialize_stimulus_steps` raises unless execution is `complete`.
4. Size budget. `MAX_DATASET_BYTES` = 64 MiB in `shared/unified_h5/common.py:20`. A real GoodBatBadBat session has 359,974 chaser rows × 280 B = 100.8 MB, so the native importer itself refuses real-size chaser tables before any adapter runs. **[V]** for the constant and the real row count; **[I]** that production unified files will have the same row count.

Also note: the core contract declares `status = development_core_definitions_not_production_admitted`, and the fixtures carry `development_schema_id = citrus.experimental_h5_core_writer_test`. This spec targets a schema Citrus has not yet frozen. **[V]**

---

## Main table

Legend for the consumers column: `N files` is a token-grep file count over `src/`. It excludes the importer, `unified_h5/`, and the migration tools. It is an approximate upper bound, because the token can also appear in non-stimulus contexts. The 92 files that mention `stimulus_runs` are the broad consumer universe.

### A. Zarr-level side effects (outside the run group)

| # | legacy output | legacy writer file:line | consumers | unified source | class | rule / notes |
|---|---|---|---|---|---|---|
| A1 | `analysis/subject_metadata_runs/<r>` (subject record: fish_id, subject_count, subject_type, genotype, line_strain, dpf, cross/dish ids, fish_count→source_dish_population_count, species, sex, parents…) | `analysis/import_stimulus_to_zarr.py:3106-3114` → `shared/subject_metadata.py:164` (`publish_subject_metadata`); also `utils/import_recording_analysis.py:439-472` (`import_experiment_setup`) via `read_h5_subject_metadata` `shared/subject_metadata.py:94` (reads `/subject_metadata` attrs) | `subject_metadata` 11 files, e.g. `utils/backfill_subject_context.py:161`, `zarr_inspector.py:150` | `/metadata/subject` attrs. Fixture has only `{subject_id: "synthetic-subject"}` **[V]** | **MISSING** | The legacy raw H5 carries 17 attrs, including `subject_count=1`, `fish_id` (UUID) and zebrobot lineage **[V]**. Unified renames `fish_id`→`subject_id` **[I]** and drops everything else. Legacy also had `/zebrobot_snapshot/snapshot_json`, which has no unified path. Citrus question Q1. |
| A2 | `analysis/experiment_setup_runs/<r>` (expected_subject_count, assignment status, source_dish_population_count) | `import_stimulus_to_zarr.py:3115-3124` → `shared/experiment_setup.py:119,286` | `experiment_setup` 9 files, e.g. `tracking/arena_assignment.py:764` | none | **MISSING** | `build_experiment_setup_record` requires `subject_count` (`experiment_setup.py:135-140`) **[V]**. Without it, setup publication raises and downstream falls back to legacy/inferred setup. The adapter must not invent `subject_count=1`. |
| A3 | `analysis/coordinate_frames/source_camera/<cam>/continuous` (source-camera physical mm authority) | `import_stimulus_to_zarr.py:3559-3569` → `shared/source_camera_physical_authority.py:516` | physical-mm consumers via `load_stimulus_physical_coordinate_authority` | `/geometry/calibration/<cam>/scale_models/*` (same attrs as legacy `/calibration_snapshot/<cam>/scale_models`) **[V]** | EXACT | Path move `/calibration_snapshot` → `/geometry/calibration`. The unified fixture also carries `tank_bottom_inner_surface` next to `projected_surface`. The legacy selector reads `projected_surface` only **[I]**. |
| A4 | `analysis/stimulus_runs` parent `latest`, `latest_complete` (+ `publication_generation`, lease) | `import_stimulus_to_zarr.py:3644` → `:2747-2843` (`activate_selector_eligible_run`, selector_attrs `("latest_complete","latest")`) | effectively all stimulus consumers resolve `latest`; registry `registry/extractors/stimulus_metadata.py:677` (`is_latest`) | n/a (native candidate is `stage_selector_eligible=False` and never touches selectors, `analysis/unified_stimulus_import.py:32-38,203-215`) **[V]** | DERIVED (policy) | The adapter run is the first selector-eligible run for a unified recording. Coexistence with a legacy-H5 run of the same recording is a design decision (D6). |

### B. Run-group attrs

| # | legacy output | writer | consumers | unified source | class | rule / notes |
|---|---|---|---|---|---|---|
| B1 | lifecycle: `palette_run_*`, `stage_selector_eligible`, `stimulus_publication_owner_uuid` | `import_stimulus_to_zarr.py:3161` (`mark_run_started`), `:3571` (`mark_run_complete`), `:2465-2740` (staging guard) | completion gate, registry | n/a | DERIVED | Produced by the adapter's own run lifecycle. |
| B2 | `run_provenance` (command, params incl. `import_version`, `repair_chaser_gaps`, `source_coordinate_policy`, `active_camera_id`, `selected_calibration_source_evidence_sha256`; `input_run_ids.source_h5`) | `import_stimulus_to_zarr.py:3571-3606` | provenance tooling | n/a | DERIVED | Must add adapter provenance (D4). |
| B3 | `source_h5`, `created_at_utc`, `import_version` (`"2.0.0"`, `shared/stimulus_coordinate_contract.py:109`), `coordinate_contract_epoch` (=1) | `import_stimulus_to_zarr.py:3494-3503`; epoch also `stimulus_coordinate_contract.py:3283` | `source_h5` 7 files. Three of them **re-open it as a legacy H5**: `visualization/overlay_arena_mask.py:178-189` (`/calibration_snapshot`), `utils/export_protocol_mermaid.py:45-52,93` (`/protocol_snapshot/*`), `analysis_workflows/protocol_semantic_chaser_selection.py:746-757` (requires `source_h5 == sealed raw_h5`, then `read_protocol_semantic_snapshot(raw_h5)`) **[V]** | n/a | DERIVED | Pointing `source_h5` at the unified file breaks those three consumers, because the legacy paths do not exist there. This is a strong argument for D1 (a legacy-shaped derivative H5). Even with D1, `protocol_semantic_chaser_selection` compares against a *sealed raw H5*, and that binding needs a decision (D7). |
| B4 | `source_stimulus_video_path` (`<h5>.mp4` next to the H5 if present) | `import_stimulus_to_zarr.py:3500-3502`, `:600-605` | 1 file: `utils/backfill_stimulus_video_paths.py:71` | none. `/video_metadata` is an empty group in both fixtures **[V]** | MISSING | Is a rendered stimulus video still produced alongside unified H5? See Q6. |
| B5 | `source_coordinate_policy`, `source_coordinate_surface_status` | `import_stimulus_to_zarr.py:3182-3197` | coordinate audit (`utils/audit_coordinate_contracts.py`) | n/a | DERIVED | Constant `canonical_required_v1` / `canonical`. |
| B6 | `protocol_json` | `import_stimulus_to_zarr.py:3389-3391` via `_read_protocol_snapshot` `:828-843` (`/protocol_snapshot/protocol_definition_json`) | `protocol_json` 17 files, e.g. `registry/extractors/stimulus_metadata.py:694`, `registry/extractors/chaser_metadata.py:93`, `analysis/stimulus_response.py:678`, `analysis/chaser_distance_runs.py:586` | `/protocol/authored/protocol_definition_json` **[V]** | EXACT | Byte-for-byte text. The registry extractors derive `protocol_name` from this payload, falling back to a run attr that the importer never writes (`stimulus_metadata.py:706-713`). `/protocol/authored@protocol_name` also exists **[V]**. |
| B7 | `arena_config_json` | `import_stimulus_to_zarr.py:3424-3432` | 5 files, e.g. `analysis/swim_bout_statistics.py:693`, `diagnostics/prepare_detect_training.py:606` | `/geometry/calibration/arena_config_json` **[V]** (keys incl. `experimental_chamber`, `selected_dish_type_name`, `active_camera_id`, `ipc_source_name`) | EXACT | Path move. |
| B8 | frame-gap stats: `gap_ranges`, `missing_frames`, `largest_gap`, `original_frames`, `interpolated_frames`, `interpolation_method`, `total_camera_frames`, `timestamp` | `import_stimulus_to_zarr.py:3265,3442-3443` (`analyze_frame_gaps`, `analysis/chaser_state_interpolator.py:87`) | `missing_frames` 16 files (token; many are other stages) | `/frames/stimulus.triggering_camera_frame_id` | DERIVED | Re-run `analyze_frame_gaps` on the unified frame rows. The gaps are trigger-ID gaps, not acquisition gaps (see D4 row). |
| B9 | `frame_metadata_interpolation_skipped(+_reason)`, `chaser_interpolation_skipped(+_reason)` | `import_stimulus_to_zarr.py:3268-3272, 3445-3480` | `analysis/chaser_state_interpolator.py:464` | n/a | DERIVED | With chaser states present, legacy skips both interpolations. **Without chaser states (OMR/loom/grating sessions), legacy runs `interpolate_metadata` and fabricates frame rows** (`:3274`). The adapter must not do this silently (D8). |
| B10 | `protocol_semantic_status/hash/snapshot_path`, `protocol_recipe_schema_id/version/step_count/mode_sequence/label` | `import_stimulus_to_zarr.py:846-935` (`_materialize_protocol_semantic_snapshot`) | `protocol_semantic_hash` 21 files; strict re-verification in `registry/extractors/stimulus_metadata.py:163-345` | `/protocol/authored/{protocol_semantic_json,protocol_semantic_hash,protocol_trial_index_json,protocol_trial_index_hash}` + attrs `schema_version`, `policy_id` **[V]** | EXACT | `shared/unified_h5/protocol.py:15-54` already calls the same `validate_protocol_semantic_snapshot` / `validate_protocol_execution_index` as legacy **[V]**, so the attrs come out bit-identical if the same code writes them. |
| B11 | `protocol_execution_status/hash/path`, `protocol_interval_axis`, `protocol_camera_frame_role`, `protocol_acquisition_containment_status`, `protocol_selector_eligibility` | `import_stimulus_to_zarr.py:938-1008` | registry extractor `stimulus_metadata.py:347-547`; `shared/protocol_execution_contract.py:439` | `/protocol/executed/{execution_index_json,execution_index_hash}` + attrs **[V]** | EXACT | **Gate:** both fixtures are `status: "interrupted"`. Legacy steps materialization raises for non-`complete` (`:1775-1781`), and it raises *inside* the import, so the whole run fails **[V]**. See D5. |
| B12 | `stimulus_renderer_snapshot_ref/_sha256` | `import_stimulus_to_zarr.py:1632-1669` | coordinate contract only | see D14 | DERIVED | Digest over the reshaped source. |
| B13 | `physical_coordinate_manifest_ref/sha256/publication_status/reason_code` + `calibration/<cam>/coordinate_frames/*` records | `import_stimulus_to_zarr.py:3486-3493` → `shared/stimulus_physical_coordinate.py` | physical-mm readers (`load_stimulus_physical_coordinate_authority`) | `/geometry/calibration/<cam>/scale_models/projected_surface` **[V]** | DERIVED | Re-run legacy publication on EXACT-moved inputs. |
| B14 | `chaser_states_coordinate_descriptor_status` | `shared/stimulus_coordinate_contract.py:3285,3848` | coordinate audit | n/a | DERIVED | `canonical` or `not_present`. |

### C. `source_metadata` group (run-local)

| # | legacy output | writer | consumers | unified source | class | rule / notes |
|---|---|---|---|---|---|---|
| C1 | `subject_metadata_ref/_sha256`, `experiment_setup_ref/_sha256`, `expected_subject_count` (or the legacy fallback: `subject_metadata` JSON + `experiment_setup`) | `import_stimulus_to_zarr.py:3200-3233` | `experiment_setup` / `subject_metadata` readers (see A1/A2) | none usable | MISSING | Depends on A1/A2. |
| C2 | `camera_metadata` JSON + `camera_config_hash` | `import_stimulus_to_zarr.py:3234-3236`, reader `:514-557` (`/camera_metadata`, `/device_metadata`, else `/recording_snapshot` → `cameras[camera_id]`) | 7 files, e.g. `utils/audit_swim_bladder_tuning_metadata.py:95`, `zarr_inspector.py:152` | `/metadata/recording/recording_snapshot_json` → `cameras[<binding.acquisition_camera_id>]` **[V]** (fixture: `{"CAM-42":{"frame_rate":100}}`) | DERIVED | Path move. Pick the camera by `/correspondence/acquisition/binding_json.acquisition_camera_id` instead of by regexing `ipc_source_name`. The payload is thin in the fixture, so its richness on real rigs is unverified (Q7). |
| C3 | `session_context` JSON (root attrs: session_uuid, session_start_iso8601_utc, rig_id, arena_id, camera_id, canvas_name, protocol_name_from_definition, loaded_protocol_filepath, stimulus_output_width/height, ipc_source_name, active_ipc_source, hostname, software_version) | `import_stimulus_to_zarr.py:3237-3239`, reader `:608-638` (reads **root attrs**) | `session_context` 9 files | `/metadata/session` attrs **[V]** (all keys present except `camera_id` and `ipc_source_name`); `ipc_source_name` is in `arena_config_json`; camera from binding | DERIVED | Move root attrs → `/metadata/session`. `camera_id` comes from the binding, not from the `cam_(\d+)` regex fallback (`:586-597`). The legacy regex would silently derive a wrong ID from `ipc_source_name`. |
| C4 | `session_uuid` | `import_stimulus_to_zarr.py:3240-3242` (root attr) | `session_uuid` 42 files (token, broad) | `/metadata/session@session_uuid` **[V]** | DERIVED (semantic change) | Format changed. Legacy is `2026-08-12T21-59-55Z_arena_2`; unified fixture is `citsess_70…`. Any consumer that parses or joins on the legacy form will break **[I]**. Q8. |
| C5 | `experimental_chamber` | `import_stimulus_to_zarr.py:3243-3245`, reader `:564-583` | 5 files, e.g. `tracking/arena_assignment.py:764` | `arena_config_json.experimental_chamber` **[V]** | EXACT | Legacy already falls back to `arena_config_json`. The root attr is gone in unified. |
| C6 | `dish_design` | `import_stimulus_to_zarr.py:3246-3250` | `dish_design` 38 files (mostly training naming) | `arena_config_json.selected_dish_type_name` **[V]** | EXACT | |

### D. Arrays and child groups under the run

| # | legacy output | writer | consumers | unified source | class | rule / notes |
|---|---|---|---|---|---|---|
| D1a | `video_metadata/frame_metadata/stimulus_frame_num` (u8) | `import_stimulus_to_zarr.py:3281-3296` (`write_columnar_dataset`) | `frame_metadata` 15 files, e.g. `analysis/chaser_distance_runs.py:238-242`, `analysis/swim_bout_statistics.py:495-500`, `analysis/diagnose_frame_alignment.py:31` | `/frames/stimulus.stimulus_frame_num` <u8 **[V]** | EXACT | |
| D1b | `…/triggering_camera_frame_id` (u8) | same | 47 files (token) | `/frames/stimulus.triggering_camera_frame_id` <u8 **[V]** | EXACT | Unified says explicitly: "recorded trigger ID; not authoritative Orange acquisition correspondence". Legacy consumers treat it as a camera frame. The fixture shows trigger 900–903 against recording frames 1–2 **[V]**. Semantics are unchanged from legacy, but the caveat is now explicit. |
| D1c | `…/timestamp_ns_epoch` (i8) | same | 14 files | `/frames/stimulus.timestamp_ns_epoch` <i8 **[V]** | EXACT | Citrus wall clock at state-output log, which can jump. Same as legacy **[I]**. |
| D1d | `…/video_frame_index` (legacy i8, always 0..N-1 in the real file) | same | 0 files outside `chaser_analysis/` **[V]** | `/frames/stimulus.video_frame_index` <u8 + `video_frame_index_valid` **[V]** (all invalid in both fixtures) | DERIVED | `valid ? int64(v) : -1` would be a **Palette-invented sentinel**, because legacy never had invalid rows. Low consumer risk. Q6. |
| D1e | `frame_metadata` group attrs `schema_version` (=2, copied from H5), `interpolated`, `original_records`, `total_records` | `import_stimulus_to_zarr.py:3282-3291` | `inspect_frame_alignment`, `diagnose_*` **[I]** | `/frames/stimulus` attrs have `schema_id=citrus.experimental_h5.stimulus_frame`, `schema_version=1`, `source_schema_version=3` **[V]** | DERIVED | Decide whether to stamp legacy `schema_version=2` (the legacy wire meaning) or to carry the unified identifiers. Nobody was found to branch on it **[I]**. |
| D2 | `video_metadata/camera_aligned_frame_metadata` (+attrs) | `import_stimulus_to_zarr.py:3298-3322`, `_build_camera_aligned_metadata` `:649-668` | 1 file: `analysis/chaser_distance_runs.py:242` | derived from D1 | DERIVED | Same function. |
| D3 | `interpolation_mask`, `camera_aligned_interpolation_mask` | `import_stimulus_to_zarr.py:3324-3340` | 6 files, e.g. `analysis/chaser_state_interpolator.py:464`, `visualization/visualize_refined_online.py:63` | derived | DERIVED | All-True when chaser rows exist. See B9/D8 for non-chaser sessions. |
| D4 | `frame_alignment@camera_frame_offset` (=min trigger ID), `frame_alignment/camera_to_metadata_index` (i8), `frame_alignment/camera_interpolation_mask` | `import_stimulus_to_zarr.py:3342-3355`, `_compute_camera_alignment` `:671-697` | 5 files, e.g. `analysis/diagnose_camera_chaser_mapping.py:191-192`, `analysis/plot_chaser_alignment.py:59-60`, `diagnostics/inspect_frame_alignment.py:97` | derived from `/frames/stimulus.triggering_camera_frame_id` | DERIVED | Reproduce the trigger-ID-based legacy semantics exactly. **Do not** substitute `/correspondence/frames/sources.source_acquisition_frame_index`: its base differs (0-based acquisition vs trigger ID 900+) and it covers only mapped frames (2 of 4 in the fixture **[V]**). |
| D5 | `enums/<name>/{id (i4), name (UTF-8)}` + `field_names`, `storage_layout` | `import_stimulus_to_zarr.py:760-826` | 6 files, e.g. `visualization/visualize_experiment_timeline.py:514`, `utils/inspect_zarr_events.py:27` | `/definitions/enums/<name>` compound72 (`id` <i8, `name` utf8[64]) **[V]** | EXACT | Narrowing i8→i4 with a range check (legacy already casts to int32, `:791`). The unified set adds 8 `appearance_*` enums **[V]** and `events` has 121 entries vs 119 in the real legacy file **[V]**. Pass the extra tables through, which is harmless because they are columnar groups keyed by name. |
| D6 | `events/<field>` columnar: `timestamp_ns_epoch` i8, `timestamp_ns_session` i8, `event_type_id` i4, `current_step_index` i4, `event_name`, `stimulus_frame_num` u8, `camera_frame_id` u8, `reserved` u1[8], `name_or_context`, `stimulus_mode_id` i4, `details_json` (+group attrs) | `import_stimulus_to_zarr.py:3362-3388` | `events` 18 files, e.g. `visualization/plot_detection_epoch_heatmaps.py:255`; also step materialization | `/events/records` compound1480 **[V]** | EXACT | Narrow `event_type_id`, `current_step_index`, `stimulus_mode_id` from i8 to i4 with range checks. `event_row_index` is dropped: it is the row ordinal. Unified text fields are null-terminated UTF-8 and legacy is `|S`. `_ensure_utf8_column` already normalizes both. **Caveat:** unified documents `camera_frame_id` as "zero may be default; not acquisition correspondence", and the appearance fixture's STEP_START/STEP_END rows carry `camera_frame_id=0` **[V]**. See D10. |
| D7 | `tracking_data/chaser_states/<56 columns>` | `import_stimulus_to_zarr.py:3357-3361` → `_copy_h5_dataset` `:726-757` → `write_columnar_dataset` | 15 files reference `chaser_states`; per-column counts in the sub-table | `/components/chaser/states` compound280 (63 fields) **[V]** | mixed | See the per-column sub-table. Row order is append order in both **[V]**. |
| D8a | `tracking_data/chaser_states/stimulus_state_key` (i8, N×2, components `[chaser_index, stimulus_frame_num]`) + `row_identity_contract` attrs | `shared/stimulus_coordinate_contract.py:3380-3395` (copied from source `/tracking_data/stimulus_state_key`, which the migration tool builds) | 8 files, e.g. `analysis/chaser_metrics_loader.py:188` | `/components/chaser/states` (`chaser_index`, `stimulus_frame_num`); `key_fields="chaser_index,stimulus_frame_num"` **[V]** | DERIVED | `column_stack([chaser_index, stimulus_frame_num]).astype('<i8')`. The component order matches legacy **[V]**. Build the contract with `build_row_identity_contract(domain="stimulus_state")`, as `utils/migrate_legacy_goodbatbadbat_stimulus_h5.py` does. |
| D8b | `…/source_acquisition_frame_index` (i8, ≥0, no validity) + sealed `source_acquisition_mapping_record` (acquisition_recording_id, acquisition_camera_id, source_total_frames, …) | `stimulus_coordinate_contract.py:3396-3420`; source contract `:1173-1283` | `source_acquisition_frame_index` 184 files (token, shared with tracking stages); via `load_bound_stimulus_coordinate_evidence` 6 files, e.g. `analysis/chaser_distance_coordinate_publication.py:607` | `/correspondence/chaser/sources.source_acquisition_frame_index` (+`_valid`), with recording id / camera from `/correspondence/acquisition/binding_json` and total from `source_semantic_record_json` (`_v6_validate_orange_source_record`) **[V]** | DERIVED | Admission guarantees every chaser row has `source_acquisition_frame_valid == 1` and `index == recording_frame_id - 1` (`shared/unified_h5/correspondence.py:329-337`) **[V]**. The copy is therefore lossless. The adapter should still assert all-valid and refuse otherwise, because legacy has no validity array for current-source rows. |
| D8c | `…/target_source_acquisition_frame_index` (i8) + `…_valid` (bool) + record | `stimulus_coordinate_contract.py:~3480-3530`, source loader `:1286-1450` | same as D8b | `/correspondence/chaser/sources.target_source_acquisition_frame_{index,valid}` **[V]** | DERIVED | Direct copy. Invalid rows carry 0, consistent with legacy. The legacy loader also cross-checks `target_source_frame_id` / `target_source_camera_id` columns. This conflicts with the sentinel issue in the sub-table, so it must be verified against the loader. |
| D8d | `…/camera_frame_ids` (i8), `…/source_row_indices` (i8) + `camera_mapping_record` | `stimulus_coordinate_contract.py:3571-3637` (`_camera_mapping_inputs`) | `camera_frame_ids` 17 files (token) | derived from D1/D7 | DERIVED | Same legacy code. It maps each stimulus row to its trigger ID through frame_metadata. |
| D8e | `…/{chaser_position_xy,target_position_xy,target_clamped_position_xy}` + `coordinate_descriptor`, `coordinate_surface_manifest`, `coordinate_import_lineage`, `coordinate_output_manifest`, frame-transform / temporal-authority records | `stimulus_coordinate_contract.py:3274-3850` | `load_bound_stimulus_coordinate_evidence` (6 files), `utils/audit_coordinate_contracts.py` | chaser attrs `coordinate_frame=arena_relative_canvas_px`, `coordinate_origin=top_left_of_active_arena`, x right / y down, `units=px` **[V]** (same as legacy v4 attrs **[V]**) | DERIVED (Palette-authored) | Citrus never wrote the descriptor or surface manifest in legacy either; the migration tool authored them. The adapter authors them the same way (`build_canonical_coordinate_descriptor`, profile `arena_relative_canvas_px.top_left_y_down.v1`, reference extent from `arena_geometry` width/height). This is a Palette claim, so record it in the provenance. |
| D9 | `tracking_data/bounding_boxes` (+ computed `centroid_x/centroid_y`) | `import_stimulus_to_zarr.py:3360`, `:736-749` | 5 files (token) | `/observations/bounding_boxes` compound76 **[V]** | N/A | **The legacy canonical preflight refuses non-empty stimulus bounding boxes** (`stimulus_coordinate_contract.py:2623-2629`) **[V]**. No canonical run has them. The adapter must omit them, or the import fails. The base fixture has 2 rows. They stay NO-LEGACY-HOME. |
| D10 | `steps` group attrs + `steps/step_N` attrs: `step_index`, `step_name`, `stimulus_mode_id`, `stimulus_mode`, `duration_s`, `raw_protocol_params_json`; **either** `start_camera_frame`/`end_camera_frame` (no execution index) **or** `start_stimulus_frame_inclusive`, `end_stimulus_frame_exclusive`, `first/last_camera_frame_id_correspondence`, `execution_completion_status`, `execution_end_reason`; plus semantic binding attrs (`protocol_semantic_step_index/_ref`, `stimulus_family`, `display_context`, `resolved_color_rgba8`, …) | `import_stimulus_to_zarr.py:1757-2053`, binding `:1220-1265` | `"steps"` 20 files; `start_camera_frame` 15 files (e.g. `analysis/stimulus_response.py:632`, `analysis/stimulus_response_io.py:241`, `reporting/discovery.py:258`); `start_stimulus_frame_inclusive` 6 files (`analysis_workflows/protocol_semantic_chaser_selection.py:577`) | `/protocol/executed/execution_index_json` + `/protocol/authored/*` + `/events/records` STEP_START/STEP_END | DERIVED | Legacy requires STEP_START/STEP_END events that cover every recipe step with matching mode IDs (`:1839-1860`), plus a `complete` execution. **Mode split matters:** with an execution index present (always the case for unified), legacy writes stimulus-frame intervals and **no `start_camera_frame`/`end_camera_frame`**. The 15 consumers that read `start_camera_frame` then fall back or break. That is legacy behavior for v2 snapshots too **[V]**. Do **not** backfill camera bounds from event `camera_frame_id` (it can be 0 in unified **[V]**). |
| D10b | `steps/step_N/moving_grating` or `/concentric_grating` attrs (authored orientation, speeds, spatial freq, temporal freq, `direction_mapping_validated=False`, …) | `import_stimulus_to_zarr.py:1671-1754, 2039-2046` | `moving_grating` 8 files, `concentric_grating` 5 files | protocol params from `/protocol/authored/protocol_definition_json` | DERIVED | Same code. Note that `/components/moving_grating/states` records realized canvas direction and `render_axis_degrees_canvas` per frame. Legacy ignores this and keeps `direction_mapping_status=unvalidated_default_zero_offset`. That is a scientific upgrade opportunity, not an adapter concern. |
| D10c | `steps/step_N/execution_phases/<phase>` | `import_stimulus_to_zarr.py:2012-2037` | 1 file + registry extractor `stimulus_metadata.py:452-470` | `execution_index_json.steps[i].chaser_phases` **[V]** (present in the appearance fixture) | DERIVED | Same code. |
| D11 | `protocol_semantic_snapshot/{protocol_semantic_json_utf8, protocol_trial_index_json_utf8}` + attrs | `import_stimulus_to_zarr.py:876-935` | 9 files; registry extractor strict check | `/protocol/authored/*` **[V]** | EXACT | Byte-exact. |
| D12 | `protocol_execution/execution_index_json_utf8` + attrs | `import_stimulus_to_zarr.py:954-987` | `protocol_execution_contract.py:439`, registry extractor | `/protocol/executed/execution_index_json` **[V]** | EXACT | Byte-exact (status gate B11). |
| D13 | `protocol_execution/frame_correspondence_proxy/{stimulus_frame_num, camera_frame_id_correspondence, …}` + manifest attrs | `import_stimulus_to_zarr.py:1025-1069` | registry extractor `stimulus_metadata.py:483-535` | derived from `/frames/stimulus` (`stimulus_frame_num`, `triggering_camera_frame_id`) | DERIVED | Same builder (`build_protocol_frame_correspondence_proxy_payload`). |
| D14 | `stimulus_renderer_snapshot/<arena>/{attrs, custom_coordinates}` + schema/record attrs | `import_stimulus_to_zarr.py:1632-1669`; classification `stimulus_coordinate_contract.py:546-700` | coordinate contract (0 direct readers) | appearance fixture: `/geometry/renderer/<arena>/custom_coordinates` with identical layout and attrs **[V]**; base fixture: absent, only the flattened `/geometry/correspondence/input/renderer` attrs **[V]** | EXACT (when `/geometry/renderer` exists) | If only the flattened form exists, reshaping it is a DERIVED fallback. The flattened form adds `evidence_kind=supplied_static_snapshot_claim`. The two producer commits disagree on whether `/geometry/renderer` exists (Q9). |
| D15 | `stimulus_coordinates` (historical renderer copy) | `import_stimulus_to_zarr.py:1623-1629` | 1 file: `analysis/chaser_quadrant_occupancy.py:445` | none | N/A | Legacy-historical path. Canonical v5 uses D14. |
| D16 | `calibration/<cam>/*` mirror (homography_matrix + attrs, homography_matrix_yml, png buffers, scale_models/*) + stamped selected-calibration manifest (`calibration@active_camera_*`, `selected_calibration_manifest`), `camera_to_selected_canvas_authority`, `selected_canvas_to_source_camera` | `import_stimulus_to_zarr.py:377-433` (`_materialize_selected_calibration_snapshot`) → `shared/selected_calibration.py` stamping | `calibration` 33 files (token), e.g. `visualization/overlay_arena_mask.py:45`, `shared/calibration.py` | `/geometry/calibration/<cam>/*` **[V]** (same node set and attrs as legacy `/calibration_snapshot/<cam>`) | EXACT (inputs) | Path move. Stamping is re-run legacy code, but it cannot complete without D18. **[V]** that `/geometry/correspondence/input@calibration_authority_ref` still names `/calibration_snapshot/CAM-42/homography_matrix` and `runtime_geometry_contract_ref` names `/runtime_geometry_contract/contract_json`: legacy paths that do not exist in the unified file (Q10). |
| D17 | `calibration/arena_geometry` attrs (+ `arena_geometry_record`) | `import_stimulus_to_zarr.py:398-404` | 9 files, e.g. `analysis/chaser_distance_runs.py:281-290`, `visualization/goodcopbadcop_interactive.py:773` | `/geometry/calibration/arena_geometry` attrs **[V]** | EXACT | Path move. |
| D18 | `display_snapshot` (attrs + `selected_output_block` text: xrandr output name, connection, `WxH+X+Y`, transform) | `import_stimulus_to_zarr.py:419-422`; **required** by `shared/selected_calibration.py:1520-1660` (`SOURCE_DISPLAY_DATASET_PATH="/display_snapshot/selected_output_block"`) | `shared/selected_calibration.py:154,303`, `utils/audit_coordinate_contracts.py:3775` | `/metadata/display` has **no** `selected_output_*` fields. The fixture has `capture_status=no_display_env` and an `nvidia_gpu_inventory_csv` **[V]**. Partial substitutes exist: `/geometry/correspondence/input@final_display_{width,height}_px`, `/geometry/presentation/contract_json` | **MISSING** | Hard blocker for selected-calibration stamping, and therefore for canonical coordinates, physical mm, and selector activation. Either Citrus records the selected output block in unified, or Palette's selected-calibration contract learns a presentation-contract display source (a Palette code change, not an adapter). Q2. |
| D19 | run-level `coordinate_frames/{arena_geometry_xywh, arena_relative_canvas, selected_canvas}`, `transforms/arena_to_selected_canvas(+_authority)` | `shared/stimulus_coordinate_contract.py` / `stimulus_frame_transform.py` (called via `:3638-3650`) | coordinate evidence loaders | derived from D16/D17/D18 | DERIVED | Same code; blocked by D18. |

### D7 sub-table: `chaser_states` column by column

Coordinate frame is identical in both: `arena_relative_canvas_px`, top-left of active arena, x right, y down, px. mm fields are "projected-surface mm derived from active pixels_per_mm". Both are declared by identical chaser dataset attrs in legacy v4 and unified **[V]**. Timestamps: `timestamp_ns_session` is Citrus `std::chrono::steady_clock` since session origin, not photon time, in both **[V]** (unified documents it; legacy inferred to be the same producer field).

| legacy column (dtype) | unified field (dtype) | class | rule |
|---|---|---|---|
| stimulus_frame_num (u8) | same (<u8) | EXACT | |
| timestamp_ns_session (i8) | same (<i8) | EXACT | |
| chaser_index (u1) | chaser_index (<u8) | EXACT-narrow | assert < 256 |
| is_chasing (u1) | same | EXACT | |
| chaser_pos_x/y, target_pos_x/y (f4 ×4) | same | EXACT | |
| target_source_frame_id (u8) | target_source_frame_id (<u8) + `_valid` | DERIVED | `valid ? v : 0`. Legacy 0 = unknown **[I]** (the real file never had 0; min 31426). Refuse if valid && v == 0. Q3 |
| target_source_camera_id (u4) | target_source_camera_id (<u8) + `_valid` | DERIVED | `valid ? v : 0`, assert < 2^32. Unified doc says "old Chaser camera zero was unknown". Legacy real data = 1 on every row; fixtures = invalid on every row **[V]**. Need Citrus to confirm the same numbering (Q3). |
| target_source_box_index_in_payload (u1) | (<u8) + `_valid` | DERIVED | `valid ? v : ?`. **No legacy sentinel known** (real data always 0, and 0 is a valid index). Unrecoverable ambiguity. Q3 |
| target_age_ms (f4) | (<f4) + `_valid` | DERIVED | `valid ? v : ?`. Legacy sentinel unknown (0.0 appears on fresh rows). Q3. 0 consumers **[V]** |
| target_freshness_state (u1) | (<u8, enum) | EXACT-narrow | enum ids identical (`chaser_target_freshness_states` 0..3) **[V]** |
| target_area_state (u1) | (<u8, enum) | EXACT-narrow | |
| target_clamped_pos_x/y (f4 ×2) | same | EXACT | 4 consumers |
| target_distance_outside_px (f4) | (<f4) + `_valid` | DERIVED | `valid ? v : -1.0`. **[I-strong]**: in the real file exactly the 1,062 `EXPIRED_OR_MISSING` rows carry -1.0 |
| chaser_radius_px/mm, target_radius_px/mm, distance_to_target_px/mm, chase_speed_px_per_s/mm_per_s (f4 ×8) | same | EXACT | |
| behavior_program_active, behavior_episode_active (u1 ×2) | same | EXACT | |
| behavior_episode_id (u8) | same | EXACT | |
| behavior_phase_index (i4) | (<u8) + `_valid` | DERIVED | `valid ? int32(v) : -1`. The real file is -1 on every row **[V]**; the fixture is valid=1, value 2 **[V]**. Assert v < 2^31. 0 consumers |
| behavior_motion_type_id (u1) | (<u8, enum) | EXACT-narrow | |
| behavior_velocity_x/y_mm_per_s, behavior_command_speed_mm_per_s (f4 ×3) | same | EXACT | |
| behavior_retreat_plan_active (u1) | same | EXACT | |
| behavior_retreat_{requested,actual}_distance_mm, …endpoint_x/y_mm, …target_x/y_mm, …angular_deviation_deg (f4 ×7) | same | EXACT | |
| visual_angle_deg, angular_velocity_deg_s, tau_ms (f4 ×3) | same | EXACT | 5/8/1 consumers |
| loom_mode, loom_phase, chaser_behavior_class_id (u1 ×3) | (<u8, enum) | EXACT-narrow | |
| l_over_v_ms, initial_distance_mm, max_angle_deg, z_eff_mm, pixels_per_mm (f4 ×5) | same | EXACT | |
| trial_state (u1) | (<u8, enum) | EXACT-narrow | 4 consumers |
| chase_sequence_active (u1) | same | EXACT | |
| chase_trial_id (u8) | (<u8) + `_valid` | DERIVED | `valid ? v : 0`. In the real file, 0 = no trial on 350,969/359,974 rows **[V]**. Refuse if valid && v == 0. 4 consumers (e.g. `analysis/chaser_escape_freeze_summary.py:191,296`) |
| time_in_state_s (f4) | same | EXACT | |

Totals: 41 EXACT, 8 EXACT-narrow, 7 DERIVED. The collapse is lossy for `target_source_box_index_in_payload` and `target_age_ms` until Citrus confirms the sentinels. The unified validity flags stay authoritative in `native_h5`.

---

## NO-LEGACY-HOME (stays in `native_h5`; the adapter must not project it)

1. `/frames/stimulus.timestamp_ns_session`: per-frame session-clock time. Legacy frame_metadata has no session clock. A new column is optional (D9 decision below).
2. `/frames/stimulus.video_frame_index_valid` (collapsed by D1d).
3. The 7 `chaser/states *_valid` flags (collapsed by the D7 sub-table).
4. `/correspondence/frames/sources` (Orange 1-based recording frame id + acquisition index per stimulus frame). This is the authoritative stimulus→acquisition map for *frames*. Legacy has only the per-chaser-row map (D8b). This is where legacy's trigger-ID `camera_frame_offset` is known to be approximate.
5. `/correspondence/acquisition/{binding_json, raw_live_identity_provenance_json, source_semantic_record_json}` and `/correspondence/receipt_json`. Used to derive D8b records; the full evidence stays native.
6. `/correspondence/{moving_grating,independent_motion_grid}/sources`.
7. `/components/visual_appearance/states` + `replay_dependency_manifest_json` (object-appearance witness, 407 B rows).
8. `/definitions/enums/appearance_*` (8 tables; they may pass through, see D5).
9. `/components/moving_grating/states`. Legacy never imported realized grating state; only authored step params (D10b).
10. `/components/independent_motion_grid/states`. Same.
11. `/observations/bounding_boxes` (the legacy canonical import refuses them, D9).
12. `/observations/region_routing`, `/observations/region_candidates`.
13. `/timing/display_submissions/frame_submissions` (compose/swap CPU times, marker codes).
14. `/timing/clock_domains/*` (steady↔system correlation; exposure/DAQ sync status).
15. `/trials/trial_index` (runtime trials). Legacy never imported `/trials` either **[V]**.
16. `/protocol/planned/chaser/*` (chaser schedule, planned onsets; legacy `/chaser_schedule`). Legacy never imported it **[V]**. A Palette importer and A0 binding for the chaser schedule are pending separately.
17. `/geometry/runtime/*` (runtime geometry contract + pointers + readiness).
18. `/geometry/presentation/contract_json` (legacy `/presentation_mapping`, not imported).
19. `/geometry/correspondence/{authority_json, input/*, receipt_json}` (geometry readiness / authority; the renderer part feeds D14's fallback).
20. `/recording_geometry_contract/*`.
21. `/definitions/source_namespaces/current_geometry_json`.
22. `/geometry/calibration/{calibration_pattern, tank/*}` (legacy also has these but does not import them **[V]**).
23. `/geometry/calibration/<cam>/scale_models/tank_bottom_inner_surface` (second plane).
24. `/metadata/rig/*` (commissioning pointers, photodiode marker json).
25. `/metadata/recording_association/{claims_json, receipt_json}` (identity claims contract).
26. `/metadata/completion_json`, `/metadata/component_outcomes_json`.
27. `/metadata/user_session`, `/metadata/cameras/camera_metadata/<cam>` attrs (configured frame rate). Part of the latter could feed C2.
28. `/evidence/integrity/*`, `/evidence/recording_binding/*`.
29. `/events/records.event_row_index` (implicit ordinal).
30. Unified per-table `schema_id`, `schema_version`, `source_schema_version`, `*_ref` attrs on every table.

---

## MISSING / questions for Citrus

- **Q1 (blocker).** Subject metadata. Unified `/metadata/subject` carries only `subject_id`. Where do `subject_count`, `subject_type`, and the zebrobot lineage go (cross_id, dish_id, fish_count/source_dish_population_count, genotype, line_strain, dpf, date_of_fertilization, parents, sex, species, queried_at_utc)? Is `/zebrobot_snapshot/snapshot_json` intentionally dropped? Is `subject_id` the old `fish_id`?
- **Q2 (blocker).** Display selected output. Will unified record the xrandr selected-output block (name, connection, `WxH+X+Y`, transform token/raw) that legacy `/display_snapshot` carried? If not, which unified node is the display-geometry authority, and will Palette's selected-calibration contract accept it?
- **Q3.** Chaser sentinels. Confirm the legacy in-band value for each validity-flagged field:
  - `target_source_frame_id` (0?)
  - `target_source_camera_id` (0 = unknown? same numbering as legacy 1?)
  - `target_source_box_index_in_payload` (none known)
  - `target_age_ms` (none known)
  - `target_distance_outside_px` (-1.0)
  - `behavior_phase_index` (-1)
  - `chase_trial_id` (0)

  Can a valid row ever carry the sentinel value?
- **Q4 (blocker for fixtures, possibly common in production).** Execution status. Both fixtures are `interrupted`. Will production sessions routinely finalize `complete`? What should happen to interrupted sessions? Legacy refuses them outright.
- **Q5.** Size. Will production unified chaser tables exceed Palette's 64 MiB per-dataset native budget? A real 2-chaser session is about 101 MB at 280 B/row. This is a Palette budget, but Citrus's row size drives it.
- **Q6.** Stimulus video. Is the rendered stimulus `.mp4` still produced, and does `video_frame_index` get populated in production (all invalid in the fixtures)?
- **Q7.** `recording_snapshot_json.cameras[<id>]` richness on real rigs (the legacy `camera_metadata` payload was used for tuning audits).
- **Q8 (answered 2026-09-23).** `session_uuid` (`citsess_…`) is intentionally the per-Arena Citrus session, one per H5; legacy `<timestamp>_arena_N` was also per arena, so the meaning is unchanged and only the format differs. It is **not** the acquisition session: that is `/correspondence/acquisition/binding_json.recording_id` (see the recording identity design).
- **Q9.** `/geometry/renderer` exists in the appearance fixture but not the base fixture. Is it required in production?
- **Q10.** `/geometry/correspondence/input@calibration_authority_ref` and `@runtime_geometry_contract_ref` point to legacy paths (`/calibration_snapshot/...`, `/runtime_geometry_contract/...`) that do not exist in a unified file. Is that a bug or an intentional legacy-name reference?
- **Scientific decision (not Citrus): decided 2026-09-23, see D12.** Legacy `camera_frame_offset` / `camera_to_metadata_index` index by trigger ID; unified provides an authoritative acquisition map (NO-LEGACY-HOME #4).

---

## Design decisions for the adapter

- **D1. Architecture.** Recommended: **unified (read from the native candidate via `load_unified_stimulus_candidate`) → write a canonical-v5-shaped derivative H5 → run the unchanged `import_stimulus_to_zarr`.** This mirrors `utils/migrate_legacy_goodbatbadbat_stimulus_h5.py`, whose job it replaces. Why:
  - it reuses the ~11k LOC coordinate, calibration, physical, selector and publication contract verbatim;
  - `source_h5` keeps pointing at a file with legacy paths, so the three re-open consumers keep working (B3);
  - consolidation and verification stay identical.

  The alternative, a direct zarr writer, would re-implement `materialize_stimulus_coordinate_contract`, selected-calibration stamping and activation, and it would drift. Cost: one derivative H5 per recording, about 100 MB for a chaser session (the storage crunch is noted), plus an extra format to keep alive. Put it under `<recording>/derived/unified_legacy_adapter/` with a receipt, like the migration tool.
- **D2. Registry guard vs `source_profile`.** Registry extractors skip any run where `run.attrs["source_profile"] is not None` (`registry/extractors/stimulus_metadata.py:48-53`; `chaser_metadata.py:79`). The adapter-produced run **must not** set `source_profile` on the run group, or the guard must learn an allowlisted adapter profile. Pick one explicitly:
  - (a) no `source_profile`; carry provenance in new attrs such as `adapter_id`, `adapter_version`, `native_source_run`, `native_manifest_sha256`, `native_source_profile="unified_experimental_h5_v1"`. The guard stays unchanged.
  - (b) `source_profile="unified_legacy_adapter_v1"` + a guard allowlist.

  Recommend (a). The guard's docstring says "legacy v5/v6 runs never declare source_profile", and (a) keeps that true. Either way, write a test proving the extractor reads the adapter run and still lists the native candidate under `unsupported_native_runs`.
- **D3. What the importer sees as `source_h5`.** Under D1 it is the derivative H5. Record the native lineage (native run name, `native_manifest_sha256` from `unified_stimulus_import.py:193`, source file sha256 from the native admission) in the derivative's receipt *and* in run attrs, and ideally in `run_provenance.input_run_ids`. The importer's `run_provenance` today carries only `source_h5` and the video path (`import_stimulus_to_zarr.py:3592-3597`). This needs a small importer change or a post-completion attr (post-completion writes conflict with immutability, so prefer the importer change).
- **D4. Adapter provenance.** At minimum:
  - adapter version;
  - native run name + `native_manifest_sha256`;
  - native `source_sha256` and finalization receipt id;
  - `native_source_profile`;
  - the explicit sentinel table used for the 7 DERIVED chaser columns;
  - the Palette-authored descriptor profile id;
  - the list of NO-LEGACY-HOME nodes intentionally not projected.
- **D5. Interrupted / incomplete execution.** Legacy refuses. Options:
  - (a) the adapter refuses too, matching legacy, but then the fixtures can never produce a legacy run;
  - (b) project steps only for steps whose `execution_completion_status` is complete. That is a legacy-contract change in `_materialize_stimulus_steps`, not an adapter rule.

  Never fall back to event `camera_frame_id` bounds (D10).
- **D6. Selector coexistence.** For a recording that has both a legacy-H5 stimulus run and an adapter run, which is `latest`? Recommend: the adapter refuses if any selector-eligible stimulus run already exists, unless explicitly overridden. Otherwise the registry sees two protocol runs with different `source_h5` for one recording.
- **D7. `protocol_semantic_chaser_selection` sealed raw-H5 binding.** It requires `run.attrs.source_h5 == source.raw_h5` (`analysis_workflows/protocol_semantic_chaser_selection.py:746-750`). Decide whether the sealed "raw H5" for unified recordings is the unified file or the derivative, and update the selection sealing accordingly.
- **D8. No silent repair or inference.** The adapter must:
  - never call `interpolate_metadata`, `interpolate_run` or `repair_chaser_gaps` (pass `repair_chaser_gaps=False`; for non-chaser sessions legacy would still interpolate frame metadata at `:3274`, so either refuse, or add a no-interpolation importer flag);
  - never derive `camera_id` from the `ipc_source_name` regex;
  - never invent `subject_count`;
  - never backfill `start_camera_frame` from events;
  - never map `/observations/bounding_boxes` into `tracking_data`;
  - assert every narrowing cast is in range and every current-source acquisition row is valid, and refuse rather than clamp.
- **D9. Frame session clock.** Either add `timestamp_ns_session` as a fifth `frame_metadata` column (additive; columnar readers select by name, so it should be harmless **[I]**) or leave it native-only. Adding it is the cheapest win for downstream timing work.
- **D10. `session_uuid` / `camera_id` sourcing.** Take `session_uuid` from `/metadata/session` as-is: it is the Arena session, as the legacy per-arena value was (do not synthesize the legacy format). Take `camera_id` from the binding. Never use `session_uuid` as the acquisition session or as a recording-ID input.
- **D11. Version the adapter against the unified schema.** The unified core is `development_core_definitions_not_production_admitted`. Pin the adapter to `(schema_id, schema_version, storage_schema/storage_version)`, and refuse unknown versions rather than best-effort mapping.
- **D12. Frame alignment method (decided 2026-09-23).** Adapter v1 keeps the
  legacy trigger-ID alignment bit-for-bit (`camera_frame_offset`,
  `camera_to_metadata_index`, step bounds), so new results stay comparable with
  every existing recording. Legacy H5s carry no exact acquisition map and
  cannot be re-aligned, so switching methods would mix two alignments within a
  cohort. The exact map stays in `native_h5`. Once real unified sessions exist,
  measure the per-session difference between the two methods, especially
  around chase onsets; check the 2026-08-20 chaser stimulus/camera temporal
  projection audit in `docs/` first. Switch only if that difference matters,
  and then as a versioned alignment method stamped on each run, with cohort
  analyses refusing to mix versions silently. The code cost of a switch is
  small (about 5 consumers read these fields); the cost is comparability.
