# Recording Registry Normalization TODO

Purpose: move to a recording-first registry model that supports queries like:
"all recordings with genotype X, DPF Y, and protocol step/task Z."

Manifest field contract for ingestion is documented in:
`docs/recording_manifest_contract.md`.
Future multi-camera / 3D analysis planning is tracked in:
`docs/multicamera_3d_analysis_todo.md`.
Analysis archive creation split planning is tracked in:
`docs/analysis_zarr_creation_contract.md` and `docs/analysis_zarr_creation_todo.md`.
Detect-stage performance registry planning is tracked in:
`docs/detect_performance_registry_todo.md`.
Crop-review registry planning is tracked in:
`docs/crop_review_registry_todo.md`.
Cross-archive string encoding standardization is tracked in:
`docs/zarr_string_encoding_todo.md`.
Registry-backed recording step/status migration is tracked in:
`docs/recording_step_status_registry_todo.md`.
Status (2026-02-09): detect-performance Phase 2A is complete
(schema v11, model-only latest views, scoped backfill defaults).

**Architectural decision:** palette registry and zebrobot.db remain separate databases.
Palette normalizes biological metadata already captured in Zarr snapshots
(cross_id, dish_id, genotype, line_strain, etc.) into proper entity tables.
zebrobot.db remains the isolated source of truth for husbandry/lifecycle data.
SQLite `ATTACH` is available as an escape hatch for zebrobot-only fields if needed.

## Current Status Snapshot

### Implemented now (Phase 1 core)

- [x] `recordings` + `recording_artifacts` tables exist and are backfilled.
- [x] `datasets` includes `recording_id` and `artifact_kind`.
- [x] `merged_training_datasets` view exists (with `artifact_kind` and `zarr_purpose`).
- [x] Maintenance backfill command exists: `--backfill-recording-entities`.
- [x] Integrity command validates recording linkage + required `behavior_v1` artifacts.
- [x] Registry one-shot hygiene command exists: `--reconcile-registry`.
- [x] Registry viewer exposes recordings: `check_training_registry --view recordings` (and `--all`).
- [x] `recording_subtype` is implemented for behavior recordings.
  - Current controlled behavior vocab: `free`, `embedded`.
  - Existing recordings currently backfilled as `behavior/free`.
- [x] Recording type/subtype vocabulary is now DB-backed.
  - Tables: `recording_type_vocab`, `recording_subtype_vocab` (seeded defaults).
  - Integrity checks read active vocab from DB (with hardcoded fallback only if tables are empty).
  - Registry summary can display allowed vocab via `--recording-summary`.
- [x] Recording manifest contract validator is implemented and in active use.
  - CLI: `scripts/py -m fisheye.utils.validate_recording_manifest ...`
  - Supports in-place defaults patching via `--apply-defaults`.
  - Current recording manifests validate cleanly after defaults rollout.
- [x] Unit-test baseline is recovered for `tests/unit/fisheye`.
  - Latest local run: `170 passed, 1 skipped`.
  - Remaining warnings are Zarr V3 string dtype stability warnings (non-blocking).

### Next up (high-priority open items)

- [x] Add `dataset_lineage` table + convenience view.
- [x] Add lineage-aware integrity checks for derived datasets and merged artifacts.
- [x] Finalize `zarr_purpose`/origin vocabulary split to avoid mixed semantics.
- [x] Define immutable vs mutable field policy and delete policy in writing.
  - See: `docs/registry_data_governance_policy.md`
- [x] Add `recording_overview` query view with common filters.
  - Implemented as SQL view `recording_overview`.
  - Exposed in registry viewer via `--view recording-overview` and included in `--all`.

### Active next actions

- [x] Execute the `dataset_id` re-key migration runbook (Phase A-F) on live registry.
- [x] Add migration notes + operator checklist for first real post-bootstrap schema change.
  - See: `docs/registry_repair_playbook.md`
- [x] Prevent legacy source IDs from being recreated during `registry_rescan` / `register_from_root`.
  - Implemented by preferring canonical source IDs (`{session_uuid}:z<path-hash>`) in
    `_resolve_effective_dataset_id()` for recording-source artifacts.
  - Verified on live DB: rescan updates did not recreate `dataset_id=session_uuid` rows.
- [x] Keep CI wiring for the targeted registry migration/integrity test subset.
  - CI entrypoint script: `scripts/ci_registry_migration_integrity_subset.sh`
  - Supports full run and fast `--smoke` precheck mode.
- [x] Execute registry-backed recording step/status plan.
  - See: `docs/recording_step_status_registry_todo.md`
  - Goal: make recording health/status queryable from registry (not filesystem traversal).
  - Implemented with runtime status hooks, status ledger views, and validation harness:
    - `scripts/validate_recording_step_status_registry.sh`
    - `scripts/validate_recording_step_status_registry_batch.sh`
  - Latest operator batch validation: `passed=52 failed=0 skipped_missing_zarr=1`
    (skipped non-recording path: `/nvme1/recordings/logs`).
  - Periodic smoke is scheduled via cron (nightly `02:30`), logging to:
    `/tmp/palette_recording_step_status_smoke_cron.log`.

## Glossary

- `dataset_id`
  - Identity of one concrete dataset artifact (typically one Zarr path).
  - Used for both source recording datasets and derived datasets (e.g. merged training exports).
  - Answers: "what artifact/file is this?"

- `set_id`
  - Identity of one training set definition/version (a selected cohort of dataset rows + config context).
  - Not a file itself; it groups multiple datasets for training.
  - Answers: "what cohort/version was this model trained from?"

- Relationship
  - One `set_id` includes many `dataset_id` rows.
  - One `dataset_id` may be reused across multiple `set_id` values over time.

## Key Decision

- [x] Model `recording` as first-class parent.
- [x] Keep one row per Zarr as `dataset` (child of recording).
- [x] Keep workflow state (review/training/export status) separate from immutable provenance (initial pass).
  - Immutable-vs-mutable policy is documented in `docs/registry_data_governance_policy.md`.

## Target Model (Phase 1)

### `recording_id` derivation

- [x] **Decided:** `recording_id = session_uuid` when available (the common case).
  - Fallback: `path-{sha256(zarr_path)[:12]}` for legacy Zarrs without `session_uuid`.
  - This mirrors the existing `resolve_dataset_id()` logic (`db.py:444-447`).
  - Stability rule: fallback `recording_id` is assigned once and persisted; it must not be recomputed from
    current path after first registration (path moves must not change identity).
  - Source dataset IDs are canonicalized as `{session_uuid}:z<path-hash>` so multiple Zarrs
    can coexist per recording without identity collisions.

### `dataset_id` re-key migration (required for multi-Zarr)

- [x] Add explicit migration plan to decouple `dataset_id` from `session_uuid`.
  - Introduce new `dataset_id` generation policy for all new rows.
  - Backfill/re-key existing rows where `dataset_id=session_uuid`.
  - Update all dependent FK references in one transaction-safe migration pass:
    - `provenance.dataset_id`
    - `training_set_datasets.dataset_id`
    - `keypoint_quality.dataset_id`
    - `detect_quality.dataset_id`
    - any other dataset-linked table introduced before rollout.
  - Add temporary compatibility view/mapping table to bridge old IDs during transition.

#### Proposed execution plan (runbook)

- [x] Phase A: preflight + safety snapshot
  - Create full DB backup before mutation.
  - Capture pre-migration counts:
    - total datasets
    - source vs derived counts
    - count of rows where `dataset_id=session_uuid`
    - count of training set members that do not resolve to current `datasets.dataset_id`.
  - Freeze write-heavy jobs during migration window (training/export/maintenance mutators).

- [x] Phase B: deterministic ID mapping
  - Build `dataset_id_remap(old_dataset_id, new_dataset_id, reason, created_utc)` table.
  - For source recording rows where `dataset_id=session_uuid`, set:
    - `new_dataset_id = {session_uuid}:{short_path_hash}`
    - keep deterministic and collision-safe.
  - Leave rows already using non-session IDs unchanged.
  - Validate remap table:
    - all `new_dataset_id` unique
    - all `old_dataset_id` exist
    - no null/empty IDs.

- [x] Phase C: transaction-safe FK rewrite
  - In one transaction, apply remap to every dataset-linked table:
    - `datasets.dataset_id` (primary identity update)
    - `provenance.dataset_id`
    - `training_set_datasets.dataset_id` (if present in this DB version)
    - `keypoint_quality.dataset_id`
    - `detect_quality.dataset_id`
    - `dataset_lineage.parent_dataset_id`
    - `dataset_lineage.child_dataset_id`
    - any later-added dataset-linked table.
  - Rebuild/refresh derived views if needed.

- [x] Phase D: training-set JSON linkage rewrite
  - Rewrite `training_sets.dataset_ids_json` entries using remap table.
  - Run built-in helper where applicable:
    - `--remap-training-set-dataset-ids`
  - Verify no unresolved IDs remain in `dataset_ids_json`.

- [x] Phase E: lineage + integrity revalidation
  - Run:
    - `--backfill-dataset-lineage`
    - `--check-integrity`
  - Verify:
    - no merged dataset missing lineage
    - no set/lineage mismatch
    - no orphan dataset references.

- [x] Phase F: compatibility + cleanup
  - Decision (2026-02-09): retire old-ID compatibility immediately (early project stage).
  - Do not maintain long-lived old-ID lookup compatibility in registry.
  - Keep canonical IDs only (`session_uuid:z...` for source recordings; readable IDs for derived merges).
  - 2026-02-09 execution note:
    - preflight/final integrity passed
    - training-set remap applied `0` row changes (already canonical)
    - lineage rebuild unchanged (`relationships_changed=0`)
    - current `dataset_id=session_uuid` count is `2`, corresponding to derived merged datasets
      (`detect_cedar_shadow_v007_merged`, `pose_cedar_shadow_manual_gray_latest_traditional_894ad574_v004_merged`);
      this is expected because source recording datasets are already re-keyed to `session_uuid:z...`.
  - 2026-02-09 Phase F verification:
    - `source_ids_equal_session_uuid = 0`
    - `unresolved_training_set_members = 0`
    - dropped temporary table: `dataset_id_remap`
    - final `--check-integrity` passed with no issues.

#### Rollback plan

- [ ] If any Phase C+ validation fails:
  - Stop mutating jobs.
  - Restore registry from preflight backup.
  - Re-run integrity checks to confirm restored baseline.
  - Record root cause and adjust remap rules before retry.

### Recording<->dataset cardinality

- [x] **Decided:** one-to-many. One recording owns many datasets.
  - `datasets.recording_id` FK is sufficient; no link table needed.

### `recordings` table

- [x] `recordings` table (one row per recording/session) added (initial schema pass).
  - Example fields:
    - `recording_id` (PK)
    - `session_uuid`
    - `recording_path`
    - `started_utc`
    - `rig_id`, `arena_id`, `canvas_name`
    - `status`, `created_utc`

- [x] `datasets` stays one row per Zarr, linked to recording.
  - Add `recording_id` FK to `datasets`.
  - Continue storing `zarr_path`, `dataset_id`, and status.

- [x] Store per-Zarr purpose and flags in child rows (initial pass).
  - `zarr_origin` (`source`, `derived`, `imported`) and `zarr_use` (`training`, `analysis`, etc.)
  - compatibility alias `zarr_purpose` remains view-level only where needed
  - artifact flags (e.g. merged export, contains crop/keypoint/detect runs)
  - quality/status flags as needed

### Multi-Zarr per Recording + Lineage (Phase 1A)

- [x] Support multiple Zarr assets per recording explicitly (schema/backfill pass).
  - A single `recording_id` can own many `datasets` rows.
  - Each dataset row carries `zarr_origin`/`zarr_use` and artifact type flags.
  - Example purposes: `source_training`, `source_analysis`, `inference_output`, `training_merged`.

- [x] Naming convention decision for behavior recordings:
  - Source training artifact: `<recording_base>_training.zarr`
  - Analysis working artifact: `<recording_base>_analysis.zarr`
  - Rationale: keep training curation immutable while allowing iterative analysis runs.

- [x] `zarr_purpose` vocabulary note: current example values mix origin and use dimensions.
  - Implemented two-field approach in `datasets`:
    - `zarr_origin` (`source`, `derived`, `imported`)
    - `zarr_use` (`training`, `analysis`, `inference`, `export`, `archive`)
  - Legacy `provenance.zarr_purpose` retained for compatibility during migration.
  - Current workflow policy:
    - `training` and `analysis` are the active uses.
    - `inference` is not used as a standalone Zarr artifact in current behavior workflows
      (inference outputs are expected inside analysis artifacts/runs).
    - `export` is reserved for standalone collated/packaged outputs (for sharing/archive).

- [x] Migration task: rename existing recording Zarrs and reconcile registry paths.
  - Renamed on disk from legacy single-artifact names to suffix-based names.
  - Rescan/reconcile updates `datasets.zarr_path` to new locations.
  - Added built-in membership remap command for ID migrations:
    - `--remap-training-set-dataset-ids`
  - Lineage + integrity recovery workflow is documented in:
    - `docs/registry_repair_playbook.md`

- [x] Keep merged/training artifacts in `datasets` (not in `recordings`).
  - Source recording Zarrs should have `recording_id` populated.
  - Derived merged exports may have:
    - same `recording_id` when strictly tied to one recording, or
    - `recording_id` NULL when derived from many recordings.
  - Use lineage edges to preserve auditable ancestry in both cases.

- [x] Add `dataset_lineage` table for parent/child derivation edges.
  - Example fields:
    - `lineage_id` (PK)
    - `child_dataset_id` FK -> `datasets.dataset_id`
    - `parent_dataset_id` FK -> `datasets.dataset_id`
    - `op_type` (`merge_for_detect_train`, `merge_for_pose_train`, `refine_detect`, `refine_pose`, ...)
    - `op_run_id` (nullable FK for backfilled/history-only edges)
    - `lineage_source` (`observed`, `backfilled`, `imported`)
    - `created_utc`
  - Add uniqueness constraint on `(child_dataset_id, parent_dataset_id, op_type)`.

- [x] Add convenience view: `dataset_lineage_current`.
  - Goal: quickly answer "what source datasets produced this merged dataset?"
  - Include parent purpose + recording linkage fields for filtering.

- [x] Add convenience view for merged training datasets.
  - Implemented as `merged_training_datasets` (includes `artifact_kind` and `zarr_purpose`).

- [x] Align existing join tables with lineage.
  - `training_set_datasets` remains user-facing set membership.
  - `dataset_lineage` remains canonical derivation graph.
  - Add consistency check in maintenance: set-membership edges must exist in lineage for merged exports.

## Phase 1 Implementation Guardrails

- [x] Add `artifact_kind` to `datasets` to avoid overloading `zarr_purpose`.
  - Suggested controlled values:
    - `source_recording`
    - `derived_analysis`
    - `derived_training_merge`
    - `model_input_export`

- [x] Add explicit `dataset_lineage` integrity constraints + indexes.
  - `CHECK(child_dataset_id <> parent_dataset_id)`
  - indexes:
    - `dataset_lineage(parent_dataset_id)`
    - `dataset_lineage(child_dataset_id)`
    - `dataset_lineage(op_type)`
  - Status: parent/child indexes implemented; self-edge enforcement is implemented via DB triggers. `op_type` is the canonical column name (used in `dataset_lineage` DDL).

- [x] Define delete policy before rollout.
  - Policy documented in `docs/registry_data_governance_policy.md`.
  - Never auto-delete source recording dataset rows.
  - Explicit-only deletion for derived artifacts.
  - File deletion stays scoped to safe derived artifact roots.

- [x] Define immutable vs mutable registry fields.
  - Policy documented in `docs/registry_data_governance_policy.md`.
  - Immutable provenance and mutable operational fields are explicitly enumerated.

- [x] Keep migration minimum viable for Phase 1 (core delivered).
  - Implemented core:
    - `recordings`
    - `datasets.recording_id`
    - `recording_artifacts`
    - backfill + compatibility views
  - Remaining Phase 1 hardening:
    - `dataset_id` re-key migration
    - policy/contract finalization

- [x] Index all Phase-1 FK columns used in WHERE or JOIN clauses.
  - SQLite does not auto-index foreign keys.
  - Implemented/indexed in current schema:
    - `datasets(recording_id)` (`idx_datasets_recording_id`)
    - `recording_artifacts(recording_id)` (`idx_recording_artifacts_recording_id`)
    - `dataset_lineage(child_dataset_id, relationship_type)` (`idx_dataset_lineage_child_rel`)
    - `dataset_lineage(parent_dataset_id, relationship_type)` (`idx_dataset_lineage_parent_rel`)
    - `training_models(run_id)` / `onnx_models(run_id)` / `tensorrt_models(run_id, precision)` / `model_exports(run_id, export_type)` are PK-backed and therefore indexed.
  - Future-phase FK index work remains tracked with the corresponding tables
    (`dishes`, `subjects`, `recording_subjects`, `trials`, `trial_metrics`).

- [x] `metadata_json` promotion rule defined: if a field stored in `metadata_json` is used
  in WHERE/GROUP BY queries more than occasionally, promote it to a real column.
  Typed columns are indexable and avoid JSON extraction overhead at query time.
  - Current policy decision (2026-02-09): document now; defer additional promotions
    until fields show sustained query demand.

### Phase 1 completion status

- [x] Phase 1 core + hardening are complete for current scope.
  - Remaining open item (`CI wiring`) is tracked as operational follow-up.

- [x] Add hard validation checks to acceptance criteria (partial — see remaining item below).
  - No orphan datasets (every source dataset linked to a valid recording).
  - Every merged training dataset has one or more lineage parents.
  - `training_set_datasets` membership is consistent with lineage edges.
  - Status: orphan-dataset, merged-parent, and set-membership consistency checks are implemented.
- [x] Add lineage-cycle detection to hard validation checks.
  - Implemented in `fisheye.registry.maintenance._check_registry_integrity`
    via DFS on `dataset_lineage_current` (emits `dataset_lineage_cycle` issues).
  - Covered by unit test:
    `tests/unit/fisheye/test_registry_maintenance.py::test_integrity_flags_dataset_lineage_cycle`.

## Subject / Dish / Cross Normalization (Phase 2)

Data source: palette registry normalizes cross/dish/subject data **from Zarr snapshots**
already captured at acquisition time via `_extract_provenance()` (`db.py:704-722`).
Fields available from snapshots: `cross_id`, `dish_id`, `genotype`, `line_strain`,
`dpf_at_acquisition`, `species`, `sex`, `fish_id` (mapped to `subject_id`),
`subject_count`, `parents`.
  - DPF extraction note (2026-02-09): current acquisition metadata stores DPF as
    `days_post_fertilization`; registry maps this value into
    `provenance.dpf_at_acquisition`.

**No sync with zebrobot.db** for the common case — snapshot data is sufficient.
zebrobot.db stays isolated as the husbandry/lifecycle source of truth
(screening, quality checks, breeding records). For future queries needing
zebrobot-only fields (screening results, transgenic indicators, cross yield),
use SQLite `ATTACH` for read-only access (see Phase 7).

- [x] Add `crosses` table.
  - `cross_id` (PK, from snapshot)
  - `line_strain`, `genotype`, `parents` summary
  - `metadata_json` for ad-hoc extras
  - Status (2026-02-09): implemented in schema migration `v6`
    (`src/fisheye/registry/db.py`, table `crosses`).

- [x] Add `dishes` table.
  - `dish_id` (PK, from snapshot)
  - `cross_id` FK -> `crosses.cross_id`
  - `species`, `metadata_json`
  - Status (2026-02-09): implemented in schema migration `v6`
    (`src/fisheye/registry/db.py`, table `dishes`).

- [x] Add `recording_subjects` join table (see Phase 6 for full definition).
  - Status (2026-02-09): initial join table implemented in schema migration `v6`
    (`src/fisheye/registry/db.py`, table `recording_subjects`).
  - Operator backfill command implemented:
    `scripts/py -m fisheye.registry.maintenance --registry /nvme1/palette_registry.sqlite --backfill-subject-dish-cross [--dry-run]`.

### Phase 2 completion status

- [x] Phase 2 schema + operator backfill are complete on live registry.
  - Execution date: 2026-02-09
  - Registry: `/nvme1/palette_registry.sqlite`
  - Backfill apply summary:
    - `source_rows_scanned=52`
    - `crosses_unique_seen=2`, `crosses_would_insert=2`
    - `dishes_unique_seen=3`, `dishes_would_insert=3`
    - `recording_subjects_unique_seen=52`, `recording_subjects_would_insert=52`
    - `rows_skipped_missing_recording_id=0`, `rows_skipped_missing_subject_id=0`
  - Post-apply verification:
    - `crosses=2`
    - `dishes=3`
    - `recording_subjects=52`
    - `--check-integrity` passed with no issues.
  - Observed identity shape:
    - `distinct_subject_ids=26`
    - `distinct_recording_subject_pairs=52`
    - Rationale: same `subject_id` can appear in multiple recordings; join key is `(recording_id, subject_id)`.

## Protocol / Camera Normalization (Phase 3)

- [ ] Add `protocol_runs` table.
  - `protocol_run_id`, `recording_id`, `protocol_name`, `protocol_hash`, timestamps.

- [ ] Add `protocol_steps` table.
  - indexed steps with names/types/timing/params.

- [ ] Add `camera_runs` table.
  - `camera_id`, model, serial, fps, exposure, gain, pixel format, metadata.

## Analysis Run Ledger (Phase 4)

- [ ] Keep artifacts as source of truth; use DB as index/ledger only.
  - Full arrays/results remain in auditable artifacts (zarr/json/report files).
  - Registry stores pointers, hashes, status, and compact query fields.

- [ ] Add `analysis_runs` table.
  - One row per processing step execution (e.g. inference, refine, behavior).
  - Example fields:
    - `analysis_run_id` (PK)
    - `recording_id` FK
    - `dataset_id` FK
    - `step_type` (`detect_infer`, `pose_infer`, `refine`, `behavior_speed`, ...)
    - `input_artifact_path`, `input_artifact_sha256`
    - `output_artifact_path`, `output_artifact_sha256`
    - `model_run_id` (optional FK to training/model registry rows)
    - `params_json`, `invocation_json`
    - `status`, `error_message`
    - `tool_version`, `git_commit`, `created_utc`, `completed_utc`

- [ ] Add typed summary tables per analysis domain (not EAV).
  - `detection_summaries`:
    - `analysis_run_id` FK
    - `total_detections INTEGER`
    - `real_detections INTEGER`
    - `interpolated_rate REAL`
    - `false_positive_rate REAL`
    - `metadata_json` (ad-hoc extras)
  - `behavior_summaries`:
    - `analysis_run_id` FK
    - `mean_speed REAL`
    - `max_speed REAL`
    - `latency_to_escape REAL`
    - `distance_to_chaser_min REAL`
    - `metadata_json` (ad-hoc extras)
  - Each typed table uses explicit metric columns for queryability;
    `metadata_json` stores ad-hoc extras without schema changes.
  - New analysis domains get new typed tables as needed.

- [ ] Add optional convenience view: `recording_analysis_overview`.
  - Shows latest successful run per step type per recording.
  - Includes output paths + top summary metrics.

## Trial / Segment Indexing (Phase 5)

- [ ] Add `trials` (or `trial_segments`) table for experiment episode windows.
  - One row per trial/segment so queries do not require full dataset re-parse.
  - Example fields:
    - `trial_id` (PK)
    - `recording_id` FK
    - `dataset_id` FK
    - `analysis_run_id` FK (which pipeline step produced this trial)
    - `session_type` (`chaser`, `feeding`, etc.)
    - `start_frame`, `end_frame`, `start_utc`, `end_utc`
    - `outcome` (`escaped`, `captured`, `timeout`, ...)
    - `outcome_confidence`
    - `trial_artifact_path`, `trial_artifact_sha256`

- [ ] Add `trial_metrics` table with typed columns.
  - Example fields:
    - `trial_id` FK
    - `analysis_run_id` FK
    - `latency_to_escape REAL`
    - `max_speed REAL`
    - `distance_to_chaser_min REAL`
    - `metadata_json` (ad-hoc extras)
  - Keep this table small and query-oriented; store dense time-series in artifacts.

- [ ] Add query views for trial-level exploration.
  - Example target query:
    - "all trials from chaser sessions where the fish escaped."

## Subject Identity Across Recordings (Phase 6)

**Naming convention:** the registry uses `subject_id` as the canonical, species-neutral
identity column. Source data in Zarr snapshots stores this as `fish_id`; the mapping
at registration time is `subject_id = snapshot["fish_id"]`.

- [x] Add `subjects` table (one row per subject across all recordings).
  - `subject_id` TEXT PK — sourced from `fish_id` UUID in Zarr files at acquisition time
    (stored in `/subject_metadata` or `analysis_metadata.zebrobot_snapshot`).
  - Already extracted into provenance table (`db.py:710`).
  - `dish_id` FK -> `dishes.dish_id` (snapshot-sourced)
  - optional `sex`, `species` (subject-level convenience fields when present)
  - normalization decision (2026-02-09): keep cross/genotype canonical in
    `dishes`/`crosses`; do not require denormalized `subjects.cross_id` or
    `subjects.genotype` for correctness.
  - identity confidence/notes metadata
  - Status (2026-02-09): implemented in schema migration `v7` and backfilled on
    `/nvme1/palette_registry.sqlite` (`subjects=26`, `subjects_missing_dish=0`).

- [x] Add `recording_subjects` join table baseline.
  - Status (2026-02-09): implemented in Phase 2 (`recording_id`, `subject_id`,
    `dataset_id`, `dish_id`, `cross_id`, `dpf_at_acquisition`, plus metadata).
  - Supports repeated subjects across many recordings (same `subject_id` can
    appear in multiple recordings/protocols).

- [ ] Extend `recording_subjects` for Phase 6 role metadata.
  - Status (2026-02-09): deferred until a concrete trial-subject producer exists
    (currently no stable source emits subject-role annotations).
  - Future intent: add explicit role fields needed by downstream trial/behavior joins.
  - Preserve existing `(recording_id, subject_id)` key semantics.

- [ ] Link trials to subjects.
  - Minimum: `trials.subject_id` for single-subject trials.
  - Preferred flexible model: `trial_subjects(trial_id, subject_id, role)`.
    - role examples: `chaser`, `target`, `bystander`.

- [ ] Add cross-recording subject queries.
  - Examples:
    - all escaped trials for `subject_id = X`
    - all target-role trials with outcome `escaped` in chaser sessions
  - priority query support (confirmed 2026-02-09):
    - all recordings from `cross_id = X`
    - all recordings where `dpf_at_acquisition = N` and `genotype = Y`
    - implementation path: `recording_subjects -> subjects -> dishes -> crosses`
      with indexed genotype and DPF filters.
  - Status (2026-02-09): baseline query view implemented in schema migration `v8`:
    `recording_subject_overview` (includes `cross_id`, `genotype`,
    `dpf_at_acquisition`, and recording context columns).
  - Example queries using the view:
    - `SELECT DISTINCT recording_id FROM recording_subject_overview WHERE cross_id = :cross_id;`
    - `SELECT DISTINCT recording_id FROM recording_subject_overview WHERE dpf_at_acquisition = :dpf AND genotype = :genotype;`

- [ ] Note: zebrobot.db's `fish_subjects`/`fish_runs` tables (currently empty) can be
  populated separately for lifecycle tracking, but palette does not depend on them.

## Cross / Dish Integration via ATTACH (Phase 7)

Palette registry normalizes cross/dish data from Zarr snapshots (Phase 2).
No sync mechanism is needed for the common case.

For future queries requiring zebrobot-only fields (screening results,
transgenic indicators, cross yield, breeding records), use SQLite `ATTACH`
for read-only access:

- [ ] Document `ATTACH` usage pattern for zebrobot.db.
  - `ATTACH DATABASE '/nvme1/zebrobot.db' AS zebrobot;`
  - Join palette entity tables with zebrobot tables in a single query.
  - Read-only: palette never writes to zebrobot.db.
  - Example: enrich palette cross data with screening outcomes from zebrobot.

- [ ] Model `dishes` as children of `crosses` (defined in Phase 2).
  - `dishes.cross_id -> crosses.cross_id`
  - preserve dish metadata used during acquisition.

- [ ] Link `subjects` to dish/cross lineage (defined in Phase 6).
  - `subjects.dish_id` (implies cross through dish)
  - optional denormalized `subjects.cross_id` remains deferred; base model is normalized.

- [x] Add Phase 6 query/performance indexes for cross/genotype workflows.
  - `CREATE INDEX ... ON crosses(genotype)`
  - `CREATE INDEX ... ON dishes(cross_id)`
  - `CREATE INDEX ... ON subjects(dish_id)`
  - `CREATE INDEX ... ON recording_subjects(subject_id, dpf_at_acquisition)`
  - `CREATE INDEX ... ON recording_subjects(recording_id)`
  - Status (2026-02-09): implemented in schema migrations `v6/v7`.

- [ ] Add convenience views that join palette + zebrobot via ATTACH.
  - Only for queries that genuinely need husbandry/lifecycle data not in snapshots.
  - Keep ATTACH optional — palette registry must function without zebrobot.db present.

## Compatibility + Write Path

- [ ] Keep current `provenance` table during migration as denormalized cache.
- [ ] Update `Registry.register_from_root(...)` to:
  - resolve/create recording
  - link dataset->recording
  - upsert subjects/dish/protocol/camera entities
  - continue writing compatibility fields

- [ ] Ensure idempotent upserts with stable keys + unique constraints.

- [ ] Update processing tools (inference/refine/behavior) to register runs.
  - Record run start (`in_progress`) and completion (`success`/`failed`).
  - Attach output artifact paths and hashes.
  - Write compact summaries for query use.

### Provenance Table Deprecation Plan

Phased deprecation of the flat `provenance` table:

1. **Phases 1-3:** dual-write to both `provenance` and new entity tables.
   - All consumers continue reading from `provenance` during this period.
2. **After Phase 3:** migrate read consumers one at a time to new entity tables.
3. **After all readers migrated:** stop writing to `provenance`; keep as read-only archive.

Consumer code paths requiring migration:
- `register_from_root()` in `db.py` (write path)
- `check_training_registry.py` (read: training/quality queries)
- `registry_tui.py` (read: browsing panels)
- `maintenance.py` (read/write: backfill, pruning, integrity checks)

## Query UX

- [x] Add `recording_overview` view for common filters.
- [x] Add recordings panel to registry viewer (`check_training_registry`).
  - `--view recordings` and `--all` now show recording rows, artifact counts, and
    required-artifact completeness for `behavior_v1`.
  - `--recording-summary` shows type/subtype counts and allowed subtype vocab.
- [ ] Add filters for:
  - [x] genotype (implemented via `registry_query --genotype` and
    `recording_subject_overview`)
  - [x] exact DPF (implemented via `registry_query --dpf` and
    `recording_subject_overview`)
  - [x] DPF range (implemented via `registry_query --dpf-min/--dpf-max`)
  - [x] cross_id (implemented via `registry_query --cross-id` and
    `recording_subject_overview`)
  - protocol step/task
  - subject count
  - camera/rig/arena
  - analysis step completion/status
  - behavior summary thresholds (e.g. mean speed)
  - trial outcomes + trial metric thresholds
  - subject-level history across recordings

## Modality Growth Rubric

Use a layered model so new recording variants do not force frequent core-schema churn.

- [ ] Separate workflow intent from modality classification.
  - `zarr_use` should answer "why this artifact exists now" (e.g. `training`, `analysis`, `export`, `archive`).
    (Legacy `zarr_purpose` has been split into `zarr_origin` + `zarr_use` — see Phase 1A above.)
  - `recording_type` / `recording_subtype` should answer "what kind of recording this is."
  - Do not use `zarr_use` to encode modality.

- [ ] Keep top-level `recording_type` small and stable.
  - Example: `behavior`, `microscopy`, `histology`.
  - Avoid creating new top-level types for every protocol variation.

- [x] Add initial controlled modality taxonomy.
  - `recording_type='behavior'`:
    - `recording_subtype IN ('free', 'embedded')` (implemented)
  - `recording_type='microscopy'`:
    - `recording_subtype IN ('lightsheet', 'confocal', '2p')` (implemented)
  - `recording_type='histology'`:
    - `recording_subtype IN ('section', 'wholemount')` (initial implemented set)
  - Keep subtype vocab in lookup tables so it can evolve without schema churn.

- [ ] Represent fine-grained variation with versioned protocol/config identifiers.
  - `protocol_id`, `acquisition_profile_id`, `schema_version`.
  - Use these IDs for most subtype distinctions.

- [ ] Use typed extension tables per modality family.
  - Keep `recordings` compact (shared/common fields only).
  - Store modality-specific fields in child tables keyed by `recording_id`.

- [x] Use controlled vocab/lookup tables for frequently filtered categorical values.
  - Avoid uncontrolled free-text for fields used in joins/filters.
  - Add version/validity windows for evolving vocabularies if needed.
  - Initial implementation: recording type/subtype vocab tables + active flags.

- [ ] Keep rare/unstable fields in `metadata_json`, then promote by usage.
  - Promotion rule: if used repeatedly in WHERE/GROUP BY, add a typed column.
  - Keep `metadata_json` for ad-hoc, low-frequency, or evolving attributes.

- [ ] Include explicit schema versions for modality payloads.
  - Each modality extension table should carry a version field where structure may evolve.
  - Add migration notes when version changes introduce semantic differences.

- [ ] Query design rule of thumb:
  - frequent filter/join field -> typed column or FK.
  - occasional annotation field -> JSON.

## Recording Classification Contract (Current Stable Workflow)

For current behavior recordings, classify recordings from declared metadata and
artifact schema, not from directory naming conventions.

- [x] Do not infer `recording_type` from folder name/path (initial behavior).
  - Folder names are operational layout, not authoritative semantics.

- [x] Declare `recording_type` + `artifact_schema_id` in metadata.
  - Source of truth: `recording_manifest.json` (or mirrored Zarr attrs).
  - Example: `recording_type='behavior'`, `recording_subtype='free'`, `artifact_schema_id='behavior_v1'`.
  - Current manifests have been normalized with validator defaults and pass validation.

- [x] Index actual artifacts per recording in a normalized table.
  - `recording_artifacts` row per file with:
    - `recording_id`
    - `artifact_type`
    - `path`
    - `status`
    - optional hash/size/timestamps

- [ ] Validate expected artifact set against the declared schema.
  - `artifact_schema_id` defines expected/required artifact types.
  - Validator checks expected vs present and reports missing/extra artifacts.
  - Keep validation output queryable in registry (pass/fail + missing list).
  - [x] Implemented for initial `behavior_v1` required artifact set in
    maintenance integrity checks and surfaced in registry viewer recordings panel.
  - [ ] Generalize to schema-driven validation for arbitrary `artifact_schema_id` values.

- [ ] Keep this contract path-independent.
  - If storage layout changes later (tier moves, reorg, migration), recording
    classification and query behavior remain unchanged.

### Quick audit query (type/subtype distribution)

```sql
SELECT recording_type, recording_subtype, COUNT(*) AS n
FROM recordings
GROUP BY recording_type, recording_subtype
ORDER BY recording_type, recording_subtype;
```

CLI equivalent (includes allowed vocab footer):

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry /path/to/palette_registry.sqlite \
  --recording-summary
```

### Vocab Change Procedure (Recording Type/Subtype)

Use this when adding a new allowed `recording_type` or `recording_subtype`.

1. Insert/activate vocab rows in registry:

```sql
INSERT OR IGNORE INTO recording_type_vocab (recording_type, active, description)
VALUES ('microscopy', 1, 'Microscopy recordings');

INSERT OR IGNORE INTO recording_subtype_vocab (recording_type, recording_subtype, active, description)
VALUES ('microscopy', 'spinningdisk', 1, 'Spinning disk microscopy');
```

2. Ensure new manifests use the new values (`recording_type`, `recording_subtype`).
3. Backfill/refresh recording entities:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry /path/to/palette_registry.sqlite \
  --backfill-recording-entities
```

4. Validate contract:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry /path/to/palette_registry.sqlite \
  --check-integrity --list-limit 100
```

5. Confirm distribution + allowed vocab:

```bash
scripts/py -m fisheye.utils.check_training_registry \
  --registry /path/to/palette_registry.sqlite \
  --recording-summary
```

## Storage Tiering + Path Mobility

Goal: support PRFS (durable/compute), NRS (scratch/recomputable), and nearline
(cold archive) without tying dataset identity to filesystem paths.

- [ ] Treat `dataset_id` as stable identity; treat filesystem paths as mutable locations.
  - Never derive identity from current storage tier/path once registered.

- [ ] Add `dataset_locations` table (one dataset may have multiple locations).
  - Suggested fields:
    - `location_id` (PK)
    - `dataset_id` FK -> `datasets.dataset_id`
    - `tier` (`prfs`, `nrs`, `nearline`)
    - `path`
    - `is_primary` (bool)
    - `state` (`online`, `offline`, `restoring`, `missing`)
    - `bytes`
    - `checksum` (optional, per-location verification)
    - `last_verified_utc`, `created_utc`
  - Constraints:
    - unique `(dataset_id, path)`
    - at most one `is_primary=1` per `dataset_id`

- [ ] Keep `datasets.zarr_path` as compatibility field during migration.
  - Reads continue to work while tools migrate.
  - Eventually replace direct path reads with `dataset_locations` resolution.

- [ ] Add location-selection policy for compute/preflight.
  - Prefer `online` + `is_primary`.
  - Fallback order by tier preference (default: `prfs` > `nrs` > `nearline`).
  - If only `offline/restoring` locations exist, fail with "restore required."

- [ ] Add maintenance commands for location health.
  - verify/update states by path checks
  - set primary location
  - move/swap location paths
  - mark restore requested/completed

- [ ] Add lifecycle policy by `artifact_kind`.
  - `source_recording`: durable retention; no auto-delete.
  - `derived_training_merge` / recomputable artifacts: prefer NRS, pruneable with policy.
  - optional archive mirror on nearline for selected artifacts.

### Storage Tier Rollout Plan

- [ ] **Now (single-machine/local-only):**
  - Add schema + compatibility fields/views only.
  - Keep one location per dataset (`tier='local'` or `tier='nrs'` equivalent).
  - Do not force workflow changes yet.

- [ ] **When network tiers go live:**
  - backfill secondary locations as data is copied/moved.
  - switch preflight and training tools to location resolver.
  - enable tier-aware maintenance and restore workflows.

- [ ] **After migration confidence:**
  - reduce reliance on `datasets.zarr_path` in readers.
  - keep it as denormalized cache or deprecate in a later schema version.

### Draft DDL (Storage Tiering)

```sql
-- 1) New location table (one dataset can have multiple physical locations).
CREATE TABLE IF NOT EXISTS dataset_locations (
  location_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  tier TEXT NOT NULL,                         -- prfs | nrs | nearline | local
  path TEXT NOT NULL,
  is_primary INTEGER NOT NULL DEFAULT 0,      -- 0/1
  state TEXT NOT NULL DEFAULT 'online',       -- online | offline | restoring | missing
  bytes INTEGER,
  checksum TEXT,
  last_verified_utc TEXT,
  created_utc TEXT NOT NULL,
  FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
  CHECK (is_primary IN (0, 1)),
  CHECK (tier IN ('prfs', 'nrs', 'nearline', 'local')),
  CHECK (state IN ('online', 'offline', 'restoring', 'missing'))
);

-- 2) Prevent duplicate path rows for same dataset.
CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_locations_dataset_path
  ON dataset_locations(dataset_id, path);

-- 3) Fast lookups by dataset and by tier/state.
CREATE INDEX IF NOT EXISTS idx_dataset_locations_dataset_id
  ON dataset_locations(dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_locations_tier_state
  ON dataset_locations(tier, state);
CREATE INDEX IF NOT EXISTS idx_dataset_locations_primary
  ON dataset_locations(dataset_id, is_primary);

-- 4) Enforce at most one primary location per dataset.
CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_locations_one_primary
  ON dataset_locations(dataset_id)
  WHERE is_primary = 1;
```

```sql
-- Compatibility view: effective location for existing readers.
-- Priority:
--   1) online + is_primary=1
--   2) online fallback in tier order (prfs > nrs > nearline > local)
CREATE VIEW IF NOT EXISTS dataset_effective_location AS
WITH ranked AS (
  SELECT
    dl.*,
    ROW_NUMBER() OVER (
      PARTITION BY dl.dataset_id
      ORDER BY
        CASE WHEN dl.state = 'online' AND dl.is_primary = 1 THEN 0 ELSE 1 END,
        CASE dl.tier
          WHEN 'prfs' THEN 0
          WHEN 'nrs' THEN 1
          WHEN 'nearline' THEN 2
          WHEN 'local' THEN 3
          ELSE 9
        END,
        COALESCE(dl.last_verified_utc, dl.created_utc, '') DESC
    ) AS rn
  FROM dataset_locations dl
)
SELECT
  dataset_id,
  tier,
  path,
  is_primary,
  state,
  bytes,
  checksum,
  last_verified_utc,
  created_utc
FROM ranked
WHERE rn = 1;
```

```sql
-- Optional backfill seed from legacy datasets.zarr_path.
-- Replace location_id generation strategy with project-standard UUID helper if available.
INSERT INTO dataset_locations (
  location_id, dataset_id, tier, path, is_primary, state, created_utc
)
SELECT
  lower(hex(randomblob(16))) AS location_id,
  d.dataset_id,
  'local' AS tier,
  d.zarr_path AS path,
  1 AS is_primary,
  CASE
    WHEN d.status = 'missing' THEN 'missing'
    ELSE 'online'
  END AS state,
  COALESCE(d.last_seen_utc, d.created_utc, datetime('now')) AS created_utc
FROM datasets d
WHERE d.zarr_path IS NOT NULL
  AND TRIM(d.zarr_path) <> ''
  AND NOT EXISTS (
    SELECT 1
    FROM dataset_locations dl
    WHERE dl.dataset_id = d.dataset_id
      AND dl.path = d.zarr_path
  );
```

## Example SQL Templates

- [ ] Query: all subjects from a cross in a given experiment at a specific DPF.

```sql
SELECT
  s.subject_id,
  r.recording_id,
  rs.dpf_at_acquisition,
  pr.protocol_name
FROM subjects s
JOIN dishes d ON d.dish_id = s.dish_id
JOIN crosses c ON c.cross_id = d.cross_id
JOIN recording_subjects rs ON rs.subject_id = s.subject_id
JOIN recordings r ON r.recording_id = rs.recording_id
LEFT JOIN protocol_runs pr ON pr.recording_id = r.recording_id
WHERE c.cross_id = :cross_id
  AND rs.dpf_at_acquisition = :dpf
  AND pr.protocol_name = :experiment_name;
```

- [ ] Query: subjects from a cross with the same experiment at 5, 6, and 7 dpf.

```sql
SELECT
  s.subject_id
FROM subjects s
JOIN dishes d ON d.dish_id = s.dish_id
JOIN crosses c ON c.cross_id = d.cross_id
JOIN recording_subjects rs ON rs.subject_id = s.subject_id
JOIN recordings r ON r.recording_id = rs.recording_id
JOIN protocol_runs pr ON pr.recording_id = r.recording_id
WHERE c.cross_id = :cross_id
  AND pr.protocol_name = :experiment_name
  AND rs.dpf_at_acquisition IN (5, 6, 7)
GROUP BY s.subject_id
HAVING COUNT(DISTINCT rs.dpf_at_acquisition) = 3;
```

- [ ] Query: escaped trials in chaser sessions for a cross at selected DPFs.

```sql
SELECT
  t.trial_id,
  s.subject_id,
  rs.dpf_at_acquisition,
  t.outcome
FROM trials t
JOIN trial_subjects ts ON ts.trial_id = t.trial_id
JOIN subjects s ON s.subject_id = ts.subject_id
JOIN dishes d ON d.dish_id = s.dish_id
JOIN crosses c ON c.cross_id = d.cross_id
JOIN recordings r ON r.recording_id = t.recording_id
JOIN recording_subjects rs
  ON rs.recording_id = r.recording_id
 AND rs.subject_id = s.subject_id
WHERE c.cross_id = :cross_id
  AND t.session_type = 'chaser'
  AND t.outcome = 'escaped'
  AND rs.dpf_at_acquisition IN (5, 6, 7);
```

## Backfill + Validation

- [x] Add maintenance command:
  - `--backfill-recording-entities`
  - `--dry-run`
  - inserted/updated/skipped/error counters

- [x] Backfill existing `/nvme1/palette_registry.sqlite` (initial run completed).

- [x] Integrity checks (initial Phase 1 scope):
  - every source dataset linked to one recording
  - referenced recording row exists
  - recording rows declare `recording_type` + `artifact_schema_id`
  - `behavior_v1` required artifact types are present
  - command: `--check-integrity`

- [x] Add one-shot reconciliation command:
  - `--reconcile-registry`
  - runs: reconcile missing dataset statuses -> delete missing dataset rows -> integrity check
  - supports `--dry-run` and `--list-limit`

- [x] Integrity checks (remaining scope):
  - [x] derived datasets either:
    - link to one recording when single-source, or
    - use `recording_id NULL` only when lineage has 2+ distinct parent recordings
  - [x] subject/protocol links consistent
  - [x] key query views return expected rows
    - Implemented as required-view existence + smoke-query checks for
      `recording_overview`, `merged_training_datasets`, `dataset_lineage_current`,
      `keypoint_quality_current`, and `detect_quality_current`.

### Schema Migration Tooling

`CREATE TABLE IF NOT EXISTS` cannot handle `ALTER TABLE` for existing databases.
To support incremental schema evolution:

- [x] Add `schema_version` table with versioned history for applied migrations.
- [x] Maintain ordered migration scripts (SQL or Python) applied incrementally.
  - Each migration bumps `schema_version` and applies DDL/DML changes.
  - Migrations are idempotent where possible.
- [x] Auto-run pending migrations on registry open (in `Registry.__init__`).
  - Implemented in `Registry._apply_schema_migrations()` with:
    - ordered migration list
    - legacy bootstrap for pre-versioned registries
    - transaction-scoped migration application and version bump on success
  - Covered by unit tests for:
    - new-registry initialization
    - legacy bootstrap
    - failed migration does not advance `schema_version`.
  - Current migration list includes:
    - `v1`: `initial_registry_schema`
    - `v2`: `reserved_noop_template` (explicit append-only template slot).

### Migration PR Template

Use this checklist for every schema-changing PR:

- [ ] **Change summary**
  - migration version:
  - migration name:
  - tables/views/indexes touched:
  - backward-compat notes:

- [ ] **Preflight evidence (from a real registry copy)**
  - integrity before apply:
  - `schema_version` / `PRAGMA user_version` before:
  - DB backup path:

- [ ] **Apply evidence**
  - command used to open/apply migrations:
  - `schema_version` / `PRAGMA user_version` after:
  - any migration logs/errors:

- [ ] **Post-apply validation**
  - integrity after apply:
  - key view smoke checks (`recording_overview`, `dataset_lineage_current`, etc.):
  - targeted pytest subset:

- [ ] **Rollback plan**
  - explicit restore command from backup:
  - expected post-rollback integrity result:

- [ ] **Docs updated**
  - `docs/registry_repair_playbook.md` if operator steps changed
  - `docs/recording_registry_normalization_todo.md` status updates

## Open Questions

- [ ] How do we assign/validate stable `subject_id` when identity is uncertain?
  (Partially resolved: `fish_id` UUID from acquisition maps to `subject_id`;
  uncertainty arises when subjects are re-identified across dishes or sessions
  without UUID continuity.)
