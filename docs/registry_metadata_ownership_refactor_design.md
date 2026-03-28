# Registry Metadata Ownership Refactor Design
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-03-18
-->

Purpose: reduce duplicated canonical metadata in the Palette registry while
preserving two requirements that remain explicitly in scope:

- same biological subject across multiple recordings,
- exact derivation provenance across pipeline stages
  (`subject_mask <- keypoints <- crop <- detect <- recording`).

Date anchored: 2026-03-18.

## Why This Refactor Exists

The current registry design has good goals but too many co-equal homes for the
same facts.

Examples of current duplication:

- recording context is split across `recordings` and `provenance`,
- biological metadata exists both as denormalized dataset fields in
  `provenance` and as normalized lineage entities
  (`subjects`, `recording_subjects`, `dishes`, `crosses`),
- query/profile tables persist repeated context fields that could be projected
  from canonical owners.

This creates three forms of drag:

1. write-time duplication,
2. integrity checks for mismatches between duplicated owners,
3. reader ambiguity about which table is authoritative.

The refactor keeps the useful structure and removes unclear ownership.

## Confirmed Requirements

### 1) Cross-recording subject identity is real

The registry must support the same `subject_id` appearing in multiple
recordings. This means a normalized `subjects` entity remains justified.

### 2) Stage derivation provenance is first-class

For produced runs, Palette must be able to answer:

- which run created this output,
- which upstream run(s) it consumed,
- which dataset/recording ultimately supplied the source data.

This is especially important for chains such as:

- `subject_mask_run -> keypoint_run -> crop_run -> detect_run -> source recording`

The refactor must not weaken this.

### 3) On-disk provenance remains canonical

Run/stage derivation provenance remains canonical in the Zarrs themselves.

Hard rule:

- every run keeps its full provenance contract on disk,
- the registry may project only the subset of lineage needed for queries,
  summaries, or operator workflows,
- registry lineage projections are derived caches, not the canonical source of
  truth,
- if a registry lineage projection disagrees with on-disk Zarr provenance, the
  Zarr provenance wins and the registry projection should be repaired from it.

This refactor is explicitly not a move away from on-disk provenance.

## Non-Goals

- Removing run-level provenance attrs from Zarr outputs.
- Flattening biological lineage into a single JSON blob.
- Replacing registry query/projection tables with ad hoc filesystem traversal.
- Solving every future external-source schema problem in this document.
  Multi-source recording intake remains covered separately by
  `docs/registry_multi_source_provenance_design.md`.

## Design Principles

1. One semantic field family should have one canonical owner.
2. Query projections are allowed, but they must be refreshable derived data.
3. Cross-recording subject identity and run lineage are worth keeping even if
   they add schema objects.
4. Dataset-level convenience fields are acceptable only when they are clearly
   derived from canonical owners.
5. Compatibility fallbacks may remain for a migration period, but they must not
   remain ambiguous long-term.

## Canonical Ownership Model

| Metadata family | Example fields | Canonical owner | Allowed derived projections | Notes |
| --- | --- | --- | --- | --- |
| Dataset identity | `dataset_id`, `session_uuid`, `recording_id`, `zarr_path`, `artifact_kind`, `zarr_origin`, `zarr_use`, `status`, `last_seen_utc` | `datasets` | overview/status/profile views | One row per concrete dataset artifact. |
| Recording context | `recording_type`, `recording_subtype`, `behavior_mode`, `artifact_schema_id`, `rig_id`, `arena_id`, `camera_id`, `canvas_name`, `protocol_name`, `dish_design`, `started_utc`, `recording_path` | `recordings` | dataset context views, status views | Recording-scoped facts should not be co-owned by `provenance`. |
| Subject identity across recordings | `subject_id`, `dish_id`, `species`, `sex` | `subjects` | `recording_subject_overview`, dataset subject context views | Required because same subject may span recordings. |
| Subject membership in a recording | `recording_id`, `subject_id`, per-recording `dish_id`, `cross_id`, `dpf_at_acquisition`, `genotype`, `line_strain` | `recording_subjects` | `recording_subject_overview`, dataset subject context views | Canonical home for biological identity as observed in one recording. |
| Dish / cross lineage | `dish_id`, `cross_id`, `genotype`, `line_strain`, `parents_json` | `dishes`, `crosses` | `recording_subject_overview` | External identity lineage, not tracking identity. |
| Dataset-local technical snapshot | `fps`, codec/compression fields, exposure/gain, camera probe fields, `has_images_ds`, `downsample_formats_json`, `snapshot_status`, `snapshot_missing_json`, `protocol_hash` | `provenance` | dataset context views, training preflight | Keep `provenance` focused on technical snapshot and capture completeness. |
| Run derivation provenance | `provenance.contract`, `provenance.stage`, `provenance.inputs`, `source_*_run`, `source_*_path`, stage params | on-disk run attrs in Zarr | registry quality/performance/profile projections | Canonical lineage stays with the run itself. Registry stores only query-critical projections. |
| Operational stage status | `step_name`, `status`, `run_name`, `coverage_pct`, `review_status_json` | `recording_step_status` | `recording_step_status_latest`, `recording_step_status_wide` | Operational ledger, not canonical acquisition metadata. |
| Query-critical derived summaries | profile/quality/performance metrics | dedicated registry projection tables | latest/current/overview views | These are caches, not owners of base identity/context. |

## Field-Family Decisions

### Recording Context

The following fields become canonical in `recordings` for source recordings:

- `rig_id`
- `arena_id`
- `camera_id`
- `canvas_name`
- `protocol_name`
- `dish_design`
- `recording_type`
- `recording_subtype`
- `behavior_mode`
- `artifact_schema_id`
- `started_utc`
- `recording_path`

Implication:

- duplicated copies in `provenance` are compatibility fields during migration,
  not long-term canonical owners.

### Biological Identity and Lineage

Canonical biological identity should be expressed through normalized lineage
tables:

- `subjects`
- `recording_subjects`
- `dishes`
- `crosses`

Important naming decision:

- `subject_id` is the canonical biological identity key.
- legacy `fish_id` should be treated as a compatibility alias or source import
  field, not a second canonical identity namespace.

Operator-facing interpretation:

- normalized registry tables, canonical views, and integrity rules should speak
  in terms of `subject_id`.
- acquisition-era snapshots may still carry `fish_id`; during migration that
  value may remain readable only as a compatibility field such as
  `legacy_fish_id`.
- query surfaces may keep `fish_id` as a compatibility filter name for users,
  but it should resolve to canonical `subject_id` first and only fall back to
  legacy provenance aliases when normalized subject lineage is absent.

`recording_subjects` remains necessary because the same `subject_id` may appear
in multiple recordings, and some attributes are recording-specific:

- `dpf_at_acquisition`
- observed `dish_id`
- observed `cross_id`
- observed genotype / strain metadata

### Dataset-Local Technical Snapshot

`provenance` remains valuable, but its scope narrows to dataset-local technical
capture/probe/snapshot fields.

Examples that remain appropriate in `provenance`:

- video probe / codec / compression metadata,
- exposure / gain / frame-rate snapshot,
- camera hardware probe fields,
- `has_images_ds`, `has_images_ds_rgb`, `downsample_formats_json`,
- `snapshot_status`, `snapshot_missing_json`,
- `protocol_hash`.

Examples that should stop being canonical in `provenance`:

- `rig_id`
- `arena_id`
- `camera_id`
- `canvas_name`
- `protocol_name`
- `dish_design`
- `fish_id`
- `dish_id`
- `cross_id`
- `genotype`
- `line_strain`
- `dpf_at_acquisition`

Those may remain readable during migration, but the registry should stop
treating them as co-equal truth with `recordings` and normalized lineage.

### Stage Derivation Provenance

Stage lineage remains canonical on disk in run attrs.

Required behavior:

- every stage writes canonical run provenance,
- provenance records direct inputs,
- source run references remain explicit,
- registry projections may copy query-critical lineage pointers only when they
  are derived directly from canonical run attrs.

This means columns such as these remain acceptable in projection tables when
needed for queryability:

- `source_detect_run`
- `source_refined_run`
- `source_crop_run`
- `source_keypoint_run`
- `source_eye_mask_run`

These are not the problem. The problem is copying unrelated recording or
biological context into every projection table.

Additional ownership rule:

- the registry must never become the only surviving copy of stage derivation
  lineage,
- full run provenance stays in Zarr attrs even when equivalent query-critical
  pointers are projected into registry tables or views.

## Allowed Denormalization Policy

### Allowed

- SQL views that join canonical tables into operator-friendly surfaces.
- Derived projection tables that cache metrics or query-critical lineage from
  run attrs.
- Aggregated convenience fields that are explicitly refreshable from canonical
  sources.

### Not Allowed

- treating two tables as equally canonical for the same semantic field,
- writing a field into multiple canonical tables and relying on integrity
  checks to discover drift later,
- adding profile-table context columns just because a consumer wants an easier
  query once.

## Proposed Query Surfaces

### 1) Keep `recording_subject_overview`

This remains the canonical join surface for per-recording biological lineage.

It already answers:

- which subject was in which recording,
- which dish/cross lineage applied,
- what genotype / DPF / species / sex were associated,
- what recording context applied.

### 2) Add `dataset_context_current`

Add a single registry view that joins:

- `datasets`
- `recordings`
- derived subject-lineage summary for the dataset's `recording_id`
- technical snapshot fields from `provenance`

This view should become the standard read surface for dataset-scoped queries.

Recommended shape:

- dataset identity fields from `datasets`,
- recording context fields from `recordings`,
- technical snapshot fields from `provenance`,
- biological summary fields derived from `recording_subject_overview`,
- explicit legacy compatibility fields for acquisition snapshots that have not
  yet been normalized, for example `legacy_fish_id`.

Biological summary should support both:

- scalar convenience fields when unambiguous,
- aggregate JSON/list fields when multiple subjects exist.

Examples:

- `subject_count_recorded`
- `subject_ids_json`
- `dish_ids_json`
- `cross_ids_json`
- `genotypes_json`
- `dpf_values_json`
- compatibility-only `legacy_fish_id`, `legacy_dish_id`, `legacy_cross_id`,
  `legacy_genotype`, `legacy_dpf_at_acquisition`
- convenience scalar `subject_id`, `dish_id`, `cross_id`, `genotype`,
  `dpf_at_acquisition` only when a single distinct value exists

This avoids forcing a false single-value biology model onto multi-subject
recordings.

Recommended subject-context semantics:

- `subject_context_source='normalized'` means canonical lineage came from
  `recording_subjects` / `recording_subject_overview`.
- `subject_context_source='legacy_provenance'` means no normalized lineage was
  available and only compatibility snapshot biology exists.
- canonical biological fields should remain null when only
  `legacy_provenance` data is present.

### 3) Make status and profile views read from context views

Views such as these should read from `dataset_context_current` rather than
pulling context fields directly from `provenance`:

- `recording_step_status_latest`
- `recording_step_status_wide`
- dataset latest/current profile views
- operator registry query surfaces

## Profile / Projection Table Policy

Projection tables should keep:

- dataset/run identity,
- run-local metrics,
- query-critical lineage pointers,
- projection payload JSON if needed.

Projection tables should stop owning repeated context fields such as:

- `rig_id`
- `camera_id`
- `arena_id`
- `dish_design`
- `canvas_name`
- `protocol_name`
- `genotype`
- `dpf_at_acquisition`

Those values should come from joined context views.

Migration note:

- existing columns may remain temporarily for compatibility,
- new writes should stop treating them as canonical,
- long-term goal is to remove them from writers and eventually from schema.

## Integrity Policy After Refactor

Integrity should validate ownership and lineage, not duplicated equality checks.

Examples of checks that remain desirable:

- every source dataset links to a valid `recording_id`,
- every `recording_subjects.subject_id` resolves to `subjects.subject_id`,
- every `recording_subjects.dish_id` resolves to `dishes.dish_id`,
- every projected source run field resolves to a real run or accepted legacy
  fallback,
- every dataset context view row is derivable without ambiguity bugs.

Checks that should become unnecessary after migration:

- `recordings.protocol_name != provenance.protocol_name`,
- `recordings.dish_design != provenance.dish_design`,
- other mismatch checks that only exist because ownership is duplicated.

## Compatibility / Migration Policy

### Phase A: Additive read surfaces

- introduce `dataset_context_current`,
- keep legacy columns and existing readers working.

### Phase B: Reader migration

- move queries and views to the new canonical read surfaces,
- stop adding new consumers of duplicated fields.

### Phase C: Writer migration

- stop dual-writing recording context into `provenance`,
- stop dual-writing biological identity into `provenance` as canonical query
  fields,
- keep legacy values populated only where needed for compatibility.

### Phase D: Projection cleanup

- stop writing repeated context columns into profile tables,
- use joins/views instead.

### Phase E: Integrity cleanup

- remove mismatch checks that only defend duplicated ownership,
- replace them with canonical-owner validation.

### Phase F: Deprecation / removal

- document deprecated columns,
- remove them only after readers and backfills no longer depend on them.

## Acceptance Criteria

- Cross-recording subject identity remains queryable and normalized.
- A user can trace a derived run back through upstream runs to the exact source
  recording without consulting duplicate registry fields.
- Full stage derivation provenance remains present and canonical in the Zarrs.
- Recording context has one canonical registry owner.
- Dataset-local technical snapshot has one canonical registry owner.
- Profile/quality/performance tables become clearly derived query caches rather
  than shadow canonical metadata stores.
- Integrity checks get simpler because they validate ownership and lineage, not
  duplicated equality.

## Related Docs

- `docs/pipeline_metadata_boundaries.md`
- `docs/provenance_contract_draft.md`
- `docs/recording_manifest_contract.md`
- `docs/registry_data_governance_policy.md`
- `docs/registry_multi_source_provenance_design.md`
- `docs/recording_registry_normalization_todo.md`
