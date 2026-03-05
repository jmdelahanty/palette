# Registry Multi-Source Provenance TODO

Purpose: turn the proposed design into an implementation plan that supports production recordings from multiple source systems (with or without Citrus/H5 metadata), while keeping registry lineage and operator workflows clean.

Design reference:
- `docs/registry_multi_source_provenance_design.md`

Date anchored: 2026-03-05.

## Decision Snapshot (Current)

- [x] Keep `recording_intent` independent from metadata completeness.
- [x] Treat non-Citrus recordings as first-class (`production + partial/minimal`), not automatically `ad_hoc`.
- [x] Keep canonical dataset identity rules unchanged (`recording_id:path-hash` style canonical IDs where applicable).
- [x] Use source adapters to map external schemas into canonical core + source payload envelope.

## Phase 0: Contract Lock

- [ ] Finalize vocabulary for:
  - `recording_intent`: `production | training | ad_hoc`
  - `metadata_level`: `full | partial | minimal`
- [ ] Define mutable/immutable policy:
  - whether `recording_intent` can change,
  - how `metadata_level` upgrades are audited.
- [ ] Define minimum required core fields for registration when metadata is minimal.
- [ ] Define adapter output contract (`source_system`, `source_schema_version`, validation payload).

## Phase 1: Schema Additions (Additive)

- [ ] Add registry columns (or equivalent table-backed representation) for:
  - `recording_intent`
  - `metadata_level`
  - `source_system`
  - `source_schema_version`
  - `source_payload_json`
  - `metadata_sources_json`
- [ ] Add constraints/checks for controlled vocab values.
- [ ] Keep migrations backward compatible with current readers/views.
- [ ] Add schema tests and migration tests.

## Phase 2: Writer/Registration Updates

- [ ] Update one-off detect registration flow to auto-upsert source rows with explicit intent/metadata level.
- [ ] Update one-off pose/eye-mask registration flows the same way.
- [ ] Update import/scan flows to set intent + metadata level defaults explicitly.
- [ ] Ensure runtime step-status writers cannot fail FK due to missing source dataset rows in one-off mode.

## Phase 3: Source Adapter Layer

- [ ] Create adapter interface for source metadata ingestion.
- [ ] Implement initial adapters:
  - Citrus/H5 adapter
  - Generic/manual/video-probe adapter
- [ ] Persist source envelope fields (`source_*`, payload JSON) through adapter outputs.
- [ ] Add deterministic adapter tests (same input -> same normalized output).

## Phase 4: Backfill Existing Registry

- [ ] Add maintenance command to classify existing rows:
  - infer `recording_intent`
  - infer `metadata_level`
  - set `source_system` for known sources when determinable
- [ ] Add dry-run mode with summary counts by inferred class.
- [ ] Add apply mode with audit summary (inserted/updated/unchanged).
- [ ] Verify no canonical identity churn during backfill.

## Phase 5: Query and UX Defaults

- [ ] Update `registry_query` filters:
  - add `--recording-intent`
  - add `--metadata-level`
  - add `--include-ad-hoc` convenience behavior where relevant
- [ ] Update `check_training_registry` views to surface intent/metadata-level.
- [ ] Make default operator views exclude `ad_hoc` unless explicitly requested.
- [ ] Add docs/examples for production-without-H5 discovery queries.

## Phase 6: Training/Lineage Integration

- [ ] Ensure training dataset builders accept source rows regardless of source system when canonical core is valid.
- [ ] Enforce that training lineage links only reference canonical dataset IDs.
- [ ] Include source-system and completeness summaries in training data-card aggregation outputs.
- [ ] Add validation command/check to flag lineage rows that rely on legacy/non-canonical IDs.

## Phase 7: Governance and Validation

- [ ] Add integrity checks for:
  - controlled vocab validity,
  - parseable source payload,
  - stage-status dataset linkage integrity,
  - canonical identity rules.
- [ ] Define archival/deletion policy for `ad_hoc` rows.
- [ ] Add operator runbook section for handling external source schema drift.

## Open Questions

- [ ] Should `recording_intent` be immutable after first set, except explicit audited override?
- [ ] Should `metadata_level` be stored directly, computed dynamically, or both?
- [ ] Do we need a dedicated adapter audit table beyond JSON payload fields?
- [ ] What is the minimum metadata threshold for classifying a non-Citrus recording as `production`?

## Suggested First Implementation Slice

- [ ] Implement Phase 1 schema additions.
- [ ] Implement detect one-off auto-registration updates (Phase 2 subset).
- [ ] Implement backfill dry-run/apply for intent + metadata level (Phase 4 subset).
- [ ] Add registry query filters for intent/ad-hoc inclusion (Phase 5 subset).

## Exit Criteria for v1

- [ ] Non-Citrus production recordings can be registered and queried as production without H5 metadata.
- [ ] One-off runs can be registered cleanly without FK step-status failures.
- [ ] Default operational views remain clean (no accidental ad-hoc pollution).
- [ ] Training/model lineage remains canonical and reproducible across source systems.
