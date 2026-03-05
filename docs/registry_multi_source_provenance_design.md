# Registry Multi-Source Provenance Design (Proposed)
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-03-05
-->

Purpose: define how Palette registry should represent recordings from multiple upstream systems (Citrus and non-Citrus), while keeping stage tracking, lineage, and training provenance consistent.

Date anchored: 2026-03-05.

## Problem Statement

Current workflows implicitly assume one dominant metadata source (Citrus/H5-style). We now have legitimate recordings that:

- are production-quality but have no H5 experiment metadata,
- may come from other acquisition systems with different schemas,
- should still participate in analysis, review, training lineage, and model provenance.

We need a design that keeps provenance strict without forcing all inputs into Citrus-specific fields.

## Goals

- Keep one canonical registry identity and lineage model across all sources.
- Decouple recording validity from Citrus/H5 availability.
- Support progressive metadata enrichment over time.
- Preserve clean operator views by separating intent (`production`, `training`, `ad_hoc`) from metadata completeness (`full`, `partial`, `minimal`).
- Keep downstream training/model provenance reproducible regardless of source system.

## Non-Goals

- Replacing existing stage writers.
- Full schema migration implementation in this phase.
- Defining every source-specific field for every future external system.

## Core Design

### 1) Two-Axis Recording Classification

Each source recording row should be classified on two independent axes:

- `recording_intent`: `production | training | ad_hoc`
- `metadata_level`: `full | partial | minimal`

Interpretation:

- `production + partial` is valid and first-class (no H5 required).
- `ad_hoc` means workflow intent, not missing metadata.
- Metadata completeness must never silently alter intent.

### 2) Canonical Core vs Source Payload

Use a stable canonical core that all systems can populate:

- canonical dataset identity (`dataset_id`, `session_uuid`, `path_hash`, `zarr_use`, `recording_id`)
- path/timestamps/status
- stage status/coverage/review lineage
- minimal video-derived fields when available (resolution/fps/codec)

Keep source-specific metadata in a source payload envelope:

- `source_system` (for example: `citrus`, `legacy_lab`, `manual_import`)
- `source_schema_version`
- `source_payload_json` (namespaced raw/normalized source content)
- `metadata_sources_json` (what was derived from filename/video probe/manifest/api)

Canonical columns remain query surface for operational tooling; source payload is audit/debug surface.

### 3) Source Adapter Contract

Add adapter modules per source system that map inbound metadata to:

- canonical core fields (best effort),
- source payload envelope,
- validation result (`valid`, `warnings`, `errors`).

Adapter invariants:

- must not mutate dataset identity rules,
- must declare parser/version used,
- must be deterministic for same input.

### 4) Progressive Enrichment

Allow rows to move from `minimal -> partial -> full` without changing identity:

- initial registration can be video-only,
- later enrichment updates metadata fields and payloads,
- provenance history should capture enrichment source/time/tool.

### 5) Query and UX Policy

Default operator views:

- include `production` and `training`,
- exclude `ad_hoc` unless explicitly requested.

Do not filter by H5 presence. Filter by intent + stage/review/quality state.

## Registry Representation (Proposed Fields)

Minimal additions (names may be refined during implementation):

- `datasets.recording_intent` TEXT NULL
- `datasets.metadata_level` TEXT NULL
- `datasets.source_system` TEXT NULL
- `datasets.source_schema_version` TEXT NULL
- `datasets.source_payload_json` TEXT NULL
- `datasets.metadata_sources_json` TEXT NULL

Constraints:

- `recording_intent` in (`production`, `training`, `ad_hoc`) when non-null.
- `metadata_level` in (`full`, `partial`, `minimal`) when non-null.

Compatibility:

- existing rows default to inferred values during migration (`production` + inferred level where possible),
- legacy rows remain readable until backfill completes.

## Identity and Lineage Invariants

- Source-recording dataset identity remains canonical (`<recording_id>:z<path_hash>` where applicable).
- No source adapter can emit non-canonical IDs.
- Training datasets and model runs must reference canonical source dataset IDs only.
- Lineage links must not depend on Citrus-specific fields.

## One-Off Workflow Policy

One-off tools should be able to register rows with:

- `recording_intent=ad_hoc`,
- `metadata_level=minimal` or `partial`,
- stable identity and stage status writes enabled.

This removes FK failures and keeps one-off provenance queryable while preserving clean default production views.

## Validation and Governance

Introduce source-agnostic registry checks:

- required core fields present,
- intent/metadata-level valid,
- identity canonical,
- source envelope parseable,
- stage rows link to existing dataset IDs.

Source-specific checks remain adapter-owned and should report warnings/errors separately from core validity.

## Migration Strategy

### Phase 1: Additive schema + read compatibility

- add new columns/tables (if needed) without changing existing behavior,
- keep all existing queries working.

### Phase 2: Writer updates

- one-off detect/pose/eye-mask entry points write intent/metadata-level/source envelope,
- registry import/scan paths set defaults explicitly.

### Phase 3: Backfill existing rows

- classify existing source recordings,
- set conservative defaults (`production` + inferred metadata level),
- preserve audit trail of migrated values.

### Phase 4: Query defaults + tooling

- registry_query/check_training_registry add intent filters (`--include-ad-hoc`),
- views and reports default to non-ad-hoc unless requested.

## Open Questions

- Should `recording_intent` be immutable after first write, or editable with audit trail?
- Do we need a separate source-adapter audit table for parse errors/history beyond JSON payload?
- Should `metadata_level` be computed dynamically from completeness checks or stored explicitly (or both)?
- What minimum metadata is required for `production` classification in non-Citrus pipelines?

## Recommended Initial Defaults

- New one-off registrations: `ad_hoc + minimal`.
- New non-Citrus production imports: `production + partial` (or `minimal` if only video probe fields exist).
- Citrus/H5 imports: `production + full` when parser confirms expected metadata.

## Acceptance Criteria (Design Phase)

- Operators can explain why a row is `ad_hoc` vs `production` without referencing H5.
- Non-Citrus recordings can be first-class in analysis and training lineage.
- Default registry views remain clean for production operations.
- Provenance and model lineage remain reproducible across source systems.
