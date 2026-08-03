# Subject identity correction boundary

Status: design boundary; no production correction publisher is implemented.

Last reviewed: 2026-08-03.

## Decision

An acquisition-time subject-ID mistake is provenance. It must not be repaired
by rewriting the raw H5, editing a completed subject-metadata run, directly
changing registry identity rows, or layering an attribute that canonical
readers do not validate.

Palette's current authorities are the selected immutable
`analysis/subject_metadata_runs/<run>` and
`analysis/experiment_setup_runs/<run>` pair. A future correction must publish
and validate a new authority pair with explicit lineage from the selected
parent pair. Existing scientific runs and capture-time records remain
unchanged.

MetaZebrobot remains the authority that mints and registers biological subject
identities. Palette must not invent a replacement UUID or treat a locally
supplied UUID as proof of registration.

## Retired branch design

The historical `agent/subject-metadata-corrections-20260718` branch proposed:

- `analysis_metadata.subject_metadata_corrections` as an append-only overlay;
- a `subject_metadata_corrections` SQLite table;
- direct updates to legacy `provenance.fish_id`; and
- a repair CLI that rejected already-normalized `recording_subjects` rows.

That implementation predates the canonical subject/setup run authorities.
Current resolvers do not consume its overlay, and normalized
`recording_subjects` are now the normal registry projection. The old modules,
migration, and repair CLI must therefore not be ported into production.

The branch contained only a placeholder repair template. It did not contain
approved recording IDs or replacement subject UUIDs, so retiring the
implementation does not discard an applied correction or production repair
specification.

## Required successor contract

Each correction publication must bind at least:

- recording and archive identity;
- the selected parent subject run path and SHA-256;
- the selected parent setup run path and SHA-256;
- the capture-time subject record SHA-256 and source H5 fingerprint;
- original and corrected canonical subject UUIDs;
- a stable correction ID and optional superseded correction ID;
- a constrained reason code and human-readable reason;
- operator identity and an immutable UTC review timestamp; and
- the MetaZebrobot record ID plus an explicit registration-verification
  assertion.

The first supported reason should remain narrowly scoped to
`acquisition_subject_id_reuse`. General metadata editing is a separate design.

The successor subject schema must preserve the capture-time metadata and make
the correction lineage executable to validate. The successor setup must bind
the exact corrected subject run and digest. Both artifacts remain
selector-ineligible until their records, digests, lineage, and mutual binding
have passed validation.

## Selection and registry rules

The two current parents have independent selectors. A production correction
must not expose an interval in which the selected setup and subject runs
disagree. Before implementation, Palette must add either:

1. one authoritative pair selector that names both immutable runs and digests;
   or
2. a recovery-safe, offline activation protocol whose pending/complete receipt
   lets all readers fail closed during an interrupted transition.

After activation, registry state is rebuilt from the selected canonical pair.
The correction workflow must not make direct identity edits to
`provenance.fish_id`, `subjects`, or `recording_subjects` as an independent
authority. Registry backup, transactional refresh, and exact post-refresh
comparison remain required operational safeguards.

Existing derived runs are immutable and are not rewritten. Exports that use a
corrected identity must retain both the derived run's source-bound subject/setup
digests and the active correction authority so acquisition-time provenance and
current biological identity remain distinguishable.

## Implementation checklist

- [ ] Freeze a versioned correction manifest and exact canonical digest rules.
- [ ] Add strict builders and parsed validators, including canonical UUIDv4,
  reason registry, correction-chain, parent-digest, and MetaZebrobot checks.
- [ ] Add an immutable corrected subject-metadata successor schema.
- [ ] Publish the paired experiment-setup successor without activating either.
- [ ] Add a single pair-selection or recovery-safe activation contract.
- [ ] Add a dry-run-first planner with collision, stale-parent, and duplicate-ID
  checks.
- [ ] Refresh the registry only from the activated pair and verify all projected
  rows transactionally.
- [ ] Cover legacy-only archives by first publishing canonical subject/setup
  authorities, never by adding a compatibility overlay.
- [ ] Test interrupted activation, stale consolidated metadata, successor
  chains, normalized multi-subject recordings, idempotent replay, and rollback.
- [ ] Obtain and independently verify real MetaZebrobot replacement IDs before
  authorizing any production repair.

## Current incident boundary

The historical investigation mentioned four RedScare recordings affected by
subject-ID reuse. No repository repair should name or alter those recordings
until distinct replacement UUIDs have been created and verified in
MetaZebrobot and the successor contract above is implemented.
