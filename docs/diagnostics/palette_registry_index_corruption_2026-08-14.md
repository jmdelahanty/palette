# Palette registry index-corruption incident, 2026-08-14

## Outcome

The canonical Palette registry had secondary-index corruption, not lost table
records. It was repaired while quiescent by rebuilding indexes in a separate
copy and atomically replacing the canonical file. No analysis Zarr, source
recording, or registry table payload was rewritten by the repair.

Canonical registry:

```text
/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

Exact preserved corrupt file:

```text
/groups/johnson/johnsonlab/jeremy/registries/backups/palette_registry_corrupt_indexes_before_rebuild_replace_20260814T015145Z.sqlite
```

## Evidence

- The backup immediately before the goodbatbadbat rescan,
  `palette_registry_before_goodbatbadbat_rescan_20260813T215209Z.sqlite`, returned
  `ok` from complete `PRAGMA integrity_check`.
- The post-rescan canonical registry returned `ok` from `PRAGMA quick_check` but
  produced 100 rows from complete `PRAGMA integrity_check` (SQLite's reporting
  limit for that invocation).
- Reported errors were missing or wrong entries in
  `recording_step_status`, `recording_step_status_history`, and associated
  primary/secondary indexes.
- The exact corrupt file has SHA-256
  `f026be5c68442ec0c3b00655391c5851a4026663dddb5d724790d3edb016f927`.
- `REINDEX` on a separate SQLite-consistent copy made complete
  `integrity_check` return `ok`.
- An index-independent, rowid-ordered logical hash matched before and after for
  all 58 tables and all 48,804 rows:
  `f86564f93c9ed210302cba641ef5f50d549e3129627ed34735db5c9042dd9456`.
- The repaired canonical file has SHA-256
  `44b150124886d10fcc15b28bd369526bb709099674f89a44bc41293d100f668b` and
  complete `integrity_check=ok`.

An apparent dataset/recording lookup mismatch disappeared after index rebuild,
confirming it was an index-lookup artifact rather than contradictory table
payload.

## Cause assessment

The clean pre-rescan backup and corrupt post-rescan file bound the incident to
the rescan/concurrent-writer window. They do not identify one process with
enough certainty to assign a unique triggering command.

The failure class is already documented in Palette: the canonical SQLite file
is on a multi-host NFS filesystem, where SQLite rollback-journal locking cannot
be treated as a reliable cross-host writer funnel. A previous production
incident had the same duplicate-page, out-of-order-rowid, and wrong-index-entry
signature. WAL remains inappropriate because its shared-memory index is not
cross-host safe.

## Operational decision

- Backup validation uses complete `PRAGMA integrity_check` and
  `PRAGMA foreign_key_check`; `quick_check` is not an acceptance gate.
- The geometry-approval terminal registry step does not mutate the shared
  SQLite file in place. It snapshots and validates the source, mutates a
  node-local candidate, checks that the canonical source did not change,
  validates the shared-filesystem staging copy, and publishes by atomic rename.
- Every such publication preserves a request-bound pre-write SQLite backup.
- Parallel workers do not write the registry. Unrelated legacy direct writers
  must not overlap shadow publication; a repository-wide single-writer funnel
  remains the systemic follow-up.
