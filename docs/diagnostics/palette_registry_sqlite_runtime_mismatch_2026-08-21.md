# Palette registry SQLite runtime mismatch, 2026-08-21

## Outcome

The canonical registry was logically intact but had a physical page-accounting
defect visible to Palette's SQLite 3.52.0 runtime. The workstation `sqlite3`
3.45.1 command reported `ok`, while Palette's complete integrity check reported
`Page 8544: never used`. The system command's result was therefore not valid
acceptance evidence for the runtime that performs Palette registry writes.

The source was repaired while quiescent with `VACUUM INTO` under Palette's
runtime. Source and candidate had the same logical `.sha3sum --schema` digest:

```text
851e42f26bbc840309a97bedac50c2c5a9b86f4b13f72e5c0c96647f
```

The guarded publication acquired the registry writer lock, refused SQLite
sidecars, rechecked exact hashes, retained the original bytes as a 0444 backup,
validated a shared-filesystem staging copy, atomically replaced the canonical
file, fsynced the parent directory, and validated the published file again.

Canonical and backup evidence:

```text
canonical repaired SHA-256:
84e96ff1bff129ef17108e335bcae6b9c7eac747ce584f1450f178820694099d

exact original SHA-256:
57a9873810060ec8ed9d003abf285008004529c9f45ec141316865105be470b2

exact original backup:
/groups/johnson/johnsonlab/jeremy/registries/backups/palette_registry.pre_clipped_cam93_96.20260821T1830Z.sqlite
```

The subsequent safe-shadow registration of the four clipped master archives
completed with four active `rolling_clips` datasets, zero scan errors, zero
missing rows, and clean source, candidate, staged, and published validations.

## Permanent guardrail

- `RegistryValidation` records the SQLite engine version, Python sqlite module
  version, interpreter path, and validation backend in every safe-shadow
  publication receipt.
- `scripts/py -m fisheye.utils.registry_integrity` is the maintained complete
  integrity and foreign-key acceptance command.
- `scripts/backup_palette_registry.sh` no longer invokes the system `sqlite3`
  program. Backup creation and source/backup validation use the same SQLite
  library loaded by `scripts/py`.
- Repository agent and governance policy explicitly prohibit treating a
  separately installed `sqlite3` verdict as Palette acceptance evidence.

This removes the split-runtime acceptance path. A system binary can still be
used diagnostically, but it cannot produce the receipt consumed by Palette's
maintained validation and backup workflows.
